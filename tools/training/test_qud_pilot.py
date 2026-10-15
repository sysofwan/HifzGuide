"""Tests for training.qud_pilot: #144's rule, the sites, the worklist and the estimate.

Torch-free and model-free: the item, its reference and every decode are synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tadabur.qud_source import QudRow
from tadabur.qud_text import ItemReference
from tadabur.truth_sites import PENDING, parse_site
from training import qud_pilot as qp
from training.carrier_windowing import (
    Clip,
    carrier_sites,
    clip_commits,
    geminate_positions,
    pair_positions,
)
from training.truth_scorer import SiteRow

SLUG = qp.PILOT[0]
# Two words: ``قَالَ``, then ``سَططَ`` (a ``س`` carrier and an in-word ``طط`` geminate).
REFERENCE = "قَالَ سَططَ"
OFFSETS = (0, 6, 11)


def _item() -> qp.PilotItem:
    row = QudRow(SLUG, 1, 1, 2000, "a b", ((1, 2, 0, 2000),),
                 ((1, 100, 900), (2, 900, 1900)), "u", 0)
    return qp.PilotItem(f"{row.audio_filename}#0", SLUG, row, 0, 0, 32000, "e" * 64,
                        ItemReference(("a", "b"), (0, 1), REFERENCE, OFFSETS))


def _rows(decodes: dict[str, str]) -> list[SiteRow]:
    """The item's carriers with every arm's outcome, read as ``training.carrier_windowing``
    reads them (in process: no pool)."""
    item = _item()
    by_span = {**{a: REFERENCE for m in qp.MODELS
                  for a in (f"{m}/spans", f"{m}/stream_b0", f"{m}/checks")}, **decodes}
    positions = tuple(sorted(pair_positions(0, REFERENCE)
                             + geminate_positions(0, REFERENCE, OFFSETS)))
    clip = Clip(item.audio_filename, "1:1", 1, 1.0, (REFERENCE,), (0,), positions, {}, by_span)
    commits = clip_commits(clip)
    return [SiteRow(site, 1.0, 1, {a: o[k] for a, o in commits.items()}, {})
            for k, position in enumerate(clip.positions)
            for site in carrier_sites(clip, position)]


def _by_carrier(rows, letter):
    return [r for r in rows if r.site.reference[r.site.reference_index] == letter]


def test_both_rule_arms_on_the_mushaf_is_agreed_either_off_is_a_disagreement():
    rows = _rows({"h448/stream_b0": "قَالَ شَططَ"})  # h448 hears the س as ش
    assert {qp.classify(r) for r in _by_carrier(rows, "س")} == {qp.DISAGREED}
    assert {qp.classify(r) for r in _by_carrier(rows, "ق")} == {qp.AGREED}
    # Another arm off the mushaf does not matter: the rule reads two arms only.
    assert {qp.classify(r) for r in _rows({"base/checks": "قَالَ شَططَ"})} == {qp.AGREED}


def test_nothing_committed_with_no_disagreement_is_abstained():
    rows = _rows({"base/spans": "قَالَ"})  # the decode never reaches سَططَ
    assert {qp.classify(r) for r in _by_carrier(rows, "س")} == {qp.ABSTAINED}


def test_sites_are_valid_qud_truth_sites_and_leave_abstained_carriers_out():
    rows = _rows({"h448/stream_b0": "قَشَطَ", "base/spans": "شَططَ"})  # ق's word left out
    item = _item()
    sites = qp.truth_sites(rows, {(item.audio_filename, 0): item})
    for site in sites:
        parse_site(site.__dict__, site.site_id)
    marks = {(s.mark, s.heard, s.assumes_competent_reciter) for s in sites}
    assert marks == {("س↔ش", PENDING, False), ("س↔ص", PENDING, False),
                     ("shaddah", PENDING, False), ("ت↔ط", "ط", True), ("ط↔ق", "ط", True)}
    assert not any(s.reference_index == 0 for s in sites)  # the abstained ق


def test_the_worklist_has_one_row_per_disagreed_carrier_with_its_excerpt():
    rows = _rows({"h448/stream_b0": "قَالَ شَطَ"})
    item = _item()
    work = qp.worklist(rows, {(item.audio_filename, 0): item})
    assert [r["stratum"] for r in work] == ["letter:س", "geminate"]
    assert len(work[0]["site_ids"]) == 2  # the س sits in two pairs
    # Its word and the one before it (no word after), padded, within the item.
    assert (work[0]["excerpt_start_sample"], work[0]["excerpt_end_sample"]) == (0, 32000)


def test_the_listening_estimate_caps_each_stratum_at_the_budget():
    rows = [{"slug": SLUG, "stratum": "geminate", "listen_seconds": 12.0}] * 30 + \
           [{"slug": SLUG, "stratum": "letter:ط", "listen_seconds": 6.0}] * 4
    estimate = qp.listening_estimate(rows)[SLUG]
    assert estimate["census_minutes"] == pytest.approx((30 * 12 + 4 * 6) / 60, abs=0.05)
    assert estimate["k20_minutes"] == pytest.approx((20 * 12 + 4 * 6) / 60, abs=0.05)


def test_every_recited_ayah_word_is_graded_once_on_its_own_muraja_word():
    item = _item()
    placed = qp.placed_words(item, lambda surah, ayah, word: word)
    assert [(w.word_index, w.reference_index, site.word) for w, site in placed] == \
        [(0, 0, 1), (1, 6, 2)]
    # Two printed words recited as one phoneme word are graded once.
    merged = qp.placed_words(item, lambda surah, ayah, word: 1)
    assert [site.word for _, site in merged] == [1]


QURAN_DB = Path(__file__).parent.parent.parent / "data" / "quran.db"


@pytest.mark.skipif(not QURAN_DB.exists(), reason="data/quran.db is not built here")
@pytest.mark.parametrize("key", [
    "qud_abdul_hamid_ghraio_2026_yt_002_061.mp3#2",  # words 31-36, which the reference alone
    "qud_anas_almiman_yt_004_171.mp3#1",              # placed from word 1 (Sol, #149)
    "qud_anas_almiman_yt_005_003.mp3#2",
])
def test_a_mid_ayah_item_starts_on_its_own_first_word(key):
    from training.muraja_policy import MurajaText

    slug = key.removeprefix("qud_").rsplit("_", 2)[0]
    line = qp.read_decodes(slug, "base")[key]
    word_from, word_to = line["segment"][:2]
    surah, ayah = map(int, line["surah_ayah"].split(":"))
    row = QudRow(slug, surah, ayah, 0, "", (tuple(line["segment"]),), (), "u", 0)
    words = tuple(range(word_from - 1, word_to))
    item = qp.PilotItem(key, slug, row, 0, line["start_sample"], line["end_sample"], "e" * 64,
                        ItemReference(tuple("w" for _ in words), words, "x" * len(words),
                                      tuple(range(len(words) + 1))))
    text = MurajaText(QURAN_DB)
    placed = qp.placed_words(item, text.phoneme_word)
    placement = qp.placement(item, placed)
    expected = sorted({text.phoneme_word(surah, ayah, w + 1) for w in words})
    assert placement.start_word == text.phoneme_word(surah, ayah, word_from) > 1
    assert sorted(site.word for site in placement.where.values()) == expected


def test_merge_takes_each_item_once_and_refuses_a_missing_item(tmp_path, monkeypatch):
    monkeypatch.setattr(qp, "DECODES_DIR", tmp_path / "out")
    for shard, keys in ((0, ["b", "a"]), (1, ["c", "a"])):
        lines = [f'{{"key": "{k}", "spans": "{shard}"}}' for k in keys]
        (tmp_path / f"{SLUG}.{shard}.base.jsonl").write_text("\n".join(lines) + "\n")
    (tmp_path / f"{SLUG}.base.identity.json").write_text('{"model": "base"}')
    assert qp.merge_decodes(tmp_path, SLUG, "base", 3) == {"model": "base"}
    merged = qp.read_decodes(SLUG, "base")
    assert list(merged) == ["a", "b", "c"] and merged["a"]["spans"] == "0"
    with pytest.raises(ValueError, match="4 expected"):
        qp.merge_decodes(tmp_path, SLUG, "base", 4)


def test_a_re_read_word_is_timed_by_its_recital_inside_the_item():
    row = QudRow(SLUG, 1, 1, 4000, "a b a b", ((1, 2, 0, 2000), (1, 2, 2000, 4000)),
                 ((1, 100, 900), (2, 900, 1900), (1, 2100, 2900), (2, 2900, 3900)), "u", 0)
    item = qp.PilotItem(f"{row.audio_filename}#1", SLUG, row, 1, 0, 70000, "e" * 64,
                        ItemReference(("a", "b"), (0, 1), REFERENCE, OFFSETS))
    pad = round(0.3 * 16000)
    assert qp.excerpt(item, 6) == (2100 * 16 - pad, 3900 * 16 + pad)


def test_every_decoded_recording_is_registered_once():
    line = {"surah_ayah": "1:1", "mp3_sha256": "e" * 64}
    rows = qp.qud_exposures({SLUG: {"a#0": line, "a#1": line}, qp.PILOT[1]: {}})
    assert [(r.slug, r.surah_ayah, r.mp3_sha256) for r in rows] == [(SLUG, "1:1", "e" * 64)]
    assert rows[0].qud_revision == qp.QUD_REVISION


# --- through the pipeline (review round 2, #149) ------------------------------------------


class FakeHarness:
    """A harness that records each request and grades each requested word: the first
    correct, the second still pending (never graded), the last minor for a ن→ل mismatch."""

    def __init__(self, quran_db: Path):
        self.quran_db = quran_db
        self.requests: list[dict] = []

    def run(self, requests):
        self.requests += requests
        results = []
        for r in requests:
            words = r["report_words"]
            final = [{"word": w, "quality": "correct", "errors": []} for w in words]
            final[1]["quality"] = "pending"
            final[-1] = {"word": words[-1], "quality": "minor", "errors": [
                {"kind": {"type": "mismatch", "expected": "ن", "heard": "ل"}, "group_index": 0}]}
            results += [{"item": r["item"], "scoring": s["name"], "final": final, "letters": []}
                        for s in r["scorings"]]
        return {"protocol_version": 3}, results


def _real_item(key: str):
    """A committed pilot item rebuilt as the pipeline builds it: its canonical words, realized
    reference and decodes."""
    from tadabur.waqf_segments import _uthmani_words, hafs_segment_reference

    slug = key.removeprefix("qud_").rsplit("_", 2)[0]
    decodes = {m: {key: qp.read_decodes(slug, m)[key]} for m in qp.MODELS}
    line = decodes["base"][key]
    surah, ayah = map(int, line["surah_ayah"].split(":"))
    canonical = _uthmani_words(line["surah_ayah"])
    row = QudRow(slug, surah, ayah, 0, " ".join(canonical), (tuple(line["segment"]),), (), "u", 0)
    reference = qp.item_reference(line["segment"], canonical, hafs_segment_reference())
    item = qp.PilotItem(key, slug, row, 0, line["start_sample"], line["end_sample"],
                        line["mp3_sha256"], reference)
    return item, decodes


@pytest.mark.skipif(not QURAN_DB.exists(), reason="data/quran.db is not built here")
def test_the_app_outcome_requests_and_grades_an_items_own_canonical_words():
    item, decodes = _real_item("qud_abdul_hamid_ghraio_2026_yt_002_061.mp3#2")
    assert item.reference.word_indices == tuple(range(30, 36))  # canonical words 31-36
    harness = FakeHarness(QURAN_DB)
    out = qp.app_outcome([item], decodes, harness)
    assert {(r["start_word"], tuple(r["report_words"])) for r in harness.requests} == \
        {(31, (31, 32, 33, 34, 35, 36))}
    view = out["h448"]["shipped_default"]
    assert (view["words"], view["graded"], view["flagged"]) == (6, 5, 1)
    assert view["substitutions"] == [{"cause": "other_consonant", "substitution": "ن→ل",
                                      "words": 1}]


@pytest.mark.skipif(not QURAN_DB.exists(), reason="data/quran.db is not built here")
def test_a_per_mushaf_rate_divides_by_the_graded_words():
    item, decodes = _real_item("qud_abdul_hamid_ghraio_2026_yt_002_061.mp3#2")
    view = qp.app_outcome([item], decodes, FakeHarness(QURAN_DB))["base"]["allowances_off"]
    assert view["per_1000"] == pytest.approx(1000 / 5)  # one flag of five graded, not six words
    assert view["per_mushaf"][item.slug]["per_1000"] == pytest.approx(1000 / 5)


def test_build_registers_a_decoded_recording_that_yields_no_item(tmp_path, monkeypatch):
    from tadabur import exposure

    item = _item()
    lines = {m: {item.key: {"surah_ayah": "1:1", "mp3_sha256": "e" * 64, "spans": REFERENCE,
                            "stream_b0": REFERENCE, "checks": [{"hop": REFERENCE}]},
                 "qud_x_090_008.mp3#0": {"surah_ayah": "90:8", "mp3_sha256": "f" * 64}}
             for m in qp.MODELS}
    decodes = {m: {item.key: {k: v for k, v in lines[m][item.key].items()
                              if k in ("spans", "stream_b0", "checks")}} for m in qp.MODELS}
    pilot = qp.Pilot([item], decodes, {slug: {"unmapped_ayat": [], "phonetizer_unsupported":
                                              ["qud_x_090_008.mp3#0"], "repeated_words_ayat": 0}
                                       for slug in qp.PILOT})
    written = {}
    monkeypatch.setattr(qp, "load_pilot", lambda: pilot)
    monkeypatch.setattr(qp, "read_decodes",
                        lambda slug, model: lines[model] if slug == SLUG else {})
    monkeypatch.setattr(qp, "SITES_DIR", tmp_path / "sites")
    monkeypatch.setattr(qp, "WORKLIST_PATH", tmp_path / "worklist.jsonl")
    monkeypatch.setattr(exposure, "write_qud_use",
                        lambda use, rows, directory=None: written.update({use: list(rows)}))
    qp.build(workers=1)
    registered = {(r.surah_ayah, r.mp3_sha256) for r in written[exposure.QUD_CORRECT_RECITATION]}
    assert registered == {("1:1", "e" * 64), ("90:8", "f" * 64)}


def test_the_doc_compares_with_the_explained_excluded_device_rates_by_protocol():
    import json

    from training.qud_pilot_doc import render

    report = json.loads(qp.REPORT_PATH.read_text(encoding="utf-8"))
    report["device_logs"] = qp.device_log_rates()  # read from #151's report, as `app` does
    doc = render(report)
    header = next(line for line in doc.splitlines() if line.startswith("| model | scoring |"))
    assert "device, all eras" in header and "device, 1 s hop" in header
    rows = [line for line in doc.splitlines() if line.startswith("| base | ")]
    shipped = next(r for r in rows if "| shipped_default |" in r)
    off = next(r for r in rows if "| allowances_off |" in r)
    assert "| 17.1 (35 of 2042) | 10.1 (15 of 1485) |" in shipped
    assert "| 22.0 (45 of 2042) | 16.2 (24 of 1485) |" in off
    assert "110.6" not in shipped and "115.0" not in off  # every graded word: diagnostic only
    assert "Diagnostic: the device sessions with every graded word" in doc


def test_the_doc_discloses_what_the_fingerprint_can_miss():
    import json

    from training.qud_pilot_doc import render

    doc = render(json.loads(qp.REPORT_PATH.read_text(encoding="utf-8")))
    assert "no overlap detected under this screen, not proven absence" in doc
    assert "`ahmed_nuayna_qdc`" in doc and "0.4365" in doc and "not retuned" in doc


def test_the_committed_doc_is_what_the_renderer_writes():
    import json

    from training.qud_pilot_doc import render

    report = json.loads(qp.REPORT_PATH.read_text(encoding="utf-8"))
    assert qp.DOC_PATH.read_text(encoding="utf-8") == render(report)

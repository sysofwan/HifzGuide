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

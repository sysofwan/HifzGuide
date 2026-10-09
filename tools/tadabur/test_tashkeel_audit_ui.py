"""Tests for the blind listening-session UI (#87): what the page is and is not told, how
answers are stored, and the audio it serves, without binding a socket."""

from __future__ import annotations

import io
import json
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tadabur.listening_session import (
    Candidate,
    shuffled,
    Verdict,
    census,
    hearable,
    read_verdicts,
    write_verdicts,
    write_worklist,
)
from tadabur.staged_audio import StagedClip
from tadabur.tashkeel_audit_ui import (
    HIDDEN_LETTER,
    SessionState,
    StaleAnswer,
    Target,
    _hide_word,
    choices,
    mapped_offset,
    masked,
    page_key,
)
from tadabur.truth_sites import PENDING, TruthSite, audio_sha256, write_truth_sites

FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"
NUM_SAMPLES = 48000

#: One site per question kind: (reference, carrier index, mark, prescribed, stratum).
SITES = {
    "fatha": (f"ق{FATHA}اال{FATHA} ب{FATHA}ل", 0, "fatha", "fatha",
              "new_audit:fatha:base_empty:h448_empty"),
    "sukun": (f"ي{FATHA}قڇت{DAMMA}ل", 2, "sukun", "sukun",
              "new_audit:sukun:base_empty:h448_empty"),
    "held": (f"ر{FATHA}بب{KASRA}ي", 2, "shaddah", "held",
             "new_audit:shaddah:held:base_single"),
    "not_held": (f"ر{FATHA}ب{KASRA}ي", 2, "shaddah", "not_held", "p35_fixture:shadda"),
    "dhal": (f"ك{FATHA}ذ{FATHA}ب", 2, "ذ↔ظ", "ذ", "new_audit:ذ→ظ:base_partner"),
    "zah": (f"ك{FATHA}ظ{FATHA}ب", 2, "ذ↔ظ", "ظ", "new_audit:ظ→ذ:base_partner"),
    "qaf": (f"ي{FATHA}قڇت{DAMMA}ل", 2, "ق↔ك", "ق", "new_audit:ق→ك:base_partner"),
}


def _clip(audio_dir: Path, name: str) -> StagedClip:
    path = audio_dir / name
    sf.write(path, np.linspace(-0.5, 0.5, NUM_SAMPLES, dtype=np.float32), 16000,
             subtype="PCM_16")
    return StagedClip(name, 40, 1, 7, "2:2", NUM_SAMPLES, audio_sha256(path), ("mining_pool",))


def _candidate(name, number, clip, reference, index, mark, prescribed, stratum,
               surah_ayah="2:2", offsets=None) -> Candidate:
    site = TruthSite(
        site_id=f"{stratum}:{name}#{number}", source=stratum.split(":")[0],
        assumes_competent_reciter=False, audio_filename=clip.audio_filename, shard=40,
        start_sample=8000, end_sample=40000, audio_sha256=clip.audio_sha256,
        surah_ayah=surah_ayah, reference=reference, reference_index=index, mark=mark,
        prescribed=prescribed, heard=PENDING, stratum=stratum, stratum_population=24,
    )
    return Candidate(site, 1.0, 0, tuple(offsets or (0, len(reference))), (16000, 32000))


def _stage(tmp_path, names=tuple(SITES), extra=()) -> dict[str, StagedClip]:
    """Stage one clip per named site (each its own ayah) plus ``extra`` candidates' clips,
    and write their worklist; returns the registry."""
    audio_dir = tmp_path / "clips"
    audio_dir.mkdir(exist_ok=True)
    registry, candidates = {}, []
    for number, name in enumerate(names):
        clip = registry.setdefault(f"{name}.wav", _clip(audio_dir, f"{name}.wav"))
        candidates.append(_candidate(name, number, clip, *SITES[name],
                                     surah_ayah=f"3:{number + 1}"))
    for number, (clip_name, fields) in enumerate(extra):
        clip = registry.setdefault(clip_name, _clip(audio_dir, clip_name))
        candidates.append(_candidate(clip_name, 100 + number, clip, **fields))
    write_worklist(census(candidates), tmp_path / "worklist.jsonl")
    return registry


def _load(tmp_path, registry: dict[str, StagedClip]) -> SessionState:
    return SessionState.load(tmp_path / "worklist.jsonl", tmp_path / "verdicts.jsonl",
                             tmp_path / "clips", registry)


def _state(tmp_path, names=tuple(SITES), extra=()) -> SessionState:
    return _load(tmp_path, _stage(tmp_path, names, extra))


def _view(state: SessionState, name: str) -> dict:
    (row,) = [r for r in state.rows if r.site.audio_filename == f"{name}.wav"]
    return state.view(row)


def _text(view: dict) -> str:
    return view["before"] + view["carrier"] + view["after"]


# --- blinding --------------------------------------------------------------------------


def test_the_page_is_sent_only_the_blind_fields(tmp_path):
    state = _state(tmp_path)
    assert set(state.payload()) == {"sites", "progress"}
    for view in state.payload()["sites"]:
        assert set(view) == {"key", "mode", "surah_ayah", "before", "carrier", "after",
                             "choices", "heard", "note"}


def test_no_withheld_field_reaches_the_page(tmp_path):
    # Model outcome (the stratum), prescription, source, id and sampling weight: none of
    # them may appear anywhere in the response the page receives.
    state = _state(tmp_path)
    page = json.dumps(state.payload(), ensure_ascii=False)
    for row in state.rows:
        for withheld in (row.site.site_id, row.site.stratum, row.site.source,
                         row.site.audio_filename, row.site.audio_sha256):
            assert withheld not in page
    for word in ("base_", "h448", "p35", "new_audit", "prescribed", "stratum", "inclusion",
                 "probability", "population", "pending", "excerpt"):
        assert word not in page


def test_the_tashkeel_answer_and_its_cues_are_hidden(tmp_path):
    state = _state(tmp_path)
    fatha, sukun = _view(state, "fatha"), _view(state, "sukun")
    # The fatha on ق and the madd that lengthens it; the qalqala that marks ق as sakin.
    assert (fatha["before"], fatha["carrier"], fatha["after"]) \
        == ("", "ق", f"ل{FATHA} ب{FATHA}ل")
    assert (sukun["before"], sukun["carrier"], sukun["after"]) \
        == (f"ي{FATHA}", "ق", f"ت{DAMMA}ل")
    assert fatha["mode"] == sukun["mode"] == "tashkeel"
    assert fatha["choices"] == sukun["choices"] == ["fatha", "damma", "kasra", "sukun", "unclear"]


def test_a_held_and_a_not_held_site_look_the_same(tmp_path):
    state = _state(tmp_path)
    held, single = _view(state, "held"), _view(state, "not_held")
    assert (held["carrier"], held["after"]) == (single["carrier"], single["after"]) == ("ب", "ي")
    assert held["choices"] == single["choices"] == ["held", "not_held", "unclear"]


def test_the_letter_under_test_and_its_qalqala_are_hidden_in_both_directions(tmp_path):
    state = _state(tmp_path)
    dhal, zah, qaf = _view(state, "dhal"), _view(state, "zah"), _view(state, "qaf")
    assert dhal["carrier"] == zah["carrier"] == f"{HIDDEN_LETTER}{FATHA}"
    assert dhal["choices"] == zah["choices"] == ["ذ", "ظ", "unclear"]
    # No qalqala after the hidden letter: ق takes it and ك does not.
    assert (qaf["carrier"], qaf["after"]) == (HIDDEN_LETTER, f"ت{DAMMA}ل")
    assert not {"ذ", "ظ"} & set(_text(dhal) + _text(zah))
    assert not {"ق", "ك", "ڇ"} & set(_text(qaf))


def test_a_doubled_letter_under_test_is_hidden_whole(tmp_path):
    reference = f"ك{FATHA}ذذ{FATHA}ب"
    state = _state(tmp_path, (), [("c.wav", dict(
        reference=reference, index=2, mark="ذ↔ظ", prescribed="ذ",
        stratum="new_audit:ذ→ظ:base_partner"))])
    view = state.payload()["sites"][0]
    assert (view["before"], view["carrier"], view["after"]) \
        == (f"ك{FATHA}", f"{HIDDEN_LETTER}{FATHA}", "ب")


#: One segment with four sites, two of them on the same carrier, and a second reciter's
#: segment of the same ayah whose words carry no site of its own.
SHARED = f"ق{FATHA}اال{FATHA} ي{FATHA}قڇت{DAMMA}ل{DAMMA} ذ{FATHA}ظظ{FATHA}"
SHARED_OFFSETS = (0, 7, 16, len(SHARED))


def _shared_sites():
    item = dict(reference=SHARED, surah_ayah="9:9", offsets=SHARED_OFFSETS)
    sites = [
        ("one.wav", dict(index=0, mark="fatha", prescribed="fatha",
                         stratum="new_audit:fatha:base_matched:h448_empty", **item)),
        ("one.wav", dict(index=9, mark="sukun", prescribed="sukun",
                         stratum="new_audit:sukun:base_empty:h448_empty", **item)),
        ("one.wav", dict(index=9, mark="ق↔ك", prescribed="ق",
                         stratum="new_audit:ق→ك:base_rest", **item)),
        ("one.wav", dict(index=16, mark="ذ↔ظ", prescribed="ذ",
                         stratum="new_audit:ذ→ظ:base_rest", **item)),
        ("one.wav", dict(index=18, mark="ذ↔ظ", prescribed="ظ",
                         stratum="new_audit:ظ→ذ:base_rest", **item)),
    ]
    other = ("two.wav", dict(index=18, mark="shaddah", prescribed="held",
                             stratum="new_audit:shaddah:held:base_rest", **item))
    return sites + [other]


def test_every_site_s_answer_is_hidden_in_every_view_that_shows_its_word(tmp_path):
    state = _state(tmp_path, (), _shared_sites())
    views = state.payload()["sites"]
    texts = {_text(view) for view in views}
    assert len(texts) == 1  # every view of these words shows the same masked text
    (text,) = texts
    hidden = HIDDEN_LETTER
    assert text == f"قل{FATHA} ي{FATHA}{hidden}ت{DAMMA}ل{DAMMA} {hidden}{FATHA}{hidden}"
    # Highlights differ: each view marks its own carrier.
    assert sorted(view["carrier"] for view in views) == sorted(
        ["ق", HIDDEN_LETTER, HIDDEN_LETTER, f"{HIDDEN_LETTER}{FATHA}", HIDDEN_LETTER,
         HIDDEN_LETTER])


def test_masking_follows_the_word_not_the_clip(tmp_path):
    # A clip with no site in a word still hides the answers another clip's sites have
    # there, but only for the same ayah and the same realized word.
    state = _state(tmp_path, (), _shared_sites()[:1] + [
        ("two.wav", dict(reference=SHARED, surah_ayah="9:9", offsets=SHARED_OFFSETS,
                         index=16, mark="ذ↔ظ", prescribed="ذ",
                         stratum="new_audit:ذ→ظ:base_rest")),
        ("three.wav", dict(reference=SHARED, surah_ayah="9:10", offsets=SHARED_OFFSETS,
                           index=16, mark="ذ↔ظ", prescribed="ذ",
                           stratum="new_audit:ذ→ظ:base_rest")),
    ])
    rows = {row.site.audio_filename: row for row in state.rows}
    two = "".join(masked(rows["two.wav"], state._targets))
    three = "".join(masked(rows["three.wav"], state._targets))
    assert two.startswith("قل")             # one.wav's fatha site, same ayah and word
    assert three.startswith(f"ق{FATHA}اا")  # another ayah: nothing of one.wav's is hidden


def test_the_highlight_stops_at_the_next_letter_even_an_equal_one(tmp_path):
    # مِممم: a kasra on م, then an idgham run of م. Running the highlight on would show the
    # fatha after the run as if it were the carrier's.
    reference = f"م{KASRA}ممم{FATHA}ل"
    state = _state(tmp_path, (), [("c.wav", dict(
        reference=reference, index=0, mark="kasra", prescribed="kasra",
        stratum="new_audit:kasra:base_empty:h448_empty"))])
    view = state.payload()["sites"][0]
    assert (view["before"], view["carrier"], view["after"]) == ("", "م", f"ممم{FATHA}ل")


def test_every_question_offers_exactly_the_answers_its_mark_can_take(tmp_path):
    for row in _state(tmp_path).rows:
        assert set(choices(row)) == hearable(row.site.mark)


def test_the_page_key_is_opaque_and_stable(tmp_path):
    state = _state(tmp_path)
    for row in state.rows:
        key = page_key(row.site.site_id)
        assert key == state.view(row)["key"] and len(key) == 16
        assert state.row(key) is row


def test_the_page_offers_no_tally_or_result_route():
    source = (Path(__file__).parent / "tashkeel_audit_ui.py").read_text(encoding="utf-8")
    page = (Path(__file__).parent / "tashkeel_audit_page.html").read_text(encoding="utf-8")
    script = (Path(__file__).parent / "tashkeel_audit_session.mjs").read_text(encoding="utf-8")
    assert "/api/results" not in source and not hasattr(SessionState, "results")
    for word in ("prescribed", "stratum", "base_", "inclusion", "site_id", "edit", "decoy",
                 "synthetic", "operation", "splice", "donor"):
        assert word not in page.lower() and word not in script.lower()
    assert 'name="viewport"' in page


# --- answers ---------------------------------------------------------------------------


def test_an_answer_is_written_to_the_verdicts_file_by_site_id(tmp_path):
    state = _state(tmp_path)
    key = _view(state, "held")["key"]
    state.record(key, "not_held", "short", None)
    stored = read_verdicts(tmp_path / "verdicts.jsonl")
    (row,) = [r for r in state.rows if r.site.audio_filename == "held.wav"]
    assert stored == {row.site.site_id: Verdict(row.site.site_id, "not_held", "short")}
    assert state.progress() == {"answered": 1, "total": len(SITES)}
    assert _view(state, "held")["heard"] == "not_held"


def test_a_re_answer_names_the_answer_it_replaces(tmp_path):
    state = _state(tmp_path)
    key = _view(state, "fatha")["key"]
    state.record(key, "kasra", "", None)
    with pytest.raises(StaleAnswer):  # made from a view that had not seen "kasra"
        state.record(key, "damma", "", None)
    state.record(key, "sukun", "", "kasra")
    assert [v.heard for v in read_verdicts(tmp_path / "verdicts.jsonl").values()] == ["sukun"]


def test_a_failed_write_records_nothing(tmp_path, monkeypatch):
    import tadabur.tashkeel_audit_ui as ui

    state = _state(tmp_path)
    key = _view(state, "fatha")["key"]

    def full_disk(*args):
        raise OSError("No space left on device")

    monkeypatch.setattr(ui, "write_verdicts", full_disk)
    with pytest.raises(OSError):
        state.record(key, "kasra", "", None)
    assert state.verdicts == {} and state.progress()["answered"] == 0
    assert _view(state, "fatha")["heard"] is None


@pytest.mark.parametrize("name, heard", [("held", "fatha"), ("fatha", "held"),
                                         ("dhal", "ز"), ("fatha", PENDING), ("sukun", None)])
def test_an_answer_outside_the_question_is_refused(tmp_path, name, heard):
    state = _state(tmp_path)
    with pytest.raises(ValueError, match="not an answer"):
        state.record(_view(state, name)["key"], heard, "", None)
    assert not (tmp_path / "verdicts.jsonl").exists()


def test_an_unknown_key_is_refused(tmp_path):
    with pytest.raises(KeyError):
        _state(tmp_path).record("0" * 16, "fatha", "", None)


def test_the_session_resumes_and_keeps_verdicts_for_sites_it_no_longer_lists(tmp_path):
    write_verdicts({"dropped-by-a-re-mine": Verdict("dropped-by-a-re-mine", "held")},
                   tmp_path / "verdicts.jsonl")
    registry = _stage(tmp_path)
    state = _load(tmp_path, registry)
    state.record(_view(state, "zah")["key"], "ذ", "", None)
    resumed = _load(tmp_path, registry)
    assert resumed.progress() == {"answered": 1, "total": len(SITES)}
    assert set(read_verdicts(tmp_path / "verdicts.jsonl")) == {
        "dropped-by-a-re-mine", *(r.site.site_id for r in state.rows
                                  if r.site.audio_filename == "zah.wav")}


# --- audio -----------------------------------------------------------------------------


def _frames(data: bytes) -> int:
    with wave.open(io.BytesIO(data), "rb") as handle:
        assert (handle.getnchannels(), handle.getframerate()) == (1, 16000)
        return handle.getnframes()


def test_the_excerpt_and_the_whole_segment_are_served_as_wav(tmp_path):
    state = _state(tmp_path)
    key = _view(state, "fatha")["key"]
    assert _frames(state.audio(key, whole=False)) == 16000
    assert _frames(state.audio(key, whole=True)) == 32000


def test_the_server_refuses_audio_that_does_not_match_the_registry(tmp_path):
    registry = _stage(tmp_path)
    sf.write(tmp_path / "clips" / "fatha.wav", np.zeros(NUM_SAMPLES, dtype=np.float32), 16000,
             subtype="PCM_16")
    with pytest.raises(ValueError, match="sha256"):
        _load(tmp_path, registry)


def test_an_excerpt_past_the_end_of_its_clip_is_refused(tmp_path):
    registry = _stage(tmp_path)
    short = {name: replace(clip, num_samples=30000) for name, clip in registry.items()}
    with pytest.raises(ValueError, match="past the end"):
        _load(tmp_path, short)


def test_a_clip_missing_from_the_registry_is_refused(tmp_path):
    _stage(tmp_path)
    with pytest.raises(ValueError, match="not in the staged-clip registry"):
        _load(tmp_path, {})


# --- the synthetic-edit check (#107) ---------------------------------------------------

#: An ayah whose words a session site and edit-check items share: a geminate ب in the
#: first word, a ذ in the second, and the same session-site fatha on ق.
EDIT_AYAH = SHARED
EDIT_OFFSETS = SHARED_OFFSETS


def _edit_check(tmp_path, specs, reference=EDIT_AYAH, surah_ayah="9:9",
                offsets=EDIT_OFFSETS) -> Path:
    """Write a blind check (truth-site skeletons, ``source: synthetic_edit``) of ``specs``
    ``(file name, index, mark, prescribed)`` on one ayah (:data:`EDIT_AYAH` by default), its
    words file and each item's audio (distinct samples, so distinct checksums); return the
    blind check's path."""
    audio_dir = tmp_path / "clips"
    audio_dir.mkdir(exist_ok=True)
    sites = []
    for number, (name, index, mark, prescribed) in enumerate(specs):
        sf.write(audio_dir / name, np.full(NUM_SAMPLES, 0.01 * (number + 1), dtype=np.float32),
                 16000, subtype="PCM_16")
        sites.append(TruthSite(
            site_id=f"synthetic_edit:{number:020x}", source="synthetic_edit",
            assumes_competent_reciter=False, audio_filename=name, shard=200, start_sample=0,
            end_sample=NUM_SAMPLES, audio_sha256=audio_sha256(audio_dir / name),
            surah_ayah=surah_ayah, reference=reference, reference_index=index, mark=mark,
            prescribed=prescribed, heard=PENDING, stratum="synthetic_edit:blind_check",
            stratum_population=334))
    write_truth_sites(sites, tmp_path / "blind_check.jsonl")
    (tmp_path / "words.json").write_text(json.dumps({"ayahs": {surah_ayah: {
        "reference": reference, "word_offsets": list(offsets)}}}), encoding="utf-8")
    return tmp_path / "blind_check.jsonl"


#: An edit and its decoy are two files with the same source reference, carrier and mark;
#: a third item tests the pair. Nothing else distinguishes them.
EDIT_SPECS = [("se_aaaa.wav", 18, "shaddah", "held"), ("se_bbbb.wav", 18, "shaddah", "held"),
              ("se_cccc.wav", 16, "ذ↔ظ", "ذ")]


def _edit_state(tmp_path, specs=EDIT_SPECS, session=("fatha",), extra=()) -> SessionState:
    registry = _stage(tmp_path, session, extra)
    blind_check = _edit_check(tmp_path, specs)
    return SessionState.load(tmp_path / "worklist.jsonl", tmp_path / "verdicts.jsonl",
                             tmp_path / "clips", registry, blind_check, tmp_path / "words.json")


def _edit_views(state: SessionState) -> dict[str, dict]:
    return {row.site.audio_filename: state.view(row) for row in state.rows
            if row.site.source == "synthetic_edit"}


def test_the_edit_check_shares_the_queue_and_asks_whether_it_sounds_natural(tmp_path):
    state = _edit_state(tmp_path)
    assert state.progress() == {"answered": 0, "total": 1 + len(EDIT_SPECS)}
    for view in state.payload()["sites"]:
        blind = {"key", "mode", "surah_ayah", "before", "carrier", "after", "choices",
                 "heard", "note"}
        asks = "natural" in view
        assert set(view) == (blind | {"natural_choices", "natural"} if asks else blind)
        if asks:
            assert view["natural_choices"] == ["natural", "unnatural", "unclear"]
    views = _edit_views(state)
    assert views["se_aaaa.wav"]["choices"] == ["held", "not_held", "unclear"]
    assert views["se_cccc.wav"]["choices"] == ["ذ", "ظ", "unclear"]


def test_an_edit_and_its_decoy_reach_the_page_identically(tmp_path):
    views = _edit_views(_edit_state(tmp_path))
    edit, decoy = views["se_aaaa.wav"], views["se_bbbb.wav"]
    assert edit["key"] != decoy["key"]
    assert {**edit, "key": None} == {**decoy, "key": None}


def test_no_withheld_field_reaches_the_page_in_the_edit_check(tmp_path):
    state = _edit_state(tmp_path)
    page = json.dumps(state.payload(), ensure_ascii=False)
    for row in state.rows:
        for withheld in (row.site.site_id, row.site.audio_filename, row.site.audio_sha256,
                         row.site.stratum, row.site.source):
            assert withheld not in page
    for word in ("edit", "decoy", "synthetic", "blind", "operation", "label", "donor",
                 "splice", "render", "source", "se_", "prescribed", "pending", "population"):
        assert word not in page.lower()


def test_the_edit_check_hides_its_carrier_and_every_shared_word_s_answer(tmp_path):
    # The session's fatha on ق shares the first word with the items; every view of the
    # ayah, the session's and the items', hides all three answers.
    state = _edit_state(tmp_path, extra=[_shared_sites()[0]])
    texts = {_text(view) for view in state.payload()["sites"]
             if view["surah_ayah"] == "9:9"}
    hidden = HIDDEN_LETTER
    assert texts == {f"قل{FATHA} ي{FATHA}قڇت{DAMMA}ل{DAMMA} {hidden}{FATHA}ظ"}
    views = _edit_views(state)
    assert views["se_aaaa.wav"]["carrier"] == "ظ"  # the geminate shown once, haraka hidden
    assert views["se_cccc.wav"]["carrier"] == f"{hidden}{FATHA}"


def test_the_queue_order_depends_on_the_site_ids_alone(tmp_path):
    def queue(state):
        return [row.site.site_id for row in state.rows]

    state = _edit_state(tmp_path)
    blind_check = tmp_path / "blind_check.jsonl"
    lines = blind_check.read_text(encoding="utf-8").splitlines(keepends=True)
    blind_check.write_text("".join(reversed(lines)), encoding="utf-8")
    again = SessionState.load(tmp_path / "worklist.jsonl", tmp_path / "verdicts.jsonl",
                              tmp_path / "clips", _stage(tmp_path, ("fatha",)), blind_check,
                              tmp_path / "words.json")
    assert queue(again) == queue(state) == [r.site.site_id for r in shuffled(state.rows)]


def test_an_edit_check_answer_needs_both_questions_and_is_one_verdict(tmp_path):
    state = _edit_state(tmp_path)
    key = _edit_views(state)["se_aaaa.wav"]["key"]
    with pytest.raises(ValueError, match="natural"):
        state.record(key, "held", "", None)
    with pytest.raises(ValueError, match="natural"):
        state.record(key, "held", "", None, "maybe")
    state.record(key, "not_held", "a click", None, "unnatural")
    (verdict,) = read_verdicts(tmp_path / "verdicts.jsonl").values()
    assert (verdict.heard, verdict.natural, verdict.note) == ("not_held", "unnatural", "a click")
    with pytest.raises(StaleAnswer):  # the page names the pair it last saw
        state.record(key, "held", "", "not_held", "natural")
    state.record(key, "held", "", {"heard": "not_held", "natural": "unnatural"}, "natural")
    view = _edit_views(state)["se_aaaa.wav"]
    assert (view["heard"], view["natural"]) == ("held", "natural")


def test_a_session_site_refuses_a_naturalness_answer(tmp_path):
    state = _edit_state(tmp_path)
    (row,) = [r for r in state.rows if r.site.audio_filename == "fatha.wav"]
    with pytest.raises(ValueError, match="natural"):
        state.record(page_key(row.site.site_id), "fatha", "", None, "natural")


def test_an_edit_check_item_plays_whole_either_way(tmp_path):
    state = _edit_state(tmp_path)
    key = _edit_views(state)["se_cccc.wav"]["key"]
    assert _frames(state.audio(key, whole=False)) == _frames(state.audio(key, whole=True)) \
        == NUM_SAMPLES


def test_the_server_refuses_edit_check_audio_that_does_not_match(tmp_path):
    with pytest.raises(FileNotFoundError, match="listening_session fetch"):
        registry = _stage(tmp_path, ("fatha",))
        blind_check = _edit_check(tmp_path, EDIT_SPECS)
        (tmp_path / "clips" / "se_bbbb.wav").unlink()
        SessionState.load(tmp_path / "worklist.jsonl", tmp_path / "verdicts.jsonl",
                          tmp_path / "clips", registry, blind_check, tmp_path / "words.json")
    sf.write(tmp_path / "clips" / "se_bbbb.wav", np.zeros(NUM_SAMPLES, dtype=np.float32),
             16000, subtype="PCM_16")
    with pytest.raises(ValueError, match="sha256"):
        SessionState.load(tmp_path / "worklist.jsonl", tmp_path / "verdicts.jsonl",
                          tmp_path / "clips", registry, blind_check, tmp_path / "words.json")


def test_the_ui_never_loads_the_edit_manifest_or_the_summary():
    # The manifest names every item's operation and role; the summary reads it. Neither
    # may be importable from the server process the page talks to.
    import subprocess
    import sys

    probe = ("import sys, tadabur.tashkeel_audit_ui; "
             "print(sorted(m for m in sys.modules if m.startswith('tadabur.')))")
    loaded = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            check=True, cwd=Path(__file__).parent.parent).stdout
    assert "tadabur.synthetic_edits'" not in loaded and "edit_check_summary" not in loaded
    assert "tadabur.edit_check'" in loaded
    source = (Path(__file__).parent / "tashkeel_audit_ui.py").read_text(encoding="utf-8")
    assert "edits.jsonl" not in source and "read_items" not in source


#: 9:91 as the committed queue has it: a session segment that ends in waqf on its 18th word
#: (``وَرَسُۥۥلِه``) and the whole ayah an edit-check item tests, in wasl there
#: (``وَرَسُۥۥلِهِۦۦ``), with a shaddah question on its س (prescribed not held).
AYAH_9_91 = ("لَيسَ عَلَ ضضُعَفَااااءِ وَلَاا عَلَ لمَرضَاا وَلَاا عَلَ للَذِۦۦنَ لَاا "
             "يَجِدُۥۥنَ مَاا يُںںںفِقُۥۥنَ حَرَجُن ءِذَاا نَصَحُۥۥ لِللَااهِ وَرَسُۥۥلِهِۦۦ "
             "مَاا عَلَ لمُحسِنِۦۦنَ مِںںںسَبِۦۦلِوووَللَااهُ غَفُۥۥرُررَحِۦۦم")
AYAH_9_91_OFFSETS = (0, 6, 11, 25, 32, 37, 46, 53, 58, 68, 73, 84, 89, 103, 111, 118, 127,
                     137, 152, 157, 162, 175, 180, 190, 200, 208, 216)
SEGMENT_9_91 = AYAH_9_91[:137] + "وَرَسُۥۥلِه"
SEGMENT_9_91_OFFSETS = (*AYAH_9_91_OFFSETS[:18], len(SEGMENT_9_91))


def test_a_carrier_is_hidden_in_every_realization_of_its_word(tmp_path):
    assert AYAH_9_91[141] == "س" and SEGMENT_9_91[141] == "س"
    registry = _stage(tmp_path, (), [("seg.wav", dict(
        reference=SEGMENT_9_91, index=59, mark="fatha", prescribed="fatha",
        stratum="new_audit:fatha:base_matched:h448_empty", surah_ayah="9:91",
        offsets=SEGMENT_9_91_OFFSETS))])
    blind_check = _edit_check(tmp_path, [("se_aaaa.wav", 141, "shaddah", "not_held")],
                              AYAH_9_91, "9:91", AYAH_9_91_OFFSETS)
    state = SessionState.load(tmp_path / "worklist.jsonl", tmp_path / "verdicts.jsonl",
                              tmp_path / "clips", registry, blind_check, tmp_path / "words.json")
    for view in state.payload()["sites"]:
        text = _text(view)
        # The damma after س would show it single (a "not held" prescription); a geminate
        # would show as سس. Both views show the letter once with its haraka hidden.
        assert "رَسُ" not in text and "رَسۥۥلِه" in text


def test_a_carrier_is_mapped_across_realizations_and_a_word_it_cannot_be_placed_in_hides():
    target = Target(("2:2", 0), f"ء{FATHA}لق{FATHA}مر", 3, "tashkeel")
    # The segment's first word lost its hamzat wasl: the carrier moves two places left.
    assert mapped_offset(target, f"لق{FATHA}مر") == 1
    assert mapped_offset(target, target.text) == 3
    # The carrier's own haraka differs: its span is not matched whole, so it is not placed.
    assert mapped_offset(target, f"ء{FATHA}لق{DAMMA}مر") is None
    shown = list(f"ب{KASRA} ء{FATHA}لق{DAMMA}مر")
    _hide_word(f"ء{FATHA}لق{DAMMA}مر", shown, 3)
    assert "".join(shown) == f"ب{KASRA} {HIDDEN_LETTER * 5}"

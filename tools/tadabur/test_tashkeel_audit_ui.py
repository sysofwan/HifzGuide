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
    choices,
    masked,
    page_key,
)
from tadabur.truth_sites import PENDING, TruthSite, audio_sha256

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
    for word in ("prescribed", "stratum", "base_", "inclusion", "site_id"):
        assert word not in page and word not in script
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

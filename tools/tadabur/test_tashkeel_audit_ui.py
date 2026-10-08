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
    blind_reference,
    choices,
    page_key,
)
from tadabur.truth_sites import PENDING, TruthSite, audio_sha256

FATHA, DAMMA, KASRA, SUKUN_SIGN = "َ", "ُ", "ِ", "ْ"
NUM_SAMPLES = 48000

#: One site per question kind: (reference, carrier index, mark, prescribed, stratum).
SITES = {
    "fatha": (f"ق{FATHA}اال{FATHA} ب{FATHA}ل", 0, "fatha", "fatha",
              "new_audit:fatha:base_empty"),
    "sukun": (f"ي{FATHA}قڇت{DAMMA}ل", 2, "sukun", "sukun", "new_audit:sukun:base_empty"),
    "held": (f"ر{FATHA}بب{KASRA}ي", 2, "shaddah", "held",
             "new_audit:shaddah:held:base_single"),
    "not_held": (f"ر{FATHA}ب{KASRA}ي", 2, "shaddah", "not_held", "p35_fixture:shadda"),
    "dhal": (f"ك{FATHA}ذ{FATHA}ب", 2, "ذ↔ظ", "ذ", "new_audit:ذ→ظ:base_partner"),
    "zah": (f"ك{FATHA}ظ{FATHA}ب", 2, "ذ↔ظ", "ظ", "new_audit:ظ→ذ:base_partner"),
}


def _clip(audio_dir: Path, name: str) -> StagedClip:
    path = audio_dir / name
    sf.write(path, np.linspace(-0.5, 0.5, NUM_SAMPLES, dtype=np.float32), 16000,
             subtype="PCM_16")
    return StagedClip(name, 40, 1, 7, "2:2", NUM_SAMPLES, audio_sha256(path), ("mining_pool",))


def _stage(tmp_path, names=tuple(SITES)) -> dict[str, StagedClip]:
    """Stage one clip per site and write their worklist; returns the registry."""
    audio_dir = tmp_path / "clips"
    audio_dir.mkdir(exist_ok=True)
    registry, candidates = {}, []
    for number, name in enumerate(names):
        reference, index, mark, prescribed, stratum = SITES[name]
        clip = registry.setdefault(f"{name}.wav", _clip(audio_dir, f"{name}.wav"))
        site = TruthSite(
            site_id=f"{stratum}:{name}#{number}", source=stratum.split(":")[0],
            assumes_competent_reciter=False, audio_filename=clip.audio_filename, shard=40,
            start_sample=8000, end_sample=40000, audio_sha256=clip.audio_sha256,
            surah_ayah="2:2", reference=reference, reference_index=index, mark=mark,
            prescribed=prescribed, heard=PENDING, stratum=stratum, stratum_population=24,
        )
        candidates.append(Candidate(site, 1.0, (16000, 32000)))
    write_worklist(census(candidates), tmp_path / "worklist.jsonl")
    return registry


def _load(tmp_path, registry: dict[str, StagedClip]) -> SessionState:
    return SessionState.load(tmp_path / "worklist.jsonl", tmp_path / "verdicts.jsonl",
                             tmp_path / "clips", registry)


def _state(tmp_path, names=tuple(SITES)) -> SessionState:
    return _load(tmp_path, _stage(tmp_path, names))


def _view(state: SessionState, name: str) -> dict:
    (row,) = [r for r in state.rows if r.site.audio_filename == f"{name}.wav"]
    return state.view(row)


# --- blinding --------------------------------------------------------------------------


def test_the_page_is_sent_only_the_blind_fields(tmp_path):
    state = _state(tmp_path)
    for row in state.rows:
        assert set(state.view(row)) == {"key", "mode", "surah_ayah", "before", "carrier",
                                        "after", "choices", "heard", "note"}


def test_no_withheld_field_reaches_the_page(tmp_path):
    # Model outcome (the stratum), prescription, source, id and sampling weight: none of
    # them may appear anywhere in what the page receives.
    state = _state(tmp_path)
    page = json.dumps([state.view(row) for row in state.rows], ensure_ascii=False)
    for row in state.rows:
        for withheld in (row.site.site_id, row.site.stratum, row.site.source,
                         row.site.audio_filename, row.site.audio_sha256):
            assert withheld not in page
    for word in ("base_", "p35", "new_audit", "prescribed", "stratum", "inclusion",
                 "probability", "population", "pending"):
        assert word not in page


def test_the_tashkeel_answer_is_hidden_from_the_reference(tmp_path):
    state = _state(tmp_path)
    fatha, sukun = _view(state, "fatha"), _view(state, "sukun")
    assert (fatha["before"], fatha["carrier"], fatha["after"]) \
        == ("", "ق", f"ل{FATHA} ب{FATHA}ل")
    assert (sukun["before"], sukun["carrier"], sukun["after"]) \
        == (f"ي{FATHA}", "ق", f"ت{DAMMA}ل")
    assert fatha["mode"] == sukun["mode"] == "tashkeel"
    assert fatha["choices"] == sukun["choices"] == ["fatha", "damma", "kasra", "sukun", "unclear"]


def test_a_held_and_a_not_held_site_both_show_one_consonant(tmp_path):
    state = _state(tmp_path)
    held, single = _view(state, "held"), _view(state, "not_held")
    assert (held["carrier"], held["after"]) == (single["carrier"], single["after"]) \
        == (f"ب{KASRA}", "ي")
    assert held["choices"] == single["choices"] == ["held", "not_held", "unclear"]


def test_the_letter_under_test_is_replaced_and_both_directions_offer_the_same_choices(tmp_path):
    state = _state(tmp_path)
    dhal, zah = _view(state, "dhal"), _view(state, "zah")
    assert dhal["carrier"] == zah["carrier"] == f"{HIDDEN_LETTER}{FATHA}"
    assert "ذ" not in dhal["before"] + dhal["after"] and "ظ" not in zah["before"] + zah["after"]
    assert dhal["choices"] == zah["choices"] == ["ذ", "ظ", "unclear"]


def test_a_doubled_letter_under_test_is_hidden_whole(tmp_path):
    state = _state(tmp_path, ("dhal",))
    row = state.rows[0]
    doubled = replace(row, site=replace(row.site, reference=f"ك{FATHA}ذذ{FATHA}ب"))
    assert blind_reference(doubled) == (f"ك{FATHA}", f"{HIDDEN_LETTER}{FATHA}", "ب")


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
    assert "/api/results" not in source and not hasattr(SessionState, "results")
    for word in ("prescribed", "stratum", "base_", "inclusion", "site_id"):
        assert word not in page
    assert 'name="viewport"' in page


# --- answers ---------------------------------------------------------------------------


def test_an_answer_is_written_to_the_verdicts_file_by_site_id(tmp_path):
    state = _state(tmp_path)
    key = _view(state, "held")["key"]
    state.record(key, "not_held", "short")
    stored = read_verdicts(tmp_path / "verdicts.jsonl")
    (row,) = [r for r in state.rows if r.site.audio_filename == "held.wav"]
    assert stored == {row.site.site_id: Verdict(row.site.site_id, "not_held", "short")}
    assert state.progress() == {"answered": 1, "total": len(SITES)}
    assert _view(state, "held")["heard"] == "not_held"


def test_a_re_answer_replaces_the_earlier_one(tmp_path):
    state = _state(tmp_path)
    key = _view(state, "fatha")["key"]
    state.record(key, "kasra", "")
    state.record(key, "sukun", "")
    assert [v.heard for v in read_verdicts(tmp_path / "verdicts.jsonl").values()] == ["sukun"]


@pytest.mark.parametrize("name, heard", [("held", "fatha"), ("fatha", "held"),
                                         ("dhal", "ز"), ("fatha", PENDING), ("sukun", None)])
def test_an_answer_outside_the_question_is_refused(tmp_path, name, heard):
    state = _state(tmp_path)
    with pytest.raises(ValueError, match="not an answer"):
        state.record(_view(state, name)["key"], heard, "")
    assert not (tmp_path / "verdicts.jsonl").exists()


def test_an_unknown_key_is_refused(tmp_path):
    with pytest.raises(KeyError):
        _state(tmp_path).record("0" * 16, "fatha", "")


def test_the_session_resumes_and_keeps_verdicts_for_sites_it_no_longer_lists(tmp_path):
    write_verdicts({"dropped-by-a-re-mine": Verdict("dropped-by-a-re-mine", "held")},
                   tmp_path / "verdicts.jsonl")
    registry = _stage(tmp_path)
    state = _load(tmp_path, registry)
    state.record(_view(state, "zah")["key"], "ذ", "")
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


def test_a_clip_missing_from_the_registry_is_refused(tmp_path):
    _stage(tmp_path)
    with pytest.raises(ValueError, match="not in the staged-clip registry"):
        _load(tmp_path, {})

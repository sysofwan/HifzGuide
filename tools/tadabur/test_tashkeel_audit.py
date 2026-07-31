"""Tests for the tashkeel audit's storage, comparison and UI logic (#60).

Exercises the adjudication round-trip, the blind view the listener is served, the
window-slice WAV encoding and the population-scaled over-strictness comparison — all
without binding a socket.
"""

from __future__ import annotations

import io
import json
import wave

import numpy as np
import pytest

from training.tashkeel_eval import DAMMA, FATHA, KASRA, MATCHED, OMITTED
from training.tashkeel_worklist import RECOVERED, REGRESSED, TashkeelSite, site_id, write_worklist

from .tashkeel_acceptance import compare, summarize_direction
from .tashkeel_audit_ui import AuditState, ClipCache, encode_wav
from .tashkeel_fixtures import (
    NONE,
    UNCLEAR,
    Adjudication,
    read_adjudications,
    write_adjudications,
)

_VOWEL_NAMES = {FATHA: "fatha", DAMMA: "damma", KASRA: "kasra"}


def _site(index: int, direction: str, vowel: str = FATHA) -> TashkeelSite:
    return TashkeelSite(
        site_id=site_id("clip.wav", 0, index),
        clip_audio_filename="clip.wav",
        surah_ayah="2:1",
        reciter_id=3,
        window_index=0,
        start_sample=16000,
        num_samples=32000,
        reference=f"م{vowel}الك",
        reference_index=1,
        reference_vowel=vowel,
        vowel_name=_VOWEL_NAMES[vowel],
        carrier="م",
        direction=direction,
        base_outcome=OMITTED if direction == RECOVERED else MATCHED,
        candidate_outcome=MATCHED if direction == RECOVERED else OMITTED,
        base_vowel=None,
        candidate_vowel=vowel,
    )


def _verdict(site: TashkeelSite, verdict: str) -> Adjudication:
    return Adjudication(site.site_id, verdict, site.clip_audio_filename, site.reference_index)


# --- fixtures -------------------------------------------------------------------------


def test_adjudications_round_trip_through_disk(tmp_path):
    path = tmp_path / "adjudications.jsonl"
    entries = {a.site_id: a for a in (_verdict(_site(1, RECOVERED), "fatha"),
                                      _verdict(_site(2, REGRESSED), UNCLEAR))}
    write_adjudications(path, entries)
    assert read_adjudications(path) == entries


def test_a_missing_adjudication_file_reads_as_an_empty_audit(tmp_path):
    assert read_adjudications(tmp_path / "nothing.jsonl") == {}


def test_a_resubmitted_verdict_replaces_the_earlier_one(tmp_path):
    path = tmp_path / "adjudications.jsonl"
    site = _site(1, RECOVERED)
    path.write_text(
        json.dumps({"site_id": site.site_id, "verdict": "fatha",
                    "clip_audio_filename": "clip.wav", "reference_index": 1}) + "\n"
        + json.dumps({"site_id": site.site_id, "verdict": UNCLEAR,
                      "clip_audio_filename": "clip.wav", "reference_index": 1}) + "\n",
        encoding="utf-8",
    )
    assert read_adjudications(path)[site.site_id].verdict == UNCLEAR


def test_an_unknown_verdict_is_refused():
    with pytest.raises(ValueError, match="not a tashkeel verdict"):
        Adjudication("abc", "probably_fatha", "clip.wav", 1)


def test_reading_rejects_a_row_missing_a_required_field(tmp_path):
    path = tmp_path / "adjudications.jsonl"
    path.write_text(json.dumps({"site_id": "abc", "verdict": "fatha"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="adjudication schema"):
        read_adjudications(path)


def test_only_a_heard_colour_maps_to_a_vowel():
    assert _verdict(_site(1, RECOVERED), "kasra").heard_vowel == KASRA
    assert _verdict(_site(1, RECOVERED), NONE).heard_vowel is None
    assert _verdict(_site(1, RECOVERED), UNCLEAR).heard_vowel is None


# --- comparison -----------------------------------------------------------------------


def test_a_site_counts_as_over_strictness_only_when_the_reference_colour_was_heard():
    sites = [_site(1, RECOVERED, FATHA), _site(2, RECOVERED, FATHA), _site(3, RECOVERED, FATHA)]
    adjudications = {
        sites[0].site_id: _verdict(sites[0], "fatha"),
        sites[1].site_id: _verdict(sites[1], "kasra"),
        sites[2].site_id: _verdict(sites[2], UNCLEAR),
    }
    result = summarize_direction(sites, adjudications, RECOVERED, population=30)
    assert (result.confirmed, result.said_otherwise, result.unclear) == (1, 1, 1)


def test_unjudged_sites_are_not_counted_as_anything():
    sites = [_site(1, RECOVERED), _site(2, RECOVERED)]
    result = summarize_direction(sites, {sites[0].site_id: _verdict(sites[0], "fatha")},
                                 RECOVERED, population=10)
    assert result.audited == 1


def test_the_audited_share_is_scaled_onto_the_mined_population():
    # Half the audited recoveries were genuinely correct recitation, so half the 100 mined
    # ones are estimated to be -- reporting 1 would read as a census of the corpus.
    sites = [_site(i, RECOVERED) for i in range(4)]
    adjudications = {}
    for i, site in enumerate(sites):
        adjudications[site.site_id] = _verdict(site, "fatha" if i < 2 else "kasra")
    result = summarize_direction(sites, adjudications, RECOVERED, population=100)
    assert result.estimate == pytest.approx(50.0)


def _population(recovered: int, regressed: int, vowels: int) -> dict:
    return {"reference_vowels": vowels, RECOVERED: recovered, REGRESSED: regressed,
            "concordant": vowels - recovered - regressed}


def test_the_gain_is_positive_when_the_base_falsely_rejects_more():
    sites = [_site(i, RECOVERED) for i in range(4)] + [_site(10 + i, REGRESSED) for i in range(4)]
    adjudications = {s.site_id: _verdict(s, "fatha") for s in sites[:4]}
    adjudications.update({s.site_id: _verdict(s, "kasra") for s in sites[4:]})
    report = compare(sites, adjudications, _population(400, 40, 10000))
    # Every mined recovery was real over-strictness; no mined regression was.
    assert report["base_false_rejection_rate"] == pytest.approx(0.04)
    assert report["candidate_false_rejection_rate"] == pytest.approx(0.0)
    assert report["acceptance_gain"] > 0


def test_the_gain_interval_is_wider_than_the_point_estimate_on_both_sides():
    sites = [_site(i, RECOVERED) for i in range(4)] + [_site(10 + i, REGRESSED) for i in range(4)]
    adjudications = {s.site_id: _verdict(s, "fatha") for s in sites[:4]}
    adjudications.update({s.site_id: _verdict(s, "fatha") for s in sites[4:]})
    report = compare(sites, adjudications, _population(400, 40, 10000))
    low, high = report["acceptance_gain_ci95"]
    assert low < report["acceptance_gain"] < high


def test_an_unaudited_worklist_reports_pending_rather_than_a_verdict():
    sites = [_site(i, RECOVERED) for i in range(3)]
    report = compare(sites, {}, _population(300, 30, 9000))
    assert report["audited"] == 0
    assert report["pending"] == 3


# --- audio ----------------------------------------------------------------------------


def test_encoded_window_audio_is_mono_16k_pcm():
    data = encode_wav(np.zeros(1600, dtype=np.float32))
    with wave.open(io.BytesIO(data), "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 16000)
        assert handle.getnframes() == 1600


def test_samples_beyond_full_scale_clip_instead_of_wrapping():
    # Without the clip, +1.5 overflows int16 and returns as a large *negative* sample --
    # audible as a crack exactly where the listener is trying to judge a vowel.
    data = encode_wav(np.array([1.5, -1.5], dtype=np.float32))
    with wave.open(io.BytesIO(data), "rb") as handle:
        pcm = np.frombuffer(handle.readframes(2), dtype="<i2")
    assert pcm.tolist() == [32767, -32767]


def test_a_clip_missing_from_the_audio_dir_names_both_layouts(tmp_path):
    with pytest.raises(FileNotFoundError, match="hash-prefixed"):
        ClipCache(tmp_path).waveform("absent.wav")


# --- UI state -------------------------------------------------------------------------


def _state(tmp_path, sites) -> AuditState:
    worklist = tmp_path / "worklist.jsonl"
    write_worklist(worklist, sites)
    return AuditState.load(
        worklist, tmp_path / "adjudications.jsonl", tmp_path,
        _population(len(sites), len(sites), 1000),
    )


def test_the_listener_is_never_told_which_model_did_what(tmp_path):
    state = _state(tmp_path, [_site(1, RECOVERED)])
    view = state.view(state.sites[0])
    assert not {"direction", "base_outcome", "candidate_outcome", "base_vowel",
                "candidate_vowel"} & view.keys()


def test_the_reference_vowel_is_withheld_and_stripped_from_the_displayed_reference(tmp_path):
    # Showing the prescribed vowel primes the listener toward the answer that scores as
    # confirmed over-strictness -- the reading that flatters the fine-tune.
    state = _state(tmp_path, [_site(1, RECOVERED, KASRA)])
    view = state.view(state.sites[0])
    assert "reference_vowel" not in view
    assert KASRA not in view["reference"]
    assert view["reference"] == "مالك"


def test_the_carrier_index_points_at_the_letter_to_judge_after_the_vowel_is_removed(tmp_path):
    state = _state(tmp_path, [_site(1, RECOVERED, FATHA)])
    view = state.view(state.sites[0])
    assert view["reference"][view["carrier_index"]] == "م"


def test_surrounding_harakat_survive_as_context(tmp_path):
    site = _site(1, RECOVERED, FATHA)
    site = TashkeelSite(**{**site.__dict__, "reference": f"م{FATHA}ل{KASRA}ك",
                           "reference_index": 1})
    state = _state(tmp_path, [site])
    view = state.view(state.sites[0])
    assert view["reference"] == f"مل{KASRA}ك"


def test_a_recorded_verdict_persists_and_shows_up_in_the_view(tmp_path):
    state = _state(tmp_path, [_site(1, RECOVERED), _site(2, REGRESSED)])
    state.record(state.sites[0].site_id, "damma", "sounded short")
    assert state.progress() == {"total": 2, "judged": 1}
    assert state.view(state.sites[0])["verdict"] == "damma"
    assert read_adjudications(state.adjudications_path)[state.sites[0].site_id].note == "sounded short"


def test_recording_against_an_unknown_site_is_an_error(tmp_path):
    state = _state(tmp_path, [_site(1, RECOVERED)])
    with pytest.raises(KeyError):
        state.record("not-a-site", "fatha", "")


def test_the_state_resumes_from_whatever_the_file_already_holds(tmp_path):
    state = _state(tmp_path, [_site(1, RECOVERED), _site(2, REGRESSED)])
    state.record(state.sites[1].site_id, "kasra", "")
    resumed = AuditState.load(
        tmp_path / "worklist.jsonl", tmp_path / "adjudications.jsonl", tmp_path,
        state.population,
    )
    assert resumed.progress() == {"total": 2, "judged": 1}
    assert resumed.view(resumed.sites[1])["verdict"] == "kasra"

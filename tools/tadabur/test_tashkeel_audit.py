"""Tests for the tashkeel audit's storage, comparison and UI logic (#60).

Exercises the adjudication round-trip, the blind view the listener is served, the
window-slice WAV encoding and the population-scaled over-strictness comparison — all
without binding a socket.
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from training.tashkeel_eval import DAMMA, FATHA, KASRA, MATCHED, OMITTED
from training.tashkeel_worklist import RECOVERED, REGRESSED, TashkeelSite, site_id, write_worklist

from .tashkeel_acceptance import compare, component_z, summarize_direction
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


def _strata(fatha: int = 0, damma: int = 0, kasra: int = 0) -> dict:
    return {"fatha": fatha, "damma": damma, "kasra": kasra}


def _population(recovered: dict, regressed: dict, vowels: int) -> dict:
    return {
        "reference_vowels": vowels,
        "strata": {RECOVERED: recovered, REGRESSED: regressed},
        RECOVERED: sum(recovered.values()),
        REGRESSED: sum(regressed.values()),
        "concordant": vowels - sum(recovered.values()) - sum(regressed.values()),
    }


def test_a_site_counts_as_over_strictness_only_when_the_reference_colour_was_heard():
    sites = [_site(1, RECOVERED, FATHA), _site(2, RECOVERED, FATHA), _site(3, RECOVERED, FATHA)]
    adjudications = {
        sites[0].site_id: _verdict(sites[0], "fatha"),
        sites[1].site_id: _verdict(sites[1], "kasra"),
        sites[2].site_id: _verdict(sites[2], UNCLEAR),
    }
    result = summarize_direction(sites, adjudications, RECOVERED, _strata(fatha=30))
    (stratum,) = [s for s in result.strata if s.vowel_name == "fatha"]
    assert (stratum.confirmed, stratum.said_otherwise, stratum.unclear) == (1, 1, 1)


def test_unclear_verdicts_leave_the_denominator_rather_than_counting_against_confirmation():
    # An inaudible recording is not evidence the reciter said the wrong vowel. Counting it
    # in the denominator would drag the estimated over-strictness toward zero.
    sites = [_site(i, RECOVERED, FATHA) for i in range(4)]
    adjudications = {sites[0].site_id: _verdict(sites[0], "fatha")}
    adjudications.update({s.site_id: _verdict(s, UNCLEAR) for s in sites[1:]})
    result = summarize_direction(sites, adjudications, RECOVERED, _strata(fatha=100))
    assert result.scoreable == 1
    assert result.estimate() == pytest.approx(100.0)


def test_unjudged_sites_are_not_counted_as_anything():
    sites = [_site(1, RECOVERED), _site(2, RECOVERED)]
    result = summarize_direction(sites, {sites[0].site_id: _verdict(sites[0], "fatha")},
                                 RECOVERED, _strata(fatha=10))
    assert result.audited == 1


def test_the_audited_share_is_scaled_onto_the_mined_population():
    # Half the audited recoveries were genuinely correct recitation, so half the 100 mined
    # ones are estimated to be -- reporting 2 would read as a census of the corpus.
    sites = [_site(i, RECOVERED) for i in range(4)]
    adjudications = {
        s.site_id: _verdict(s, "fatha" if i < 2 else "kasra") for i, s in enumerate(sites)
    }
    result = summarize_direction(sites, adjudications, RECOVERED, _strata(fatha=100))
    assert result.estimate() == pytest.approx(50.0)


def test_each_colour_is_scaled_onto_its_own_population_not_the_pooled_share():
    # The worklist samples equally per colour, so pooling weights colours by *sample* size.
    # Here every fatha recovery is genuine and every kasra one is not; with 1000 fatha and
    # 100 kasra in the corpus the answer is 1000, but a pooled 50% share would say 550.
    fathas = [_site(i, RECOVERED, FATHA) for i in range(4)]
    kasras = [_site(10 + i, RECOVERED, KASRA) for i in range(4)]
    adjudications = {s.site_id: _verdict(s, "fatha") for s in fathas}
    adjudications.update({s.site_id: _verdict(s, "damma") for s in kasras})
    result = summarize_direction(
        fathas + kasras, adjudications, RECOVERED, _strata(fatha=1000, kasra=100)
    )
    assert result.estimate() == pytest.approx(1000.0)


def test_a_colour_nobody_has_reached_yet_widens_the_interval_to_its_whole_population():
    sites = [_site(i, RECOVERED, FATHA) for i in range(4)]
    adjudications = {s.site_id: _verdict(s, "fatha") for s in sites}
    result = summarize_direction(
        sites, adjudications, RECOVERED, _strata(fatha=100, kasra=500)
    )
    low, high = result.bounds(component_z(2))
    assert low < 100 and high > 500


def test_the_gain_is_positive_when_the_base_falsely_rejects_more():
    sites = [_site(i, RECOVERED) for i in range(4)] + [_site(10 + i, REGRESSED) for i in range(4)]
    adjudications = {s.site_id: _verdict(s, "fatha") for s in sites[:4]}
    adjudications.update({s.site_id: _verdict(s, "kasra") for s in sites[4:]})
    report = compare(sites, adjudications, _population(_strata(fatha=400), _strata(fatha=40), 10000))
    # Every mined recovery was real over-strictness; no mined regression was.
    assert report["base_false_rejection_rate"] == pytest.approx(0.04)
    assert report["candidate_false_rejection_rate"] == pytest.approx(0.0)
    assert report["acceptance_gain"] > 0


def test_the_gain_interval_is_wider_than_the_point_estimate_on_both_sides():
    sites = [_site(i, RECOVERED) for i in range(4)] + [_site(10 + i, REGRESSED) for i in range(4)]
    adjudications = {s.site_id: _verdict(s, "fatha") for s in sites}
    report = compare(sites, adjudications, _population(_strata(fatha=400), _strata(fatha=40), 10000))
    low, high = report["acceptance_gain_ci95"]
    assert low < report["acceptance_gain"] < high


def test_component_bounds_are_widened_so_the_reported_interval_holds_at_95_percent():
    # Differencing two plain 95% bounds gives at most 90.25% joint coverage; six strata
    # would leave ~74%. Each component must be widened, never left at 1.96.
    assert component_z(1) == pytest.approx(1.95996, abs=1e-4)
    assert component_z(6) == pytest.approx(2.63826, abs=1e-4)
    with pytest.raises(ValueError):
        component_z(0)


def test_the_reported_interval_names_the_method_it_used():
    report = compare([], {}, _population(_strata(fatha=4), _strata(fatha=4), 100))
    assert report["interval_method"]["components"] == 6
    assert report["interval_method"]["component_z"] > 1.96


def test_a_worklist_and_summary_from_different_runs_is_an_error():
    sites = [_site(1, RECOVERED, KASRA)]
    adjudications = {sites[0].site_id: _verdict(sites[0], "kasra")}
    with pytest.raises(ValueError, match="different mining runs"):
        summarize_direction(sites, adjudications, RECOVERED, {"fatha": 10})


def test_an_unaudited_worklist_reports_pending_rather_than_a_verdict():
    sites = [_site(i, RECOVERED) for i in range(3)]
    report = compare(sites, {}, _population(_strata(fatha=300), _strata(fatha=30), 9000))
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


def test_a_clip_name_escaping_the_audio_dir_is_refused(tmp_path):
    # read_worklist validates field *names*, not contents, and this server is meant to be
    # bound to 0.0.0.0 -- a traversing clip name must not reach the filesystem.
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"RIFF....WAVEfmt ")
    audio_dir = tmp_path / "clips"
    audio_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        ClipCache(audio_dir).waveform("../secret.wav")


# --- UI state -------------------------------------------------------------------------


def _state(tmp_path, sites) -> AuditState:
    worklist = tmp_path / "worklist.jsonl"
    write_worklist(worklist, sites)
    return AuditState.load(worklist, tmp_path / "adjudications.jsonl", tmp_path)


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


def test_an_interrupted_write_cannot_destroy_the_verdicts_already_recorded(tmp_path):
    # The UI rewrites the whole file on every keystroke; writing in place would put the
    # accumulated audit inside each truncation window.
    path = tmp_path / "adjudications.jsonl"
    first = {a.site_id: a for a in [_verdict(_site(1, RECOVERED), "fatha")]}
    write_adjudications(path, first)
    original = path.read_bytes()

    class Exploding(dict):
        def __getitem__(self, key):
            raise RuntimeError("crash mid-write")

    with pytest.raises(RuntimeError):
        write_adjudications(path, Exploding({"x": None}))
    assert path.read_bytes() == original


def test_the_ui_exposes_no_result_route_to_probe(tmp_path):
    # A listener who can watch the recovered/regressed tallies move can submit a verdict,
    # see which way it pushed the gain, and revise it -- undoing the blinding entirely.
    assert not hasattr(AuditState, "results")
    source = (Path(__file__).parent / "tashkeel_audit_ui.py").read_text(encoding="utf-8")
    assert "/api/results" not in source


def test_the_state_resumes_from_whatever_the_file_already_holds(tmp_path):
    state = _state(tmp_path, [_site(1, RECOVERED), _site(2, REGRESSED)])
    state.record(state.sites[1].site_id, "kasra", "")
    resumed = AuditState.load(
        tmp_path / "worklist.jsonl", tmp_path / "adjudications.jsonl", tmp_path
    )
    assert resumed.progress() == {"total": 2, "judged": 1}
    assert resumed.view(resumed.sites[1])["verdict"] == "kasra"

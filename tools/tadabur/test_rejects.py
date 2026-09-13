"""Tests for the reject sink — cause labelling, the clean-re-read predicate, the writer.

The gate itself is used unchanged (``tadabur.scorer``); nothing here needs the GPU
model. Cause labelling is exercised both against real gate verdicts and against
hand-built :class:`~tadabur.scorer.GateResult` values, so the multi-cause and
added-shadda branches are covered without having to find a decode that provokes them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.rejects import (
    CAUSE_ADDED_SHADDA,
    CAUSE_INSERTION_RUN,
    CAUSE_LOW_RATIO,
    CAUSE_MIN_QUERY,
    CAUSE_NO_ALIGNMENT,
    CLEAN_RE_READ_MIN_RATIO,
    RejectRecord,
    RejectSink,
    build_reject_record,
    is_clean_re_read,
    read_reject_records,
    reject_causes,
    write_clip_wav,
)
from tadabur.scorer import BALANCED_SCORER, MAX_INSERTION_RUN, GateResult

# A 20-phoneme reference and a decode that repeats its phonemes 5..10 in the middle —
# the shape of a real re-read. The repeat is an interior insertion run of exactly
# MAX_INSERTION_RUN, and 20 matches less one 5-long affine gap over 25 query phonemes
# leaves match_ratio at 0.764, so it clears the clean-re-read ratio floor. Kept off the
# clip edges on purpose: a repeat at either end is trimmed by the local aligner and
# shows up as leading/trailing_trim, not as an insertion run.
REFERENCE = "بتثجحخدذرزسشصضطظعغفق"
RE_READ_DECODE = REFERENCE[:10] + REFERENCE[5:10] + REFERENCE[10:]


def _result(**overrides) -> GateResult:
    """A rejected GateResult with clean-re-read-shaped defaults, overridable per test."""
    fields = {
        "passed": False,
        "match_ratio": 0.8,
        "max_insertion_run": MAX_INSERTION_RUN,
        "leading_trim": 0,
        "trailing_trim": 0,
        "added_shadda": False,
    }
    return GateResult(**{**fields, **overrides})


# --- the clean-re-read predicate ---------------------------------------------------


def test_real_gate_rejects_a_repeated_span_as_a_clean_re_read():
    # The end-to-end claim ADR-0016 rests on: a recitation that matches its reference
    # except for a repeated phrase fails the gate on the insertion run alone, while
    # match_ratio stays high — which is exactly what makes it minable.
    result = BALANCED_SCORER.gate(RE_READ_DECODE, REFERENCE)

    assert not result.passed
    assert result.max_insertion_run == MAX_INSERTION_RUN
    assert result.match_ratio >= CLEAN_RE_READ_MIN_RATIO
    assert not result.added_shadda
    assert is_clean_re_read(
        result.match_ratio, result.max_insertion_run, result.added_shadda
    )


def test_a_short_insertion_run_is_not_a_clean_re_read():
    # One below the gate's own reject bar: such a clip would have passed, so by
    # construction it cannot be in the reject pile this predicate mines.
    assert not is_clean_re_read(0.9, MAX_INSERTION_RUN - 1, False)
    assert is_clean_re_read(0.9, MAX_INSERTION_RUN, False)


def test_ratio_floor_is_half_open_at_the_threshold():
    assert is_clean_re_read(CLEAN_RE_READ_MIN_RATIO, MAX_INSERTION_RUN, False)
    assert not is_clean_re_read(CLEAN_RE_READ_MIN_RATIO - 0.01, MAX_INSERTION_RUN, False)


def test_added_shadda_disqualifies_an_otherwise_clean_re_read():
    # A repeat plus a gemination the reference lacks is a mispronunciation as well as a
    # re-read, which is not the thing the word-space oracle can judge.
    assert not is_clean_re_read(0.9, MAX_INSERTION_RUN, True)


def test_record_exposes_the_predicate_over_its_own_fields():
    record = build_reject_record(
        audio_filename="r.wav",
        surah_ayah="3:82",
        reciter_id=7,
        ayah_duration_s=12.0,
        predicted=RE_READ_DECODE,
        result=BALANCED_SCORER.gate(RE_READ_DECODE, REFERENCE),
        scorer=BALANCED_SCORER,
    )

    assert record.is_clean_re_read
    assert record.causes == (CAUSE_INSERTION_RUN,)
    assert record.predicted_phonemes == RE_READ_DECODE
    assert record.ayah_duration_s == 12.0
    assert record.reciter_id == 7


# --- cause labelling ---------------------------------------------------------------


def test_a_passer_has_no_causes():
    result = BALANCED_SCORER.gate(REFERENCE, REFERENCE)
    assert result.passed
    assert reject_causes(REFERENCE, result, BALANCED_SCORER) == ()


def test_too_short_a_decode_is_labelled_min_query():
    result = BALANCED_SCORER.gate("بت", REFERENCE)
    assert reject_causes("بت", result, BALANCED_SCORER) == (CAUSE_MIN_QUERY,)


def test_normalization_collapse_can_make_a_long_decode_min_query():
    # Six identical phonemes normalize to one, so a decode that *looks* long enough is
    # still a min_query reject. The count has to be taken after normalization, as the
    # gate takes it, or this clip would be mislabelled.
    decode = "مممممم"
    result = BALANCED_SCORER.gate(decode, "بتثج")
    assert reject_causes(decode, result, BALANCED_SCORER) == (CAUSE_MIN_QUERY,)


def test_a_zero_ratio_decode_is_labelled_no_alignment():
    # A long-enough query with no positive-scoring alignment. Rare in practice — soft
    # pairs mean almost any pair of Arabic phonemes earns some partial credit — but it
    # is the gate's other early return, and the stored fields alone cannot tell it
    # apart from min_query, which is why the cause is recorded rather than derived.
    result = _result(match_ratio=0.0, max_insertion_run=0)
    assert reject_causes(REFERENCE, result, BALANCED_SCORER) == (CAUSE_NO_ALIGNMENT,)


def test_terminal_causes_are_reported_alone():
    # The gate returns early on a short query, so max_insertion_run is a default rather
    # than a measurement; reporting it as a cause would be reading noise.
    result = _result(match_ratio=0.0, max_insertion_run=0)
    assert reject_causes("بت", result, BALANCED_SCORER) == (CAUSE_MIN_QUERY,)


def test_a_low_ratio_decode_is_labelled_low_ratio():
    result = _result(match_ratio=0.4, max_insertion_run=0)
    assert reject_causes(REFERENCE, result, BALANCED_SCORER) == (CAUSE_LOW_RATIO,)


def test_added_shadda_is_labelled_on_its_own():
    result = _result(match_ratio=0.9, max_insertion_run=0, added_shadda=True)
    assert reject_causes(REFERENCE, result, BALANCED_SCORER) == (CAUSE_ADDED_SHADDA,)


def test_every_applicable_cause_is_reported():
    # Causes overlap; a breakdown that picked one "primary" cause would be inventing an
    # ordering the gate does not have.
    result = _result(match_ratio=0.3, max_insertion_run=9, added_shadda=True)
    assert reject_causes(REFERENCE, result, BALANCED_SCORER) == (
        CAUSE_LOW_RATIO,
        CAUSE_INSERTION_RUN,
        CAUSE_ADDED_SHADDA,
    )


# --- the sink ----------------------------------------------------------------------


def _reject(name: str, **overrides) -> RejectRecord:
    fields = {
        "audio_filename": name,
        "surah_ayah": "3:82",
        "reciter_id": 88,
        "ayah_duration_s": 11.5,
        "match_ratio": 0.8,
        "max_insertion_run": 7,
        "leading_trim": 0,
        "trailing_trim": 2,
        "added_shadda": False,
        "predicted_phonemes": "بتثج",
        "causes": (CAUSE_INSERTION_RUN,),
    }
    return RejectRecord(**{**fields, **overrides})


def test_sink_appends_records_and_counts_them(tmp_path):
    path = tmp_path / "rejects.jsonl"
    with RejectSink.open(path) as sink:
        sink.append_batch([_reject("a.wav"), _reject("b.wav")])
        assert sink.rejects_written == 2

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert [r["audio_filename"] for r in rows] == ["a.wav", "b.wav"]
    assert rows[0]["causes"] == [CAUSE_INSERTION_RUN]
    assert rows[0]["max_insertion_run"] == 7


def test_sink_carries_the_whole_gate_result(tmp_path):
    # The point of the sink: nothing the gate computed is summarised away, so any
    # predicate over rejects can be re-derived offline without another GPU pass.
    path = tmp_path / "rejects.jsonl"
    with RejectSink.open(path) as sink:
        sink.append_batch([_reject("a.wav", leading_trim=13, added_shadda=True)])

    (row,) = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert set(row) == {
        "audio_filename", "surah_ayah", "reciter_id", "ayah_duration_s",
        "match_ratio", "max_insertion_run", "leading_trim", "trailing_trim",
        "added_shadda", "predicted_phonemes", "causes",
    }
    assert row["leading_trim"] == 13
    assert row["added_shadda"] is True


def test_reopening_the_sink_dedupes_a_replayed_batch(tmp_path):
    # The crash window: a batch's rejects were written but the checkpoint bump is
    # replayed. The seen-set must keep the sink duplicate-free, like the manifest.
    path = tmp_path / "rejects.jsonl"
    with RejectSink.open(path) as sink:
        sink.append_batch([_reject("a.wav"), _reject("b.wav")])

    with RejectSink.open(path) as sink:
        assert sink.rejects_written == 2
        sink.append_batch([_reject("a.wav"), _reject("b.wav"), _reject("c.wav")])
        assert sink.rejects_written == 3

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert [r["audio_filename"] for r in rows] == ["a.wav", "b.wav", "c.wav"]


def test_records_round_trip_through_the_reader(tmp_path):
    path = tmp_path / "rejects.jsonl"
    original = [_reject("a.wav"), _reject("b.wav", match_ratio=0.2, causes=(CAUSE_LOW_RATIO,))]
    with RejectSink.open(path) as sink:
        sink.append_batch(original)

    assert read_reject_records(path) == original


def test_reader_tolerates_a_row_without_causes(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps(
            {
                "audio_filename": "old.wav", "surah_ayah": "1:1", "reciter_id": 3,
                "ayah_duration_s": 5.0, "match_ratio": 0.8, "max_insertion_run": 6,
                "leading_trim": 0, "trailing_trim": 0, "added_shadda": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (record,) = read_reject_records(path)
    assert record.causes == ()
    assert record.predicted_phonemes == ""


# --- staged audio ------------------------------------------------------------------


def test_clip_wav_is_written_at_16k_under_the_clips_own_name(tmp_path):
    waveform = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)

    path = write_clip_wav(tmp_path / "staged", "c.wav", waveform)

    assert path == tmp_path / "staged" / "c.wav"
    read_back, rate = sf.read(path, dtype="float32")
    assert rate == TARGET_SAMPLE_RATE
    assert read_back == pytest.approx(waveform, abs=1e-6)


def test_rewriting_a_clip_reproduces_identical_bytes(tmp_path):
    # A replayed batch re-stages its audio; the directory must stay idempotent.
    waveform = np.linspace(-0.5, 0.5, 4000, dtype=np.float32)
    first = write_clip_wav(tmp_path / "staged", "c.wav", waveform).read_bytes()
    second = write_clip_wav(tmp_path / "staged", "c.wav", waveform).read_bytes()
    assert first == second

"""Tests for waqf-aware segmentation, realized references, and the offsets manifest."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import tadabur.waqf_segments as ws
from tadabur.manifest import ManifestRecord
from tadabur.waqf_segments import (
    SegmentRecord,
    WordAlignment,
    build_clip_segments,
    clip_speech_rms,
    hafs_phonetizer,
    is_silent_gap,
    parse_word_alignments,
    read_segment_manifest,
    shadda_contrast_report,
    split_at_pauses,
    write_segment_manifest,
)


def _align(word: str, start: float, end: float) -> WordAlignment:
    return WordAlignment(word=word, start_s=start, end_s=end)


def _passing(name: str, surah_ayah: str, predicted: str = "") -> ManifestRecord:
    return ManifestRecord(
        audio_filename=name,
        surah_ayah=surah_ayah,
        match_ratio=0.9,
        ayah_duration_s=8.0,
        reciter_id=106,
        predicted_phonemes=predicted,
    )


# --- parsing ----------------------------------------------------------------


def test_parse_word_alignments_reads_json_string():
    metadata = (
        '{"word_alignments": [{"word": "a", "start": 0.0, "end": 1.0}, '
        '{"word": "b", "start": 1.2, "end": 2.0}]}'
    )
    alignments = parse_word_alignments(metadata)
    assert alignments == [_align("a", 0.0, 1.0), _align("b", 1.2, 2.0)]


def test_parse_word_alignments_missing_field_fails_loudly():
    with pytest.raises(ValueError):
        parse_word_alignments('{"text_ar_simple": "x"}')


# --- pause detection --------------------------------------------------------


def test_continuous_recitation_is_one_segment():
    # Overlapping/abutting words (wasl) never split.
    alignments = [_align("a", 0.0, 1.34), _align("b", 1.26, 3.9), _align("c", 3.7, 5.1)]
    assert split_at_pauses(alignments, 0.25) == [(0, 3)]


def test_gap_above_threshold_splits():
    alignments = [_align("a", 0.0, 1.0), _align("b", 1.4, 2.0)]  # gap 0.4
    assert split_at_pauses(alignments, 0.25) == [(0, 1), (1, 2)]


def test_gap_below_threshold_does_not_split():
    alignments = [_align("a", 0.0, 1.0), _align("b", 1.1, 2.0)]  # gap 0.1
    assert split_at_pauses(alignments, 0.25) == [(0, 2)]


def test_gap_exactly_at_threshold_splits():
    alignments = [_align("a", 0.0, 1.0), _align("b", 1.25, 2.0)]  # gap 0.25
    assert split_at_pauses(alignments, 0.25) == [(0, 1), (1, 2)]


def test_multiple_pauses_make_multiple_segments():
    alignments = [
        _align("a", 0.0, 1.0),
        _align("b", 1.5, 2.0),  # pause before b
        _align("c", 2.1, 3.0),  # continuous
        _align("d", 3.6, 4.0),  # pause before d
    ]
    assert split_at_pauses(alignments, 0.25) == [(0, 1), (1, 3), (3, 4)]


def test_empty_alignments_yield_no_segments():
    assert split_at_pauses([], 0.25) == []


# --- acoustic silence gate (madd vs waqf) -----------------------------------


def _tone(sample_rate: int, seconds: float, amplitude: float) -> np.ndarray:
    """A voiced-like sine burst; ``amplitude`` scales its loudness (0 = silence)."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def test_is_silent_gap_true_for_silence():
    sr = 16000
    waveform = np.concatenate([_tone(sr, 0.5, 0.5), np.zeros(int(0.4 * sr), np.float32),
                               _tone(sr, 0.5, 0.5)])
    clip_rms = clip_speech_rms(waveform)
    # The [0.5, 0.9) window is silent → a real stop.
    assert is_silent_gap(waveform, sr, 0.5, 0.9, clip_rms, 0.15) is True


def test_is_silent_gap_false_for_voiced_madd():
    sr = 16000
    # The "gap" is a held tone (a madd), not silence, at the same amplitude as speech.
    waveform = np.concatenate([_tone(sr, 0.5, 0.5), _tone(sr, 0.4, 0.5),
                               _tone(sr, 0.5, 0.5)])
    clip_rms = clip_speech_rms(waveform)
    assert is_silent_gap(waveform, sr, 0.5, 0.9, clip_rms, 0.15) is False


def test_is_silent_gap_false_for_zero_width_window():
    sr = 16000
    waveform = _tone(sr, 1.0, 0.5)
    assert is_silent_gap(waveform, sr, 0.5, 0.5, clip_speech_rms(waveform), 0.15) is False


def test_acoustic_gate_keeps_madd_in_one_segment():
    # Two words with a >0.25 s timestamp gap that is actually a sustained madd:
    # the audio confirmation must NOT split them (one wasl segment).
    sr = 16000
    phonetize = hafs_phonetizer()
    words = ["عَنِ", "ٱلنَّبَإِ", "ٱلْعَظِيمِ"]
    alignments = [
        _align("عَنِ", 0.0, 1.0),
        _align("ٱلنَّبَإِ", 1.1, 2.0),
        _align("ٱلْعَظِيمِ", 2.5, 3.5),  # 0.5 s timestamp gap before the last word
    ]
    voiced = _tone(sr, 4.0, 0.5)  # continuous voicing across the whole clip
    record = _passing("clip.wav", "78:2")

    segments = build_clip_segments(
        record, alignments, words, phonetize, 0.25, waveform=voiced, sample_rate=sr
    )
    assert len(segments) == 1
    assert segments[0].realized_reference_phonemes == phonetize(" ".join(words))


def test_acoustic_gate_splits_on_true_silence():
    # Same timestamps, but the gap [2.0, 2.5) is silent → a real waqf, so it splits.
    sr = 16000
    phonetize = hafs_phonetizer()
    words = ["عَنِ", "ٱلنَّبَإِ", "ٱلْعَظِيمِ"]
    alignments = [
        _align("عَنِ", 0.0, 1.0),
        _align("ٱلنَّبَإِ", 1.1, 2.0),
        _align("ٱلْعَظِيمِ", 2.5, 3.5),
    ]
    waveform = np.concatenate([
        _tone(sr, 2.0, 0.5),               # [0.0, 2.0) recitation
        np.zeros(int(0.5 * sr), np.float32),  # [2.0, 2.5) silent stop
        _tone(sr, 1.0, 0.5),               # [2.5, 3.5) recitation
    ])
    record = _passing("clip.wav", "78:2")

    segments = build_clip_segments(
        record, alignments, words, phonetize, 0.25, waveform=waveform, sample_rate=sr
    )
    assert [(s.word_start, s.word_end) for s in segments] == [(0, 2), (2, 3)]


# --- realized reference (waqf vs wasl, real phonetizer) ---------------------


def test_single_segment_matches_full_ayah_reference():
    # 78:2 — عَنِ ٱلنَّبَإِ ٱلْعَظِيمِ, recited continuously → one segment whose
    # realized reference is the whole-ayah phonetization.
    phonetize = hafs_phonetizer()
    words = ["عَنِ", "ٱلنَّبَإِ", "ٱلْعَظِيمِ"]
    alignments = [_align(w, i, i + 0.9) for i, w in enumerate(words)]
    record = _passing("clip.wav", "78:2")

    segments = build_clip_segments(record, alignments, words, phonetize, 0.25)
    assert len(segments) == 1
    assert segments[0].realized_reference_phonemes == phonetize(" ".join(words))


def test_split_puts_terminal_word_in_waqf_form():
    # A pause after ٱلنَّبَإِ: the first segment's terminal word takes waqf form
    # (final kasra dropped → ...نَبَء, not ...نَبَءِ), interior word stays wasl.
    phonetize = hafs_phonetizer()
    words = ["عَنِ", "ٱلنَّبَإِ", "ٱلْعَظِيمِ"]
    alignments = [
        _align("عَنِ", 0.0, 1.0),
        _align("ٱلنَّبَإِ", 1.1, 2.0),
        _align("ٱلْعَظِيمِ", 2.5, 3.5),  # pause 0.5 before the last word
    ]
    record = _passing("clip.wav", "78:2")

    segments = build_clip_segments(record, alignments, words, phonetize, 0.25)

    assert [(s.word_start, s.word_end) for s in segments] == [(0, 2), (2, 3)]
    assert segments[0].realized_reference_phonemes == phonetize("عَنِ ٱلنَّبَإِ")
    assert segments[1].realized_reference_phonemes == phonetize("ٱلْعَظِيمِ")
    # The whole-ayah (wasl) reference differs from the concatenated realized one:
    # the phantom pre-waqf form is exactly what this stage removes.
    full = phonetize(" ".join(words))
    realized = " ".join(s.realized_reference_phonemes for s in segments)
    assert realized != full
    # Segment offsets come from the alignment.
    assert (segments[0].start_s, segments[0].end_s) == (0.0, 2.0)
    assert (segments[1].start_s, segments[1].end_s) == (2.5, 3.5)


def test_build_requires_matching_word_counts():
    phonetize = hafs_phonetizer()
    alignments = [_align("a", 0.0, 1.0)]
    with pytest.raises(AssertionError):
        build_clip_segments(_passing("c.wav", "78:2"), alignments, ["a", "b"], phonetize, 0.25)


# --- manifest determinism / idempotency -------------------------------------


def _segment(name: str, index: int, ref: str = "ءَبت") -> SegmentRecord:
    return SegmentRecord(
        audio_filename=name,
        surah_ayah="78:2",
        reciter_id=106,
        segment_index=index,
        word_start=index,
        word_end=index + 1,
        start_s=float(index),
        end_s=float(index) + 1.0,
        realized_reference_phonemes=ref,
    )


def test_manifest_round_trips(tmp_path):
    path = tmp_path / "segments.jsonl"
    records = [_segment("a.wav", 0), _segment("a.wav", 1)]
    write_segment_manifest(path, records)
    assert read_segment_manifest(path) == records


def test_manifest_is_deterministic_regardless_of_input_order(tmp_path):
    records = [_segment("b.wav", 0), _segment("a.wav", 1), _segment("a.wav", 0)]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_segment_manifest(first, records)
    write_segment_manifest(second, list(reversed(records)))
    assert first.read_bytes() == second.read_bytes()
    # Sorted by (audio_filename, segment_index).
    keys = [(r.audio_filename, r.segment_index) for r in read_segment_manifest(first)]
    assert keys == [("a.wav", 0), ("a.wav", 1), ("b.wav", 0)]


def test_manifest_rewrite_is_idempotent(tmp_path):
    path = tmp_path / "segments.jsonl"
    records = [_segment("a.wav", 1), _segment("a.wav", 0)]
    write_segment_manifest(path, records)
    before = path.read_bytes()
    write_segment_manifest(path, records)
    assert path.read_bytes() == before


def test_manifest_is_utf8_and_preserves_arabic(tmp_path):
    path = tmp_path / "segments.jsonl"
    write_segment_manifest(path, [_segment("a.wav", 0, ref="عَنِ ننننَبَء")])
    text = path.read_text(encoding="utf-8")
    assert "عَنِ ننننَبَء" in text  # not \u-escaped


# --- shadda before/after report ---------------------------------------------


def test_report_counts_phantom_pre_waqf_shadda_removed():
    # Model decode has a single ب where the full-ayah (wasl) reference doubles it
    # (a phantom pre-waqf gemination); the realized (segmented) reference drops the
    # doubling, so the shadda contrast is present "before" and gone "after".
    passing = [_passing("clip.wav", "1:1", predicted="ءَبَتَ")]
    segments = [
        _segment("clip.wav", 0, ref="ءَب"),
        _segment("clip.wav", 1, ref="تَ"),
    ]
    references = {"1:1": "ءببت"}  # normalized full-ayah reference (doubled ب)

    report = shadda_contrast_report(passing, segments, references)
    assert report["clips_with_waqf"] == 1
    assert report["shadda_before"] == 1
    assert report["shadda_after"] == 0
    assert report["phantom_removed"] == 1


def test_report_ignores_single_segment_clips():
    passing = [_passing("clip.wav", "1:1", predicted="ءَب")]
    segments = [_segment("clip.wav", 0, ref="ءَب")]
    report = shadda_contrast_report(passing, segments, {"1:1": "ءَب"})
    assert report == {}


# --- missing passing clips (data-integrity guard) ---------------------------


def test_full_build_raises_when_a_passing_clip_is_never_streamed(tmp_path, monkeypatch):
    # A full build (no --limit) whose stream drops a passing clip would emit a
    # partial label source; that must fail loudly, naming the misses.
    monkeypatch.setattr(ws, "_stream_passing_rows", lambda *a, **k: iter(()))
    passing = [_passing("a.wav", "78:2"), _passing("b.wav", "78:2")]
    with pytest.raises(ValueError, match="were not found"):
        ws.build_segments(passing, lambda text: "x", audio_dir=tmp_path)


def test_limited_run_records_missing_due_to_limit_instead_of_raising(tmp_path, monkeypatch):
    # A --limit smoke run may legitimately stop before reaching every clip; the
    # shortfall is tallied, not raised.
    monkeypatch.setattr(ws, "_stream_passing_rows", lambda *a, **k: iter(()))
    passing = [_passing("a.wav", "78:2"), _passing("b.wav", "78:2")]
    records, skips = ws.build_segments(
        passing, lambda text: "x", audio_dir=tmp_path, limit=1
    )
    assert records == []
    assert skips["missing_due_to_limit"] == 2


def test_found_clips_are_not_reported_missing(tmp_path, monkeypatch):
    record = _passing("a.wav", "78:2")
    row = {
        "metadata": '{"word_alignments": [{"word": "عَنِ", "start": 0.0, "end": 1.0}]}',
        "audio": {"bytes": b""},
    }
    monkeypatch.setattr(ws, "_stream_passing_rows", lambda *a, **k: iter([(row, record)]))
    monkeypatch.setattr(ws, "_uthmani_words", lambda surah_ayah: ["عَنِ"])
    monkeypatch.setattr(ws, "decode_to_mono_16k", lambda b: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(ws, "_save_local_clip", lambda *a, **k: None)

    records, skips = ws.build_segments([record], lambda text: "X", audio_dir=tmp_path)
    assert "missing_due_to_limit" not in skips
    assert len(records) == 1


# --- torch-free offline stage -----------------------------------------------


def test_importing_waqf_segments_does_not_import_torch():
    # The stage must run in the plain macOS/CPU env without pulling in the GPU
    # inference path; a fresh interpreter proves nothing imports torch.
    code = (
        "import sys; import tadabur.waqf_segments; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr

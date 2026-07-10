"""Tests for waqf-aware segmentation, realized references, and the offsets manifest."""

from __future__ import annotations

import pytest

from tadabur.manifest import ManifestRecord
from tadabur.waqf_segments import (
    SegmentRecord,
    WordAlignment,
    build_clip_segments,
    hafs_phonetizer,
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

"""Tests for the whole-clip windowed CTC labels, preflight, and reciter split (#25).

Pure logic over the segment manifest + per-clip status sidecar — no GPU, no
quran-transcript. The fixtures below stand in for what ``tadabur.segment_score`` emits:
kept segments (with their word ranges + realized references) and one status per clip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tadabur.clip_status import ClipStatus, read_clip_status, write_clip_status
from training.waqf_distill import WindowContract, muaalem_lattice_length
from training.windowed_labels import (
    EXCLUDE_DROPPED_SEGMENT,
    EXCLUDE_EMPTY_WINDOW,
    EXCLUDE_OVER_LONG,
    EXCLUDE_TARGET_TOO_LONG,
    Segment,
    assert_no_reciter_leakage,
    build_clip_windows,
    build_windowed_labels,
    read_segments,
    split_by_reciter,
    write_labels,
)

CONTRACT = WindowContract()  # 5 s window (250 frames), frozen 4 s hop (200 frames)


def _seg(clip, index, w0, w1, s0, s1, ref, reciter=1, surah="78:2"):
    return Segment(
        clip_audio_filename=clip,
        surah_ayah=surah,
        reciter_id=reciter,
        segment_index=index,
        word_start=w0,
        word_end=w1,
        start_s=s0,
        end_s=s1,
        reference_phonemes=ref,
    )


def _status(clip, n_words, duration_s, reciter=1, surah="78:2", skip=None):
    return ClipStatus(
        audio_filename=clip,
        surah_ayah=surah,
        reciter_id=reciter,
        n_words=n_words,
        duration_s=duration_s,
        skip_reason=skip,
    )


# --- single-window happy path ------------------------------------------------


def test_short_single_segment_makes_one_full_window():
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت")]
    status = _status("a.wav", n_words=3, duration_s=4.0)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert reason is None
    assert len(labels) == 1
    label = labels[0]
    assert label.phoneme_label == "ءبت"
    assert (label.word_start, label.word_end) == (0, 3)
    assert label.segment_indices == (0,)
    assert label.start_sample == 0
    assert label.logit_frames == muaalem_lattice_length(label.feature_frames)
    assert len(label.phoneme_label) < label.logit_frames


def test_multi_segment_window_concatenates_realized_references():
    # Three short segments inside one 4.5 s window → one window owns all three, its label
    # is their references concatenated in order with contiguous word coverage.
    segs = [
        _seg("a.wav", 0, 0, 1, 0.0, 1.5, "ءا"),
        _seg("a.wav", 1, 1, 2, 1.5, 3.0, "بب"),
        _seg("a.wav", 2, 2, 4, 3.0, 4.4, "تت"),
    ]
    status = _status("a.wav", n_words=4, duration_s=4.4)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert reason is None
    assert len(labels) == 1
    assert labels[0].phoneme_label == "ءاببتت"
    assert labels[0].segment_indices == (0, 1, 2)
    assert (labels[0].word_start, labels[0].word_end) == (0, 4)


# --- two windows, center-band ownership --------------------------------------


def test_two_segments_split_across_two_windows_by_center_band():
    # 6 s recitation with an interior pause at 3 s → two segments, each owned by the
    # window whose center band holds its midpoint (window 0 ← [0,3 s], window 1 ← [3,6 s]).
    segs = [
        _seg("a.wav", 0, 0, 2, 0.0, 3.0, "ءبت"),
        _seg("a.wav", 1, 2, 4, 3.0, 6.0, "جحخ"),
    ]
    status = _status("a.wav", n_words=4, duration_s=6.0)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert reason is None
    assert [l.window_index for l in labels] == [0, 1]
    assert [l.phoneme_label for l in labels] == ["ءبت", "جحخ"]
    assert [(l.word_start, l.word_end) for l in labels] == [(0, 2), (2, 4)]
    # The tail window is shorter than a full window; its 40 ms length is checked too.
    assert labels[1].feature_frames < labels[0].feature_frames


# --- clip-level eligibility --------------------------------------------------


def test_skip_reason_excludes_whole_clip():
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت")]
    status = _status("a.wav", n_words=3, duration_s=4.0, skip="repeated_recitation")

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == "repeated_recitation"


def test_dropped_segment_leaves_word_gap_and_excludes_clip():
    # A middle segment [2,4) was dropped by the scorer, so the kept segments no longer
    # cover [0, n_words) contiguously — the clip is ineligible.
    segs = [
        _seg("a.wav", 0, 0, 2, 0.0, 2.0, "ءب"),
        _seg("a.wav", 2, 4, 6, 4.0, 6.0, "تث"),
    ]
    status = _status("a.wav", n_words=6, duration_s=6.0)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_DROPPED_SEGMENT


def test_trailing_dropped_segment_excludes_clip():
    segs = [_seg("a.wav", 0, 0, 2, 0.0, 3.0, "ءب")]  # n_words says there is a word 2..3
    status = _status("a.wav", n_words=3, duration_s=3.0)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_DROPPED_SEGMENT


def test_over_long_recitation_is_flagged_not_truncated():
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت")]
    status = _status("a.wav", n_words=3, duration_s=4.0)

    # 4 s recitation is 200 feature frames; a cap of 100 makes it over-long.
    labels, reason = build_clip_windows(segs, status, CONTRACT, cap_feature_frames=100)

    assert labels == []
    assert reason == EXCLUDE_OVER_LONG


def test_long_single_segment_leaves_empty_window():
    # A 6 s single-breath run with no interior pause: its whole audio is one segment, so
    # the second needed window has no owned segment → flagged, not mislabelled.
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 6.0, "ءبت")]
    status = _status("a.wav", n_words=3, duration_s=6.0)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_EMPTY_WINDOW


def test_target_longer_than_logit_frames_excludes_clip():
    # A ~0.5 s window (25 feature frames → 13 logit frames) with a 20-phoneme label can't
    # be a valid CTC target (target_len must be < logit_frames).
    segs = [_seg("a.wav", 0, 0, 1, 0.0, 0.5, "ءبتثجحخدذرزسشصضط")]
    status = _status("a.wav", n_words=1, duration_s=0.5)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_TARGET_TOO_LONG


# --- corpus build + exclusion report -----------------------------------------


def test_build_windowed_labels_reports_exclusions_by_reason():
    segs = [
        _seg("keep.wav", 0, 0, 3, 0.0, 4.0, "ءبت", reciter=1),
        _seg("skip.wav", 0, 0, 3, 0.0, 4.0, "ءبت", reciter=2),
        _seg("gap.wav", 0, 0, 2, 0.0, 3.0, "ءب", reciter=3),
    ]
    statuses = [
        _status("keep.wav", 3, 4.0, reciter=1),
        _status("skip.wav", 3, 4.0, reciter=2, skip="low_alignment"),
        _status("gap.wav", 3, 3.0, reciter=3),
        _status("missing.wav", 3, 0.0, reciter=4, skip="clip_missing"),
    ]

    built = build_windowed_labels(segs, statuses, CONTRACT)

    assert built.clips_kept == 1
    assert {c for c, _ in built.exclusions} == {"skip.wav", "gap.wav", "missing.wav"}
    assert dict(built.reasons) == {
        "low_alignment": 1,
        EXCLUDE_DROPPED_SEGMENT: 1,
        "clip_missing": 1,
    }


def test_build_windowed_labels_is_deterministic():
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت")]
    statuses = [_status("a.wav", 3, 4.0)]

    first = build_windowed_labels(segs, statuses, CONTRACT)
    second = build_windowed_labels(segs, statuses, CONTRACT)

    assert first.labels == second.labels


def test_orphan_segment_without_status_raises():
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت")]
    with pytest.raises(ValueError, match="no clip-status"):
        build_windowed_labels(segs, [], CONTRACT)


# --- whole-clip reciter split ------------------------------------------------


def _labels_for(clips):
    segs = [_seg(clip, 0, 0, 3, 0.0, 4.0, "ءبت", reciter=rec) for clip, rec in clips]
    statuses = [_status(clip, 3, 4.0, reciter=rec) for clip, rec in clips]
    return build_windowed_labels(segs, statuses, CONTRACT).labels


def test_reciter_split_has_no_leakage():
    labels = _labels_for(
        [("a.wav", 1), ("b.wav", 1), ("c.wav", 2), ("d.wav", 3), ("e.wav", 4)]
    )

    train, val = split_by_reciter(labels, val_fraction=0.4, seed=0)

    proof = assert_no_reciter_leakage(train, val)
    assert proof["val_windows"] > 0 and proof["train_windows"] > 0
    # Reciter 1 owns two clips (a.wav, b.wav); both must resolve to the same partition.
    val_clips = {l.clip_audio_filename for l in val}
    train_clips = {l.clip_audio_filename for l in train}
    assert {"a.wav", "b.wav"} <= val_clips or {"a.wav", "b.wav"} <= train_clips


def test_reciter_split_is_deterministic_by_seed():
    labels = _labels_for([("a.wav", 1), ("b.wav", 2), ("c.wav", 3), ("d.wav", 4)])

    t1, v1 = split_by_reciter(labels, 0.25, seed=7)
    t2, v2 = split_by_reciter(labels, 0.25, seed=7)
    assert [l.clip_audio_filename for l in v1] == [l.clip_audio_filename for l in v2]


def test_zero_val_fraction_keeps_all_in_train():
    labels = _labels_for([("a.wav", 1), ("b.wav", 2)])
    train, val = split_by_reciter(labels, 0.0, seed=0)
    assert val == [] and len(train) == len(labels)


# --- IO round-trips ----------------------------------------------------------


def test_read_segments_round_trips_manifest_rows(tmp_path: Path):
    row = {
        "clip_audio_filename": "a.wav", "surah_ayah": "78:2", "reciter_id": 5,
        "segment_index": 0, "word_start": 0, "word_end": 3,
        "start_s": 0.0, "end_s": 4.0, "reference_phonemes": "ءبت",
    }
    path = tmp_path / "segments.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    segments = read_segments(path)

    assert segments == [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت", reciter=5)]


def test_clip_status_sidecar_round_trips(tmp_path: Path):
    statuses = [
        _status("b.wav", 3, 4.0, reciter=2, skip="low_alignment"),
        _status("a.wav", 5, 6.0, reciter=1),
    ]
    path = tmp_path / "status.jsonl"
    write_clip_status(path, statuses)

    loaded = read_clip_status(path)

    # Written key-sorted by audio_filename → deterministic order.
    assert [s.audio_filename for s in loaded] == ["a.wav", "b.wav"]
    assert loaded[1].skip_reason == "low_alignment"


def test_write_labels_is_sorted_and_tagged(tmp_path: Path):
    labels = _labels_for([("b.wav", 1), ("a.wav", 2)])
    path = tmp_path / "labels.jsonl"

    write_labels(path, labels, "train")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["clip_audio_filename"] for r in rows] == ["a.wav", "b.wav"]
    assert all(r["split"] == "train" for r in rows)
    assert all(isinstance(r["segment_indices"], list) for r in rows)

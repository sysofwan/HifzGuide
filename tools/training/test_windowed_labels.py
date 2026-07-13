"""Tests for the whole-clip windowed CTC labels, preflight, and reciter split (#25).

Pure logic over the segment manifest + per-clip status sidecar — no GPU, no
quran-transcript. The fixtures below stand in for what ``tadabur.segment_score`` emits:
kept segments (with their word ranges + realized references) and one status per clip
(carrying the recitation span both label artifacts window over). The waqf-side windowing
(:func:`training.waqf_distill.slice_recitation_windows`) is numpy-only, so the shared-grid
identity is checked here too without loading the VAD.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tadabur.clip_status import ClipStatus, read_clip_status, write_clip_status
from training.waqf_distill import (
    TARGET_SAMPLE_RATE,
    WindowContract,
    muaalem_lattice_length,
    recitation_window_span,
    slice_recitation_windows,
)
from training.windowed_labels import (
    EXCLUDE_DROPPED_SEGMENT,
    EXCLUDE_EMPTY_WINDOW,
    EXCLUDE_OVER_LONG,
    EXCLUDE_SEGMENT_CROSSES_WINDOW,
    EXCLUDE_TARGET_TOO_LONG,
    Segment,
    assert_no_reciter_leakage,
    build_clip_windows,
    build_windowed_labels,
    main,
    read_labels,
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


def _status(clip, n_words, duration_s, rec_start=0.0, rec_end=None, reciter=1,
            surah="78:2", skip=None):
    return ClipStatus(
        audio_filename=clip,
        surah_ayah=surah,
        reciter_id=reciter,
        n_words=n_words,
        duration_s=duration_s,
        recitation_start_s=rec_start,
        recitation_end_s=duration_s if rec_end is None else rec_end,
        skip_reason=skip,
    )


def _status_for(clip, segs, n_words, **kw):
    """Status whose recitation span is the first/last segment edges — the pipeline's rule."""
    return _status(clip, n_words, segs[-1].end_s, rec_start=segs[0].start_s,
                   rec_end=segs[-1].end_s, **kw)


# --- single-window happy path ------------------------------------------------


def test_short_single_segment_makes_one_full_window():
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت")]
    status = _status_for("a.wav", segs, n_words=3)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert reason is None
    assert len(labels) == 1
    label = labels[0]
    assert label.phoneme_label == "ءبت"
    assert (label.word_start, label.word_end) == (0, 3)
    assert label.segment_indices == (0,)
    assert label.start_sample == 0
    assert label.recitation_start_sample == 0
    assert label.logit_frames == muaalem_lattice_length(label.feature_frames)
    assert len(label.phoneme_label) < label.logit_frames


def test_multi_segment_window_concatenates_realized_references():
    # Three short segments inside one <5 s window → one window owns all three, its label
    # is their references concatenated in order with contiguous word coverage. Interior
    # waqf is retained because the whole recitation fits a single fixed window.
    segs = [
        _seg("a.wav", 0, 0, 1, 0.0, 1.5, "ءا"),
        _seg("a.wav", 1, 1, 2, 1.5, 3.0, "بب"),
        _seg("a.wav", 2, 2, 4, 3.0, 4.4, "تت"),
    ]
    status = _status_for("a.wav", segs, n_words=4)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert reason is None
    assert len(labels) == 1
    assert labels[0].phoneme_label == "ءاببتت"
    assert labels[0].segment_indices == (0, 1, 2)
    assert (labels[0].word_start, labels[0].word_end) == (0, 4)


# --- multi-window: a waqf pause spanning the window overlap keeps the clip ----


def test_multi_window_kept_when_waqf_pause_covers_the_overlap():
    # 8 s recitation → two windows ([0,5 s], [4,8 s]) sharing the [4,5 s] overlap. A long
    # waqf pause [3.9,5.1 s] covers that whole overlap, so neither segment crosses a window
    # edge: seg0 lives only in window 0, seg1 only in window 1. Both are kept.
    segs = [
        _seg("a.wav", 0, 0, 2, 0.0, 3.9, "ءبت"),
        _seg("a.wav", 1, 2, 4, 5.1, 8.0, "جحخ"),
    ]
    status = _status_for("a.wav", segs, n_words=4)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert reason is None
    assert [l.window_index for l in labels] == [0, 1]
    assert [l.phoneme_label for l in labels] == ["ءبت", "جحخ"]
    assert [(l.word_start, l.word_end) for l in labels] == [(0, 2), (2, 4)]
    # Clip-relative start samples on the frozen 4 s hop (64000 samples).
    assert [l.start_sample for l in labels] == [0, 64000]


# --- boundary/straddle cases: a crossing segment excludes the clip -----------


def test_segment_crossing_a_window_boundary_excludes_the_clip():
    # The reviewer's corruption case: window 0 spans 0-5 s and segment 1 spans 3-6 s, so
    # window 0's audio 3-5 s holds a partly-spoken word. Segment-level timing cannot prove
    # a full target for it, so the whole clip is excluded rather than mislabelled.
    segs = [
        _seg("a.wav", 0, 0, 2, 0.0, 3.0, "ءبت"),
        _seg("a.wav", 1, 2, 4, 3.0, 6.0, "جحخ"),
    ]
    status = _status_for("a.wav", segs, n_words=4)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_SEGMENT_CROSSES_WINDOW


def test_segment_starting_before_a_window_end_and_extending_past_excludes():
    # A segment [4,7 s] starts before window 0's end (5 s) but reaches past it → its audio
    # 4-5 s is in window 0 with only a partial word, so the clip is excluded.
    segs = [
        _seg("a.wav", 0, 0, 2, 0.0, 4.0, "ءب"),
        _seg("a.wav", 1, 2, 4, 4.0, 7.0, "جح"),
    ]
    status = _status_for("a.wav", segs, n_words=4)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_SEGMENT_CROSSES_WINDOW


def test_long_single_segment_crosses_the_second_window():
    # A 6 s single-breath run (one segment, no interior pause): its audio crosses window 0's
    # 5 s edge, so it cannot be cleanly labelled and the clip is excluded.
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 6.0, "ءبت")]
    status = _status_for("a.wav", segs, n_words=3)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_SEGMENT_CROSSES_WINDOW


# --- lead-in-trimmed clip: clip-relative start sample ------------------------


def test_lead_in_clip_windows_are_clip_relative():
    # A staged clip whose recitation starts at 0.6 s (neighbour-ayah lead-in before it).
    # The window start_sample is clip-relative (the recitation offset folded in) and the
    # offset is persisted, so the phoneme window joins the waqf soft label on one grid.
    segs = [_seg("a.wav", 0, 0, 3, 0.6, 4.6, "ءبت")]
    status = _status("a.wav", n_words=3, duration_s=5.0, rec_start=0.6, rec_end=4.6)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert reason is None
    offset = round(0.6 * TARGET_SAMPLE_RATE)  # 9600, already a whole 40 ms frame pair
    assert labels[0].start_sample == offset
    assert labels[0].recitation_start_sample == offset


# --- shared window grid with the waqf soft labels ----------------------------


def test_phoneme_and_waqf_windows_share_the_same_grid_for_a_lead_in_clip():
    # ADR-0004 requires the phoneme CTC label and the waqf soft label of a window to pair
    # on ONE grid. For a lead-in-trimmed clip (recitation 0.6-8.6 s inside a 10 s clip),
    # the phoneme label windows and the waqf soft-label windows must share identical
    # (window_index, start_sample, num_samples). The two segments break outside the
    # window overlap band (a waqf pause spans 4.6-5.6 s) so neither crosses a window edge.
    segs = [
        _seg("a.wav", 0, 0, 2, 0.6, 4.6, "ءب"),
        _seg("a.wav", 1, 2, 4, 5.6, 8.6, "جح"),
    ]
    status = _status("a.wav", n_words=4, duration_s=10.0, rec_start=0.6, rec_end=8.6)

    labels, reason = build_clip_windows(segs, status, CONTRACT)
    assert reason is None
    assert len(labels) == 2  # a genuine multi-window clip, both windows kept
    phoneme_grid = [(l.window_index, l.start_sample, l.num_samples) for l in labels]

    # Waqf side: window the SAME recitation span over the whole 10 s clip waveform.
    waveform = np.zeros(int(10.0 * TARGET_SAMPLE_RATE), dtype=np.float32)
    start_sample, num_samples = recitation_window_span(
        status.recitation_start_s, status.recitation_end_s
    )
    waqf_grid = [
        (w.index, w.start_sample, w.num_samples)
        for w, _ in slice_recitation_windows(waveform, start_sample, num_samples, CONTRACT)
    ]

    assert phoneme_grid == waqf_grid
    assert all(l.recitation_start_sample == start_sample for l in labels)


# --- clip-level eligibility --------------------------------------------------


def test_skip_reason_excludes_whole_clip():
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت")]
    status = _status_for("a.wav", segs, n_words=3, skip="repeated_recitation")

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
    status = _status_for("a.wav", segs, n_words=6)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_DROPPED_SEGMENT


def test_trailing_dropped_segment_excludes_clip():
    segs = [_seg("a.wav", 0, 0, 2, 0.0, 3.0, "ءب")]  # n_words says there is a word 2..3
    status = _status_for("a.wav", segs, n_words=3)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_DROPPED_SEGMENT


def test_over_long_recitation_is_flagged_not_truncated():
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت")]
    status = _status_for("a.wav", segs, n_words=3)

    # 4 s recitation is 200 feature frames; a cap of 100 makes it over-long.
    labels, reason = build_clip_windows(segs, status, CONTRACT, cap_feature_frames=100)

    assert labels == []
    assert reason == EXCLUDE_OVER_LONG


def test_window_overlapping_no_segment_is_flagged_empty():
    # Degenerate span: the recitation_end_s claims 6 s but the only segment ends at 1 s, so
    # the second window (4-6 s) overlaps no segment. Guarded rather than mislabelled.
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 1.0, "ءبت")]
    status = _status("a.wav", n_words=3, duration_s=6.0, rec_start=0.0, rec_end=6.0)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_EMPTY_WINDOW


def test_target_longer_than_logit_frames_excludes_clip():
    # A ~0.5 s window (25 feature frames → 13 logit frames) with a 16-phoneme label can't
    # be a valid CTC target (target_len must be < logit_frames).
    segs = [_seg("a.wav", 0, 0, 1, 0.0, 0.5, "ءبتثجحخدذرزسشصضط")]
    status = _status_for("a.wav", segs, n_words=1)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_TARGET_TOO_LONG


# --- corpus build + exclusion report -----------------------------------------


def test_build_windowed_labels_reports_exclusions_by_reason():
    keep = [_seg("keep.wav", 0, 0, 3, 0.0, 4.0, "ءبت", reciter=1)]
    skip = [_seg("skip.wav", 0, 0, 3, 0.0, 4.0, "ءبت", reciter=2)]
    gap = [_seg("gap.wav", 0, 0, 2, 0.0, 3.0, "ءب", reciter=3)]
    segs = keep + skip + gap
    statuses = [
        _status_for("keep.wav", keep, 3, reciter=1),
        _status_for("skip.wav", skip, 3, reciter=2, skip="low_alignment"),
        _status_for("gap.wav", gap, 3, reciter=3),
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
    statuses = [_status_for("a.wav", segs, 3)]

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
    statuses = [_status_for(clip, [s], 3, reciter=rec)
                for (clip, rec), s in zip(clips, segs)]
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
        _status("b.wav", 3, 4.0, rec_start=0.1, rec_end=3.9, reciter=2, skip="low_alignment"),
        _status("a.wav", 5, 6.0, rec_start=0.2, rec_end=5.7, reciter=1),
    ]
    path = tmp_path / "status.jsonl"
    write_clip_status(path, statuses)

    loaded = read_clip_status(path)

    # Written key-sorted by audio_filename → deterministic order.
    assert [s.audio_filename for s in loaded] == ["a.wav", "b.wav"]
    assert loaded[0].recitation_start_s == 0.2 and loaded[0].recitation_end_s == 5.7
    assert loaded[1].skip_reason == "low_alignment"


def test_write_labels_is_sorted_and_tagged(tmp_path: Path):
    labels = _labels_for([("b.wav", 1), ("a.wav", 2)])
    path = tmp_path / "labels.jsonl"

    write_labels(path, labels, "train")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["clip_audio_filename"] for r in rows] == ["a.wav", "b.wav"]
    assert all(r["split"] == "train" for r in rows)
    assert all(isinstance(r["segment_indices"], list) for r in rows)
    assert all("recitation_start_sample" in r for r in rows)


# --- CLI smoke test ----------------------------------------------------------


def _write_segments(path: Path, segments) -> None:
    """Serialize ``Segment`` rows to the JSONL manifest shape ``read_segments`` expects."""
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(asdict(seg), ensure_ascii=False) + "\n")


def test_main_cli_builds_labels_and_report(tmp_path, monkeypatch):
    """End-to-end ``main()`` invocation guards the CLI wiring (parser + ``--segments``).

    A previous regression deleted the ``ArgumentParser``/``--segments`` setup, so ``main()``
    raised ``NameError`` before building anything. Running it here exercises argument parsing
    and the ``read_segments(args.segments)`` path so a broken CLI fails loudly.
    """
    segs = [_seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت")]
    seg_path = tmp_path / "segments.jsonl"
    status_path = tmp_path / "status.jsonl"
    labels_path = tmp_path / "windowed_labels.jsonl"
    report_path = tmp_path / "report.json"
    _write_segments(seg_path, segs)
    write_clip_status(status_path, [_status_for("a.wav", segs, n_words=3)])

    monkeypatch.setattr(sys, "argv", [
        "windowed_labels",
        "--segments", str(seg_path),
        "--clip-status", str(status_path),
        "--out-labels", str(labels_path),
        "--out-report", str(report_path),
    ])
    main()

    by_split = read_labels(labels_path)
    assert sum(len(v) for v in by_split.values()) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["clips_kept"] == 1
    assert report["windows_total"] == 1


def test_main_cli_requires_segments(monkeypatch):
    """``main()`` must fail loudly (nonzero exit) when the required ``--segments`` is absent."""
    monkeypatch.setattr(sys, "argv", [
        "windowed_labels",
        "--clip-status", "x.jsonl",
        "--out-labels", "y.jsonl",
        "--out-report", "z.json",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0

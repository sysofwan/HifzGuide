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
    EXCLUDE_RE_READ,
    EXCLUDE_SEGMENT_CROSSES_WINDOW,
    EXCLUDE_TARGET_TOO_LONG,
    Segment,
    assert_no_reciter_leakage,
    build_clip_windows,
    build_windowed_labels,
    EXCLUDE_HELD_OUT_EVAL_CLIP,
    read_held_out_clips,
    resolve_held_out_clips,
    main,
    read_labels,
    read_segments,
    split_by_reciter,
    write_labels,
)

CONTRACT = WindowContract()  # 5 s window (250 frames), frozen 4 s hop (200 frames)


def _seg(clip, index, w0, w1, s0, s1, ref, reciter=1, surah="78:2", offsets=()):
    return Segment(
        clip_audio_filename=clip,
        surah_ayah=surah,
        reciter_id=reciter,
        segment_index=index,
        word_start=w0,
        word_end=w1,
        start_s=s0,
        end_s=s1,
        label_phonemes=ref,
        label_word_offsets=offsets,
    )


def _status(clip, n_words, duration_s, rec_start=0.0, rec_end=None, reciter=1,
            surah="78:2", skip=None, re_reads=0, word_times=()):
    return ClipStatus(
        audio_filename=clip,
        surah_ayah=surah,
        reciter_id=reciter,
        n_words=n_words,
        duration_s=duration_s,
        recitation_start_s=rec_start,
        recitation_end_s=duration_s if rec_end is None else rec_end,
        skip_reason=skip,
        re_reads=re_reads,
        word_times=word_times,
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


def test_reread_clip_excluded_under_own_reason():
    # A re-read clip's segments overlap in words (segment 1 re-covers word 1) and cannot
    # tile the recitation contiguously, so it is excluded — under EXCLUDE_RE_READ, not the
    # misleading dropped_segment.
    segs = [
        _seg("a.wav", 0, 0, 2, 0.0, 4.0, "ءبت"),
        _seg("a.wav", 1, 1, 3, 4.5, 8.0, "بتث"),
    ]
    status = _status_for("a.wav", segs, n_words=3, re_reads=1)

    labels, reason = build_clip_windows(segs, status, CONTRACT)

    assert labels == []
    assert reason == EXCLUDE_RE_READ


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
        "start_s": 0.0, "end_s": 4.0,
        # The label is the tashkeel-bearing raw reference, sliced by the offsets that
        # index *it* — never the vowel-stripped one (ADR-0003).
        "raw_reference_phonemes": "ءَبِتُ", "raw_word_offsets": [0, 2, 4, 6],
        "reference_phonemes": "ءبت", "word_offsets": [0, 1, 2, 3],
    }
    path = tmp_path / "segments.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    segments = read_segments(path)

    assert segments == [
        _seg("a.wav", 0, 0, 3, 0.0, 4.0, "ءَبِتُ", reciter=5, offsets=(0, 2, 4, 6))
    ]


def test_read_segments_rejects_a_manifest_without_raw_word_offsets(tmp_path: Path):
    """A pre-ADR-0003 manifest must fail loudly, not silently drop every short vowel."""
    row = {
        "clip_audio_filename": "a.wav", "surah_ayah": "78:2", "reciter_id": 5,
        "segment_index": 0, "word_start": 0, "word_end": 3,
        "start_s": 0.0, "end_s": 4.0,
        "raw_reference_phonemes": "ءَبِتُ",
        "reference_phonemes": "ءبت", "word_offsets": [0, 1, 2, 3],
    }
    path = tmp_path / "segments.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(KeyError, match="raw_word_offsets"):
        read_segments(path)


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
    """Serialize ``Segment`` rows to the JSONL manifest shape ``read_segments`` expects.

    The manifest names the label fields after the string they index, so a ``Segment``'s
    ``label_phonemes`` / ``label_word_offsets`` are written as the tashkeel-bearing
    ``raw_reference_phonemes`` / ``raw_word_offsets`` (ADR-0003).
    """
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            row = asdict(seg)
            row["raw_reference_phonemes"] = row.pop("label_phonemes")
            row["raw_word_offsets"] = list(row.pop("label_word_offsets"))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        # the build now refuses to silently train on the #34 eval clips
        "--allow-eval-clips-in-training",
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


# --- word-snapped windowing (long recitations) --------------------------------


def _one_char_per_word(clip, n_words, seconds_per_word, index=0, w0=0):
    """One segment of ``n_words`` evenly-spaced words, one phoneme char each."""
    ref = "".join(chr(0x0621 + i) for i in range(n_words))
    return _seg(
        clip,
        index,
        w0,
        w0 + n_words,
        w0 * seconds_per_word,
        (w0 + n_words) * seconds_per_word,
        ref,
        offsets=tuple(range(n_words + 1)),
    )


def test_long_recitation_is_windowed_at_word_edges_not_excluded():
    """With per-word timing a 12 s recitation trains instead of being discarded."""
    seg = _one_char_per_word("a.wav", n_words=12, seconds_per_word=1.0)
    word_times = tuple(float(i) for i in range(13))
    status = _status_for("a.wav", [seg], n_words=12, word_times=word_times)

    labels, reason = build_clip_windows([seg], status, CONTRACT)

    assert reason is None
    assert len(labels) > 1
    # Every window's audio holds exactly the words its label spells, at word edges.
    for label in labels:
        assert label.phoneme_label == seg.slice_words(label.word_start, label.word_end)
        assert label.start_sample >= round(word_times[label.word_start] * 16000) - 640
        end = label.start_sample + label.num_samples
        assert end <= round(word_times[label.word_end] * 16000) + 640
    # ...and the same clip is excluded outright without the timing.
    _, no_timing = build_clip_windows(
        [seg], _status_for("a.wav", [seg], n_words=12), CONTRACT
    )
    assert no_timing == EXCLUDE_SEGMENT_CROSSES_WINDOW


def test_window_spanning_a_waqf_concatenates_both_segments_word_sliced():
    """An interior waqf inside a window is labelled from both segments' own phonetizations."""
    first = _one_char_per_word("a.wav", n_words=4, seconds_per_word=1.0, index=0, w0=0)
    second = _one_char_per_word("a.wav", n_words=8, seconds_per_word=1.0, index=1, w0=4)
    word_times = tuple(float(i) for i in range(13))
    status = _status_for("a.wav", [first, second], n_words=12, word_times=word_times)

    labels, reason = build_clip_windows([first, second], status, CONTRACT)

    assert reason is None
    spanning = [lab for lab in labels if len(lab.segment_indices) == 2]
    assert spanning, "expected a window straddling the interior waqf"
    for label in spanning:
        assert label.phoneme_label == first.slice_words(
            label.word_start, first.word_end
        ) + second.slice_words(second.word_start, label.word_end)


def test_phoneme_and_soft_label_grids_match_under_word_snapping():
    """Both artifacts enumerate the identical snapped grid — the joint pairing contract."""
    from training.waqf_distill import clip_recitation_windows, recitation_window_span

    seg = _one_char_per_word("a.wav", n_words=12, seconds_per_word=1.0)
    word_times = tuple(float(i) for i in range(13))
    status = _status_for("a.wav", [seg], n_words=12, word_times=word_times)

    labels, _ = build_clip_windows([seg], status, CONTRACT)
    start_sample, num_samples = recitation_window_span(
        status.recitation_start_s, status.recitation_end_s
    )
    windows = clip_recitation_windows(start_sample, num_samples, CONTRACT, word_times)

    assert [(lab.window_index, lab.start_sample, lab.num_samples) for lab in labels] == [
        (w.index, w.start_sample, w.num_samples) for w in windows
    ]


def test_word_longer_than_the_window_overlap_is_left_untrained_not_mislabelled():
    """A word in no window contributes no label — and no window holds its audio either."""
    ref = "".join(chr(0x0621 + i) for i in range(3))
    seg = _seg("a.wav", 0, 0, 3, 0.0, 11.0, ref, offsets=(0, 1, 2, 3))
    # Word 1 spans 2 s -> 8 s: it straddles every 5 s window edge on the 4 s hop grid.
    word_times = (0.0, 2.0, 8.0, 11.0)
    status = _status_for("a.wav", [seg], n_words=3, word_times=word_times)

    labels, reason = build_clip_windows([seg], status, CONTRACT)

    assert reason is None
    assert all(1 not in range(lab.word_start, lab.word_end) for lab in labels)
    for label in labels:
        end = label.start_sample + label.num_samples
        assert end <= 2 * 16000 + 640 or label.start_sample >= 8 * 16000 - 640


# --- ADR-0003: tashkeel must survive into the CTC target ----------------------


#: The Muaalem phoneme head's three short-vowel output classes (ADR-0003).
FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"


def test_short_vowels_survive_manifest_to_encoded_ctc_target(tmp_path: Path):
    """End-to-end guard: fatha/damma/kasra reach the encoded CTC target as ids 32-34.

    The label path once sliced the ``.balanced``-normalized ``reference_phonemes``, which
    strips every short vowel, so classes 32-34 had **no positive target anywhere in the
    corpus**. Because the phoneme head trains in full (``modules_to_save``), that actively
    suppressed them — a fine-tuned checkpoint emitted zero vowels where the base emitted
    the reference's full count. Every gate normalizes both sides before scoring, so none
    of them could see it; this test is the tripwire that can.
    """
    from tadabur.phoneme_vocab import PHONEME_CHAR_TO_ID
    from training.windowed_batch import encode_phoneme_label

    raw = f"\u0621{FATHA}\u0628{KASRA}\u062a{DAMMA}"  # ءَبِتُ — one vowel per word
    seg = _seg("a.wav", 0, 0, 3, 0.0, 4.0, raw, offsets=(0, 2, 4, 6))
    status = _status_for("a.wav", [seg], n_words=3, word_times=(0.0, 1.0, 2.0, 4.0))

    path = tmp_path / "segments.jsonl"
    _write_segments(path, [seg])
    labels, reason = build_clip_windows(read_segments(path), status, CONTRACT)

    assert reason is None and labels
    label = "".join(lab.phoneme_label for lab in labels)
    assert {FATHA, DAMMA, KASRA} <= set(label)
    encoded = set(encode_phoneme_label(label))
    assert {PHONEME_CHAR_TO_ID[v] for v in (FATHA, DAMMA, KASRA)} <= encoded
    assert {32, 33, 34} <= encoded


def test_the_label_is_the_raw_reference_not_the_normalized_one(tmp_path: Path):
    """The two manifest references differ; the label must come from the tashkeel-bearing one."""
    raw = f"\u0621{FATHA}\u0628{KASRA}"
    row = {
        "clip_audio_filename": "a.wav", "surah_ayah": "78:2", "reciter_id": 1,
        "segment_index": 0, "word_start": 0, "word_end": 2,
        "start_s": 0.0, "end_s": 4.0,
        "raw_reference_phonemes": raw, "raw_word_offsets": [0, 2, 4],
        "reference_phonemes": "\u0621\u0628", "word_offsets": [0, 1, 2],
    }
    path = tmp_path / "segments.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    (segment,) = read_segments(path)

    assert segment.label_phonemes == raw
    assert segment.slice_words(0, 2) == raw
    assert segment.slice_words(0, 1) == f"\u0621{FATHA}"


def test_held_out_eval_clips_never_become_training_examples(tmp_path: Path):
    """A clip named in the waqf freeze is excluded even though it is otherwise eligible.

    The #34 event eval scores those clips; if they are also training examples the reported
    waqf F1 measures memorization of the exact eval audio.
    """
    keep = [_seg("keep.wav", 0, 0, 3, 0.0, 4.0, "ءبت", reciter=1)]
    held = [_seg("held.wav", 0, 0, 3, 0.0, 4.0, "ءبت", reciter=2)]
    statuses = [_status_for("keep.wav", keep, 3, reciter=1),
                _status_for("held.wav", held, 3, reciter=2)]

    # Both clips are eligible when nothing is held out.
    assert build_windowed_labels(keep + held, statuses, CONTRACT).clips_kept == 2

    built = build_windowed_labels(
        keep + held, statuses, CONTRACT, held_out_clips=frozenset({"held.wav"})
    )

    assert built.clips_kept == 1
    assert {c for c, _ in built.exclusions} == {"held.wav"}
    assert dict(built.reasons) == {EXCLUDE_HELD_OUT_EVAL_CLIP: 1}
    assert all(lab.clip_audio_filename != "held.wav" for lab in built.labels)


def test_read_held_out_clips_unions_calibration_and_test(tmp_path: Path):
    path = tmp_path / "waqf_partition.json"
    path.write_text(json.dumps({
        "calibration_clips": ["a.wav", "b.wav"], "test_clips": ["c.wav"],
        "must_exclude_reciters": [1, 2],
    }), encoding="utf-8")

    assert read_held_out_clips(path) == frozenset({"a.wav", "b.wav", "c.wav"})


def test_read_held_out_clips_rejects_a_non_partition_json(tmp_path: Path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"test_clips": ["c.wav"]}), encoding="utf-8")

    with pytest.raises(KeyError, match="calibration_clips"):
        read_held_out_clips(path)


def test_resolve_held_out_clips_refuses_a_silently_leaky_build():
    """Omitting the partition must fail, not quietly train on the #34 eval clips."""
    with pytest.raises(SystemExit, match="refusing to build labels"):
        resolve_held_out_clips(None, allow_leak=False)


def test_resolve_held_out_clips_allows_an_explicitly_leaky_build():
    assert resolve_held_out_clips(None, allow_leak=True) == frozenset()


def test_resolve_held_out_clips_reads_the_partition_when_given(tmp_path: Path):
    path = tmp_path / "partition.json"
    path.write_text(
        json.dumps({"calibration_clips": ["a.wav"], "test_clips": ["b.wav"]}),
        encoding="utf-8",
    )

    assert resolve_held_out_clips(path, allow_leak=False) == frozenset({"a.wav", "b.wav"})

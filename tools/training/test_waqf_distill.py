"""Tests for the torch-free waqf distillation pooling, windowing, and soft-label store.

The pooling and per-window frame-alignment (:mod:`training.waqf_distill`) are the pinned
rule the whole ADR-0004 waqf head depends on: a 1–2 frame shift between the 20 ms VAD
teacher and the 40 ms Muaalem student moves a boundary snap across a word edge. These
golden fixtures prove the correspondence exactly — per training window — without a GPU.
The VAD forward pass (torch) lives in ``tadabur.vad`` and is not exercised here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from training.waqf_distill import (
    DEPLOYED_WINDOW_FEATURE_FRAMES,
    SoftLabelStore,
    Window,
    WindowContract,
    clip_window_soft_labels,
    enumerate_windows,
    muaalem_lattice_length,
    pool_silence_2to1,
    window_silence_soft_labels,
)


# --- muaalem_lattice_length: the exact 20 ms → 40 ms conv relation -----------


@pytest.mark.parametrize(
    "feature_frames,expected",
    [
        (250, 125),   # the fixed 5 s export window (ADR-0004)
        (249, 125),   # 5 s as the VAD actually frames it → still 125
        (251, 126),   # odd tail adds a student frame (ceil, not floor)
        (2, 1),
        (1, 1),
    ],
)
def test_lattice_length_matches_adapter_conv(feature_frames, expected):
    assert muaalem_lattice_length(feature_frames) == expected


def test_lattice_length_is_ceil_of_half():
    for t in range(1, 400):
        assert muaalem_lattice_length(t) == -(-t // 2)  # ceil(t/2)


# --- pool_silence_2to1: golden frame-correspondence fixtures -----------------


def test_student_frame_owns_its_two_teacher_frames():
    # A silence run over teacher frames [4, 8) must land on student frames [2, 4):
    # student i owns teacher {2i, 2i+1}, so frames 4-7 → students 2 and 3, and no
    # neighbouring student frame is contaminated.
    teacher = np.zeros(12, dtype=np.float32)
    teacher[4:8] = 1.0
    student = pool_silence_2to1(teacher, num_student_frames=6)
    assert student.tolist() == [0, 0, 1, 1, 0, 0]


def test_one_teacher_frame_shift_moves_the_student_boundary():
    # The drift warning made concrete: shifting the silence onset by a single 20 ms
    # teacher frame (4→5) must NOT bleed into student frame 1 — student 2 owns teacher
    # {4,5}, and min-pool keeps it silent only when BOTH are silent.
    aligned = np.zeros(12, dtype=np.float32)
    aligned[4:8] = 1.0
    shifted = np.zeros(12, dtype=np.float32)
    shifted[5:9] = 1.0  # onset one teacher frame later

    assert pool_silence_2to1(aligned, 6).tolist() == [0, 0, 1, 1, 0, 0]
    # teacher {4,5}=(0,1)→0, {6,7}=(1,1)→1, {8,9}=(1,0)→0: the half-covered edge
    # student frames go to speech, so the boundary snaps one student frame inward.
    assert pool_silence_2to1(shifted, 6).tolist() == [0, 0, 0, 1, 0, 0]


def test_silent_iff_both_teacher_frames_silent():
    # min-pool: a student frame silence posterior is the min of its two teacher frames,
    # so a half-silent 40 ms window is scored as speech (max-pool speech) — the pinned
    # "silent iff both" rule, not an average.
    teacher = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.95], dtype=np.float32)
    student = pool_silence_2to1(teacher, num_student_frames=3)
    np.testing.assert_allclose(student, [0.1, 0.7, 0.2])


# --- reconcile: drift is absorbed at the tail, not an interior boundary -------


def test_odd_teacher_tail_is_edge_held():
    # 5 teacher frames, 3 student frames need 6: the missing final frame is edge-held
    # (repeat frame 4), so student 2 = min(teacher[4], teacher[4]).
    teacher = np.array([0.0, 0.0, 1.0, 1.0, 0.3], dtype=np.float32)
    student = pool_silence_2to1(teacher, num_student_frames=3)
    np.testing.assert_allclose(student, [0.0, 1.0, 0.3])


def test_extra_teacher_tail_is_truncated():
    # 8 teacher frames but only 3 student frames requested: the trailing 2 frames are
    # dropped (left-anchored), never folded back into an interior frame.
    teacher = np.arange(8, dtype=np.float32) / 10.0
    student = pool_silence_2to1(teacher, num_student_frames=3)
    np.testing.assert_allclose(student, [0.0, 0.2, 0.4])


def test_empty_teacher_with_frames_needed_raises():
    with pytest.raises(ValueError, match="no silence signal"):
        pool_silence_2to1(np.array([], dtype=np.float32), num_student_frames=1)


def test_zero_student_frames_is_empty():
    assert pool_silence_2to1(np.array([0.5, 0.5], dtype=np.float32), 0).tolist() == []


# --- WindowContract: the deployed 5 s window + provisional even-frame spacing --


def test_default_contract_is_the_deployed_non_overlapping_5s_window():
    contract = WindowContract()
    assert contract.feature_frames == DEPLOYED_WINDOW_FEATURE_FRAMES == 250
    assert contract.hop_feature_frames == 250  # non-overlapping (provisional, #24)
    assert contract.student_frames == 125


@pytest.mark.parametrize("bad", [0, -2, 251, 3])
def test_contract_rejects_non_positive_or_odd_frames(bad):
    # Odd window/hop would split a teacher pair across two windows and reintroduce the
    # ±1-frame drift the alignment pins — rejected up front.
    with pytest.raises(ValueError):
        WindowContract(feature_frames=bad)
    with pytest.raises(ValueError):
        WindowContract(hop_feature_frames=bad)


# --- enumerate_windows: deterministic tiling on the teacher grid --------------


def test_non_overlapping_tiling_covers_the_clip():
    # 600 teacher frames, 250-frame non-overlapping windows: [0,250), [250,500),
    # [500,600) — the tail window carries the 100 remaining frames only.
    windows = enumerate_windows(600, WindowContract())
    assert [(w.index, w.start_feature_frame, w.num_feature_frames) for w in windows] == [
        (0, 0, 250),
        (1, 250, 250),
        (2, 500, 100),
    ]
    # Each window's student count is its own slice's exact Muaalem 40 ms length.
    assert [w.num_student_frames for w in windows] == [125, 125, 50]
    # Even starts → student start is exactly start // 2 (clip-lattice aligned).
    assert [w.start_student_frame for w in windows] == [0, 125, 250]


def test_clip_shorter_than_one_window_is_a_single_window():
    windows = enumerate_windows(249, WindowContract())
    assert len(windows) == 1
    assert windows[0].num_feature_frames == 249
    assert windows[0].num_student_frames == muaalem_lattice_length(249) == 125


def test_no_teacher_frames_yields_no_windows():
    assert enumerate_windows(0, WindowContract()) == []


def test_overlapping_hop_shares_student_start_grid():
    # A 50/25-frame overlapping contract: starts step by the hop, every start even so
    # every window still lands on the clip's 40 ms lattice.
    contract = WindowContract(feature_frames=50, hop_feature_frames=24)
    windows = enumerate_windows(100, contract)
    assert [w.start_feature_frame for w in windows] == [0, 24, 48, 72, 96]
    assert [w.start_student_frame for w in windows] == [0, 12, 24, 36, 48]


# --- window_silence_soft_labels: per-window teacher→student correspondence ----


def test_window_target_maps_teacher_run_to_the_right_student_frames():
    # A clip-wide teacher with a silence run at teacher frames [254, 258). Under the
    # 250/250 tiling that run is inside window 1 (teacher [250,500) → student [125,250)),
    # at window-local teacher frames [4,8) → window-local student frames [2,4). It must
    # NOT appear in window 0, proving windowing preserves the pinned 2:1 mapping.
    teacher = np.zeros(600, dtype=np.float32)
    teacher[254:258] = 1.0
    windows = enumerate_windows(len(teacher), WindowContract())

    w0 = window_silence_soft_labels(teacher, windows[0])
    w1 = window_silence_soft_labels(teacher, windows[1])
    assert w0.max() == 0.0  # nothing bled into the previous window
    assert w1[2] == 1.0 and w1[3] == 1.0
    assert w1[:2].max() == 0.0 and w1[4:].max() == 0.0


def test_window_boundary_run_is_owned_by_exactly_one_window():
    # A silence run straddling a window edge at teacher frame 250 (the window-1 start):
    # frames [248,252). Non-overlapping windows split it — frames 248-249 belong to
    # window 0's last student frame, 250-251 to window 1's first — and neither window
    # sees the other half. This is the drift-sensitive edge the fixtures must pin.
    teacher = np.zeros(600, dtype=np.float32)
    teacher[248:252] = 1.0
    windows = enumerate_windows(len(teacher), WindowContract())

    w0 = window_silence_soft_labels(teacher, windows[0])  # student [0,125)
    w1 = window_silence_soft_labels(teacher, windows[1])  # student [0,125)
    assert w0[124] == 1.0 and w0[:124].max() == 0.0  # last student frame of window 0
    assert w1[0] == 1.0 and w1[1:].max() == 0.0       # first student frame of window 1


def test_tail_window_target_length_tracks_its_own_slice():
    teacher = np.random.default_rng(3).random(600).astype(np.float32)
    windows = enumerate_windows(len(teacher), WindowContract())
    labels = window_silence_soft_labels(teacher, windows[2])  # 100-frame tail window
    assert len(labels) == windows[2].num_student_frames == 50


# --- clip_window_soft_labels: whole-clip per-window artifact ------------------


def test_clip_window_soft_labels_pairs_each_window_with_its_target():
    teacher = np.random.default_rng(0).random(600).astype(np.float32)
    pairs = clip_window_soft_labels(teacher, WindowContract())
    assert [w.index for w, _ in pairs] == [0, 1, 2]
    for window, labels in pairs:
        assert len(labels) == window.num_student_frames


def test_clip_window_soft_labels_are_deterministic():
    teacher = np.random.default_rng(1).random(320).astype(np.float32)
    a = clip_window_soft_labels(teacher, WindowContract())
    b = clip_window_soft_labels(teacher, WindowContract())
    for (_, la), (_, lb) in zip(a, b):
        np.testing.assert_array_equal(la, lb)


# --- SoftLabelStore: per-window, manifest-keyed, idempotent, resumable --------


def _windows_for(labels_by_index):
    return [
        (
            Window(
                index=i,
                start_feature_frame=i * 250,
                start_student_frame=i * 125,
                num_feature_frames=250,
                num_student_frames=len(lab),
            ),
            lab,
        )
        for i, lab in enumerate(labels_by_index)
    ]


def test_store_persists_each_window_and_indexes_the_clip_once(tmp_path):
    w0 = np.array([0.1, 0.9], dtype=np.float32)
    w1 = np.array([0.4, 0.2, 0.7], dtype=np.float32)
    with SoftLabelStore.open(tmp_path) as store:
        store.write_clip("clip_a.wav", _windows_for([w0, w1]), num_teacher_frames=500)

    # One index line for the clip, listing both windows keyed by window_index.
    lines = (tmp_path / SoftLabelStore.INDEX_NAME).read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["audio_filename"] == "clip_a.wav"
    assert [w["window_index"] for w in record["windows"]] == [0, 1]
    assert [w["start_student_frame"] for w in record["windows"]] == [0, 125]

    arrays_dir = tmp_path / SoftLabelStore.ARRAYS_SUBDIR
    np.testing.assert_array_equal(np.load(arrays_dir / "clip_a.wav#w0.npy"), w0)
    np.testing.assert_array_equal(np.load(arrays_dir / "clip_a.wav#w1.npy"), w1)


def test_store_write_clip_is_idempotent(tmp_path):
    labels = _windows_for([np.array([0.2, 0.8], dtype=np.float32)])
    with SoftLabelStore.open(tmp_path) as store:
        store.write_clip("clip_a.wav", labels, num_teacher_frames=250)
        store.write_clip("clip_a.wav", labels, num_teacher_frames=250)  # replay → no-op

    lines = (tmp_path / SoftLabelStore.INDEX_NAME).read_text().strip().splitlines()
    assert len(lines) == 1


def test_store_resumes_skipping_written_clips(tmp_path):
    with SoftLabelStore.open(tmp_path) as store:
        store.write_clip(
            "clip_a.wav", _windows_for([np.array([0.5], dtype=np.float32)]), num_teacher_frames=2
        )

    resumed = SoftLabelStore.open(tmp_path)
    assert resumed.has("clip_a.wav")
    assert not resumed.has("clip_b.wav")
    resumed.close()


# --- torch-free offline stage ------------------------------------------------


def test_importing_waqf_distill_does_not_import_torch():
    # The pooling/windowing/alignment must run in the plain CPU env without pulling in
    # the GPU inference path; a fresh interpreter proves the import stays torch-free.
    code = (
        "import sys; import training.waqf_distill; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr

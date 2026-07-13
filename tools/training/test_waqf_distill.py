"""Tests for the torch-free waqf distillation pooling, alignment, and soft-label store.

The pooling and frame-alignment (:mod:`training.waqf_distill`) are the pinned rule the
whole ADR-0004 waqf head depends on: a 1–2 frame shift between the 20 ms VAD teacher and
the 40 ms Muaalem student moves a boundary snap across a word edge. These golden fixtures
prove the correspondence exactly, without a GPU. The VAD forward pass (torch) lives in
``tadabur.vad`` and is not exercised here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from training.waqf_distill import (
    SoftLabelStore,
    clip_silence_soft_labels,
    muaalem_lattice_length,
    pool_silence_2to1,
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


# --- clip_silence_soft_labels: whole-clip pooling via the conv length --------


def test_clip_soft_labels_length_tracks_the_lattice():
    # A 249-frame teacher (5 s) pools to a 125-frame student clip, matching Muaalem.
    teacher = np.random.default_rng(0).random(249).astype(np.float32)
    labels = clip_silence_soft_labels(teacher)
    assert len(labels) == muaalem_lattice_length(249) == 125


def test_clip_soft_labels_are_deterministic():
    teacher = np.random.default_rng(1).random(60).astype(np.float32)
    np.testing.assert_array_equal(
        clip_silence_soft_labels(teacher), clip_silence_soft_labels(teacher)
    )


# --- SoftLabelStore: deterministic, idempotent, manifest-keyed ---------------


def test_store_persists_and_reloads_keyed_by_audio_filename(tmp_path):
    labels = np.array([0.1, 0.9, 0.4], dtype=np.float32)
    with SoftLabelStore.open(tmp_path) as store:
        store.write("clip_a.wav", labels, num_teacher_frames=6)

    reopened = SoftLabelStore.open(tmp_path)
    assert reopened.has("clip_a.wav")
    reopened.close()

    saved = np.load(tmp_path / SoftLabelStore.ARRAYS_SUBDIR / "clip_a.wav.npy")
    np.testing.assert_array_equal(saved, labels)


def test_store_write_is_idempotent(tmp_path):
    labels = np.array([0.2, 0.8], dtype=np.float32)
    with SoftLabelStore.open(tmp_path) as store:
        store.write("clip_a.wav", labels, num_teacher_frames=4)
        store.write("clip_a.wav", labels, num_teacher_frames=4)  # replayed → no-op

    index_lines = (tmp_path / SoftLabelStore.INDEX_NAME).read_text().strip().splitlines()
    assert len(index_lines) == 1


def test_store_resumes_skipping_written_clips(tmp_path):
    with SoftLabelStore.open(tmp_path) as store:
        store.write("clip_a.wav", np.array([0.5], dtype=np.float32), num_teacher_frames=2)

    resumed = SoftLabelStore.open(tmp_path)
    assert resumed.has("clip_a.wav")
    assert not resumed.has("clip_b.wav")
    resumed.close()


# --- torch-free offline stage ------------------------------------------------


def test_importing_waqf_distill_does_not_import_torch():
    # The pooling/alignment must run in the plain CPU env without pulling in the GPU
    # inference path; a fresh interpreter proves the module import stays torch-free.
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

"""Tests for the torch-free waqf distillation pooling, windowing, and soft-label store.

The pooling and per-window frame-alignment (:mod:`training.waqf_distill`) are the pinned
rule the whole ADR-0004 waqf head depends on: a 1–2 frame shift between the 20 ms VAD
teacher and the 40 ms Muaalem student moves a boundary snap across a word edge. These
golden fixtures prove the correspondence exactly — per training window — without a GPU.
The VAD forward pass (torch) lives in ``tadabur.vad`` and is not exercised here; the
teacher posteriors below stand in for what the VAD emits for a single window waveform.
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
    SAMPLES_PER_STUDENT_FRAME,
    SAMPLES_PER_TEACHER_FRAME,
    FEATURE_FRAME_SAMPLE_OFFSET,
    WINDOW_ORIGIN_RECITATION,
    WINDOW_ORIGIN_WHOLE_CLIP,
    SoftLabelStore,
    Window,
    WindowContract,
    _read_index_spans,
    enumerate_recitation_windows,
    enumerate_windows,
    generation_contract,
    muaalem_lattice_length,
    pool_silence_2to1,
    pool_window_posteriors,
    window_student_frames,
    feature_frames_for_samples,
    recitation_window_span,
    slice_recitation_windows,
    slice_windows,
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


# --- pool_window_posteriors: teacher→student correspondence per window --------


def test_window_posteriors_map_teacher_run_to_the_right_student_frames():
    # The acceptance-criterion golden fixture, now per window: the VAD emits 20 ms
    # posteriors for ONE window waveform, with a silence run at window teacher frames
    # [4, 8). Pooling must land it on window student frames [2, 4) and nowhere else —
    # student j owns window teacher {2j, 2j+1}.
    window_teacher = np.zeros(250, dtype=np.float32)  # a full 5 s window's VAD output
    window_teacher[4:8] = 1.0
    student = pool_window_posteriors(window_teacher, samples_for_feature_frames(250))
    assert len(student) == muaalem_lattice_length(250) == 125
    assert student[2] == 1.0 and student[3] == 1.0
    assert student[:2].max() == 0.0 and student[4:].max() == 0.0


def test_window_posteriors_one_frame_shift_moves_the_boundary():
    # A single-teacher-frame drift in a window's own posteriors moves the silence
    # boundary by exactly one student frame — the word-edge risk the head must survive.
    aligned = np.zeros(20, dtype=np.float32)
    aligned[4:8] = 1.0
    shifted = np.zeros(20, dtype=np.float32)
    shifted[5:9] = 1.0
    span = samples_for_feature_frames(20)
    assert pool_window_posteriors(aligned, span)[:5].tolist() == [0, 0, 1, 1, 0]
    assert pool_window_posteriors(shifted, span)[:5].tolist() == [0, 0, 0, 1, 0]


def test_window_posteriors_odd_length_ceils_and_edge_holds_tail():
    # A 249-frame window span gets 125 student frames (ceil), the missing 250th teacher
    # frame edge-held — drift absorbed at the window tail.
    window_teacher = np.zeros(249, dtype=np.float32)
    window_teacher[-1] = 0.7
    student = pool_window_posteriors(window_teacher, samples_for_feature_frames(249))
    assert len(student) == 125
    assert student[-1] == pytest.approx(0.7)  # min(teacher[248], edge-held teacher[248])


@pytest.mark.parametrize("vad_frames", [205, 206, 207, 208, 209])
def test_window_student_length_follows_the_span_not_the_vad_frame_count(vad_frames):
    # The VAD's emitted frame count drifts against ``num_samples // 320`` on a window
    # shorter than the full 5 s, but the student lattice is fixed by the window's audio
    # span — the identical expression ``training.windowed_labels`` uses for
    # ``logit_frames``. Deriving it from ``len(vad_output)`` instead gave a
    # 66287-sample window a 103-frame teacher against its 104-frame phoneme target, and
    # the joint loader rejected the pair as "different window grids".
    num_samples = 66287
    # The real extractor emits 206 feature frames here — the naive ``num_samples // 320``
    # says 207, and the extra frame is exactly what broke the joint loss.
    assert feature_frames_for_samples(num_samples) == 206
    assert num_samples // SAMPLES_PER_TEACHER_FRAME == 207
    expected = muaalem_lattice_length(206)
    assert expected == 103
    assert window_student_frames(num_samples) == expected
    student = pool_window_posteriors(np.zeros(vad_frames, dtype=np.float32), num_samples)
    assert len(student) == expected


# --- WindowContract: the deployed 5 s window + frozen center-trusted overlap --


def test_default_contract_is_the_frozen_center_trusted_overlap_5s_window():
    contract = WindowContract()
    assert contract.feature_frames == DEPLOYED_WINDOW_FEATURE_FRAMES == 250
    assert contract.hop_feature_frames == 200  # frozen 1 s overlap (4 s hop), #24 A2
    assert contract.student_frames == 125
    # 250 teacher frames × 320 samples = 80 000 samples ≈ 5 s at 16 kHz.
    assert contract.window_samples == 250 * SAMPLES_PER_TEACHER_FRAME == 80000
    # 200 teacher frames × 320 samples = 64 000 samples ≈ 4 s hop (1 s overlap).
    assert contract.hop_samples == 200 * SAMPLES_PER_TEACHER_FRAME == 64000


@pytest.mark.parametrize("bad", [0, -2, 251, 3])
def test_contract_rejects_non_positive_or_odd_frames(bad):
    # Odd window/hop would split a teacher pair across two windows and reintroduce the
    # ±1-frame drift the alignment pins — rejected up front.
    with pytest.raises(ValueError):
        WindowContract(feature_frames=bad)
    with pytest.raises(ValueError):
        WindowContract(hop_feature_frames=bad)


# --- enumerate_windows: deterministic sample-domain tiling --------------------


def test_non_overlapping_tiling_covers_the_clip():
    # 600 teacher frames of audio (600×320 samples), 250-frame non-overlapping windows
    # (explicit hop == window): sample spans [0,80k), [80k,160k), [160k,192k) — the tail
    # window carries the remaining 100 teacher frames (32 000 samples) only.
    num_samples = 600 * SAMPLES_PER_TEACHER_FRAME
    windows = enumerate_windows(num_samples, WindowContract(hop_feature_frames=250))
    assert [(w.index, w.start_sample, w.num_samples) for w in windows] == [
        (0, 0, 80000),
        (1, 80000, 80000),
        (2, 160000, 32000),
    ]
    # Even starts → student start is exactly start_feature_frame // 2 (clip-lattice aligned).
    assert [w.start_feature_frame for w in windows] == [0, 250, 500]
    assert [w.start_student_frame for w in windows] == [0, 125, 250]


def test_frozen_center_trusted_overlap_windows_step_by_the_4s_hop():
    # 700 teacher frames of audio, frozen default (250-frame window, 200-frame hop = 1 s
    # overlap): windows start every 64 000 samples and overlap the previous by 16 000.
    num_samples = 700 * SAMPLES_PER_TEACHER_FRAME
    windows = enumerate_windows(num_samples, WindowContract())
    assert [(w.index, w.start_sample, w.num_samples) for w in windows] == [
        (0, 0, 80000),
        (1, 64000, 80000),
        (2, 128000, 80000),
        (3, 192000, 32000),
    ]
    assert [w.start_feature_frame for w in windows] == [0, 200, 400, 600]
    assert [w.start_student_frame for w in windows] == [0, 100, 200, 300]


def test_clip_no_longer_than_one_hop_is_a_single_window():
    # Under the frozen 200-frame hop, a clip no longer than one hop yields a single
    # window (the next start would fall at/after the clip end).
    windows = enumerate_windows(200 * SAMPLES_PER_TEACHER_FRAME, WindowContract())
    assert len(windows) == 1
    assert windows[0].num_samples == 200 * SAMPLES_PER_TEACHER_FRAME


def test_no_samples_yields_no_windows():
    assert enumerate_windows(0, WindowContract()) == []


def test_overlapping_hop_shares_student_start_grid():
    # A 50/24-frame overlapping contract: starts step by the hop_samples, every start on
    # an even teacher frame so every window still lands on the clip's 40 ms lattice.
    contract = WindowContract(feature_frames=50, hop_feature_frames=24)
    windows = enumerate_windows(100 * SAMPLES_PER_TEACHER_FRAME, contract)
    assert [w.start_feature_frame for w in windows] == [0, 24, 48, 72, 96]
    assert [w.start_student_frame for w in windows] == [0, 12, 24, 36, 48]


# --- slice_windows: the waveform is cut on window boundaries ------------------


def test_slice_windows_cuts_the_waveform_on_window_boundaries():
    # A clip 2.5 windows long: the VAD must see each window's OWN samples, so slicing
    # (not whole-clip posterior slicing) is what the generator feeds the model. Use an
    # explicit non-overlapping tiling so the slices reconstruct the clip exactly.
    contract = WindowContract(hop_feature_frames=250)
    waveform = np.arange(int(2.5 * contract.window_samples), dtype=np.float32)
    windows = slice_windows(waveform, contract)
    assert [w.index for w, _ in windows] == [0, 1, 2]
    # Each slice is exactly the window's sample span from the clip.
    for window, wave_slice in windows:
        expected = waveform[window.start_sample : window.start_sample + window.num_samples]
        np.testing.assert_array_equal(wave_slice, expected)
    # Non-overlapping tiling → concatenating the slices reconstructs the clip.
    np.testing.assert_array_equal(np.concatenate([s for _, s in windows]), waveform)


def test_slice_windows_is_deterministic():
    contract = WindowContract(feature_frames=50, hop_feature_frames=50)
    waveform = np.random.default_rng(1).random(3 * contract.window_samples).astype(np.float32)
    a = slice_windows(waveform, contract)
    b = slice_windows(waveform, contract)
    for (_, sa), (_, sb) in zip(a, b):
        np.testing.assert_array_equal(sa, sb)


# --- recitation-span windowing: the shared clip-relative grid ----------------


def test_recitation_window_span_floors_the_start_to_a_student_frame_pair():
    # The recitation onset is floored to a whole 40 ms student-frame pair so window starts
    # stay on the 40 ms lattice (pulling in <=40 ms of lead-in, within the edge pad).
    start_sample, num_samples = recitation_window_span(0.641, 4.641)
    assert start_sample % SAMPLES_PER_STUDENT_FRAME == 0
    assert start_sample == (round(0.641 * 16000) // SAMPLES_PER_STUDENT_FRAME) * SAMPLES_PER_STUDENT_FRAME
    assert num_samples == round(4.641 * 16000) - start_sample


def test_recitation_windows_are_clip_relative_and_match_the_zero_based_grid():
    # A lead-in-trimmed recitation windows on the SAME 0-based grid as the whole clip, only
    # shifted by the clip-relative onset, so the phoneme and waqf artifacts pair per window.
    contract = WindowContract()
    start_sample, num_samples = recitation_window_span(0.6, 8.6)  # 9600, 128000
    windows = enumerate_recitation_windows(start_sample, num_samples, contract)
    base = enumerate_windows(num_samples, contract)
    assert [w.index for w in windows] == [w.index for w in base]
    assert [w.num_samples for w in windows] == [w.num_samples for w in base]
    assert [w.start_sample for w in windows] == [start_sample + w.start_sample for w in base]


def test_recitation_windows_drop_the_redundant_overlap_tail():
    # A recitation just past the 4 s hop yields a trailing window that is pure overlap the
    # previous window already covers (its audio ends no later). The inference stitch discards
    # it, so the shared grid must too — otherwise a segment crossing that tail's edge would
    # wrongly exclude the clip and the two artifacts could disagree on window count.
    contract = WindowContract()
    start_sample, num_samples = recitation_window_span(0.0, 4.4)  # 0, 70400 (< 5 s, > 4 s hop)
    base = enumerate_windows(num_samples, contract)
    windows = enumerate_recitation_windows(start_sample, num_samples, contract)
    assert len(base) == 2 and base[1].start_sample + base[1].num_samples == num_samples
    assert [w.index for w in windows] == [0]  # the redundant second window is dropped


def test_slice_recitation_windows_cuts_the_clip_on_the_shared_grid():
    # The waqf side slices the WHOLE clip waveform on the shared clip-relative grid, so each
    # slice is exactly the window's clip span — the same grid the phoneme labels enumerate.
    contract = WindowContract()
    clip = np.arange(int(10.0 * 16000), dtype=np.float32)
    start_sample, num_samples = recitation_window_span(0.6, 8.6)
    sliced = slice_recitation_windows(clip, start_sample, num_samples, contract)
    windows = enumerate_recitation_windows(start_sample, num_samples, contract)
    assert [w.index for w, _ in sliced] == [w.index for w in windows]
    for window, wave_slice in sliced:
        expected = clip[window.start_sample : window.start_sample + window.num_samples]
        np.testing.assert_array_equal(wave_slice, expected)


def test_enumerate_recitation_windows_rejects_an_unaligned_onset():
    with pytest.raises(ValueError):
        enumerate_recitation_windows(9601, 64000, WindowContract())


# --- SoftLabelStore: per-window, manifest-keyed, idempotent, resumable --------


def _windows_for(labels_by_index, contract=None):
    contract = contract or WindowContract()
    return [
        (
            Window(
                index=i,
                start_sample=i * contract.hop_samples,
                num_samples=contract.window_samples,
            ),
            lab,
        )
        for i, lab in enumerate(labels_by_index)
    ]


def test_store_persists_each_window_and_indexes_the_clip_once(tmp_path):
    w0 = np.array([0.1, 0.9], dtype=np.float32)
    w1 = np.array([0.4, 0.2, 0.7], dtype=np.float32)
    with SoftLabelStore.open(tmp_path, WindowContract()) as store:
        store.write_clip("clip_a.wav", _windows_for([w0, w1]), num_samples=500 * SAMPLES_PER_TEACHER_FRAME)

    # One index line for the clip, listing both windows keyed by window_index.
    lines = (tmp_path / SoftLabelStore.INDEX_NAME).read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["audio_filename"] == "clip_a.wav"
    assert [w["window_index"] for w in record["windows"]] == [0, 1]
    assert [w["start_student_frame"] for w in record["windows"]] == [0, 100]
    assert [w["start_sample"] for w in record["windows"]] == [0, 64000]

    arrays_dir = tmp_path / SoftLabelStore.ARRAYS_SUBDIR
    np.testing.assert_array_equal(np.load(arrays_dir / "clip_a.wav#w0.npy"), w0)
    np.testing.assert_array_equal(np.load(arrays_dir / "clip_a.wav#w1.npy"), w1)


def test_store_write_clip_is_idempotent(tmp_path):
    labels = _windows_for([np.array([0.2, 0.8], dtype=np.float32)])
    with SoftLabelStore.open(tmp_path, WindowContract()) as store:
        store.write_clip("clip_a.wav", labels, num_samples=80000)
        store.write_clip("clip_a.wav", labels, num_samples=80000)  # replay → no-op

    lines = (tmp_path / SoftLabelStore.INDEX_NAME).read_text().strip().splitlines()
    assert len(lines) == 1


def test_store_resumes_skipping_written_clips(tmp_path):
    with SoftLabelStore.open(tmp_path, WindowContract()) as store:
        store.write_clip(
            "clip_a.wav", _windows_for([np.array([0.5], dtype=np.float32)]), num_samples=80000
        )

    resumed = SoftLabelStore.open(tmp_path, WindowContract())
    assert resumed.has("clip_a.wav")
    assert not resumed.has("clip_b.wav")
    resumed.close()


# --- SoftLabelStore contract metadata: no silent cross-contract corruption ----


def test_store_records_the_generation_contract(tmp_path):
    contract = WindowContract(feature_frames=50, hop_feature_frames=24)
    with SoftLabelStore.open(tmp_path, contract):
        pass
    stored = json.loads((tmp_path / SoftLabelStore.CONTRACT_NAME).read_text())
    assert stored == generation_contract(contract, WINDOW_ORIGIN_WHOLE_CLIP)
    assert stored["window_feature_frames"] == 50
    assert stored["hop_feature_frames"] == 24
    assert stored["pooling_rule"] == "min-silence-2to1-left-anchored-span"
    assert stored["window_origin"] == WINDOW_ORIGIN_WHOLE_CLIP


def test_store_resume_across_window_origins_fails_fast(tmp_path):
    # A whole-clip store re-opened for the ADR-0004 recitation grid must NOT silently skip
    # its whole-clip clips and leave their (0-origin) windows to be paired with
    # recitation-relative phoneme labels — the joint-label misalignment guard.
    with SoftLabelStore.open(tmp_path, WindowContract(), WINDOW_ORIGIN_WHOLE_CLIP):
        pass
    with pytest.raises(ValueError, match="different contract"):
        SoftLabelStore.open(tmp_path, WindowContract(), WINDOW_ORIGIN_RECITATION)


def test_store_needs_write_rejects_a_drifted_recitation_span(tmp_path):
    # Same recitation-grid origin, but a clip already written at one recitation span is
    # re-run with a different (stale) span: the stored windows are on a different grid, so
    # resuming must raise rather than skip the clip with its old windows. Both axes of the
    # span — the start AND the num_samples — are validated: a clip whose floored start is
    # unchanged but whose recitation length drifted (different decode/segmentation → a
    # different tail-window length / window count) must also refuse, not skip with stale
    # windows that would pair with full-length phoneme CTC targets.
    with SoftLabelStore.open(tmp_path, WindowContract(), WINDOW_ORIGIN_RECITATION) as store:
        store.write_clip(
            "clip_a.wav",
            _windows_for([np.array([0.5], dtype=np.float32)]),
            num_samples=96000,
            recitation_start_sample=640,
            recitation_num_samples=80000,
        )
    resumed = SoftLabelStore.open(tmp_path, WindowContract(), WINDOW_ORIGIN_RECITATION)
    assert resumed.needs_write("clip_b.wav", 0, 80000) is True  # unseen clip → write it
    assert resumed.needs_write("clip_a.wav", 640, 80000) is False  # same span → skip
    with pytest.raises(ValueError, match="different recitation grid"):
        resumed.needs_write("clip_a.wav", 1280, 80000)  # drifted start → refuse to skip
    with pytest.raises(ValueError, match="different recitation grid"):
        # Same floored start, drifted recitation length (end_s drift): still refuse. This is
        # the num_samples-axis corruption the start-only guard missed — the tail window's
        # 40 ms length / window count differs while the start is unchanged.
        resumed.needs_write("clip_a.wav", 640, 81920)
    resumed.close()


def test_store_needs_write_recovers_full_span_from_the_index(tmp_path):
    # The persisted index carries BOTH recitation_start_sample and recitation_num_samples,
    # so a fresh store reopened from disk validates the full span (not just the start).
    with SoftLabelStore.open(tmp_path, WindowContract(), WINDOW_ORIGIN_RECITATION) as store:
        store.write_clip(
            "clip_a.wav",
            _windows_for([np.array([0.5], dtype=np.float32)]),
            num_samples=96000,
            recitation_start_sample=640,
            recitation_num_samples=80000,
        )
    record = json.loads(
        (tmp_path / SoftLabelStore.INDEX_NAME).read_text().strip().splitlines()[0]
    )
    assert record["recitation_start_sample"] == 640
    assert record["recitation_num_samples"] == 80000

    reopened = _read_index_spans(tmp_path / SoftLabelStore.INDEX_NAME)
    assert reopened["clip_a.wav"] == (640, 80000)


def test_read_index_spans_defaults_old_rows_to_the_whole_clip_span(tmp_path):
    # An older index line (written before recitation_num_samples existed) defaults to the
    # whole-clip span (0, num_samples) so needs_write never mistakes a legacy row for a
    # drifted recitation grid.
    index_path = tmp_path / SoftLabelStore.INDEX_NAME
    index_path.write_text(
        json.dumps({"audio_filename": "legacy.wav", "num_samples": 48000, "windows": []})
        + "\n",
        encoding="utf-8",
    )
    assert _read_index_spans(index_path) == {"legacy.wav": (0, 48000)}


def test_store_resume_under_a_different_contract_fails_fast(tmp_path):
    # Re-running with a changed window/hop must NOT silently skip existing clips and
    # leave stale arrays — that is silent training-label corruption. It must raise.
    with SoftLabelStore.open(tmp_path, WindowContract(hop_feature_frames=250)):
        pass
    with pytest.raises(ValueError, match="different contract"):
        SoftLabelStore.open(tmp_path, WindowContract(hop_feature_frames=124))


def test_store_resume_under_the_same_contract_is_allowed(tmp_path):
    contract = WindowContract(feature_frames=50, hop_feature_frames=24)
    with SoftLabelStore.open(tmp_path, contract) as store:
        store.write_clip("clip_a.wav", _windows_for([np.array([0.5], dtype=np.float32)], contract), num_samples=16000)
    resumed = SoftLabelStore.open(tmp_path, contract)  # identical contract → no raise
    assert resumed.has("clip_a.wav")
    resumed.close()


# --- generate_soft_labels: no silent whole-clip reuse under the recitation grid ----


def test_generate_refuses_to_resume_a_whole_clip_store_with_recitation_spans(tmp_path):
    # The reviewer's scenario: an existing store was generated WITHOUT recitation spans
    # (whole-clip, 0-origin windows). Re-running generate_soft_labels with a lead-in
    # recitation span must REFUSE — reusing the stale whole-clip windows would pair them
    # with the new recitation-relative phoneme labels (silent joint-label corruption).
    pytest.importorskip("torch")  # generate_soft_labels imports torch before it opens the store
    from tadabur.manifest import ManifestRecord
    from training.waqf_distill import generate_soft_labels

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    out_dir = tmp_path / "soft_labels"

    # A pre-existing whole-clip store (origin whole-clip, one clip at start_sample 0).
    with SoftLabelStore.open(out_dir, WindowContract(), WINDOW_ORIGIN_WHOLE_CLIP) as store:
        store.write_clip(
            "clip_a.wav",
            _windows_for([np.array([0.5], dtype=np.float32)]),
            num_samples=80000,
            recitation_start_sample=0,
        )

    records = [ManifestRecord("clip_a.wav", "1:1", 1.0, 5.0, 7)]
    # A lead-in span (start_sample > 0): the recitation-grid origin differs from the stored
    # whole-clip origin, so the run must raise before touching the VAD.
    with pytest.raises(ValueError, match="different contract"):
        generate_soft_labels(
            records,
            clips_dir,
            out_dir,
            recitation_spans={"clip_a.wav": (SAMPLES_PER_STUDENT_FRAME, 60000)},
        )


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


# --- SoftLabelReader: the read-side join partner for the joint fine-tune ------


def _write_recitation_store(tmp_path, windows):
    """Write a recitation-grid store (the only grid the joint run may join) and return root."""
    with SoftLabelStore.open(
        tmp_path, WindowContract(), window_origin=WINDOW_ORIGIN_RECITATION
    ) as store:
        store.write_clip(
            "clip_a.wav",
            _windows_for(windows),
            num_samples=500 * SAMPLES_PER_TEACHER_FRAME,
            recitation_num_samples=500 * SAMPLES_PER_TEACHER_FRAME,
        )
    return tmp_path


def test_reader_serves_each_window_by_key(tmp_path):
    from training.waqf_distill import SoftLabelReader

    w0 = np.array([0.1, 0.9], dtype=np.float32)
    w1 = np.array([0.4, 0.2, 0.7], dtype=np.float32)
    root = _write_recitation_store(tmp_path, [w0, w1])

    reader = SoftLabelReader.open(root)
    assert reader.has("clip_a.wav", 0)
    assert not reader.has("clip_a.wav", 5)
    target = reader.target("clip_a.wav", 1)
    assert target.window_index == 1
    assert target.start_sample == WindowContract().hop_samples
    np.testing.assert_array_equal(target.silence_40ms, w1)


def test_reader_rejects_a_whole_clip_store(tmp_path):
    from training.waqf_distill import SoftLabelReader

    # A whole-clip store must NEVER be joined with recitation-relative phoneme labels.
    with SoftLabelStore.open(
        tmp_path, WindowContract(), window_origin=WINDOW_ORIGIN_WHOLE_CLIP
    ) as store:
        store.write_clip(
            "clip_a.wav", _windows_for([np.array([0.5], dtype=np.float32)]), num_samples=80000
        )
    with pytest.raises(ValueError, match="grid"):
        SoftLabelReader.open(tmp_path)


def test_reader_raises_on_unknown_window(tmp_path):
    from training.waqf_distill import SoftLabelReader

    root = _write_recitation_store(tmp_path, [np.array([0.5, 0.5], dtype=np.float32)])
    reader = SoftLabelReader.open(root)
    with pytest.raises(KeyError, match="no window"):
        reader.target("clip_a.wav", 9)


def test_reader_raises_on_truncated_array(tmp_path):
    from training.waqf_distill import SoftLabelReader

    root = _write_recitation_store(tmp_path, [np.array([0.5, 0.5, 0.5], dtype=np.float32)])
    # Corrupt the stored array so its length no longer matches the index.
    np.save(root / SoftLabelStore.ARRAYS_SUBDIR / "clip_a.wav#w0.npy",
            np.array([0.5], dtype=np.float32))
    reader = SoftLabelReader.open(root)
    with pytest.raises(ValueError, match="student frames"):
        reader.target("clip_a.wav", 0)


def test_reader_raises_when_store_missing(tmp_path):
    from training.waqf_distill import SoftLabelReader

    with pytest.raises(FileNotFoundError):
        SoftLabelReader.open(tmp_path / "nonexistent")


def samples_for_feature_frames(feature_frames: int) -> int:
    """The shortest audio span the feature extractor frames as ``feature_frames``."""
    return feature_frames * SAMPLES_PER_TEACHER_FRAME + FEATURE_FRAME_SAMPLE_OFFSET


@pytest.mark.parametrize(
    "num_samples,expected_frames",
    # Golden values read off the real ``SeamlessM4TFeatureExtractor`` for
    # ``obadx/muaalem-model-v3_2`` — the grid the model (and the VAD, which shares the
    # extractor) actually produces. The naive ``num_samples // 320`` disagrees on most of
    # these, which is the off-by-one that made the joint loss reject its own targets.
    [(66287, 206), (58870, 183), (62574, 195), (55648, 173), (47024, 146), (80000, 249)],
)
def test_feature_frames_match_the_real_extractor(num_samples, expected_frames):
    assert feature_frames_for_samples(num_samples) == expected_frames


def test_feature_frames_never_negative_for_sub_frame_audio():
    assert feature_frames_for_samples(0) == 0
    assert feature_frames_for_samples(FEATURE_FRAME_SAMPLE_OFFSET) == 0

"""Tests for the torch-free fixed-window geometry (:mod:`training.windowing`).

The window grid is what every whole-clip label builder cuts, so these golden fixtures pin
the 20 ms → 40 ms lattice relation, the real feature extractor's frame count, the frozen
5 s / 4 s-hop contract, the clip-relative recitation grid and the inward word snap —
without a GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from training.windowing import (
    DEPLOYED_WINDOW_FEATURE_FRAMES,
    FEATURE_FRAME_SAMPLE_OFFSET,
    SAMPLES_PER_STUDENT_FRAME,
    SAMPLES_PER_TEACHER_FRAME,
    Window,
    WindowContract,
    clip_recitation_windows,
    enumerate_recitation_windows,
    enumerate_windows,
    feature_frames_for_samples,
    muaalem_lattice_length,
    recitation_window_span,
    snap_window_to_words,
)

# --- muaalem_lattice_length: the exact 20 ms → 40 ms conv relation -----------


@pytest.mark.parametrize(
    "feature_frames,expected",
    [
        (250, 125),   # the fixed 5 s export window (ADR-0004)
        (249, 125),   # 5 s as the extractor actually frames it → still 125
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


def test_window_lattice_follows_the_extractor_frame_count_not_the_naive_one():
    # A window shorter than the full 5 s: the real extractor emits 206 feature frames for
    # 66 287 samples, the naive ``num_samples // 320`` says 207. The 40 ms lattice the CTC
    # target is checked against must come from the real count.
    num_samples = 66287
    assert feature_frames_for_samples(num_samples) == 206
    assert num_samples // SAMPLES_PER_TEACHER_FRAME == 207
    assert muaalem_lattice_length(feature_frames_for_samples(num_samples)) == 103


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
    # shifted by the clip-relative onset, so every consumer keys the same windows.
    contract = WindowContract()
    start_sample, num_samples = recitation_window_span(0.6, 8.6)  # 9600, 128000
    windows = enumerate_recitation_windows(start_sample, num_samples, contract)
    base = enumerate_windows(num_samples, contract)
    assert [w.index for w in windows] == [w.index for w in base]
    assert [w.num_samples for w in windows] == [w.num_samples for w in base]
    assert [w.start_sample for w in windows] == [start_sample + w.start_sample for w in base]


def test_recitation_windows_drop_the_redundant_overlap_tail():
    # A recitation just past the 4 s hop yields a trailing window that is pure overlap the
    # previous window already covers (its audio ends no later). The grid drops it — otherwise
    # a segment crossing that tail's edge would wrongly exclude the clip.
    contract = WindowContract()
    start_sample, num_samples = recitation_window_span(0.0, 4.4)  # 0, 70400 (< 5 s, > 4 s hop)
    base = enumerate_windows(num_samples, contract)
    windows = enumerate_recitation_windows(start_sample, num_samples, contract)
    assert len(base) == 2 and base[1].start_sample + base[1].num_samples == num_samples
    assert [w.index for w in windows] == [0]  # the redundant second window is dropped


def test_enumerate_recitation_windows_rejects_an_unaligned_onset():
    with pytest.raises(ValueError):
        enumerate_recitation_windows(9601, 64000, WindowContract())


# --- feature_frames_for_samples: the real extractor's frame count ------------


@pytest.mark.parametrize(
    "num_samples,expected_frames",
    # Golden values read off the real ``SeamlessM4TFeatureExtractor`` for
    # ``obadx/muaalem-model-v3_2`` — the grid the model actually produces. The naive
    # ``num_samples // 320`` disagrees on most of these.
    [(66287, 206), (58870, 183), (62574, 195), (55648, 173), (47024, 146), (80000, 249)],
)
def test_feature_frames_match_the_real_extractor(num_samples, expected_frames):
    assert feature_frames_for_samples(num_samples) == expected_frames


def test_feature_frames_never_negative_for_sub_frame_audio():
    assert feature_frames_for_samples(0) == 0
    assert feature_frames_for_samples(FEATURE_FRAME_SAMPLE_OFFSET) == 0


# --- word-snapped windows ------------------------------------------------------


def test_snap_window_shrinks_to_whole_words_on_the_student_lattice():
    # Words at 0.5-1.9 s, 1.9-3.1 s, 3.1-6.0 s within a 0-5 s window.
    snapped = snap_window_to_words(
        Window(index=0, start_sample=0, num_samples=5 * 16000), (0.5, 1.9, 3.1, 6.0)
    )

    assert snapped is not None
    assert snapped.index == 0
    assert snapped.start_sample % 640 == 0 and snapped.num_samples % 640 == 0
    # Rounded *in*: no audio from the words it excludes.
    assert snapped.start_sample >= round(0.5 * 16000)
    assert snapped.start_sample + snapped.num_samples <= round(3.1 * 16000)


def test_snap_window_returns_none_when_no_whole_word_fits():
    # One 9 s word: no whole word fits in the 5 s window.
    assert snap_window_to_words(
        Window(index=1, start_sample=0, num_samples=5 * 16000), (0.0, 9.0)
    ) is None


def test_clip_recitation_windows_without_word_times_is_the_fixed_grid():
    contract = WindowContract()
    assert clip_recitation_windows(0, 12 * 16000, contract) == (
        enumerate_recitation_windows(0, 12 * 16000, contract)
    )


def test_clip_recitation_windows_drops_duplicate_snapped_spans():
    contract = WindowContract()
    # Two windows both snap onto the same single word run -> one training example.
    word_times = (0.0, 1.0, 2.0, 3.0, 20.0)
    windows = clip_recitation_windows(0, 20 * 16000, contract, word_times)

    spans = [(w.start_sample, w.num_samples) for w in windows]
    assert len(spans) == len(set(spans))


def test_snapped_windows_stay_inside_their_nominal_grid_window():
    """#24 parity: snapping never moves a window off the frozen 5 s / 4 s-hop grid.

    The frozen contract owns *which* audio a training example may see; word snapping may
    only shrink that span. Asserted over randomised word layouts so the property holds
    for every clip shape the corpus produces, not just a hand-picked one.
    """
    import random

    contract = WindowContract()
    rng = random.Random(24)
    for _ in range(200):
        duration = rng.uniform(1.0, 40.0)
        edges = sorted(rng.uniform(0.0, duration) for _ in range(rng.randint(1, 30)))
        word_times = (0.0, *edges, duration)
        num_samples = round(duration * 16000)

        nominal = {w.index: w for w in enumerate_recitation_windows(0, num_samples, contract)}
        for snapped in clip_recitation_windows(0, num_samples, contract, word_times):
            base = nominal[snapped.index]
            assert snapped.start_sample >= base.start_sample
            assert (
                snapped.start_sample + snapped.num_samples
                <= base.start_sample + base.num_samples
            )
            assert (snapped.start_sample - base.start_sample) % SAMPLES_PER_STUDENT_FRAME == 0
            assert snapped.num_samples % SAMPLES_PER_STUDENT_FRAME == 0
            assert snapped.num_samples > 0
            assert snapped == snap_window_to_words(base, word_times)


# --- torch-free ------------------------------------------------------------------


def test_importing_windowing_does_not_import_torch():
    # The window geometry runs in the plain CPU env; a fresh interpreter proves the import
    # stays torch-free.
    code = (
        "import sys; import training.windowing; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr

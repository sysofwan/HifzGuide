"""Tests for corpus windowing and splitting.

The windowing and split logic is torch-free and pure, so it is tested directly. The
feature-extraction path needs transformers and real audio and is exercised by the training
smoke run rather than here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training import distill_data as dd


# --- Windowing ----------------------------------------------------------------------


def test_exact_window_yields_one_start():
    assert dd.window_starts(dd.WINDOW_SAMPLES) == [0]


def test_clip_shorter_than_minimum_yields_nothing():
    """Too little audio to be worth a teacher pass."""
    assert dd.window_starts(int(0.5 * dd.SAMPLE_RATE)) == []


def test_short_clip_still_yields_one_padded_window():
    """Short windows are in-distribution: the device pads its preview passes too."""
    starts = dd.window_starts(int(2.0 * dd.SAMPLE_RATE))
    assert starts == [0]


def test_hop_controls_overlap():
    """A 10 s clip at a 2.5 s stride covers the clip without running off the end."""
    ten_seconds = 10 * dd.SAMPLE_RATE
    hop = int(2.5 * dd.SAMPLE_RATE)
    starts = dd.window_starts(ten_seconds, hop_samples=hop)
    assert starts[0] == 0
    assert all(second - first == hop for first, second in zip(starts, starts[1:]))
    assert max(starts) < ten_seconds


def test_walk_stops_once_a_window_reaches_the_end():
    """No window should be emitted that is mostly padding past the clip."""
    clip = 11 * dd.SAMPLE_RATE
    hop = int(2.5 * dd.SAMPLE_RATE)
    starts = dd.window_starts(clip, hop_samples=hop)
    # The last start must be the first one whose window covers the tail.
    assert starts[-1] + dd.WINDOW_SAMPLES >= clip
    assert starts[-2] + dd.WINDOW_SAMPLES < clip


def test_smaller_hop_yields_more_windows():
    clip = 30 * dd.SAMPLE_RATE
    coarse = dd.window_starts(clip, hop_samples=int(2.5 * dd.SAMPLE_RATE))
    fine = dd.window_starts(clip, hop_samples=int(1.25 * dd.SAMPLE_RATE))
    assert len(fine) > len(coarse)


def test_window_starts_rejects_nonpositive_hop():
    with pytest.raises(ValueError, match="hop_samples"):
        dd.window_starts(dd.WINDOW_SAMPLES, hop_samples=0)


# --- Splitting ----------------------------------------------------------------------


def test_split_is_deterministic():
    """Same name, same side -- across runs, machines and directory orderings."""
    assert dd.clip_split("abc.wav") == dd.clip_split("abc.wav")


def test_split_depends_on_the_name_not_the_path():
    first = dd.split_clips([Path("/a/b/clip.wav")], val_fraction=0.5)
    second = dd.split_clips([Path("/completely/other/clip.wav")], val_fraction=0.5)
    assert bool(first[1]) == bool(second[1])


def test_split_respects_the_requested_fraction():
    names = [Path(f"clip_{i:05d}.wav") for i in range(20_000)]
    train, val = dd.split_clips(names, val_fraction=0.02)
    assert len(train) + len(val) == len(names)
    assert len(val) / len(names) == pytest.approx(0.02, abs=0.004)


def test_zero_fraction_keeps_everything_in_train():
    names = [Path(f"c{i}.wav") for i in range(500)]
    train, val = dd.split_clips(names, val_fraction=0.0)
    assert not val
    assert len(train) == 500


def test_split_rejects_out_of_range_fraction():
    with pytest.raises(ValueError, match="val_fraction"):
        dd.clip_split("x.wav", val_fraction=1.0)


def test_train_and_val_clips_are_disjoint():
    """Window-level leakage is the failure this split exists to prevent."""
    names = [Path(f"clip_{i}.wav") for i in range(5_000)]
    train, val = dd.split_clips(names, val_fraction=0.1)
    assert set(train).isdisjoint(set(val))


# --- Contract constants -------------------------------------------------------------


def test_window_matches_the_deployed_shape_contract():
    """5 s at 16 kHz -> 250 stride-2 frames; anything else is not a drop-in."""
    assert dd.WINDOW_SAMPLES == 80_000
    assert dd.FEATURE_FRAMES == 250
    assert dd.FEATURE_DIM == 160


def test_window_ref_key_is_unique_per_window():
    first = dd.WindowRef(Path("a/clip.wav"), 0)
    second = dd.WindowRef(Path("b/clip.wav"), 40_000)
    assert first.key != second.key
    assert first.key == "clip.wav#0"

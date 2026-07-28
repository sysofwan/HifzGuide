"""Tests for the rung-(1) segment-scoped CTC labels (#33 ladder control).

Pure logic over the same segment-manifest / clip-status fixtures
``test_windowed_labels`` uses. What matters here is that rung (1) differs from rung (2)
in *exactly one* variable — the training unit — so these tests pin the shared parts
(clip population, frozen grid, word snapping, slice-never-re-phonetize) and the one part
that must differ (a window never spans two waqf segments).
"""

from __future__ import annotations

import pytest

from training.segmented_labels import (
    EXCLUDE_NO_SEGMENT_WINDOW,
    build_clip_segment_windows,
    build_segmented_labels,
)
from training.test_windowed_labels import CONTRACT, _seg, _status, _status_for
from training.waqf_distill import TARGET_SAMPLE_RATE
from training.windowed_labels import (
    EXCLUDE_DROPPED_SEGMENT,
    EXCLUDE_RE_READ,
    build_windowed_labels,
)


def _two_segment_clip():
    """A 9 s clip of two waqf segments — longer than one 5 s window, so the two builds differ."""
    segs = [
        _seg("c.wav", 0, 0, 2, 0.0, 3.0, "AABB", offsets=(0, 2, 4)),
        _seg("c.wav", 1, 2, 4, 3.0, 9.0, "CCDD", offsets=(0, 2, 4)),
    ]
    status = _status_for("c.wav", segs, 4, word_times=(0.0, 1.5, 3.0, 4.5, 9.0))
    return segs, status


def test_every_window_comes_from_exactly_one_segment():
    segs, status = _two_segment_clip()

    labels, reason = build_clip_segment_windows(segs, status, CONTRACT)

    assert reason is None and labels
    assert all(len(label.segment_indices) == 1 for label in labels)
    for label in labels:
        seg = segs[label.segment_indices[0]]
        assert seg.word_start <= label.word_start < label.word_end <= seg.word_end


def test_window_audio_never_crosses_its_segment_boundary():
    segs, status = _two_segment_clip()

    labels, _ = build_clip_segment_windows(segs, status, CONTRACT)

    for label in labels:
        seg = segs[label.segment_indices[0]]
        assert label.start_sample >= round(seg.start_s * TARGET_SAMPLE_RATE)
        assert (
            label.start_sample + label.num_samples
            <= round(seg.end_s * TARGET_SAMPLE_RATE)
        )


def test_labels_are_sliced_from_the_segment_never_rephonetized():
    segs, status = _two_segment_clip()

    labels, _ = build_clip_segment_windows(segs, status, CONTRACT)

    for label in labels:
        seg = segs[label.segment_indices[0]]
        assert label.phoneme_label == seg.slice_words(label.word_start, label.word_end)


def test_window_index_is_unique_within_a_clip():
    segs, status = _two_segment_clip()

    labels, _ = build_clip_segment_windows(segs, status, CONTRACT)

    indices = [label.window_index for label in labels]
    assert indices == sorted(set(indices)) == list(range(len(labels)))


def test_the_whole_clip_build_does_span_a_segment_boundary():
    """The contrast the ladder measures: rung (2)'s windows may hold two segments."""
    segs, status = _two_segment_clip()

    whole = build_windowed_labels(segs, [status], CONTRACT).labels

    assert any(len(label.segment_indices) > 1 for label in whole)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"skip": "low_alignment"}, "low_alignment"),
        ({"re_reads": 1}, EXCLUDE_RE_READ),
    ],
)
def test_clip_level_gates_match_the_whole_clip_build(kwargs, expected):
    """Both rungs must train on the same clip population, or (1)->(2) is not isolated."""
    segs, _ = _two_segment_clip()
    status = _status_for("c.wav", segs, 4, word_times=(0.0, 1.5, 3.0, 4.5, 9.0), **kwargs)

    assert build_clip_segment_windows(segs, status, CONTRACT)[1] == expected
    assert build_windowed_labels(segs, [status], CONTRACT).reasons[expected] == 1


def test_dropped_segment_excludes_the_clip_in_both_builds():
    segs = [_seg("c.wav", 0, 0, 2, 0.0, 3.0, "AABB", offsets=(0, 2, 4))]
    status = _status_for("c.wav", segs, 4, word_times=(0.0, 1.5, 3.0, 4.5, 9.0))

    assert build_clip_segment_windows(segs, status, CONTRACT)[1] == EXCLUDE_DROPPED_SEGMENT


def test_a_segment_too_short_to_hold_a_word_is_dropped_not_the_clip():
    """An unusable *segment* leaves no un-targeted audio elsewhere, so the clip survives."""
    segs = [
        _seg("c.wav", 0, 0, 2, 0.0, 3.0, "AABB", offsets=(0, 2, 4)),
        # Word 2 runs 4.0-10.0 s: longer than a 5 s window, so no whole word fits.
        _seg("c.wav", 1, 2, 3, 4.0, 10.0, "CC", offsets=(0, 2)),
    ]
    status = _status_for("c.wav", segs, 3, word_times=(0.0, 2.0, 4.0, 10.0))

    labels, reason = build_clip_segment_windows(segs, status, CONTRACT)

    assert reason is None
    assert {label.segment_indices[0] for label in labels} == {0}


def test_a_clip_whose_every_segment_is_unusable_is_excluded():
    segs = [_seg("c.wav", 0, 0, 1, 0.0, 9.0, "AA", offsets=(0, 2))]
    status = _status_for("c.wav", segs, 1, word_times=(0.0, 9.0))

    assert (
        build_clip_segment_windows(segs, status, CONTRACT)[1] == EXCLUDE_NO_SEGMENT_WINDOW
    )


def test_build_raises_on_a_segment_whose_clip_has_no_status():
    segs, status = _two_segment_clip()
    other = _status("other.wav", 4, 9.0, word_times=(0.0, 1.5, 3.0, 4.5, 9.0))

    with pytest.raises(ValueError, match="no clip-status record"):
        build_segmented_labels(segs, [other], CONTRACT)

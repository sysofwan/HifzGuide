"""Unit tests for the torch-free half of VAD pause detection (``tadabur.vad``).

Only :func:`pauses_from_intervals` — the pure interval→gap logic — is covered here; it
needs no torch, transformers, GPU, or the VAD weights. The GPU orchestration
(:func:`tadabur.vad.compute_clip_pauses`) is exercised by the end-to-end pipeline run.
"""

from __future__ import annotations

from tadabur.vad import pauses_from_intervals


def test_pauses_are_the_gaps_between_speech_intervals():
    intervals = [(0.0, 5.0), (5.4, 9.0), (9.7, 12.0)]
    assert pauses_from_intervals(intervals) == [(5.0, 5.4), (9.0, 9.7)]


def test_no_interior_pause_for_a_single_speech_interval():
    assert pauses_from_intervals([(0.0, 8.0)]) == []


def test_empty_intervals_yield_no_pauses():
    assert pauses_from_intervals([]) == []


def test_leading_and_trailing_silence_are_not_interior_pauses():
    # Speech starting after 0.7s and ending before the clip end are edges, not stops:
    # only the gap *between* the two speech spans is a pause.
    intervals = [(0.7, 5.0), (6.0, 10.0)]
    assert pauses_from_intervals(intervals) == [(5.0, 6.0)]


def test_touching_intervals_yield_no_pause():
    # Padding can make two spans meet (or cross); a non-positive gap is not a pause.
    assert pauses_from_intervals([(0.0, 5.0), (5.0, 9.0)]) == []
    assert pauses_from_intervals([(0.0, 5.1), (5.0, 9.0)]) == []

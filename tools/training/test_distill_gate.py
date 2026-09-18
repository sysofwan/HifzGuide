"""Tests for gate-decision agreement.

The scoring half needs two resident models and a GPU. What is tested here is the part that
silently produces a *plausible but wrong* number if it drifts: the surah indexing, and the
aggregation that turns per-clip outcomes into the headline.
"""

from __future__ import annotations

import pytest

from training import distill_gate as dg


# --- The indexing trap --------------------------------------------------------------


def test_surah_is_shifted_from_tadabur_zero_indexing():
    """Filenames carry a 0-indexed surah; the reference cache is keyed 1-indexed.

    ``tadabur.filter.canonical_surah_ayah`` warns that without this shift every clip gates
    against the wrong ayah and nothing passes -- a failure that looks like a bad model
    rather than a bad key.
    """
    assert dg.parse_surah_ayah("tadabur_spk0039_S5_A31_60e0c708_000042.wav") == "6:31"
    # S77 is Al-Naba, the 78th surah -- the example the filter's docstring gives.
    assert dg.parse_surah_ayah("tadabur_spk0001_S77_A1_abc_000000.wav") == "78:1"


def test_ayah_is_not_shifted():
    assert dg.parse_surah_ayah("x_S0_A1_y.wav") == "1:1"


def test_unparseable_names_return_none_rather_than_guessing():
    assert dg.parse_surah_ayah("no_surah_here.wav") is None
    assert dg.parse_surah_ayah("") is None


def test_segment_suffixes_do_not_break_parsing():
    assert dg.parse_surah_ayah("tadabur_spk0008_S18_A90_977dc332_000027__seg0.wav") == "19:90"


# --- Aggregation --------------------------------------------------------------------


def _pair(teacher_passed, student_passed, teacher_ratio=0.8, student_ratio=0.8):
    return (teacher_passed, teacher_ratio, student_passed, student_ratio)


def test_identical_decisions_are_full_agreement():
    report = dg.compare_gates([_pair(True, True), _pair(False, False)])
    assert report.decision_agreement == pytest.approx(1.0)
    assert report.both_passed == 1
    assert report.both_failed == 1


def test_disagreements_are_split_by_direction():
    """Which way a disagreement goes matters: the student passing a clip the teacher
    failed is a different product risk from the reverse."""
    report = dg.compare_gates([_pair(True, False), _pair(False, True), _pair(True, True)])
    assert report.teacher_only_passed == 1
    assert report.student_only_passed == 1
    assert report.decision_agreement == pytest.approx(1 / 3)


def test_ratio_delta_is_absolute():
    """A student scoring above the teacher is as much a divergence as scoring below."""
    report = dg.compare_gates(
        [_pair(True, True, 0.9, 0.7), _pair(True, True, 0.7, 0.9)]
    )
    assert report.mean_abs_ratio_delta == pytest.approx(0.2)
    assert report.mean_teacher_ratio == pytest.approx(0.8)
    assert report.mean_student_ratio == pytest.approx(0.8)


def test_empty_corpus_does_not_divide_by_zero():
    report = dg.compare_gates([])
    assert report.num_clips == 0
    assert report.decision_agreement == pytest.approx(0.0)


def test_agreement_counts_decisions_not_ratios():
    """Two clips can agree on pass/fail while differing widely in score, and that is the
    point of the metric: the gate is a threshold, so only the side of it matters."""
    report = dg.compare_gates([_pair(True, True, 0.99, 0.66)])
    assert report.decision_agreement == pytest.approx(1.0)
    assert report.mean_abs_ratio_delta == pytest.approx(0.33)

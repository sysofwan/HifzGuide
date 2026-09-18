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


# --- Token -> phoneme mapping -------------------------------------------------------


def test_tokens_map_to_phonemes_by_index():
    """PHONEME_ID_TO_CHAR is a tuple indexed by class id, not a mapping.

    Pinned because getting this wrong is silent: an `id in PHONEME_ID_TO_CHAR` membership
    test asks whether the integer is one of the characters, always false, so every decode
    becomes the empty string and every gate scores 0.0. That reads as a model producing
    nothing rather than a broken lookup, which is exactly how it was first observed.
    """
    pytest.importorskip("tadabur.phoneme_vocab")
    from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR

    assert dg.tokens_to_phonemes([1, 2, 3]) == "".join(PHONEME_ID_TO_CHAR[i] for i in (1, 2, 3))
    assert dg.tokens_to_phonemes([]) == ""


def test_blank_is_dropped_and_out_of_range_ids_are_ignored():
    pytest.importorskip("tadabur.phoneme_vocab")
    from tadabur.phoneme_vocab import PHONEME_PAD_ID

    assert dg.tokens_to_phonemes([PHONEME_PAD_ID]) == ""
    assert dg.tokens_to_phonemes([9999, -1]) == ""


def test_a_real_decode_is_not_empty():
    """The assertion that would have caught the bug immediately."""
    pytest.importorskip("tadabur.phoneme_vocab")
    assert len(dg.tokens_to_phonemes([7, 7, 32, 10, 32, 26])) == 6


# --- Recalibration must beat the trivial policy -------------------------------------


def test_pass_everything_baseline_is_the_teacher_pass_rate():
    """The number any threshold search has to beat, and usually does not.

    If the teacher passes 88 of 100 clips, gating nothing scores 88% agreement. A "best
    threshold" near zero is the search rediscovering that, not a recalibration win -- which
    is exactly what the first h384 measurement produced (best bar 0.01 -> 89.0% against an
    88% baseline).
    """
    pairs = [_pair(True, True) for _ in range(88)] + [_pair(False, False) for _ in range(12)]
    report = dg.compare_gates(pairs)
    assert report.always_pass_agreement == pytest.approx(0.88)


def test_recalibration_flag_is_false_when_it_only_matches_trivial():
    # Student ratios carry no information: every clip scores the same, so no threshold
    # can separate them and the best available is pass-everything.
    pairs = [_pair(True, True, 0.9, 0.5) for _ in range(9)] + [_pair(False, False, 0.1, 0.5)]
    report = dg.compare_gates(pairs)
    assert report.as_dict()["recalibration_beats_trivial"] is False


def test_recalibration_flag_is_true_when_a_threshold_genuinely_separates():
    """A student that is merely *offset* is recoverable, and must be reported as such."""
    pairs = [_pair(True, False, 0.9, 0.55) for _ in range(6)] + [
        _pair(False, False, 0.3, 0.2) for _ in range(4)
    ]
    report = dg.compare_gates(pairs)
    assert report.as_dict()["recalibration_beats_trivial"] is True
    assert report.best_threshold_agreement == pytest.approx(1.0)

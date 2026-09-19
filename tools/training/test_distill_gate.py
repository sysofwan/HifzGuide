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


# --- Scoring against a frozen evaluation set ---


def _eval_clip(name, ratio, passed, stratum=None, split="dev"):
    from training.gate_evalset import EvalClip, stratum_for

    return EvalClip(
        filename=name,
        surah_ayah="78:1",
        reciter_id=1,
        shard=20,
        duration_s=4.0,
        stratum=stratum or stratum_for(ratio, 0.65),
        split=split,
        in_population=True,
        in_boundary=True,
        teacher_text="ab",
        teacher_ratio=ratio,
        teacher_passed=passed,
        teacher_insertion_run=0,
        teacher_added_shadda=False,
    )


def _decision(name, ratio, passed, insertion_run=0, added_shadda=False):
    from training.distill_gate import StudentDecision

    return StudentDecision(
        filename=name,
        student_text="ab",
        student_ratio=ratio,
        student_passed=passed,
        student_insertion_run=insertion_run,
        student_added_shadda=added_shadda,
    )


def test_view_report_separates_false_rejections_from_false_acceptances():
    from training.distill_gate import build_view_report

    clips = [
        _eval_clip("a.wav", 0.90, True),
        _eval_clip("b.wav", 0.70, True),     # student flips it to fail -> false rejection
        _eval_clip("c.wav", 0.40, False),
        _eval_clip("d.wav", 0.60, False),    # student flips it to pass -> false acceptance
    ]
    decisions = {
        "a.wav": _decision("a.wav", 0.91, True),
        "b.wav": _decision("b.wav", 0.60, False),
        "c.wav": _decision("c.wav", 0.41, False),
        "d.wav": _decision("d.wav", 0.68, True),
    }
    report = build_view_report("boundary", "dev", clips, decisions, {"near_pass": 10}, 0.65)

    assert report.num_clips == 4
    assert report.agreement == 0.5
    assert report.flip_rate == 0.5
    assert report.errors.false_fails == 1
    assert report.errors.false_passes == 1
    # Same aggregate agreement, opposite product consequences: the report must not merge them.
    assert report.errors.false_fail_rate == 0.5
    assert report.errors.false_pass_rate == 0.5


def test_view_report_ratio_error_records_direction_not_only_magnitude():
    from training.distill_gate import build_view_report

    clips = [_eval_clip("a.wav", 0.80, True), _eval_clip("b.wav", 0.80, True)]
    decisions = {
        "a.wav": _decision("a.wav", 0.70, True),
        "b.wav": _decision("b.wav", 0.70, True),
    }
    report = build_view_report("population", "dev", clips, decisions, {"pass_clear": 5}, 0.65)
    assert report.ratio_offset == pytest.approx(-0.10)
    assert report.ratio_rmse == pytest.approx(0.10)
    assert report.ratio_p95_abs_delta == pytest.approx(0.10)


def test_view_report_flags_a_student_that_only_matches_pass_everything():
    """A rubber stamp must be called one, whatever its raw agreement looks like."""
    from training.distill_gate import build_view_report

    clips = [_eval_clip(f"p{i}.wav", 0.90, True) for i in range(9)]
    clips.append(_eval_clip("f0.wav", 0.10, False))
    decisions = {clip.filename: _decision(clip.filename, 0.9, True) for clip in clips}

    report = build_view_report("population", "dev", clips, decisions, {"pass_clear": 9}, 0.65)
    assert report.agreement == 0.9
    assert report.always_pass_agreement == 0.9
    assert not report.trivial_guard.as_dict()["a_beats_b"]
    assert "NOT significantly better" in __import__(
        "training.distill_gate", fromlist=["format_view_report"]
    ).format_view_report(report)


def test_view_report_reweights_enriched_strata_back_to_the_population():
    from training.distill_gate import build_view_report

    # The boundary view is half near-bar, but near-bar clips are 4% of the real corpus.
    clips = [_eval_clip(f"n{i}.wav", 0.60, False, stratum="near_fail") for i in range(2)]
    clips += [_eval_clip(f"p{i}.wav", 0.95, True, stratum="pass_clear") for i in range(2)]
    decisions = {clip.filename: _decision(clip.filename, clip.teacher_ratio, True) for clip in clips}
    # Both near_fail clips flip, both pass_clear agree -> raw 50%.
    report = build_view_report(
        "boundary", "dev", clips, decisions, {"near_fail": 4, "pass_clear": 96}, 0.65
    )
    assert report.agreement == 0.5
    assert report.reweighted_agreement == pytest.approx(0.96)


def test_agreement_is_reported_under_each_definition_of_the_gate():
    """One number over the shipped gate answers two questions and reports neither.

    On the real baseline the same 8.6-point gap is 3.2 points of decode fidelity and 5.4
    points of one asymmetric poison heuristic, and only the split says which to work on.
    """
    from training.distill_gate import build_view_report

    # Teacher rejects both on added shadda (ratio high, no insertion run); the student
    # reproduces neither. Under the shipped gate that is two flips; under either relaxed
    # definition both sides agree.
    clips = [
        _eval_clip("a.wav", 0.90, False, stratum="pass_clear"),
        _eval_clip("b.wav", 0.88, False, stratum="pass_clear"),
        _eval_clip("c.wav", 0.30, False, stratum="fail_clear"),
    ]
    decisions = {
        "a.wav": _decision("a.wav", 0.89, True),
        "b.wav": _decision("b.wav", 0.87, True),
        "c.wav": _decision("c.wav", 0.31, False),
    }
    report = build_view_report(
        "population", "dev", clips, decisions, {"pass_clear": 10}, 0.65
    )
    assert report.agreement_by_condition["full"] == pytest.approx(1 / 3)
    assert report.agreement_by_condition["no_added_shadda"] == pytest.approx(1.0)
    assert report.agreement_by_condition["ratio_only"] == pytest.approx(1.0)
    assert "REJECT_ADDED_SHADDA" in __import__(
        "training.distill_gate", fromlist=["format_view_report"]
    ).format_view_report(report)

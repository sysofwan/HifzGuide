"""Tests for the `.balanced` scorer gate (the Tadabur filter, ADR-0001)."""

import pytest

from tadabur.scorer import (
    BALANCED,
    BALANCED_SCORER,
    MIN_QUERY_PHONEMES,
    Scorer,
    ScoringParameters,
)


def test_balanced_params_verbatim():
    # Muraja's ScoringParameters.balanced values.
    assert BALANCED == ScoringParameters(
        correct_threshold=0.65, soft_pairs_enabled=True, shaddah_suppression=True
    )
    assert BALANCED_SCORER.params is BALANCED


def test_gate_perfect_match_passes():
    result = BALANCED_SCORER.gate("بتثج", "بتثج")
    assert result.passed
    assert result.match_ratio == pytest.approx(1.0, abs=1e-3)


def test_gate_match_ratio_is_score_over_query_phonemes():
    # Query is a clean substring of the reference: score 4 over 4 query phonemes.
    result = BALANCED_SCORER.gate("بتثج", "ابتثجحخ")
    assert result.match_ratio == pytest.approx(1.0, abs=1e-3)
    assert result.passed


def test_gate_short_query_fails():
    # Fewer than MIN_QUERY_PHONEMES non-space phonemes → no trusted alignment.
    assert MIN_QUERY_PHONEMES == 3
    result = BALANCED_SCORER.gate("بت", "بتثج")
    assert not result.passed
    assert result.match_ratio == 0.0


def test_gate_spaces_not_counted_toward_query_length():
    # Spaces don't count as phonemes: "ب ت" has only 2 → below the minimum.
    result = BALANCED_SCORER.gate("ب ت", "ب ت")
    assert not result.passed


def test_gate_unrelated_strings_fail():
    result = BALANCED_SCORER.gate("بتث", "محك")
    assert not result.passed
    assert result.match_ratio < BALANCED.correct_threshold


def test_gate_soft_pair_substitution_scores_high():
    # ص in place of س (a balanced soft pair) is heavily credited by graduated
    # mismatch, so a clip differing only by a soft pair still clears the bar.
    baseline = BALANCED_SCORER.gate("سلم", "سلم")
    soft = BALANCED_SCORER.gate("صلم", "سلم")
    assert baseline.passed
    assert soft.passed
    assert soft.match_ratio < baseline.match_ratio


def test_gate_reference_normalization_idempotent():
    # A pre-normalized reference (the cache format) gates identically to its raw
    # shadda-run form, since normalization collapses the run either way.
    normalized = BALANCED_SCORER.gate("رب", "رب لمشرقين")
    raw = BALANCED_SCORER.gate("رب", "رَببُ لمشرقين")
    assert normalized == raw


def test_is_soft_mismatch_respects_mode():
    dhal, zai = "\u0630", "\u0632"
    assert BALANCED_SCORER.is_soft_mismatch(dhal, zai)
    strict = Scorer(ScoringParameters(0.75, soft_pairs_enabled=False, shaddah_suppression=False))
    assert not strict.is_soft_mismatch(dhal, zai)

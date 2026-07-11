"""Tests for the `.balanced` scorer gate (the Tadabur filter, ADR-0001)."""

import pytest

from tadabur.scorer import (
    BALANCED,
    BALANCED_SCORER,
    MAX_INSERTION_RUN,
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


def test_gate_uses_reference_verbatim_preserving_shadda():
    # normalize_phonemes is a Swift-faithful port and is NOT idempotent: it uses
    # combining marks to keep shadda gemination doubled, then strips them, so a
    # second pass collapses the doubling (رَببُ → ربب → رب). The gate therefore
    # must take an already-normalized reference (the cache form) verbatim, or it
    # would wrongly collapse the reference's shadda. Passing the RAW reference
    # (which the gate would re-normalize once, keeping shadda) and its correctly
    # pre-normalized form must gate identically.
    from tadabur.normalization import normalize_phonemes

    raw_ref = "رَببُ لمشرقين"
    pre_normalized = normalize_phonemes(raw_ref).normalized
    assert pre_normalized == "ربب لمشرقين"  # shadda kept doubled (single pass)
    # Query heard the doubled shadda too (raw, with marks, so its single
    # normalization keeps the doubling); against the correctly-normalized
    # reference both carry ربب, so it matches cleanly.
    result = BALANCED_SCORER.gate("رَببُ لمشرقين", pre_normalized)
    assert result.passed
    assert result.match_ratio == pytest.approx(1.0, abs=1e-3)


def test_gate_rejects_repeated_phrase_despite_passing_ratio():
    # A reciter who repeats a phrase inserts a run of query phonemes the reference
    # lacks. The local aligner shrugs those off (match_ratio stays above the bar), but
    # the poison policy rejects any interior insertion run >= MAX_INSERTION_RUN.
    assert MAX_INSERTION_RUN == 5
    ref = "بتثجحخدذرزسشصضطظعغفقكلمنهوي"
    repeated = "بتثجحخدذرز" + "سشصضطظ" + "سشصضطظعغفقكلمنهوي"  # سشصضطظ said twice
    result = BALANCED_SCORER.gate(repeated, ref)
    assert result.max_insertion_run >= MAX_INSERTION_RUN
    assert result.match_ratio >= BALANCED.correct_threshold  # would pass on ratio alone
    assert not result.passed  # but the insertion-run poison policy rejects it


def test_gate_tolerates_short_insertion_run():
    # A single spurious inserted phoneme (below the run threshold) does not trip the
    # poison policy; the pair still passes on match_ratio.
    ref = "بتثجحخدذرزسشصضطظعغفقكلمنهوي"
    one_extra = "بتثجحخدذرز" + "س" + "سشصضطظعغفقكلمنهوي"
    result = BALANCED_SCORER.gate(one_extra, ref)
    assert result.max_insertion_run < MAX_INSERTION_RUN
    assert result.passed


def test_gate_perfect_match_has_no_insertion_run():
    assert BALANCED_SCORER.gate("بتثج", "بتثج").max_insertion_run == 0


def test_gate_exposes_edge_trims_without_failing():
    # Extra query phonemes trailing past the reference are trimmed by the local aligner,
    # not scored as interior insertions. The gate exposes the trim observationally; it
    # does NOT fail the pair (a whole clip may legitimately end mid-phrase — only
    # segment_score treats an interior-boundary trim as a mislabel).
    ref = "بتثجحخدذرز"
    overrun = ref + "سشصضطظعغف"  # 9 extra trailing phonemes
    result = BALANCED_SCORER.gate(overrun, ref)
    assert result.trailing_trim >= 5
    assert result.leading_trim == 0
    assert result.max_insertion_run == 0  # a trailing overrun is trimmed, not an insertion


def test_gate_perfect_match_has_no_edge_trims():
    result = BALANCED_SCORER.gate("بتثج", "بتثج")
    assert result.leading_trim == 0 and result.trailing_trim == 0


def test_is_soft_mismatch_respects_mode():
    dhal, zai = "\u0630", "\u0632"
    assert BALANCED_SCORER.is_soft_mismatch(dhal, zai)
    strict = Scorer(ScoringParameters(0.75, soft_pairs_enabled=False, shaddah_suppression=False))
    assert not strict.is_soft_mismatch(dhal, zai)

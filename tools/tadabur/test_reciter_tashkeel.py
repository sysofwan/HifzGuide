"""Tests for the reciter-layer tashkeel filter (ADR-0005)."""

import pytest

from tadabur.reciter_tashkeel import (
    MAX_SWAP_RATE,
    MIN_REFERENCE_VOWELS,
    judge,
    score_reciters,
    summarize,
    wilson_lower_bound,
)
from training.tashkeel_eval import VowelCounts


def _counts(swapped: int, matched: int) -> VowelCounts:
    return VowelCounts(matched=matched, swapped=swapped)


def test_a_reciter_who_never_swaps_is_kept():
    verdicts = judge({"clean": _counts(0, 5000)})
    assert verdicts[0].excluded is False
    assert verdicts[0].judged is True


def test_a_systematically_deviating_reciter_is_excluded():
    """The case the filter exists for: a different qira'ah swaps vowels at percent scale."""
    verdicts = judge({"non_hafs": _counts(300, 4700)})
    assert verdicts[0].swap_rate == pytest.approx(0.06)
    assert verdicts[0].excluded is True


def test_a_reciter_with_too_few_vowels_is_never_excluded():
    """Below the evidence floor the answer is "unknown", not "guilty".

    Otherwise the filter would preferentially drop reciters who simply have few clips.
    """
    verdicts = judge({"sparse": _counts(20, 20)})
    assert verdicts[0].swap_rate == pytest.approx(0.5)
    assert verdicts[0].judged is False
    assert verdicts[0].excluded is False


def test_the_same_swap_rate_is_judged_differently_at_different_volumes():
    """Judged on the lower bound, so a point estimate above the ceiling is not enough.

    At the evidence floor a reciter can sit at nearly twice the threshold and still be kept --
    the interval does not yet support the claim. Ten times the volume at the *same* rate does
    support it. This is the whole reason the filter reads the bound and not the estimate.
    """
    swaps, matched = 9, MIN_REFERENCE_VOWELS - 9
    sparse = judge({"sparse": VowelCounts(matched=matched, swapped=swaps)})[0]
    dense = judge({"dense": VowelCounts(matched=matched * 10, swapped=swaps * 10)})[0]

    assert sparse.swap_rate == pytest.approx(dense.swap_rate)
    assert sparse.swap_rate > MAX_SWAP_RATE
    assert sparse.excluded is False
    assert dense.excluded is True


def test_omission_is_never_grounds_for_exclusion():
    """ADR-0005: omission tracks audio quality and pace, not reciter error.

    Gating on it would delete hard-but-correct audio, the opposite of ADR-0003's goal.
    """
    verdicts = judge({"mumbler": VowelCounts(matched=3000, omitted=2000, swapped=0)})
    assert verdicts[0].excluded is False


def test_unanchored_vowels_are_not_counted_as_swaps():
    """A right vowel on a carrier the model missed is a model miss, not a reciter deviation.

    This is the bucket the ADR-0003 preview's 1.3% "swap" actually turned out to be.
    """
    verdicts = judge({"r": VowelCounts(matched=3000, unanchored=2000, swapped=0)})
    assert verdicts[0].swap_rate == 0.0
    assert verdicts[0].excluded is False


def test_reciters_are_reported_worst_first():
    verdicts = judge({"a": _counts(0, 5000), "b": _counts(300, 4700), "c": _counts(10, 4990)})
    assert [v.reciter_id for v in verdicts] == ["b", "c", "a"]


def test_scoring_groups_rows_by_reciter():
    fatha = "\u064e"
    word = f"\u0642{fatha}\u062f"
    rows = [
        {"reciter_id": 7, "predicted_phonemes": word, "raw_reference_phonemes": word},
        {"reciter_id": 7, "predicted_phonemes": word, "raw_reference_phonemes": word},
        {"reciter_id": 9, "predicted_phonemes": word, "raw_reference_phonemes": word},
    ]
    per = score_reciters(rows)
    assert set(per) == {"7", "9"}
    assert per["7"].reference_total == 2 * per["9"].reference_total


def test_summary_reports_the_distribution_not_just_the_exclusions():
    """The shape of the distribution is the actual evidence about contamination.

    A unimodal spread with no outliers is what says "no non-Hafs population here" -- an
    exclusion count of zero alone would not distinguish that from a broken filter.
    """
    summary = summarize(judge({f"r{i}": _counts(i, 5000) for i in range(10)}))
    assert summary["reciters"] == 10
    assert summary["judged"] == 10
    assert summary["excluded"] == 0
    assert summary["swap_rate_max"] > summary["swap_rate_median"]
    assert summary["corpus_swap_rate"] is not None


def test_summary_names_the_excluded_reciters():
    summary = summarize(judge({"bad": _counts(300, 4700), "good": _counts(0, 5000)}))
    assert summary["excluded"] == 1
    assert summary["excluded_reciters"] == ["bad"]


@pytest.mark.parametrize("successes,total", [(0, 100), (1, 1000), (50, 100)])
def test_wilson_lower_bound_never_exceeds_the_point_estimate(successes, total):
    assert wilson_lower_bound(successes, total) <= successes / total


def test_wilson_lower_bound_is_zero_with_no_observations():
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(0, 100) == 0.0

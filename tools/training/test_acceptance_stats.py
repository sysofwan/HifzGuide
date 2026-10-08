"""The statistics and verdicts of the acceptance rules (§1, §2, §5, §9), pinned before any candidate."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from training.acceptance_stats import (
    BIAS_SPLIT_SALT,
    CANNOT_CERTIFY,
    FAIL,
    LOWER,
    PASS,
    REPLICATES,
    SCORE,
    SEED,
    TUNE,
    UPPER,
    AgreementRecord,
    Resample,
    agreement_terms,
    aggregate,
    difference_verdict,
    independent_equal_weight,
    is_degenerate,
    is_sparse,
    lower_bound,
    project_to_teacher,
    quantile_type7,
    ratio,
    ratio_verdict,
    reciter_half,
    relative_change_verdict,
    upper_bound,
)


def test_frozen_constants():
    assert (REPLICATES, SEED) == (10_000, 20261008)
    assert BIAS_SPLIT_SALT == "issue-91-blank-bias-2026-10"


# --- aggregation truth table -------------------------------------------------------------


@pytest.mark.parametrize(
    "verdicts, expected",
    [
        ([PASS, PASS], PASS),
        ([PASS, FAIL], FAIL),
        ([CANNOT_CERTIFY, FAIL], FAIL),
        ([PASS, CANNOT_CERTIFY], CANNOT_CERTIFY),
        ([], CANNOT_CERTIFY),
    ],
)
def test_aggregate_truth_table(verdicts, expected):
    assert aggregate(verdicts) == expected


def test_aggregate_refuses_a_non_verdict():
    with pytest.raises(ValueError):
        aggregate(["passed"])


# --- percentiles and adverse endpoints ---------------------------------------------------


def test_type7_matches_numpy_on_finite_samples():
    values = np.random.default_rng(0).normal(size=137)
    for q in (0.0, 0.025, 0.5, 0.975, 1.0):
        assert quantile_type7(values, q) == pytest.approx(np.percentile(values, 100 * q))


def test_undefined_replicates_take_the_adverse_endpoint():
    replicates = np.array([np.nan] * 300 + list(np.linspace(0.5, 1.0, 9_700)))
    assert lower_bound(replicates) == -math.inf  # 3% undefined reaches the 2.5th percentile
    assert upper_bound(replicates) == math.inf  # an upper bound is pushed the other way
    few = np.array([np.nan] * 100 + [0.9] * 9_900)
    assert lower_bound(few) == 0.9  # 1% undefined stays inside the tail it cannot reach


def test_interpolating_next_to_an_infinite_end_returns_it():
    assert quantile_type7(np.array([-math.inf, 1.0]), 0.5) == -math.inf
    assert quantile_type7(np.array([1.0, math.inf]), 0.5) == math.inf


def test_ratio_is_undefined_on_a_zero_denominator():
    assert math.isnan(ratio(np.zeros(3), np.zeros(3)))
    assert ratio(np.array([1.0, 0.0]), np.array([2.0, 2.0])) == 0.25


# --- the clustered bootstrap -------------------------------------------------------------


def test_resample_is_deterministic_and_draws_whole_clusters():
    clusters = [3, 3, 1, 2, 2, 2]
    a, b = Resample.by_cluster(clusters), Resample.by_cluster(list(reversed(clusters)))
    assert a.clusters == (1, 2, 3)
    assert np.array_equal(a.counts, b.counts)  # the draw depends on the sorted ids only
    assert a.counts.shape == (REPLICATES, 3)
    assert np.all(a.counts.sum(axis=1) == 3)
    totals = a.totals(np.ones(6))  # each replicate's site count: clusters of size 1, 3, 2
    assert set(np.unique(totals)) <= {3, 4, 5, 6, 7, 8, 9}


def test_a_paired_difference_shares_one_draw():
    clusters = list(range(30))
    num = np.array([1.0] * 20 + [0.0] * 10)
    resample = Resample.by_cluster(clusters)
    same = resample.ratios(num, np.ones(30)) - resample.ratios(num.copy(), np.ones(30))
    assert np.all(same == 0)


def test_replicate_ratios_are_undefined_where_the_denominator_is_drawn_empty():
    resample = Resample.by_cluster([0, 1])
    replicates = resample.ratios(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert np.isnan(replicates).any() and not np.isnan(replicates).all()


def test_sparse_and_degenerate():
    assert is_sparse(19, 10) and is_sparse(20, 9) and not is_sparse(20, 10)
    assert is_degenerate(np.zeros(5)) and is_degenerate(np.full(5, 0.7))
    assert not is_degenerate(np.array([0.1, 0.2]))


def test_independence_needs_one_site_per_reciter_unit_weights_and_fixed_denominators():
    assert independent_equal_weight([1, 2, 3], np.ones(3))
    assert not independent_equal_weight([1, 1, 3], np.ones(3))
    assert not independent_equal_weight([1, 2, 3], np.ones(3), weights=np.array([1, 2, 1]))
    assert not independent_equal_weight([1, 2, 3], np.array([1.0, 0.0, 1.0]))


# --- verdicts ----------------------------------------------------------------------------


def _sites(successes: int, failures: int, per_reciter: int = 1):
    n = successes + failures
    clusters = [i // per_reciter for i in range(n)]
    return clusters, np.array([1.0] * successes + [0.0] * failures), np.ones(n)


def test_bootstrap_lower_bound_pass_and_fail():
    rng = np.random.default_rng(1)
    clusters = [i // 4 for i in range(400)]
    num = (rng.random(400) < 0.97).astype(float)
    assert ratio_verdict(clusters, num, np.ones(400), 0.90, LOWER).verdict == PASS
    result = ratio_verdict(clusters, num, np.ones(400), 0.99, LOWER)
    assert result.verdict == FAIL and result.method == "bootstrap"


def test_a_zero_denominator_never_passes():
    result = ratio_verdict([1, 2], np.zeros(2), np.zeros(2), 0.0, LOWER)
    assert result.verdict == CANNOT_CERTIFY and result.bound is None


def test_wilson_flawless_floors_of_the_rules():
    # §4: 73 flawless correct sites put the Wilson upper bound on false flags at 5.0%.
    clusters, num, den = _sites(0, 73)
    result = ratio_verdict(clusters, num, den, 0.05, UPPER)
    assert result.method == "wilson" and result.bound == pytest.approx(3.8416 / 76.8416, abs=1e-4)
    assert result.verdict == PASS
    # §8: 35 flawless commits give a Wilson lower bound of 90.1% on committed accuracy.
    clusters, num, den = _sites(35, 0)
    result = ratio_verdict(clusters, num, den, 0.90, LOWER)
    assert result.method == "wilson" and result.bound == pytest.approx(35 / 38.8416, abs=1e-4)


def test_a_sparse_cell_that_is_not_independent_cannot_certify():
    clusters, num, den = _sites(10, 0, per_reciter=2)
    assert ratio_verdict(clusters, num, den, 0.5, LOWER).verdict == CANNOT_CERTIFY
    clusters, num, den = _sites(10, 0)
    weighted = ratio_verdict(clusters, 2 * num, 2 * den, 0.5, LOWER, weights=np.full(10, 2.0))
    assert weighted.verdict == CANNOT_CERTIFY


def test_a_degenerate_large_cell_falls_back_to_the_exact_rule():
    clusters, num, den = _sites(200, 0)  # every replicate is exactly 1.0
    result = ratio_verdict(clusters, num, den, 0.95, LOWER)
    assert result.method == "wilson" and result.verdict == PASS


def test_tango_flawless_floor_of_the_rules():
    # §8: a fixed-denominator paired guard at 2 pts needs n >= 189 with zero discordances.
    clusters, num, den = _sites(189, 0)
    result = difference_verdict(clusters, (num, den), (num.copy(), den), 0.02, UPPER)
    assert result.method == "tango" and result.bound == pytest.approx(3.8416 / 192.8416, abs=1e-4)
    assert result.verdict == PASS


def test_paired_difference_is_recomputed_per_replicate_not_on_the_intersection():
    rng = np.random.default_rng(2)
    clusters = [i // 5 for i in range(500)]
    arm = (rng.random(500) < 0.9).astype(float)
    comparator = (rng.random(500) < 0.6).astype(float)
    result = difference_verdict(clusters, (arm, np.ones(500)), (comparator, np.ones(500)), 0.2, LOWER)
    assert result.method == "bootstrap" and result.verdict == PASS


def test_relative_change_needs_a_nonzero_comparator():
    clusters = [i // 2 for i in range(100)]
    zero = (np.zeros(100), np.ones(100))
    assert relative_change_verdict(clusters, zero, zero, -0.25, UPPER).verdict == CANNOT_CERTIFY
    today = ((np.arange(100) % 5 == 0).astype(float), np.ones(100))
    candidate = ((np.arange(100) % 25 == 0).astype(float), np.ones(100))
    result = relative_change_verdict(clusters, candidate, today, -0.25, UPPER)
    assert result.verdict == PASS and result.bound < -0.25


# --- teacher agreement -------------------------------------------------------------------


def test_projection_removes_sukun_and_expands_shaddah():
    baa, fatha, sukun, shaddah = "ب", "َ", "ْ", "ّ"
    assert project_to_teacher(baa + sukun) == baa
    assert project_to_teacher(baa + fatha + shaddah) == baa + baa + fatha
    assert project_to_teacher(baa + shaddah + fatha) == baa + baa + fatha


def test_agreement_is_one_minus_edits_over_teacher_tokens():
    records = [
        AgreementRecord(1, "abcd", "abcd"),
        AgreementRecord(1, "abcd", "abxd"),
        AgreementRecord(2, "ab", "a"),
    ]
    clusters, num, den = agreement_terms(records)
    assert clusters == [1, 1, 2]
    assert ratio(num, den) == pytest.approx(1 - 2 / 10)


# --- reciter split -----------------------------------------------------------------------


def test_reciter_split_algorithm_is_pinned():
    digest = hashlib.sha256(f"{BIAS_SPLIT_SALT}:0".encode("utf-8")).digest()
    assert int.from_bytes(digest[:8], "big") < 2**63
    assert [reciter_half(i) for i in range(8)] == [TUNE] * 4 + [SCORE] + [TUNE] * 3
    tune = sum(reciter_half(i) == TUNE for i in range(1000))
    assert 450 < tune < 550


@pytest.mark.parametrize("bad", [-1, True, "7", 1.0])
def test_reciter_split_refuses_a_non_canonical_id(bad):
    with pytest.raises(ValueError):
        reciter_half(bad)

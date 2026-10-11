"""The §8 simulation machinery (lifted from #105): generator, draws, bounds, search. Torch-free."""

from __future__ import annotations

import math
from collections import Counter
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tadabur.power_frame import FIELDS
from training import power_simulation as ps
from training.acceptance_stats import FAIL, LOWER, PASS, UPPER, Resample, difference_verdict, ratio_verdict

_FIELD = {name: i for i, name in enumerate(FIELDS)}


def stratum(name: str, sites_per_reciter, *, first_reciter=0, **rates) -> ps.Stratum:
    """A synthetic stratum: ``sites_per_reciter`` on consecutive reciters, each ``rates`` field
    that share of every reciter's sites."""
    sites = np.asarray(sites_per_reciter, dtype=np.int64)
    stats = np.zeros((len(sites), len(FIELDS)))
    stats[:, _FIELD["sites"]] = sites
    stats[:, _FIELD["listen_seconds"]] = 12.0 * sites
    for field, rate in rates.items():
        stats[:, _FIELD[field]] = rate * sites
    reciters = np.arange(first_reciter, first_reciter + len(sites), dtype=np.int64)
    return ps.Stratum(name, "consonant", "ت↔ط", reciters, sites, stats, 12.0)


# --- known answers -----------------------------------------------------------------------


def test_the_flawless_floors_in_the_committed_functions():
    """§8: a fixed-denominator paired guard at 2 pts needs n ≥ 189 with no discordance; a
    flawless Wilson lower bound reaches 90% at 35 commits."""
    for n, verdict in ((189, PASS), (188, FAIL)):
        ones = np.ones(n)
        assert difference_verdict(range(n), (ones, ones), (ones, ones), -0.02, LOWER).verdict == verdict
    for n, verdict in ((35, PASS), (34, FAIL)):
        ones = np.ones(n)
        assert ratio_verdict(range(n), ones, ones, 0.90, LOWER).verdict == verdict


@pytest.mark.parametrize("n, verdict", [(73, PASS), (72, FAIL)])
def test_a_flawless_paired_guard_at_5_points_needs_73_reciters(n, verdict):
    """One site per reciter: Tango at zero discordance gives z²/(n + z²) ≤ 5% from n = 73
    (3.8416 / 76.84 = 4.9995%)."""
    assert ps.paired_unit_verdict(np.zeros(n), np.zeros(n), 0.05, UPPER) == verdict


def test_the_point_verdict_reads_a_census():
    """A census point rule passes exactly at the threshold; an undefined value cannot certify."""
    assert ps.point_verdict(0.7, 0.7, LOWER) == PASS and ps.point_verdict(0.69, 0.7, LOWER) == FAIL
    assert ps.point_verdict(math.nan, 0.7, LOWER) == "cannot_certify"


def test_the_sign_test_veto_is_exact():
    # P(Bin(10, ½) ≥ 9) = 11 / 1024 = 1.07%: a veto at 5%, not at 5% / 14.
    assert ps.sign_test_harm(9, 1, 0.05) and not ps.sign_test_harm(9, 1, 0.05 / 14)
    assert not ps.sign_test_harm(0, 0, 0.05)


# --- the generator -----------------------------------------------------------------------


def test_exact_redraw_keeps_both_marginals_and_hits_the_discordance():
    rng = np.random.default_rng(2)
    p = np.array([0.88, 0.01, 0.11])
    rows = np.tile(p, (200_000, 1))
    control = ps.categorical(rows, rng)
    probe = ps.paired(control, rows, 0.05, rng)
    for arm in (control, probe):
        assert np.bincount(arm, minlength=3) / len(arm) == pytest.approx(p, abs=0.003)
    assert (control != probe).mean() == pytest.approx(0.05, abs=0.002)


def test_an_impossible_discordance_is_reported():
    p = np.array([0.99, 0.005, 0.005])
    assert ps.exact_redraw_share(p, 0.05) > 1
    assert ps.exact_redraw_share(p, 0.0) == 0.0


@pytest.mark.parametrize("p", [0.01, 0.3, 0.9])
def test_intercept_for_hits_the_marginal(p):
    sigma = ps.sigma_for(0.15)
    b = ps.intercept_for(np.array([p]), sigma)[0]
    draws = ps._expit(b + sigma * np.random.default_rng(4).standard_normal(400_000))
    assert draws.mean() == pytest.approx(p, abs=0.002)


def test_sigma_for_is_the_latent_icc():
    sigma2 = ps.sigma_for(0.15) ** 2
    assert sigma2 / (sigma2 + math.pi**2 / 3) == pytest.approx(0.15)


def test_clustered_outcomes_keep_the_marginal_and_cluster_by_reciter():
    rng = np.random.default_rng(3)
    reciters = np.repeat(np.arange(2_000), 20)
    outcomes = ps.clustered_bernoulli(0.3, reciters, ps.sigma_for(0.15), rng)
    assert outcomes.mean() == pytest.approx(0.3, abs=0.01)
    per_reciter = outcomes.reshape(2_000, 20).mean(axis=1)
    assert per_reciter.var() > 2 * 0.3 * 0.7 / 20  # more spread than independent sites


# --- draws and the design -------------------------------------------------------------------


def test_allocate_is_proportional_capped_and_whole():
    assert ps.allocate({"a": 1.0, "b": 1.0, "c": 2.0}, {"a": 5, "b": 100, "c": 100}, 45) == \
        {"a": 5, "b": 13, "c": 27}
    assert ps.allocate({"a": 1.0, "b": 0.0}, {"a": 3, "b": 9}, 50, minimum=2) == {"a": 3, "b": 2}


def test_spread_splits_by_largest_remainder():
    assert ps._spread(10, np.array([1.0, 1.0, 1.0])).tolist() == [4, 3, 3]
    assert ps._spread(0, np.array([1.0])).tolist() == [0]


def test_the_union_takes_each_strata_largest_size():
    assert ps.union([{"a": 4}, {"a": 2, "b": 10}, {"c": 0}]) == {"a": 4, "b": 10}


def test_draw_takes_the_design():
    strata = {"a": stratum("a", [3, 1, 6]), "b": stratum("b", [5, 5], first_reciter=10)}
    design = ps.Design({"x": {"a": 4}, "y": {"a": 2, "b": 10}})
    sample = ps.draw(strata, design, np.random.default_rng(9))
    assert np.bincount(sample.stratum).tolist() == [4, 10]
    assert sample.weight.tolist() == [10 / 4] * 4 + [1.0] * 10
    assert sample.component(design, "y").sum() == 12 and sample.component(design, "x").sum() == 4
    assert set(sample.reciter[sample.stratum == 1].tolist()) == {10, 11}
    assert design.minutes(strata) == pytest.approx(14 * 12.0 / 60)


def test_bootstrap_rates_redraw_the_reciters():
    s = stratum("a", [10, 10, 10, 10], h448_mushaf=0.5)
    s.rates_from[0, _FIELD["h448_mushaf"]] = 10  # one reciter always right
    assert s.rates()[_FIELD["h448_mushaf"]] == pytest.approx(25 / 40)
    draws = [s.bootstrap_rates(np.random.default_rng(seed))[_FIELD["h448_mushaf"]]
             for seed in range(200)]
    assert min(draws) >= 0.5 and max(draws) <= 1.0 and np.std(draws) > 0


def test_one_per_reciter_takes_one_site_from_each():
    reciters = np.array([1, 1, 2, 3, 3, 3])
    mask = np.array([True, True, False, True, True, True])
    chosen = ps.one_per_reciter(mask, reciters, np.random.default_rng(8))
    assert sorted(reciters[chosen].tolist()) == [1, 3] and mask[chosen].all()


def test_a_weighted_choice_per_reciter_follows_the_weights():
    reciters = np.zeros(3, dtype=np.int64)
    weights = np.array([1.0, 3.0, 0.0])
    rng = np.random.default_rng(11)
    picks = Counter(int(ps.one_per_reciter(np.ones(3, bool), reciters, rng, weights)[0])
                    for _ in range(4_000))
    assert picks[2] == 0 and picks[1] / 4_000 == pytest.approx(0.75, abs=0.03)


def test_shared_resample_is_by_clusters_draw():
    reciters = np.random.default_rng(10).choice([3, 17, 40, 41, 900], size=60)
    labels, resample = ps.shared_resample(reciters)
    reference = Resample.by_cluster(reciters.tolist())
    assert labels.tolist() == reference.index.tolist()
    assert np.array_equal(resample.counts, reference.counts)


# --- cluster samples ---------------------------------------------------------------------


def test_pps_inclusion_is_proportional_capped_and_sums_to_n():
    pi = ps.pps_inclusion(np.array([50, 1, 1, 2, 0]), 3)
    assert pi[0] == 1.0 and pi[4] == 0.0 and pi.sum() == pytest.approx(3.0)
    assert pi[3] == pytest.approx(2 * pi[1])


def test_the_mean_limit_is_the_chernoff_tail_at_2_5_percent():
    for k in (1, 3):
        mu = ps.poisson_binomial_mean_limit(k)
        assert math.exp(-mu) * (math.e * mu / k) ** k == pytest.approx(0.025, rel=1e-6)


def test_the_cluster_bound_counts_certain_reciters_and_loads_the_rest():
    # No mistake heard: the certain reciter's 2 are counted; the other (π = ½, cost ln 2 of the
    # −ln 0.025 budget) could hold all 4 of its sites unseen.
    bound = ps.cluster_mistake_bound(np.array([1.0, 0.5]), np.array([5, 4]), np.array([2, 0]),
                                     np.array([True, False]))
    assert bound == pytest.approx(6.0)
    # One mistake heard: Σπ ≤ 6.57 lets all three half-sampled reciters be full of mistakes.
    bound = ps.cluster_mistake_bound(np.full(3, 0.5), np.array([10, 10, 10]), np.array([1, 0, 0]),
                                     np.array([True, False, False]))
    assert bound == pytest.approx(30.0)


def test_no_mistake_heard_gets_the_exact_design_bound():
    """200 reciters with one site each, half sampled (π = ½), none a mistake: each unseen
    reciter costs ln 2 of the −ln 0.025 budget, so −ln 0.025 / ln 2 = 5.32 mistakes can hide."""
    holding = np.ones(200, dtype=np.int64)
    pi = ps.pps_inclusion(holding, 100)
    sampled = np.arange(200) % 2 == 0
    bound = ps.cluster_mistake_bound(pi, holding, np.zeros(200, dtype=np.int64), sampled)
    assert bound == pytest.approx(-math.log(0.025) / math.log(2))


def test_concentrated_mistakes_are_counted_whole():
    """Three reciters hold 30 of 163 sites, all mistakes: a sampled one shows all of its
    mistakes, so with every reciter sampled the bound is the true 30."""
    holding = np.array([10, 10, 10] + [3] * 44 + [1])
    mistakes = np.array([10, 10, 10] + [0] * 45)
    pi = ps.pps_inclusion(holding, len(holding))
    assert ps.cluster_mistake_bound(pi, holding, mistakes, np.ones(len(holding), bool)) == 30.0


def test_expected_cluster_sites_weigh_holdings_by_inclusion():
    holding = np.array([50, 1, 1, 2, 0])
    assert ps.expected_cluster_sites(holding, 3) == pytest.approx(
        float(ps.pps_inclusion(holding, 3) @ holding))
    assert ps.expected_cluster_sites(holding, 10) == 54


# --- the power search --------------------------------------------------------------------


def test_rng_for_is_stable_and_keyed():
    assert ps.rng_for("a", 1).random() == ps.rng_for("a", 1).random()
    assert ps.rng_for("a", 1).random() != ps.rng_for("a", 2).random()


def test_grid_is_geometric_and_ends_at_the_cap():
    assert ps.grid(30) == [10, 12, 15, 18, 22, 27, 30]
    assert ps.grid(5) == [5]


def test_size_component_finds_the_flawless_wilson_floor():
    """A rule every site passes: the Wilson lower bound of n/n reaches 90% from n = 35, so
    the search lands on the first grid size at or above it (10, 12, …, 33, 40)."""
    def verdicts(n: int, rng) -> dict[str, str]:
        ones = np.ones(n)
        return {"floor": ratio_verdict(range(n), ones, ones, 0.90, LOWER).verdict}

    found = ps.size_component(ps.Component("flawless", 60, verdicts), repetitions=20, screen=5)
    assert found == {"size": 40, "power": 1.0, "reachable": True}
    assert ps.size_component(ps.Component("short", 30, verdicts), repetitions=20, screen=5) == \
        {"size": 30, "power": 0.0, "reachable": False}


def test_evaluate_reports_each_condition_and_the_joint_pass():
    def verdicts(n: int, rng) -> dict[str, str]:
        return {"always": PASS, "coin": PASS if rng.random() < 0.5 else FAIL}

    result = ps.evaluate(ps.Component("coin", 1, verdicts), 1, 2_000)
    assert result["conditions"]["always"][PASS] == 1.0
    assert result["pass"] == pytest.approx(0.5, abs=0.05)


def test_the_simulation_needs_no_torch():
    code = "import sys; import training.power_simulation; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)

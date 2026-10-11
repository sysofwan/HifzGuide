"""Reciter-clustered power simulation: the reusable machinery of acceptance rules §8 (#126).

Lifted from #105 (branch ``agent/issue-105`` at ``3726ef4``) without its Ship 1 gate, its
sittings or its app model: what is left is torch-free and knows nothing about any one gate.
:mod:`training.ship_a_sizing` uses it to size Ship A's listening (ADR-0012); a later gate
(#61, #95) can reuse it as is.

Strata and draws
----------------
A :class:`Stratum` holds a population (its sites per canonical reciter) and the per-reciter
sums of :data:`tadabur.power_frame.FIELDS` its rates are read from: :meth:`Stratum.rates`
at face value, :meth:`Stratum.bootstrap_rates` redrawn from a reciter bootstrap of those
sums (baseline uncertainty). :func:`load_frame` reads them from Ship A's frame.
:func:`allocate` sizes strata in proportion to shares with caps, a :class:`Design` is the
union of several components' sizes (they nest by rank), and :func:`draw` simulates one
draw of a design: each stratum's sites spread over its reciters by a multivariate
hypergeometric, each site with its scorer weight and its rank in the stratum.

Outcomes
--------
ICC is the **latent** intra-reciter correlation of a logistic random effect,
σ² = ICC·(π²/3)/(1 − ICC) (:func:`sigma_for`), not the observed binary correlation.
:func:`clustered_bernoulli` draws one outcome per site with a set marginal and its
reciter's effect; :func:`intercept_for` finds the intercept that keeps the marginal
(Gauss-Hermite quadrature). :func:`paired` draws a second arm on the same sites from the
first, at an exact share of sites that differ, keeping both marginals
(:func:`exact_redraw_share`).

Verdicts
--------
Every verdict is the acceptance rules' own (:mod:`training.acceptance_stats`).
:func:`shared_resample` gives the §1 bootstrap draw for a set of sites, cached by its
number of reciters, so several rules on one repetition share one draw.
:func:`one_per_reciter` keeps one site per reciter, so a cell's observations are
independent and equally weighted and the Wilson and Tango bounds apply where it is sparse.
:func:`point_verdict`, :func:`paired_unit_verdict` and the exact one-sided
:func:`sign_test_harm` wrap the rest.

Cluster samples
---------------
:func:`pps_inclusion` draws reciters with probability proportional to the sites they hold
(capped at 1); every site of a drawn reciter is adjudicated. :func:`cluster_mistake_bound`
is a design-valid one-sided 97.5% bound on the mistakes among all the sites from such a
sample (exact with no mistake heard, Chernoff otherwise), and
:func:`expected_cluster_sites` the sites the sample is expected to adjudicate.

Power search
------------
:func:`rng_for` seeds every repetition stream from §8's 20261008 and a digest of its key.
A :class:`Component` is one sized part of a listening plan: given a size and a generator
it simulates one repetition and returns each condition's verdict. :func:`evaluate` runs
repetitions and returns the per-condition verdict shares and the joint pass probability;
:func:`size_component` finds the smallest size on a geometric :func:`grid` at which the
conditions pass jointly with the target probability, screened at
:data:`SCREEN_REPETITIONS` and confirmed at :data:`REPETITIONS`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from tadabur.power_frame import FIELDS, FRAME_PATH
from training.acceptance_stats import (
    CANNOT_CERTIFY,
    FAIL,
    LOWER,
    PASS,
    SEED,
    Resample,
    aggregate,
    difference_verdict,
)

# --- pre-registered (§8) ---------------------------------------------------------------------
REPETITIONS = 2_500
#: The owner's power target for sizing the listening (§8).
TARGET_POWER = 0.80
RECITER_ICCS = (0.0, 0.05, 0.15)

# --- the search ------------------------------------------------------------------------------
SCREEN_REPETITIONS = 400
GRID_START = 10
GRID_RATIO = 1.2

_FIELD = {name: i for i, name in enumerate(FIELDS)}
_NODES, _NODE_WEIGHTS = hermegauss(48)
_NODE_WEIGHTS = _NODE_WEIGHTS / _NODE_WEIGHTS.sum()


# --- strata ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stratum:
    """One stratum of the frame: its population and the per-reciter sums its rates come from.

    ``reciters`` are canonical reciter ids, ``sites`` how many of the stratum's sites each
    holds, ``rates_from`` each reciter's sums of :data:`FIELDS`, and ``seconds_per_site``
    the stratum's mean listening time under #87's model."""

    name: str
    mode: str
    family: str
    reciters: np.ndarray
    sites: np.ndarray
    rates_from: np.ndarray
    seconds_per_site: float

    @property
    def population(self) -> int:
        return int(self.sites.sum())

    def field(self, name: str) -> np.ndarray:
        """Each reciter's sum of one of :data:`FIELDS`."""
        return self.rates_from[:, _FIELD[name]]

    def rates(self) -> np.ndarray:
        """Every field as a share of the stratum's sites."""
        totals = self.rates_from.sum(axis=0)
        return totals / totals[_FIELD["sites"]]

    def bootstrap_rates(self, rng: np.random.Generator) -> np.ndarray:
        """:meth:`rates` over one reciter bootstrap of the stratum's reciters."""
        k = len(self.rates_from)
        totals = rng.multinomial(k, np.full(k, 1.0 / k)) @ self.rates_from
        return totals / totals[_FIELD["sites"]]


def load_frame(path: Path = FRAME_PATH) -> dict[str, Stratum]:
    """Ship A's frame (:mod:`tadabur.power_frame`) as strata, checked against :data:`FIELDS`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if tuple(data["fields"]) != FIELDS:
        raise ValueError(f"{path}: fields {data['fields']} are not {FIELDS}")
    strata = {}
    for name, entry in data["strata"].items():
        reciters = np.array(sorted(int(r) for r in entry["per_reciter"]), dtype=np.int64)
        stats = np.array([entry["per_reciter"][str(r)] for r in reciters], dtype=float)
        sites = stats[:, _FIELD["sites"]].astype(np.int64)
        if int(sites.sum()) != entry["population"]:
            raise ValueError(f"{path}: {name} per-reciter sites do not sum to its population")
        seconds = float(stats[:, _FIELD["listen_seconds"]].sum() / sites.sum())
        strata[name] = Stratum(name, entry["mode"], entry["family"], reciters, sites, stats,
                               seconds)
    return strata


# --- designs ---------------------------------------------------------------------------------


def _spread(total: int, shares: np.ndarray) -> np.ndarray:
    """``total`` split in proportion to ``shares`` by largest remainder."""
    if total == 0 or shares.sum() == 0:
        return np.zeros(len(shares), dtype=np.int64)
    exact = total * shares / shares.sum()
    counts = np.floor(exact).astype(np.int64)
    counts[np.argsort(-(exact - counts), kind="stable")[: total - counts.sum()]] += 1
    return counts


def allocate(
    shares: Mapping[str, float], caps: Mapping[str, int], total: int, minimum: int = 0
) -> dict[str, int]:
    """Integer sizes in proportion to ``shares``, summing to ``total`` (or every cap): a stratum
    whose share exceeds its population is taken whole and its excess spread over the rest.
    Each stratum then gets at least ``minimum`` (its population if smaller)."""
    free = {s for s, share in shares.items() if share > 0 and caps[s] > 0}
    sizes = {s: 0.0 for s in shares}
    remaining = min(total, sum(caps[s] for s in free))
    while free:
        weight = sum(shares[s] for s in free)
        capped = {s for s in free if remaining * shares[s] / weight >= caps[s]}
        if not capped:
            sizes.update({s: remaining * shares[s] / weight for s in free})
            break
        for s in capped:
            sizes[s] = caps[s]
            remaining -= caps[s]
        free -= capped
    whole = {s: int(math.floor(v)) for s, v in sizes.items()}
    short = round(sum(sizes.values())) - sum(whole.values())
    for s in sorted(sizes, key=lambda s: (whole[s] - sizes[s], s))[:short]:
        whole[s] += 1
    sized = {s: max(n, min(minimum, caps[s])) for s, n in sorted(whole.items())}
    return {s: n for s, n in sized.items() if n}


def union(designs: Iterable[Mapping[str, int]]) -> dict[str, int]:
    """Each stratum's largest size: one draw serving every component (they nest by rank)."""
    merged: dict[str, int] = {}
    for design in designs:
        for s, n in design.items():
            merged[s] = max(merged.get(s, 0), n)
    return dict(sorted((s, n) for s, n in merged.items() if n))


def design_minutes(strata: Mapping[str, Stratum], sizes: Mapping[str, int]) -> float:
    """Listening minutes under #87's model (each site's excerpt twice, plus 4 s to answer)."""
    return sum(n * strata[s].seconds_per_site for s, n in sizes.items()) / 60


@dataclass(frozen=True)
class Design:
    """A draw: per component, its sizes per stratum, the top of each stratum's rank order.
    The draw is their union (:attr:`sizes`)."""

    components: Mapping[str, Mapping[str, int]]

    @property
    def sizes(self) -> dict[str, int]:
        return union(self.components.values())

    def minutes(self, strata: Mapping[str, Stratum]) -> float:
        return design_minutes(strata, self.sizes)


@dataclass(frozen=True)
class Sample:
    """One draw of a design: per site, its stratum (an index into ``strata``), canonical
    reciter, scorer weight (stratum population / sites drawn) and rank in its stratum."""

    strata: tuple[Stratum, ...]
    stratum: np.ndarray
    reciter: np.ndarray
    weight: np.ndarray
    rank: np.ndarray

    def component(self, design: Design, name: str) -> np.ndarray:
        """The sites a component reads: the top ``sizes[stratum]`` of each stratum's ranks."""
        sizes = design.components.get(name, {})
        limit = np.array([sizes.get(s.name, 0) for s in self.strata] or [0], dtype=np.int64)
        return self.rank < limit[self.stratum]


def draw(strata: Mapping[str, Stratum], design: Design, rng: np.random.Generator) -> Sample:
    """One draw of ``design``: each stratum's sites spread over its reciters."""
    sizes = design.sizes
    chosen = tuple(strata[name] for name in sizes)
    index, reciters, weights, ranks = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)], \
        [np.zeros(0)], [np.zeros(0, np.int64)]
    for position, (name, n) in enumerate(sizes.items()):
        stratum = strata[name]
        sites = np.repeat(stratum.reciters, rng.multivariate_hypergeometric(stratum.sites, n))
        index.append(np.full(n, position))
        reciters.append(rng.permutation(sites))
        weights.append(np.full(n, stratum.population / n))
        ranks.append(np.arange(n))
    return Sample(chosen, *map(np.concatenate, (index, reciters, weights, ranks)))


# --- outcomes --------------------------------------------------------------------------------


def sigma_for(icc: float) -> float:
    """The SD of a logit-scale reciter effect whose **latent** correlation is ``icc``."""
    return math.sqrt(icc / (1 - icc) * math.pi**2 / 3)


def _expit(x) -> np.ndarray:
    return 0.5 * (1 + np.tanh(np.asarray(x, dtype=float) / 2))


def intercept_for(p: np.ndarray, sigma: float) -> np.ndarray:
    """The logit intercept b with E[expit(b + σZ)] = p, Z standard normal (quadrature)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        if sigma == 0:
            return np.log(p) - np.log1p(-p)
        low, high = np.full(p.shape, -40.0), np.full(p.shape, 40.0)
        for _ in range(60):
            mid = (low + high) / 2
            below = _expit(mid[..., None] + sigma * _NODES) @ _NODE_WEIGHTS < p
            low, high = np.where(below, mid, low), np.where(below, high, mid)
        return np.where(p <= 0, -np.inf, np.where(p >= 1, np.inf, (low + high) / 2))


def clustered_bernoulli(
    p, reciters: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """One outcome per site with marginal ``p`` and its own reciter effect N(0, σ²)."""
    p = np.broadcast_to(np.asarray(p, dtype=float), reciters.shape)
    values, inverse = np.unique(p, return_inverse=True)
    ids, by_reciter = np.unique(reciters, return_inverse=True)
    effect = sigma * rng.standard_normal(len(ids))[by_reciter]
    return rng.random(len(reciters)) < _expit(intercept_for(values, sigma)[inverse] + effect)


def categorical(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw per row of ``probabilities`` (rows summing to 1)."""
    if not len(probabilities):
        return np.zeros(0, dtype=np.int64)
    cumulative = np.cumsum(probabilities, axis=1)
    picked = (rng.random(len(probabilities))[:, None] > cumulative).sum(axis=1)
    return np.minimum(picked, probabilities.shape[1] - 1)


def exact_redraw_share(probabilities: np.ndarray, discordance: float) -> float:
    """The redraw probability giving P(arms differ) = ``discordance`` exactly, d / (1 − Σp²);
    the joint (1 − s)·diag(p) + s·p pᵀ keeps both marginals p. Above 1 is impossible."""
    if not discordance:
        return 0.0
    spread = 1 - float(np.sum(np.asarray(probabilities) ** 2))
    return discordance / spread if spread > 0 else math.inf


def paired(control: np.ndarray, rows: np.ndarray, discordance: float, rng) -> np.ndarray:
    """The second arm: ``control`` redrawn per site from its row at the exact-discordance share."""
    if not len(rows):
        return control
    shares = np.array([exact_redraw_share(r, discordance) for r in rows])
    return np.where(rng.random(len(control)) < shares, categorical(rows, rng), control)


# --- verdicts --------------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _canonical_resample(k: int) -> Resample:
    """by_cluster's draw for K clusters, counts held as floats (exact: small integers)."""
    base = Resample.by_cluster(range(k))
    return Resample(base.clusters, base.index, base.counts.astype(float))


def shared_resample(reciters: np.ndarray) -> tuple[np.ndarray, Resample]:
    """The §1 bootstrap draw for these sites, clusters relabelled 0..K−1 in sorted order:
    :meth:`Resample.by_cluster` draws from the count of sorted distinct ids alone, so this is
    exactly its draw for ``reciters`` (pinned by a test), cached by K."""
    ids, labels = np.unique(reciters, return_inverse=True)
    base = _canonical_resample(len(ids))
    return labels, Resample(base.clusters, labels, base.counts)


def one_per_reciter(
    mask: np.ndarray, reciters: np.ndarray, rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """One site per reciter among ``mask``'s sites, as sorted indices: uniformly chosen, or
    with chance proportional to ``weights`` (exponential keys, the least per reciter), so a
    reciter's total weight times the chosen site's outcome is unbiased for its weighted sum."""
    candidates = np.flatnonzero(mask)
    if weights is None:
        candidates = rng.permutation(candidates)
    else:
        with np.errstate(divide="ignore"):  # a zero weight is never chosen (key +inf)
            keys = rng.exponential(size=len(candidates)) / np.asarray(weights)[candidates]
        candidates = candidates[np.argsort(keys, kind="stable")]
    _, first = np.unique(reciters[candidates], return_index=True)
    return np.sort(candidates[first])


def point_verdict(value: float, threshold: float, kind: str) -> str:
    """A point estimate against a threshold; an undefined one is ``cannot_certify``."""
    if not math.isfinite(value):
        return CANNOT_CERTIFY
    return PASS if (value >= threshold if kind == LOWER else value <= threshold) else FAIL


def paired_unit_verdict(
    arm: np.ndarray, comparator: np.ndarray, threshold: float, kind: str
) -> str:
    """A paired difference on independent unit-weight observations (one per reciter): the
    bootstrap, or Tango where sparse or degenerate (§1)."""
    n = len(arm)
    if not n:
        return CANNOT_CERTIFY
    labels, resample = shared_resample(np.arange(n))
    ones = np.ones(n)
    return difference_verdict(labels, (arm.astype(float), ones), (comparator.astype(float), ones),
                              threshold, kind, ones, resample).verdict


def sign_test_harm(worse: int, better: int, alpha: float) -> bool:
    """One-sided exact sign test that an arm is worse: P(Bin(n, ½) ≥ worse) < alpha."""
    n = worse + better
    return bool(n) and sum(math.comb(n, k) for k in range(worse, n + 1)) / 2**n < alpha


# --- cluster samples -------------------------------------------------------------------------


def pps_inclusion(sizes: np.ndarray, n: int) -> np.ndarray:
    """Inclusion probabilities proportional to ``sizes`` summing to min(n, holders), each capped
    at 1 (a capped unit is taken with certainty and the rest re-scaled)."""
    sizes = np.asarray(sizes, dtype=float)
    pi = np.zeros_like(sizes)
    free = sizes > 0
    target = float(min(n, np.count_nonzero(free)))
    while target > 0 and free.any():
        trial = target * sizes / sizes[free].sum()
        capped = free & (trial >= 1)
        if not capped.any():
            pi[free] = trial[free]
            break
        pi[capped] = 1.0
        target -= capped.sum()
        free &= ~capped
    return pi


#: §1's one-sided level: a bound is the 97.5% end.
_TAIL = 0.025


def cluster_mistake_bound(
    pi: np.ndarray, holding: np.ndarray, mistakes: np.ndarray, sampled: np.ndarray
) -> float:
    """A one-sided 97.5% bound on the mistakes among every site held, from a cluster sample.

    Every site of a sampled reciter is adjudicated, so a reciter with any mistake shows it
    whenever it is sampled. Reciters taken with certainty (π_r = 1) are a census: their
    mistakes count as heard. For the rest, an allocation that loads mistakes on a set S of
    reciters makes K (the sampled ones in S, those heard to hold a mistake) a sum of
    independent Bernoulli(π_r). With no such reciter heard, P(K = 0) = Π_S (1 − π_r) must stay
    ≥ 2.5%: each loaded reciter costs −ln(1 − π_r) of a −ln 0.025 budget. With k heard, P(K ≤ k)
    ≥ 2.5% needs Σ_S π_r ≤ :func:`poisson_binomial_mean_limit` (the Chernoff lower tail): each
    costs π_r. The bound is the census's mistakes plus the most sites S can hold, every one a
    mistake, under that budget: a fractional knapsack (largest M_r per unit cost first), which
    bounds the exact 0/1 problem from above."""
    certain = pi >= 1
    open_ = (pi > 0) & ~certain
    heard = int(np.count_nonzero(sampled & open_ & (mistakes > 0)))
    if heard:
        budget, cost = poisson_binomial_mean_limit(heard), pi[open_]
    else:
        budget, cost = -math.log(_TAIL), -np.log1p(-pi[open_])
    value = holding[open_].astype(float)
    total = float(mistakes[certain].sum())
    for r in np.argsort(-(value / cost), kind="stable"):
        take = min(1.0, budget / cost[r])
        total += take * value[r]
        budget -= take * cost[r]
        if budget <= 0:
            break
    return total


def poisson_binomial_mean_limit(heard: int) -> float:
    """The largest mean μ of a sum K of independent Bernoullis at which P(K ≤ heard) can still
    be ≥ 2.5%: by the Chernoff lower tail, P(K ≤ k) ≤ e^{−μ}(eμ/k)^k for μ > k."""
    target, low, high = math.log(_TAIL), float(heard), float(heard) + 100.0
    for _ in range(100):
        mid = (low + high) / 2
        if -mid + heard + heard * math.log(mid / heard) >= target:
            low = mid
        else:
            high = mid
    return high


def expected_cluster_sites(holding: np.ndarray, n: int) -> float:
    """The sites a cluster sample of ``n`` expected reciters adjudicates: Σ π_r·M_r."""
    holding = np.asarray(holding, dtype=float)
    return float(pps_inclusion(holding, n) @ holding)


# --- power -----------------------------------------------------------------------------------


def rng_for(*key) -> np.random.Generator:
    """A generator seeded from §8's 20261008 and a stable digest of ``key``."""
    digest = hashlib.sha256(
        json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).digest()
    return np.random.default_rng([SEED, int.from_bytes(digest[:8], "big")])


@dataclass(frozen=True)
class Component:
    """One sized part of a listening plan. ``verdicts(size, rng)`` simulates one repetition
    at ``size`` and returns each condition's verdict; ``cap`` is the largest size there is."""

    name: str
    cap: int
    verdicts: Callable[[int, np.random.Generator], Mapping[str, str]]


def evaluate(component: Component, size: int, repetitions: int) -> dict:
    """Per condition, the share of repetitions in each verdict, and the pass probability: the
    share of repetitions in which every condition passes."""
    rng = rng_for(component.name, size, repetitions)
    counts: dict[str, Counter] = {}
    joint = 0
    for _ in range(repetitions):
        verdicts = component.verdicts(size, rng)
        for name, verdict in verdicts.items():
            counts.setdefault(name, Counter())[verdict] += 1
        joint += aggregate(list(verdicts.values())) == PASS
    return {
        "pass": joint / repetitions,
        "repetitions": repetitions,
        "conditions": {name: {v: c[v] / repetitions for v in (PASS, FAIL, CANNOT_CERTIFY)}
                       for name, c in sorted(counts.items())},
    }


def grid(cap: int) -> list[int]:
    """Sizes from :data:`GRID_START` up by :data:`GRID_RATIO`, ending at ``cap``."""
    totals, total = [], GRID_START
    while total < cap:
        totals.append(total)
        total = math.ceil(total * GRID_RATIO)
    return totals + [cap]


def size_component(
    component: Component,
    target: float = TARGET_POWER,
    repetitions: int = REPETITIONS,
    screen: int = SCREEN_REPETITIONS,
) -> dict:
    """The smallest grid size at which ``component``'s conditions pass jointly with probability
    ≥ ``target``: screened at ``screen`` repetitions (a doubling search, then bisection),
    confirmed at ``repetitions``; ``reachable: false`` with the power at ``cap`` when even
    that falls short."""
    totals = grid(component.cap)

    def power(i: int, reps: int) -> float:
        return evaluate(component, totals[i], reps)["pass"]

    low, i, step = -1, 0, 1
    while i < len(totals) - 1 and power(i, screen) < target:
        low, i, step = i, min(i + step, len(totals) - 1), step * 2
    high = i
    while high - low > 1:
        mid = (low + high) // 2
        if power(mid, screen) >= target:
            high = mid
        else:
            low = mid
    for i in range(high, len(totals)):
        p = power(i, repetitions)
        if p >= target or i == len(totals) - 1:
            return {"size": totals[i], "power": p, "reachable": p >= target}
    raise AssertionError("the loop returns at the last grid size")

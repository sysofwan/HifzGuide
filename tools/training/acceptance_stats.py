"""The statistics and verdicts of ``docs/acceptance-rules.md``, frozen as code before any candidate.

Every gate from #84 onward (the probe gate #95, the bias probe #91, the ship criterion #97,
allowance retirement) reads its numbers through this module, so each rule is written once
and pinned by ``test_acceptance_stats.py``. It is torch-free: plain arrays in, numbers and
verdicts out. :mod:`training.truth_scorer` turns truth sites and decodes into the arrays.

Rates (§1)
----------
A rate is a **weighted ratio** of sums over observations: commit rate ΣwC / Σw, committed
accuracy ΣwA / ΣwC, and so on. Each observation carries its weighted numerator and
denominator (``num``, ``den``) and the **cluster** it belongs to (the canonical reciter id).
A ratio with a zero denominator is undefined (``nan``), never 0.

Intervals (§1)
--------------
:class:`Resample` is the reciter-clustered bootstrap: :data:`REPLICATES` = 10,000 draws of
the clusters with replacement, seeded with :data:`SEED` = 20261008. The draw is a function of
the **sorted** cluster ids alone, so every statistic computed on one observation set (both
arms, both protocols, every rate) shares one draw: that is what makes a difference **paired**.
A comparison of two ratios is the difference of the two ratios inside every replicate,
never a test on the observations both arms committed.

Bounds are type-7 percentiles (:func:`quantile_type7`, NumPy's default linear
interpolation): the lower bound is the 2.5th percentile, the upper the 97.5th. A replicate
whose statistic is undefined takes the **adverse endpoint**: −∞ in a lower bound, +∞ in an
upper bound. Replicates are never dropped.

Sparse cells and exact bounds (§1)
----------------------------------
A cell with fewer than :data:`SPARSE_MIN_RECITERS` reciters or :data:`SPARSE_MIN_SITES`
sites, or whose replicates are all identical (all zero included), is **sparse**. A sparse
cell may still be judged by a Wilson score bound (one proportion) or a Tango score bound
(a paired difference) **only** when its observations are independent and equally weighted:
one observation per reciter, unit weights, and a fixed denominator (every ``den`` is 1, in
both arms for a difference). Any other sparse or degenerate cell is ``cannot_certify``. A
zero denominator never passes: it is ``cannot_certify``.

Verdicts (§1)
-------------
Every rule returns :data:`PASS`, :data:`FAIL` or :data:`CANNOT_CERTIFY`, and
:func:`aggregate` folds a gate's conditions: any fail → fail; otherwise any
cannot_certify → cannot_certify; otherwise pass.

Teacher agreement (§2, §9)
--------------------------
``1 − Σ edit distance / Σ teacher tokens`` over clip records of the frozen
``decode_evalset`` dev manifest, decoded at b = 0 with no bias. Before the edit distance,
the student string is **projected** onto the teacher's vocabulary
(:func:`project_to_teacher`): the explicit sukun mark is removed and a geminate written with
the shaddah mark is expanded to the legacy doubled consonant. Clip records are resampled
by reciter: the one exception to the one-site observation unit.

Reciter split (§5, §9)
----------------------
:func:`reciter_half` assigns a canonical reciter id to the bias probe's ``tune`` or
``score`` half: SHA-256 of the UTF-8 string ``"<salt>:<reciter id in decimal>"``, the first
8 bytes of the digest read as a big-endian unsigned integer, ``tune`` below 2**63 and
``score`` at or above it. The salt is :data:`BIAS_SPLIT_SALT`.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from training.counterfactual_eval import paired_score_interval, wilson_interval
from training.decode_evalset import levenshtein
from training.tashkeel_eval import SUKUN_MARK

# --- frozen constants (§1) -------------------------------------------------------------
REPLICATES = 10_000
SEED = 20261008
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975
#: The normal quantile the Wilson and Tango bounds use: the same one-sided 97.5% end.
Z = 1.96
SPARSE_MIN_RECITERS = 10
SPARSE_MIN_SITES = 20

PASS = "pass"
FAIL = "fail"
CANNOT_CERTIFY = "cannot_certify"
VERDICTS = (PASS, FAIL, CANNOT_CERTIFY)

LOWER = "lower"
UPPER = "upper"

#: The salt of the bias probe's reciter split (§5).
BIAS_SPLIT_SALT = "issue-91-blank-bias-2026-10"
TUNE = "tune"
SCORE = "score"

#: The Unicode shaddah. No decode carries it today; if a gemination class is ever written
#: with it (#92), teacher agreement expands it to the legacy doubled consonant.
SHADDAH_MARK = "ّ"


def aggregate(verdicts: Sequence[str]) -> str:
    """The §1 truth table: any fail → fail; else any cannot_certify → cannot_certify; else pass.

    An empty gate has no evidence and is ``cannot_certify``.
    """
    unknown = set(verdicts) - set(VERDICTS)
    if unknown:
        raise ValueError(f"not verdicts: {sorted(unknown)}")
    if FAIL in verdicts:
        return FAIL
    if CANNOT_CERTIFY in verdicts or not verdicts:
        return CANNOT_CERTIFY
    return PASS


def quantile_type7(values: np.ndarray, q: float) -> float:
    """Hyndman-Fan type 7 (NumPy's default linear interpolation), exact at infinite ends.

    Interpolating between −∞ and a finite neighbour is −∞ (and likewise +∞), where NumPy
    would return ``nan``: an adverse endpoint inside the interpolation window must reach the
    bound rather than erase it.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0 or np.isnan(ordered).any():
        raise ValueError("quantile of an empty sample or one holding nan")
    h = (ordered.size - 1) * q
    low = math.floor(h)
    high = min(low + 1, ordered.size - 1)
    fraction = h - low
    a, b = ordered[low], ordered[high]
    if fraction == 0 or a == b:
        return float(a)
    if math.isinf(a) or math.isinf(b):
        return float(a) if math.isinf(a) else float(b)
    return float(a + fraction * (b - a))


def lower_bound(replicates: np.ndarray) -> float:
    """The 2.5th percentile, an undefined replicate counting as −∞."""
    return quantile_type7(np.where(np.isnan(replicates), -np.inf, replicates), LOWER_QUANTILE)


def upper_bound(replicates: np.ndarray) -> float:
    """The 97.5th percentile, an undefined replicate counting as +∞."""
    return quantile_type7(np.where(np.isnan(replicates), np.inf, replicates), UPPER_QUANTILE)


def ratio(num: np.ndarray, den: np.ndarray) -> float:
    """Σnum / Σden, ``nan`` when the denominator is zero."""
    total = float(np.sum(den))
    return float(np.sum(num)) / total if total else math.nan


@dataclass(frozen=True)
class Resample:
    """One reciter-clustered bootstrap draw, shared by every statistic on an observation set.

    ``clusters`` are the sorted distinct cluster ids, ``index`` maps each observation to its
    cluster's position, and ``counts[r, k]`` is how often replicate ``r`` drew cluster ``k``.
    """

    clusters: tuple[int, ...]
    index: np.ndarray
    counts: np.ndarray

    @classmethod
    def by_cluster(
        cls, cluster_ids: Sequence[int], replicates: int = REPLICATES, seed: int = SEED
    ) -> "Resample":
        clusters = tuple(sorted(set(int(c) for c in cluster_ids)))
        position = {cluster: k for k, cluster in enumerate(clusters)}
        index = np.array([position[int(c)] for c in cluster_ids], dtype=np.int64)
        draws = np.random.default_rng(seed).integers(
            0, len(clusters), size=(replicates, len(clusters))
        )
        counts = np.zeros((replicates, len(clusters)), dtype=np.int64)
        np.add.at(counts, (np.arange(replicates)[:, None], draws), 1)
        return cls(clusters, index, counts)

    def totals(self, values: np.ndarray) -> np.ndarray:
        """Each replicate's sum of a per-observation quantity."""
        per_cluster = np.bincount(
            self.index, weights=np.asarray(values, dtype=float), minlength=len(self.clusters)
        )
        return self.counts @ per_cluster

    def ratios(self, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        """Each replicate's Σnum / Σden, ``nan`` where the denominator is zero."""
        top, bottom = self.totals(num), self.totals(den)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(bottom > 0, top / np.where(bottom > 0, bottom, 1.0), np.nan)


def is_sparse(num_sites: int, num_reciters: int) -> bool:
    """Fewer than 10 reciters or 20 sites (§1)."""
    return num_reciters < SPARSE_MIN_RECITERS or num_sites < SPARSE_MIN_SITES


def is_degenerate(replicates: np.ndarray) -> bool:
    """Replicates that are all identical (all zero included) carry no interval."""
    return bool(replicates.size) and bool(np.all(replicates == replicates[0]))


def independent_equal_weight(clusters: Sequence[int], *dens: np.ndarray, weights=None) -> bool:
    """One observation per reciter, unit weights and a fixed denominator in every arm.

    The only cells where a Wilson (one proportion) or Tango (paired difference) bound may
    stand in for the bootstrap. ``dens`` are the arms' per-observation denominators.
    """
    one_per_reciter = len(set(clusters)) == len(clusters)
    unit = weights is None or bool(np.all(np.asarray(weights) == 1))
    fixed = all(bool(np.all(np.asarray(den) == 1)) for den in dens)
    return one_per_reciter and unit and fixed


BOOTSTRAP = "bootstrap"
WILSON = "wilson"
TANGO = "tango"
#: No admissible bound: the statistic is undefined, or the cell is sparse or degenerate and
#: not independent and equally weighted. Any rule read from it is ``cannot_certify``.
NO_BOUND = "none"


@dataclass(frozen=True)
class Interval:
    """The §1 interval of one statistic and the method that produced it.

    A bootstrap bound may be infinite (the adverse endpoint of undefined replicates). With
    :data:`NO_BOUND` both ends are ``None`` and ``reason`` says why.
    """

    lower: float | None
    upper: float | None
    method: str
    reason: str = ""

    def bound(self, kind: str) -> float | None:
        return self.lower if kind == LOWER else self.upper

    def as_dict(self) -> dict:
        def end(value: float | None):
            if value is None or math.isfinite(value):
                return value
            return "-inf" if value < 0 else "inf"

        return {
            "lower": end(self.lower),
            "upper": end(self.upper),
            "method": self.method,
            "reason": self.reason,
        }


_NOT_INDEPENDENT = "sparse or degenerate, and not independent and equally weighted"


def _arrays(*pairs) -> list[tuple[np.ndarray, np.ndarray]]:
    return [tuple(np.asarray(x, dtype=float) for x in pair) for pair in pairs]


def ratio_interval(
    clusters: Sequence[int],
    num: np.ndarray,
    den: np.ndarray,
    weights: np.ndarray | None = None,
    resample: Resample | None = None,
) -> Interval:
    """A single weighted ratio's interval under §1's rules.

    The reciter-clustered bootstrap when the cell is neither sparse nor degenerate;
    otherwise the Wilson score interval if the observations are independent and equally
    weighted (``weights`` are the site weights behind ``num`` and ``den``); otherwise none.
    A zero denominator has none. ``resample`` lets several statistics share one draw.
    """
    ((num, den),) = _arrays((num, den))
    if not np.sum(den):
        return Interval(None, None, NO_BOUND, "zero denominator")
    replicates = (resample or Resample.by_cluster(clusters)).ratios(num, den)
    if not (is_sparse(len(clusters), len(set(clusters))) or is_degenerate(replicates)):
        return Interval(lower_bound(replicates), upper_bound(replicates), BOOTSTRAP)
    if not independent_equal_weight(clusters, den, weights=weights):
        return Interval(None, None, NO_BOUND, _NOT_INDEPENDENT)
    low, high = wilson_interval(int(round(np.sum(num))), len(clusters), Z)
    return Interval(low, high, WILSON, "sparse or degenerate")


def difference_interval(
    clusters: Sequence[int],
    arm: tuple[np.ndarray, np.ndarray],
    comparator: tuple[np.ndarray, np.ndarray],
    weights: np.ndarray | None = None,
    resample: Resample | None = None,
) -> Interval:
    """The paired difference arm − comparator of two ratios on one observation set (§1).

    Both ratios are recomputed inside every replicate of one shared draw; the observations
    are never restricted to the ones both arms committed. A sparse or degenerate cell takes
    the Tango score interval if independent and equally weighted with a common fixed
    denominator, otherwise none.
    """
    (num_a, den_a), (num_b, den_b) = _arrays(arm, comparator)
    if not np.sum(den_a) or not np.sum(den_b):
        return Interval(None, None, NO_BOUND, "zero denominator")
    resample = resample or Resample.by_cluster(clusters)
    replicates = resample.ratios(num_a, den_a) - resample.ratios(num_b, den_b)
    if not (is_sparse(len(clusters), len(set(clusters))) or is_degenerate(replicates)):
        return Interval(lower_bound(replicates), upper_bound(replicates), BOOTSTRAP)
    if not independent_equal_weight(clusters, den_a, den_b, weights=weights):
        return Interval(None, None, NO_BOUND, _NOT_INDEPENDENT)
    gains = int(np.sum((num_a == 1) & (num_b == 0)))
    losses = int(np.sum((num_a == 0) & (num_b == 1)))
    low, high = paired_score_interval(gains, losses, len(clusters), Z)
    return Interval(low, high, TANGO, "sparse or degenerate")


@dataclass(frozen=True)
class BoundVerdict:
    """One rule's outcome: the verdict, the bound it was read from and how it was obtained."""

    verdict: str
    bound: float | None
    method: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "bound": None if self.bound is None or not math.isfinite(self.bound) else self.bound,
            "method": self.method,
            "reason": self.reason,
        }


def _verdict(interval: Interval, threshold: float, kind: str) -> BoundVerdict:
    bound = interval.bound(kind)
    if interval.method == NO_BOUND:
        return BoundVerdict(CANNOT_CERTIFY, None, NO_BOUND, interval.reason)
    holds = bound >= threshold if kind == LOWER else bound <= threshold
    return BoundVerdict(PASS if holds else FAIL, bound, interval.method, interval.reason)


def ratio_verdict(
    clusters: Sequence[int],
    num: np.ndarray,
    den: np.ndarray,
    threshold: float,
    kind: str,
    weights: np.ndarray | None = None,
) -> BoundVerdict:
    """A single weighted ratio's ``kind`` bound against ``threshold`` (:func:`ratio_interval`)."""
    return _verdict(ratio_interval(clusters, num, den, weights), threshold, kind)


def difference_verdict(
    clusters: Sequence[int],
    arm: tuple[np.ndarray, np.ndarray],
    comparator: tuple[np.ndarray, np.ndarray],
    threshold: float,
    kind: str,
    weights: np.ndarray | None = None,
) -> BoundVerdict:
    """A paired difference's ``kind`` bound against ``threshold`` (:func:`difference_interval`)."""
    return _verdict(difference_interval(clusters, arm, comparator, weights), threshold, kind)


def relative_change_verdict(
    clusters: Sequence[int],
    arm: tuple[np.ndarray, np.ndarray],
    comparator: tuple[np.ndarray, np.ndarray],
    threshold: float,
    kind: str,
) -> BoundVerdict:
    """(arm − comparator) / comparator, e.g. the §3 false-flag headline.

    A comparator whose rate is zero leaves the change undefined: ``cannot_certify``. No exact
    bound exists for a relative change, so a sparse or degenerate cell is ``cannot_certify``.
    """
    (num_a, den_a), (num_b, den_b) = _arrays(arm, comparator)
    if not np.sum(den_a) or not np.sum(den_b) or not np.sum(num_b):
        return BoundVerdict(CANNOT_CERTIFY, None, NO_BOUND, "zero denominator or zero comparator rate")
    resample = Resample.by_cluster(clusters)
    base = resample.ratios(num_b, den_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        replicates = np.where(base > 0, (resample.ratios(num_a, den_a) - base) / base, np.nan)
    if is_sparse(len(clusters), len(set(clusters))) or is_degenerate(replicates):
        return BoundVerdict(CANNOT_CERTIFY, None, NO_BOUND, "sparse or degenerate")
    interval = Interval(lower_bound(replicates), upper_bound(replicates), BOOTSTRAP)
    return _verdict(interval, threshold, kind)


# --- teacher agreement (§2) ------------------------------------------------------------


def project_to_teacher(student: str) -> str:
    """The student decode in the teacher's vocabulary: sukun removed, geminates doubled.

    A cluster carrying the shaddah mark becomes the bare consonant followed by the consonant
    with its remaining marks, the legacy encoding (``بَّ`` → ``ببَ``).
    """
    projected = student.replace(SUKUN_MARK, "")
    clusters: list[str] = []
    for char in projected:
        if clusters and unicodedata.combining(char):
            clusters[-1] += char
        else:
            clusters.append(char)
    return "".join(
        cluster[0] + cluster.replace(SHADDAH_MARK, "") if SHADDAH_MARK in cluster else cluster
        for cluster in clusters
    )


@dataclass(frozen=True)
class AgreementRecord:
    """One clip of the dev manifest: its reciter, the cached teacher decode, a student's."""

    reciter_id: int
    teacher: str
    student: str


def agreement_terms(records: Sequence[AgreementRecord]) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Clusters and per-clip ``(teacher tokens − edits, teacher tokens)``: agreement is their ratio.

    Feed them to :func:`ratio_verdict` or, for probe − control, :func:`difference_verdict`
    with the two arms' terms over the same records.
    """
    clusters = [record.reciter_id for record in records]
    tokens = np.array([len(record.teacher) for record in records], dtype=float)
    edits = np.array(
        [levenshtein(record.teacher, project_to_teacher(record.student)) for record in records],
        dtype=float,
    )
    return clusters, tokens - edits, tokens


# --- reciter split (§5) ----------------------------------------------------------------


def reciter_half(reciter_id: int, salt: str = BIAS_SPLIT_SALT) -> str:
    """``tune`` or ``score`` for one canonical reciter id (module docstring for the rule)."""
    if isinstance(reciter_id, bool) or not isinstance(reciter_id, int) or reciter_id < 0:
        raise ValueError(f"a canonical reciter id is a non-negative int, got {reciter_id!r}")
    digest = hashlib.sha256(f"{salt}:{reciter_id}".encode("utf-8")).digest()
    return TUNE if int.from_bytes(digest[:8], "big") < 2**63 else SCORE

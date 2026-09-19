"""A gate-evaluation set on which ">95% agreement" is a statement about the model.

``training.distill_gate`` scores student-vs-teacher gate decisions on whatever clips it is
pointed at. Pointed at a random sample it measures almost nothing: on the 200 random clips
used throughout ADR-0010, **88.0% of them pass the gate**, so a student that rubber-stamped
every clip scores 88.0% and the measured 91.5% is 3.5 points of daylight at n=200 --
McNemar p ~ 0.23. Nothing done to the model can move a number that saturated.

This module builds the set that makes the number move, and it builds **two views of one
sample** because neither alone supports a ship decision:

* a **population** sample -- a uniform draw over eligible clips, which answers "what would
  a user see"; and
* a **boundary** sample -- enriched near the ``.balanced`` bar (``correct_threshold`` 0.65),
  which answers "is this student better than a rubber stamp", because that is where a
  student can plausibly flip a decision and where the flip costs something.

Optimising the population number is hopeless (it saturates); shipping on the boundary number
alone is dishonest (it is deliberately unrepresentative). Both are reported, always.

**Clips are stored as 32-bit float.** The manifest caches the teacher's decode of the
waveform in memory and the student reads the file back, so anything lossy between them is a
difference charged entirely to the student. A PCM_16 round trip is lossy enough to matter
here: measured on 60 clips it changed the teacher's decoded string on 12 and moved
``match_ratio`` by up to 0.062, and Tadabur audio peaks above 1.0 so it clips as well.

**The teacher is run once and frozen.** Every gate evaluation so far decoded *both* models,
paying the teacher's 42 ms/window again for every student measured. The teacher is
deterministic and the protocol is fixed, so its decode is a property of the clip: it is
computed here and written into the manifest, and :mod:`training.distill_gate` then scores a
student against cached decisions. The manifest records the teacher id, the confirmation
split and the scorer bar it was built under, and :func:`check_provenance` refuses a manifest
whose rules no longer match the code loading it -- a cache that outlives the thing it caches
is worse than no cache.

**Provenance: strided shards, never contiguous.** Candidates come from Tadabur shards that
no training run may touch. Shards 0-19 already produced the staged ``clips_v2`` corpus, so
one eval set drawn from a block at the *top* of the range would be valid -- but a contiguous
block is a poor population sample if shard order groups reciters or recording sources, which
nothing guarantees it does not. :data:`GATE_EVAL_SHARDS` therefore takes every 24th shard
across the whole training range, and :func:`training_shard_spec` returns the complement for
``--stream-shards``. The cost is 15 shards (~4%) of training data for a sample that spans
the corpus.

**A dev/test boundary inside the set.** Tuning against the whole set eventually makes all of
it development data. Clips are split by **reciter** (not by clip -- the same reciter's
recordings share channel and style), so a finalist can be scored on clips no intermediate
decision was ever made against.

**The candidates are unfiltered.** Tadabur's staging filter never saw these rows. That is
deliberate: gate agreement is a teacher-vs-student question, and the clips the filter would
have rejected -- the poor, the partial, the mis-assigned -- are exactly the ambiguous ones
that land near the bar. It does mean this set measures agreement on a *broader* input
distribution than ``clips_v2``, so the two are not comparable and the manifest says which
one it is.

Usage::

    # Build the frozen set (downloads and deletes one shard at a time)
    python -m training.gate_evalset --out-dir ../tadabur/gate_eval

    # Report what a built set contains, without a GPU
    python -m training.gate_evalset --out-dir ../tadabur/gate_eval --describe

    # The shard spec a training run must use so it never sees these clips
    python -m training.gate_evalset --print-training-shards

Linux + CUDA for the build (the teacher must be resident); loading, the statistics helpers
and the shard arithmetic are torch-free so they can be unit-tested anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

# Bumped whenever the manifest layout, the strata, or anything the cached teacher decisions
# depend on changes, so a stale set is refused rather than silently reused.
SCHEMA_VERSION = "gate-evalset-v1"

MANIFEST_FILENAME = "manifest.json"
CLIPS_DIRNAME = "clips"

# Every 19th shard from 20: 20 shards spread across the corpus rather than a block at one
# end (see the module docstring on why contiguous is the wrong choice). The stride is set by
# what the boundary view needs, not by taste: measured on two shards, only ~4% of raw Tadabur
# clips fall below the 0.65 bar under the teacher, so filling a 500-clip failing quota takes
# ~13,000 candidates. 20 shards yield ~20,000, which leaves margin; 15 would not have.
GATE_EVAL_SHARD_STRIDE = 19
GATE_EVAL_SHARD_START = 20
NUM_TADABUR_SHARDS = 385

# Half-width of the band counted as "near the bar". 0.20 around 0.65 spans [0.45, 0.85]:
# wide enough that a real corpus populates it, narrow enough that a clip inside it is one a
# plausible student could flip.
NEAR_BAND_HALF_WIDTH = 0.20

# Share of the BOUNDARY sample each stratum gets. Near-bar clips dominate because they carry
# the information; the clear tails are kept so a student that breaks the easy cases is still
# caught. Chosen so the teacher's pass rate on the boundary sample lands near 50%, which is
# what turns ``always_pass_agreement`` into an informative floor instead of the headline.
BOUNDARY_QUOTA_SHARE: dict[str, float] = {
    "fail_clear": 0.20,
    "near_fail": 0.30,
    "near_pass": 0.30,
    "pass_clear": 0.20,
}
STRATA = tuple(BOUNDARY_QUOTA_SHARE)

# n=1000 is not a free choice. At an observed 95.0%, the 95% Wilson interval is roughly
# [93.5%, 96.2%] -- so 1000 clips cannot *establish* ">95%", only fail to rule it out. 2000
# narrows that to about [94.0%, 95.8%]. The target is sized for the population sample, which
# is the one a ship claim rests on; the boundary sample is a comparison instrument and is
# smaller.
DEFAULT_POPULATION_TARGET = 2000
DEFAULT_BOUNDARY_TARGET = 1000

# Matches ``tadabur.filter.MAX_AYAH_DURATION_S``: the staging filter drops clips longer than
# this, so including them would measure the student on audio the product pipeline never
# reaches -- and each one costs ~46 teacher window passes.
MAX_CLIP_SECONDS = 50.0
# Below this there is not enough audio for one confirmed window to mean anything.
MIN_CLIP_SECONDS = 1.5

# Above this many discordant pairs the exact sum is replaced by the normal approximation:
# the exact form costs O(n^2) big-integer work and the approximation is indistinguishable at
# these counts, while ``2.0 ** trials`` would simply overflow past 1023.
EXACT_BINOMIAL_LIMIT = 1000

DEFAULT_SEED = 20260918
# Stable across rebuilds so "the test half" means the same reciters in every set.
SPLIT_SALT = "gate-evalset-reciter-split-v1"


def gate_eval_shards() -> list[int]:
    """The shards reserved for evaluation -- strided, never trained on."""
    return list(
        range(GATE_EVAL_SHARD_START, NUM_TADABUR_SHARDS, GATE_EVAL_SHARD_STRIDE)
    )


def training_shard_spec(
    held_out_below: int = GATE_EVAL_SHARD_START,
    num_shards: int = NUM_TADABUR_SHARDS,
    reserved_shards: list[int] | None = None,
) -> str:
    """The ``--stream-shards`` spec that excludes both held-out blocks.

    Shards below ``held_out_below`` produced the staged corpus; the evaluation shards are
    removed on top of that. Returned as the compact range spec ``training.distill_stream``
    already parses, so a run cannot be configured with a hand-typed range that clips the
    reservation.

    ``reserved_shards`` must be the shards a set was **actually built on**, which is not
    always :func:`gate_eval_shards`: ``--shards`` can override it. Defaulting to the
    canonical block while a set was built on some other shard is how the advertised leakage
    guard quietly stops guarding, so the CLI reads this from the built manifest.
    """
    reserved = set(gate_eval_shards() if reserved_shards is None else reserved_shards)
    available = [i for i in range(held_out_below, num_shards) if i not in reserved]
    if not available:
        raise ValueError("no training shards left after the reservation")

    terms: list[str] = []
    start = previous = available[0]
    for index in available[1:]:
        if index == previous + 1:
            previous = index
            continue
        terms.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    terms.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(terms)


def stratum_for(match_ratio: float, threshold: float) -> str:
    """Which stratum a clip belongs to, from the **teacher's** match_ratio.

    The split is around the shipped bar, not around the middle of [0, 1]: "near" has to mean
    "a student could plausibly flip this", which is a statement about distance from 0.65.
    """
    if match_ratio < threshold - NEAR_BAND_HALF_WIDTH:
        return "fail_clear"
    if match_ratio < threshold:
        return "near_fail"
    if match_ratio < threshold + NEAR_BAND_HALF_WIDTH:
        return "near_pass"
    return "pass_clear"


def boundary_quotas(target: int) -> dict[str, int]:
    """Per-stratum caps for the boundary sample, summing to ``target``."""
    quotas = {name: int(target * share) for name, share in BOUNDARY_QUOTA_SHARE.items()}
    # Truncation can leave several clips unassigned, not just one per stratum. They go to
    # the near-bar strata, which is where an extra clip buys the most resolution.
    near = ("near_fail", "near_pass")
    for offset in range(target - sum(quotas.values())):
        quotas[near[offset % len(near)]] += 1
    return quotas


def reciter_split(reciter_id: int) -> str:
    """``"dev"`` or ``"test"`` for one reciter, stably and without a stored table.

    Split by reciter rather than by clip: one reciter's recordings share channel, pace and
    style, so a clip-level split puts near-duplicates on both sides and a "held-out" test
    number quietly measures the dev set again.
    """
    digest = hashlib.sha256(f"{SPLIT_SALT}:{reciter_id}".encode("utf-8")).digest()
    return "test" if digest[0] % 2 else "dev"


@dataclass(frozen=True)
class EvalClip:
    """One frozen evaluation clip and the teacher decision cached for it."""

    filename: str
    surah_ayah: str
    reciter_id: int
    shard: int
    duration_s: float
    stratum: str
    split: str
    in_population: bool
    in_boundary: bool
    teacher_text: str
    teacher_ratio: float
    teacher_passed: bool
    # Recorded but unused by any distillation metric: they are the inputs to the Tadabur
    # corpus filter's poison rejects, which belong to the ADR-0001 fine-tune track. Kept in
    # the manifest so a filter-side question can be asked of this set later without a rebuild.
    teacher_insertion_run: int
    teacher_added_shadda: bool

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "surah_ayah": self.surah_ayah,
            "reciter_id": self.reciter_id,
            "shard": self.shard,
            "duration_s": round(self.duration_s, 2),
            "stratum": self.stratum,
            "split": self.split,
            "in_population": self.in_population,
            "in_boundary": self.in_boundary,
            "teacher_text": self.teacher_text,
            "teacher_ratio": round(self.teacher_ratio, 6),
            "teacher_passed": self.teacher_passed,
            "teacher_insertion_run": self.teacher_insertion_run,
            "teacher_added_shadda": self.teacher_added_shadda,
        }


@dataclass(frozen=True)
class EvalSet:
    """A built evaluation set: its clips, its provenance, and what the scan saw.

    ``scanned_by_stratum`` is the field that is easy to omit and impossible to reconstruct
    later. Without it the boundary sample is a bag of clips with no route back to the
    population it came from, and every number computed on it silently reads as a population
    number while being an enriched one.
    """

    schema_version: str
    clips: tuple[EvalClip, ...]
    scanned_by_stratum: dict[str, int]
    num_scanned: int
    num_skipped: int
    provenance: dict

    def fingerprint(self) -> str:
        """A stable id for *this* evaluation: its clips, their cached teacher truth, its rules.

        Two sets built from the same shards under different protocols share filenames, so
        comparing a saved decisions file by name alone can silently score one checkpoint
        against another's truth. Hashing the clip list with the teacher's decodes and the
        provenance makes that mismatch detectable.
        """
        payload = json.dumps(
            {
                "schema": self.schema_version,
                "provenance": self.provenance,
                "clips": [
                    [clip.filename, clip.teacher_text, round(clip.teacher_ratio, 6)]
                    for clip in self.clips
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def subset(self, view: str, split: str | None = None) -> tuple[EvalClip, ...]:
        """The clips in one view (``"population"`` / ``"boundary"`` / ``"all"``)."""
        if view == "population":
            chosen = [clip for clip in self.clips if clip.in_population]
        elif view == "boundary":
            chosen = [clip for clip in self.clips if clip.in_boundary]
        elif view == "all":
            chosen = list(self.clips)
        else:
            raise ValueError(f"unknown view {view!r}")
        if split is not None:
            chosen = [clip for clip in chosen if clip.split == split]
        return tuple(chosen)

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance,
            "num_scanned": self.num_scanned,
            "num_skipped": self.num_skipped,
            "scanned_by_stratum": self.scanned_by_stratum,
            "num_population": sum(1 for clip in self.clips if clip.in_population),
            "num_boundary": sum(1 for clip in self.clips if clip.in_boundary),
            "clips": [clip.as_dict() for clip in self.clips],
        }


# --- Statistics. Torch-free, so the numbers a ship decision turns on are unit-testable. ---


def population_weights(scanned_by_stratum: dict[str, int]) -> dict[str, float]:
    """Each stratum's share of the *scanned* population, for un-stratifying an estimate."""
    total = sum(scanned_by_stratum.values())
    if total <= 0:
        return {name: 0.0 for name in scanned_by_stratum}
    return {name: count / total for name, count in scanned_by_stratum.items()}


MIN_REWEIGHTING_COVERAGE = 0.95


def reweighted_agreement(
    per_stratum_agreement: dict[str, float],
    scanned_by_stratum: dict[str, int],
    min_coverage: float = MIN_REWEIGHTING_COVERAGE,
) -> float | None:
    """Population agreement implied by per-stratum agreement and the scan counts.

    Renormalised over the strata that actually have samples -- but **only when those strata
    carry almost all of the population**. Renormalising over a sliver is not an estimate of
    anything: a sample covering one 1%-mass stratum at 100% agreement would otherwise report
    "100% population agreement" while saying nothing about the other 99%. Returns ``None``
    when coverage is below ``min_coverage``, so the caller has to print "unavailable" rather
    than a confident wrong number.
    """
    weights = population_weights(scanned_by_stratum)
    covered = sum(weights.get(name, 0.0) for name in per_stratum_agreement)
    if covered < min_coverage:
        return None
    return sum(
        agreement * weights.get(name, 0.0) / covered
        for name, agreement in per_stratum_agreement.items()
    )


def binomial_two_sided_p(successes: int, trials: int) -> float:
    """Exact two-sided binomial p-value at p=0.5 -- the McNemar test's exact form.

    The chi-square approximation is unreliable at the discordant-pair counts this evaluation
    produces (often under 25), and it is the approximation that would decide whether a
    reported gain is real. ``math.comb`` is exact and fast enough at these sizes.
    """
    if trials <= 0:
        return 1.0
    if trials > EXACT_BINOMIAL_LIMIT:
        # Beyond this the exact sum is slow and the normal approximation is excellent (the
        # counts are in the hundreds and the distribution is symmetric). Continuity-corrected.
        from statistics import NormalDist

        deviation = abs(successes - trials / 2) - 0.5
        if deviation <= 0:
            return 1.0
        return min(1.0, 2 * NormalDist().cdf(-deviation / math.sqrt(trials / 4)))
    weights = [math.comb(trials, k) for k in range(trials + 1)]
    observed = weights[successes]
    tail = sum(weight for weight in weights if weight <= observed)
    # Integer division, not ``2.0 ** trials``: the float power overflows at 1024 trials, and a
    # student that disagrees with the teacher on a thousand clips is exactly when this gets
    # called. ``int / int`` is correctly rounded at any size.
    return min(1.0, tail / (2**trials))


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- the one that stays inside [0, 1] near the extremes.

    A 95% target is measured close enough to 1.0 that the normal-approximation interval
    overshoots, and the entire purpose of this evaluation set is to say whether an interval
    excludes a value.
    """
    if trials <= 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    spread = (
        z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
    ) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass(frozen=True)
class PairedComparison:
    """One decision policy against another, on the same clips, as predictors of the teacher.

    Built for two uses: student versus the pass-everything policy (is this better than a
    rubber stamp?) and student versus another student (did this intervention help?). The
    second is the one that should decide an experiment -- against pass-everything, any
    competent student wins once the sample is balanced near the bar, so a small p-value there
    is not evidence the target is met.
    """

    name_a: str
    name_b: str
    a_only_correct: int
    b_only_correct: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.a_only_correct + self.b_only_correct

    def as_dict(self) -> dict:
        return {
            "comparison": f"{self.name_a} vs {self.name_b}",
            f"{self.name_a}_only_correct": self.a_only_correct,
            f"{self.name_b}_only_correct": self.b_only_correct,
            "discordant": self.discordant,
            "p_value": round(self.p_value, 6),
            "a_beats_b": bool(
                self.a_only_correct > self.b_only_correct and self.p_value < 0.05
            ),
        }


def paired_comparison(
    correct_a: list[bool], correct_b: list[bool], name_a: str, name_b: str
) -> PairedComparison:
    """Exact McNemar over two per-clip correctness vectors of the same length."""
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"paired comparison needs matched clips: {len(correct_a)} vs {len(correct_b)}"
        )
    a_only = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    b_only = sum(1 for a, b in zip(correct_a, correct_b) if b and not a)
    return PairedComparison(
        name_a=name_a,
        name_b=name_b,
        a_only_correct=a_only,
        b_only_correct=b_only,
        p_value=binomial_two_sided_p(min(a_only, b_only), a_only + b_only),
    )


@dataclass(frozen=True)
class DirectionalErrors:
    """Which way the student is wrong -- the thing an aggregate percentage hides.

    ``false_fail_rate`` is what a user feels: the teacher would have accepted the recitation
    and the student rejects it. ``false_pass_rate`` is what silently degrades the product:
    the student accepts a recitation the teacher would have flagged. A student can reach the
    same agreement by being too strict or too lax and the two call for opposite fixes.
    """

    teacher_passes: int
    teacher_fails: int
    false_fails: int
    false_passes: int

    @property
    def false_fail_rate(self) -> float:
        return self.false_fails / max(1, self.teacher_passes)

    @property
    def false_pass_rate(self) -> float:
        return self.false_passes / max(1, self.teacher_fails)

    def as_dict(self) -> dict:
        return {
            "teacher_passes": self.teacher_passes,
            "teacher_fails": self.teacher_fails,
            "false_fails": self.false_fails,
            "false_passes": self.false_passes,
            "false_fail_rate": round(self.false_fail_rate, 4),
            "false_pass_rate": round(self.false_pass_rate, 4),
        }


def directional_errors(decisions: list[tuple[bool, bool]]) -> DirectionalErrors:
    """Split disagreements by direction over ``(teacher_passed, student_passed)`` pairs."""
    return DirectionalErrors(
        teacher_passes=sum(1 for teacher, _ in decisions if teacher),
        teacher_fails=sum(1 for teacher, _ in decisions if not teacher),
        false_fails=sum(
            1 for teacher, student in decisions if teacher and not student
        ),
        false_passes=sum(
            1 for teacher, student in decisions if not teacher and student
        ),
    )


# The gate's three conditions, as predicates over one side's ``GateResult`` fields. Scoring
# agreement under each in turn is what separates "the student decodes differently" from "one
# asymmetric heuristic is unreproducible", and on the h384 baseline those are 3.2 and 5.4
# points of the same 8.6-point gap.
@dataclass(frozen=True)
class GateDefinition:
    """One decision the scorer can be asked to make, named by its threshold.

    **The Tadabur poison rejects are not here, and that is deliberate.** ``Scorer.gate``
    layers two of them on the Muraja-faithful ``match_ratio`` -- a long interior insertion
    run and an added shadda -- and its own comments call both "NOT a Muraja parameter",
    "Tadabur-only", "filter-side". They decide which clips enter the ADR-0001 **fine-tune**
    corpus. A size distillation is behavioural cloning of the teacher's decode; the corpus
    filter is not part of that question, and ADR-0008 (Accepted) already ruled that this gate
    "should not be the headline metric at all".

    What is left is the threshold, and it is not a formality. ADR-0005 records that Muraja's
    **advancement** decision is ``matchRatio`` against a hard-coded ``0.70`` which
    ``scoringMode`` does not touch. ``.balanced``'s ``0.65`` is the filter's bar. A clip at
    0.67 advances under one and not the other.
    """

    name: str
    label: str
    threshold: float

    def verdict(self, match_ratio: float) -> bool:
        return match_ratio >= self.threshold


GATE_DEFINITIONS: tuple[GateDefinition, ...] = (
    GateDefinition("advancement", "Muraja advancement, ratio >= 0.70 (ADR-0005)", 0.70),
    GateDefinition("bar_0_65", "at the Tadabur filter's 0.65 bar, for comparison", 0.65),
)

# The decision the **product** makes. Deliberately not the definition that scores best.
DISTILLATION_CRITERION = "advancement"


def gate_verdicts(match_ratio: float) -> dict[str, bool]:
    """Every definition's verdict on one side's ``match_ratio``."""
    return {d.name: d.verdict(match_ratio) for d in GATE_DEFINITIONS}


def gate_definition(name: str) -> GateDefinition:
    for definition in GATE_DEFINITIONS:
        if definition.name == name:
            return definition
    raise ValueError(f"unknown gate definition {name!r}")


def cluster_bootstrap_interval(
    outcomes: list[bool],
    clusters: list,
    iterations: int = 4000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap resampling **reciters**, not clips.

    :func:`wilson_interval` assumes the observations are independent. These are not: 2,000
    evaluation clips come from ~286 reciters, and whether the student agrees with the teacher
    on a clip is correlated within a voice -- same channel, same pace, same articulation. The
    independent-sample interval is therefore too narrow, and it is too narrow on exactly the
    question a ship decision asks ("is the lower bound above the bar?").

    Resampling whole reciters with replacement makes no assumption about the size of that
    correlation, which is the right trade when there is one clustering variable and enough of
    them to resample. The total clip count varies between draws, as it should -- a corpus
    with a different set of reciters really would have a different number of clips.

    Returns the naive interval unchanged when there is effectively no clustering to find (one
    cluster, or fewer than two), because a bootstrap over one cluster is not an interval.
    """
    if not outcomes or len(outcomes) != len(clusters):
        raise ValueError(
            f"need one cluster label per outcome: {len(outcomes)} vs {len(clusters)}"
        )
    grouped: dict = {}
    for outcome, cluster in zip(outcomes, clusters):
        grouped.setdefault(cluster, []).append(outcome)
    keys = list(grouped)
    if len(keys) < 2:
        return wilson_interval(sum(outcomes), len(outcomes))

    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        hits = total = 0
        for _ in keys:
            drawn = grouped[keys[rng.randrange(len(keys))]]
            hits += sum(drawn)
            total += len(drawn)
        estimates.append(hits / total)
    estimates.sort()
    low = estimates[int(alpha / 2 * iterations)]
    high = estimates[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (low, high)


def load_manifest(out_dir: Path) -> EvalSet:
    """Read a built set, refusing one this code cannot honestly interpret."""
    path = Path(out_dir) / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SystemExit(
            f"{path} was written by schema {version!r}; this code speaks {SCHEMA_VERSION!r}. "
            f"Rebuild it -- the cached teacher decisions are only valid for the protocol "
            f"and scorer the manifest was built under."
        )
    return EvalSet(
        schema_version=version,
        clips=tuple(
            EvalClip(
                filename=record["filename"],
                surah_ayah=record["surah_ayah"],
                reciter_id=record["reciter_id"],
                shard=record["shard"],
                duration_s=record["duration_s"],
                stratum=record["stratum"],
                split=record["split"],
                in_population=record["in_population"],
                in_boundary=record["in_boundary"],
                teacher_text=record["teacher_text"],
                teacher_ratio=record["teacher_ratio"],
                teacher_passed=record["teacher_passed"],
                teacher_insertion_run=record["teacher_insertion_run"],
                teacher_added_shadda=record["teacher_added_shadda"],
            )
            for record in payload["clips"]
        ),
        scanned_by_stratum=payload["scanned_by_stratum"],
        num_scanned=payload["num_scanned"],
        num_skipped=payload["num_skipped"],
        provenance=payload["provenance"],
    )


def check_provenance(evalset: EvalSet, teacher_model_id: str, threshold: float) -> None:
    """Refuse a manifest whose cached teacher decisions were made under other rules.

    A different teacher, a different decode protocol, a different confirmation split or a
    different pass bar all change the cached decisions without changing the file. The failure
    mode is a plausible-looking agreement number computed against stale truth, which is
    exactly the class of silent wrongness these tools exist not to produce. Every such field
    is checked **here**, in one place -- a second provenance check somewhere else is how one
    of them ends up unchecked.
    """
    from training.distill_eval import PROTOCOL_VERSION
    from training.distill_loss import CONFIRM_TIMESTEPS

    expected = {
        "teacher_model_id": teacher_model_id,
        "correct_threshold": threshold,
        "confirm_timesteps": CONFIRM_TIMESTEPS,
        "protocol_version": PROTOCOL_VERSION,
    }
    mismatches = [
        f"  {key}: manifest={evalset.provenance.get(key)!r} current={value!r}"
        for key, value in expected.items()
        if evalset.provenance.get(key) != value
    ]
    if mismatches:
        raise SystemExit(
            "refusing to use this evaluation set: its cached teacher decisions were "
            "produced under different rules.\n" + "\n".join(mismatches)
        )


class _Reservoir:
    """Uniform sample of at most ``quota`` items, in one streaming pass.

    Two passes over the candidates would be simpler, but a pass costs a re-download and a
    full teacher decode of every shard, so the sample must be chosen while streaming.
    Reservoir sampling keeps the sample uniform over everything seen; taking the first N
    instead would sample by shard order, i.e. by reciter.
    """

    def __init__(self, quota: int, seed: int) -> None:
        self.quota = quota
        self.rng = random.Random(seed)
        self.seen = 0
        self.kept: list = []

    def offer(self, item) -> bool:
        """Consider one candidate; returns whether it is currently in the sample."""
        self.seen += 1
        if len(self.kept) < self.quota:
            self.kept.append(item)
            return True
        index = self.rng.randrange(self.seen)
        if index < self.quota:
            self.kept[index] = item
            return True
        return False

    @property
    def is_full(self) -> bool:
        return len(self.kept) >= self.quota


def _clip_filename(shard: int, row_index: int, surah_id: int, ayah_id: int) -> str:
    """A name that carries its own provenance and parses with the staged-clip pattern.

    ``surah_id`` is Tadabur's **0-indexed** array position, matching the staged
    ``clips_v2`` filenames, so ``training.distill_gate.parse_surah_ayah`` reads these
    exactly as it reads the staged ones. The manifest carries the canonical key as well;
    the filename is for humans and for tools that only have a path.
    """
    return f"tadabur_sh{shard:03d}_i{row_index:05d}_S{surah_id}_A{ayah_id}_.wav"


def build(
    out_dir: Path,
    shards: list[int],
    population_target: int,
    boundary_target: int,
    row_sample: float,
    max_scan: int,
    seed: int,
    batch_size: int,
) -> EvalSet:
    """Scan the reserved shards once, decode each clip with the teacher, sample, freeze.

    One pass does everything: the teacher decode is the expensive part and is never repeated,
    the two samples are drawn with reservoirs so they stay uniform over the whole scan, and
    shard blobs are deleted as they are consumed so peak disk is one shard.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import SeamlessM4TFeatureExtractor

    from tadabur.audio import decode_to_mono_16k
    from tadabur.filter import canonical_surah_ayah
    from tadabur.reference_phonemes import load_reference_phonemes
    from tadabur.scorer import BALANCED, BALANCED_SCORER
    from tadabur.shard_reader import iter_shard_rows
    from training.distill_data import SAMPLE_RATE
    from training.distill_eval import PROTOCOL_VERSION, confirmed_stream
    from training.distill_loss import CONFIRM_TIMESTEPS
    from training.distill_gate import tokens_to_phonemes
    from training.distill_student import TEACHER_MODEL_ID
    from training.distill_train import load_teacher

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to build the evaluation set")
    device = torch.device("cuda")

    clips_dir = Path(out_dir) / CLIPS_DIRNAME
    clips_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_teacher(device)
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(TEACHER_MODEL_ID)
    references = load_reference_phonemes()

    population = _Reservoir(population_target, seed)
    quotas = boundary_quotas(boundary_target)
    boundary = {
        name: _Reservoir(quota, seed + index + 1)
        for index, (name, quota) in enumerate(sorted(quotas.items()))
    }

    # Rows are skipped *before* the teacher pass, so a sub-1.0 rate buys coverage across all
    # reserved shards at a fraction of the GPU cost -- which is the right trade: a scan that
    # exhausts its budget inside the first few shards samples reciters, not the corpus.
    sampler = random.Random(seed ^ 0x5EED)

    scanned_by_stratum = {name: 0 for name in STRATA}
    written: set[str] = set()
    num_scanned = num_skipped = 0
    started = time.time()

    for shard_position, shard in enumerate(shards):
        for row_index, row in enumerate(iter_shard_rows([shard], delete_after=True)):
            if max_scan and num_scanned >= max_scan:
                break
            if row_sample < 1.0 and sampler.random() >= row_sample:
                continue

            key = canonical_surah_ayah(int(row["surah_id"]), int(row["ayah_id"]))
            reference = references.get(key)
            if reference is None:
                num_skipped += 1
                continue
            try:
                samples = decode_to_mono_16k(row["audio"]["bytes"])
            except Exception:
                num_skipped += 1
                continue
            duration = len(samples) / SAMPLE_RATE
            if not MIN_CLIP_SECONDS <= duration <= MAX_CLIP_SECONDS:
                num_skipped += 1
                continue

            text = tokens_to_phonemes(
                confirmed_stream(teacher, extractor, samples, device, batch_size)
            )
            result = BALANCED_SCORER.gate(text, reference)
            stratum = stratum_for(result.match_ratio, BALANCED.correct_threshold)
            scanned_by_stratum[stratum] += 1
            num_scanned += 1

            reciter_id = int(row["reciter_id"])
            candidate = EvalClip(
                filename=_clip_filename(
                    shard, row_index, int(row["surah_id"]), int(row["ayah_id"])
                ),
                surah_ayah=key,
                reciter_id=reciter_id,
                shard=shard,
                duration_s=duration,
                stratum=stratum,
                split=reciter_split(reciter_id),
                in_population=False,
                in_boundary=False,
                teacher_text=text,
                teacher_ratio=result.match_ratio,
                teacher_passed=result.passed,
                teacher_insertion_run=result.max_insertion_run,
                teacher_added_shadda=result.added_shadda,
            )

            # ``|=``, never ``or``: both reservoirs must see every candidate, and ``or``
            # would skip the boundary offer whenever the population sample accepted first --
            # silently biasing the boundary view toward the clips the population view rejected.
            kept = population.offer(candidate)
            kept |= boundary[stratum].offer(candidate)
            if kept and candidate.filename not in written:
                # 32-bit float, not PCM_16, and this is not a preference. The manifest caches
                # the teacher's decode of the waveform *in memory*; the student will read the
                # file back. Any difference between the two is charged entirely to the
                # student. Measured on 60 clips, a PCM_16 round trip changed the teacher's
                # decoded string on 12 of them and moved match_ratio by up to 0.062 -- on a
                # set built to be dense at the 0.65 bar, that is flipped decisions attributed
                # to a model that did nothing. Tadabur peaks above 1.0 (measured 1.037), so
                # PCM_16 also clips real signal. FLOAT round-trips bit-exactly.
                sf.write(
                    str(clips_dir / candidate.filename),
                    np.asarray(samples, dtype="float32"),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )
                written.add(candidate.filename)

            if num_scanned % 100 == 0:
                rate = num_scanned / max(1e-6, time.time() - started)
                fill = " ".join(
                    f"{name}:{len(boundary[name].kept)}/{boundary[name].quota}"
                    for name in sorted(boundary)
                )
                print(
                    f"  shard {shard} ({shard_position + 1}/{len(shards)}) "
                    f"scanned {num_scanned} at {rate:.2f} clips/s -- "
                    f"pop {len(population.kept)}/{population.quota} {fill}",
                    flush=True,
                )
        if max_scan and num_scanned >= max_scan:
            break

    selected: dict[str, EvalClip] = {}
    for clip in population.kept:
        selected[clip.filename] = clip
    for reservoir in boundary.values():
        for clip in reservoir.kept:
            selected[clip.filename] = clip

    population_names = {clip.filename for clip in population.kept}
    boundary_names = {
        clip.filename for reservoir in boundary.values() for clip in reservoir.kept
    }
    final = tuple(
        replace(
            clip,
            in_population=clip.filename in population_names,
            in_boundary=clip.filename in boundary_names,
        )
        for clip in sorted(selected.values(), key=lambda c: c.filename)
    )

    # Reservoir sampling writes every clip it ever accepts and evicts some of them later, so
    # the clips directory is a superset of the final sample. Sweep the difference rather than
    # refcounting during the scan: a clip can sit in either sample or both, and getting that
    # bookkeeping wrong deletes audio the manifest still references.
    for path in clips_dir.glob("*.wav"):
        if path.name not in selected:
            path.unlink()

    evalset = EvalSet(
        schema_version=SCHEMA_VERSION,
        clips=final,
        scanned_by_stratum=scanned_by_stratum,
        num_scanned=num_scanned,
        num_skipped=num_skipped,
        provenance={
            "teacher_model_id": TEACHER_MODEL_ID,
            "correct_threshold": BALANCED.correct_threshold,
            "confirm_timesteps": CONFIRM_TIMESTEPS,
            # The cached decodes are only meaningful under the protocol that produced them;
            # the flush alone moves a short clip's transcript from one second of audio to all
            # of it. training.distill_gate refuses a mismatch.
            "protocol_version": PROTOCOL_VERSION,
            "shards": shards,
            "row_sample": row_sample,
            "max_scan": max_scan,
            "seed": seed,
            "population_target": population_target,
            "boundary_target": boundary_target,
            "near_band_half_width": NEAR_BAND_HALF_WIDTH,
            "min_clip_seconds": MIN_CLIP_SECONDS,
            "max_clip_seconds": MAX_CLIP_SECONDS,
            "source": "tadabur-shards-unfiltered",
            "elapsed_s": round(time.time() - started, 1),
        },
    )
    (Path(out_dir) / MANIFEST_FILENAME).write_text(
        json.dumps(evalset.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return evalset


def describe(evalset: EvalSet) -> str:
    """A human summary of what a built set contains and what it can resolve."""
    lines = [
        f"gate evaluation set -- {evalset.schema_version}",
        f"  scanned                 {evalset.num_scanned} clips "
        f"({evalset.num_skipped} skipped) from shards {evalset.provenance['shards']}",
        "  population of the scan:",
    ]
    weights = population_weights(evalset.scanned_by_stratum)
    for name in STRATA:
        lines.append(
            f"    {name:<12} {evalset.scanned_by_stratum.get(name, 0):>6} "
            f"({weights.get(name, 0.0):.1%})"
        )

    for view in ("population", "boundary"):
        clips = evalset.subset(view)
        if not clips:
            continue
        passes = sum(1 for clip in clips if clip.teacher_passed)
        low, high = wilson_interval(int(0.95 * len(clips)), len(clips))
        lines.append(f"  {view} sample: {len(clips)} clips")
        lines.append(
            f"    teacher pass rate     {passes / len(clips):.1%} "
            f"-- the pass-everything floor"
        )
        for split in ("dev", "test"):
            lines.append(
                f"    {split:<20}  {len(evalset.subset(view, split))} clips"
            )
        lines.append(
            f"    resolution            a measured 95.0% here has a 95% CI of "
            f"[{low:.1%}, {high:.1%}]"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen, stratified gate-evaluation set"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("../tadabur/gate_eval"))
    parser.add_argument(
        "--shards",
        default="",
        help="override the reserved shard list (default: every "
        f"{GATE_EVAL_SHARD_STRIDE}th shard from {GATE_EVAL_SHARD_START})",
    )
    parser.add_argument("--population-target", type=int, default=DEFAULT_POPULATION_TARGET)
    parser.add_argument("--boundary-target", type=int, default=DEFAULT_BOUNDARY_TARGET)
    parser.add_argument(
        "--row-sample",
        type=float,
        default=1.0,
        help="fraction of rows to decode, applied BEFORE the teacher pass. Below 1.0 it "
        "buys coverage of every reserved shard at a fraction of the GPU cost.",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="stop after this many decoded clips (0 = the whole reservation). A budget that "
        "runs out inside the first shards samples reciters, not the corpus -- prefer "
        "--row-sample.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--describe", action="store_true", help="summarise a built set")
    parser.add_argument(
        "--print-training-shards",
        action="store_true",
        help="print the --stream-shards spec that excludes every reserved shard",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.print_training_shards:
        # From the built manifest when there is one, so a set built with --shards does not
        # get a spec that still permits the shards it was drawn from.
        manifest = Path(args.out_dir) / MANIFEST_FILENAME
        reserved = (
            load_manifest(args.out_dir).provenance.get("shards")
            if manifest.exists()
            else None
        )
        print(training_shard_spec(reserved_shards=reserved))
        return

    if args.describe:
        evalset = load_manifest(args.out_dir)
        print(json.dumps(evalset.as_dict(), indent=2, ensure_ascii=False)
              if args.json else describe(evalset))
        return

    from tadabur.shard_reader import parse_shard_spec

    shards = parse_shard_spec(args.shards) if args.shards else gate_eval_shards()
    evalset = build(
        out_dir=args.out_dir,
        shards=shards,
        population_target=args.population_target,
        boundary_target=args.boundary_target,
        row_sample=args.row_sample,
        max_scan=args.max_scan,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print()
    print(describe(evalset))
    print(f"\nwrote {args.out_dir / MANIFEST_FILENAME}")
    print(
        f"training runs must use --stream-shards "
        f"{training_shard_spec(reserved_shards=shards)}"
    )


if __name__ == "__main__":
    main()

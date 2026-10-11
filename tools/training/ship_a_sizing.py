"""Ship A's sizing (#126): the verification budget and what the held-out half can certify.

It answers the four questions of #126 from committed inputs only, torch-free, with the
machinery of :mod:`training.power_simulation` and the verdicts of
:mod:`training.acceptance_stats`. It reads no candidate, and it **chooses no threshold, floor or
margin**: every power is reported over a grid of values for the owner to pick from (#128).
:mod:`training.ship_a_sizing_doc` renders ``docs/ship-a-sizing.md`` from the results.

1. **Verification budget** (:func:`budget`): Ship A's sites (:mod:`tadabur.ship_a`) per pair
   direction, per family and for geminates, how many the base teacher already gets right, and
   how they fall into the train and held-out halves of the confirmed 50/50 split (D11). A
   held-out cell with fewer than 10 reciters or 20 sites is sparse (§1): it is
   pooled into its family and reported as insufficient evidence (§7).
2. **Power on the held-out half** (:func:`held_out_power`): per evaluation cell, the held-out
   sites listened to, each a real mistake with probability θ (:data:`THETAS`); the candidate
   recovers a verified-correct site (now commits the mushaf letter or holds the geminate), or
   retains a verified mistake, with probability :data:`TRUE_RATES`; reciter effects at each
   ICC of :data:`training.power_simulation.RECITER_ICCS`. Each repetition takes the §1 lower
   bound of the rate, read two ways: **one site per reciter** (independent and equally
   weighted, so Wilson where sparse) and, for the pooled cells, **every site** with the
   reciter-clustered bootstrap. The power at each of :data:`FLOORS` is the share of
   repetitions whose bound reaches it. :func:`sites_needed` gives the reciters one site each
   needs for 80% power at each true rate and floor.
3. **New-error harm** (:func:`harm_power`): a candidate changes a share (:data:`CHURNS`) of the
   held-out carriers where ``h448`` commits the mushaf value (pair carriers once each, and
   in-word geminates), reciter-clustered; a share of the changed sites are new errors
   (:data:`NEW_ERROR_SHARES`). The changed sites are a census once the candidate exists; a
   blind, candidate-locked PPS cluster sample of ``n`` reciters (:data:`CLUSTER_RECITERS`)
   adjudicates every changed site of each reciter drawn, and
   :func:`training.power_simulation.cluster_mistake_bound` bounds the new errors. Reported:
   the minutes, the median bound per 1,000 carriers and the chance it reaches each of
   :data:`HARM_RATES`. The word sample's power for a lower false-flag rate is
   :func:`word_sample_power`.
4. **Mix of real mistakes** (:func:`mistake_mix`): on the P3.5 fixtures (the fixture-side
   report, rebuilt from the committed decode caches at every run,
   :func:`tadabur.eval_harness.build`, so the guards and the verdicts are the current
   ones), where ``h448`` commits an off value, how many are should-accept (correct recitation) and how many
   should-reject (pending, an upper bound on the share of mistakes: a clip-level reject does
   not say the site was the mistake), and the verified mistakes the "keep both outcomes" term
   gets on the train half at each θ.

Every population is read **after the shared exclusion** (:mod:`tadabur.exclusions`): a site
or sampled word ``waqf_realized`` at the current verdicts (:mod:`tadabur.waqf_guard`, #134:
its prescribed mark only exists in wasl, and the reciter paused there) or ``onset_trimmed``
(:mod:`tadabur.onset_guard`, #138: its clip's opening sound is cut) leaves every cell;
``sizing.json`` lists them per guard with the guard's checksum. The planning frame (question
3's carriers ``h448`` gets right) is not guarded: the guards classify the verified
populations, not the frame.

Usage (from ``tools/``; torch-free, CPU; ``--workers`` processes)::

  python -m training.ship_a_sizing run [--workers 6]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tadabur.eval_harness import build as fixture_side_report
from tadabur.exclusions import GUARD_PATHS, REASONS, Exclusions, current
from tadabur.listening_session import VERDICTS_PATH
from tadabur.onset_guard import ONSET_TRIMMED
from tadabur.power_frame import FRAME_PATH, SHIP_A_DIR, input_checksums
from tadabur.ship_a import (
    GEMINATE,
    HELD_OUT,
    SITES_PATH,
    TRAIN,
    ShipASite,
    listening_weights,
    load_sites,
    reciter_half,
)
from tadabur.truth_sites import SHADDAH, SOFT_PAIRS, UNFORGIVEN_PAIRS
from tadabur.waqf_guard import WAQF_REALIZED
from tadabur.word_sample import WORD_SAMPLE_PATH, WordSite, load_words
from training.acceptance_stats import (
    LOWER,
    NO_BOUND,
    PASS,
    SPARSE_MIN_RECITERS,
    SPARSE_MIN_SITES,
    UPPER,
    Interval,
    difference_verdict,
    is_sparse,
    ratio_interval,
    ratio_verdict,
    unit_if_equal,
)
from training.power_simulation import (
    RECITER_ICCS,
    REPETITIONS,
    TARGET_POWER,
    Component,
    clustered_bernoulli,
    cluster_mistake_bound,
    load_frame,
    one_per_reciter,
    pps_inclusion,
    rng_for,
    shared_resample,
    sigma_for,
    size_component,
)
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import DEFAULT_MODELS

SIZING_PATH = SHIP_A_DIR / "sizing.json"

# --- planning ranges: this sizing's grids, never thresholds -----------------------------------
#: θ, the share of real mistakes among the sites ``h448`` flags (question 4 bounds it).
THETAS = (0.05, 0.15, 0.5)
#: The candidate's true recovery of verified-correct sites, or retention of verified mistakes.
TRUE_RATES = (0.5, 0.7, 0.9)
#: Values a §1 lower bound is read at; the owner picks the floor (#128).
FLOORS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
#: Share of held-out carriers ``h448`` gets right that a candidate changes.
CHURNS = (0.0025, 0.01, 0.025)
#: Share of the changed sites that are new errors.
NEW_ERROR_SHARES = (0.1, 0.5, 1.0)
#: Reciters a PPS cluster sample of changed sites is expected to draw.
CLUSTER_RECITERS = (10, 20, 40, 80)
#: New errors per 1,000 held-out carriers a harm bound is read at.
HARM_RATES = (1.0, 2.0, 5.0, 10.0)
#: ``h448``'s app false-flag rate on correctly recited words: unknown until #123's replay.
FALSE_FLAG_RATES = (0.02, 0.05, 0.10)
#: The candidate's relative reduction of those false flags.
REDUCTIONS = (0.25, 0.5, 0.75)
#: Share of sampled words heard as recited correctly (``correct``), the rest mistakes or unclear.
WORD_CORRECT_SHARE = 0.9
#: Upper-bound values of the word sample's paired difference (candidate − ``h448``).
WORD_MARGINS = (0.0, -0.01, -0.02)

RECOVERY = "recovery"
RETENTION = "retention"
ONE_PER_RECITER = "one_per_reciter"
EVERY_SITE = "every_site"
GEMINATE_FRAME_STRATUM = "shaddah:held:in_word"
#: The Muraja soft pairs pooled (ADR-0012: one switch forgives them all).
SOFT_PAIRS_POOLED = "soft pairs (pooled)"
GEMINATES = "geminates (base not held)"


def _bucket(sites: list[ShipASite]) -> dict:
    """Sites and reciters per half, and how many the base teacher already gets right."""
    out = {"sites": len(sites), "base_already_right": sum(s.base_already_right for s in sites),
           "listened": sum(s.listened for s in sites),
           "partner_swaps": sum(s.partner_swap for s in sites)}
    for half in (TRAIN, HELD_OUT):
        group = [s for s in sites if s.half == half]
        listened = [s for s in group if s.listened]
        out[half] = {"sites": len(group), "listened": len(listened),
                     "listened_reciters": len({s.reciter_id for s in listened})}
    held_out = out[HELD_OUT]
    out["held_out_sparse"] = is_sparse(held_out["listened"], held_out["listened_reciters"])
    return out


def _direction_of(site: ShipASite) -> str:
    """A partner swap's direction, or the carrier letter of any other off-letter commit."""
    if site.kind == GEMINATE:
        return SHADDAH
    if site.partner_swap:
        return f"{site.site.prescribed}→{site.h448_committed} (partner)"
    return f"{site.site.prescribed}→other letter"


def budget(sites: list[ShipASite]) -> dict:
    """Question 1: per direction, per family and per evaluation cell."""
    def grouped(key) -> dict:
        groups: dict[str, list[ShipASite]] = {}
        for site in sites:
            groups.setdefault(key(site), []).append(site)
        return {name: _bucket(group) for name, group in sorted(groups.items())}

    return {
        "directions": grouped(_direction_of),
        "families": grouped(lambda s: s.family),
        "cells": {name: _bucket(group) for name, group in cells(sites).items()},
        "sparse_rule": {"min_reciters": SPARSE_MIN_RECITERS, "min_sites": SPARSE_MIN_SITES},
    }


def cells(sites: list[ShipASite]) -> dict[str, list[ShipASite]]:
    """The evaluation cells: the Muraja soft pairs pooled, each target pair on its own (the
    three no allowance covers never pooled with them), and the geminates listened to."""
    return {
        SOFT_PAIRS_POOLED: [s for s in sites if s.family in SOFT_PAIRS],
        **{family: [s for s in sites if s.family == family]
           for family in (*sorted(SOFT_PAIRS), *sorted(UNFORGIVEN_PAIRS))},
        GEMINATES: [s for s in sites if s.kind == GEMINATE and s.listened],
    }


# --- question 2: power on the held-out half ---------------------------------------------------


@dataclass(frozen=True)
class SideScenario:
    """One power scenario on one cell: its held-out listened sites' reciters and design weights
    (the inverse of each worklist row's inclusion probability), θ, the true rate, the ICC,
    which side is read and how (one site per reciter, or every site)."""

    cell: str
    reciters: tuple[int, ...]
    weights: tuple[float, ...]
    theta: float
    rate: float
    icc: float
    side: str
    rule: str

    def key(self) -> tuple:
        return ("side", self.cell, self.theta, self.rate, self.icc, self.side, self.rule)


def cell_interval(reciters: np.ndarray, weights: np.ndarray, outcome: np.ndarray) -> Interval:
    """§1's interval of a cell's weighted rate Σw·outcome / Σw, read as the #84 scorer reads it
    under design weights: a cell of one common weight is equally weighted
    (:func:`training.acceptance_stats.unit_if_equal`), and the bootstrap is the scorer's draw."""
    weights = unit_if_equal(weights)
    labels, resample = shared_resample(reciters)
    return ratio_interval(labels, weights * outcome, weights, weights, resample)


def side_bounds(scenario: SideScenario, repetitions: int) -> dict:
    """The §1 lower bound of the side's rate per repetition, and the power at each floor."""
    rng = rng_for(*scenario.key(), repetitions)
    reciters = np.asarray(scenario.reciters, dtype=np.int64)
    weights = np.asarray(scenario.weights, dtype=float)
    _, reciters_index = np.unique(reciters, return_inverse=True)
    sigma = sigma_for(scenario.icc)
    bounds = np.full(repetitions, -math.inf)
    sites, held = np.zeros(repetitions), np.zeros(repetitions)
    for k in range(repetitions):
        correct = clustered_bernoulli(1 - scenario.theta, reciters, sigma, rng)
        on_side = correct if scenario.side == RECOVERY else ~correct
        outcome = clustered_bernoulli(scenario.rate, reciters, sigma, rng).astype(float)
        if scenario.rule == ONE_PER_RECITER:
            chosen = one_per_reciter(on_side, reciters, rng, weights)
            per_reciter = np.bincount(reciters_index, weights=weights * on_side)
            den = per_reciter[reciters_index[chosen]]
        else:
            chosen = np.flatnonzero(on_side)
            den = weights[chosen]
        if not len(chosen):
            continue
        interval = cell_interval(reciters[chosen], den, outcome[chosen])
        sites[k], held[k] = len(chosen), len(set(reciters[chosen].tolist()))
        if interval.method != NO_BOUND:
            bounds[k] = interval.lower
    return {
        "power": {str(f): float(np.mean(bounds >= f)) for f in FLOORS},
        "median_bound": float(np.median(bounds)) if np.isfinite(np.median(bounds)) else None,
        "mean_sites": float(sites.mean()),
        "mean_reciters": float(held.mean()),
    }


def kish(weights: Iterable[float]) -> float:
    """Kish's effective sample size, (Σw)² / Σw²."""
    weights = list(weights)
    return round(sum(weights) ** 2 / sum(w * w for w in weights), 1) if weights else 0.0


def effective_sizes(sites: list[ShipASite]) -> dict[str, dict[str, float]]:
    """Per cell, its held-out listened sites and their Kish effective size under the
    listening weights (:func:`tadabur.ship_a.listening_weights`)."""
    sizes = {}
    for cell, members in cells(sites).items():
        held_out = [s for s in members if s.listened and s.half == HELD_OUT]
        sizes[cell] = {"sites": len(held_out),
                       "effective": kish(listening_weights(held_out).values())}
    return sizes


def side_scenarios(sites: list[ShipASite]) -> list[SideScenario]:
    """Every cell, θ, true rate, ICC and side; every site as well for the pooled cells."""
    scenarios = []
    for cell, members in cells(sites).items():
        held_out = [s for s in members if s.listened and s.half == HELD_OUT]
        reciters = tuple(s.reciter_id for s in held_out)
        weights = tuple(listening_weights(held_out)[s.site.site_id] for s in held_out)
        rules = (ONE_PER_RECITER, EVERY_SITE) if cell in (SOFT_PAIRS_POOLED, GEMINATES) \
            else (ONE_PER_RECITER,)
        scenarios += [
            SideScenario(cell, reciters, weights, theta, rate, icc, side, rule)
            for theta in THETAS for rate in TRUE_RATES for icc in RECITER_ICCS
            for side in (RECOVERY, RETENTION) for rule in rules
        ]
    return scenarios


def sites_needed(rate: float, floor: float, repetitions: int = REPETITIONS) -> dict:
    """Reciters, one site each, for the §1 lower bound of a true ``rate`` to reach ``floor``
    with :data:`training.power_simulation.TARGET_POWER` (Wilson where sparse)."""
    def verdicts(n: int, rng) -> dict[str, str]:
        outcome = (rng.random(n) < rate).astype(float)
        ones = np.ones(n)
        labels, resample = shared_resample(np.arange(n))
        return {"floor": ratio_verdict(labels, outcome, ones, floor, LOWER, ones,
                                       resample).verdict}

    return size_component(Component(f"sites_needed:{rate}:{floor}", 400, verdicts),
                          TARGET_POWER, repetitions)


# --- question 3: new-error harm and the word sample -------------------------------------------


@dataclass(frozen=True)
class HarmScenario:
    """A candidate's changes on the held-out carriers ``h448`` gets right, read by a PPS
    cluster sample of ``n`` reciters."""

    carriers: tuple[int, ...]
    seconds_per_site: float
    churn: float
    new_error_share: float
    icc: float
    n: int

    def key(self) -> tuple:
        return ("harm", self.churn, self.new_error_share, self.icc, self.n)


def harm_power(scenario: HarmScenario, repetitions: int) -> dict:
    """Per repetition, the 97.5% bound on new errors per 1,000 carriers and the sites heard."""
    rng = rng_for(*scenario.key(), repetitions)
    carriers = np.asarray(scenario.carriers, dtype=np.int64)
    site_reciters = np.repeat(np.arange(len(carriers)), carriers)
    sigma = sigma_for(scenario.icc)
    bounds, heard = np.zeros(repetitions), np.zeros(repetitions)
    for k in range(repetitions):
        changed_sites = clustered_bernoulli(scenario.churn, site_reciters, sigma, rng)
        changed = np.bincount(site_reciters[changed_sites], minlength=len(carriers))
        errors = rng.binomial(changed, scenario.new_error_share)
        pi = pps_inclusion(changed, scenario.n)
        sampled = rng.random(len(pi)) < pi
        bounds[k] = 1000 * cluster_mistake_bound(pi, changed, errors, sampled) / carriers.sum()
        heard[k] = changed[sampled].sum()
    return {
        "true_rate_per_1000": 1000 * scenario.churn * scenario.new_error_share,
        "expected_changed_sites": scenario.churn * int(carriers.sum()),
        "mean_sites_heard": float(heard.mean()),
        "mean_minutes": float(heard.mean() * scenario.seconds_per_site / 60),
        "median_bound_per_1000": float(np.median(bounds)),
        "power": {str(x): float(np.mean(bounds <= x)) for x in HARM_RATES},
    }


def held_out_carriers(frame_path: Path = FRAME_PATH) -> tuple[tuple[int, ...], float]:
    """Per held-out reciter, the pool carriers ``h448`` gets right (each pair carrier once,
    under the first direction its letter has, since ``h448``'s commit there is pair-free; and
    the in-word geminates), and their mean listening seconds."""
    strata = load_frame(frame_path)
    first_direction = {}
    for name in sorted(strata):
        if name.startswith("pair:"):
            first_direction.setdefault(name.split(":")[1][0], name)
    chosen = [strata[name] for name in (*first_direction.values(), GEMINATE_FRAME_STRATUM)]
    per_reciter: dict[int, int] = {}
    for stratum in chosen:
        for reciter, right in zip(stratum.reciters, stratum.field("h448_mushaf")):
            if reciter_half(int(reciter)) == HELD_OUT:
                per_reciter[int(reciter)] = per_reciter.get(int(reciter), 0) + int(right)
    seconds = sum(s.seconds_per_site * s.population for s in chosen) / sum(
        s.population for s in chosen)
    return tuple(v for _, v in sorted(per_reciter.items()) if v), seconds


def harm_scenarios(carriers: tuple[int, ...], seconds: float) -> list[HarmScenario]:
    return [HarmScenario(carriers, seconds, churn, share, icc, n)
            for churn in CHURNS for share in NEW_ERROR_SHARES for icc in RECITER_ICCS
            for n in CLUSTER_RECITERS]


@dataclass(frozen=True)
class WordScenario:
    """The frozen word sample (one word per reciter), each word weighing the inverse of its
    selection probability within the sample's frame (:func:`word_weights`). A word is heard
    as recited correctly with probability :data:`WORD_CORRECT_SHARE`; on those, ``h448``
    flags each with probability ``rate``, and the candidate keeps each flag with probability
    1 - ``reduction`` and flags no word ``h448`` does not."""

    weights: tuple[float, ...]
    rate: float
    reduction: float

    def key(self) -> tuple:
        return ("words", len(self.weights), self.rate, self.reduction)


def word_sample_power(scenario: WordScenario, repetitions: int) -> dict:
    """The chance the upper bound of the weighted paired difference in false-flag rate
    (candidate - ``h448``, over the correctly recited words, one per reciter: the bootstrap,
    or Tango where sparse and equally weighted, §1) reaches each of :data:`WORD_MARGINS`."""
    rng = rng_for(*scenario.key(), repetitions)
    weights = np.asarray(scenario.weights, dtype=float)
    passes = {m: 0 for m in WORD_MARGINS}
    for _ in range(repetitions):
        w = unit_if_equal(weights[rng.random(len(weights)) < WORD_CORRECT_SHARE])
        today = rng.random(len(w)) < scenario.rate
        candidate = today & (rng.random(len(w)) >= scenario.reduction)
        labels, resample = shared_resample(np.arange(len(w)))
        for margin in WORD_MARGINS:
            verdict = difference_verdict(labels, (w * candidate, w), (w * today, w), margin,
                                         UPPER, w, resample).verdict
            passes[margin] += verdict == PASS
    return {"power": {str(m): passes[m] / repetitions for m in WORD_MARGINS}}


def word_weights(words: Iterable[WordSite]) -> tuple[float, ...]:
    """Each sampled word's weight: the inverse of its selection probability within the
    sample's frame (the reciter draw times one of the reciter's words), never the pool's clip
    probability (owner decision 2026-10-10: the estimand is the frozen sample)."""
    return tuple(1 / w.draw_probability for w in words)


# --- question 4: the mix of real mistakes -----------------------------------------------------


def mistake_mix(fixture_sides: Mapping, sites: list[ShipASite]) -> dict:
    """Per family, the P3.5 sites where ``h448`` (streamed at b = 0) commits an off value, by
    fixture side; and the verified mistakes the train half would hold at each θ."""
    by_family: dict[str, dict[str, int]] = {}
    for site in fixture_sides["sites"]:
        committed = site["arms"]["h448/stream_b0"]["committed"]
        if committed is None or committed == site["prescribed"]:
            continue
        family = "soft pairs (pooled)" if site["family"] in SOFT_PAIRS else site["family"]
        sides = by_family.setdefault(family, {"should_accept": 0, "should_reject": 0})
        sides[site["fixture_side"]] += 1
    for sides in by_family.values():
        total = sides["should_accept"] + sides["should_reject"]
        sides["reject_share_upper"] = sides["should_reject"] / total
    train = {cell: sum(s.listened and s.half == TRAIN for s in members)
             for cell, members in cells(sites).items()}
    return {
        "p35_h448_off": dict(sorted(by_family.items())),
        "train_listened": train,
        "expected_train_mistakes": {cell: {str(t): round(n * t, 1) for t in THETAS}
                                    for cell, n in train.items()},
    }


# --- the run ----------------------------------------------------------------------------------


def _run(job: tuple[str, object, int]) -> tuple[tuple, dict]:
    kind, scenario, repetitions = job
    run = {"side": side_bounds, "harm": harm_power, "words": word_sample_power}[kind]
    return scenario.key(), run(scenario, repetitions)


#: Each reason's block in ``sizing.json``, and the key its ids are listed under.
GUARD_BLOCKS = {WAQF_REALIZED: ("waqf_guard", "realized"),
                ONSET_TRIMMED: ("onset_guard", "trimmed")}


def guard_block(reason: str, every_site: list[ShipASite], words: list[WordSite],
                excluded_items: Exclusions) -> dict:
    """What one guard takes out: its sites per evaluation cell and family, its words, and
    every id it names (``sizing.json`` is a function of them). An item two guards take out
    counts under the first (:mod:`tadabur.exclusions`)."""
    taken = [s for s in every_site if excluded_items.reason(s.site.site_id) == reason]
    return {
        "guard": _sha256(GUARD_PATHS[reason]),
        "verdicts": _sha256(VERDICTS_PATH),
        "cells": {cell: {"sites": len(members),
                         "held_out_listened": sum(s.listened and s.half == HELD_OUT
                                                  for s in members)}
                  for cell, members in cells(taken).items() if members},
        "families": dict(sorted(Counter(s.family for s in taken).items())),
        "sites": len(taken),
        "words": sum(excluded_items.reason(w.site.site_id) == reason for w in words),
        GUARD_BLOCKS[reason][1]: sorted(excluded_items.by_reason[reason]),
    }


def guarded(every_site: list[ShipASite], words: list[WordSite], excluded_items: Exclusions
            ) -> tuple[list[ShipASite], list[WordSite]]:
    """The sites and sampled words the shared exclusion keeps."""
    return ([s for s in every_site if s.site.site_id not in excluded_items],
            [w for w in words if w.site.site_id not in excluded_items])


def run(workers: int, repetitions: int = REPETITIONS) -> dict:
    every_site, every_word, excluded_items = load_sites(), load_words(), current()
    sites, words = guarded(every_site, every_word, excluded_items)
    fixtures = fixture_side_report(DEFAULT_MODELS, None, excluded_items)
    carriers, seconds = held_out_carriers()
    sampled_word_weights = word_weights(words)
    jobs: list[tuple[str, object, int]] = [
        *(("side", s, repetitions) for s in side_scenarios(sites)),
        *(("harm", s, repetitions) for s in harm_scenarios(carriers, seconds)),
        *(("words", WordScenario(sampled_word_weights, rate, reduction), repetitions)
          for rate in FALSE_FLAG_RATES for reduction in REDUCTIONS),
    ]
    with ProcessPoolExecutor(workers) as pool:
        results = dict(pool.map(_run, jobs, chunksize=4))
    side = [{"cell": s.cell, "theta": s.theta, "rate": s.rate, "icc": s.icc,
             "side": s.side, "rule": s.rule, "held_out_sites": len(s.reciters),
             **results[s.key()]} for _, s, _ in jobs if isinstance(s, SideScenario)]
    harm = [{"churn": s.churn, "new_error_share": s.new_error_share, "icc": s.icc, "n": s.n,
             **results[s.key()]} for _, s, _ in jobs if isinstance(s, HarmScenario)]
    words = [{"words": len(s.weights), "rate": s.rate, "reduction": s.reduction,
              **results[s.key()]} for _, s, _ in jobs if isinstance(s, WordScenario)]
    return {
        "inputs": {**input_checksums(), **{
            name: _sha256(path) for name, path in (
                ("ship_a/sites.jsonl", SITES_PATH),
                ("ship_a/word_sample.jsonl", WORD_SAMPLE_PATH))}},
        "fixture_report": {"inputs": fixtures["inputs"],
                           **{reason: fixtures[reason] for reason in REASONS}},
        "repetitions": repetitions,
        "grids": {"thetas": THETAS, "true_rates": TRUE_RATES, "floors": FLOORS,
                  "iccs": RECITER_ICCS, "churns": CHURNS, "new_error_shares": NEW_ERROR_SHARES,
                  "cluster_reciters": CLUSTER_RECITERS, "harm_rates": HARM_RATES,
                  "false_flag_rates": FALSE_FLAG_RATES, "reductions": REDUCTIONS,
                  "word_margins": WORD_MARGINS, "word_correct_share": WORD_CORRECT_SHARE},
        **{block: guard_block(reason, every_site, every_word, excluded_items)
           for reason, (block, _) in GUARD_BLOCKS.items()},
        "budget": budget(sites),
        "effective_sizes": effective_sizes(sites),
        "word_sample_effective": {"words": len(sampled_word_weights),
                                  "effective": kish(sampled_word_weights)},
        "held_out_power": side,
        "sites_needed": [{"rate": rate, "floor": floor, **sites_needed(rate, floor, repetitions)}
                         for rate in TRUE_RATES for floor in FLOORS if floor < rate],
        "harm": {"held_out_carriers": int(sum(carriers)), "reciters": len(carriers),
                 "seconds_per_site": seconds, "scenarios": harm},
        "word_sample": words,
        "mistake_mix": mistake_mix(fixtures, sites),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_ = commands.add_parser("run", help="run the sizing and write ship_a/sizing.json")
    run_.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    results = run(args.workers)
    write_text_atomically(SIZING_PATH, json.dumps(results, indent=1, ensure_ascii=False,
                                                  sort_keys=True) + "\n")
    print(f"wrote {SIZING_PATH}")


if __name__ == "__main__":
    main()

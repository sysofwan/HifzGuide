"""The truth-site scorer (ADR-0011 §5, acceptance rules §1): sites + decodes in, a report out.

Plain data in and out, torch-free. :func:`score` takes

* the truth sites (:mod:`tadabur.truth_sites`), every one of them, ``unclear`` and
  ``pending`` included, so exclusions are counted rather than lost;
* each site's **canonical reciter id** (from the staged-clip registry), the cluster every
  interval resamples;
* the decodes of every item, per **arm**: a model under a decode protocol (whole spans, or
  the stream at b = 0);

and returns a JSON-ready report. Per-site outcomes come from :mod:`training.site_outcomes`,
Muraja's site-level grades from :mod:`training.muraja_policy`, and every rate, interval and
verdict from :mod:`training.acceptance_stats`.

One record per physical site (§1)
---------------------------------
:func:`reconcile` runs before anything is scored or exported: records that share a
:func:`physical_site` (audio checksum, item span, carrier and mark) count once, under the
directly adjudicated record (a direct ``unclear`` still beats a weak label, and goes to
the exclusions; a ``pending`` record yields to an adjudication, never to a weak label), with
weights computed over every record so the frozen sampling weights stand. Conflicting labels
raise.

Populations (§1), never pooled with each other
----------------------------------------------
:func:`population` assigns each site from its truth record alone:

* ``headline``: directly adjudicated sites from the listening session (source
  ``new_audit``): the sampled subpopulation every headline rate is over;
* ``pause``: the sukun-at-pause sites (``waqf_boundary``, not a weak label). Their own row,
  not the in-scope sukun population;
* ``targeted_safeguard``: the re-located P3.5 fixtures (``p35_fixture``). Their sites were
  **selected** where the base teacher's decode showed the contrast (#83), so they measure
  behaviour at known base errors, not a rate over correct recitation;
* ``synthetic_edit``: never in a real-mistake rate;
* ``diagnostic``: every label that ``assumes_competent_reciter`` (the wasl boundaries and the
  long-vowel haraka at a pause). Shown, in no gate.

Cells
-----
Within a population, sites with a verdict are split by side (correct recitation or real
mistake, :func:`training.site_outcomes.side`) and indexed by the human-heard value: a
haraka or sukun (family ``tashkeel``), a gemination state (``shaddah``), or a letter of a
pair (family ``a↔b``, so each direction of each pair is its own cell, §7). Each population
and side also has a pooled ``all`` cell (the §3 headline and the §2 pooled missed mistakes).
Correct recitation and real mistakes are never in one cell.

Each site is weighted by its stratum: population / sites sampled in the stratum, counting
the ``unclear`` and ``pending`` sites that were sampled too (weights are never inflated to
cover them). A cell reports its sites, reciters, whether it is **sparse** (fewer than 10
reciters or 20 sites), and, per arm, every rate with its §1 interval and the method behind
it (:func:`training.acceptance_stats.ratio_interval`): the reciter-clustered bootstrap, a
Wilson bound where a sparse or degenerate cell is independent and equally weighted, or
``none`` where no bound is admissible. Shaddah cells are ``provisional`` until #92.

Rates per arm, by side
----------------------
Correct recitation: ``commit_rate`` ΣwC/Σw, ``committed_accuracy`` ΣwA/ΣwC,
``decode_flagged`` ΣwF/Σw, Muraja's ``false_flags`` Σw[WRONG]/Σw and ``coverage``
Σw[graded]/Σw. Real mistakes: ``commit_rate``, ``committed_accuracy``,
``missed_mistakes`` Σw(1−F)/Σw (an abstention is a miss), ``silent_corrections``
Σw[committed = prescribed]/Σw and Muraja's ``muraja_missed`` Σw[not WRONG]/Σw. A cell whose
heard value is sukun also reports ``spurious_haraka``: Σw[any haraka emitted]/Σw (§5).

Comparisons between models (and between protocols) are paired: one draw per cell serves
every arm, and each difference is recomputed inside every replicate; a sparse or degenerate
difference takes the Tango bound where admissible, otherwise none
(:func:`training.acceptance_stats.difference_interval`).

Exclusions
----------
A site heard ``unclear`` or ``pending`` leaves every denominator. Each cell lists the excluded
sites that could belong to it (same population and family, a prescribed value compatible
with the cell's side and heard value) and the best and worst case of its commit rate and
committed accuracy with all of them counted as successes or as failures.

Required cells (§1, §9)
-----------------------
:data:`REQUIRED_CELLS` freezes, before any candidate result, every cell the §2 probe gate,
the §3 ship criterion and the §5 bias probe must judge. Unsupported cells stay on the list
and come out ``cannot_certify``. Pair cells are directional; shaddah cells are provisional.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tadabur.truth_sites import (
    GEMINATION_STATES,
    HARAKA_CHARS,
    NEW_AUDIT,
    P35_FIXTURE,
    PENDING,
    SHADDAH,
    SUKUN,
    SYNTHETIC_EDIT,
    UNCLEAR,
    WAQF_BOUNDARY,
    TruthSite,
)
from training.acceptance_stats import (
    REPLICATES,
    SEED,
    Resample,
    difference_interval,
    is_sparse,
    ratio,
    ratio_interval,
)
from training.muraja_policy import ALLOWANCES, NOT_GRADED, TODAY, WRONG, MurajaConfig, grade
from training.site_outcomes import (
    CORRECT_SIDE,
    MISTAKE_SIDE,
    TARGET_PAIRS,
    TASHKEEL,
    SiteOutcome,
    family,
    item_outcomes,
    side,
)

HEADLINE = "headline"
PAUSE = "pause"
SAFEGUARD = "targeted_safeguard"
SYNTHETIC = "synthetic_edit"
DIAGNOSTIC = "diagnostic"
POPULATIONS = (HEADLINE, PAUSE, SAFEGUARD, SYNTHETIC, DIAGNOSTIC)
_POPULATION_OF_SOURCE = {
    NEW_AUDIT: HEADLINE,
    WAQF_BOUNDARY: PAUSE,
    P35_FIXTURE: SAFEGUARD,
    SYNTHETIC_EDIT: SYNTHETIC,
}

#: The pooled family / heard value of a population's ``all`` cell.
ALL = "all"
HARAKAT = tuple(sorted(HARAKA_CHARS))


def population(site: TruthSite) -> str:
    """The population a site belongs to (module docstring), from its truth record alone."""
    if site.assumes_competent_reciter:
        return DIAGNOSTIC
    return _POPULATION_OF_SOURCE[site.source]


@dataclass(frozen=True, order=True)
class Cell:
    """A population's sites on one side, heard as one value of one family (or ``all``)."""

    population: str
    side: str
    family: str
    heard: str

    @property
    def label(self) -> str:
        if self.family == ALL:
            return ALL
        if self.family == TASHKEEL:
            return self.heard
        if self.family == SHADDAH:
            return f"shaddah:{self.heard}"
        return f"{self.family}:{self.heard}"

    def holds(self, site: TruthSite) -> bool:
        """Whether a site with a verdict belongs to this cell."""
        return (
            population(site) == self.population
            and side(site) == self.side
            and (self.family == ALL or (family(site.mark) == self.family and site.heard == self.heard))
        )

    def could_hold(self, site: TruthSite) -> bool:
        """Whether an excluded site could belong here once adjudicated."""
        if population(site) != self.population:
            return False
        if self.family == ALL:
            return True
        if family(site.mark) != self.family:
            return False
        return (site.prescribed == self.heard) == (self.side == CORRECT_SIDE)

    def as_dict(self) -> dict:
        return {
            "population": self.population,
            "side": self.side,
            "family": self.family,
            "heard": self.heard,
            "label": self.label,
        }


@dataclass(frozen=True)
class RequiredCell:
    """One cell a gate must judge: frozen with the scorer, before any candidate (§1)."""

    gate: str
    rule: str
    cell: Cell
    #: ``always``, or ``when_supported`` for §5's "spurious haraka per supported row".
    when: str = "always"


def _heard_cells(side_: str) -> list[tuple[str, str]]:
    """(family, heard) of every per-mark and per-direction cell on one side."""
    marks = [(TASHKEEL, mark) for mark in HARAKAT]
    if side_ == MISTAKE_SIDE:
        marks.append((TASHKEEL, SUKUN))
    gemination = [(SHADDAH, state) for state in sorted(GEMINATION_STATES)]
    pairs = [(pair, letter) for pair in sorted(TARGET_PAIRS) for letter in sorted(pair.split("↔"))]
    return marks + gemination + pairs


def _required_cells() -> tuple[RequiredCell, ...]:
    def cells(gate, rule, side_, population_=HEADLINE, heard=None, when="always"):
        targets = heard if heard is not None else _heard_cells(side_)
        return [RequiredCell(gate, rule, Cell(population_, side_, f, h), when) for f, h in targets]

    harakat = [(TASHKEEL, mark) for mark in HARAKAT]
    sukun = [(TASHKEEL, SUKUN)]
    pooled = [(ALL, ALL)]
    probe, ship, bias = "§2 probe", "§3 ship", "§5 bias"
    return tuple(
        cells(probe, "sukun commit rate", CORRECT_SIDE, heard=sukun)
        + cells(probe, "sukun committed accuracy", CORRECT_SIDE, heard=sukun)
        + cells(probe, "commit rate", CORRECT_SIDE)
        + cells(probe, "committed accuracy", CORRECT_SIDE)
        + cells(probe, "committed accuracy", MISTAKE_SIDE)
        + cells(probe, "missed mistakes", MISTAKE_SIDE, heard=pooled + _heard_cells(MISTAKE_SIDE))
        + cells(probe, "silent corrections", MISTAKE_SIDE, heard=pooled)
        + cells(ship, "false flags (relative change)", CORRECT_SIDE, heard=pooled)
        + cells(ship, "coverage", CORRECT_SIDE, heard=pooled)
        + cells(ship, "missed mistakes", MISTAKE_SIDE, heard=pooled + _heard_cells(MISTAKE_SIDE))
        + cells(bias, "committed accuracy", CORRECT_SIDE, heard=harakat)
        + cells(bias, "committed accuracy", MISTAKE_SIDE)
        + cells(bias, "missed mistakes", MISTAKE_SIDE, heard=pooled)
        + cells(bias, "silent corrections", MISTAKE_SIDE, heard=pooled)
        + cells(bias, "spurious haraka", CORRECT_SIDE, heard=sukun, when="when_supported")
        + cells(bias, "spurious haraka", MISTAKE_SIDE, heard=sukun, when="when_supported")
        + cells(bias, "spurious haraka", CORRECT_SIDE, PAUSE, sukun, "when_supported")
    )


#: Every required cell of §2, §3 and §5. The §2 teacher-agreement guard is not a site cell:
#: it is read on the frozen ``decode_evalset`` dev manifest (``acceptance_stats``).
REQUIRED_CELLS: tuple[RequiredCell, ...] = _required_cells()


def item_key(site: TruthSite) -> str:
    """The decoded item a site sits on: its clip and sample span."""
    return f"{site.audio_filename}@{site.start_sample}:{site.end_sample}"


def site_weights(sites: Sequence[TruthSite]) -> dict[str, float]:
    """Stratum population / sites sampled in the stratum, unclear and pending included."""
    sampled = Counter(site.stratum for site in sites)
    return {site.site_id: site.stratum_population / sampled[site.stratum] for site in sites}


def physical_site(site: TruthSite) -> tuple:
    """The identity of the labelled position itself: the audio (its checksum once staged, its
    filename before), the item's span, the carrier and the mark under test."""
    audio = site.audio_sha256 or site.audio_filename
    return (audio, site.start_sample, site.end_sample, site.reference_index, site.mark)


def _precedence(site: TruthSite) -> int:
    """Which record of one physical site stands for it: the lowest value wins.

    A direct record (one that does not assume a competent reciter) always outranks a weak
    label. Among direct records a verdict outranks ``unclear`` (an adjudication that is
    complete but undecided, so the site is excluded rather than given the weak label's
    mark), which outranks ``pending`` (not adjudicated yet). Only among weak labels does a
    verdict outrank the rest.
    """
    rank = {UNCLEAR: 1, PENDING: 2}.get(site.heard, 0)
    return rank if not site.assumes_competent_reciter else 3 + (side(site) is None)


@dataclass(frozen=True)
class Reconciled:
    """One record per physical site, with every record's frozen sampling weight."""

    sites: list[TruthSite]
    weights: dict[str, float]
    #: Per physical site held by several records: the kept id and the dropped ids.
    merged: list[dict]


def reconcile(sites: Sequence[TruthSite]) -> Reconciled:
    """One record per physical site (§1's observation unit), before anything is scored.

    Weights are computed over every record first, so a stratum's sampled count stays the
    frozen one; a dropped duplicate does not inflate the kept record's weight. Among
    several records of one physical site the one of highest :func:`_precedence` stands for
    it: a direct verdict, then a direct ``unclear``, then a direct ``pending``, and a weak
    label only when no direct record exists; ties go to the lowest ``(source, site_id)``.
    Records that disagree on what the mushaf prescribes, or two verdicts of the winning rank
    that disagree on what was heard, are an error.
    """
    if len({site.site_id for site in sites}) != len(sites):
        raise ValueError("duplicate site ids across the truth-site files")
    weights = site_weights(sites)
    groups: dict[tuple, list[TruthSite]] = defaultdict(list)
    for site in sites:
        groups[physical_site(site)].append(site)
    kept_ids: set[str] = set()
    merged = []
    for key, records in groups.items():
        if len({r.prescribed for r in records}) > 1:
            raise ValueError(f"records {[r.site_id for r in records]} disagree on the prescribed mark")
        best = min(_precedence(r) for r in records)
        ranked = sorted(
            (r for r in records if _precedence(r) == best), key=lambda r: (r.source, r.site_id)
        )
        verdicts = {r.heard for r in ranked if side(r) is not None}
        if len(verdicts) > 1:
            raise ValueError(
                f"records {[r.site_id for r in ranked]} of one physical site disagree: {sorted(verdicts)}"
            )
        kept_ids.add(ranked[0].site_id)
        if len(records) > 1:
            dropped = sorted(r.site_id for r in records if r is not ranked[0])
            merged.append({"kept": ranked[0].site_id, "dropped": dropped})
    kept = [site for site in sites if site.site_id in kept_ids]
    return Reconciled(kept, {s.site_id: weights[s.site_id] for s in kept}, merged)


@dataclass(frozen=True)
class SiteRow:
    """One site with everything a rate reads: weight, reciter and each arm's outcome."""

    site: TruthSite
    weight: float
    reciter_id: int
    outcomes: Mapping[str, SiteOutcome]


def estimate(
    clusters: Sequence[int], terms: tuple[np.ndarray, np.ndarray], weights: np.ndarray,
    resample: Resample,
) -> dict:
    """A rate's point and its §1 interval, with the method that produced the interval."""
    num, den = terms
    point = ratio(num, den)
    return {
        "point": point if math.isfinite(point) else None,
        **ratio_interval(clusters, num, den, weights, resample).as_dict(),
        "num": float(num.sum()),
        "den": float(den.sum()),
    }


def _indicator(values) -> np.ndarray:
    return np.array(list(values), dtype=float)


def _rate_terms(
    rows: Sequence[SiteRow], arm: str, side_: str, config: MurajaConfig, sukun_cell: bool
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each rate's weighted per-site (numerator, denominator) for one arm on one side."""
    w = _indicator(row.weight for row in rows)
    outcomes = [row.outcomes[arm] for row in rows]
    commits = _indicator(o.commits for o in outcomes)
    correct = _indicator(o.correct(row.site) for o, row in zip(outcomes, rows))
    flagged = _indicator(o.flagged(row.site) for o, row in zip(outcomes, rows))
    grades = [grade(row.site, o, config) for o, row in zip(outcomes, rows)]
    wrong = _indicator(g == WRONG for g in grades)
    terms = {
        "commit_rate": (w * commits, w),
        "committed_accuracy": (w * correct, w * commits),
    }
    if side_ == CORRECT_SIDE:
        terms["decode_flagged"] = (w * flagged, w)
        terms["false_flags"] = (w * wrong, w)
        terms["coverage"] = (w * _indicator(g != NOT_GRADED for g in grades), w)
    else:
        silent = _indicator(o.committed == row.site.prescribed for o, row in zip(outcomes, rows))
        terms["missed_mistakes"] = (w * (1 - flagged), w)
        terms["silent_corrections"] = (w * silent, w)
        terms["muraja_missed"] = (w * (1 - wrong), w)
    if sukun_cell:
        haraka = _indicator(any(m in HARAKA_CHARS for m in o.marks) for o in outcomes)
        terms["spurious_haraka"] = (w * haraka, w)
    return terms


def exclusion_range(num: float, den: float, excluded: float) -> dict:
    """§1's sensitivity of the rate Σnum / Σden to the excluded sites that could belong to it:
    their weight counted as failures (worst) or as successes (best); ``None`` when undefined."""
    total = den + excluded
    if not total:
        return {"worst": None, "best": None}
    return {"worst": num / total, "best": (num + excluded) / total}


def _sensitivity(rows: Sequence[SiteRow], excluded_weights: Sequence[float], arm: str) -> dict:
    """Best / worst case of commit rate and committed accuracy with exclusions counted in."""
    w = sum(row.weight for row in rows)
    wc = sum(row.weight * row.outcomes[arm].commits for row in rows)
    wa = sum(row.weight * row.outcomes[arm].correct(row.site) for row in rows)
    wx = sum(excluded_weights)
    return {
        "commit_rate": exclusion_range(wc, w, wx),
        "committed_accuracy": exclusion_range(wa, wc, wx),
    }


def _score_cell(
    cell: Cell,
    rows: Sequence[SiteRow],
    excluded: Sequence[tuple[TruthSite, float]],
    arms: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
    config: MurajaConfig,
) -> dict:
    clusters = [row.reciter_id for row in rows]
    summary = {
        **cell.as_dict(),
        "sites": len(rows),
        "reciters": len(set(clusters)),
        "sparse": is_sparse(len(rows), len(set(clusters))),
        "provisional": cell.family == SHADDAH,
        "excluded": len(excluded),
    }
    if not rows:
        return summary
    resample = Resample.by_cluster(clusters)
    w = _indicator(row.weight for row in rows)
    sukun_cell = cell.family == TASHKEEL and cell.heard == SUKUN
    terms = {arm: _rate_terms(rows, arm, cell.side, config, sukun_cell) for arm in arms}
    summary["arms"] = {
        arm: {
            "rates": {name: estimate(clusters, t, w, resample) for name, t in terms[arm].items()},
            "sensitivity": _sensitivity(rows, [x for _, x in excluded], arm) if excluded else None,
        }
        for arm in arms
    }
    summary["differences"] = {}
    for arm, other in comparisons:
        differences = {}
        for name, arm_terms in terms[arm].items():
            points = [summary["arms"][a]["rates"][name]["point"] for a in (arm, other)]
            interval = difference_interval(clusters, arm_terms, terms[other][name], w, resample)
            differences[name] = {
                "point": None if None in points else points[0] - points[1],
                **interval.as_dict(),
            }
        summary["differences"][f"{arm} - {other}"] = differences
    return summary


def _allowance_view(
    population_: str,
    rows: Sequence[SiteRow],
    arms: Sequence[str],
    config: MurajaConfig,
) -> list[dict]:
    """Per allowance: false flags and missed mistakes on the sites it affects, on vs off (§4)."""
    view = []
    for allowance in ALLOWANCES:
        off = allowance.switch_off(config)
        entry = {"allowance": allowance.name, "population": population_, "sides": {}}
        for side_, rate in ((CORRECT_SIDE, "false_flags"), (MISTAKE_SIDE, "missed_mistakes")):
            affected = [r for r in rows if side(r.site) == side_ and allowance.affects(r.site)]
            clusters = [r.reciter_id for r in affected]
            block = {
                "sites": len(affected),
                "reciters": len(set(clusters)),
                "sparse": is_sparse(len(affected), len(set(clusters))),
            }
            if affected:
                resample = Resample.by_cluster(clusters)
                w = _indicator(r.weight for r in affected)
                for arm in arms:
                    block[arm] = {}
                    for state, cfg in (("on", config), ("off", off)):
                        # A false flag is a WRONG grade; a missed mistake is anything else.
                        wrong = _indicator(grade(r.site, r.outcomes[arm], cfg) == WRONG for r in affected)
                        num = w * (wrong if side_ == CORRECT_SIDE else 1 - wrong)
                        block[arm][state] = estimate(clusters, (num, w), w, resample)
            entry["sides"][rate] = block
        view.append(entry)
    return view


def outcomes_by_arm(
    sites: Sequence[TruthSite], decodes: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, SiteOutcome]]:
    """Every site's outcome under every arm. An arm missing an item's decode is an error:
    a site must never drop out of one arm of a paired comparison."""
    by_item: dict[str, list[TruthSite]] = defaultdict(list)
    for site in sites:
        by_item[item_key(site)].append(site)
    outcomes: dict[str, dict[str, SiteOutcome]] = {}
    for arm, items in decodes.items():
        missing = sorted(set(by_item) - set(items))
        if missing:
            raise ValueError(f"arm {arm} has no decode for {len(missing)} item(s), e.g. {missing[0]}")
        outcomes[arm] = {}
        for key, item_sites in by_item.items():
            outcomes[arm].update(item_outcomes(item_sites, items[key]))
    return outcomes


def _cells_of(rows: Sequence[SiteRow]) -> list[Cell]:
    observed = {
        Cell(population(r.site), side(r.site), family(r.site.mark), r.site.heard) for r in rows
    }
    pooled = {Cell(c.population, c.side, ALL, ALL) for c in observed}
    required = {required.cell for required in REQUIRED_CELLS}
    return sorted(observed | pooled | required)


def score(
    sites: Sequence[TruthSite],
    reciter_of: Mapping[str, int],
    decodes: Mapping[str, Mapping[str, str]],
    comparisons: Sequence[tuple[str, str]] = (),
    config: MurajaConfig = TODAY,
) -> dict:
    """The report: per-cell rates for every arm, paired differences, the allowance view,
    exclusions and required-cell support. ``decodes[arm][item_key]`` is a decode string,
    ``reciter_of[audio_filename]`` a canonical reciter id."""
    reconciled = reconcile(sites)
    sites, weights = reconciled.sites, reconciled.weights
    arms = sorted(decodes)
    outcomes = outcomes_by_arm(sites, decodes)
    rows = [
        SiteRow(
            site,
            weights[site.site_id],
            reciter_of[site.audio_filename],
            {arm: outcomes[arm][site.site_id] for arm in arms},
        )
        for site in sites
        if side(site) is not None
    ]
    excluded = [(site, weights[site.site_id]) for site in sites if side(site) is None]

    cells = []
    for cell in _cells_of(rows):
        members = [row for row in rows if cell.holds(row.site)]
        could = [(site, w) for site, w in excluded if cell.could_hold(site)]
        cells.append(_score_cell(cell, members, could, arms, comparisons, config))

    allowances = []
    for population_ in POPULATIONS:
        members = [row for row in rows if population(row.site) == population_]
        if members:
            allowances.extend(_allowance_view(population_, members, arms, config))

    support = {(c["population"], c["side"], c["family"], c["heard"]): c for c in cells}
    required = [
        {
            "gate": r.gate,
            "rule": r.rule,
            "when": r.when,
            **r.cell.as_dict(),
            "sites": support[(r.cell.population, r.cell.side, r.cell.family, r.cell.heard)]["sites"],
            "reciters": support[(r.cell.population, r.cell.side, r.cell.family, r.cell.heard)]["reciters"],
        }
        for r in REQUIRED_CELLS
    ]
    return {
        "bootstrap": {"replicates": REPLICATES, "seed": SEED, "cluster": "canonical reciter id"},
        "muraja_config": config.as_dict(),
        "arms": arms,
        "sites": _site_counts(sites, reciter_of),
        "physical_sites_merged": reconciled.merged,
        "exclusions": _exclusions(excluded),
        "cells": cells,
        "allowances": allowances,
        "required_cells": required,
    }


def _site_counts(sites: Sequence[TruthSite], reciter_of: Mapping[str, int]) -> list[dict]:
    """Sites, verdicts and reciters per population and stratum."""
    groups: dict[tuple[str, str], list[TruthSite]] = defaultdict(list)
    for site in sites:
        groups[(population(site), site.stratum)].append(site)
    return [
        {
            "population": population_,
            "stratum": stratum,
            "stratum_population": members[0].stratum_population,
            "sites": len(members),
            "with_verdict": sum(side(s) is not None for s in members),
            "items": len({item_key(s) for s in members}),
            "reciters": len({reciter_of[s.audio_filename] for s in members}),
        }
        for (population_, stratum), members in sorted(groups.items())
    ]


def _exclusions(excluded: Sequence[tuple[TruthSite, float]]) -> list[dict]:
    counts = Counter(
        (population(s), s.stratum, s.mark, s.prescribed, s.heard) for s, _ in excluded
    )
    return [
        dict(zip(("population", "stratum", "mark", "prescribed", "heard"), key), sites=n)
        for key, n in sorted(counts.items())
    ]


def power_inputs(
    sites: Sequence[TruthSite],
    reciter_of: Mapping[str, int],
    decodes: Mapping[str, Mapping[str, str]],
    config: MurajaConfig = TODAY,
) -> dict:
    """The §8 power simulation's baseline inputs: per-reciter sufficient statistics.

    For every arm, population, stratum, side and per-site cell: per canonical reciter, the
    site count and the weighted sums the simulation's outcome generator and its
    reciter-clustered baseline bootstrap need (Σw, ΣwC, ΣwA, ΣwF, Σw[Muraja WRONG],
    Σw[graded]). Only correct- and mistake-side sites with a verdict; ``excluded`` counts the
    rest per stratum.
    """
    reconciled = reconcile(sites)
    sites, weights = reconciled.sites, reconciled.weights
    outcomes = outcomes_by_arm(sites, decodes)
    cells: dict[tuple, dict[int, list[float]]] = defaultdict(dict)
    for arm in sorted(decodes):
        for site in sites:
            side_ = side(site)
            if side_ is None:
                continue
            outcome = outcomes[arm][site.site_id]
            verdict = grade(site, outcome, config)
            w = weights[site.site_id]
            key = (arm, population(site), site.stratum, side_, family(site.mark), site.heard)
            sums = cells[key].setdefault(reciter_of[site.audio_filename], [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for i, value in enumerate(
                (1, w, w * outcome.commits, w * outcome.correct(site), w * outcome.flagged(site),
                 w * (verdict == WRONG), w * (verdict != NOT_GRADED)),
            ):
                sums[i] += value
    fields = ("sites", "w", "w_commit", "w_correct", "w_flagged", "w_muraja_wrong", "w_graded")
    return {
        "muraja_config": config.as_dict(),
        "per_reciter_fields": list(fields),
        "cells": [
            {
                "arm": arm,
                "population": population_,
                "stratum": stratum,
                "side": side_,
                "family": family_,
                "heard": heard,
                "per_reciter": {str(r): sums for r, sums in sorted(per_reciter.items())},
            }
            for (arm, population_, stratum, side_, family_, heard), per_reciter in sorted(cells.items())
        ],
        "excluded": dict(sorted(Counter(s.stratum for s in sites if side(s) is None).items())),
    }


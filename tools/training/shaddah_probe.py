"""Shaddah probe (#86): where a decode emitted one consonant at a geminate, what did the model hear?

ADR-0011 needs three shaddah states -- held, not held, unsure -- and asks whether a decode
rule over today's encoding (a doubled consonant, which CTC can only emit with a blank
between the two) can supply them, or a gemination output class is needed. This module
measures, at every consonant site of a decoded segment, what the model's own output says
beyond its greedy decode. The pre-registered definitions, thresholds and verdict rules are
in ``docs/shaddah-probe-preregistration.md``; the constants below are those values.

**Sites.** A reference *site* is a maximal run of one consonant (:data:`CONSONANTS`) in a
segment's realized reference: a run of two or more is a **geminate**, a run of one a
**single**. The decode's tokens are aligned to the reference (Levenshtein,
:func:`training.edit_decomposition.align`, word spaces dropped) and a site's ``decoded``
count is how many tokens of its consonant land on the run, or are inserted against it. A
geminate with ``decoded == 1`` is **collapsed**: the decode emitted one consonant.

**What is measured at a site** (all from the same pass's log-posteriors):

* ``log_ratio`` -- the doubled transcript's posterior against the single one,
  ``log P(..c c..) - log P(..c..)`` (:func:`training.ctc_paths.ctc_log_likelihood`), with
  the rest of the decode held fixed. At a collapsed site the doubled transcript inserts a
  second ``c``; at a site decoded double it is the decode itself.
* ``second_peak`` -- the largest posterior the consonant gets on any frame of its
  **interval** outside its own decoded run: the sub-argmax second spike, if any.
* ``extra_location`` -- where the best alignment of the doubled transcript
  (:func:`training.ctc_paths.ctc_best_path`) puts the extra consonant relative to the
  decoded run: ``before``, ``after``, ``split`` (it carves the run, a mid-run dip) or
  ``elsewhere``.
* ``interval_frames`` -- the site's **interval**: the frames strictly between the decoded
  token before the site and the one after it, which hold every spike of the consonant. It
  is the held segment the stretch edit lengthens and the duration compared across sites.
  ``vv_context`` marks a site whose neighbouring tokens are both harakat, so intervals are
  compared in one phonetic context.

Everything here is numpy over posteriors and strings, so it is tested on synthetic
posteriors; the runner (:mod:`training.shaddah_probe_run`) does the I/O and the forwards.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tadabur.phoneme_vocab import PHONEME_CHAR_TO_ID, PHONEME_ID_TO_CHAR
from tadabur.truth_sites import CONSONANTS, HARAKA_CHARS
from training.ctc_paths import ctc_best_path, ctc_log_likelihood
from training.decoding import scan_ctc
from training.edit_decomposition import align

# --- Pre-registered values (docs/shaddah-probe-preregistration.md) ---------------------

#: One CTC frame of the Muaalem lattice: 25 frames per second of 16 kHz audio.
FRAME_SAMPLES = 640
FRAME_MS = 40

#: ``log_ratio`` at or above this calls the second consonant's mass **present**: the doubled
#: transcript has at least a tenth of the single one's posterior.
PRESENT_LOG_RATIO = math.log(0.1)
#: Below this the mass is **absent** (under a thousandth); between the two it is **weak**.
ABSENT_LOG_RATIO = math.log(0.001)
#: The curve ``log_ratio`` is reported on, in nats.
LOG_RATIO_GRID = (-30.0, -20.0, -15.0, -10.0, ABSENT_LOG_RATIO, -6.0, -5.0, -4.0, -3.0,
                  PRESENT_LOG_RATIO, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0)
#: A second spike is **present** when the consonant reaches this posterior off its run.
PRESENT_SECOND_PEAK = 0.1
SECOND_PEAK_GRID = (0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
#: The stretch factors: the issue's two, and 2.0 to see where the curve goes.
STRETCH_FACTORS = (1.25, 1.5, 2.0)
#: Intervals normalized by the median single-consonant interval of the same segment need
#: at least this many such singles.
MIN_SINGLES_FOR_RATE = 3
#: Reciter-clustered percentile bootstrap, as docs/acceptance-rules.md fixes for its gates.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261008

HARAKAT = frozenset(HARAKA_CHARS.values())

BEFORE, AFTER, SPLIT, ELSEWHERE = "before", "after", "split", "elsewhere"


# --- Tokens, sites and the alignment between them -------------------------------------


@dataclass(frozen=True)
class Token:
    """One decoded token and the frames of its greedy run (inclusive)."""

    char: str
    start: int
    end: int


def decode_tokens(log_posteriors: np.ndarray) -> list[Token]:
    """The greedy decode of one pass's log-posteriors, token by token."""
    return [
        Token(PHONEME_ID_TO_CHAR[seg.token_id], seg.start_step, seg.end_step)
        for seg in scan_ctc(np.asarray(log_posteriors).argmax(axis=1))
    ]


@dataclass(frozen=True)
class ReferenceSite:
    """A maximal run of one consonant in the raw reference: ``length >= 2`` is a geminate."""

    index: int  # the run's first character in the raw reference
    length: int
    consonant: str

    @property
    def is_geminate(self) -> bool:
        return self.length >= 2


def reference_sites(reference: str) -> list[ReferenceSite]:
    """Every consonant run of ``reference``, in order."""
    sites = []
    i = 0
    while i < len(reference):
        j = i
        while j < len(reference) and reference[j] == reference[i]:
            j += 1
        if reference[i] in CONSONANTS:
            sites.append(ReferenceSite(i, j - i, reference[i]))
        i = j
    return sites


def token_slots(reference: str, tokens: Sequence[Token]) -> list[tuple[int, bool]]:
    """Each token's place in the raw reference: ``(index, inserted)``.

    A matched or substituted token sits on the reference character ``index``; an inserted
    one sits just after it (``-1`` before the first). Word spaces are dropped before the
    alignment, as the decode has none, and indices are mapped back to the raw reference.
    """
    raw_index = [i for i, char in enumerate(reference) if char != " "]
    stripped = [reference[i] for i in raw_index]
    slots: list[tuple[int, bool] | None] = [None] * len(tokens)
    last = -1
    for op, ref, dec in align(stripped, [t.char for t in tokens]):
        if op in ("match", "substitution"):
            slots[dec] = (raw_index[ref], False)
        elif op == "insertion":
            slots[dec] = (last, True)
        if ref is not None:
            last = raw_index[ref]
    return slots  # type: ignore[return-value]


def site_tokens(
    site: ReferenceSite, tokens: Sequence[Token], slots: Sequence[tuple[int, bool]]
) -> list[int]:
    """The indices of the tokens of ``site``'s consonant on its run or inserted against it."""
    first, last = site.index, site.index + site.length - 1
    found = []
    for j, (token, (index, inserted)) in enumerate(zip(tokens, slots)):
        low = first - 1 if inserted else first
        if token.char == site.consonant and low <= index <= last:
            found.append(j)
    return found


# --- What the posteriors say at one site -----------------------------------------------


@dataclass(frozen=True)
class SiteMeasure:
    """One site of one decoded segment, measured as the module docstring defines."""

    reference_index: int
    length: int
    consonant: str
    decoded: int
    vv_context: bool | None
    interval_frames: int | None
    run_frames: int | None
    log_ratio: float | None
    second_peak: float | None
    extra_location: str | None

    @property
    def is_geminate(self) -> bool:
        return self.length >= 2


def _labels(tokens: Sequence[Token]) -> list[int]:
    return [PHONEME_CHAR_TO_ID[t.char] for t in tokens]


def interval_span(tokens: Sequence[Token], first: int, last: int) -> tuple[int, int] | None:
    """The interval of tokens ``first..last``: frames ``[start, end)`` strictly between the
    token before them and the token after, or ``None`` at either edge of the decode."""
    if first == 0 or last == len(tokens) - 1:
        return None
    return tokens[first - 1].end + 1, tokens[last + 1].start


def _extra_location(run: Token, spans: Sequence[tuple[int, int]]) -> str:
    """Where the best doubled alignment's extra consonant sits relative to ``run``."""
    def overlap(span: tuple[int, int]) -> int:
        return max(0, min(span[1], run.end) - max(span[0], run.start) + 1)

    first, second = spans
    if overlap(first) and overlap(second):
        return SPLIT
    if not overlap(first) and not overlap(second):
        return ELSEWHERE
    extra = second if overlap(first) else first
    return BEFORE if extra[1] < run.start else AFTER


def measure_site(
    site: ReferenceSite,
    tokens: Sequence[Token],
    slots: Sequence[tuple[int, bool]],
    log_posteriors: np.ndarray,
    decode_log_likelihood: float,
) -> SiteMeasure:
    """Measure ``site``. ``decode_log_likelihood`` is the decode's own CTC log-likelihood
    under ``log_posteriors``, shared by every site of the segment."""
    found = site_tokens(site, tokens, slots)
    labels = _labels(tokens)
    consonant_id = PHONEME_CHAR_TO_ID[site.consonant]
    vv_context = interval = run_frames = log_ratio = second_peak = location = None
    if found:
        first, last = found[0], found[-1]
        span = interval_span(tokens, first, last)
        if span is not None:
            interval = span[1] - span[0]
            vv_context = tokens[first - 1].char in HARAKAT and tokens[last + 1].char in HARAKAT
    if len(found) == 1:
        (j,) = found
        run = tokens[j]
        run_frames = run.end - run.start + 1
        doubled = labels[: j + 1] + [consonant_id] + labels[j + 1 :]
        log_ratio = ctc_log_likelihood(log_posteriors, doubled) - decode_log_likelihood
        if math.isfinite(log_ratio):
            location = _extra_location(run, ctc_best_path(log_posteriors, doubled)[j : j + 2])
        if span is not None:
            off_run = [t for t in range(*span) if not run.start <= t <= run.end]
            second_peak = float(np.exp(log_posteriors[off_run, consonant_id]).max()) if off_run else 0.0
    elif len(found) == 2 and found[1] == found[0] + 1:
        single = labels[: found[1]] + labels[found[1] + 1 :]
        log_ratio = decode_log_likelihood - ctc_log_likelihood(log_posteriors, single)
    return SiteMeasure(
        reference_index=site.index,
        length=site.length,
        consonant=site.consonant,
        decoded=len(found),
        vv_context=vv_context,
        interval_frames=interval,
        run_frames=run_frames,
        log_ratio=log_ratio,
        second_peak=second_peak,
        extra_location=location,
    )


@dataclass(frozen=True)
class SegmentMeasure:
    """One decoded segment: its decode and every requested site's measure."""

    decode: str
    sites: tuple[SiteMeasure, ...]


def measure_segment(
    reference: str, log_posteriors: np.ndarray, only: frozenset[int] | None = None
) -> SegmentMeasure:
    """Decode one pass greedily and measure every consonant site of ``reference`` (or only
    the sites whose run starts at an index in ``only``)."""
    tokens = decode_tokens(log_posteriors)
    slots = token_slots(reference, tokens)
    decode_log_likelihood = ctc_log_likelihood(log_posteriors, _labels(tokens))
    sites = tuple(
        measure_site(site, tokens, slots, log_posteriors, decode_log_likelihood)
        for site in reference_sites(reference)
        if only is None or site.index in only
    )
    return SegmentMeasure("".join(t.char for t in tokens), sites)


def site_interval(reference: str, log_posteriors: np.ndarray, index: int) -> tuple[int, int] | None:
    """The interval ``[start, end)`` in frames of the site whose run starts at ``index``."""
    tokens = decode_tokens(log_posteriors)
    slots = token_slots(reference, tokens)
    (site,) = (s for s in reference_sites(reference) if s.index == index)
    found = site_tokens(site, tokens, slots)
    return interval_span(tokens, found[0], found[-1]) if found else None


# --- Populations and statistics -------------------------------------------------------

#: A geminate the decode emitted once; a geminate of two decoded as two; a single
#: consonant decoded as one. Every other site (a consonant missing or substituted, a
#: geminate run of three or more decoded double, a single decoded double) is counted but
#: belongs to none of the three.
COLLAPSED, DOUBLE, SINGLE = "collapsed_geminate", "double_geminate", "single"
POPULATIONS = (COLLAPSED, DOUBLE, SINGLE)


@dataclass(frozen=True)
class Observation:
    """One measured site of one pool segment, with its sampling weight and its cluster.

    ``weight`` is the inverse of the clip's inclusion probability (#83).
    ``census_collapsed`` says whether the #83 census
    (:func:`tadabur.contrast_attribution.contrast_sites`) counts this run as a dropped
    gemination. ``rate_normalized`` is the interval over the median interval of the
    segment's single, harakah-flanked consonants (``None`` with fewer than
    :data:`MIN_SINGLES_FOR_RATE`).
    """

    segment: str
    reciter_id: int
    weight: float
    census_collapsed: bool
    rate_normalized: float | None
    site: SiteMeasure

    @property
    def population(self) -> str | None:
        site = self.site
        if site.is_geminate:
            if site.decoded == 1:
                return COLLAPSED
            return DOUBLE if site.decoded == 2 and site.length == 2 else None
        return SINGLE if site.decoded == 1 else None


def segment_observations(
    segment: str,
    reciter_id: int,
    inclusion_probability: float,
    measure: SegmentMeasure,
    census_dropped: frozenset[int],
) -> list[Observation]:
    """The observations of one segment. ``census_dropped`` holds the raw reference indices
    of the census's dropped gemination sites (each the first of its doubled pair)."""
    single_intervals = [
        s.interval_frames
        for s in measure.sites
        if not s.is_geminate and s.decoded == 1 and s.vv_context
    ]
    rate = float(np.median(single_intervals)) if len(single_intervals) >= MIN_SINGLES_FOR_RATE else None
    return [
        Observation(
            segment=segment,
            reciter_id=reciter_id,
            weight=1.0 / inclusion_probability,
            census_collapsed=any(
                site.reference_index <= i < site.reference_index + site.length for i in census_dropped
            ),
            rate_normalized=(
                site.interval_frames / rate
                if rate and site.vv_context and site.interval_frames is not None
                else None
            ),
            site=site,
        )
        for site in measure.sites
    ]


def share(observations: Sequence, predicate) -> dict:
    """The share of ``observations`` (anything with a ``weight``) where ``predicate`` holds:
    unweighted and weighted."""
    if not observations:
        return {"n": 0, "unweighted": None, "weighted": None}
    hits = np.array([bool(predicate(o)) for o in observations])
    weights = np.array([o.weight for o in observations])
    return {
        "n": len(observations),
        "unweighted": float(hits.mean()),
        "weighted": float((weights * hits).sum() / weights.sum()),
    }


def clustered_interval(
    observations: Sequence, value, resamples: int = BOOTSTRAP_RESAMPLES
) -> list[float] | None:
    """A reciter-clustered percentile 95% interval for the weighted mean of ``value``.

    ``value`` maps an observation (anything with ``weight`` and ``reciter_id``) to a number
    or a bool (a share). Reciters are resampled with replacement; a replicate's statistic
    is the weighted mean over the sites of the reciters it drew, so a replicate with no
    sites cannot occur.
    """
    if not observations:
        return None
    reciters = sorted({o.reciter_id for o in observations})
    position = {r: i for i, r in enumerate(reciters)}
    numerator = np.zeros(len(reciters))
    denominator = np.zeros(len(reciters))
    for o in observations:
        numerator[position[o.reciter_id]] += o.weight * float(value(o))
        denominator[position[o.reciter_id]] += o.weight
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(reciters), size=(resamples, len(reciters)))
    ratios = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
    return [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]


def weighted_quantiles(values: Sequence[float], weights: Sequence[float], qs: Sequence[float]):
    """Weighted quantiles: the smallest value whose cumulative weight share reaches ``q``."""
    if not len(values):
        return None
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    cumulative /= cumulative[-1]
    return [float(sorted_values[min(np.searchsorted(cumulative, q), len(values) - 1)]) for q in qs]


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _distribution(observations: Sequence[Observation], value) -> dict:
    kept = [(value(o), o.weight) for o in observations if value(o) is not None]
    return {
        "n": len(kept),
        "quantiles": dict(
            zip(
                [f"p{int(q * 100)}" for q in QUANTILES],
                weighted_quantiles([v for v, _ in kept], [w for _, w in kept], QUANTILES) or [],
            )
        ),
    }


def _mass_state(o: Observation) -> str:
    """The pre-registered three-way reading of ``log_ratio``."""
    if o.site.log_ratio >= PRESENT_LOG_RATIO:
        return "present"
    return "weak" if o.site.log_ratio >= ABSENT_LOG_RATIO else "absent"


def rule_state(o: Observation) -> str:
    """The candidate decode rule's shaddah state at a site: a decoded double is ``held``;
    a decoded single is ``held`` / ``unsure`` / ``not_held`` as its mass is present / weak
    / absent."""
    if o.site.decoded >= 2:
        return "held"
    return {"present": "held", "weak": "unsure", "absent": "not_held"}[_mass_state(o)]


def mass_report(observations: Sequence[Observation]) -> dict:
    """Q1 for one population: the ``log_ratio`` and second-spike curves, the pre-registered
    states and where the extra consonant sits."""
    scored = [o for o in observations if o.site.log_ratio is not None]
    report = {
        "sites": len(observations),
        "reciters": len({o.reciter_id for o in observations}),
        "scored": len(scored),
        "log_ratio_states": {
            state: share(scored, lambda o, s=state: _mass_state(o) == s)
            for state in ("present", "weak", "absent")
        },
        "present_interval_95": clustered_interval(scored, lambda o: _mass_state(o) == "present"),
        "rule_states": {
            state: share(scored, lambda o, s=state: rule_state(o) == s)
            for state in ("held", "unsure", "not_held")
        },
        "log_ratio_curve": [
            {"threshold": g, "share_at_or_above": share(scored, lambda o, g=g: o.site.log_ratio >= g)["weighted"]}
            for g in LOG_RATIO_GRID
        ],
        "log_ratio": _distribution(scored, lambda o: o.site.log_ratio),
    }
    peaks = [o for o in observations if o.site.second_peak is not None]
    if peaks:
        report["second_peak_curve"] = [
            {"threshold": g, "share_at_or_above": share(peaks, lambda o, g=g: o.site.second_peak >= g)["weighted"]}
            for g in SECOND_PEAK_GRID
        ]
        report["second_peak_present"] = share(peaks, lambda o: o.site.second_peak >= PRESENT_SECOND_PEAK)
        located = [o for o in observations if o.site.extra_location is not None]
        report["extra_location"] = {
            where: share(located, lambda o, w=where: o.site.extra_location == w)
            for where in (BEFORE, AFTER, SPLIT, ELSEWHERE)
        }
        report["run_frames"] = _distribution(observations, lambda o: o.site.run_frames)
    return report


def duration_report(by_population: dict[str, list[Observation]]) -> dict:
    """Q3: the interval at harakah-flanked sites, raw (ms) and rate-normalized, per
    population, and the share of collapsed geminates whose normalized interval is
    geminate-like (at least midway between the double-geminate and single medians)."""
    flanked = {name: [o for o in obs if o.site.vv_context] for name, obs in by_population.items()}
    report: dict = {
        name: {
            "interval_ms": _distribution(obs, lambda o: o.site.interval_frames * FRAME_MS),
            "rate_normalized": _distribution(obs, lambda o: o.rate_normalized),
            "rate_normalized_curve": [
                {"at_least": r, "share": share(
                    [o for o in obs if o.rate_normalized is not None],
                    lambda o, r=r: o.rate_normalized >= r,
                )["weighted"]}
                for r in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
            ],
        }
        for name, obs in flanked.items()
    }
    medians = [report[name]["rate_normalized"]["quantiles"].get("p50") for name in (DOUBLE, SINGLE)]
    if None in medians:
        report["geminate_like"] = None
        return report
    midpoint = (medians[0] + medians[1]) / 2
    normalized = [o for o in flanked[COLLAPSED] if o.rate_normalized is not None]
    report["geminate_like"] = {
        "midpoint": midpoint,
        **share(normalized, lambda o: o.rate_normalized >= midpoint),
        "interval_95": clustered_interval(normalized, lambda o: o.rate_normalized >= midpoint),
    }
    return report


# --- The stretch test -----------------------------------------------------------------


@dataclass(frozen=True)
class StretchTrial:
    """One site re-measured after its interval was stretched by ``factor``
    (:func:`tadabur.time_stretch.stretch_span`), and on the equal-length decoy: the same
    unedited audio with as many zero samples appended. ``*_changes`` is the Levenshtein
    distance of that pass's decode from the unedited one."""

    segment: str
    reciter_id: int
    weight: float
    population: str  # COLLAPSED, or SINGLE for a collapsed site's matched control
    reference_index: int
    factor: float
    edited: SiteMeasure
    decoy: SiteMeasure
    edited_changes: int
    decoy_changes: int


def match_controls(observations: Sequence[Observation]) -> list[tuple[Observation, Observation | None]]:
    """Each collapsed geminate with an interval, paired with one single site of the same
    consonant: in the same segment if possible (the nearest), else in another segment of
    the same clip. A control is used once; a collapsed site without one is paired with
    ``None``."""
    def clip(o: Observation) -> str:
        return o.segment.rsplit("#", 1)[0]

    def stretchable(o: Observation) -> bool:
        return bool(o.site.interval_frames)

    singles: dict[tuple[str, str], list[Observation]] = {}
    for o in observations:
        if o.population == SINGLE and stretchable(o):
            singles.setdefault((clip(o), o.site.consonant), []).append(o)
    used: set[tuple[str, int]] = set()
    pairs = []
    for o in sorted(observations, key=lambda o: (o.segment, o.site.reference_index)):
        if o.population != COLLAPSED or not stretchable(o):
            continue
        candidates = [
            c for c in singles.get((clip(o), o.site.consonant), [])
            if (c.segment, c.site.reference_index) not in used
        ]
        candidates.sort(key=lambda c: (
            c.segment != o.segment,
            abs(c.site.reference_index - o.site.reference_index) if c.segment == o.segment else 0,
            c.segment,
            c.site.reference_index,
        ))
        control = candidates[0] if candidates else None
        if control is not None:
            used.add((control.segment, control.site.reference_index))
        pairs.append((o, control))
    return pairs


def _flipped(site: SiteMeasure) -> bool:
    return site.decoded >= 2


def stretch_report(trials: Sequence[StretchTrial]) -> dict:
    """Q2: per factor and population, how often the site decodes double after the stretch
    and on the decoy, the net of the two (with a reciter-clustered interval), and the
    ``log_ratio`` curve along the factors."""
    report = {}
    for factor in STRETCH_FACTORS:
        report[str(factor)] = {}
        for population in (COLLAPSED, SINGLE):
            rows = [t for t in trials if t.factor == factor and t.population == population]
            edited = share(rows, lambda t: _flipped(t.edited))
            decoy = share(rows, lambda t: _flipped(t.decoy))
            report[str(factor)][population] = {
                "trials": len(rows),
                "double_after_stretch": edited,
                "double_on_decoy": decoy,
                "net": {
                    "weighted": (
                        None if not rows else edited["weighted"] - decoy["weighted"]
                    ),
                    "interval_95": clustered_interval(
                        rows, lambda t: _flipped(t.edited) - _flipped(t.decoy)
                    ),
                },
                "log_ratio_after_stretch": _distribution(rows, lambda t: t.edited.log_ratio),
                "log_ratio_on_decoy": _distribution(rows, lambda t: t.decoy.log_ratio),
                "decode_changes_after_stretch": _distribution(rows, lambda t: t.edited_changes),
                "decode_changes_on_decoy": _distribution(rows, lambda t: t.decoy_changes),
            }
    return report


# --- The verdict ----------------------------------------------------------------------

#: Pre-registered verdict thresholds (docs/shaddah-probe-preregistration.md, "Verdict").
GEMINATE_LIKE_MAJORITY = 0.5
EVIDENCE_GAP = 0.10
RULE_MAX_SINGLE_HELD = 0.02
RULE_MAX_SINGLE_HELD_OR_UNSURE = 0.10


def verdict(mass: dict, durations: dict, stretch: dict) -> dict:
    """The pre-registered representation-vs-data call for one model, with its evidence."""
    geminate_like = (durations.get("geminate_like") or {}).get("weighted")
    present = {
        pop: mass[pop]["log_ratio_states"]["present"]["weighted"] for pop in (COLLAPSED, SINGLE)
    }
    at = stretch[str(1.5)]
    nets = [at[pop]["net"]["weighted"] for pop in (COLLAPSED, SINGLE)]
    rule = {pop: {k: v["weighted"] for k, v in mass[pop]["rule_states"].items()} for pop in (COLLAPSED, SINGLE)}
    if None in (geminate_like, *present.values(), *nets):
        return {"call": "insufficient_evidence", "posterior_rule_looks_viable": None}
    mass_gap = present[COLLAPSED] - present[SINGLE]
    stretch_gap = nets[0] - nets[1]
    rule_viable = (
        rule[SINGLE]["held"] <= RULE_MAX_SINGLE_HELD
        and rule[SINGLE]["held"] + rule[SINGLE]["unsure"] <= RULE_MAX_SINGLE_HELD_OR_UNSURE
        and rule[COLLAPSED]["held"] - rule[SINGLE]["held"] >= EVIDENCE_GAP
    )
    if geminate_like is not None and geminate_like >= GEMINATE_LIKE_MAJORITY:
        call = "representation"
    elif mass_gap < EVIDENCE_GAP and stretch_gap < EVIDENCE_GAP:
        call = "data"
    else:
        call = "mixed"
    return {
        "call": call,
        "geminate_like": geminate_like,
        "mass_present": present,
        "mass_gap": mass_gap,
        "stretch_net_gap_at_1.5": stretch_gap,
        "rule_states": rule,
        "posterior_rule_looks_viable": rule_viable,
    }


# --- One model's report ---------------------------------------------------------------

#: Consonants with at least this many collapsed sites get their own Q1 / Q3 row.
MIN_SITES_PER_CONSONANT = 10


def _decoded_bucket(decoded: int) -> str:
    return str(decoded) if decoded < 3 else "3+"


def collapsed_rows(observations: Sequence[Observation], trials: Sequence[StretchTrial]) -> list[dict]:
    """Every collapsed geminate as one row, with its stretch outcomes: the per-site record
    the decision issue (#92) can re-read without re-running anything."""
    outcomes: dict[tuple[str, int], dict] = {}
    for t in trials:
        if t.population == COLLAPSED:
            outcomes.setdefault((t.segment, t.reference_index), {})[str(t.factor)] = {
                "edited_decoded": t.edited.decoded,
                "edited_log_ratio": t.edited.log_ratio,
                "decoy_decoded": t.decoy.decoded,
                "decoy_log_ratio": t.decoy.log_ratio,
            }
    rows = []
    for o in sorted(observations, key=lambda o: (o.segment, o.site.reference_index)):
        if o.population != COLLAPSED:
            continue
        site = o.site
        rows.append({
            "segment": o.segment,
            "reciter_id": o.reciter_id,
            "weight": o.weight,
            "reference_index": site.reference_index,
            "run_length": site.length,
            "consonant": site.consonant,
            "census_collapsed": o.census_collapsed,
            "vv_context": site.vv_context,
            "interval_ms": None if site.interval_frames is None else site.interval_frames * FRAME_MS,
            "rate_normalized": o.rate_normalized,
            "log_ratio": site.log_ratio,
            "second_peak": site.second_peak,
            "extra_location": site.extra_location,
            "run_frames": site.run_frames,
            "rule_state": None if site.log_ratio is None else rule_state(o),
            "stretch": outcomes.get((o.segment, site.reference_index), {}),
        })
    return rows


def model_report(observations: Sequence[Observation], trials: Sequence[StretchTrial]) -> dict:
    """Q1-Q3 and the verdict for one model's observations and stretch trials."""
    from collections import Counter

    by_population = {p: [o for o in observations if o.population == p] for p in POPULATIONS}
    mass = {p: mass_report(obs) for p, obs in by_population.items()}
    durations = duration_report(by_population)
    stretch = stretch_report(trials)
    census = Counter(
        f"decoded_{_decoded_bucket(o.site.decoded)}|census_collapsed_{o.census_collapsed}"
        for o in observations
        if o.site.is_geminate
    )
    per_consonant = Counter(o.site.consonant for o in by_population[COLLAPSED])
    by_consonant = {
        consonant: {
            p: {
                "sites": len(obs := [o for o in by_population[p] if o.site.consonant == consonant]),
                "mass_present": share(
                    [o for o in obs if o.site.log_ratio is not None],
                    lambda o: _mass_state(o) == "present",
                ),
                "interval_ms": _distribution(
                    [o for o in obs if o.site.vv_context], lambda o: o.site.interval_frames * FRAME_MS
                ),
            }
            for p in POPULATIONS
        }
        for consonant, n in sorted(per_consonant.items())
        if n >= MIN_SITES_PER_CONSONANT
    }
    return {
        "sites": {
            "observed": len(observations),
            "geminate_runs": sum(o.site.is_geminate for o in observations),
            "by_population": {p: len(obs) for p, obs in by_population.items()},
            "geminates_by_decode_and_census": dict(sorted(census.items())),
            "single_decoded_double": sum(
                not o.site.is_geminate and o.site.decoded == 2 for o in observations
            ),
        },
        "mass": mass,
        "by_consonant": by_consonant,
        "durations": durations,
        "stretch": stretch,
        "verdict": verdict(mass, durations, stretch),
        "collapsed_sites": collapsed_rows(observations, trials),
    }

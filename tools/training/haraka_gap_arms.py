"""The arms and sites of the haraka-gap diagnosis (#85): what is decoded, and how it is scored.

The base teacher matches **0.9617** of reference harakat on whole waqf segments (ADR-0005)
but only **0.841** on the <=5 s word-snapped windows of ADR-0007. This tool decodes the
same audio four ways, with the base teacher and ``h448``, and says where the difference
comes from. Every number it produces is **agreement with the mushaf** (the realized
reference), not truth: a haraka the model leaves empty may be one the reciter did not
say. It recommends nothing about the product.

The four arms
-------------

All arms decode the mining pool (:mod:`tadabur.mining_pool`) clips that
:func:`training.windowed_labels.build_clip_windows` accepts, from the staged WAV, through
:class:`training.decoding.Decoder` with one precision and batch size
(:data:`training.haraka_gap.WEIGHTS_DTYPE`, ``DECODE_BATCH_SIZE``), and score them with
:func:`training.tashkeel_eval.vowel_sites`:

* ``segment`` -- each kept waqf segment decoded whole against its realized reference
  (ADR-0005's unit);
* ``window`` -- the ADR-0007 windows: a 5 s grid with a 4 s hop over the recitation,
  each snapped inward to whole words, decoded whole against its sliced label;
* ``stream_b0`` / ``stream_b1`` -- the recitation span streamed through the deployed
  protocol (5 s windows, 1 s hop, commit block ``b``, startup rule, tail flush) and scored
  against the concatenated segment references.

**Sites are fixed across arms and models.** A site is one fatha, damma or kasra of a kept
segment's realized reference, keyed by ``(clip, segment, reference index)``; every arm maps
its own reference string back onto those keys. The population is every haraka of an
eligible clip that at least one window contains (a word longer than the grid's 1 s
overlap fits in no window), so every arm scores the same sites over the same denominator.
A site seen by two overlapping windows carries a share of 1/2 in each (``window``), so it
still counts once; ``window_occ`` counts every occurrence once, which is how ADR-0007
computed 0.841. ``segment_all`` scores every kept segment of the whole pool, its sites
classed ``windowed``, ``word_in_no_window`` or ``clip_not_windowed``, to measure what
restricting to the fixed population itself moves.

Weights are the clip's 1 / ``inclusion_probability``: the pool over-samples clips with
consonant and gemination events, so an unweighted rate describes the pool, a weighted one
the drawn reciters' frame (:mod:`training.haraka_gap_report` has the statistics).

Classes
-------

A **window** occurrence is classified by where its word sits in the window
(:func:`window_position`): the only word, the first or last word -- each split by whether
that edge is also a segment edge (the audio there is a real pause) or cuts through a
segment (wasl audio continues past the edge) -- or an interior word.

A **stream** site is classified by when it is spoken (:func:`stream_region`): committed by
the first window (startup), in the last window's flushed blocks, past the last full window
(never decoded), or in steady state, where the fifth of a second either side of a commit
boundary is the **seam** and the rest the block middle. The time is the
one the **base teacher's whole-segment decode** places the haraka (or its nearest aligned
neighbour in the same word) at, so it is the same for both models and every arm; a word
the base decode aligned nothing in falls back to interpolating its forced-alignment span.

The decode, the scoring run and the report are driven by :mod:`training.haraka_gap`.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import NamedTuple

from tadabur.mining_pool import PoolClip, PoolSegment, load_manifest, segment_key
from tadabur.smith_waterman import smith_waterman
from training.tashkeel_eval import (
    MATCHED,
    OMITTED,
    SPURIOUS,
    SWAPPED,
    UNANCHORED,
    UNANCHORED_WRONG,
    vowel_sites,
)
from training.tashkeel_worklist import VOWEL_NAMES
from training.windowed_labels import WindowLabel, build_clip_windows
from training.windowed_labels import Segment as LabelSegment
from training.windowing import TARGET_SAMPLE_RATE, WindowContract, recitation_window_span

#: The two commit blocks compared: the deployed b=0 and b=1 (ADR-0010).
BLOCKS = (0, 1)

#: 40 ms CTC timesteps, and the width of the seam band either side of a commit boundary.
STEP_SECONDS = 0.04
SEAM_SECONDS = 0.2
WINDOW_SECONDS = 5

# --- Arms, classes, outcomes --------------------------------------------------------

SEGMENT_ALL = "segment_all"
SEGMENT = "segment"
WINDOW = "window"
WINDOW_OCC = "window_occ"
STREAM = {block: f"stream_b{block}" for block in BLOCKS}
#: The arms every model is reported on, in report order.
ARMS = (SEGMENT, WINDOW, WINDOW_OCC, *STREAM.values())

WHOLE = "whole"
#: ``segment_all`` sites by why they are, or are not, in the fixed population.
CLIP_NOT_WINDOWED = "clip_not_windowed"
WORD_IN_NO_WINDOW = "word_in_no_window"
WINDOWED = "windowed"
SEGMENT_ALL_CLASSES = (CLIP_NOT_WINDOWED, WORD_IN_NO_WINDOW, WINDOWED)
INTERIOR = "interior"
ONLY_WORD = "only_word"
FIRST_AT_PAUSE = "first@segment_start"
FIRST_MID_SEGMENT = "first@mid_segment"
LAST_AT_PAUSE = "last@segment_end"
LAST_MID_SEGMENT = "last@mid_segment"
WINDOW_CLASSES = (
    FIRST_AT_PAUSE, FIRST_MID_SEGMENT, LAST_AT_PAUSE, LAST_MID_SEGMENT, ONLY_WORD, INTERIOR
)
EDGE_CLASSES = frozenset(WINDOW_CLASSES) - {INTERIOR}

STARTUP = "startup"
SEAM_START = "seam:block_start"
BLOCK_MID = "block_mid"
SEAM_END = "seam:block_end"
FLUSH = "flush"
UNDECODED = "undecoded"
STREAM_CLASSES = (STARTUP, SEAM_START, BLOCK_MID, SEAM_END, FLUSH, UNDECODED)
SEAM_CLASSES = frozenset({SEAM_START, SEAM_END})

#: How a reference haraka fared, grouped as the report reads it. ``empty`` is the empty
#: tashkeel slot (``omitted``); ``unanchored`` pools both carrier-missed outcomes.
EMPTY = "empty"
OUTCOME_GROUP = {
    MATCHED: MATCHED,
    OMITTED: EMPTY,
    SWAPPED: SWAPPED,
    UNANCHORED: "unanchored",
    UNANCHORED_WRONG: "unanchored",
}
OUTCOME_GROUPS = (MATCHED, EMPTY, SWAPPED, "unanchored")
HARAKAT = ("fatha", "damma", "kasra")
ALL_HARAKAT = "all"

#: Where a haraka sits among its word's harakat: the window edges cut words, so the first
#: haraka of a window's first word and the last of its last word are where an edge acts.
FIRST_IN_WORD, INNER_IN_WORD, LAST_IN_WORD, SOLE_IN_WORD = "first", "inner", "last", "sole"
WORD_POSITIONS = (FIRST_IN_WORD, INNER_IN_WORD, LAST_IN_WORD, SOLE_IN_WORD)

# Where a site's time came from (:func:`site_times`).
TIME_OWN, TIME_NEIGHBOUR, TIME_WORD = "own_token", "neighbour_token", "word_interpolation"


class Site(NamedTuple):
    """One reference haraka: the same key in every arm and for every model."""

    clip: str
    segment_index: int
    ref_index: int


# --- The plan: what is decoded --------------------------------------------------------


@dataclass(frozen=True)
class ClipPlan:
    """One pool clip's kept segments and, when it is eligible, its windows and stream span.

    ``windows`` is empty and ``exclusion`` names the reason for a clip the ADR-0007 label
    build refuses; such a clip enters only the ``segment_all`` arm.
    """

    clip: PoolClip
    segments: tuple[PoolSegment, ...]
    windows: tuple[WindowLabel, ...]
    exclusion: str | None
    recitation_start_sample: int
    recitation_num_samples: int

    @property
    def eligible(self) -> bool:
        return self.exclusion is None

    @property
    def stream_key(self) -> str:
        return self.clip.audio_filename


def plan_pool() -> list[ClipPlan]:
    """Every committed pool clip, planned (:func:`plan_clip`)."""
    return [plan_clip(clip) for clip in load_manifest()]


def window_key(audio_filename: str, window_index: int) -> str:
    return f"{audio_filename}@w{window_index}"


def stream_unit_key(audio_filename: str, block: int) -> str:
    return f"{audio_filename}~b{block}"


def plan_clip(clip: PoolClip) -> ClipPlan:
    """The clip's kept segments and the windows ``training.windowed_labels`` cuts from it.

    The windows are built by the label build itself, from the pool's segmentation, so they
    are exactly ADR-0007's geometry: the recitation grid, word snapping, and every
    clip-level exclusion (re-reads, dropped segments, over-long clips).
    """
    from tadabur.clip_status import ClipStatus

    kept = tuple(sorted((s for s in clip.segments if s.kept), key=lambda s: s.segment_index))
    status = ClipStatus(
        audio_filename=clip.audio_filename,
        surah_ayah=clip.surah_ayah,
        reciter_id=clip.reciter_id,
        n_words=clip.n_words,
        duration_s=clip.recitation_end_s,
        recitation_start_s=clip.recitation_start_s,
        recitation_end_s=clip.recitation_end_s,
        skip_reason=clip.skip_reason,
        re_reads=clip.re_reads,
        recited_words=clip.recited_words,
        word_times=clip.word_times,
    )
    label_segments = [
        LabelSegment(
            clip_audio_filename=clip.audio_filename,
            surah_ayah=clip.surah_ayah,
            reciter_id=clip.reciter_id,
            segment_index=seg.segment_index,
            word_start=seg.word_start,
            word_end=seg.word_end,
            start_s=seg.start_sample / TARGET_SAMPLE_RATE,
            end_s=seg.end_sample / TARGET_SAMPLE_RATE,
            label_phonemes=seg.reference,
            label_word_offsets=seg.raw_word_offsets,
        )
        for seg in kept
    ]
    windows, exclusion = build_clip_windows(label_segments, status, WindowContract())
    start, num = recitation_window_span(clip.recitation_start_s, clip.recitation_end_s)
    return ClipPlan(clip, kept, tuple(windows), exclusion, start, num)


# --- Classes --------------------------------------------------------------------------


def word_of(segment: PoolSegment, ref_index: int) -> int:
    """The ayah word index ``ref_index`` of the segment's reference falls in."""
    return segment.word_start + bisect_right(segment.raw_word_offsets, ref_index) - 1


def word_position(segment: PoolSegment, ref_index: int) -> str:
    """Whether the haraka at ``ref_index`` is its word's first, last, only or an inner one."""
    k = bisect_right(segment.raw_word_offsets, ref_index) - 1
    lo, hi = segment.raw_word_offsets[k], segment.raw_word_offsets[k + 1]
    harakat = [i for i in range(lo, hi) if segment.reference[i] in VOWEL_NAMES]
    if len(harakat) == 1:
        return SOLE_IN_WORD
    if ref_index == harakat[0]:
        return FIRST_IN_WORD
    return LAST_IN_WORD if ref_index == harakat[-1] else INNER_IN_WORD


def window_position(word: int, window: WindowLabel, segment: PoolSegment) -> str:
    """Where ``word`` (of ``segment``) sits in ``window``'s whole-word range.

    An edge word is split by whether the window edge is also a segment edge -- the audio
    beside it is a real pause -- or cuts through the segment, so speech continues just
    past the window's audio.
    """
    if not window.word_start <= word < window.word_end:
        raise ValueError(f"word {word} is outside window [{window.word_start}, {window.word_end})")
    if window.word_end - window.word_start == 1:
        return ONLY_WORD
    if word == window.word_start:
        return FIRST_AT_PAUSE if word == segment.word_start else FIRST_MID_SEGMENT
    if word == window.word_end - 1:
        return LAST_AT_PAUSE if word == segment.word_end - 1 else LAST_MID_SEGMENT
    return INTERIOR


def stream_region(seconds: float, num_samples: int, block: int) -> str:
    """Which part of the streaming protocol commits audio at ``seconds`` into the stream.

    Mirrors :func:`training.decoding.commit_bounds` on the time axis: window ``w`` starts at
    second ``w`` and commits ``[w+b, w+b+1)``. **Startup** is everything the first window
    commits, ``[0, b+1)``: the blocks before ``b`` by the startup rule and block ``b``
    itself, which at b=0 is the recitation's first second, decoded with no audio before it
    at all. **Flush** is what the last window commits after its block ``b``. A stream at
    least one window long leaves the audio past its last full window **undecoded**. The
    rest is steady state, split into the seam bands either side of a commit boundary and
    the block middle.
    """
    from training.decoding import WINDOW_SAMPLES, clip_windows

    last = len(clip_windows(num_samples)) - 1
    if num_samples >= WINDOW_SAMPLES and seconds >= last + WINDOW_SECONDS:
        return UNDECODED
    if seconds >= last + block + 1:
        return FLUSH
    if seconds < block + 1:
        return STARTUP
    into_block = seconds - math.floor(seconds)
    if into_block < SEAM_SECONDS:
        return SEAM_START
    if into_block >= 1 - SEAM_SECONDS:
        return SEAM_END
    return BLOCK_MID


# --- Units: one decoded string and the sites its reference holds ------------------------


@dataclass(frozen=True)
class Unit:
    """One string an arm decodes: its reference, and per reference index the site there
    and the site's class in this unit."""

    key: str
    reciter_id: int
    weight: float
    reference: str
    sites: Mapping[int, Site]
    classes: Mapping[int, str]


def haraka_sites(segment: PoolSegment) -> list[int]:
    """The reference indices of the segment's fatha, damma and kasra."""
    return [i for i, char in enumerate(segment.reference) if char in VOWEL_NAMES]


def segment_unit(plan: ClipPlan, segment: PoolSegment, population: frozenset[Site]) -> Unit:
    """A kept segment decoded whole, every haraka classed by whether it is in
    ``population`` and, if not, why: its clip has no windows, or no window holds its word."""
    sites = {
        r: Site(plan.clip.audio_filename, segment.segment_index, r)
        for r in haraka_sites(segment)
    }
    outside = WORD_IN_NO_WINDOW if plan.eligible else CLIP_NOT_WINDOWED
    return Unit(
        key=segment_key(plan.clip.audio_filename, segment.segment_index),
        reciter_id=plan.clip.reciter_id,
        weight=1 / plan.clip.inclusion_probability,
        reference=segment.reference,
        sites=sites,
        classes={r: WINDOWED if site in population else outside for r, site in sites.items()},
    )


def windowed_only(unit: Unit) -> Unit:
    """``unit`` restricted to its ``windowed`` sites: the ``segment`` arm's unit."""
    sites = {r: site for r, site in unit.sites.items() if unit.classes[r] == WINDOWED}
    return replace(unit, sites=sites, classes={r: WHOLE for r in sites})


def window_unit(plan: ClipPlan, window: WindowLabel) -> Unit:
    """One ADR-0007 window: its label rebuilt from segment slices, so every label index
    maps back to the segment site it was sliced from."""
    pieces: list[str] = []
    sites: dict[int, Site] = {}
    classes: dict[int, str] = {}
    offset = 0
    for seg in plan.segments:
        lo, hi = max(seg.word_start, window.word_start), min(seg.word_end, window.word_end)
        if lo >= hi:
            continue
        begin = seg.raw_word_offsets[lo - seg.word_start]
        end = seg.raw_word_offsets[hi - seg.word_start]
        for r in range(begin, end):
            if seg.reference[r] in VOWEL_NAMES:
                index = offset + r - begin
                sites[index] = Site(plan.clip.audio_filename, seg.segment_index, r)
                classes[index] = window_position(word_of(seg, r), window, seg)
        pieces.append(seg.reference[begin:end])
        offset += end - begin
    reference = "".join(pieces)
    if reference != window.phoneme_label:
        raise AssertionError(f"window {window.window_index} of {plan.clip.audio_filename} "
                             "does not rebuild from its segments")
    return Unit(
        key=window_key(plan.clip.audio_filename, window.window_index),
        reciter_id=plan.clip.reciter_id,
        weight=1 / plan.clip.inclusion_probability,
        reference=reference,
        sites=sites,
        classes=classes,
    )


def stream_unit(
    plan: ClipPlan, block: int, population: frozenset[Site], times: Mapping[Site, float]
) -> Unit:
    """The whole recitation streamed at ``block``, scored against the concatenated
    segment references; each site classed by the protocol region its time falls in."""
    sites: dict[int, Site] = {}
    classes: dict[int, str] = {}
    offset = 0
    for seg in plan.segments:
        for r in haraka_sites(seg):
            site = Site(plan.clip.audio_filename, seg.segment_index, r)
            if site in population:
                sites[offset + r] = site
                classes[offset + r] = stream_region(times[site], plan.recitation_num_samples, block)
        offset += len(seg.reference)
    return Unit(
        key=stream_unit_key(plan.clip.audio_filename, block),
        reciter_id=plan.clip.reciter_id,
        weight=1 / plan.clip.inclusion_probability,
        reference="".join(seg.reference for seg in plan.segments),
        sites=sites,
        classes=classes,
    )


def windowed_population(plans: Iterable[ClipPlan]) -> frozenset[Site]:
    """Every haraka of an eligible clip that at least one window contains."""
    return frozenset(
        site
        for plan in plans
        if plan.eligible
        for window in plan.windows
        for site in window_unit(plan, window).sites.values()
    )


# --- Site times (for the stream regions) -----------------------------------------------


def time_in_segment(
    ref_index: int, word_span: tuple[int, int], aligned: Mapping[int, int], token_mids: list[float]
) -> tuple[float, str] | None:
    """Seconds into the segment at which its decode emits ``ref_index``, or ``None``.

    ``aligned`` maps reference index -> decode index; ``token_mids`` is each decode
    token's midpoint in seconds. The haraka's own token is used when it was aligned,
    otherwise the nearest aligned character of the same word (its carrier first).
    """
    if ref_index in aligned:
        return token_mids[aligned[ref_index]], TIME_OWN
    lo, hi = word_span
    for neighbour in [*range(ref_index - 1, lo - 1, -1), *range(ref_index + 1, hi)]:
        if neighbour in aligned:
            return token_mids[aligned[neighbour]], TIME_NEIGHBOUR
    return None


def site_times(
    plans: Iterable[ClipPlan],
    population: frozenset[Site],
    base_segments: Mapping[str, dict],
) -> tuple[dict[Site, float], Counter]:
    """Each site's time in seconds from its clip's stream start, and where it came from.

    Taken from the base teacher's whole-segment decode (``base_segments``: per segment key
    the decode and each token's doubled midpoint step), so it is one time per site for
    every arm and model. Falls back to the word's forced-alignment span.
    """
    times: dict[Site, float] = {}
    sources: Counter = Counter()
    for plan in plans:
        if not plan.eligible:
            continue
        origin = plan.recitation_start_sample / TARGET_SAMPLE_RATE
        for seg in plan.segments:
            cached = base_segments[segment_key(plan.clip.audio_filename, seg.segment_index)]
            alignment = smith_waterman(cached["decode"], seg.reference)
            aligned = {
                alignment.ref_start + i: q for i, q in enumerate(alignment.ref_to_query) if q >= 0
            }
            token_mids = [(mid2 / 2 + 0.5) * STEP_SECONDS for mid2 in cached["mid2"]]
            start = seg.start_sample / TARGET_SAMPLE_RATE - origin
            for r in haraka_sites(seg):
                site = Site(plan.clip.audio_filename, seg.segment_index, r)
                if site not in population:
                    continue
                word = word_of(seg, r)
                k = word - seg.word_start
                span = (seg.raw_word_offsets[k], seg.raw_word_offsets[k + 1])
                found = time_in_segment(r, span, aligned, token_mids)
                if found is None:
                    t0, t1 = plan.clip.word_times[word], plan.clip.word_times[word + 1]
                    fraction = (r - span[0] + 0.5) / (span[1] - span[0])
                    times[site] = max(0.0, t0 + fraction * (t1 - t0) - origin)
                    sources[TIME_WORD] += 1
                else:
                    times[site] = max(0.0, start + found[0])
                    sources[found[1]] += 1
    return times, sources


# --- Scoring ------------------------------------------------------------------------------


def unit_outcomes(decode: str, unit: Unit) -> dict[int, str]:
    """The ``vowel_sites`` outcome of every site in ``unit``, by reference index."""
    found = {
        site.reference_index: site.outcome
        for site in vowel_sites(decode, unit.reference)
        if site.outcome != SPURIOUS and site.reference_index in unit.sites
    }
    if found.keys() != unit.sites.keys():
        raise AssertionError(f"{unit.key}: vowel_sites did not classify every site")
    return found


def _score(job: tuple[str, Unit]) -> dict[int, str]:
    return unit_outcomes(*job)


class JointKey(NamedTuple):
    """One cell of the joint table: a site's outcome in an arm beside its outcome on the
    whole segment, for one model, haraka, class and position in its word."""

    model: str
    arm: str
    haraka: str
    cls: str
    word_position: str
    segment: str  # outcome group of the same site in the segment arm
    outcome: str  # outcome group in this arm


@dataclass(frozen=True)
class Observation:
    reciter_id: int
    key: JointKey
    weight: float  # inclusion weight x share
    share: float


def arm_observations(
    model: str,
    arm: str,
    units: Iterable[Unit],
    outcomes: Mapping[str, Mapping[int, str]],
    segment_outcome: Mapping[Site, str],
    word_positions: Mapping[Site, str],
    share_per_occurrence: bool,
) -> list[Observation]:
    """One observation per site occurrence in ``units``.

    With ``share_per_occurrence`` every occurrence counts 1; otherwise a site seen by
    ``n`` units carries 1/n in each, so the arm counts every site once.
    """
    units = list(units)
    seen = Counter(site for unit in units for site in unit.sites.values())
    observations = []
    for unit in units:
        for index, site in unit.sites.items():
            share = 1.0 if share_per_occurrence else 1 / seen[site]
            key = JointKey(
                model=model,
                arm=arm,
                haraka=VOWEL_NAMES[unit.reference[index]],
                cls=unit.classes[index],
                word_position=word_positions[site],
                segment=segment_outcome[site],
                outcome=OUTCOME_GROUP[outcomes[unit.key][index]],
            )
            observations.append(Observation(unit.reciter_id, key, unit.weight * share, share))
    return observations


# --- Assembling the arms ---------------------------------------------------------------


@dataclass(frozen=True)
class Arms:
    """Every arm's units over a fixed site population, and every site's word position."""

    population: frozenset[Site]
    units: dict[str, list[Unit]]  # arm -> units; WINDOW_OCC shares WINDOW's units
    word_positions: dict[Site, str]


def build_arms(plans: list[ClipPlan], times: Mapping[Site, float]) -> Arms:
    population = windowed_population(plans)
    eligible = [plan for plan in plans if plan.eligible]
    windows = [window_unit(plan, window) for plan in eligible for window in plan.windows]
    segments = [segment_unit(plan, seg, population) for plan in plans for seg in plan.segments]
    units = {
        SEGMENT_ALL: segments,
        SEGMENT: [windowed_only(unit) for unit in segments],
        WINDOW: windows,
        WINDOW_OCC: windows,
        **{STREAM[b]: [stream_unit(plan, b, population, times) for plan in eligible]
           for b in BLOCKS},
    }
    word_positions = {
        Site(plan.clip.audio_filename, seg.segment_index, r): word_position(seg, r)
        for plan in plans for seg in plan.segments for r in haraka_sites(seg)
    }
    return Arms(population, units, word_positions)


def score_model(arms: Arms, decodes: Mapping[str, str], workers: int) -> dict[str, dict[int, str]]:
    """Every unit's outcomes for one model's decodes, keyed by unit key (the segment arms
    share their units' decodes and so their outcomes)."""
    from concurrent.futures import ProcessPoolExecutor

    unique = {unit.key: unit for arm in (SEGMENT_ALL, WINDOW, *STREAM.values())
              for unit in arms.units[arm]}
    jobs = [(decodes[key], unit) for key, unit in sorted(unique.items())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_score, jobs, chunksize=32))
    return dict(zip(sorted(unique), results))


def model_observations(model: str, arms: Arms, outcomes: Mapping[str, Mapping[int, str]]
                       ) -> list[Observation]:
    """The joint observations of every arm of one model."""
    segment_outcome = {
        site: OUTCOME_GROUP[outcomes[unit.key][index]]
        for unit in arms.units[SEGMENT_ALL]
        for index, site in unit.sites.items()
    }
    observations = []
    for arm, units in arms.units.items():
        observations += arm_observations(
            model, arm, units, outcomes, segment_outcome, arms.word_positions,
            share_per_occurrence=arm != WINDOW,
        )
    return observations



"""Windowing vs weights at target-pair carriers, on unselected pool audio (#146).

#90's verified sites cannot say whether ``h448``'s target-pair swaps come from the **window
protocol** or from its **weights**: they were mined where ``h448``'s b=0 stream swapped, and
stratified by the base teacher's whole-segment decode. This reads the same question off
carriers nobody selected: every target-pair carrier and in-word geminate of the #83 mining
pool's clips that the #85 haraka-gap cache holds, from decodes already committed. It needs no
GPU and no torch, and it **measures only**: no threshold, no protocol choice.

Arms
----
Each of the base teacher and ``h448`` under three protocols, all from
``haraka_gap_cache/{base,h448}.json`` (one precision and batch size for every arm,
:func:`training.haraka_gap.validated_settings`):

* ``whole``: each kept waqf segment decoded whole (#85's ``segment`` arm);
* ``stream_b0`` / ``stream_b1``: the clip's recitation span streamed at commit block b (5 s
  windows, 1 s hop, startup rule, tail flush), read against the clip's kept-segment
  references joined by one space (:data:`SEGMENT_SEPARATOR`), so a seam never makes two
  segments' letters one run.

Carriers and outcomes
---------------------
Carriers are a function of the realized reference alone, as #87 defines them
(:mod:`tadabur.listening_session`): a **pair carrier** is the first of each run of a target-pair
letter (a geminate is one carrier, on its first half), an **in-word geminate** the first of a
doubled consonant whose run stays inside one word. A ``ذ``, ``ظ``, ``س``, ``ط`` or ``ق`` carrier
sits in two pairs, so it counts in both **directions** (``ط→ق`` is a ``ط`` whose partner is
``ق``). What a decode commits there is the #84 scorer's rule, by position
(:func:`training.site_outcomes.pair_outcome`: the letter
:func:`tadabur.contrast_attribution.aligned_consonants` aligns to the carrier;
:func:`~training.site_outcomes.shaddah_outcome` for a geminate). Against the **mushaf**, never
against what was heard:

* ``swap``: the partner letter (a geminate: one consonant);
* ``mushaf``: the mushaf's letter (a geminate held);
* ``other``: anything else, another letter or nothing aligned at the carrier.

Rows are the eighteen directions of the nine target pairs and the in-word geminates; the
six soft pairs pool into one family row and each other pair into its own (Muraja forgives
only the soft pairs, so they are never pooled with the rest).

Statistics
----------
Every rate is a weighted ratio, each carrier weighted by 1 / its clip's inclusion probability
(``mining_pool/clips.jsonl``), with acceptance rules §1's interval
(:func:`training.truth_scorer.estimate`: the reciter-clustered bootstrap, B = 10,000, seed
20261008). Every arm reads the same carriers, so each **gap** is paired: both swap rates are
recomputed inside every replicate of one reciter draw
(:func:`training.acceptance_stats.difference_interval`). Windowing is ``stream_b − whole`` per
model; weights are ``h448 − base`` per protocol.

Population
----------
The cache holds the pool clips ADR-0007's window build accepts (2,084 of 2,508); of those,
the clips whose recording is tied to anything ``h448`` was trained or selected on are left
out (:func:`tadabur.power_frame.unseen_clips`, as Ship A does), because a weights comparison
on audio one model has heard is not one. The whole cache is reported beside it.

Cross-checks
------------
* **Owner verdicts** (#90): at Ship A's partner swaps where the base teacher was right, what
  the owner heard, per pair. They are selected sites and enter no estimate; they say for
  which pairs a swap against the mushaf reads as a model error.
* **App checks**: the truth baseline's diagnostic items (``truth_sites/waqf_boundaries.jsonl``,
  unselected on any model outcome), whose committed decodes include the ``checks`` arm, the
  transcript Muraja commits from its own check sequence. Every item weighs 1.

Usage (from ``tools/``; torch-free, about a minute)::

  python -m training.carrier_windowing
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from tadabur.contrast_attribution import CarrierAlignment, align_on_carriers
from tadabur.exposure import unseen_violations
from tadabur.listening_session import (
    VERDICTS_PATH,
    Verdict,
    crosses_words,
    read_verdicts,
    run_start,
)
from tadabur.mining_pool import PoolSegment, segment_key
from tadabur.power_frame import input_checksums, partner, unseen_clips
from tadabur.ship_a import SITES_PATH, ShipASite, load_sites
from tadabur.staged_audio import StagedClip, load_staged_clips
from tadabur.truth_sites import (
    CONSONANTS,
    HELD,
    NO_VERDICT,
    NOT_HELD,
    SOFT_PAIRS,
    TARGET_PAIRS,
    TRUTH_SITES_DIR,
    load_truth_sites,
)
from training.acceptance_stats import REPLICATES, SEED, Resample, difference_interval, ratio
from training.decoding import committed_text
from training.haraka_gap import (
    CACHE_DIR,
    POOL_AUDIO,
    load_cache,
    validated_settings,
    verified_provenance,
)
from training.haraka_gap_arms import STREAM, ClipPlan, plan_pool, stream_unit_key
from training.site_outcomes import pair_outcome, shaddah_outcome
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import (
    CHECK_PROTOCOLS,
    CHECK_STORE,
    CHECKS,
    DECODE_STORE,
    DEFAULT_MODELS,
    PROTOCOLS,
    check_cache,
    read_cache,
)
from training.truth_scorer import estimate, item_key

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
DOC_PATH = DOCS_DIR / "carrier-windowing.md"
JSON_PATH = DOCS_DIR / "carrier-windowing.json"
DIAGNOSTIC_SITES = TRUTH_SITES_DIR / "waqf_boundaries.jsonl"

MODELS = ("base", "h448")
WHOLE = "whole"
POOL_PROTOCOLS = (WHOLE, *STREAM.values())
#: The truth baseline's protocols: the item whole (``spans``), its b=0 stream, and the
#: transcript Muraja commits from its own checks.
APP_PROTOCOLS = (*PROTOCOLS, CHECKS)
SPANS = "spans"
#: Joins a clip's kept-segment references into the stream's reference: the word separator
#: the references already use, so a segment ending in a letter the next one starts with stays
#: two runs (a geminate's run, or a normalized group, never spans a seam).
SEGMENT_SEPARATOR = " "

UNSEEN = "h448_unseen"
CACHE = "cache"

GEMINATE = "geminate"
SOFT_PAIRS_FAMILY = "soft pairs"
ALL_DIRECTIONS = "all"
SWAP, MUSHAF, OTHER = "swap", "mushaf", "other"


def arm(model: str, protocol: str) -> str:
    return f"{model}/{protocol}"


def pair_directions(pair: str) -> tuple[str, str]:
    a, b = pair.split("↔")
    return f"{a}→{b}", f"{b}→{a}"


#: Every family and its directions, in report order: the soft pairs, the target pairs Muraja
#: does not forgive, then the in-word geminates.
FAMILIES: dict[str, tuple[str, ...]] = {
    SOFT_PAIRS_FAMILY: tuple(d for p in sorted(SOFT_PAIRS) for d in pair_directions(p)),
    **{p: pair_directions(p) for p in sorted(TARGET_PAIRS - SOFT_PAIRS)},
    GEMINATE: (GEMINATE,),
}

#: Each target-pair letter's directions (two for a letter in two pairs).
LETTER_DIRECTIONS: dict[str, tuple[str, ...]] = {
    letter: tuple(d for p in sorted(TARGET_PAIRS) for d in pair_directions(p)
                  if d.startswith(letter))
    for letter in sorted({c for p in TARGET_PAIRS for c in p.split("↔")})
}


def mushaf_and_swap(direction: str) -> tuple[str, str]:
    """What a decode commits for ``mushaf`` and for ``swap`` in a direction."""
    if direction == GEMINATE:
        return HELD, NOT_HELD
    prescribed, other = direction.split("→")
    return prescribed, other


class RowKey(NamedTuple):
    family: str
    direction: str  # one direction, or ALL_DIRECTIONS for the family pooled


#: Every row in report order: per family, each direction, then the family pooled.
ROWS: tuple[RowKey, ...] = tuple(
    RowKey(family, direction)
    for family, directions in FAMILIES.items()
    for direction in (*directions, *((ALL_DIRECTIONS,) if len(directions) > 1 else ()))
)


def row_directions(key: RowKey) -> tuple[str, ...]:
    return FAMILIES[key.family] if key.direction == ALL_DIRECTIONS else (key.direction,)


# --- Carriers: positions in a reference, and what a decode commits there ----------------------


class Position(NamedTuple):
    """A carrier's place in its clip: the kept segment (in order), the index in that
    segment's reference, and whether it is an in-word geminate (else a pair letter)."""

    segment: int
    index: int
    geminate: bool


def pair_positions(segment: int, reference: str) -> list[Position]:
    """The first of each run of a target-pair letter."""
    return [Position(segment, i, False) for i, char in enumerate(reference)
            if char in LETTER_DIRECTIONS and run_start(reference, i) == i]


def geminate_positions(segment: int, pool_segment: PoolSegment) -> list[Position]:
    """The first of each doubled consonant whose run stays inside one word."""
    reference, offsets = pool_segment.reference, list(pool_segment.raw_word_offsets)
    return [
        Position(segment, i, True)
        for i, char in enumerate(reference)
        if char in CONSONANTS and reference[i + 1:i + 2] == char and run_start(reference, i) == i
        and not crosses_words(reference, offsets, i)
    ]


def committed(alignment: CarrierAlignment, index: int, geminate: bool) -> str | None:
    """The scorer's committed value at a carrier: a letter, a gemination state, or ``None``."""
    rule = shaddah_outcome if geminate else pair_outcome
    return rule(alignment, index).committed


def joined_reference(references: Sequence[str]) -> tuple[str, list[int]]:
    """The references joined by :data:`SEGMENT_SEPARATOR`, and each one's offset in it."""
    offsets, start = [], 0
    for reference in references:
        offsets.append(start)
        start += len(reference) + len(SEGMENT_SEPARATOR)
    return SEGMENT_SEPARATOR.join(references), offsets


@dataclass(frozen=True)
class Clip:
    """One clip (or truth-baseline item): who recited it, its weight, its kept-segment
    references and carriers, and every arm's decode, either one per segment
    (``by_segment``: the whole-segment arms) or one of the span the segments cover
    (``by_span``: a stream, or a whole item)."""

    name: str
    reciter_id: int
    weight: float
    references: tuple[str, ...]
    positions: tuple[Position, ...]
    by_segment: Mapping[str, tuple[str, ...]]
    by_span: Mapping[str, str]


def clip_commits(clip: Clip) -> dict[str, list[str | None]]:
    """Per arm, the committed value at each of the clip's carriers, in position order."""
    commits: dict[str, list[str | None]] = {}
    for arm_name, decodes in clip.by_segment.items():
        alignments = [align_on_carriers(decode, reference)
                      for decode, reference in zip(decodes, clip.references, strict=True)]
        commits[arm_name] = [committed(alignments[p.segment], p.index, p.geminate)
                             for p in clip.positions]
    reference, offsets = joined_reference(clip.references)
    for arm_name, decode in clip.by_span.items():
        alignment = align_on_carriers(decode, reference)
        commits[arm_name] = [committed(alignment, offsets[p.segment] + p.index, p.geminate)
                             for p in clip.positions]
    return commits


@dataclass(frozen=True)
class Carrier:
    """One carrier: its clip, reciter and weight, the reference letter, and per arm what the
    decode committed there."""

    clip: str
    reciter_id: int
    weight: float
    letter: str
    geminate: bool
    committed: Mapping[str, str | None]

    @property
    def directions(self) -> tuple[str, ...]:
        return (GEMINATE,) if self.geminate else LETTER_DIRECTIONS[self.letter]


def carriers_of(clips: Sequence[Clip], workers: int) -> list[Carrier]:
    """Every carrier of ``clips``, read in ``workers`` processes (alignment is the cost)."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(clip_commits, clips, chunksize=8))
    return [
        Carrier(clip.name, clip.reciter_id, clip.weight, clip.references[p.segment][p.index],
                p.geminate, {arm_name: values[k] for arm_name, values in commits.items()})
        for clip, commits in zip(clips, results)
        for k, p in enumerate(clip.positions)
    ]


# --- Rows: rates and paired gaps -----------------------------------------------------------


def row_members(carriers: Iterable[Carrier], key: RowKey) -> list[tuple[str, Carrier]]:
    """The row's carriers, each with its direction in the row (never more than one: no
    letter sits in two pairs of one family)."""
    wanted = set(row_directions(key))
    members = []
    for carrier in carriers:
        held = [d for d in carrier.directions if d in wanted]
        if len(held) > 1:
            raise AssertionError(f"{carrier.letter} is in {held} of one row")
        if held:
            members.append((held[0], carrier))
    return members


def outcome_indicators(members: Sequence[tuple[str, Carrier]], arm_name: str
                       ) -> dict[str, np.ndarray]:
    """Per rate, 1 where the carrier's committed value is that outcome in its direction."""
    values = [(mushaf_and_swap(d), c.committed[arm_name]) for d, c in members]
    swap = np.array([v == pair[1] for pair, v in values], dtype=float)
    mushaf = np.array([v == pair[0] for pair, v in values], dtype=float)
    return {SWAP: swap, MUSHAF: mushaf, OTHER: 1 - swap - mushaf}


def gap_name(first: str, second: str) -> str:
    return f"{first} − {second}"


def paired_gap(clusters: Sequence[int], weights: np.ndarray, first: np.ndarray,
               second: np.ndarray, resample: Resample) -> dict:
    """``first`` − ``second`` swap rate on the same carriers, both recomputed in every replicate."""
    point = ratio(weights * first, weights) - ratio(weights * second, weights)
    interval = difference_interval(clusters, (weights * first, weights),
                                   (weights * second, weights), weights, resample)
    return {"point": point if math.isfinite(point) else None, **interval.as_dict()}


def row_report(members: Sequence[tuple[str, Carrier]], arms: Sequence[str],
               gaps: Sequence[tuple[str, str]]) -> dict:
    """Carriers and reciters, and when there are any, every arm's rates (each with its
    unweighted ``count``) and every paired swap-rate gap, all on one reciter draw."""
    clusters = [carrier.reciter_id for _, carrier in members]
    entry: dict = {"carriers": len(members), "reciters": len(set(clusters))}
    if not members:
        return entry
    weights = np.array([carrier.weight for _, carrier in members], dtype=float)
    resample = Resample.by_cluster(clusters)
    indicators = {a: outcome_indicators(members, a) for a in arms}
    entry["rates"] = {
        a: {rate: {**estimate(clusters, (weights * ind, weights), weights, resample),
                   "count": int(ind.sum())}
            for rate, ind in indicators[a].items()}
        for a in arms
    }
    entry["gaps"] = {
        gap_name(first, second): paired_gap(
            clusters, weights, indicators[first][SWAP], indicators[second][SWAP], resample)
        for first, second in gaps
    }
    return entry


def table(carriers: Sequence[Carrier], arms: Sequence[str], gaps: Sequence[tuple[str, str]]
          ) -> list[dict]:
    return [{"family": key.family, "direction": key.direction,
             **row_report(row_members(carriers, key), arms, gaps)} for key in ROWS]


WINDOWING, WEIGHTS = "windowing", "weights"


def arm_gaps(protocols: Sequence[str], reference: str) -> dict[str, list[tuple[str, str]]]:
    """The paired gaps: **windowing** per model (each protocol − ``reference``) and
    **weights** per protocol (``h448`` − base)."""
    return {
        WINDOWING: [(arm(m, p), arm(m, reference)) for m in MODELS for p in protocols
                    if p != reference],
        WEIGHTS: [(arm("h448", p), arm("base", p)) for p in protocols],
    }


def gap_names(gaps: Mapping[str, list[tuple[str, str]]]) -> dict[str, list[str]]:
    return {kind: [gap_name(*pair) for pair in pairs] for kind, pairs in gaps.items()}


def all_gaps(gaps: Mapping[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    return [pair for pairs in gaps.values() for pair in pairs]


def uncommitted(carriers: Sequence[Carrier], arms: Sequence[str]) -> dict:
    """Per arm and kind, carriers with nothing committed (no consonant aligned): the part of
    ``other`` that is the alignment's, not a letter's."""
    kinds = {"pair": [c for c in carriers if not c.geminate],
             GEMINATE: [c for c in carriers if c.geminate]}
    return {a: {kind: {"carriers": len(group),
                       "uncommitted": sum(c.committed[a] is None for c in group)}
                for kind, group in kinds.items()}
            for a in arms}


# --- The pool ----------------------------------------------------------------------------


def pool_clip(plan: ClipPlan, caches: Mapping[str, dict]) -> Clip:
    """One cached clip: its carriers and its six arms' decodes."""
    name = plan.clip.audio_filename
    positions = tuple(
        p for n, s in enumerate(plan.segments)
        for p in sorted(pair_positions(n, s.reference) + geminate_positions(n, s)))
    return Clip(
        name=name,
        reciter_id=plan.clip.reciter_id,
        weight=1 / plan.clip.inclusion_probability,
        references=tuple(s.reference for s in plan.segments),
        positions=positions,
        by_segment={arm(model, WHOLE): tuple(
            caches[model]["segments"][segment_key(name, s.segment_index)]["decode"]
            for s in plan.segments) for model in MODELS},
        by_span={arm(model, protocol): caches[model]["streams"][stream_unit_key(name, block)]
                 for model in MODELS for block, protocol in STREAM.items()},
    )


def cached_plans(plans: Sequence[ClipPlan], caches: Mapping[str, dict]) -> list[ClipPlan]:
    """The pool clips the cache streamed: exactly the ones the window build accepts."""
    cached = [plan for plan in plans if plan.eligible]
    names = {plan.clip.audio_filename for plan in cached}
    for model, cache in caches.items():
        if set(cache["stream_rows_clips"]) != names:
            raise ValueError(f"{model}'s cache does not hold exactly the window-eligible clips")
    return cached


def population(plans: Sequence[ClipPlan], cached: Sequence[ClipPlan], unseen: set[str]) -> dict:
    """Who is in the population and what it leaves out, in clips and reciters."""
    def reciters(group: Iterable[ClipPlan]) -> int:
        return len({plan.clip.reciter_id for plan in group})

    kept = [plan for plan in cached if plan.clip.audio_filename in unseen]
    return {
        "pool": {"clips": len(plans), "reciters": reciters(plans)},
        "not_in_cache": dict(sorted(Counter(p.exclusion for p in plans if p.exclusion).items())),
        CACHE: {"clips": len(cached), "reciters": reciters(cached),
                "segments": sum(len(p.segments) for p in cached)},
        "h448_linked_clips": len(cached) - len(kept),
        UNSEEN: {"clips": len(kept), "reciters": reciters(kept),
                 "segments": sum(len(p.segments) for p in kept),
                 "inclusion_probability_1": sum(p.clip.inclusion_probability == 1 for p in kept)},
    }


def pool_report(carriers: Sequence[Carrier], unseen: set[str]) -> dict:
    arms = [arm(m, p) for m in MODELS for p in POOL_PROTOCOLS]
    gaps = arm_gaps(POOL_PROTOCOLS, WHOLE)
    report = {}
    for name, group in ((UNSEEN, [c for c in carriers if c.clip in unseen]), (CACHE, carriers)):
        report[name] = {"rows": table(group, arms, all_gaps(gaps)),
                        "uncommitted": uncommitted(group, arms)}
    return {"arms": arms, "gaps": gap_names(gaps), "populations": report}


# --- Cross-checks ----------------------------------------------------------------------------


def owner_verdicts(sites: Iterable[ShipASite], verdicts: Mapping[str, Verdict]) -> dict:
    """Per target pair, at Ship A's answered partner swaps the base teacher got right, how
    often the owner heard the mushaf letter, the partner, another letter or could not tell."""
    counts: dict[str, Counter] = {pair: Counter() for pair in sorted(TARGET_PAIRS)}
    for site in sites:
        verdict = verdicts.get(site.site.site_id)
        if verdict is None or not (site.partner_swap and site.base_already_right):
            continue
        heard = verdict.heard
        outcome = (MUSHAF if heard == site.site.prescribed else SWAP if heard == partner(site.site)
                   else "unclear" if heard in NO_VERDICT else OTHER)
        counts[site.family][outcome] += 1
    soft = sum((counts[pair] for pair in SOFT_PAIRS), Counter())
    return {**{pair: dict(c) for pair, c in counts.items()}, SOFT_PAIRS_FAMILY: dict(soft)}


def app_check_clips(registry: Mapping[str, StagedClip]) -> tuple[list[Clip], dict]:
    """The diagnostic items, each with its pair carriers and six arms' decodes, and the
    items' counts. The caches are checked against the models and protocols they must hold."""
    references = {item_key(s): s for s in load_truth_sites(DIAGNOSTIC_SITES)}
    barred = unseen_violations({s.audio_filename: registry[s.audio_filename]
                                for s in references.values()})
    decodes: dict[str, dict[str, str]] = {}
    for model in MODELS:
        cache, checks = read_cache(DECODE_STORE.path(model)), read_cache(CHECK_STORE.path(model))
        check_cache(DECODE_STORE.path(model), cache, DEFAULT_MODELS[model], PROTOCOLS)
        check_cache(CHECK_STORE.path(model), checks, DEFAULT_MODELS[model], CHECK_PROTOCOLS)
        for protocol in PROTOCOLS:
            decodes[arm(model, protocol)] = {k: cache["items"][k][protocol] for k in references}
        decodes[arm(model, CHECKS)] = {k: committed_text(checks["items"][k][CHECKS])
                                       for k in references}
    clips = [
        Clip(name=key, reciter_id=registry[site.audio_filename].reciter_id, weight=1.0,
             references=(site.reference,), positions=tuple(pair_positions(0, site.reference)),
             by_segment={}, by_span={a: by_item[key] for a, by_item in decodes.items()})
        for key, site in sorted(references.items()) if site.audio_filename not in barred
    ]
    items = {"items": len(references), "h448_linked_items": len(barred),
             "kept_items": len(clips), "reciters": len({clip.reciter_id for clip in clips})}
    return clips, items


def app_check_report(carriers: Sequence[Carrier], items: dict) -> dict:
    arms = [arm(m, p) for m in MODELS for p in APP_PROTOCOLS]
    gaps = arm_gaps(APP_PROTOCOLS, SPANS)
    return {**items, "arms": arms, "gaps": gap_names(gaps),
            "rows": table(carriers, arms, all_gaps(gaps)),
            "uncommitted": uncommitted(carriers, arms)}


# --- Run ----------------------------------------------------------------------------------


def run(workers: int) -> dict:
    provenance = verified_provenance(CACHE_DIR, list(MODELS))
    caches = {model: load_cache(CACHE_DIR, model) for model in MODELS}
    settings = validated_settings(caches)
    registry = load_staged_clips()
    plans = plan_pool()
    cached = cached_plans(plans, caches)
    unseen = {clip.audio_filename for clip in unseen_clips([p.clip for p in cached], registry)[0]}
    pool = carriers_of([pool_clip(plan, caches) for plan in cached], workers)
    check_clips, items = app_check_clips(registry)
    other_inputs = {
        **{f"truth_baseline/decodes/{m}.json": DECODE_STORE.path(m) for m in MODELS},
        **{f"truth_baseline/checks/{m}.json": CHECK_STORE.path(m) for m in MODELS},
        "truth_sites/waqf_boundaries.jsonl": DIAGNOSTIC_SITES,
        "ship_a/sites.jsonl": SITES_PATH,
        "listening_session/verdicts.jsonl": VERDICTS_PATH,
    }
    return {
        "issue": 146,
        "inputs": {
            "haraka_gap_cache": provenance["files"],
            "sources": provenance["sources"],
            "models": provenance["models"],
            "pool_audio": provenance[POOL_AUDIO],
            "other": input_checksums(other_inputs),
        },
        "decode_settings": settings,
        "bootstrap": {"replicates": REPLICATES, "seed": SEED, "cluster": "reciter"},
        "segment_separator": SEGMENT_SEPARATOR,
        "population": population(plans, cached, unseen),
        "pool": pool_report(pool, unseen),
        "owner_verdicts": owner_verdicts(load_sites(), read_verdicts()),
        "app_checks": app_check_report(carriers_of(check_clips, workers), items),
    }


def main() -> None:
    from training.carrier_windowing_doc import render

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=JSON_PATH)
    parser.add_argument("--out-md", type=Path, default=DOC_PATH)
    args = parser.parse_args()
    report = run(args.workers)
    write_text_atomically(args.out_json, json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    write_text_atomically(args.out_md, render(report))
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()

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
:func:`~training.site_outcomes.shaddah_outcome` for a geminate). Each carrier becomes one
unlabelled truth site per direction (``heard`` pending), so the directions, families and the
``swap`` / ``mushaf`` / ``other`` rates are :mod:`training.direction_table`'s, the rule #90's
verified sites are read with, against the **mushaf**, never against what was heard.

Statistics
----------
Every rate is a weighted ratio, each carrier weighted by 1 / its clip's inclusion probability
(``mining_pool/clips.jsonl``), with acceptance rules §1's interval
(:func:`training.direction_table.rates`: the reciter-clustered bootstrap, B = 10,000, seed
20261008). Every arm reads the same carriers, so each **gap** is paired: both swap rates are
recomputed inside every replicate of one reciter draw (:func:`training.direction_table.difference`).
Windowing is ``stream_b − whole`` per model; weights are ``h448 − base`` per protocol.

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
* **App checks**: the truth baseline's unselected items, the diagnostic clips
  (``truth_sites/waqf_boundaries.jsonl``) and the word sample's segments
  (``ship_a/word_sample.jsonl``), whose committed decodes include the ``checks`` arm, the
  transcript Muraja commits from its own check sequence. Their pair carriers only; every item
  weighs 1.

Usage (from ``tools/``; torch-free, about a minute)::

  python -m training.carrier_windowing
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

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
from tadabur.power_frame import input_checksums, unseen_clips
from tadabur.ship_a import SITES_PATH, ShipASite, load_sites
from tadabur.staged_audio import StagedClip, load_staged_clips
from tadabur.truth_sites import (
    CONSONANTS,
    HELD,
    NEW_AUDIT,
    NO_VERDICT,
    PENDING,
    SHADDAH,
    SOFT_PAIRS,
    TARGET_PAIRS,
    TRUTH_SITES_DIR,
    TruthSite,
    load_truth_sites,
)
from tadabur.word_sample import WORD_SAMPLE_PATH, load_words
from training.acceptance_stats import REPLICATES, SEED
from training.decoding import committed_text
from training.direction_table import (
    ALL_DIRECTIONS,
    FAMILIES,
    GEMINATE,
    MUSHAF,
    OTHER,
    SOFT_PAIRS_FAMILY,
    SWAP,
    counts,
    difference,
    direction,
    partner,
    summary,
)
from training.haraka_gap import (
    CACHE_DIR,
    POOL_AUDIO,
    load_cache,
    validated_settings,
    verified_provenance,
)
from training.haraka_gap_arms import STREAM, ClipPlan, plan_pool, stream_unit_key
from training.site_outcomes import SiteOutcome, pair_outcome, shaddah_outcome
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import (
    CHECK_STORE,
    CHECKS,
    DECODE_STORE,
    DEFAULT_MODELS,
    PROTOCOLS,
    Located,
    items_of,
    model_caches,
)
from training.truth_scorer import SiteRow

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
#: Every carrier is a census site of this stratum, defined by the reference alone.
CARRIER_STRATUM = "carrier_windowing:census"

UNSEEN = "h448_unseen"
CACHE = "cache"
DIAGNOSTIC = "diagnostic"
WORD_SAMPLE = "word_sample"
WINDOWING, WEIGHTS = "windowing", "weights"

#: Every target-pair letter and the pairs it sits in (two for ``ذ ظ س ط ق``).
LETTER_PAIRS: dict[str, tuple[str, ...]] = {
    letter: tuple(p for p in sorted(TARGET_PAIRS) if letter in p.split("↔"))
    for letter in sorted({c for p in TARGET_PAIRS for c in p.split("↔")})
}


def arm(model: str, protocol: str) -> str:
    return f"{model}/{protocol}"


class RowKey(NamedTuple):
    family: str
    direction: str  # one direction, or ALL_DIRECTIONS for the family pooled


#: Every row in report order (``direction_table.FAMILIES``): per family, each direction,
#: then the family pooled.
ROWS: tuple[RowKey, ...] = tuple(
    RowKey(family, d)
    for family, directions in FAMILIES.items()
    for d in (*directions, *((ALL_DIRECTIONS,) if len(directions) > 1 else ()))
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
            if char in LETTER_PAIRS and run_start(reference, i) == i]


def geminate_positions(segment: int, pool_segment: PoolSegment) -> list[Position]:
    """The first of each doubled consonant whose run stays inside one word."""
    reference, offsets = pool_segment.reference, list(pool_segment.raw_word_offsets)
    return [
        Position(segment, i, True)
        for i, char in enumerate(reference)
        if char in CONSONANTS and reference[i + 1:i + 2] == char and run_start(reference, i) == i
        and not crosses_words(reference, offsets, i)
    ]


def committed(alignment: CarrierAlignment, index: int, geminate: bool) -> SiteOutcome:
    """The scorer's outcome at a carrier: the letter, or gemination state, it commits."""
    rule = shaddah_outcome if geminate else pair_outcome
    return rule(alignment, index)


def joined_reference(references: Sequence[str]) -> tuple[str, list[int]]:
    """The references joined by :data:`SEGMENT_SEPARATOR`, and each one's offset in it."""
    offsets, start = [], 0
    for reference in references:
        offsets.append(start)
        start += len(reference) + len(SEGMENT_SEPARATOR)
    return SEGMENT_SEPARATOR.join(references), offsets


@dataclass(frozen=True)
class Clip:
    """One clip (or truth-baseline item): who recited it, its weight, its kept segments
    (references and first samples) and carriers, and every arm's decode, either one per
    segment (``by_segment``: the whole-segment arms) or one of the span the segments cover
    (``by_span``: a stream, or a whole item)."""

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    weight: float
    references: tuple[str, ...]
    starts: tuple[int, ...]
    positions: tuple[Position, ...]
    by_segment: Mapping[str, tuple[str, ...]]
    by_span: Mapping[str, str]


def clip_commits(clip: Clip) -> dict[str, list[SiteOutcome]]:
    """Per arm, the outcome at each of the clip's carriers, in position order."""
    commits: dict[str, list[SiteOutcome]] = {}
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


def carrier_sites(clip: Clip, position: Position) -> list[TruthSite]:
    """The carrier as unlabelled truth sites (``heard`` pending, no staging fields), one per
    direction it sits in: per target pair its letter belongs to, or the held geminate. They
    are what ``direction_table`` reads a direction and a swap from."""
    reference = clip.references[position.segment]
    letter = reference[position.index]
    marks = ([(SHADDAH, HELD)] if position.geminate
             else [(pair, letter) for pair in LETTER_PAIRS[letter]])
    return [
        TruthSite(
            site_id=f"{clip.audio_filename}@{clip.starts[position.segment]}:"
                    f"{position.index}:{mark}",
            source=NEW_AUDIT, assumes_competent_reciter=False,
            audio_filename=clip.audio_filename, shard=None,
            start_sample=clip.starts[position.segment], end_sample=None, audio_sha256=None,
            surah_ayah=clip.surah_ayah, reference=reference, reference_index=position.index,
            mark=mark, prescribed=prescribed, heard=PENDING, stratum=CARRIER_STRATUM,
            stratum_population=0,
        )
        for mark, prescribed in marks
    ]


def site_rows(clips: Sequence[Clip], workers: int) -> list[SiteRow]:
    """Every carrier of ``clips`` as one :class:`~training.truth_scorer.SiteRow` per
    direction, with each arm's outcome, read in ``workers`` processes (alignment is the cost)."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(clip_commits, clips, chunksize=8))
    return [
        SiteRow(site, clip.weight, clip.reciter_id,
                {arm_name: outcomes[k] for arm_name, outcomes in commits.items()}, {})
        for clip, commits in zip(clips, results)
        for k, position in enumerate(clip.positions)
        for site in carrier_sites(clip, position)
    ]


# --- Rows: rates and paired gaps (``direction_table``) -----------------------------------------


def row_members(rows: Iterable[SiteRow], key: RowKey) -> list[SiteRow]:
    wanted = set(row_directions(key))
    return [row for row in rows if direction(row.site) in wanted]


def gap_name(first: str, second: str) -> str:
    return f"{first} − {second}"


def row_report(rows: Sequence[SiteRow], arms: Sequence[str],
               gaps: Sequence[tuple[str, str]]) -> dict:
    """``direction_table.summary`` (sites, reciters, every arm's rates), each arm's
    unweighted counts, and every paired swap-rate gap."""
    entry = summary(rows, arms)
    if rows:
        entry["counts"] = {a: counts(rows, a) for a in arms}
        entry["gaps"] = {gap_name(first, second): difference(rows, first, second)
                         for first, second in gaps}
    return entry


def table(rows: Sequence[SiteRow], arms: Sequence[str], gaps: Sequence[tuple[str, str]]
          ) -> list[dict]:
    return [{"family": key.family, "direction": key.direction,
             **row_report(row_members(rows, key), arms, gaps)} for key in ROWS]


def arm_gaps(protocols: Sequence[str], reference: str) -> dict[str, list[tuple[str, str]]]:
    """The paired gaps: **windowing** per model (each protocol − ``reference``) and
    **weights** per protocol (``h448`` − base)."""
    return {
        WINDOWING: [(arm(m, p), arm(m, reference)) for m in MODELS for p in protocols
                    if p != reference],
        WEIGHTS: [(arm("h448", p), arm("base", p)) for p in protocols],
    }


def gap_report(rows: Sequence[SiteRow], protocols: Sequence[str], reference: str) -> dict:
    """Every row's rates and gaps over ``rows`` for both models under ``protocols``, and how
    often nothing was committed."""
    arms = [arm(m, p) for m in MODELS for p in protocols]
    gaps = arm_gaps(protocols, reference)
    return {
        "arms": arms,
        "gaps": {kind: [gap_name(*pair) for pair in pairs] for kind, pairs in gaps.items()},
        "rows": table(rows, arms, [pair for pairs in gaps.values() for pair in pairs]),
        "uncommitted": uncommitted(rows, arms),
    }


def uncommitted(rows: Sequence[SiteRow], arms: Sequence[str]) -> dict:
    """Per arm and kind, carriers (each once, whatever its directions) with nothing committed:
    the part of ``other`` that is the alignment's, not a letter's."""
    carriers = {(r.site.audio_filename, r.site.start_sample, r.site.reference_index,
                 r.site.mark == SHADDAH): r for r in rows}
    kinds = {"pair": [r for k, r in carriers.items() if not k[3]],
             GEMINATE: [r for k, r in carriers.items() if k[3]]}
    return {a: {kind: {"carriers": len(group),
                       "uncommitted": sum(not r.outcomes[a].commits for r in group)}
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
        audio_filename=name,
        surah_ayah=plan.clip.surah_ayah,
        reciter_id=plan.clip.reciter_id,
        weight=1 / plan.clip.inclusion_probability,
        references=tuple(s.reference for s in plan.segments),
        starts=tuple(s.start_sample for s in plan.segments),
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


def pool_report(rows: Sequence[SiteRow], unseen: set[str]) -> dict:
    """The h448-unseen population and the whole cache, each over every pool arm."""
    groups = {UNSEEN: [r for r in rows if r.site.audio_filename in unseen], CACHE: rows}
    return {name: gap_report(group, POOL_PROTOCOLS, WHOLE) for name, group in groups.items()}


# --- Cross-checks ----------------------------------------------------------------------------


def owner_verdicts(sites: Iterable[ShipASite], verdicts: Mapping[str, Verdict]) -> dict:
    """Per target pair, at Ship A's answered partner swaps the base teacher got right, how
    often the owner heard the mushaf letter, the partner, another letter or could not tell."""
    tally: dict[str, Counter] = {pair: Counter() for pair in sorted(TARGET_PAIRS)}
    for site in sites:
        verdict = verdicts.get(site.site.site_id)
        if verdict is None or not (site.partner_swap and site.base_already_right):
            continue
        heard = verdict.heard
        outcome = (MUSHAF if heard == site.site.prescribed else SWAP if heard == partner(site.site)
                   else "unclear" if heard in NO_VERDICT else OTHER)
        tally[site.family][outcome] += 1
    soft = sum((tally[pair] for pair in sorted(SOFT_PAIRS)), Counter())
    return {**{pair: dict(c) for pair, c in tally.items()}, SOFT_PAIRS_FAMILY: dict(soft)}


def weights_of(identity: Mapping[str, str]) -> dict[str, str]:
    """A model identity's immutable part (a checkpoint's sha256 or the hub commit), without
    the reference it was loaded from, which two records may spell differently."""
    return {k: v for k, v in identity.items() if k not in ("ref", "model_ref")}


def app_decodes(items: Mapping[str, Located], registry: Mapping[str, StagedClip],
                pool_models: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    """Each item's decode per arm (both models, :data:`APP_PROTOCOLS`), read through the
    truth baseline's own checks (:func:`training.truth_baseline.model_caches`: the models'
    protocols, one set of weights for decodes and checks, every item decoded from its own
    audio) and refused unless the weights are the pool cache's (``pool_models``)."""
    decodes: dict[str, dict[str, str]] = {}
    for model in MODELS:
        cache, recorded = model_caches(model, DEFAULT_MODELS[model], items, registry, None)
        if weights_of(cache["identity"]) != weights_of(pool_models[model]):
            raise ValueError(f"{model}: the truth baseline decoded {cache['identity']}, the "
                             f"pool cache {pool_models[model]}")
        for protocol in PROTOCOLS:
            decodes[arm(model, protocol)] = {k: cache["items"][k][protocol] for k in items}
        decodes[arm(model, CHECKS)] = {k: committed_text(recorded["items"][k][CHECKS])
                                       for k in items}
    return decodes


def app_check(items: Mapping[str, Located], decodes: Mapping[str, Mapping[str, str]],
              registry: Mapping[str, StagedClip], workers: int) -> dict:
    """The pair carriers of truth-baseline items (``training.truth_baseline.items_of``: truth
    sites or sampled words by item), each item weighing 1, those tied to ``h448`` left out."""
    barred = unseen_violations({one.audio_filename: registry[one.audio_filename]
                                for one in items.values()})
    clips = [
        Clip(audio_filename=one.audio_filename, surah_ayah=one.surah_ayah,
             reciter_id=registry[one.audio_filename].reciter_id, weight=1.0,
             references=(one.reference,), starts=(one.start_sample,),
             positions=tuple(pair_positions(0, one.reference)), by_segment={},
             by_span={a: by_item[key] for a, by_item in decodes.items()})
        for key, one in sorted(items.items()) if one.audio_filename not in barred
    ]
    return {"items": len(items), "h448_linked_items": len(items) - len(clips),
            "kept_items": len(clips), "reciters": len({clip.reciter_id for clip in clips}),
            **gap_report(site_rows(clips, workers), APP_PROTOCOLS, SPANS)}


# --- Run ----------------------------------------------------------------------------------


def run(workers: int) -> dict:
    provenance = verified_provenance(CACHE_DIR, list(MODELS))
    caches = {model: load_cache(CACHE_DIR, model) for model in MODELS}
    settings = validated_settings(caches)
    registry = load_staged_clips()
    plans = plan_pool()
    cached = cached_plans(plans, caches)
    unseen = {clip.audio_filename for clip in unseen_clips([p.clip for p in cached], registry)[0]}
    pool = site_rows([pool_clip(plan, caches) for plan in cached], workers)
    diagnostic = items_of(load_truth_sites(DIAGNOSTIC_SITES), registry)
    words = items_of([word.site for word in load_words()], registry)
    decodes = app_decodes({**diagnostic, **words}, registry, provenance["models"])
    other_inputs = {
        **{f"truth_baseline/decodes/{m}.json": DECODE_STORE.path(m) for m in MODELS},
        **{f"truth_baseline/checks/{m}.json": CHECK_STORE.path(m) for m in MODELS},
        "truth_sites/waqf_boundaries.jsonl": DIAGNOSTIC_SITES,
        "ship_a/word_sample.jsonl": WORD_SAMPLE_PATH,
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
        "app_checks": {
            DIAGNOSTIC: app_check(diagnostic, decodes, registry, workers),
            WORD_SAMPLE: app_check(words, decodes, registry, workers),
        },
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

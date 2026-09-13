"""Yield arithmetic over a filtering run's reject sink — the shard-budget decision input.

A run with ``--rejects`` (:mod:`tadabur.rejects`) leaves three things behind: the passing
manifest, the reject sink, and the progress checkpoint saying how many clips were
consumed. This turns that trio into the numbers a corpus-scale commitment is decided on
— what fraction of clips the gate rejected, what those rejects failed on, and how many
of them are the **clean re-reads** Muraja ADR-0016 actually wants.

Two distributions are reported over the whole reject pile rather than just the matching
subset, on purpose: ``max_insertion_run`` and ``match_ratio`` are the two thresholds the
clean-re-read predicate is built from, so seeing their shape across *all* rejects is what
lets the thresholds be second-guessed from data instead of re-argued from first
principles. Causes are counted independently and therefore **overlap** — a clip can fail
the ratio bar and carry a long insertion run — so the shares sum to more than 100%; the
alternative, a made-up precedence order, would hide exactly the clips worth looking at.

Torch-free: everything it needs was already computed and written by the run.

Usage:
  python -m tadabur.reject_yield --passing passing.jsonl --rejects rejects.jsonl
    [--reject-audio-dir reject_audio/] [--json yield.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .manifest import read_clips_processed, read_records
from .rejects import REJECT_CAUSES, RejectRecord, read_reject_records

# Width of a ``match_ratio`` histogram bucket. 0.05 puts the two bars that matter — the
# ``.balanced`` pass bar at 0.65 and the clean-re-read floor at 0.75 — on bucket edges,
# so neither threshold is straddled by the bucket that is supposed to justify it.
RATIO_BUCKET = 0.05


@dataclass(frozen=True)
class RejectYield:
    """What one filtering run yielded, from the gate's verdicts alone.

    ``clips_processed`` is the checkpoint's count — every clip consumed from the stream,
    including the ``skipped_before_gate`` ones (over-long, or without a cached reference)
    that were dropped before the decode and so are in neither other bucket. Rates are
    fractions of ``clips_processed`` except ``clean_re_read_share_of_rejects``, which is
    the one that says how rich the *reject pile* is and therefore sizes the shard budget.
    """

    clips_processed: int
    passers: int
    rejects: int
    skipped_before_gate: int
    pass_rate: float
    reject_rate: float
    clean_re_reads: int
    clean_re_read_share_of_rejects: float
    clean_re_read_share_of_clips: float
    clean_re_read_reciters: int
    clean_re_read_ayat: int
    clean_re_read_audio_seconds: float
    cause_counts: dict[str, int] = field(default_factory=dict)
    cause_shares_of_rejects: dict[str, float] = field(default_factory=dict)
    insertion_run_histogram: dict[str, int] = field(default_factory=dict)
    match_ratio_histogram: dict[str, int] = field(default_factory=dict)
    clean_re_read_top_reciters: list[tuple[int, int]] = field(default_factory=list)


def compute_yield(
    passers: int, rejects: list[RejectRecord], clips_processed: int
) -> RejectYield:
    """Aggregate a run's rejects against its passer and clip counts."""
    clean = [record for record in rejects if record.is_clean_re_read]
    causes = Counter()
    for record in rejects:
        causes.update(record.causes)
    by_reciter = Counter(record.reciter_id for record in clean)

    return RejectYield(
        clips_processed=clips_processed,
        passers=passers,
        rejects=len(rejects),
        skipped_before_gate=clips_processed - passers - len(rejects),
        pass_rate=_share(passers, clips_processed),
        reject_rate=_share(len(rejects), clips_processed),
        clean_re_reads=len(clean),
        clean_re_read_share_of_rejects=_share(len(clean), len(rejects)),
        clean_re_read_share_of_clips=_share(len(clean), clips_processed),
        clean_re_read_reciters=len(by_reciter),
        clean_re_read_ayat=len({record.surah_ayah for record in clean}),
        clean_re_read_audio_seconds=round(
            sum(record.ayah_duration_s for record in clean), 1
        ),
        cause_counts={cause: causes[cause] for cause in REJECT_CAUSES if causes[cause]},
        cause_shares_of_rejects={
            cause: _share(causes[cause], len(rejects))
            for cause in REJECT_CAUSES
            if causes[cause]
        },
        insertion_run_histogram=_histogram(
            Counter(record.max_insertion_run for record in rejects)
        ),
        match_ratio_histogram=_histogram(
            Counter(_ratio_bucket(record.match_ratio) for record in rejects)
        ),
        clean_re_read_top_reciters=by_reciter.most_common(10),
    )


def shards_for(target: int, clean_re_reads: int, shards_run: int) -> float:
    """Shards needed to reach ``target`` clean re-reads at this run's measured rate.

    Deliberately unrounded and deliberately naive — it extrapolates one linear rate and
    nothing else. A run yielding none returns ``inf`` rather than a number, because the
    honest answer there is "this measurement cannot size a budget", not a large integer.
    """
    if clean_re_reads <= 0:
        return float("inf")
    return target * shards_run / clean_re_reads


def _share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _ratio_bucket(ratio: float) -> str:
    """Label the ``RATIO_BUCKET``-wide half-open bucket ``ratio`` falls in."""
    low = min(int(ratio / RATIO_BUCKET) * RATIO_BUCKET, 1.0 - RATIO_BUCKET)
    return f"{low:.2f}-{low + RATIO_BUCKET:.2f}"


def _histogram(counter: Counter) -> dict[str, int]:
    """Render a counter as a key-ordered ``{str: count}`` dict, for stable JSON."""
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passing", type=Path, required=True, help="The run's passing-subset manifest."
    )
    parser.add_argument(
        "--rejects", type=Path, required=True, help="The run's reject sink JSONL."
    )
    parser.add_argument(
        "--shards-run",
        type=int,
        default=1,
        help="Shards this run covered, for the shard-budget extrapolation.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=500,
        help="Clean re-read clips the budget should reach (default: 500).",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write the yield as JSON here."
    )
    args = parser.parse_args()

    result = compute_yield(
        passers=len(read_records(args.passing)),
        rejects=read_reject_records(args.rejects),
        clips_processed=read_clips_processed(args.passing),
    )
    payload = asdict(result)
    payload["shards_for_target"] = round(
        shards_for(args.target, result.clean_re_reads, args.shards_run), 2
    )
    payload["target"] = args.target
    payload["shards_run"] = args.shards_run
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

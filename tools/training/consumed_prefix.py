"""The shard prefix a streamed run consumed, and the shards a continuation has left.

``StreamingWindowDataset`` walks a run's shard list in a seeded permutation and prints one
line per shard as it leaves it, so a finished run's log states the order it actually
streamed. That order is reproducible -- and it has to be, because a continuation that
retrains on audio the earlier run already streamed is not the experiment its note claims.

The pieces are kept apart so each can be checked on its own: the permutation
(:func:`shard_permutation`), the log's reading of it (:func:`completion_order`), the
cross-check that turns the two into the consumed set (:func:`consumed_prefix`), and the
complement (:func:`remaining_shards`) with the ``--stream-shards`` spec it launches with
(:func:`shard_spec`).

One subtlety of the log: a shard's line is printed when the *next* shard's first row
arrives, so the last completed shard is not the last shard entered. A run stopped at step
N was inside one more, and that one counts as consumed: some of its rows were streamed.

Usage::

    # The --stream-shards spec a continuation runs with, from the source run's log
    python -m training.consumed_prefix --log /root/scratch/issue-94/logs/aug94_candidate_v2.log
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

#: The line ``training.distill_stream.exclude_rows`` prints when it leaves a shard.
COMPLETION_LINE = re.compile(r"\[data\] exposure exclusion through shard (\d+)\b")

#: #94's seed. ``h448`` used 1234; both #94 arms used 9494, so that is C95's stream order.
AUG94_SEED = 9494


def shard_permutation(shards: Iterable[int], seed: int) -> list[int]:
    """The order a one-worker stream walks ``shards`` in, for ``seed``.

    Sorted first: the stream shuffles the list its spec parsed to and a spec is sorted, so
    this is that order without silently depending on how the caller built its list.
    """
    ordered = sorted(shards)
    random.Random(seed).shuffle(ordered)
    return ordered


def completion_order(log_text: str) -> list[int]:
    """The shards a run's log reports completed, in the order it reported them."""
    return [int(match.group(1)) for match in COMPLETION_LINE.finditer(log_text)]


def consumed_prefix(
    shards: Iterable[int], completed: Sequence[int], seed: int
) -> tuple[int, ...]:
    """The shards a run entered: every completed one, plus the one it stopped inside.

    ``completed`` must be the head of the permutation, which is the cross-check: a log from
    another seed, another shard list or another machine would otherwise be accepted as
    evidence for this one, and the shards it really consumed would be trained on again.
    """
    permutation = shard_permutation(shards, seed)
    observed = list(completed)
    if not observed:
        # A rotated, truncated or wrongly-pathed log would otherwise "prove" an empty prefix
        # and silently move the whole corpus back into a continuation's training set.
        raise ValueError(
            "the log has no shard-completion lines: it is empty, truncated, or not a "
            f"streamed run's log (expected '[data] exposure exclusion through shard N')"
        )
    if observed != permutation[: len(observed)]:
        first = next(
            (i for i, (seen, expected) in enumerate(zip(observed, permutation)) if seen != expected),
            len(observed),
        )
        raise ValueError(
            f"the log's shard order is not the seed-{seed} permutation of these shards: "
            f"position {first} is {observed[first:first + 1]} in the log and "
            f"{permutation[first:first + 1]} in the permutation"
        )
    return tuple(permutation[: len(observed) + 1])


def remaining_shards(shards: Iterable[int], consumed: Iterable[int]) -> list[int]:
    """``shards`` minus ``consumed``, sorted."""
    return sorted(set(shards) - set(consumed))


def shard_spec(shards: Iterable[int]) -> str:
    """The compact ``--stream-shards`` spec for a shard list, as ``parse_shard_spec`` reads
    it: ``[1, 2, 3, 7]`` → ``"1-3,7"``."""
    indices = sorted(set(shards))
    if not indices:
        raise ValueError("no shards: an empty spec would stream nothing")
    terms: list[str] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        terms.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    terms.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(terms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, required=True,
                        help="the source run's stdout log, with its shard-completion lines")
    parser.add_argument("--seed", type=int, default=AUG94_SEED)
    parser.add_argument(
        "--shards", default="",
        help="the source run's --stream-shards (default: h448's training spec, the only "
             "shard set a training run may use)",
    )
    args = parser.parse_args()

    from tadabur.shard_reader import parse_shard_spec
    from training.decode_evalset import training_shard_spec

    shards = parse_shard_spec(args.shards or training_shard_spec())
    completed = completion_order(args.log.read_text(encoding="utf-8"))
    consumed = consumed_prefix(shards, completed, args.seed)
    remaining = remaining_shards(shards, consumed)
    print(
        f"{len(completed)} completions, {len(consumed)} shards consumed, "
        f"{len(remaining)} remaining",
        file=sys.stderr,
    )
    print(shard_spec(remaining))


if __name__ == "__main__":
    main()

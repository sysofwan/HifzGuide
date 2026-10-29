"""The shard prefix a streamed run consumed, and the shards a continuation has left.

``StreamingWindowDataset`` walks a run's shard list in a seeded permutation and prints one
line per shard as it leaves it, so a finished run's log states the order it actually
streamed. That order is reproducible -- and it has to be, because a continuation that
retrains on audio the earlier run already streamed is not the experiment its note claims.

The pieces are kept apart so each can be checked on its own: the permutation
(:func:`shard_permutation`), the log's reading of it (:func:`completion_order`), the
cross-check that turns the two into the consumed set (:func:`consumed_prefix`), and the
complement (:func:`remaining_shards`) written as a ``--stream-shards`` spec by
:func:`tadabur.shard_reader.shard_spec`, the parser's inverse.

One subtlety of the log: a shard's line is printed when the *next* shard's first row
arrives, so the last completed shard is not the last shard entered. A run stopped at step
N was inside one more, and that one counts as consumed: some of its rows were streamed.

A continuation also needs a shard it does *not* train on: validation clips staged from a shard
the run streams turn the health signal in-sample. :func:`monitoring_shard` names that shard
(the last of the run's own seed permutation, which its step budget reaches last) and the
completion count at which the run enters it, and :func:`monitor_reached` checks a finished
log against it -- so "the validation audio was never streamed" is asserted, not assumed.

Usage::

    # The --stream-shards spec a continuation runs with, from the source run's log
    python -m training.consumed_prefix --log /root/scratch/issue-94/logs/aug94_candidate_v2.log

    # ... with the monitoring shard and the point it would be entered (monitor=, position=)
    python -m training.consumed_prefix --log <log> --monitor-seed 178

    # Two rounds deep (:func:`remaining_after_logs`), what a second continuation trains on
    remaining_after_logs([(c95_log, AUG94_SEED), (continuation_log, 178)])
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from tadabur.shard_reader import shard_spec

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


def monitoring_shard(shards: Iterable[int], seed: int) -> tuple[int, int]:
    """The shard a run monitors on, and how many shards it must finish to reach it.

    The monitor shard is the last of the run's seed permutation, so a run of the expected
    step budget never enters it -- but that is arithmetic over steps per shard, not a
    property of the stream. :func:`monitor_reached` turns the log's completions into a check.
    """
    permutation = shard_permutation(shards, seed)
    return permutation[-1], len(permutation) - 1


def monitor_reached(shards: Iterable[int], completed: Sequence[int], seed: int) -> bool:
    """Whether the run whose shard-completion lines these are entered the monitor shard.

    A completion line means the shard was *left*, so the run is inside shard *i* once *i*
    earlier shards are done: the monitor shard is entered at its own position, not after it.
    """
    _, position = monitoring_shard(shards, seed)
    return len(list(completed)) >= position


def remaining_after_logs(
    runs: Sequence[tuple[Path, int]], *, shards: str = ""
) -> list[int]:
    """The shards left after each run in ``runs`` consumed its prefix of what was left.

    ``runs`` is ``(log, stream seed)`` in the order the runs happened: a continuation inherits
    its predecessor's remainder as its own shard list, so a second round's corpus is the
    intersection of what both earlier rounds left. Each log's order is checked against its own
    seed permutation of that remainder (:func:`consumed_prefix`), so a log from another run
    cannot be chained in.

    ``shards`` overrides the first run's ``--stream-shards``; by default it is ``h448``'s
    training spec, the only shard set a training run may stream.
    """
    from tadabur.shard_reader import parse_shard_spec
    from training.decode_evalset import training_shard_spec

    remaining = parse_shard_spec(shards or training_shard_spec())
    for log, seed in runs:
        completed = completion_order(log.read_text(encoding="utf-8"))
        remaining = remaining_shards(remaining, consumed_prefix(remaining, completed, seed))
    return remaining


def remaining_from_log(
    log: Path, *, seed: int = AUG94_SEED, shards: str = ""
) -> list[int]:
    """The shards a continuation of the run whose log this is has left."""
    return remaining_after_logs([(log, seed)], shards=shards)


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
    parser.add_argument(
        "--monitor-seed", type=int, default=None,
        help="also report the shard a run with this seed monitors on and the completion "
             "count at which it would enter it, as 'monitor=<n>' and 'position=<n>' lines "
             "ahead of the spec",
    )
    args = parser.parse_args()

    remaining = remaining_from_log(args.log, seed=args.seed, shards=args.shards)
    print(f"{len(remaining)} shards remaining after the source run's consumed prefix",
          file=sys.stderr)
    if args.monitor_seed is not None:
        monitor, position = monitoring_shard(remaining, args.monitor_seed)
        print(f"monitor shard {monitor} is entered after {position} shards are completed",
              file=sys.stderr)
        print(f"monitor={monitor}")
        print(f"position={position}")
        print(f"stream={shard_spec(remaining)}")
        return
    print(shard_spec(remaining))


if __name__ == "__main__":
    main()

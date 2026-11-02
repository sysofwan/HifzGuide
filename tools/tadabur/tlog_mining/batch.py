"""One batch of phase A's work (#263): whole shards per worker, one part per shard, resume by key.

A worker takes the shards ``sorted(shards)[index::count]``, so two workers never share a shard
and the rows of a shard stay together — the same split :mod:`tadabur.tlog_eval_decode` decodes
with. Each shard's records are appended to ``word_times.<shard_file>.jsonl`` as each clip
finishes, which is what makes a batch preemptible: the give-way loop stops only between batches,
and a batch stops only between rows (the budget is checked **before** a row, so no clip is ever
cut off). A resumed batch re-reads the part's keys
(:func:`training.external_decode.resume_keys`, which also cuts off a half-written last line from
a killed run) and aligns only the rows the part does not hold yet.

The record's ``key`` is checked against the row's before it is written: a record for another
clip is a broken aligner, not a record to keep.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from training.external_decode import resume_keys
from training.nonpro_decode import item_key


@dataclass(frozen=True)
class BatchResult:
    """One batch's outcome: whether no row is left, and how many it wrote."""

    done: bool
    written: int


def part_path(parts_dir: Path, shard_file: str) -> Path:
    """One shard's part file under ``parts_dir``."""
    return Path(parts_dir) / f"word_times.{shard_file}.jsonl"


def worker_rows(rows: Sequence[Mapping], *, shard_index: int, shard_count: int) -> list[dict]:
    """The rows of ``sorted(shards)[shard_index::shard_count]``, in the input's order."""
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), not {shard_index}")
    by_shard: dict[str, list[dict]] = {}
    for row in rows:
        by_shard.setdefault(row["shard_file"], []).append(dict(row))
    return [row for shard in sorted(by_shard)[shard_index::shard_count]
            for row in by_shard[shard]]


def run_batch(rows: Sequence[Mapping], parts_dir: Path, *,
              align_row: Callable[[Mapping], dict], budget_seconds: float,
              clock: Callable[[], float],
              prepare_shard: Callable[[str, Sequence[Mapping]], None] | None = None) -> BatchResult:
    """Align the pending rows of ``rows`` in (shard_file, row) order within ``budget_seconds``.

    The budget is measured from the batch's start and checked before each row; a row already in
    its part is skipped without costing budget. A failing ``align_row`` propagates and leaves the
    rows written so far in place. ``done`` is true only when every row of ``rows`` is held.

    ``prepare_shard(shard_file, pending)`` is called once per shard with rows left, before its
    first row: the place to stream the shard's clips once (the source blob is deleted after a
    read, so fetching per row would re-download the whole shard per clip).
    """
    started = clock()
    by_shard: dict[str, list[Mapping]] = {}
    for row in rows:
        by_shard.setdefault(row["shard_file"], []).append(row)
    written = 0
    stopped = False
    for shard_file in sorted(by_shard):
        path = part_path(parts_dir, shard_file)
        done = resume_keys(path)
        left = [row for row in sorted(by_shard[shard_file], key=lambda row: row["row"])
                if item_key(row["audio_path"]) not in done]
        if not left or stopped:
            continue
        if prepare_shard is not None:
            prepare_shard(shard_file, left)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in left:
                if clock() - started >= budget_seconds:
                    stopped = True
                    break
                record = align_row(row)
                expected = item_key(row["audio_path"])
                if record.get("key") != expected:
                    raise ValueError(f"{expected}: the aligner returned a record whose key is "
                                     f"{record.get('key')!r}")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                written += 1
    return BatchResult(done=not pending(rows, parts_dir), written=written)


def pending(rows: Sequence[Mapping], parts_dir: Path) -> bool:
    """Whether any of ``rows`` has no record in its shard's part yet."""
    return not _all_held(rows, parts_dir)


def _all_held(rows: Sequence[Mapping], parts_dir: Path) -> bool:
    """Whether every row's key is in its shard's part (after a batch's writes)."""
    by_shard: dict[str, set[str]] = {}
    for row in rows:
        by_shard.setdefault(row["shard_file"], set()).add(item_key(row["audio_path"]))
    for shard_file, keys in by_shard.items():
        if not keys <= resume_keys(part_path(parts_dir, shard_file)):
            return False
    return True

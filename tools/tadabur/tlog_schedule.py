"""The frozen, seeded shard schedule of the tlog training reader (#224).

``tlog_pool.iter_training_rows`` used to walk the training rows in ``(split, shard_file, row)``
order. Clean shards are filename-sorted and each holds a narrow surah range, so a 10k-step run
read a prefix of shards 0-121: 34 of 114 surahs and no ``unclean`` row. The schedule fixes the
draw without touching a row: the shards holding at least one ``tlog.training`` row, both splits,
are shuffled by :data:`SEED` and read **eight at a time**, one row from each open shard per
round, so any prefix of the stream is representative.

``tools/tadabur/tlog/training_schedule.json`` is metadata only -- no audio, label, decode or
agreement -- and is committed in the round-2 freeze, whose pre-registration cites its SHA-256
(``tadabur.tlog_index.sha256_file``; the file cannot hold its own hash). It records the seed, the
interleave width, the overlay commit, and the ``tlog.training`` use's count and ``rows_sha256``,
so the reader can refuse a schedule built for another overlay.

``tools/tadabur/tlog/training_prefix.json`` is the plan for the round-2 run's window budget: the
rows, clips, windows and real-audio seconds of each split, the shards touched, the share of the
pool and the clean surahs covered, walking the same :func:`interleave_order` the reader uses.

Usage (from ``tools/``)::

  python -m tadabur.tlog_schedule build                  # write training_schedule.json
  python -m tadabur.tlog_schedule plan --windows 240000  # write training_prefix.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .nonpro_sources import TLOG_SPLITS
from .tlog_index import sha256_file, write_json

#: The frozen seed: the round-2 pre-registration's draw, reproducible from this string alone.
SEED = "round2-tlog-shards-2026-10"
#: The architect's interleave: eight shards open at once (r29).
INTERLEAVE_WIDTH = 8
#: The round-2 run's tlog window budget (about 10k steps at 24 tlog windows per batch).
DEFAULT_PREFIX_WINDOWS = 240_000
SCHEDULE_PATH = Path(__file__).parent / "tlog" / "training_schedule.json"
PREFIX_PATH = Path(__file__).parent / "tlog" / "training_prefix.json"

_FIELDS = frozenset({"seed", "interleave_width", "overlay_commit", "rows_sha256", "count",
                     "shards"})
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Schedule:
    """The frozen read order: ``shards`` are ``(split, shard_file)`` pairs, read in this order."""

    seed: str
    interleave_width: int
    overlay_commit: str
    rows_sha256: str
    count: int
    shards: tuple[tuple[str, str], ...]


def shard_of(row: Mapping[str, Any]) -> tuple[str, str]:
    """One row's shard: ``(split, shard_file)``."""
    return row["split"], row["shard_file"]


def _positive_width(width: Any) -> int:
    """``width`` as an interleave width: a positive int, never a bool or a float."""
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError(f"an interleave width is a positive int, not {width!r}")
    return width


def build_schedule(rows: Iterable[Mapping[str, Any]], use, *, seed: str = SEED,
                   width: int = INTERLEAVE_WIDTH) -> Schedule:
    """The seeded shuffle of every shard holding a row of ``rows``.

    ``use`` is the ``tlog.training`` :class:`tadabur.exposure.RowSetExposure` whose overlay
    commit, count and ``rows_sha256`` the schedule records. Only the shard set is drawn: the
    rows themselves never move, so the pool's membership is unchanged.
    """
    width = _positive_width(width)
    shards = sorted({shard_of(row) for row in rows})
    random.Random(seed).shuffle(shards)
    return Schedule(seed, width, use.overlay_commit, use.rows_sha256, use.count, tuple(shards))


def write_schedule(schedule: Schedule, path: Path = SCHEDULE_PATH) -> None:
    """Write the schedule as its one JSON object, atomically and with stable bytes."""
    write_json(asdict(schedule), Path(path))


def read_schedule(path: Path = SCHEDULE_PATH) -> Schedule:
    """Parse and validate a schedule file; a missing or extra field, or a malformed value,
    refuses rather than being guessed at."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not a readable schedule: {error}") from error
    return _parse_schedule(data, where=str(path))


def _parse_schedule(data: Any, *, where: str) -> Schedule:
    """One schedule object from JSON, validated field by field."""
    if not isinstance(data, dict) or set(data) != _FIELDS:
        raise ValueError(f"{where} does not hold exactly the fields {sorted(_FIELDS)}")
    if not isinstance(data["seed"], str) or not data["seed"]:
        raise ValueError(f"{where}: seed is not a non-empty string")
    try:
        width = _positive_width(data["interleave_width"])
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
    if not isinstance(data["overlay_commit"], str) or not _HEX_40.fullmatch(
            data["overlay_commit"]):
        raise ValueError(f"{where}: overlay_commit is not 40 lowercase hex")
    if not isinstance(data["rows_sha256"], str) or not _HEX_64.fullmatch(data["rows_sha256"]):
        raise ValueError(f"{where}: rows_sha256 is not 64 lowercase hex")
    count = data["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{where}: count is not a non-negative int")
    shards = data["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"{where}: shards is not a non-empty list")
    parsed: list[tuple[str, str]] = []
    for shard in shards:
        if (not isinstance(shard, list) or len(shard) != 2
                or not all(isinstance(part, str) for part in shard)
                or shard[0] not in TLOG_SPLITS or not shard[1]):
            raise ValueError(f"{where}: {shard!r} is not a [split, shard_file] pair")
        parsed.append((shard[0], shard[1]))
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{where} lists a shard twice")
    return Schedule(data["seed"], width, data["overlay_commit"], data["rows_sha256"], count,
                    tuple(parsed))


def rows_by_shard(rows: Iterable[Mapping[str, Any]],
                  schedule: Schedule) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """``rows`` grouped by shard in row order, refusing a schedule that is not their shards.

    The schedule must name every shard holding a row exactly once: a missing, duplicated or
    extra shard refuses, so a schedule built for another overlay cannot be walked silently.
    """
    listed = list(schedule.shards)
    if len(set(listed)) != len(listed):
        raise ValueError("the schedule lists a shard twice")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda row: (row["split"], row["shard_file"], row["row"])):
        grouped[shard_of(row)].append(row)
    missing = sorted(set(grouped) - set(listed))
    extra = sorted(set(listed) - set(grouped))
    if missing or extra:
        raise ValueError(f"the schedule does not cover the rows' shards exactly: "
                         f"missing {missing}, extra {extra}")
    return dict(grouped)


def interleave_streams(streams: Mapping[tuple[str, str], Iterable[Any]],
                       schedule: Schedule) -> Iterator[Any]:
    """One item per open shard per round, ``schedule.interleave_width`` shards at a time.

    Slots fill from the front of ``schedule.shards`` and an exhausted slot is refilled at once
    by the next scheduled shard, so the stream stays interleaved until one shard is left. Each
    ``streams`` value is consumed in order and once, and every scheduled shard must hold at
    least one item; the caller owns closing them, so a raise, a ``close()`` or an early stop can
    drop whatever is still open.
    """
    scheduled = list(schedule.shards)
    width = schedule.interleave_width
    slots = [(shard, iter(streams[shard])) for shard in scheduled[:width]]
    pending = iter(scheduled[width:])
    while slots:
        live = []
        for shard, stream in slots:
            try:
                item = next(stream)
            except StopIteration:
                replacement = next(pending, None)
                if replacement is None:
                    continue
                shard = replacement
                stream = iter(streams[shard])
                item = next(stream)
            yield item
            live.append((shard, stream))
        slots = live


def interleave_order(rows: Iterable[Mapping[str, Any]],
                     schedule: Schedule) -> Iterator[Mapping[str, Any]]:
    """``rows`` in the stream order the reader yields them in: the shards of the schedule,
    ``interleave_width`` open at once, row order within a shard. The plan walks this too, so the
    planned prefix is the run's."""
    return interleave_streams(rows_by_shard(rows, schedule), schedule)


def split_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int | float]]:
    """Per split: ``rows``, ``clips``, ``windows`` and ``audio_seconds`` (real audio read).

    A clip is one row, so rows and clips agree; windows are
    :func:`tadabur.tlog_pool._windows_per_clip` at the round-2 5 s window and 2.5 s hop, and
    ``audio_seconds`` the sum of the rows' ``duration_s``, rounded to the millisecond so a
    committed report shows seconds rather than float noise.
    """
    from .tlog_pool import _windows_per_clip

    totals: dict[str, dict[str, int | float]] = {}
    for row in rows:
        entry = totals.setdefault(row["split"], {"rows": 0, "clips": 0, "windows": 0,
                                                 "audio_seconds": 0.0})
        entry["rows"] += 1
        entry["clips"] += 1
        entry["windows"] += _windows_per_clip(row)
        entry["audio_seconds"] += row["duration_s"]
    for entry in totals.values():
        entry["audio_seconds"] = round(entry["audio_seconds"], 3)
    return totals


def _surah(row: Mapping[str, Any]) -> int | None:
    """A clean row's surah, from its ``surah:ayah`` filename label; None without one."""
    label = row.get("filename_label")
    return int(label.split(":", 1)[0]) if label else None


def _clean_surahs(rows: Iterable[Mapping[str, Any]]) -> list[int]:
    """The sorted surahs of the clean rows that carry a filename label."""
    return sorted({surah for row in rows
                   if row["split"] == "clean" and (surah := _surah(row)) is not None})


def plan_prefix(rows: Iterable[Mapping[str, Any]], schedule: Schedule, *, windows: int) -> dict:
    """The planned prefix of a run of ``windows`` tlog windows, walking
    :func:`interleave_order`.

    The walk stops at the first row that brings the cumulative window count to ``windows`` (that
    row is included, so the plan never under-reads the budget), and reports the rows, clips,
    windows and real seconds per split, the shards touched, the prefix's share of the pool, and
    the clean surahs covered against the pool's. Surah coverage covers clean rows only: unclean
    rows carry no filename label.
    """
    if windows < 1:
        raise ValueError(f"a prefix window target is positive, not {windows!r}")
    from .tlog_pool import _windows_per_clip

    pool = list(rows)
    taken: list[Mapping[str, Any]] = []
    running = 0
    for row in interleave_order(pool, schedule):
        taken.append(row)
        running += _windows_per_clip(row)
        if running >= windows:
            break
    totals = split_totals(taken)
    return {
        "target_windows": windows,
        "rows": len(taken),
        "clips": len(taken),
        "windows": running,
        "shards_touched": len({shard_of(row) for row in taken}),
        "share_of_pool": len(taken) / len(pool) if pool else 0.0,
        "splits": {split: {**total,
                           "shards_touched": len({shard_of(row) for row in taken
                                                  if row["split"] == split})}
                   for split, total in totals.items()},
        "surahs_covered": _clean_surahs(taken),
        "surahs_in_pool": _clean_surahs(pool),
    }


# --- CLI ----------------------------------------------------------------------------------------


def build(*, schedule_path: Path = SCHEDULE_PATH) -> Schedule:
    """The committed use's schedule, written to ``schedule_path``."""
    from . import exposure
    from .tlog_pool import load_training_rows

    use = exposure.load_registry().row_sets[exposure.TLOG_TRAINING]
    schedule = build_schedule(load_training_rows(), use)
    write_schedule(schedule, schedule_path)
    return schedule


def plan(*, windows: int = DEFAULT_PREFIX_WINDOWS, schedule_path: Path = SCHEDULE_PATH,
         prefix_path: Path = PREFIX_PATH) -> dict:
    """The committed schedule's planned prefix, written to ``prefix_path``."""
    from .tlog_pool import load_training_rows

    report = plan_prefix(load_training_rows(), read_schedule(schedule_path), windows=windows)
    write_json(report, Path(prefix_path))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build_ = commands.add_parser("build", help="write the frozen shard schedule")
    build_.add_argument("--schedule-path", type=Path, default=SCHEDULE_PATH)
    plan_ = commands.add_parser("plan", help="write the planned prefix of a window target")
    plan_.add_argument("--windows", type=int, default=DEFAULT_PREFIX_WINDOWS)
    plan_.add_argument("--schedule-path", type=Path, default=SCHEDULE_PATH)
    plan_.add_argument("--prefix-path", type=Path, default=PREFIX_PATH)
    args = parser.parse_args(argv)
    if args.command == "build":
        schedule = build(schedule_path=args.schedule_path)
        result = {"count": schedule.count, "interleave_width": schedule.interleave_width,
                  "overlay_commit": schedule.overlay_commit, "rows_sha256": schedule.rows_sha256,
                  "seed": schedule.seed, "shards": len(schedule.shards),
                  "schedule_sha256": sha256_file(Path(args.schedule_path))}
    else:
        result = plan(windows=args.windows, schedule_path=args.schedule_path,
                      prefix_path=args.prefix_path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

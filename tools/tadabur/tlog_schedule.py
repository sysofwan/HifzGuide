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
so :func:`require_schedule` can refuse a schedule built for another overlay or another pool
before the reader downloads a shard or the planner writes a report. Injected rows are the
synthetic seam: they are checked against the count, hash and shard coverage, not the registry.

``tools/tadabur/tlog/training_prefix.json`` is the plan for the round-2 run's window budget: the
rows, clips, windows and real-audio seconds of each split, the shards touched, the share of the
pool and the clean surahs covered, walking the same :func:`interleave_order` the reader uses.

``tools/tadabur/tlog/consumed/`` holds the keys a run consumed -- round 2's, byte for byte
(``consonant2.jsonl.gz`` beside its ``meta.json``). A later round continues the frozen walk past
them: :func:`require_walk_prefix` proves the key set is the walk's first rows,
:func:`tadabur.tlog_pool.iter_training_rows` starts there, and ``plan --after`` reports the next
prefix with the cumulative surah coverage (#250).

Usage (from ``tools/``)::

  python -m tadabur.tlog_schedule build                  # write training_schedule.json
  python -m tadabur.tlog_schedule plan --windows 240000  # write training_prefix.json
  python -m tadabur.tlog_schedule plan --after tadabur/tlog/consumed/consonant2.jsonl.gz \\
      --windows 240000 --out /tmp/training_prefix.round3.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
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
_KEY_FIELDS = frozenset({"split", "shard_file", "row"})
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


def require_schedule(schedule: Schedule, rows: Sequence[Mapping[str, Any]], *, use=None
                     ) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """Refuse unless ``schedule`` is this run's, checked before any download or report.

    ``use`` is the verified :class:`tadabur.exposure.RowSetExposure` the rows came from: the
    schedule's recorded overlay must be that use's, because a corrected overlay with the same
    membership can still change the window budget its report froze. The rows' count and
    ``rows_sha256`` must be the schedule's, and its shards must be exactly theirs. Returns the
    rows grouped by shard in row order -- the grouping every walk needs anyway.
    """
    from .tlog_pool import rows_digest

    if use is not None and schedule.overlay_commit != use.overlay_commit:
        raise ValueError(f"the schedule is for overlay {schedule.overlay_commit}, not the "
                         f"verified use's {use.overlay_commit}")
    by_shard = rows_by_shard(rows, schedule)
    digest = rows_digest(rows)
    if (len(rows), digest) != (schedule.count, schedule.rows_sha256):
        raise ValueError(f"the schedule is for {schedule.count} rows with rows_sha256 "
                         f"{schedule.rows_sha256}, not these {len(rows)} rows with {digest}")
    return by_shard


def interleave_order(rows: Iterable[Mapping[str, Any]],
                     schedule: Schedule) -> Iterator[Mapping[str, Any]]:
    """``rows`` in the stream order the reader yields them in: the shards of the schedule,
    ``interleave_width`` open at once, row order within a shard. The plan walks this too, so the
    planned prefix is the run's."""
    return interleave_streams(rows_by_shard(rows, schedule), schedule)


# --- the walk prefix a later round starts after (#250) -------------------------------------------

Key = tuple[str, str, int]


@dataclass(frozen=True)
class ConsumedPrefix:
    """The walk rows a run already consumed: a later round starts after them (#250).

    ``keys`` is the set of ``(split, shard_file, row)`` the consumed-keys log holds, ``length``
    the walk length they cover, and ``line_order_is_walk_order`` whether the log's lines are in
    the walk's order -- reported, never required: the reader appends as it yields, but a copy
    may be reordered without changing what was consumed.
    """

    keys: frozenset[Key]
    length: int
    line_order_is_walk_order: bool


def read_consumed_keys(path: Path) -> list[Key]:
    """A consumed-keys log's keys in line order: one ``{"split", "shard_file", "row"}`` object
    per line and nothing else, which is exactly what the reader appends.

    The walk is rebuilt from the pool and the schedule, never from this file: the file says
    which rows a run read, and :func:`require_walk_prefix` says whether they are the walk's
    first rows.
    """
    keys: list[Key] = []
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not JSON: {error}") from error
            if not isinstance(record, dict) or set(record) != _KEY_FIELDS:
                raise ValueError(f"{path}:{number} is not a consumed-keys record: {record!r}")
            split, shard, row = record["split"], record["shard_file"], record["row"]
            if (not isinstance(split, str) or split not in TLOG_SPLITS
                    or not isinstance(shard, str) or not shard
                    or isinstance(row, bool) or not isinstance(row, int) or row < 0):
                raise ValueError(f"{path}:{number} is not a consumed key: {record!r}")
            keys.append((split, shard, row))
    return keys


def _walk_head(by_shard: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
               schedule: Schedule, length: int) -> list[Mapping[str, Any]]:
    """The walk's first ``length`` rows from already-grouped rows; a shorter walk refuses."""
    head = list(islice(interleave_streams(by_shard, schedule), length))
    if len(head) != length:
        raise ValueError(f"the consumed keys name {length} rows but the walk has {len(head)}")
    return head


def _require_walk_keys(keys: Iterable[Key], head: Sequence[Mapping[str, Any]], *,
                       where: str) -> list[Key]:
    """Refuse unless ``keys`` is, as a set, exactly ``head``'s keys; return the walk's keys."""
    from .tlog_pool import row_key

    walked = [row_key(row) for row in head]
    named, named_walk = frozenset(keys), frozenset(walked)
    if named != named_walk:
        absent = sorted(named_walk - named)
        foreign = sorted(named - named_walk)
        raise ValueError(f"{where} are not the walk's first {len(head)} rows: "
                         f"{len(absent)} of the walk's rows are absent (e.g. {absent[:1]}), "
                         f"{len(foreign)} are not in it (e.g. {foreign[:1]})")
    return walked


def require_walk_prefix(keys: Iterable[Key], rows: Iterable[Mapping[str, Any]],
                        schedule: Schedule) -> ConsumedPrefix:
    """Verify that ``keys`` (in line order) are exactly the walk's first ``len(keys)`` rows.

    The walk is :func:`interleave_order`; its key set, not the line order, must equal the
    keys'. A duplicate, a key that is not in the prefix, a missing key, a key no row of the
    pool has, and a file longer than the walk all refuse. ``rows`` must be the schedule's rows:
    :func:`require_schedule` is the caller's, because this walk is metadata only and never
    touches a shard.
    """
    keys = list(keys)
    if len(set(keys)) != len(keys):
        raise ValueError(f"the consumed keys repeat a key: {len(keys)} lines, "
                         f"{len(set(keys))} distinct")
    head = _walk_head(rows_by_shard(rows, schedule), schedule, len(keys))
    walked = _require_walk_keys(keys, head, where="the consumed keys")
    return ConsumedPrefix(frozenset(keys), len(keys), walked == keys)


def next_walk_key(keys: Iterable[Key], rows: Iterable[Mapping[str, Any]],
                  schedule: Schedule) -> Key:
    """The walk row a resumed run reads first after ``keys``: the smoke gate's offset
    assertion (#252).

    ``keys`` are re-checked as the walk's first rows, exactly as :func:`require_walk_prefix`
    does, so a log that names rows the walk does not hold -- or a duplicate -- refuses here
    rather than shifting the offset silently.
    """
    keys = list(keys)
    if len(set(keys)) != len(keys):
        raise ValueError(f"the consumed keys repeat a key: {len(keys)} lines, "
                         f"{len(set(keys))} distinct")
    head = _walk_head(rows_by_shard(rows, schedule), schedule, len(keys) + 1)
    _require_walk_keys(keys, head[:-1], where="the consumed keys")
    from .tlog_pool import row_key

    return row_key(head[-1])


def require_prefix_by_shard(after: ConsumedPrefix,
                            by_shard: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
                            schedule: Schedule) -> dict[tuple[str, str], int]:
    """The reader's re-check of a prefix verified elsewhere, and its per-shard counts.

    ``by_shard`` is the reader's validated grouping (:func:`require_schedule`). A prefix whose
    key count is not its walk length, or whose keys are not this walk's first rows, refuses
    here, before a shard is opened: a prefix verified against other rows cannot start this run
    into rows it already read. The counts are what each shard's stream starts past, so the
    shards open at the cut continue mid-shard rather than being re-interleaved from scratch.
    """
    if len(after.keys) != after.length:
        raise ValueError(f"the consumed prefix is {after.length} walk rows but holds "
                         f"{len(after.keys)} keys")
    head = _walk_head(by_shard, schedule, after.length)
    _require_walk_keys(after.keys, head, where="the consumed prefix")
    return dict(Counter(shard_of(row) for row in head))


def add_split_row(totals: dict[str, dict[str, int | float]],
                  row: Mapping[str, Any]) -> None:
    """Add one row to a per-split totals map; :func:`split_totals` and the reader's live
    summary share this one definition of what a row contributes."""
    from .tlog_pool import _windows_per_clip

    entry = totals.setdefault(row["split"], {"rows": 0, "clips": 0, "windows": 0,
                                             "audio_seconds": 0.0})
    entry["rows"] += 1
    entry["clips"] += 1
    entry["windows"] += _windows_per_clip(row)
    entry["audio_seconds"] += row["duration_s"]


def split_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int | float]]:
    """Per split: ``rows``, ``clips``, ``windows`` and ``audio_seconds`` (real audio read).

    A clip is one row, so rows and clips agree; windows are
    :func:`tadabur.tlog_pool._windows_per_clip` at the round-2 5 s window and 2.5 s hop, and
    ``audio_seconds`` the sum of the rows' ``duration_s``, rounded to the millisecond so a
    committed report shows seconds rather than float noise.
    """
    totals: dict[str, dict[str, int | float]] = {}
    for row in rows:
        add_split_row(totals, row)
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


def plan_prefix(rows: Iterable[Mapping[str, Any]], schedule: Schedule, *, windows: int,
                use=None, after: ConsumedPrefix | None = None) -> dict:
    """The planned prefix of a run of ``windows`` tlog windows, walking
    :func:`interleave_order`.

    The walk stops at the first row that brings the cumulative window count to ``windows`` (that
    row is included, so the plan never under-reads the budget), and reports the rows, clips,
    windows and real seconds per split, the shards touched, the prefix's share of the pool, and
    the clean surahs covered against the pool's. Surah coverage covers clean rows only: unclean
    rows carry no filename label.

    ``after`` makes the plan a later round's: the walk starts at its first row past
    ``after.length`` and the window target is counted from that boundary, so the report is what
    the run reads. It adds the cumulative report -- ``surahs_new`` (the prefix's clean surahs
    the consumed rows lacked), ``surahs_cumulative`` (consumed plus prefix) and
    ``share_of_pool_cumulative`` (consumed plus prefix rows over the pool) -- and leaves round
    2's report untouched. ``after`` must be a verified prefix (:func:`require_walk_prefix`).

    :func:`require_schedule` runs first, so a plan is never reported for a schedule the reader
    would refuse: the commit's ``use`` binds the recorded overlay, and the count, hash and shard
    coverage are checked against the pool.
    """
    if windows < 1:
        raise ValueError(f"a prefix window target is positive, not {windows!r}")
    from .tlog_pool import _windows_per_clip

    pool = list(rows)
    by_shard = require_schedule(schedule, pool, use=use)
    walk = interleave_streams(by_shard, schedule)
    consumed = [] if after is None else _walk_head(by_shard, schedule, after.length)
    taken: list[Mapping[str, Any]] = []
    running = 0
    for row in islice(walk, 0 if after is None else after.length, None):
        taken.append(row)
        running += _windows_per_clip(row)
        if running >= windows:
            break
    totals = split_totals(taken)
    report = {
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
    if after is not None:
        covered = set(report["surahs_covered"])
        consumed_surahs = set(_clean_surahs(consumed))
        report["surahs_new"] = sorted(covered - consumed_surahs)
        report["surahs_cumulative"] = sorted(covered | consumed_surahs)
        report["share_of_pool_cumulative"] = (
            (after.length + len(taken)) / len(pool) if pool else 0.0)
    return report


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
         prefix_path: Path = PREFIX_PATH, after: Path | None = None) -> dict:
    """The committed schedule's planned prefix, written to ``prefix_path``.

    The schedule is bound to the committed use before the report is written: a schedule for
    another overlay, or one whose count, hash or shards are not the pool's, refuses.

    ``after`` is a consumed-keys log read as given (this planner runs on any machine's copy) and
    verified as the walk's first rows before anything is written, so a keys file that is not a
    prefix never produces a report.
    """
    from .tlog_pool import load_training_rows, training_use

    rows = load_training_rows()
    schedule = read_schedule(schedule_path)
    prefix = None
    if after is not None:
        prefix = require_walk_prefix(read_consumed_keys(after), rows, schedule)
    report = plan_prefix(rows, schedule, windows=windows, use=training_use(), after=prefix)
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
    plan_.add_argument("--prefix-path", "--out", dest="prefix_path", type=Path,
                       default=PREFIX_PATH)
    plan_.add_argument("--after", type=Path,
                       help="a consumed-keys log to start after (e.g. "
                       "tadabur/tlog/consumed/consonant2.jsonl.gz): the plan is then the next "
                       "prefix past its walk rows, with the cumulative report (#250)")
    args = parser.parse_args(argv)
    if args.command == "build":
        schedule = build(schedule_path=args.schedule_path)
        result = {"count": schedule.count, "interleave_width": schedule.interleave_width,
                  "overlay_commit": schedule.overlay_commit, "rows_sha256": schedule.rows_sha256,
                  "seed": schedule.seed, "shards": len(schedule.shards),
                  "schedule_sha256": sha256_file(Path(args.schedule_path))}
    else:
        result = plan(windows=args.windows, schedule_path=args.schedule_path,
                      prefix_path=args.prefix_path, after=args.after)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

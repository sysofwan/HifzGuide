"""Acceptance tests for the frozen, seeded, interleaved tlog training schedule (#224).

Committed before the code exists and **locked**: later turns may add tests but never weaken or
rewrite these. Decisions are the brief's: the schedule is metadata only, covers both splits and is
read ``K = 8`` shards at a time; it carries no self-hash (the README cites ``sha256_file`` of the
file); the prefix cut includes the row that crosses the window target; the reader and the plan share
one interleave-order function; surah coverage counts clean rows only.

Contract this file pins
-----------------------
``tadabur.tlog_schedule`` (a new module; ``tlog_pool`` may import it, so it imports ``tlog_pool``
lazily or not at all at module level):

* ``SEED = "round2-tlog-shards-2026-10"``, ``INTERLEAVE_WIDTH = 8``, ``SCHEDULE_PATH`` =
  ``tools/tadabur/tlog/training_schedule.json``, ``PREFIX_PATH`` =
  ``tools/tadabur/tlog/training_prefix.json``.
* ``Schedule(seed, interleave_width, overlay_commit, rows_sha256, count, shards)``: a frozen
  dataclass; ``shards`` is a tuple of ``(split, shard_file)`` pairs.
* ``build_schedule(rows, use, *, seed=SEED, width=INTERLEAVE_WIDTH)``: the sorted unique
  ``(split, shard_file)`` of ``rows`` shuffled by ``random.Random(seed).shuffle``; ``use`` is the
  ``tlog.training`` ``RowSetExposure`` whose ``overlay_commit``, ``rows_sha256`` and ``count`` it
  records. A ``width`` below 1 (or a bool) is a ``ValueError``.
* ``write_schedule(schedule, path=SCHEDULE_PATH)`` writes a JSON object with exactly the
  ``Schedule`` field names (``shards`` as a list of two-element lists), deterministic bytes;
  ``read_schedule(path=SCHEDULE_PATH)`` parses and validates it (``ValueError`` on a missing or extra
  field or a malformed value).
* ``interleave_order(rows, schedule)``: yields ``rows`` in stream order -- shards opened in schedule
  order, ``width`` slots filled from the front, one row per open shard in slot order each round,
  rows within a shard in row order, an exhausted shard's slot refilled at once by the next scheduled
  shard. At ``width`` 1 it is shard by shard.
* ``split_totals(rows)``: ``{split: {"rows", "clips", "windows", "audio_seconds"}}``; a clip is one
  row, windows are ``tlog_pool._windows_per_clip``, ``audio_seconds`` the sum of ``duration_s``.
* ``plan_prefix(rows, schedule, *, windows)``: walks ``interleave_order`` until the cumulative
  windows reach ``windows`` (the crossing row is included) and reports ``target_windows``, ``rows``,
  ``clips``, ``windows``, ``shards_touched``, ``share_of_pool`` (prefix rows over all rows),
  ``splits`` (per split: ``rows``, ``clips``, ``windows``, ``audio_seconds``, ``shards_touched``),
  ``surahs_covered`` and ``surahs_in_pool`` (sorted ints from clean rows' ``filename_label``).
* ``main(argv=None)`` with the sub-commands ``build`` and ``plan --windows N``.

``tadabur.tlog_pool.iter_training_rows(rows=None, *, cache_dir, schedule, stream=None,
load_audio=None, run_log=None)``: ``schedule`` is required (none, or ``None``, refuses); the
schedule must cover exactly the shards of ``rows`` (``ValueError`` before any download); rows come
in ``interleave_order``; at most ``schedule.interleave_width`` shards are on disk at any time; every
open shard is closed on exhaustion, a raise or ``close()``; every consumed row key is appended as a
JSON line (with at least ``split``, ``shard_file``, ``row``) to the gzip'd ``run_log`` (default
``cache_dir / tlog_pool.RUN_LOG_NAME``) and the end of the run prints one line per split with its
rows, clips, windows and audio seconds.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import subprocess
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tadabur import exposure, tlog_pool, tlog_schedule
from tadabur.tlog_index import sha256_file
from tadabur.tlog_overlap import iter_pool_rows

ROOT = Path(__file__).resolve().parents[2]
TLOG_DIR = Path(__file__).parent / "tlog"
CLEAN_0, CLEAN_1 = "clean-00000-of-00411", "clean-00001-of-00411"
PREFIX_WINDOWS = 240_000
SEED = "round2-tlog-shards-2026-10"
SCHEDULE_FIELDS = {"seed", "interleave_width", "overlay_commit", "rows_sha256", "count", "shards"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key(row) -> tuple[str, str, int]:
    return row["split"], row["shard_file"], row["row"]


def _shard_of(row) -> tuple[str, str]:
    return row["split"], row["shard_file"]


def _stem(split: str, index: int) -> str:
    return f"clean-{index:05d}-of-00411" if split == "clean" else f"unclean-{index:05d}-of-00332"


def _world(layout) -> tuple[list[dict], dict[str, list[bytes]]]:
    """Fake training rows and their shards' audio bytes from ``(split, shard_file, count[, label])``.

    Every clip is 12 s (four windows at a 5 s window and 2.5 s hop).
    """
    rows, payloads = [], {}
    for split, stem, count, *label in layout:
        data = [f"{stem}:{index}".encode() for index in range(count)]
        payloads[f"data/{stem}.parquet"] = data
        rows += [{"split": split, "shard_file": stem, "row": index,
                  "audio_path": f"{stem}-{index}.flac", "audio_sha256": _sha256(data[index]),
                  "duration_s": 12.0, "role": "train", "usable": True,
                  "filename_label": label[0] if label else None}
                 for index in range(count)]
    return rows, payloads


def _use(rows) -> exposure.RowSetExposure:
    return exposure.RowSetExposure("b" * 40, dict(tlog_pool.TRAINING_PREDICATE), len(rows),
                                   tlog_pool.rows_digest(rows))


def _scheduled(rows, order=None, **options) -> tlog_schedule.Schedule:
    """A schedule over ``rows``; ``order`` (``(split, shard_file)`` pairs) replaces the shuffle."""
    schedule = tlog_schedule.build_schedule(rows, _use(rows), **options)
    return schedule if order is None else replace(schedule, shards=tuple(order))


class _Disk:
    """A fake shard stream that models the real one: the download happens on the first ``next()``
    and the blob is dropped when the generator is closed or finished."""

    def __init__(self, payloads):
        self.payloads, self.calls, self.on_disk, self.peak = payloads, [], set(), 0

    def __call__(self, shard: str):
        self.calls.append(shard)
        return self._rows(shard)

    def _rows(self, shard: str):
        self.on_disk.add(shard)
        self.peak = max(self.peak, len(self.on_disk))
        try:
            for index, payload in enumerate(self.payloads[shard]):
                yield {"audio": {"path": f"{index}.flac", "bytes": payload}}
        finally:
            self.on_disk.discard(shard)


def _load(data: bytes):
    return np.zeros(16, dtype=np.float32)


def _read(rows, payloads, schedule, tmp_path, **options):
    disk = _Disk(payloads)
    reader = tlog_pool.iter_training_rows(rows, cache_dir=tmp_path, schedule=schedule, stream=disk,
                                          load_audio=_load, **options)
    return reader, disk


def _logged(path: Path) -> list[tuple[str, str, int]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [(record["split"], record["shard_file"], record["row"])
                for record in map(json.loads, handle.read().splitlines())]


#: Seven shards of uneven length over both splits, in the order the schedule will list them.
UNEVEN = [("clean", _stem("clean", 0), 3), ("unclean", _stem("unclean", 0), 1),
          ("clean", _stem("clean", 1), 4), ("unclean", _stem("unclean", 1), 2),
          ("clean", _stem("clean", 2), 5), ("unclean", _stem("unclean", 2), 1),
          ("clean", _stem("clean", 3), 2)]


@lru_cache(maxsize=1)
def _real_training() -> tuple[dict, ...]:
    return tuple(tlog_pool.training_rows(iter_pool_rows()))


def _real_use() -> exposure.RowSetExposure:
    return exposure.load_registry().row_sets[exposure.TLOG_TRAINING]


# --- the schedule -------------------------------------------------------------------------------


def test_the_schedule_vocabulary_is_the_briefs():
    assert tlog_schedule.SEED == SEED
    assert tlog_schedule.INTERLEAVE_WIDTH == 8
    assert tlog_schedule.SCHEDULE_PATH == TLOG_DIR / "training_schedule.json"
    assert tlog_schedule.PREFIX_PATH == TLOG_DIR / "training_prefix.json"


def test_the_schedule_is_the_seeded_shuffle_of_the_sorted_shards_of_both_splits():
    layout = [(split, _stem(split, index), 2) for split in ("clean", "unclean")
              for index in range(12)]
    rows, _ = _world(layout)
    schedule = tlog_schedule.build_schedule(list(reversed(rows)), _use(rows))
    expected = sorted({_shard_of(row) for row in rows})
    random.Random(SEED).shuffle(expected)
    assert list(schedule.shards) == expected != sorted(expected)
    assert {split for split, _ in schedule.shards} == {"clean", "unclean"}
    assert (schedule.seed, schedule.interleave_width) == (SEED, 8)


def test_the_schedule_records_the_use_it_covers():
    rows, _ = _world(UNEVEN)
    use = _use(rows)
    schedule = tlog_schedule.build_schedule(rows, use)
    assert (schedule.overlay_commit, schedule.rows_sha256, schedule.count) == (
        use.overlay_commit, use.rows_sha256, use.count)


def test_the_schedule_lists_a_shard_once_however_many_rows_it_holds():
    rows, _ = _world(UNEVEN)
    assert sorted(tlog_schedule.build_schedule(rows, _use(rows)).shards) == sorted(
        {_shard_of(row) for row in rows})


def test_the_schedule_depends_on_the_seed_and_on_nothing_else():
    rows, _ = _world([("clean", _stem("clean", index), 1) for index in range(12)])
    use = _use(rows)
    first = tlog_schedule.build_schedule(rows, use)
    assert tlog_schedule.build_schedule(list(reversed(rows)), use) == first
    assert tlog_schedule.build_schedule(rows, use, seed="another-seed").shards != first.shards
    assert tlog_schedule.build_schedule(rows, use, width=3).interleave_width == 3


@pytest.mark.parametrize("width", [0, -1, True, 2.5])
def test_a_schedule_needs_a_positive_integer_width(width):
    rows, _ = _world(UNEVEN)
    with pytest.raises(ValueError):
        tlog_schedule.build_schedule(rows, _use(rows), width=width)


def test_a_schedule_round_trips_through_its_json_file_with_stable_bytes(tmp_path):
    rows, _ = _world(UNEVEN)
    schedule = _scheduled(rows, width=3)
    path = tmp_path / "schedule.json"
    tlog_schedule.write_schedule(schedule, path)
    first = path.read_bytes()
    data = json.loads(first.decode("utf-8"))
    assert set(data) == SCHEDULE_FIELDS  # no self-hash: the file cannot hold its own SHA-256
    assert data["shards"] == [list(shard) for shard in schedule.shards]
    assert tlog_schedule.read_schedule(path) == schedule
    tlog_schedule.write_schedule(schedule, path)
    assert path.read_bytes() == first


@pytest.mark.parametrize("change", [
    {"drop": "seed"}, {"drop": "shards"}, {"add": {"note": "x"}},
    {"set": {"interleave_width": 0}}, {"set": {"interleave_width": True}},
    {"set": {"count": -1}}, {"set": {"overlay_commit": "B" * 40}},
    {"set": {"rows_sha256": "c" * 63}}, {"set": {"shards": [["clean"]]}},
    {"set": {"shards": "clean-00000"}},
])
def test_a_malformed_schedule_file_is_refused(change, tmp_path):
    rows, _ = _world(UNEVEN)
    tlog_schedule.write_schedule(_scheduled(rows), tmp_path / "schedule.json")
    data = json.loads((tmp_path / "schedule.json").read_text(encoding="utf-8"))
    data.pop(change.get("drop"), None)
    data.update(change.get("add", {}))
    data.update(change.get("set", {}))
    (tmp_path / "broken.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        tlog_schedule.read_schedule(tmp_path / "broken.json")


# --- the interleave order -----------------------------------------------------------------------


def test_the_order_alternates_the_open_shards_and_refills_when_both_finish_together():
    rows, _ = _world([("clean", _stem("clean", index), 2) for index in range(4)])
    order = [(_stem("clean", index)) for index in range(4)]
    schedule = _scheduled(rows, [("clean", stem) for stem in order], width=2)
    s0, s1, s2, s3 = order
    assert [_key(row) for row in tlog_schedule.interleave_order(rows, schedule)] == [
        ("clean", s0, 0), ("clean", s1, 0), ("clean", s0, 1), ("clean", s1, 1),
        ("clean", s2, 0), ("clean", s3, 0), ("clean", s2, 1), ("clean", s3, 1)]


def test_a_width_of_one_reads_shard_by_shard_in_schedule_order_whatever_the_row_order():
    rows, _ = _world(UNEVEN)
    shards = [("unclean", _stem("unclean", 1)), ("clean", _stem("clean", 2)),
              ("clean", _stem("clean", 0)), ("unclean", _stem("unclean", 0)),
              ("clean", _stem("clean", 1)), ("clean", _stem("clean", 3)),
              ("unclean", _stem("unclean", 2))]
    schedule = _scheduled(rows, shards, width=1)
    out = [_key(row) for row in tlog_schedule.interleave_order(list(reversed(rows)), schedule)]
    assert out == sorted(out, key=lambda key: (shards.index(key[:2]), key[2]))
    assert len(out) == len(rows)


def test_the_order_yields_every_row_once_in_row_order_within_its_shard_with_k_open():
    rows, _ = _world(UNEVEN)
    shards = [(split, stem) for split, stem, _ in UNEVEN]
    width = 3
    order = list(tlog_schedule.interleave_order(list(reversed(rows)),
                                                _scheduled(rows, shards, width=width)))
    assert sorted(map(_key, order)) == sorted(map(_key, rows))  # each row exactly once
    assert len({_key(row) for row in order}) == len(rows)
    for shard in shards:
        assert [r["row"] for r in order if _shard_of(r) == shard] == sorted(
            r["row"] for r in rows if _shard_of(r) == shard)
    first_seen = list(dict.fromkeys(_shard_of(row) for row in order))
    assert first_seen == shards  # shards open in schedule order
    left = {shard: sum(1 for r in rows if _shard_of(r) == shard) for shard in shards}
    live: set[tuple[str, str]] = set()
    for position, row in enumerate(order):
        shard = _shard_of(row)
        live.add(shard)
        assert len(live) <= width
        left[shard] -= 1
        if not left[shard]:
            live.discard(shard)
        if position + 1 < len(order) and _shard_of(order[position + 1]) == shard:
            # a shard follows itself only once it is the last one left
            assert {_shard_of(r) for r in order[position + 1:]} == {shard}
    assert len({_shard_of(row) for row in order[:width]}) == width  # K different shards first


# --- the reader ---------------------------------------------------------------------------------


def test_a_training_call_without_a_schedule_refuses_before_any_download(tmp_path):
    rows, payloads = _world(UNEVEN)
    disk = _Disk(payloads)
    with pytest.raises((TypeError, ValueError)):
        list(tlog_pool.iter_training_rows(rows, cache_dir=tmp_path, stream=disk,
                                          load_audio=_load))
    with pytest.raises((TypeError, ValueError)):
        list(tlog_pool.iter_training_rows(rows, cache_dir=tmp_path, schedule=None, stream=disk,
                                          load_audio=_load))
    assert disk.calls == [] and disk.peak == 0


def test_a_two_split_overlay_yields_both_splits_within_the_first_k_shards(tmp_path):
    rows, payloads = _world([("clean", CLEAN_0, 2), ("unclean", _stem("unclean", 0), 2),
                             ("clean", CLEAN_1, 2), ("unclean", _stem("unclean", 1), 2)])
    shards = [("clean", CLEAN_0), ("unclean", _stem("unclean", 0)), ("clean", CLEAN_1),
              ("unclean", _stem("unclean", 1))]
    reader, _ = _read(rows, payloads, _scheduled(rows, shards, width=2), tmp_path)
    first = [next(reader)[0] for _ in range(4)]
    reader.close()
    assert {split for split, _, _ in first} == {"clean", "unclean"}
    assert {(split, shard) for split, shard, _ in first} == set(shards[:2])


def test_the_reader_yields_the_shared_interleave_order_and_opens_shards_in_schedule_order(tmp_path):
    rows, payloads = _world(UNEVEN)
    schedule = _scheduled(rows, width=3)
    reader, disk = _read(rows, payloads, schedule, tmp_path)
    out = [key for key, _ in reader]
    expected = [_key(row) for row in tlog_schedule.interleave_order(rows, schedule)]
    assert out == expected
    assert disk.calls == [f"data/{stem}.parquet" for _, stem in schedule.shards]
    assert len(out) == len(rows) and len(set(out)) == len(rows)


def test_k_shards_are_open_at_once_never_more_and_all_are_deleted_by_the_end(tmp_path):
    rows, payloads = _world([("clean" if index % 2 else "unclean", _stem(
        "clean" if index % 2 else "unclean", index), 3) for index in range(10)])
    reader, disk = _read(rows, payloads, _scheduled(rows, width=3), tmp_path)
    seen = []
    previous = None
    for key, _ in reader:
        seen.append(len(disk.on_disk))
        assert key[:2] != previous or len(disk.on_disk) == 1  # consecutive rows differ while K open
        previous = key[:2]
    assert max(seen) <= 3 and disk.peak == 3
    assert disk.on_disk == set()


def test_every_shard_is_deleted_when_a_row_raises_with_k_open(tmp_path):
    rows, payloads = _world(UNEVEN)
    shards = [(split, stem) for split, stem, _ in UNEVEN]
    bad = next(index for index, row in enumerate(rows) if _shard_of(row) == shards[1])
    rows[bad] = {**rows[bad], "audio_sha256": "0" * 64}
    reader, disk = _read(rows, payloads, _scheduled(rows, shards, width=3), tmp_path)
    with pytest.raises(ValueError):
        list(reader)
    assert disk.peak <= 3 and disk.on_disk == set()


def test_every_shard_is_deleted_when_the_consumer_stops_early(tmp_path):
    rows, payloads = _world([("clean", _stem("clean", index), 3) for index in range(5)])
    reader, disk = _read(rows, payloads, _scheduled(rows, width=3), tmp_path)
    for _ in range(3):
        next(reader)  # one row from each of the three open shards
    assert len(disk.on_disk) == 3
    reader.close()
    assert disk.on_disk == set()


@pytest.mark.parametrize("defect", ["missing", "duplicated", "extra"])
def test_a_schedule_that_does_not_cover_exactly_the_shards_refuses_before_any_download(
        defect, tmp_path):
    rows, payloads = _world(UNEVEN)
    schedule = _scheduled(rows, width=3)
    shards = {"missing": schedule.shards[:-1],
              "duplicated": schedule.shards + schedule.shards[:1],
              "extra": schedule.shards + (("clean", _stem("clean", 99)),)}[defect]
    reader, disk = _read(rows, payloads, replace(schedule, shards=shards), tmp_path)
    with pytest.raises(ValueError):
        list(reader)
    assert disk.calls == [] and disk.peak == 0


@pytest.fixture
def cached_shards(tmp_path, monkeypatch):
    """Two tlog shards served by a fake ``hf_hub_download``: the parquet files by shard name."""
    import huggingface_hub

    files = {}
    for name in (CLEAN_0, CLEAN_1):
        path = tmp_path / f"{name}.parquet"
        payloads = [f"{name}:{index}".encode() for index in range(2)]
        pq.write_table(pa.table({"audio": pa.array(
            [{"path": f"{index}.flac", "bytes": payload} for index, payload in enumerate(payloads)],
            type=pa.struct([("path", pa.string()), ("bytes", pa.binary())]))}), path)
        files[f"data/{name}.parquet"] = path
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, filename, **kwargs: str(files[filename]))
    rows, _ = _world([("clean", CLEAN_0, 2), ("clean", CLEAN_1, 2)])
    return files, rows


def test_two_real_downloads_are_both_deleted_after_a_full_read(cached_shards, tmp_path):
    files, rows = cached_shards
    out = list(tlog_pool.iter_training_rows(rows, cache_dir=tmp_path / "cache",
                                            schedule=_scheduled(rows, width=2),
                                            load_audio=_load))
    assert len(out) == 4
    assert not any(path.exists() for path in files.values())


def test_two_real_downloads_are_both_deleted_when_the_consumer_stops_early(cached_shards, tmp_path):
    files, rows = cached_shards
    reader = tlog_pool.iter_training_rows(rows, cache_dir=tmp_path / "cache",
                                          schedule=_scheduled(rows, width=2), load_audio=_load)
    next(reader)
    next(reader)  # one row from each open shard
    assert all(path.exists() for path in files.values())
    reader.close()
    assert not any(path.exists() for path in files.values())


# --- the run log --------------------------------------------------------------------------------


def test_the_run_log_holds_every_consumed_row_key_in_stream_order(tmp_path):
    rows, payloads = _world(UNEVEN)
    log = tmp_path / "run" / "consumed.jsonl.gz"
    log.parent.mkdir()
    reader, _ = _read(rows, payloads, _scheduled(rows, width=3), tmp_path, run_log=log)
    consumed = [key for key, _ in reader]
    assert consumed and _logged(log) == consumed


def test_the_run_log_defaults_to_a_file_in_the_cache_dir(tmp_path):
    rows, payloads = _world(UNEVEN[:2])
    reader, _ = _read(rows, payloads, _scheduled(rows, width=2), tmp_path)
    keys = [key for key, _ in reader]
    assert _logged(tmp_path / tlog_pool.RUN_LOG_NAME) == keys


def test_the_run_log_appends_across_readers_and_stops_with_the_consumer(tmp_path):
    rows, payloads = _world(UNEVEN)
    log = tmp_path / "consumed.jsonl.gz"
    schedule = _scheduled(rows, width=3)
    reader, _ = _read(rows, payloads, schedule, tmp_path, run_log=log)
    taken = [next(reader)[0] for _ in range(3)]
    reader.close()
    assert _logged(log) == taken
    reader, _ = _read(rows, payloads, schedule, tmp_path, run_log=log)
    everything = [key for key, _ in reader]
    assert _logged(log) == taken + everything


def test_the_run_ends_with_one_line_per_split_of_rows_clips_windows_and_audio_seconds(
        tmp_path, capsys):
    rows, payloads = _world([("clean", CLEAN_0, 3), ("unclean", _stem("unclean", 0), 2)])
    reader, _ = _read(rows, payloads, _scheduled(rows, width=2), tmp_path)
    list(reader)
    lines = capsys.readouterr().out.splitlines()[1:]  # the first line is the pre-run estimate
    for split, count, windows, seconds in (("clean", 3, 12, 36), ("unclean", 2, 8, 24)):
        [line] = [line for line in lines if re.search(rf"(?<![\w-]){split}(?![\w-])", line)]
        for number in (count, windows, seconds):
            assert re.search(rf"(?<![\d.]){number}(\.0+)?(?![\d.])", line), (split, number, line)


def test_split_totals_count_rows_clips_windows_and_real_audio_seconds():
    rows, _ = _world([("clean", CLEAN_0, 3), ("unclean", _stem("unclean", 0), 2)])
    rows[0] = {**rows[0], "duration_s": 2.0}  # shorter than a window: one window
    totals = tlog_schedule.split_totals(rows)
    assert totals["clean"]["rows"] == 3 and totals["clean"]["clips"] == 3
    assert totals["clean"]["windows"] == 1 + 4 + 4
    assert totals["clean"]["audio_seconds"] == pytest.approx(26.0)
    assert totals["unclean"] == {"rows": 2, "clips": 2, "windows": 8, "audio_seconds": 24.0}


# --- the planned prefix -------------------------------------------------------------------------


def _plan_world():
    rows, _ = _world([("clean", _stem("clean", 0), 3, "100:1"),
                      ("unclean", _stem("unclean", 0), 3),
                      ("clean", _stem("clean", 1), 3, "36:2")])
    schedule = _scheduled(rows, [("clean", _stem("clean", 0)), ("unclean", _stem("unclean", 0)),
                                 ("clean", _stem("clean", 1))], width=2)
    return rows, schedule


@pytest.mark.parametrize(("target", "taken"), [(16, 4), (17, 5), (20, 5), (21, 6), (22, 6)])
def test_the_prefix_cut_includes_the_row_that_crosses_the_window_target(target, taken):
    rows, schedule = _plan_world()
    report = tlog_schedule.plan_prefix(rows, schedule, windows=target)
    assert report["target_windows"] == target
    assert report["rows"] == report["clips"] == taken
    assert report["windows"] == 4 * taken
    expected = list(tlog_schedule.interleave_order(rows, schedule))[:taken]
    assert report["splits"]["clean"]["rows"] == sum(row["split"] == "clean" for row in expected)


def test_a_target_beyond_the_pool_plans_the_whole_pool():
    rows, schedule = _plan_world()
    report = tlog_schedule.plan_prefix(rows, schedule, windows=10_000)
    assert report["rows"] == len(rows) and report["share_of_pool"] == 1.0


def test_the_prefix_reports_each_split_its_shards_its_share_and_the_clean_surahs():
    rows, schedule = _plan_world()
    report = tlog_schedule.plan_prefix(rows, schedule, windows=18)  # A0 B0 A1 B1 A2
    assert (report["rows"], report["windows"], report["shards_touched"]) == (5, 20, 2)
    assert report["share_of_pool"] == pytest.approx(5 / 9)
    assert report["splits"]["clean"] == {
        "rows": 3, "clips": 3, "windows": 12, "audio_seconds": 36.0, "shards_touched": 1}
    assert report["splits"]["unclean"] == {
        "rows": 2, "clips": 2, "windows": 8, "audio_seconds": 24.0, "shards_touched": 1}
    assert report["surahs_covered"] == [100] and report["surahs_in_pool"] == [36, 100]


def test_the_cli_offers_build_and_plan_with_a_window_target(capsys):
    with pytest.raises(SystemExit) as exit_:
        tlog_schedule.main(["--help"])
    assert exit_.value.code == 0
    assert "build" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_:
        tlog_schedule.main(["plan", "--help"])
    assert exit_.value.code == 0
    assert "--windows" in capsys.readouterr().out


# --- the committed files ------------------------------------------------------------------------


def test_the_committed_schedule_is_regenerated_byte_for_byte_from_the_seed_and_overlay(tmp_path):
    expected = tlog_schedule.build_schedule(_real_training(), _real_use())
    tlog_schedule.write_schedule(expected, tmp_path / "schedule.json")
    assert tlog_schedule.SCHEDULE_PATH.read_bytes() == (tmp_path / "schedule.json").read_bytes()
    assert tlog_schedule.read_schedule() == expected


def test_the_committed_schedule_covers_every_training_shard_once_across_both_splits():
    schedule = tlog_schedule.read_schedule()
    shards = {_shard_of(row) for row in _real_training()}
    assert sorted(schedule.shards) == sorted(shards) and len(schedule.shards) == len(shards)
    assert {split for split, _ in schedule.shards} == {"clean", "unclean"}
    expected = sorted(shards)
    random.Random(SEED).shuffle(expected)
    assert list(schedule.shards) == expected
    use = _real_use()
    assert (schedule.seed, schedule.interleave_width, schedule.count) == (SEED, 8, use.count)
    assert (schedule.overlay_commit, schedule.rows_sha256) == (use.overlay_commit, use.rows_sha256)


def test_the_readme_cites_the_schedule_files_sha256_for_the_pre_registration():
    readme = (TLOG_DIR / "README.md").read_text(encoding="utf-8")
    assert sha256_file(tlog_schedule.SCHEDULE_PATH) in readme


def test_the_committed_prefix_is_the_plan_for_240000_windows_over_the_committed_schedule():
    report = tlog_schedule.plan_prefix(_real_training(), tlog_schedule.read_schedule(),
                                       windows=PREFIX_WINDOWS)
    committed = json.loads(tlog_schedule.PREFIX_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(report))
    assert committed["target_windows"] == PREFIX_WINDOWS and committed["windows"] >= PREFIX_WINDOWS


def test_the_committed_prefix_is_no_longer_the_biased_clean_only_prefix():
    committed = json.loads(tlog_schedule.PREFIX_PATH.read_text(encoding="utf-8"))
    assert set(committed["splits"]) == {"clean", "unclean"}
    assert committed["splits"]["unclean"]["rows"] > 0
    assert len(committed["surahs_covered"]) > 34  # the sorted walk reached 34 of 114
    assert len(committed["surahs_in_pool"]) == 114


@pytest.mark.parametrize("path", [tlog_schedule.SCHEDULE_PATH, tlog_schedule.PREFIX_PATH],
                         ids=lambda path: path.name)
def test_the_schedule_files_are_not_gitignored(path):
    assert subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT).returncode == 1

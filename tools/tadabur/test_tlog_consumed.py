"""Acceptance tests for round 3's tlog continuation (#250), written before the code.

Locked: later turns may add tests but never weaken or rewrite these.

Round 3 trains on the frozen schedule's next prefix past round 2's 73,990 consumed rows. This
file pins what lets the reader and the planner start there.

Contract this file pins
-----------------------
* ``tools/tadabur/tlog/consumed/consonant2.jsonl.gz``: round 2's consumed keys, byte for byte the
  NAS copy (SHA-256 ``5e95c3c0...9238``), one ``{"split", "shard_file", "row"}`` JSON object per
  line. ``consumed/meta.json`` beside it holds ``sha256``, ``keys`` (73,990), ``splits``
  (``{"clean": 49784, "unclean": 24206}``), ``shard_files`` (130), ``checkpoint_sha256`` (the
  round-2 candidate's, ``8935...ea58``) and the two source paths (the NAS copy and the cuda-dev
  run directory, anywhere in the file's text). Neither file is gitignored.
* ``tlog_schedule.read_consumed_keys(path) -> list[(split, shard_file, row)]``: the file's keys in
  line order.
* ``tlog_schedule.ConsumedPrefix``: a frozen dataclass of ``keys`` (a ``frozenset`` of
  ``(split, shard_file, row)``), ``length`` (the walk length, an int) and
  ``line_order_is_walk_order`` (bool), in that order.
* ``tlog_schedule.require_walk_prefix(keys, rows, schedule) -> ConsumedPrefix``: ``keys`` in line
  order. ``ValueError`` unless the keys are distinct and equal, as a set, exactly the first
  ``len(keys)`` rows of ``interleave_order(rows, schedule)``. The line order is reported, never
  required.
* ``iter_training_rows(..., after=None)``: ``after`` is a ``ConsumedPrefix``. The stream is
  ``interleave_order`` from walk row ``after.length + 1`` on, continuing the K-way interleave
  across the boundary; shards the prefix fully consumed are never opened; the run log holds only
  the rows yielded. A ``ConsumedPrefix`` that is not the first ``length`` walk rows of the rows
  it is given refuses before any download (``ValueError``). ``after=None`` is the round-2 walk.
* ``plan_prefix(rows, schedule, *, windows, use=None, after=None)`` walks from the same position
  and, with ``after``, adds ``surahs_new`` (the prefix's clean surahs the consumed rows lacked),
  ``surahs_cumulative`` (consumed plus prefix) and ``share_of_pool_cumulative`` (consumed plus
  prefix rows over the pool). Without ``after`` the report is exactly round 2's.
* ``plan(..., after=None)`` and ``python -m tadabur.tlog_schedule plan --after PATH --out PATH``
  (``--out`` is a second spelling of ``--prefix-path``). The keys file is checked against the
  walk before anything is written.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from tadabur import exposure, tlog_pool, tlog_schedule
from tadabur.tlog_index import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONSUMED_DIR = Path(__file__).parent / "tlog" / "consumed"
CONSUMED = CONSUMED_DIR / "consonant2.jsonl.gz"
META = CONSUMED_DIR / "meta.json"
PINNED_SHA256 = "5e95c3c06f498768323781fc04b4394093233c2a7717ae193f80feda87929238"
CHECKPOINT_SHA256 = "893523710186cb16854dd62a258331644e2284abbee8b2cb54e8ab9d32dbea58"
NAS_PATH = "/Volumes/NetBackup/HifzGuide/round2-consumed/tlog_consumed.jsonl.gz"
CUDA_DEV_DIR = "root@cuda-dev:/root/scratch/issue-228/runs/consonant2_candidate/"
CONSUMED_ROWS = 73_990
PREFIX_WINDOWS = 240_000


# --- a synthetic pool ---------------------------------------------------------------------------


def _stem(split: str, index: int) -> str:
    return f"clean-{index:05d}-of-00411" if split == "clean" else f"unclean-{index:05d}-of-00332"


def _key(row) -> tuple[str, str, int]:
    return row["split"], row["shard_file"], row["row"]


def _world(layout):
    """Fake training rows and their shards' audio bytes from ``(split, stem, count[, label])``.

    Every clip is 12 s: four windows at a 5 s window and a 2.5 s hop.
    """
    rows, payloads = [], {}
    for split, stem, count, *label in layout:
        data = [f"{stem}:{index}".encode() for index in range(count)]
        payloads[f"data/{stem}.parquet"] = data
        rows += [{"split": split, "shard_file": stem, "row": index,
                  "audio_path": f"{stem}-{index}.flac",
                  "audio_sha256": hashlib.sha256(data[index]).hexdigest(),
                  "duration_s": 12.0, "role": "train", "usable": True,
                  "filename_label": label[0] if label else None}
                 for index in range(count)]
    return rows, payloads


@lru_cache(maxsize=1)
def _real_rows() -> tuple[dict, ...]:
    return tuple(tlog_pool.load_training_rows())


def _use(rows) -> exposure.RowSetExposure:
    return exposure.RowSetExposure("b" * 40, dict(tlog_pool.TRAINING_PREDICATE), len(rows),
                                   tlog_pool.rows_digest(rows))


class _Disk:
    """A fake shard stream: the download happens on the first ``next()`` and the blob is dropped
    when the generator is closed or finished."""

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


#: Four shards of uneven length, read three at a time in this order. The walk is, per round, one
#: row from each open shard; a short shard is replaced at once by the next scheduled one.
LAYOUT = [("clean", _stem("clean", 0), 5, "100:1"), ("unclean", _stem("unclean", 0), 2),
          ("clean", _stem("clean", 1), 6, "36:2"), ("unclean", _stem("unclean", 1), 4),
          ("clean", _stem("clean", 2), 3, "55:3")]
WIDTH = 3
#: Walk rows consumed by the prefix under test: the short shard unclean-0 is finished and replaced
#: by unclean-1, and three shards are open at the boundary.
AFTER_ROWS = 8


@pytest.fixture
def world():
    rows, payloads = _world(LAYOUT)
    shards = [(split, stem) for split, stem, *_ in LAYOUT]
    schedule = replace(tlog_schedule.build_schedule(rows, _use(rows), width=WIDTH),
                       shards=tuple(shards))
    walk = [_key(row) for row in tlog_schedule.interleave_order(rows, schedule)]
    return rows, payloads, schedule, walk


def _prefix(rows, schedule, walk, length=AFTER_ROWS) -> tlog_schedule.ConsumedPrefix:
    return tlog_schedule.require_walk_prefix(walk[:length], rows, schedule)


def _read(rows, payloads, schedule, tmp_path, **options):
    disk = _Disk(payloads)
    reader = tlog_pool.iter_training_rows(rows, cache_dir=tmp_path, schedule=schedule, stream=disk,
                                          load_audio=_load, **options)
    return reader, disk


def _logged(path: Path) -> list[tuple[str, str, int]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [(record["split"], record["shard_file"], record["row"])
                for record in map(json.loads, handle.read().splitlines())]


def test_the_synthetic_boundary_is_mid_interleave(world):
    """Guards the fixture: the prefix ends with three shards open, two of them mid-shard."""
    rows, _, _, walk = world
    before, after = walk[:AFTER_ROWS], walk[AFTER_ROWS:]
    assert len({key[:2] for key in after[:WIDTH]}) == WIDTH
    started = {key[:2] for key in before}
    assert sum(key[:2] in started for key in after[:WIDTH]) >= 2
    assert len(after) > WIDTH and len(walk) == len(rows)


# --- the committed keys -------------------------------------------------------------------------


def test_the_committed_keys_are_the_pinned_nas_copy_byte_for_byte():
    assert sha256_file(CONSUMED) == PINNED_SHA256


def test_the_committed_keys_are_73990_distinct_keys_of_two_splits_over_130_shard_files():
    keys = tlog_schedule.read_consumed_keys(CONSUMED)
    assert len(keys) == len(set(keys)) == CONSUMED_ROWS
    assert Counter(split for split, _, _ in keys) == {"clean": 49_784, "unclean": 24_206}
    assert len({(split, shard) for split, shard, _ in keys}) == 130


def test_the_committed_keys_are_one_json_object_per_line_with_exactly_three_fields():
    with gzip.open(CONSUMED, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle.read().splitlines()]
    assert len(records) == CONSUMED_ROWS
    assert all(set(record) == {"split", "shard_file", "row"} for record in records)
    assert all(isinstance(record["row"], int) for record in records)


def test_the_meta_records_what_the_file_holds_and_where_it_came_from():
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["sha256"] == sha256_file(CONSUMED) == PINNED_SHA256
    assert meta["keys"] == CONSUMED_ROWS
    assert meta["splits"] == {"clean": 49_784, "unclean": 24_206}
    assert meta["shard_files"] == 130
    assert meta["checkpoint_sha256"] == CHECKPOINT_SHA256
    text = META.read_text(encoding="utf-8")
    assert NAS_PATH in text and CUDA_DEV_DIR in text


def test_the_meta_counts_are_the_files_own_counts():
    meta = json.loads(META.read_text(encoding="utf-8"))
    keys = tlog_schedule.read_consumed_keys(CONSUMED)
    assert meta["keys"] == len(keys)
    assert meta["splits"] == dict(Counter(split for split, _, _ in keys))
    assert meta["shard_files"] == len({(split, shard) for split, shard, _ in keys})


@pytest.mark.parametrize("path", [CONSUMED, META], ids=lambda path: path.name)
def test_the_committed_keys_and_meta_are_tracked_not_gitignored(path):
    assert subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT).returncode == 1


# --- round 3's consumed keys (#254) --------------------------------------------------------------

CONSUMED3 = CONSUMED_DIR / "consonant3.jsonl.gz"
META3 = CONSUMED_DIR / "consonant3.meta.json"
PINNED3_SHA256 = "abc3e168632b0c83ca0f49c7a0da3b59dbd1ce0506bb75476c5391265837d222"
CHECKPOINT3_SHA256 = "1193e2117b56381ccc078a5b5e21d61d0ad0903565dcca0cb110b39ce24ed4a9"
NAS3_PATH = "/Volumes/NetBackup/HifzGuide/round3-consumed/tlog_consumed.jsonl.gz"
CUDA_DEV3_DIR = "root@cuda-dev:/root/scratch/issue-252/runs/consonant3_candidate/"
CONSUMED3_ROWS = 75_553


def test_round_3_s_keys_are_the_walk_s_rows_past_round_2_s():
    """#254's run starts where round 2's stopped: the committed keys are the walk's rows
    73,991 onward, disjoint from round 2's, and the meta is the file's own counts."""
    keys = tlog_schedule.read_consumed_keys(CONSUMED3)
    round2 = tlog_schedule.read_consumed_keys(CONSUMED)
    schedule = tlog_schedule.read_schedule(tlog_schedule.SCHEDULE_PATH)
    joined = tlog_schedule.require_walk_prefix([*round2, *keys], _real_rows(), schedule)
    assert len(keys) == CONSUMED3_ROWS and not set(keys) & set(round2)
    assert joined.length == len(round2) + len(keys)
    meta = json.loads(META3.read_text(encoding="utf-8"))
    assert meta["sha256"] == sha256_file(CONSUMED3) == PINNED3_SHA256
    assert meta["keys"] == len(keys)
    assert meta["splits"] == dict(Counter(split for split, _, _ in keys))
    assert meta["shard_files"] == len({(split, shard) for split, shard, _ in keys})
    assert (meta["first_walk_row"], meta["last_walk_row"]) == (len(round2) + 1, joined.length)
    assert meta["checkpoint_sha256"] == CHECKPOINT3_SHA256
    text = META3.read_text(encoding="utf-8")
    assert NAS3_PATH in text and CUDA_DEV3_DIR in text
    for path in (CONSUMED3, META3):
        assert subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=ROOT).returncode == 1


def test_reading_keys_returns_them_in_line_order(tmp_path):
    path = tmp_path / "keys.jsonl.gz"
    keys = [("unclean", _stem("unclean", 3), 7), ("clean", _stem("clean", 1), 0),
            ("clean", _stem("clean", 1), 2)]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps({"split": s, "shard_file": f, "row": r}) + "\n"
                          for s, f, r in keys)
    assert tlog_schedule.read_consumed_keys(path) == keys


# --- the prefix property ------------------------------------------------------------------------


def test_the_consumed_prefix_is_a_frozen_set_of_keys_with_its_walk_length_and_order_flag():
    prefix = tlog_schedule.ConsumedPrefix(frozenset({("clean", "s", 0)}), 1, True)
    assert (prefix.keys, prefix.length, prefix.line_order_is_walk_order) == (
        frozenset({("clean", "s", 0)}), 1, True)
    with pytest.raises(FrozenInstanceError):
        prefix.length = 2


def test_the_walk_prefix_of_the_synthetic_pool_is_accepted_and_reported(world):
    rows, _, schedule, walk = world
    prefix = tlog_schedule.require_walk_prefix(walk[:AFTER_ROWS], rows, schedule)
    assert isinstance(prefix, tlog_schedule.ConsumedPrefix)
    assert prefix.keys == frozenset(walk[:AFTER_ROWS])
    assert prefix.length == AFTER_ROWS
    assert prefix.line_order_is_walk_order is True


def test_a_prefix_whose_lines_are_not_in_walk_order_is_accepted_and_reported_as_such(world):
    rows, _, schedule, walk = world
    keys = list(reversed(walk[:AFTER_ROWS]))
    prefix = tlog_schedule.require_walk_prefix(keys, rows, schedule)
    assert prefix.keys == frozenset(walk[:AFTER_ROWS])
    assert prefix.length == AFTER_ROWS
    assert prefix.line_order_is_walk_order is False


def test_the_whole_walk_and_the_empty_prefix_are_walk_prefixes(world):
    rows, _, schedule, walk = world
    assert tlog_schedule.require_walk_prefix(walk, rows, schedule).length == len(walk)
    empty = tlog_schedule.require_walk_prefix([], rows, schedule)
    assert (empty.keys, empty.length) == (frozenset(), 0)


@pytest.mark.parametrize("defect", ["out_of_prefix", "missing", "duplicate", "not_in_pool"])
def test_a_key_set_that_is_not_exactly_the_first_walk_rows_refuses(defect, world):
    rows, _, schedule, walk = world
    keys = list(walk[:AFTER_ROWS])
    keys = {
        # the next walk row stands in for an early one: the right size, one key past the prefix
        "out_of_prefix": keys[:-1] + [walk[AFTER_ROWS]],
        # an early key dropped: the right keys are not all there
        "missing": keys[1:],
        # a repeated key: the right size, one distinct key short
        "duplicate": keys[:-1] + [keys[0]],
        # a key no row of the pool has
        "not_in_pool": keys[:-1] + [("clean", _stem("clean", 99), 0)],
    }[defect]
    with pytest.raises(ValueError):
        tlog_schedule.require_walk_prefix(keys, rows, schedule)


def test_a_file_longer_than_the_walk_refuses(world):
    rows, _, schedule, walk = world
    with pytest.raises(ValueError):
        tlog_schedule.require_walk_prefix(walk + [("clean", _stem("clean", 99), 0)], rows,
                                          schedule)


def test_the_committed_keys_are_exactly_the_first_73990_rows_of_the_real_walk():
    """The prefix property on the real committed schedule and pool (metadata only)."""
    rows = _real_rows()
    schedule = tlog_schedule.read_schedule()
    keys = tlog_schedule.read_consumed_keys(CONSUMED)
    prefix = tlog_schedule.require_walk_prefix(keys, rows, schedule)
    assert prefix.length == len(prefix.keys) == CONSUMED_ROWS
    assert prefix.line_order_is_walk_order is True


def test_a_planted_defect_in_the_real_keys_refuses():
    rows = _real_rows()
    schedule = tlog_schedule.read_schedule()
    keys = tlog_schedule.read_consumed_keys(CONSUMED)
    after_prefix = _key(next(islice(tlog_schedule.interleave_order(rows, schedule),
                                    CONSUMED_ROWS, None)))
    for planted in (keys[:-1] + [after_prefix], keys[1:], keys[:-1] + [keys[0]]):
        with pytest.raises(ValueError):
            tlog_schedule.require_walk_prefix(planted, rows, schedule)


# --- the reader's offset ------------------------------------------------------------------------


def test_the_reader_starts_at_the_next_walk_row_and_yields_the_rest_of_the_walk(world, tmp_path):
    rows, payloads, schedule, walk = world
    reader, _ = _read(rows, payloads, schedule, tmp_path, after=_prefix(rows, schedule, walk))
    yielded = [key for key, _ in reader]
    assert yielded[0] == walk[AFTER_ROWS]
    assert yielded == walk[AFTER_ROWS:]


def test_no_consumed_key_is_yielded_or_logged(world, tmp_path):
    rows, payloads, schedule, walk = world
    after = _prefix(rows, schedule, walk)
    reader, _ = _read(rows, payloads, schedule, tmp_path, after=after,
                      run_log=tmp_path / "round3.jsonl.gz")
    yielded = [key for key, _ in reader]
    assert not set(yielded) & after.keys
    assert set(yielded) | after.keys == set(walk)
    assert _logged(tmp_path / "round3.jsonl.gz") == walk[AFTER_ROWS:]


@pytest.mark.parametrize("length", [1, 3, AFTER_ROWS, 13, len(LAYOUT) * 3])
def test_the_interleave_is_continued_across_the_boundary_wherever_the_walk_is_cut(
        length, world, tmp_path):
    """Whatever the cut, the stream is the walk minus its first ``length`` rows, so the K shards
    open at the boundary continue mid-shard rather than a fresh interleave of what is left."""
    rows, payloads, schedule, walk = world
    after = _prefix(rows, schedule, walk, length)
    reader, _ = _read(rows, payloads, schedule, tmp_path, after=after)
    assert [key for key, _ in reader] == walk[length:]


def test_shards_the_prefix_fully_consumed_are_never_opened_and_at_most_k_are_open(
        world, tmp_path):
    rows, payloads, schedule, walk = world
    after = _prefix(rows, schedule, walk)
    reader, disk = _read(rows, payloads, schedule, tmp_path, after=after)
    list(reader)
    total = Counter(key[:2] for key in walk)
    consumed = Counter(key[:2] for key in walk[:AFTER_ROWS])
    finished = {shard for shard, count in consumed.items() if count == total[shard]}
    assert finished, "the fixture must finish a shard inside the prefix"
    assert not {f"data/{shard_file}.parquet" for _, shard_file in finished} & set(disk.calls)
    assert 0 < disk.peak <= WIDTH
    assert disk.on_disk == set()


def test_a_prefix_that_covers_the_pool_yields_nothing_and_opens_nothing(world, tmp_path):
    rows, payloads, schedule, walk = world
    reader, disk = _read(rows, payloads, schedule, tmp_path,
                         after=_prefix(rows, schedule, walk, len(walk)))
    assert list(reader) == []
    assert disk.calls == []


def test_a_prefix_that_is_not_the_first_walk_rows_of_the_given_rows_refuses_before_any_download(
        world, tmp_path):
    """A ``ConsumedPrefix`` verified against other rows (here: one key shifted by a row) is
    re-checked cheaply by the reader, which refuses before it opens a shard."""
    rows, payloads, schedule, walk = world
    shifted = tlog_schedule.ConsumedPrefix(frozenset(walk[1:AFTER_ROWS + 1]), AFTER_ROWS, True)
    reader, disk = _read(rows, payloads, schedule, tmp_path, after=shifted)
    with pytest.raises(ValueError):
        list(reader)
    assert disk.calls == [] and disk.peak == 0


def test_a_prefix_with_a_wrong_length_refuses_before_any_download(world, tmp_path):
    rows, payloads, schedule, walk = world
    after = _prefix(rows, schedule, walk)
    reader, disk = _read(rows, payloads, schedule, tmp_path, after=replace(after, length=3))
    with pytest.raises(ValueError):
        list(reader)
    assert disk.calls == []


def test_the_default_walk_is_unchanged_and_none_is_the_default(world, tmp_path):
    rows, payloads, schedule, walk = world
    omitted, _ = _read(rows, payloads, schedule, tmp_path / "a")
    explicit, _ = _read(rows, payloads, schedule, tmp_path / "b", after=None)
    assert [key for key, _ in omitted] == walk
    assert [key for key, _ in explicit] == walk


def test_the_reader_with_a_prefix_still_validates_the_schedule(world, tmp_path):
    rows, payloads, schedule, walk = world
    after = _prefix(rows, schedule, walk)
    reader, disk = _read(rows, payloads, replace(schedule, rows_sha256="0" * 64), tmp_path,
                         after=after)
    with pytest.raises(ValueError):
        list(reader)
    assert disk.calls == []


# --- the planner's offset -----------------------------------------------------------------------


def _plan_world():
    """Nine rows, K = 2: clean-0 (surah 100) and unclean-0 are read together, then clean-1
    (surah 36). The walk is c0 u0 c0 u0 c0 u0 c1 c1 c1."""
    rows, _ = _world([("clean", _stem("clean", 0), 3, "100:1"),
                      ("unclean", _stem("unclean", 0), 3),
                      ("clean", _stem("clean", 1), 3, "36:2")])
    schedule = replace(tlog_schedule.build_schedule(rows, _use(rows), width=2),
                       shards=(("clean", _stem("clean", 0)), ("unclean", _stem("unclean", 0)),
                               ("clean", _stem("clean", 1))))
    walk = [_key(row) for row in tlog_schedule.interleave_order(rows, schedule)]
    return rows, schedule, walk


def test_a_plan_after_a_prefix_walks_the_rows_after_it_and_adds_the_cumulative_report():
    rows, schedule, walk = _plan_world()
    after = tlog_schedule.require_walk_prefix(walk[:6], rows, schedule)
    report = tlog_schedule.plan_prefix(rows, schedule, windows=8, after=after)
    assert (report["rows"], report["clips"], report["windows"]) == (2, 2, 8)
    assert report["splits"] == {"clean": {"rows": 2, "clips": 2, "windows": 8,
                                          "audio_seconds": 24.0, "shards_touched": 1}}
    assert report["shards_touched"] == 1
    assert report["share_of_pool"] == pytest.approx(2 / 9)
    assert report["surahs_covered"] == [36]
    assert report["surahs_new"] == [36]
    assert report["surahs_cumulative"] == [36, 100]
    assert report["surahs_in_pool"] == [36, 100]
    assert report["share_of_pool_cumulative"] == pytest.approx(8 / 9)


def test_a_surah_the_consumed_rows_already_read_is_not_new():
    rows, schedule, walk = _plan_world()
    after = tlog_schedule.require_walk_prefix(walk[:2], rows, schedule)  # c0 u0
    report = tlog_schedule.plan_prefix(rows, schedule, windows=8, after=after)  # c0 u0
    assert report["splits"]["clean"]["rows"] == report["splits"]["unclean"]["rows"] == 1
    assert report["surahs_covered"] == [100]
    assert report["surahs_new"] == []
    assert report["surahs_cumulative"] == [100]
    assert report["share_of_pool_cumulative"] == pytest.approx(4 / 9)


def test_a_plan_after_a_prefix_cuts_on_the_window_target_counted_from_the_boundary():
    rows, schedule, walk = _plan_world()
    after = tlog_schedule.require_walk_prefix(walk[:4], rows, schedule)
    for target, taken in [(1, 1), (4, 1), (5, 2), (100, 5)]:
        report = tlog_schedule.plan_prefix(rows, schedule, windows=target, after=after)
        assert (report["rows"], report["windows"]) == (taken, 4 * taken)


def test_a_plan_without_a_prefix_is_exactly_round_twos_report():
    rows, schedule, _ = _plan_world()
    plain = tlog_schedule.plan_prefix(rows, schedule, windows=18)
    assert plain == tlog_schedule.plan_prefix(rows, schedule, windows=18, after=None)
    assert not {"surahs_new", "surahs_cumulative", "share_of_pool_cumulative"} & set(plain)


def test_the_committed_round_two_plan_is_unchanged_by_the_offset():
    report = tlog_schedule.plan_prefix(_real_rows(),
                                       tlog_schedule.read_schedule(), windows=PREFIX_WINDOWS)
    committed = json.loads(tlog_schedule.PREFIX_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(report))


def _patch_pool(monkeypatch, rows):
    monkeypatch.setattr(tlog_pool, "load_training_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(tlog_pool, "training_use", lambda *args, **kwargs: _use(rows))


def _write_keys(path: Path, keys) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps({"split": s, "shard_file": f, "row": r}) + "\n"
                          for s, f, r in keys)


def test_plan_writes_the_report_after_the_keys_file(tmp_path, monkeypatch):
    rows, schedule, walk = _plan_world()
    _patch_pool(monkeypatch, rows)
    schedule_path, keys_path = tmp_path / "schedule.json", tmp_path / "keys.jsonl.gz"
    tlog_schedule.write_schedule(schedule, schedule_path)
    _write_keys(keys_path, walk[:6])
    out = tmp_path / "prefix.round3.json"
    report = tlog_schedule.plan(windows=8, schedule_path=schedule_path, prefix_path=out,
                                after=keys_path)
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(report))
    assert (report["rows"], report["surahs_new"]) == (2, [36])


def test_plan_refuses_a_keys_file_that_is_not_a_walk_prefix_before_writing(tmp_path, monkeypatch):
    rows, schedule, walk = _plan_world()
    _patch_pool(monkeypatch, rows)
    schedule_path, keys_path = tmp_path / "schedule.json", tmp_path / "keys.jsonl.gz"
    tlog_schedule.write_schedule(schedule, schedule_path)
    _write_keys(keys_path, walk[1:7])  # a shifted window of the walk
    out = tmp_path / "prefix.round3.json"
    with pytest.raises(ValueError):
        tlog_schedule.plan(windows=8, schedule_path=schedule_path, prefix_path=out,
                           after=keys_path)
    assert not out.exists()


def test_the_cli_offers_after_and_out_on_plan(capsys):
    with pytest.raises(SystemExit) as exit_:
        tlog_schedule.main(["plan", "--help"])
    assert exit_.value.code == 0
    help_ = capsys.readouterr().out
    assert "--after" in help_ and "--out" in help_


def test_the_cli_out_is_the_prefix_path(tmp_path, monkeypatch, capsys):
    rows, schedule, walk = _plan_world()
    _patch_pool(monkeypatch, rows)
    schedule_path, keys_path = tmp_path / "schedule.json", tmp_path / "keys.jsonl.gz"
    tlog_schedule.write_schedule(schedule, schedule_path)
    _write_keys(keys_path, walk[:6])
    out = tmp_path / "prefix.round3.json"
    tlog_schedule.main(["plan", "--after", str(keys_path), "--windows", "8",
                        "--schedule-path", str(schedule_path), "--out", str(out)])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed == json.loads(out.read_text(encoding="utf-8"))
    assert printed["surahs_new"] == [36]


def test_the_cli_plans_the_real_round_three_prefix_without_touching_the_committed_files(
        tmp_path, capsys):
    out = tmp_path / "training_prefix.round3.json"
    before = sha256_file(tlog_schedule.PREFIX_PATH)
    tlog_schedule.main(["plan", "--after", str(CONSUMED), "--windows", str(PREFIX_WINDOWS),
                        "--out", str(out)])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed == json.loads(out.read_text(encoding="utf-8"))
    assert printed["target_windows"] == PREFIX_WINDOWS
    # The cut includes the row that crosses the target: at most one 300 s clip (119 windows) over.
    assert PREFIX_WINDOWS <= printed["windows"] < PREFIX_WINDOWS + 130
    assert printed["rows"] == printed["clips"] > 0
    assert set(printed["splits"]) <= {"clean", "unclean"}
    pool = len(_real_rows())
    assert printed["share_of_pool_cumulative"] == pytest.approx(
        (CONSUMED_ROWS + printed["rows"]) / pool)
    assert set(printed["surahs_cumulative"]) == (
        set(printed["surahs_covered"]) | _round_two_surahs())
    assert sha256_file(tlog_schedule.PREFIX_PATH) == before


def _round_two_surahs() -> set[int]:
    """The surahs of the clean rows round 2 consumed, from their filename labels."""
    keys = frozenset(tlog_schedule.read_consumed_keys(CONSUMED))
    return {int(row["filename_label"].split(":")[0]) for row in _real_rows()
            if row["split"] == "clean" and row.get("filename_label") and _key(row) in keys}

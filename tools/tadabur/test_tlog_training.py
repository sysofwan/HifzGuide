"""Acceptance tests for tlog S8: the ``tlog.training`` row-set use and the streaming reader (#193).

Committed before the code exists and **locked**: later turns may add tests but never weaken or
rewrite these. The decisions are the brief's: the owner-approved gating amendment
(``docs/tlog-gating-amendment.md`` §4) leaves the training pool **without** a ``non_recitation``
term, so nothing here decodes the pool; ``tlog.training`` joins ``EXTERNAL_USES`` (orchestrator
Q1), ``overlay_commit`` is the last commit touching the overlay *data* (Q2), and the reader reads
from the verified committed use (Q3).

Contract this file pins
-----------------------
``tadabur.exposure`` gains:

* ``TLOG_TRAINING = "tlog.training"``, a member of ``EXTERNAL_USES`` and of neither ``CLIP_USES``
  nor ``TLOG_USES`` (those stay the evaluation uses).
* ``RowSetExposure(overlay_commit, predicate, count, rows_sha256)``: a frozen dataclass.
  ``overlay_commit`` is 40 lowercase hex, ``predicate`` is an object with exactly the non-empty
  strings ``text`` and ``code``, ``count`` a non-negative int (not a bool), ``rows_sha256`` 64
  lowercase hex. ``parse_row_set_exposure(data, where)`` validates one (a ``ValueError`` names
  ``where``); ``write_row_set_use(use, row_set, directory=EXPOSURE_DIR)`` writes
  ``<directory>/<use>.rows.json``; only ``TLOG_TRAINING`` is a row-set use (``ValueError``
  otherwise, on write and on load). ``write_empty(TLOG_TRAINING, directory)`` writes a row set of
  ``count`` 0 and the SHA-256 of no keys, so the suites that ``write_empty`` every missing use
  keep working.
* ``ExposureRegistry.row_sets``: use -> ``RowSetExposure``, loaded by ``load_registry``;
  ``missing()`` lists ``tlog.training`` until its file is written.
* ``check_disjoint`` refuses ``tlog.training`` in either position (a ``ValueError``), like every
  external use.
* ``check_row_set_disjoint(rows, *, registry=None, eval_groups=None)``: the training rows against
  **every** evaluation use. ``rows`` are pool rows (``split, shard_file, row, audio_path,
  audio_sha256, group_key, eval_overlap, duplicate_of``). It raises ``ExposureOverlap`` whose
  message contains the kind — ``audio_path`` (a clip id of any ``registry.clips`` use equals a
  row's ``audio_path``), ``audio_sha256`` (a clip's or a QUD ayah's checksum equals a row's),
  ``fingerprint`` (a row has S4 ``eval_overlap`` hits, or its ``duplicate_of`` is not itself one
  of ``rows``) and ``group_key`` (a row's group is in ``eval_groups``, a ``use -> {group_key}``
  map of the tlog evaluation uses; default: the split tree's groups of each ``tlog.<role>``).

``tadabur.tlog_pool`` gains:

* ``TRAINING_PREDICATE = {"text": ..., "code": "tadabur.tlog_pool.training_rows"}``; the text
  has no ``non_recitation`` term.
* ``training_rows(pool_rows)``: the rows with ``role == "train"`` and ``usable``, in row-key order.
* ``rows_digest(rows)``: SHA-256 hex of the row keys sorted as ``(split, shard_file, row)`` and
  serialised ``f"{split}:{shard_file}:{row}\\n"`` each, concatenated, UTF-8.
* ``OVERLAY_PATHS`` (the repo-relative overlay data), ``overlay_data_commit(repo=ROOT)`` (the last
  commit touching them) and ``require_overlay_at(commit, repo=ROOT)`` (``ValueError`` unless the
  overlay in the working tree is exactly that commit's).
* ``load_training_rows(registry=None, *, pool_rows=None, repo=ROOT)``: the verified rows — overlay
  at the use's commit, predicate equal to ``TRAINING_PREDICATE``, count and hash matching
  (``ValueError`` naming ``overlay``, ``predicate``, ``count`` or ``rows_sha256``).
* ``iter_training_rows(rows=None, *, cache_dir, stream=None, load_audio=None)``: yields
  ``((split, shard_file, row), samples)`` for exactly ``rows`` (default ``load_training_rows()``),
  shard by shard in ``(split, shard_file)`` order and row order, reading ``stream(f"data/{stem}
  .parquet")`` (default :func:`tadabur.tlog_pool.shard_rows`-backed, deleting each shard after
  use even when the reader stops early or raises). ``load_audio`` defaults to
  ``default_audio``. A row that is not a predicate row, a checksum mismatch and a wanted row the
  shard lacks are ``ValueError`` before that row is decoded. The first log line reports the
  windows per clip at the round-2 5 s window and 2.5 s hop.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tadabur import exposure, tlog_pool
from tadabur.exposure import (
    EXTERNAL_USES,
    ClipExposure,
    ExposureIncomplete,
    ExposureOverlap,
    QudExposure,
    check_disjoint,
    load_registry,
    require_complete,
    write_clip_use,
    write_qud_use,
)
from tadabur.nonpro_sources import TLOG
from tadabur.tlog_overlap import iter_pool_rows

ROOT = Path(__file__).resolve().parents[2]
TLOG_DIR = Path(__file__).parent / "tlog"
USE_PATH = Path(__file__).parent / "exposure" / "tlog.training.rows.json"
CLEAN_0, CLEAN_1, UNCLEAN_0 = "clean-00000-of-00411", "clean-00001-of-00411", "unclean-00000-of-00332"
FROZEN_TRAINING_COUNT = 426_825  # overlap/meta.json usable_after_s4["train"]
REVISION = "a" * 40
TRAINING_PATH_FIELDS = ("overlay_commit", "predicate", "count", "rows_sha256")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key(row) -> tuple[str, str, int]:
    return row["split"], row["shard_file"], row["row"]


def _digest(keys) -> str:
    """The pinned ``rows_sha256``, restated here so the committed use is checked against the rule."""
    text = "".join(f"{split}:{shard}:{row}\n" for split, shard, row in sorted(keys))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload(row: int) -> bytes:
    return f"audio-{row}".encode()


def _pool_row(row: int, *, split: str = "clean", shard: str = CLEAN_0, role: str = "train",
              usable: bool = True, payload: bytes | None = None, **overrides) -> dict:
    """One pool row (``iter_pool_rows``' fields the S8 code reads)."""
    record = {"split": split, "shard_file": shard, "row": row,
              "audio_path": f"100_{row}_{row}.flac",
              "audio_sha256": _sha256(_payload(row) if payload is None else payload),
              "duration_s": 12.0, "role": role, "usable": usable, "group_key": f"group-{row}",
              "eval_overlap": [], "duplicate_of": None}
    record.update(overrides)
    return record


@lru_cache(maxsize=1)
def _real_pool() -> tuple[dict, ...]:
    return tuple(iter_pool_rows())


@lru_cache(maxsize=1)
def _real_training() -> tuple[dict, ...]:
    return tuple(sorted((row for row in _real_pool() if row["role"] == "train" and row["usable"]),
                        key=_key))


# --- the vocabulary -----------------------------------------------------------------------------


def test_tlog_training_is_the_one_external_use_that_is_not_an_evaluation_use():
    assert exposure.TLOG_TRAINING == "tlog.training"
    assert exposure.TLOG_TRAINING in EXTERNAL_USES
    assert exposure.TLOG_TRAINING not in exposure.CLIP_USES
    assert exposure.TLOG_TRAINING not in exposure.TLOG_USES
    assert exposure.TLOG_USES == {f"tlog.{role}" for role in
                                  ("look", "confirm.1", "confirm.2", "confirm.3", "confirm.4")}


@pytest.mark.parametrize("position", ["first", "last"])
def test_check_disjoint_refuses_the_training_use_in_any_evaluation_context(position, tmp_path):
    exposure.write_use(exposure.LABEL_WAQF_EVENTS,
                       [exposure.Exposure("w.wav", 1, 1, 1, None, None, None)], tmp_path)
    exposure.write_empty(exposure.TLOG_TRAINING, tmp_path)
    registry = load_registry(tmp_path)
    args = ((exposure.TLOG_TRAINING, exposure.LABEL_WAQF_EVENTS) if position == "first"
            else (exposure.LABEL_WAQF_EVENTS, exposure.TLOG_TRAINING))
    with pytest.raises(ValueError):
        check_disjoint(*args, registry=registry)


# --- the row-set file ---------------------------------------------------------------------------


def _row_set(**overrides) -> exposure.RowSetExposure:
    fields = {"overlay_commit": "b" * 40, "predicate": dict(tlog_pool.TRAINING_PREDICATE),
              "count": 3, "rows_sha256": "c" * 64}
    fields.update(overrides)
    return exposure.RowSetExposure(**fields)


def test_a_row_set_use_round_trips_through_its_rows_json_file(tmp_path):
    row_set = _row_set()
    exposure.write_row_set_use(exposure.TLOG_TRAINING, row_set, tmp_path)
    path = tmp_path / "tlog.training.rows.json"
    assert set(json.loads(path.read_text(encoding="utf-8"))) == set(TRAINING_PATH_FIELDS)
    assert load_registry(tmp_path).row_sets == {exposure.TLOG_TRAINING: row_set}


def test_the_row_set_use_is_missing_until_written_and_empty_is_a_zero_row_set(tmp_path):
    exposure.write_use(exposure.LABEL_WAQF_EVENTS,
                       [exposure.Exposure("w.wav", 1, 1, 1, None, None, None)], tmp_path)
    assert exposure.TLOG_TRAINING in load_registry(tmp_path).missing()
    exposure.write_empty(exposure.TLOG_TRAINING, tmp_path)
    registry = load_registry(tmp_path)
    assert exposure.TLOG_TRAINING not in registry.missing()
    assert registry.row_sets[exposure.TLOG_TRAINING].count == 0
    assert registry.row_sets[exposure.TLOG_TRAINING].rows_sha256 == _digest([])


@pytest.mark.parametrize("change", [
    {"overlay_commit": "B" * 40}, {"overlay_commit": "b" * 39}, {"rows_sha256": "c" * 63},
    {"count": -1}, {"count": True}, {"count": "3"},
    {"predicate": {"text": "role == train"}}, {"predicate": {"text": "", "code": "x"}},
    {"predicate": {"text": "t", "code": "c", "extra": "e"}}, {"predicate": "role == train"},
])
def test_a_malformed_row_set_is_refused_naming_where(change):
    data = {**{"overlay_commit": "b" * 40, "predicate": {"text": "t", "code": "c"}, "count": 3,
               "rows_sha256": "c" * 64}, **change}
    with pytest.raises(ValueError, match="here:1"):
        exposure.parse_row_set_exposure(data, "here:1")


def test_a_row_set_with_a_missing_or_extra_field_is_refused():
    data = {"overlay_commit": "b" * 40, "predicate": {"text": "t", "code": "c"}, "count": 3,
            "rows_sha256": "c" * 64}
    for broken in ({k: v for k, v in data.items() if k != "count"}, {**data, "note": "x"}):
        with pytest.raises(ValueError):
            exposure.parse_row_set_exposure(broken, "here")


def test_only_the_training_use_is_a_row_set_use(tmp_path):
    with pytest.raises(ValueError):
        exposure.write_row_set_use("tlog.look", _row_set(), tmp_path)
    (tmp_path / "tlog.look.rows.json").write_text(json.dumps(
        {"overlay_commit": "b" * 40, "predicate": {"text": "t", "code": "c"}, "count": 1,
         "rows_sha256": "c" * 64}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registry(tmp_path)


# --- the predicate, the digest and the training rows -------------------------------------------


def test_the_predicate_names_its_code_and_has_no_non_recitation_term():
    predicate = tlog_pool.TRAINING_PREDICATE
    assert set(predicate) == {"text", "code"}
    assert predicate["code"] == "tadabur.tlog_pool.training_rows"
    assert "non_recitation" not in predicate["text"]  # the gating amendment §4 withdrew the term
    assert "train" in predicate["text"] and "silence" in predicate["text"]


def test_training_rows_are_the_usable_train_rows_in_key_order_and_no_tranche_row():
    pool = [_pool_row(2), _pool_row(0, shard=CLEAN_1), _pool_row(1, usable=False),
            _pool_row(3, role="look"), _pool_row(4, role="confirm.1"),
            _pool_row(5, role="confirm.4"), _pool_row(6, split="unclean", shard=UNCLEAN_0),
            _pool_row(1)]
    kept = tlog_pool.training_rows(iter(pool))
    assert [_key(row) for row in kept] == [("clean", CLEAN_0, 1), ("clean", CLEAN_0, 2),
                                           ("clean", CLEAN_1, 0), ("unclean", UNCLEAN_0, 6)]
    assert {row["role"] for row in kept} == {"train"} and all(row["usable"] for row in kept)


def test_rows_digest_hashes_the_sorted_keys_whatever_the_input_order():
    rows = [_pool_row(2), _pool_row(0, shard=CLEAN_1), _pool_row(1), _pool_row(9, split="unclean",
                                                                              shard=UNCLEAN_0)]
    expected = _digest(_key(row) for row in rows)
    assert tlog_pool.rows_digest(rows) == expected
    assert tlog_pool.rows_digest(list(reversed(rows))) == expected
    assert tlog_pool.rows_digest(rows[:-1]) != expected
    assert tlog_pool.rows_digest([]) == _sha256(b"")


# --- the overlay commit -------------------------------------------------------------------------


def _git(directory: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
                           "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                          cwd=directory, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def overlay_repo(tmp_path) -> Path:
    """A temporary repository holding every overlay path, committed once."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    for number, relative in enumerate(tlog_pool.OVERLAY_PATHS):
        file = repo / relative if relative.endswith(".json") else repo / relative / "x.txt"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"v1-{number}", encoding="utf-8")
    (repo / "other.txt").write_text("a", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "overlay")
    return repo


def _first_overlay_file(repo: Path) -> Path:
    return next(path for path in sorted((repo / "tools").rglob("*")) if path.is_file())


def test_the_overlay_paths_are_the_four_data_trees_and_the_silence_floors():
    assert set(tlog_pool.OVERLAY_PATHS) == {
        "tools/tadabur/tlog/index", "tools/tadabur/tlog/pass1", "tools/tadabur/tlog/split",
        "tools/tadabur/tlog/overlap", "tools/tadabur/tlog/silence_floors.json"}


def test_the_overlay_commit_is_the_last_commit_touching_the_overlay_data(overlay_repo):
    first = _git(overlay_repo, "rev-parse", "HEAD")
    assert tlog_pool.overlay_data_commit(overlay_repo) == first
    (overlay_repo / "other.txt").write_text("b", encoding="utf-8")
    _git(overlay_repo, "commit", "-aqm", "not overlay")
    assert tlog_pool.overlay_data_commit(overlay_repo) == first
    _first_overlay_file(overlay_repo).write_text("v2", encoding="utf-8")
    _git(overlay_repo, "commit", "-aqm", "overlay again")
    assert tlog_pool.overlay_data_commit(overlay_repo) == _git(overlay_repo, "rev-parse", "HEAD")


def test_require_overlay_at_accepts_exactly_the_overlay_of_that_commit(overlay_repo):
    first = _git(overlay_repo, "rev-parse", "HEAD")
    tlog_pool.require_overlay_at(first, overlay_repo)
    (overlay_repo / "other.txt").write_text("b", encoding="utf-8")
    _git(overlay_repo, "commit", "-aqm", "not overlay")
    tlog_pool.require_overlay_at(first, overlay_repo)  # the overlay did not move


@pytest.mark.parametrize("change", ["modified", "new_file", "deleted", "later_commit"])
def test_require_overlay_at_refuses_an_overlay_that_moved(change, overlay_repo):
    first = _git(overlay_repo, "rev-parse", "HEAD")
    target = _first_overlay_file(overlay_repo)
    if change == "modified":
        target.write_text("changed", encoding="utf-8")
    elif change == "new_file":
        (target.parent / "new.txt").write_text("new", encoding="utf-8")
    elif change == "deleted":
        target.unlink()
    else:
        target.write_text("changed", encoding="utf-8")
        _git(overlay_repo, "commit", "-aqm", "overlay again")
    with pytest.raises(ValueError, match="overlay"):
        tlog_pool.require_overlay_at(first, overlay_repo)


def test_require_overlay_at_refuses_an_unknown_commit(overlay_repo):
    with pytest.raises(ValueError, match="overlay"):
        tlog_pool.require_overlay_at("f" * 40, overlay_repo)


# --- the verified loader ------------------------------------------------------------------------


def _loader_inputs(tmp_path, overlay_repo, **change):
    """A tmp registry holding one verified-by-construction row set, and the pool it came from."""
    pool = [_pool_row(0), _pool_row(1), _pool_row(2, usable=False), _pool_row(3, role="look"),
            _pool_row(4, shard=CLEAN_1)]
    training = [row for row in pool if row["role"] == "train" and row["usable"]]
    fields = {"overlay_commit": _git(overlay_repo, "rev-parse", "HEAD"),
              "predicate": dict(tlog_pool.TRAINING_PREDICATE), "count": len(training),
              "rows_sha256": _digest(_key(row) for row in training)}
    fields.update(change)
    directory = tmp_path / "registry"
    exposure.write_row_set_use(exposure.TLOG_TRAINING, exposure.RowSetExposure(**fields),
                               directory)
    return load_registry(directory), pool, training


def test_the_loader_re_materialises_and_returns_the_verified_rows(tmp_path, overlay_repo):
    registry, pool, training = _loader_inputs(tmp_path, overlay_repo)
    rows = tlog_pool.load_training_rows(registry, pool_rows=iter(pool), repo=overlay_repo)
    assert [_key(row) for row in rows] == [_key(row) for row in training]


@pytest.mark.parametrize(("change", "word"), [
    ({"count": 2}, "count"), ({"rows_sha256": "d" * 64}, "rows_sha256"),
    ({"predicate": {"text": "role == train", "code": "tadabur.tlog_pool.training_rows"}},
     "predicate"),
])
def test_the_loader_refuses_a_count_hash_or_predicate_that_does_not_match(
        change, word, tmp_path, overlay_repo):
    registry, pool, _ = _loader_inputs(tmp_path, overlay_repo, **change)
    with pytest.raises(ValueError, match=word):
        tlog_pool.load_training_rows(registry, pool_rows=pool, repo=overlay_repo)


def test_the_loader_refuses_a_pool_that_gained_or_lost_a_row(tmp_path, overlay_repo):
    registry, pool, _ = _loader_inputs(tmp_path, overlay_repo)
    with pytest.raises(ValueError):
        tlog_pool.load_training_rows(registry, pool_rows=pool + [_pool_row(7)],
                                     repo=overlay_repo)
    with pytest.raises(ValueError):
        tlog_pool.load_training_rows(registry, pool_rows=pool[:1], repo=overlay_repo)


def test_the_loader_refuses_an_overlay_that_is_not_the_recorded_commit(tmp_path, overlay_repo):
    registry, pool, _ = _loader_inputs(tmp_path, overlay_repo)
    _first_overlay_file(overlay_repo).write_text("moved", encoding="utf-8")
    with pytest.raises(ValueError, match="overlay"):
        tlog_pool.load_training_rows(registry, pool_rows=pool, repo=overlay_repo)


def test_the_loader_needs_the_use_to_be_written(tmp_path, overlay_repo):
    exposure.write_use(exposure.LABEL_WAQF_EVENTS,
                       [exposure.Exposure("w.wav", 1, 1, 1, None, None, None)], tmp_path)
    with pytest.raises(ExposureIncomplete):
        tlog_pool.load_training_rows(load_registry(tmp_path), pool_rows=[], repo=overlay_repo)


# --- disjointness: a planted collision of each kind --------------------------------------------


def _eval_clip(name="eval-1.flac", sha="e", dataset="obadx/qdat") -> ClipExposure:
    return ClipExposure(dataset, REVISION, name, "1:1", sha * 64 if len(sha) == 1 else sha,
                        "unknown")


def _qud(sha="9") -> QudExposure:
    return QudExposure(REVISION, "mushaf", "1:1", sha * 64 if len(sha) == 1 else sha, "reciter")


def _disjoint_registry(tmp_path, *, clips=None, qud=()):
    clips = {exposure.NONPRO_EVALUATION: [_eval_clip()]} if clips is None else clips
    for use, rows in clips.items():
        write_clip_use(use, rows, tmp_path)
    write_qud_use(exposure.QUD_CORRECT_RECITATION, list(qud), tmp_path)
    return load_registry(tmp_path)


def _training(count=3):
    return [_pool_row(row) for row in range(count)]


def test_a_clean_training_set_is_disjoint_from_every_evaluation_use(tmp_path):
    registry = _disjoint_registry(tmp_path, clips={
        exposure.NONPRO_EVALUATION: [_eval_clip()],
        "tlog.look": [_eval_clip("look-1.flac", "1", TLOG.repo)],
        "tlog.confirm.1": [_eval_clip("c1.flac", "c", TLOG.repo)]}, qud=[_qud()])
    exposure.check_row_set_disjoint(_training(), registry=registry,
                           eval_groups={"tlog.look": {"look-group"}})


@pytest.mark.parametrize("use", [exposure.NONPRO_EVALUATION, "tlog.look", "tlog.confirm.3"])
def test_a_planted_clip_id_collision_is_found_in_every_clip_use(use, tmp_path):
    clips = {exposure.NONPRO_EVALUATION: [_eval_clip()], use: [_eval_clip("100_1_1.flac", "d")]}
    registry = _disjoint_registry(tmp_path, clips=clips)
    with pytest.raises(ExposureOverlap, match="audio_path"):
        exposure.check_row_set_disjoint(_training(), registry=registry, eval_groups={})


@pytest.mark.parametrize("where", ["clip", "tlog_clip", "qud"])
def test_a_planted_checksum_collision_is_found_in_clips_and_qud_ayat(where, tmp_path):
    stolen = _pool_row(1)["audio_sha256"]
    clips = {exposure.NONPRO_EVALUATION: [_eval_clip()]}
    qud = []
    if where == "clip":
        clips[exposure.NONPRO_EVALUATION].append(_eval_clip("other.wav", stolen))
    elif where == "tlog_clip":
        clips["tlog.confirm.2"] = [_eval_clip("another-path.flac", stolen, TLOG.repo)]
    else:
        qud = [_qud(stolen)]
    registry = _disjoint_registry(tmp_path, clips=clips, qud=qud)
    with pytest.raises(ExposureOverlap, match="audio_sha256"):
        exposure.check_row_set_disjoint(_training(), registry=registry, eval_groups={})


def test_a_row_with_a_fingerprint_match_to_an_evaluation_set_is_a_collision(tmp_path):
    rows = _training()
    rows[1] = {**rows[1], "eval_overlap": [{"set": "qdat", "clip": "q-9", "ber": 0.1}]}
    with pytest.raises(ExposureOverlap, match="fingerprint"):
        exposure.check_row_set_disjoint(rows, registry=_disjoint_registry(tmp_path), eval_groups={})


def test_a_duplicate_of_a_row_outside_the_training_set_is_a_collision(tmp_path):
    registry = _disjoint_registry(tmp_path)
    outside = {"split": "clean", "shard_file": CLEAN_0, "row": 99}
    inside = {"split": "clean", "shard_file": CLEAN_0, "row": 0}
    rows = _training()
    exposure.check_row_set_disjoint([{**rows[0]}, {**rows[1], "duplicate_of": inside}, rows[2]],
                           registry=registry, eval_groups={})
    with pytest.raises(ExposureOverlap, match="fingerprint"):
        exposure.check_row_set_disjoint([rows[0], {**rows[1], "duplicate_of": outside}, rows[2]],
                               registry=registry, eval_groups={})


def test_a_training_row_in_a_tlog_evaluation_group_is_a_collision(tmp_path):
    registry = _disjoint_registry(tmp_path)
    with pytest.raises(ExposureOverlap, match="group_key") as error:
        exposure.check_row_set_disjoint(_training(), registry=registry,
                               eval_groups={"tlog.confirm.2": {"group-1", "other"}})
    assert "tlog.confirm.2" in str(error.value)
    assert isinstance(error.value, ValueError)


# --- the reader on a fake shard ----------------------------------------------------------------


def _fake_stream(shards: dict[str, list[bytes]], calls: list[str]):
    """``stream(shard)`` over fake shards of raw audio bytes, recording each call."""
    def stream(shard: str):
        calls.append(shard)
        return iter([{"audio": {"path": f"{index}.flac", "bytes": payload}}
                     for index, payload in enumerate(shards[shard])])
    return stream


def _recording_loader(loaded: list[bytes], samples=None):
    def load(data: bytes):
        loaded.append(data)
        return np.zeros(192_000, dtype=np.float32) if samples is None else samples
    return load


def _shard_payloads(count=4, tag="") -> list[bytes]:
    return [f"{tag}audio-{row}".encode() for row in range(count)]


def _wanted(payloads, rows, *, split="clean", shard=CLEAN_0, **overrides):
    return [_pool_row(row, split=split, shard=shard, payload=payloads[row], **overrides)
            for row in rows]


def test_the_reader_yields_exactly_the_wanted_rows_with_their_decoded_audio(tmp_path):
    payloads = _shard_payloads()
    calls, loaded = [], []
    samples = np.arange(192_000, dtype=np.float32)
    rows = _wanted(payloads, [3, 1])
    stream = _fake_stream({"data/clean-00000-of-00411.parquet": payloads}, calls)
    out = list(tlog_pool.iter_training_rows(
        rows, cache_dir=tmp_path, stream=stream, load_audio=_recording_loader(loaded, samples)))
    assert [key for key, _ in out] == [("clean", CLEAN_0, 1), ("clean", CLEAN_0, 3)]
    assert all(audio is samples for _, audio in out)
    assert loaded == [payloads[1], payloads[3]]  # rows 0 and 2 are never decoded
    assert calls == ["data/clean-00000-of-00411.parquet"]


def test_the_reader_goes_shard_by_shard_in_split_then_shard_order(tmp_path):
    calls, loaded = [], []
    names = {"data/clean-00000-of-00411.parquet": _shard_payloads(2, "a"),
             "data/clean-00001-of-00411.parquet": _shard_payloads(2, "b"),
             "data/unclean-00000-of-00332.parquet": _shard_payloads(2, "c")}
    rows = (_wanted(names["data/unclean-00000-of-00332.parquet"], [1, 0], split="unclean",
                    shard=UNCLEAN_0)
            + _wanted(names["data/clean-00001-of-00411.parquet"], [0], shard=CLEAN_1)
            + _wanted(names["data/clean-00000-of-00411.parquet"], [1]))
    out = list(tlog_pool.iter_training_rows(
        rows, cache_dir=tmp_path, stream=_fake_stream(names, calls),
        load_audio=_recording_loader(loaded)))
    assert calls == list(names)  # clean-00000, clean-00001, unclean-00000
    assert [key for key, _ in out] == [("clean", CLEAN_0, 1), ("clean", CLEAN_1, 0),
                                       ("unclean", UNCLEAN_0, 0), ("unclean", UNCLEAN_0, 1)]


def test_the_reader_refuses_a_checksum_mismatch_before_decoding_it(tmp_path):
    payloads = _shard_payloads()
    rows = _wanted(payloads, [0, 1])
    rows[1] = {**rows[1], "audio_sha256": "0" * 64}
    calls, loaded = [], []
    reader = tlog_pool.iter_training_rows(
        rows, cache_dir=tmp_path,
        stream=_fake_stream({"data/clean-00000-of-00411.parquet": payloads}, calls),
        load_audio=_recording_loader(loaded))
    with pytest.raises(ValueError, match="audio_sha256|checksum|recorded"):
        list(reader)
    assert loaded == [payloads[0]]  # the bad row's audio was never decoded


def test_the_reader_refuses_a_wanted_row_the_shard_lacks(tmp_path):
    payloads = _shard_payloads(2)
    rows = _wanted(payloads, [1]) + [_pool_row(7, payload=b"missing")]
    with pytest.raises(ValueError):
        list(tlog_pool.iter_training_rows(
            rows, cache_dir=tmp_path,
            stream=_fake_stream({"data/clean-00000-of-00411.parquet": payloads}, []),
            load_audio=_recording_loader([])))


@pytest.mark.parametrize("bad", [{"role": "confirm.1"}, {"role": "look"}, {"role": "confirm.4"},
                                 {"usable": False}])
def test_the_reader_refuses_a_row_outside_the_predicate_before_reading_any_shard(bad, tmp_path):
    payloads = _shard_payloads()
    rows = _wanted(payloads, [0]) + _wanted(payloads, [1], **bad)
    calls, loaded = [], []
    with pytest.raises(ValueError):
        list(tlog_pool.iter_training_rows(
            rows, cache_dir=tmp_path,
            stream=_fake_stream({"data/clean-00000-of-00411.parquet": payloads}, calls),
            load_audio=_recording_loader(loaded)))
    assert calls == [] and loaded == []


def test_the_reader_defaults_to_the_verified_committed_use(tmp_path, monkeypatch):
    payloads = _shard_payloads(2)
    verified = _wanted(payloads, [1])
    seen = []
    monkeypatch.setattr(tlog_pool, "load_training_rows",
                        lambda *args, **kwargs: seen.append(True) or verified)
    out = list(tlog_pool.iter_training_rows(
        cache_dir=tmp_path,
        stream=_fake_stream({"data/clean-00000-of-00411.parquet": payloads}, []),
        load_audio=_recording_loader([])))
    assert seen and [key for key, _ in out] == [("clean", CLEAN_0, 1)]


def test_the_first_log_line_reports_windows_per_clip_at_the_round_2_window_and_hop(
        tmp_path, capsys):
    """A 12 s clip at a 5 s window and 2.5 s hop starts windows at 0, 2.5, 5 and 7.5 s."""
    payloads = _shard_payloads()
    list(tlog_pool.iter_training_rows(
        _wanted(payloads, [0, 1, 2]), cache_dir=tmp_path,
        stream=_fake_stream({"data/clean-00000-of-00411.parquet": payloads}, []),
        load_audio=_recording_loader([])))
    first = capsys.readouterr().out.splitlines()[0]
    assert re.search(r"windows per clip\D*4(\.0+)?\b", first)


def test_the_default_decoder_yields_16_khz_mono_float32(tmp_path):
    import io

    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, np.zeros(8_000, dtype=np.float32), 8_000, format="FLAC")
    payload = buffer.getvalue()  # one second at 8 kHz
    row = _pool_row(0, payload=payload)
    stream = _fake_stream({"data/clean-00000-of-00411.parquet": [payload]}, [])
    [(key, audio)] = list(tlog_pool.iter_training_rows([row], cache_dir=tmp_path, stream=stream))
    assert key == ("clean", CLEAN_0, 0)
    assert audio.dtype == np.float32 and audio.ndim == 1 and len(audio) == 16_000


def _write_parquet(path: Path, payloads: list[bytes]) -> None:
    table = pa.table({"audio": pa.array(
        [{"path": f"{index}.flac", "bytes": payload} for index, payload in enumerate(payloads)],
        type=pa.struct([("path", pa.string()), ("bytes", pa.binary())]))})
    pq.write_table(table, path)


@pytest.fixture
def cached_shard(tmp_path, monkeypatch):
    """One tlog shard served by a fake ``hf_hub_download``: the parquet and the calls made."""
    import huggingface_hub

    parquet = tmp_path / "cached.parquet"
    payloads = _shard_payloads(3)
    _write_parquet(parquet, payloads)
    calls = []

    def fake_download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs.get("revision")))
        return str(parquet)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return parquet, payloads, calls


def test_the_reader_streams_the_pinned_revision_and_deletes_the_shard_after_use(
        cached_shard, tmp_path):
    parquet, payloads, calls = cached_shard
    out = list(tlog_pool.iter_training_rows(
        _wanted(payloads, [0, 2]), cache_dir=tmp_path / "cache",
        load_audio=_recording_loader([])))
    assert [key for key, _ in out] == [("clean", CLEAN_0, 0), ("clean", CLEAN_0, 2)]
    assert calls == [(TLOG.repo, "data/clean-00000-of-00411.parquet", TLOG.revision)]
    assert not parquet.exists()


def test_the_reader_deletes_the_shard_when_a_checksum_fails(cached_shard, tmp_path):
    parquet, payloads, _ = cached_shard
    rows = _wanted(payloads, [0, 1])
    rows[1] = {**rows[1], "audio_sha256": "0" * 64}
    with pytest.raises(ValueError) as error:
        list(tlog_pool.iter_training_rows(rows, cache_dir=tmp_path / "cache",
                                          load_audio=_recording_loader([])))
    # ``error`` keeps the failing frames alive, so only an explicit close deletes the shard
    assert error.type is ValueError
    assert not parquet.exists()


def test_the_reader_deletes_the_shard_when_the_consumer_stops_early(cached_shard, tmp_path):
    parquet, payloads, _ = cached_shard
    reader = tlog_pool.iter_training_rows(
        _wanted(payloads, [0, 1, 2]), cache_dir=tmp_path / "cache",
        load_audio=_recording_loader([]))
    next(reader)
    reader.close()
    assert not parquet.exists()


# --- the committed use --------------------------------------------------------------------------


def test_the_training_predicate_selects_the_frozen_426825_rows_of_the_train_role():
    rows = tlog_pool.training_rows(_real_pool())
    assert len(rows) == FROZEN_TRAINING_COUNT
    overlap = json.loads((TLOG_DIR / "overlap" / "meta.json").read_text(encoding="utf-8"))
    assert overlap["usable_after_s4"]["train"] == FROZEN_TRAINING_COUNT
    assert {row["role"] for row in rows} == {"train"} and all(row["usable"] for row in rows)
    assert [_key(row) for row in rows] == [_key(row) for row in _real_training()]


def test_the_committed_use_is_the_predicates_count_and_hash_at_the_overlay_commit():
    data = json.loads(USE_PATH.read_text(encoding="utf-8"))
    assert set(data) == set(TRAINING_PATH_FIELDS)
    assert data["count"] == FROZEN_TRAINING_COUNT
    assert data["rows_sha256"] == _digest(_key(row) for row in _real_training())
    assert data["predicate"] == tlog_pool.TRAINING_PREDICATE
    assert data["overlay_commit"] == tlog_pool.overlay_data_commit()
    tlog_pool.require_overlay_at(data["overlay_commit"])


def test_the_committed_registry_loads_the_use_and_stays_complete():
    registry = load_registry()
    require_complete(registry)
    assert set(registry.row_sets) == {exposure.TLOG_TRAINING}
    row_set = registry.row_sets[exposure.TLOG_TRAINING]
    assert (row_set.count, row_set.rows_sha256) == (
        FROZEN_TRAINING_COUNT, _digest(_key(row) for row in _real_training()))
    assert set(registry.clips) == {exposure.NONPRO_EVALUATION} | exposure.TLOG_USES


def test_the_loader_verifies_the_committed_use_against_the_committed_overlay():
    rows = tlog_pool.load_training_rows(pool_rows=iter(_real_pool()))
    assert [_key(row) for row in rows] == [_key(row) for row in _real_training()]
    assert not {row["role"] for row in rows} & {"look", "confirm.1", "confirm.2", "confirm.3",
                                                "confirm.4"}


def test_the_committed_training_set_is_disjoint_from_every_evaluation_use():
    exposure.check_row_set_disjoint(_real_training())


def test_the_use_file_is_not_gitignored():
    result = subprocess.run(["git", "check-ignore", "-q", str(USE_PATH)], cwd=ROOT)
    assert result.returncode == 1

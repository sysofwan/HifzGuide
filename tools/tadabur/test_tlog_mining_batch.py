"""One batch of phase A's work, and the merge of its parts (#263), torch-free.

The interface, in ``tadabur.tlog_mining.batch``:

* ``part_path(parts_dir, shard_file)``: ``<parts_dir>/word_times.<shard_file>.jsonl``, one file
  per shard, one record per line, appended as each clip finishes.
* ``worker_rows(rows, *, shard_index, shard_count)``: the rows of ``sorted(shards)[index::count]``
  (whole shards, as ``tlog_eval_decode`` splits them), so P workers never share a shard.
* ``run_batch(rows, parts_dir, *, align_row, budget_seconds, clock) -> BatchResult``: for each row
  in (shard_file, row) order whose key is not in its part yet, calls ``align_row(row)`` (the
  forward pass and :func:`~tadabur.tlog_mining.word_times.build_record`) and appends the returned
  record. The budget is checked **before** each row (``clock() - start >= budget`` stops), so a
  row is never cut off. ``BatchResult`` has ``done`` (no row of ``rows`` is left) and
  ``written``. A killed run's half-written last line is cut off on resume
  (:func:`training.external_decode.resume_keys`); a record whose ``key`` is not the row's
  :func:`training.nonpro_decode.item_key` is a ``ValueError``; a failing ``align_row`` propagates
  and keeps what was already written.

``tadabur.tlog_mining.merge.merge_word_times(parts_dir, out_dir, *, manifests, provenance)``
gathers every ``word_times.*.jsonl`` part into ``<out_dir>/<tranche>.jsonl.gz`` (sorted by key,
:func:`tadabur.tlog_index.write_shard`) and ``<out_dir>/meta.json`` (the provenance, plus
``tranches[<tranche>] = {rows, aligned, fallback, fallback_rate}``). ``manifests`` is exactly
``{"look", "confirm.1", "confirm.2"}`` mapped to their manifest rows; each tranche must hold
one record per **gating** row, no missing, repeated or foreign key, and no clip in two tranches.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tadabur.tlog_mining.batch import part_path, run_batch, worker_rows
from tadabur.tlog_mining.merge import merge_word_times

SHARDS = ("clean-00001-of-00411", "clean-00002-of-00411", "clean-00003-of-00411")


def _row(shard: str, number: int, stratum: str = "gating", prefix: str = "a") -> dict:
    return {"split": "clean", "shard_file": shard, "row": number,
            "audio_path": f"{prefix}_{shard.split('-')[1]}_{number}.flac", "stratum": stratum,
            "audio_sha256": "00" * 32, "filename_label": "2:1"}


def _key(row: dict) -> str:
    return f"{row['audio_path']}#0"


def _record(row: dict, *, aligned: bool = True) -> dict:
    return {"key": _key(row), "split": row["split"], "shard_file": row["shard_file"],
            "row": row["row"], "audio_path": row["audio_path"],
            "audio_sha256": row["audio_sha256"], "surah_ayah": row["filename_label"],
            "aligned": aligned, "reason": None if aligned else "too few frames",
            "word_times": [[0.0, 1.0]] if aligned else None}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def aligner(self, seconds: float = 10.0, calls: list | None = None):
        def align(row):
            if calls is not None:
                calls.append(row["row"])
            self.now += seconds
            return _record(row)
        return align


def _lines(parts: Path, shard: str) -> list[dict]:
    return [json.loads(line) for line in part_path(parts, shard).read_text("utf-8").splitlines()]


# --- the workers' shards --------------------------------------------------------------------------


def test_the_part_file_is_one_per_shard_under_the_parts_dir(tmp_path):
    assert part_path(tmp_path, SHARDS[0]) == tmp_path / f"word_times.{SHARDS[0]}.jsonl"


def test_workers_get_disjoint_whole_shards_that_cover_every_row():
    rows = [_row(shard, n) for shard in SHARDS for n in (1, 2)]
    shares = [worker_rows(rows, shard_index=i, shard_count=2) for i in range(2)]
    assert {row["shard_file"] for row in shares[0]} == {SHARDS[0], SHARDS[2]}
    assert {row["shard_file"] for row in shares[1]} == {SHARDS[1]}
    assert sorted(row["audio_path"] for share in shares for row in share) == sorted(
        row["audio_path"] for row in rows)


@pytest.mark.parametrize("index, count", [(2, 2), (-1, 2), (0, 0)])
def test_a_worker_index_outside_the_count_is_refused(index, count):
    with pytest.raises(ValueError):
        worker_rows([_row(SHARDS[0], 1)], shard_index=index, shard_count=count)


# --- a batch --------------------------------------------------------------------------------------


def test_a_batch_appends_one_record_per_row_in_shard_and_row_order(tmp_path):
    rows = [_row(SHARDS[1], 2), _row(SHARDS[0], 5), _row(SHARDS[0], 1)]
    calls: list[int] = []
    clock = Clock()
    result = run_batch(rows, tmp_path, align_row=clock.aligner(calls=calls),
                       budget_seconds=1000, clock=clock)
    assert result.done is True and result.written == 3
    assert calls == [1, 5, 2]
    assert [line["row"] for line in _lines(tmp_path, SHARDS[0])] == [1, 5]
    assert [line["row"] for line in _lines(tmp_path, SHARDS[1])] == [2]


def test_the_budget_is_checked_before_each_row_so_a_row_is_never_cut_off(tmp_path):
    rows = [_row(SHARDS[0], n) for n in range(1, 7)]
    clock = Clock()
    first = run_batch(rows, tmp_path, align_row=clock.aligner(10), budget_seconds=25, clock=clock)
    assert first.done is False and first.written == 3
    assert [line["row"] for line in _lines(tmp_path, SHARDS[0])] == [1, 2, 3]


def test_the_next_batch_resumes_by_key_and_finishes(tmp_path):
    rows = [_row(SHARDS[0], n) for n in range(1, 7)]
    clock, calls = Clock(), []
    run_batch(rows, tmp_path, align_row=clock.aligner(10, calls), budget_seconds=25, clock=clock)
    calls.clear()
    second = run_batch(rows, tmp_path, align_row=clock.aligner(10, calls),
                       budget_seconds=1000, clock=clock)
    assert second.done is True and second.written == 3
    assert calls == [4, 5, 6]
    assert [line["row"] for line in _lines(tmp_path, SHARDS[0])] == [1, 2, 3, 4, 5, 6]


def test_a_batch_with_nothing_left_writes_nothing_and_is_done(tmp_path):
    rows = [_row(SHARDS[0], 1)]
    clock = Clock()
    run_batch(rows, tmp_path, align_row=clock.aligner(), budget_seconds=1000, clock=clock)
    again = run_batch(rows, tmp_path, align_row=clock.aligner(), budget_seconds=1000, clock=clock)
    assert again.done is True and again.written == 0
    assert len(_lines(tmp_path, SHARDS[0])) == 1


def test_a_half_written_last_line_from_a_killed_run_is_cut_off_and_redone(tmp_path):
    rows = [_row(SHARDS[0], n) for n in (1, 2)]
    path = part_path(tmp_path, SHARDS[0])
    path.write_text(json.dumps(_record(rows[0])) + '\n{"key": "a_00001_2.fl', encoding="utf-8")
    clock, calls = Clock(), []
    run_batch(rows, tmp_path, align_row=clock.aligner(calls=calls), budget_seconds=1000,
              clock=clock)
    assert calls == [2]
    assert [line["row"] for line in _lines(tmp_path, SHARDS[0])] == [1, 2]


def test_a_failing_row_propagates_and_keeps_the_rows_already_written(tmp_path):
    rows = [_row(SHARDS[0], n) for n in (1, 2, 3)]

    def align(row):
        if row["row"] == 3:
            raise RuntimeError("forward failed")
        return _record(row)

    with pytest.raises(RuntimeError, match="forward failed"):
        run_batch(rows, tmp_path, align_row=align, budget_seconds=1000, clock=Clock())
    assert [line["row"] for line in _lines(tmp_path, SHARDS[0])] == [1, 2]


def test_a_record_for_another_clip_is_refused_and_not_written(tmp_path):
    rows = [_row(SHARDS[0], 1)]
    with pytest.raises(ValueError, match="key"):
        run_batch(rows, tmp_path, align_row=lambda row: _record(_row(SHARDS[0], 9)),
                  budget_seconds=1000, clock=Clock())
    assert not part_path(tmp_path, SHARDS[0]).exists() or not _lines(tmp_path, SHARDS[0])


# --- the merge ------------------------------------------------------------------------------------

TRANCHE_SHARDS = {"look": SHARDS[0], "confirm.1": SHARDS[1], "confirm.2": SHARDS[2]}
PROVENANCE = {"code_sha": "c" * 40, "pull_request": 270, "pull_request_ref": "refs/pull/270/head"}


def _manifests() -> dict[str, list[dict]]:
    return {tranche: [*(_row(shard, n, prefix=tranche[:1] + tranche[-1]) for n in (1, 2, 3, 4)),
                      _row(shard, 9, "reported:unclean", prefix=tranche[:1] + tranche[-1])]
            for tranche, shard in TRANCHE_SHARDS.items()}


def _write_parts(parts: Path, manifests, *, unaligned: tuple[int, ...] = (), skip=(), extra=()):
    parts.mkdir(exist_ok=True)
    for tranche, rows in manifests.items():
        shard = TRANCHE_SHARDS[tranche]
        records = [_record(row, aligned=row["row"] not in unaligned)
                   for row in rows if row["stratum"] == "gating" and (tranche, row["row"]) not in skip]
        records += [_record(row) for row in extra if row["shard_file"] == shard]
        with part_path(parts, shard).open("a", encoding="utf-8") as handle:
            handle.writelines(json.dumps(record) + "\n" for record in records)


def _merge(tmp_path, manifests=None, **kwargs):
    manifests = _manifests() if manifests is None else manifests
    return merge_word_times(tmp_path / "parts", tmp_path / "out", manifests=manifests,
                            provenance=PROVENANCE, **kwargs)


def _read(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_the_merge_writes_one_sorted_file_per_tranche_of_its_gating_rows(tmp_path):
    manifests = _manifests()
    _write_parts(tmp_path / "parts", manifests)
    _merge(tmp_path, manifests)
    for tranche, shard in TRANCHE_SHARDS.items():
        records = _read(tmp_path / "out" / f"{tranche}.jsonl.gz")
        assert [record["key"] for record in records] == sorted(
            _key(row) for row in manifests[tranche] if row["stratum"] == "gating")


def test_the_meta_reports_the_fallback_rate_per_tranche_with_the_provenance(tmp_path):
    manifests = _manifests()
    _write_parts(tmp_path / "parts", manifests, unaligned=(2,))
    _merge(tmp_path, manifests)
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["code_sha"] == PROVENANCE["code_sha"] and meta["pull_request"] == 270
    assert set(meta["tranches"]) == set(TRANCHE_SHARDS)
    for counts in meta["tranches"].values():
        assert counts == {"rows": 4, "aligned": 3, "fallback": 1, "fallback_rate": 0.25}


def test_an_unaligned_clip_is_committed_with_its_reason(tmp_path):
    manifests = _manifests()
    _write_parts(tmp_path / "parts", manifests, unaligned=(2,))
    _merge(tmp_path, manifests)
    (bad,) = [r for r in _read(tmp_path / "out" / "look.jsonl.gz") if not r["aligned"]]
    assert bad["word_times"] is None and bad["reason"]


def test_a_missing_gating_clip_is_refused_so_a_partial_run_never_looks_complete(tmp_path):
    manifests = _manifests()
    _write_parts(tmp_path / "parts", manifests, skip={("confirm.2", 3)})
    with pytest.raises(ValueError):
        _merge(tmp_path, manifests)
    assert not (tmp_path / "out" / "look.jsonl.gz").exists()


def test_a_repeated_clip_is_refused(tmp_path):
    manifests = _manifests()
    _write_parts(tmp_path / "parts", manifests)
    _write_parts(tmp_path / "parts", {"look": manifests["look"][:1]})
    with pytest.raises(ValueError):
        _merge(tmp_path, manifests)


def test_a_record_for_a_row_that_is_not_gating_is_refused(tmp_path):
    manifests = _manifests()
    _write_parts(tmp_path / "parts", manifests, extra=[manifests["look"][-1]])
    with pytest.raises(ValueError):
        _merge(tmp_path, manifests)


def test_a_clip_in_two_tranches_is_refused(tmp_path):
    manifests = _manifests()
    manifests["confirm.2"] = [*manifests["confirm.2"], manifests["confirm.1"][0]]
    _write_parts(tmp_path / "parts", manifests)
    with pytest.raises(ValueError):
        _merge(tmp_path, manifests)


@pytest.mark.parametrize("tranches", [("look", "confirm.1"), ("look", "confirm.1", "confirm.2",
                                                              "confirm.3")])
def test_only_the_three_phase_a_tranches_are_merged(tmp_path, tranches):
    manifests = {tranche: _manifests().get(tranche, _manifests()["look"]) for tranche in tranches}
    with pytest.raises(ValueError):
        _merge(tmp_path, manifests)

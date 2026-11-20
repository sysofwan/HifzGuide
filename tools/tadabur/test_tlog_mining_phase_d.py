"""Phase D of #263 (the teacher's word times for the mined rows): pins, rows, worker, merge, driver.

Phase D is phase A's teacher-only forced alignment, run over **phase B's row set** — the labelled
``reported:*`` rows of ``confirm.1`` and ``confirm.2`` (10,642 decodable, the 26 #248 rows listed)
— so #264 can snip single words from a mined row. It adds no alignment logic and no decode of any
student model. Everything here is torch-free; the launcher is
``test_tlog_mining_phase_d_launcher.py``.

Contract this file pins
-----------------------
``tadabur.tlog_mining.selection``

* ``PHASE_D == "D"``, ``PHASE_D_TRANCHES == ("confirm.1", "confirm.2")``.
* ``require_phase("D", *, amendment_path=..., round3_dir=...)`` passes only when the file at
  ``amendment_path`` hashes to the pinned SHA-256 (a missing or changed file is a ``ValueError``
  naming ``amendment`` and ``SHA-256``, as for phase B). Round 3's choice and verdict are **not**
  required: phase D reads no round-3 output.
* ``phase_d_rows(tranche, *, tranches_dir, split_dir, layout, spec, phonetizable=None)
  -> (decodable, unphonetizable)`` returns exactly what ``phase_b_rows`` returns, with the same
  claim, committed-choice and split-tree checks; any tranche but ``confirm.1``/``confirm.2`` is a
  ``ValueError`` naming ``confirm.1`` before a file opens.

``tadabur.tlog_mining.merge.merge_mined_word_times(parts_dir, out_dir, *, rows, unphonetizable,
provenance) -> meta``: ``rows`` and ``unphonetizable`` map ``confirm.1`` and ``confirm.2`` to
``phase_d_rows``' two lists; the parts are phase A's (``batch.part_path``). Every decodable row
must hold exactly one record; a missing, repeated, foreign or listed row, a clip in both
tranches or a wrong tranche set is a ``ValueError`` before anything is written. It writes **only**
``mined.confirm.1.jsonl.gz``, ``mined.confirm.2.jsonl.gz`` (records verbatim, sorted by key) and
``meta.mined.json``: ``provenance`` as given, ``tranches`` (per tranche ``rows``, ``aligned``,
``fallback``, ``fallback_rate`` as phase A's ``meta.json``), ``label_sources`` (a count over the
rows) and ``unphonetizable`` (``tranche``, ``key``, ``audio_filename``, ``assigned_label``,
``stratum`` per listed row). Phase A's ``meta.json`` and its three files are never written. The
bytes depend on the records only: not on part order, and a re-merge is byte-identical. Returns
the meta it wrote.

``tadabur.tlog_mining.__main__``

* ``align --phase D`` aligns each mined row against its **``assigned_label``** ayah (the filename
  label is missing for a third of the mined rows and differs from the assigned one for more),
  and writes phase A's record shape (same keys) with ``surah_ayah`` the label aligned. Its rows
  come from ``phase_d_rows`` only; ``decode`` refuses phase D.
* ``run --phase D`` reads both tranches through ``phase_d_rows`` only, writes the
  ``unphonetizable.<tranche>.json`` listing and the worklists, runs **one** worker whatever the
  card offers, and refuses existing ``word_times.*.jsonl`` parts without ``--resume``. Its batch
  spawns ``align`` workers and reads phase A's parts.
* ``merge --phase D`` pools the parts through ``merge_mined_word_times`` with the run's teacher
  record in the provenance, and prints per tranche ``<tranche>: N rows, A aligned, F fallback
  (R%)``, the label sources and ``unphonetizable: N rows listed (#248)``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tadabur.test_tlog_mining_cli import FRAMES, OFFSETS, REFERENCE, SHARD, _identity
from tadabur.test_tlog_mining_phase_b import (AMENDMENT, MINED, SHARD_OF, _claimed, _decode_args,
                                              _driver_args, _merge_inputs, _phonetizable,
                                              _quiet_box, _tranche)
from tadabur.test_tlog_mining_selection import _split_tree
from tadabur.tlog_ident import BASE_MODEL, BASE_REVISION
from tadabur.tlog_index import read_jsonl
from tadabur.tlog_mining import __main__ as cli
from tadabur.tlog_mining import merge, selection
from tadabur.tlog_mining.batch import part_path
from tadabur.tlog_mining.selection import phase_b_rows, require_phase
from tadabur.tlog_pass1 import read_jsonl_gz
from training.decoding import SPANS
from training.nonpro_decode import item_key

REPO = Path(__file__).resolve().parents[2]
MINING = REPO / "tools" / "tadabur" / "tlog_mining"
WORD_TIMES = MINING / "word_times"
DECODES = MINING / "decodes"
#: Phase A's committed files and their SHA-256 (``docs/tlog-mining-run.md``): phase D never edits
#: them.
PHASE_A_SHA256 = {
    "look.jsonl.gz": "09ae1a578fc86dab360eeaa93ad10241ac7036cbc698fe0d20fc1e4b9b0e077d",
    "confirm.1.jsonl.gz": "52d5619727b65ed9b0d5db48cc8285fbb5c246823dca528ffc4968b818f1b769",
    "confirm.2.jsonl.gz": "9a5206e0ff98c1c7a5d52fab029842ff21f9eb8b526a8c5d99ff9d0250d94e32",
    "meta.json": "5e84888f89e591cb08a2063795a2cb080b187bd01ec1249985eafdafc940ca06",
}
OUT_NAMES = ["mined.confirm.1.jsonl.gz", "mined.confirm.2.jsonl.gz", "meta.mined.json"]


# --- the pins ---------------------------------------------------------------------------------


def test_phase_d_runs_under_the_committed_amendment_alone(tmp_path):
    assert selection.PHASE_D == "D" and selection.PHASE_D_TRANCHES == MINED
    require_phase("D")
    # Round 3's choice and verdict are phase C's pins, not phase D's.
    require_phase("D", amendment_path=AMENDMENT, round3_dir=tmp_path / "no-round-3")


def test_phase_d_is_refused_for_a_changed_or_missing_amendment(tmp_path):
    changed = tmp_path / "amendment.md"
    changed.write_bytes(AMENDMENT.read_bytes() + b"\nA late edit.\n")
    for path in (changed, tmp_path / "absent.md"):
        with pytest.raises(ValueError, match="amendment") as raised:
            require_phase("D", amendment_path=path)
        assert "SHA-256" in str(raised.value)


def test_the_committed_phase_a_artifacts_are_the_ones_the_run_note_pins():
    """Phase D must leave phase A's files as they are: the SHA-256 the note recorded."""
    found = {name: hashlib.sha256((WORD_TIMES / name).read_bytes()).hexdigest()
             for name in PHASE_A_SHA256}
    assert found == PHASE_A_SHA256


# --- the rows ---------------------------------------------------------------------------------


@pytest.fixture
def pool(tmp_path):
    """Both claims, the committed choice and manifests with every stratum, plus the split tree."""
    return _claimed(tmp_path, {role: _tranche(role) for role in MINED})


def _select(fixture, tranche, rows=None, **kwargs):
    round_, split = fixture
    kwargs.setdefault("phonetizable", _phonetizable)
    rows = rows or selection.phase_d_rows
    return rows(tranche, tranches_dir=round_.tranches, split_dir=split, layout=round_.layout,
                spec=round_.spec, **kwargs)


def _keys(rows) -> list[str]:
    return sorted(item_key(row["audio_path"]) for row in rows)


@pytest.mark.parametrize("tranche", MINED)
def test_phase_d_rows_are_phase_bs_decodable_rows_and_phase_bs_listing(pool, tranche):
    decodable, listed = _select(pool, tranche)
    assert (decodable, listed) == _select(pool, tranche, phase_b_rows)
    assert sorted(row["row"] for row in decodable) == [3, 4, 5, 6]
    assert [row["row"] for row in listed] == [8]


def test_phase_d_never_takes_a_gating_or_an_unlabelled_unclean_row(pool):
    decodable, listed = _select(pool, "confirm.1")
    taken = {row["row"] for row in [*decodable, *listed]}
    assert not taken & {1, 2, 7}


def test_phonetizability_is_asked_of_the_assigned_label(pool):
    asked = []
    _select(pool, "confirm.1", phonetizable=lambda label: asked.append(label) or True)
    assert sorted(set(asked)) == ["2:1", "2:2", "55:17"]


@pytest.mark.parametrize("tranche", ["look", "confirm.3", "confirm.4", "train", "confirm", ""])
def test_no_other_tranche_is_selected(pool, tranche):
    with pytest.raises(ValueError, match="confirm.1"):
        _select(pool, tranche)


def test_a_refused_tranche_opens_no_file(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="confirm.1"):
        selection.phase_d_rows("look", tranches_dir=empty, split_dir=empty,
                               phonetizable=_phonetizable)


def test_phase_d_still_needs_the_committed_claim(tmp_path):
    fixture = _claimed(tmp_path, {role: _tranche(role) for role in MINED}, claims=())
    with pytest.raises(ValueError, match="claim"):
        _select(fixture, "confirm.1")


def test_a_row_the_split_tree_gives_to_train_is_refused(pool, tmp_path):
    round_, _ = pool
    victim = _tranche("confirm.1")[2]
    split = _split_tree(tmp_path / "train-split", {role: _tranche(role) for role in MINED},
                        override={(victim["split"], victim["shard_file"], victim["row"]): "train"})
    with pytest.raises(ValueError, match="train"):
        _select((round_, split), "confirm.1")


def test_the_real_trees_give_phase_bs_mined_keys_and_its_26_listed_rows_none_of_phase_as():
    mined_keys = {line["key"] for line in read_jsonl_gz(DECODES / "tlog.mined.base.jsonl.gz")}
    meta = json.loads((DECODES / "meta.json").read_text(encoding="utf-8"))
    decodable_keys: list[str] = []
    listed_keys: list[str] = []
    for tranche in MINED:
        decodable, listed = selection.phase_d_rows(tranche, phonetizable=_phonetizable)
        decodable_keys += _keys(decodable)
        listed_keys += _keys(listed)
    assert len(decodable_keys) == len(set(decodable_keys)) == len(mined_keys) == 10_642
    assert set(decodable_keys) == mined_keys
    assert sorted(listed_keys) == sorted(entry["key"] for entry in meta["unphonetizable"])
    assert len(listed_keys) == 26
    gated = {record["key"] for tranche in MINED
             for record in read_jsonl_gz(WORD_TIMES / f"{tranche}.jsonl.gz")}
    assert not gated & set(decodable_keys)


# --- the worker -------------------------------------------------------------------------------


def _mined_row(number: int, payload: bytes, *, filename_label: str | None) -> dict:
    """A mined row: read against ``2:1`` whatever its filename says."""
    return {"split": "unclean", "shard_file": SHARD, "row": number,
            "audio_path": f"-1_-1_{number}.flac",
            "audio_sha256": hashlib.sha256(payload).hexdigest(),
            "filename_label": filename_label, "assigned_label": "2:1",
            "label_source": "identified", "stratum": "reported:identified"}


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A worklist of two cached mined clips (no filename label; another ayah's), stubbed teacher."""
    parts, clips = tmp_path / "parts", tmp_path / "clips"
    parts.mkdir()
    clips.mkdir()
    rows = []
    for number, filename_label in ((1, None), (2, "9:9")):
        payload = f"mined-{number}".encode()
        rows.append(_mined_row(number, payload, filename_label=filename_label))
        (clips / f"{rows[-1]['audio_sha256']}.audio").write_bytes(payload)
    worklist = parts / "worklist.confirm.1.0.jsonl"
    worklist.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for role in ("confirm.1", "look"):
        (parts / f"unphonetizable.{role}.json").write_text(
            json.dumps({"tranche": role, "labels": []}), encoding="utf-8")
    asked: list[str] = []
    monkeypatch.setattr(cli, "phase_d_rows", lambda role, **kwargs: ([dict(r) for r in rows], []))
    monkeypatch.setattr(cli, "load_decoder", lambda: _identity())
    monkeypatch.setattr(cli, "default_audio", lambda data: np.zeros(FRAMES))
    monkeypatch.setattr(cli, "hafs_word_reference", lambda: (lambda words: (REFERENCE, OFFSETS)))
    monkeypatch.setattr(cli, "_canonical_words",
                        lambda label: asked.append(label) or ["word-1", "word-2"])
    return {"parts": parts, "clips": clips, "worklist": worklist, "rows": rows, "asked": asked,
            "shard_cache": tmp_path / "shards", "tranches": tmp_path / "tranches",
            "split": tmp_path / "split"}


def _align(box, monkeypatch, *extra: str, role: str = "confirm.1") -> int:
    monkeypatch.setattr(cli.sys, "argv", [
        "tlog_mining", "align", "--role", role, "--shard-index", "0", "--shard-count", "1",
        "--parts-dir", str(box["parts"]), "--clip-dir", str(box["clips"]),
        "--shard-cache", str(box["shard_cache"]), "--worklist", str(box["worklist"]),
        "--budget-seconds", "60", "--tranches-dir", str(box["tranches"]),
        "--split-dir", str(box["split"]), "--phase", "D", *extra])
    with pytest.raises(SystemExit) as thrown:
        cli.main()
    return 0 if thrown.value.code in (None, 0) else 1


def _written(box) -> list[dict]:
    path = part_path(box["parts"], SHARD)
    return read_jsonl(path) if path.exists() else []


def test_the_worker_aligns_each_mined_row_against_its_assigned_label(box, monkeypatch):
    assert _align(box, monkeypatch) == 0
    records = _written(box)
    assert [record["key"] for record in records] == [
        item_key(row["audio_path"]) for row in box["rows"]]
    assert box["asked"] == ["2:1", "2:1"]
    assert all(record["surah_ayah"] == "2:1" and record["aligned"] is True
               for record in records)
    assert all(len(record["word_times"]) == 2 and record["ll_per_frame"] < 0
               for record in records)


def test_a_mined_record_has_phase_as_committed_shape(box, monkeypatch):
    assert _align(box, monkeypatch) == 0
    committed = read_jsonl_gz(WORD_TIMES / "look.jsonl.gz")[0]
    for record in _written(box):
        assert set(record) == set(committed)
    row = box["rows"][0]
    first = _written(box)[0]
    assert {name: first[name] for name in ("split", "shard_file", "row", "audio_path",
                                            "audio_sha256")} == {
        name: row[name] for name in ("split", "shard_file", "row", "audio_path", "audio_sha256")}


def test_a_second_batch_writes_nothing(box, monkeypatch):
    assert _align(box, monkeypatch) == 0
    first = part_path(box["parts"], SHARD).read_bytes()
    assert _align(box, monkeypatch) == 0
    assert part_path(box["parts"], SHARD).read_bytes() == first
    assert len(box["asked"]) == 2


def test_a_worklist_row_outside_the_selection_is_refused(box, monkeypatch):
    kept = box["rows"][:1]
    monkeypatch.setattr(cli, "phase_d_rows", lambda role, **kwargs: ([dict(r) for r in kept], []))
    assert _align(box, monkeypatch) == 1
    assert _written(box) == []


def test_the_worker_refuses_look_through_phase_ds_own_selection(box, monkeypatch, capsys):
    """``align`` takes ``look`` as a role for phase A; phase D's selection must refuse it."""
    monkeypatch.undo()
    monkeypatch.setattr(cli, "load_decoder", lambda: pytest.fail("the teacher was loaded"))
    assert _align(box, monkeypatch, role="look") == 1
    assert "confirm.1" in capsys.readouterr().err
    assert _written(box) == []


def test_the_decode_worker_refuses_phase_d(tmp_path):
    args = _decode_args(tmp_path)
    args.phase = "D"
    with pytest.raises(ValueError, match="phase D does not decode"):
        cli.decode(args)


# --- the driver -------------------------------------------------------------------------------


def test_the_phase_d_driver_reads_both_tranches_through_phase_d_selection_only(
        tmp_path, monkeypatch):
    _quiet_box(monkeypatch)
    asked = []
    listed_row = {**_tranche("confirm.1")[7], "shard_file": SHARD_OF["confirm.1"]}

    def rows(tranche, **kwargs):
        asked.append(tranche)
        return ([], [listed_row]) if tranche == "confirm.1" else ([], [])

    def never(*args, **kwargs):
        raise AssertionError("another phase's selection is not phase D's")

    monkeypatch.setattr(cli, "phase_d_rows", rows)
    for name in ("phase_a_rows", "phase_b_rows", "phase_c_rows"):
        monkeypatch.setattr(cli, name, never)
    assert cli.run(_driver_args(tmp_path, "D")) == 0
    assert asked == list(MINED)
    listing = json.loads((tmp_path / "unphonetizable.confirm.1.json").read_text(encoding="utf-8"))
    assert listing["labels"] == ["55:17"]


def test_the_driver_writes_the_worklists_and_runs_one_phase_d_worker(tmp_path, monkeypatch):
    _quiet_box(monkeypatch)
    rows = [{**row, "shard_file": SHARD_OF["confirm.1"]} for row in _tranche("confirm.1")[2:4]]
    monkeypatch.setattr(cli, "phase_d_rows",
                        lambda tranche, **kwargs: (rows if tranche == "confirm.1" else [], []))
    args = _driver_args(tmp_path, "D")
    assert cli.run(args) == 0
    assert args.parallel == 1
    assert cli._worker_count(4, 40_000, "D") == 1
    assert read_jsonl(tmp_path / "worklist.confirm.1.0.jsonl") == rows


def test_the_phase_d_driver_refuses_existing_parts_without_resume(tmp_path, monkeypatch):
    _quiet_box(monkeypatch)
    monkeypatch.setattr(cli, "phase_d_rows", lambda tranche, **kwargs: ([], []))
    part_path(tmp_path, SHARD).write_text("", encoding="utf-8")
    args = _driver_args(tmp_path, "D")
    args.resume = False
    with pytest.raises(SystemExit, match="--resume"):
        cli.run(args)


def test_the_phase_d_batch_reads_phase_as_parts_and_spawns_align_workers(tmp_path, monkeypatch):
    rows = {role: [{**row, "shard_file": SHARD_OF[role]} for row in _tranche(role)[2:4]]
            for role in MINED}
    spawned = []
    monkeypatch.setattr(cli, "_run_workers",
                        lambda role, indexes, budget, args: spawned.append((role, indexes)))
    args = _driver_args(tmp_path, "D")
    args.parallel = 1
    batch = cli._batch(rows, args)
    assert batch(10.0) is False and spawned == [("confirm.1", [0])]
    for role in MINED:
        with part_path(tmp_path, SHARD_OF[role]).open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps({"key": item_key(row["audio_path"])}) + "\n"
                              for row in rows[role])
    assert batch(10.0) is True and spawned == [("confirm.1", [0])]


def test_a_phase_d_worker_is_spawned_as_align_with_the_phase(tmp_path, monkeypatch):
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(cli.subprocess, "Popen", Process)
    cli._run_workers("confirm.1", [0], 5.0, _driver_args(tmp_path, "D"))
    (command,) = commands
    assert command[command.index("-m") + 1:command.index("-m") + 3] == [
        "tadabur.tlog_mining", "align"]
    assert command[command.index("--phase") + 1] == "D"


# --- the merge --------------------------------------------------------------------------------


def _record(row: dict, *, aligned: bool = True) -> dict:
    """Phase A's committed record shape for a mined row (read against its assigned label)."""
    return {"key": item_key(row["audio_path"]), "split": row["split"],
            "shard_file": row["shard_file"], "row": row["row"], "audio_path": row["audio_path"],
            "audio_sha256": row["audio_sha256"], "surah_ayah": row["assigned_label"],
            "aligned": aligned, "reason": None if aligned else "too few frames",
            "word_times": [[0.0, 0.5], [0.5, 1.0]] if aligned else None,
            "seconds_per_frame": 0.04 if aligned else None,
            "ll_per_frame": -0.5 if aligned else None}


def _write_parts(parts: Path, rows, *, unaligned=(), skip=(), extra=(), reverse=False) -> None:
    """One part per shard; ``unaligned`` and ``skip`` are ``(tranche, row)`` pairs, ``extra`` is
    ``(shard, record)`` pairs appended to the part of that shard."""
    parts.mkdir(parents=True, exist_ok=True)
    for tranche, tranche_rows in rows.items():
        records = [_record(row, aligned=(tranche, row["row"]) not in unaligned)
                   for row in tranche_rows if (tranche, row["row"]) not in skip]
        records += [record for shard, record in extra if shard == SHARD_OF[tranche]]
        with part_path(parts, SHARD_OF[tranche]).open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(record) + "\n"
                              for record in (reversed(records) if reverse else records))


PROVENANCE = {"code_sha": "abc", "pull_request": None, "pull_request_ref": None,
              "teacher": {"identity": {"model_ref": "teacher"}, "fingerprint": {"mode": "spans"}}}


def _merge(tmp_path, rows, listed, *, parts="parts", out="out"):
    return merge.merge_mined_word_times(tmp_path / parts, tmp_path / out, rows=rows,
                                  unphonetizable=listed, provenance=PROVENANCE)


def _names(tmp_path, out="out") -> list[str]:
    directory = tmp_path / out
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


def test_the_merge_writes_one_sorted_file_per_tranche_of_the_records_verbatim(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows, unaligned={("confirm.1", 4)})
    _merge(tmp_path, rows, listed)
    assert _names(tmp_path) == sorted(OUT_NAMES)
    for tranche in MINED:
        lines = read_jsonl_gz(tmp_path / "out" / f"mined.{tranche}.jsonl.gz")
        expected = sorted((_record(row, aligned=(tranche, row["row"]) != ("confirm.1", 4))
                           for row in rows[tranche]), key=lambda record: record["key"])
        assert lines == expected


def test_the_meta_reports_counts_label_sources_the_listed_rows_and_the_provenance(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows, unaligned={("confirm.1", 4), ("confirm.2", 3)})
    returned = _merge(tmp_path, rows, listed)
    meta = json.loads((tmp_path / "out" / "meta.mined.json").read_text(encoding="utf-8"))
    assert returned == meta
    assert {name: meta[name] for name in PROVENANCE} == PROVENANCE
    assert meta["tranches"] == {tranche: {"rows": 4, "aligned": 3, "fallback": 1,
                                          "fallback_rate": 0.25} for tranche in MINED}
    assert meta["label_sources"] == dict(Counter(
        row["label_source"] for tranche in MINED for row in rows[tranche]))
    assert sorted((entry["tranche"], entry["key"], entry["assigned_label"],
                   entry["stratum"], entry["audio_filename"]) for entry in meta["unphonetizable"]
                  ) == sorted((tranche, item_key(row["audio_path"]), row["assigned_label"],
                               row["stratum"], row["audio_path"])
                              for tranche in MINED for row in listed[tranche])
    assert [entry["assigned_label"] for entry in meta["unphonetizable"]] == ["55:17", "55:17"]


def test_a_listed_row_is_in_the_meta_and_in_no_file(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    _merge(tmp_path, rows, listed)
    listed_keys = {item_key(row["audio_path"]) for tranche in MINED for row in listed[tranche]}
    pooled = {line["key"] for tranche in MINED
              for line in read_jsonl_gz(tmp_path / "out" / f"mined.{tranche}.jsonl.gz")}
    assert listed_keys and not listed_keys & pooled


@pytest.mark.parametrize("problem", ["missing", "repeated", "listed", "foreign"])
def test_a_partial_or_foreign_part_is_refused_and_nothing_is_written(tmp_path, problem):
    rows, listed = _merge_inputs(tmp_path)
    first, shard = rows["confirm.1"][0], SHARD_OF["confirm.1"]
    stray = {**rows["confirm.2"][0], "audio_path": "stray.flac"}
    extra = {"repeated": [(shard, _record(first))],
             "listed": [(shard, _record(listed["confirm.1"][0]))],
             "foreign": [(shard, _record(stray))]}.get(problem, ())
    skip = {("confirm.2", rows["confirm.2"][1]["row"])} if problem == "missing" else ()
    _write_parts(tmp_path / "parts", rows, skip=skip, extra=extra)
    with pytest.raises(ValueError):
        _merge(tmp_path, rows, listed)
    assert _names(tmp_path) == []


def test_a_record_repeated_in_another_shards_part_is_refused(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    other = part_path(tmp_path / "parts", "clean-00999-of-00411")
    other.write_text(json.dumps(_record(rows["confirm.1"][0])) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _merge(tmp_path, rows, listed)
    assert _names(tmp_path) == []


def test_a_clip_in_both_tranches_is_refused(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    rows["confirm.2"][2] = {**rows["confirm.1"][2], "shard_file": SHARD_OF["confirm.2"]}
    _write_parts(tmp_path / "parts", rows)
    with pytest.raises(ValueError, match="both"):
        _merge(tmp_path, rows, listed)
    assert _names(tmp_path) == []


@pytest.mark.parametrize("tranches", [("confirm.1",), ("confirm.1", "confirm.2", "look")])
def test_the_merge_takes_exactly_the_two_mined_tranches(tmp_path, tranches):
    rows, listed = _merge_inputs(tmp_path)
    rows = {tranche: rows.get(tranche, []) for tranche in tranches}
    with pytest.raises(ValueError, match="confirm.1"):
        merge.merge_mined_word_times(tmp_path / "parts", tmp_path / "out", rows=rows,
                                     unphonetizable=listed, provenance=PROVENANCE)
    assert _names(tmp_path) == []


def _bytes(tmp_path, out="out") -> dict[str, bytes]:
    return {name: (tmp_path / out / name).read_bytes() for name in _names(tmp_path, out)}


def test_a_re_merge_is_byte_identical_in_place_and_whatever_the_part_order(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    _merge(tmp_path, rows, listed)
    first = _bytes(tmp_path)
    assert sorted(first) == sorted(OUT_NAMES)
    _merge(tmp_path, rows, listed)
    assert _bytes(tmp_path) == first
    _write_parts(tmp_path / "reversed", rows, reverse=True)
    _merge(tmp_path, rows, listed, parts="reversed", out="out-reversed")
    assert _bytes(tmp_path, "out-reversed") == first


def test_the_merge_never_writes_phase_as_files(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    out = tmp_path / "out"
    out.mkdir()
    for name in PHASE_A_SHA256:
        (out / name).write_bytes(f"phase A's {name}".encode())
    _merge(tmp_path, rows, listed)
    for name in PHASE_A_SHA256:
        assert (out / name).read_bytes() == f"phase A's {name}".encode()
    assert sorted(set(_names(tmp_path)) - set(PHASE_A_SHA256)) == sorted(OUT_NAMES)


# --- the merge command ------------------------------------------------------------------------


def _merge_args(tmp_path):
    return SimpleNamespace(phase="D", parts_dir=tmp_path / "parts", out_dir=tmp_path / "out",
                           tranches_dir=tmp_path, split_dir=tmp_path, code_sha="abc123",
                           pull_request=None)


def test_the_merge_command_pools_phase_ds_rows_with_the_teacher_record(tmp_path, monkeypatch,
                                                                       capsys):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows, unaligned={("confirm.1", 4)})
    teacher = {"identity": {"hub_revision": BASE_REVISION, "model_ref": BASE_MODEL},
               "fingerprint": {"mode": SPANS}}
    (tmp_path / "parts" / cli.TEACHER_RECORD).write_text(json.dumps(teacher), encoding="utf-8")

    def never(*args, **kwargs):
        raise AssertionError("another phase's selection is not phase D's")

    monkeypatch.setattr(cli, "phase_d_rows",
                        lambda tranche, **kwargs: (rows[tranche], listed[tranche]))
    for name in ("phase_a_rows", "phase_b_rows", "phase_c_rows"):
        monkeypatch.setattr(cli, name, never)
    assert cli.merge(_merge_args(tmp_path)) == 0
    assert _names(tmp_path) == sorted(OUT_NAMES)
    meta = json.loads((tmp_path / "out" / "meta.mined.json").read_text(encoding="utf-8"))
    assert meta["code_sha"] == "abc123" and meta["pull_request"] is None
    assert meta["teacher"] == teacher
    printed = capsys.readouterr().out
    assert "confirm.1: 4 rows, 3 aligned, 1 fallback (25.00%)" in printed
    assert "confirm.2: 4 rows, 4 aligned, 0 fallback (0.00%)" in printed
    assert "label sources:" in printed
    assert "unphonetizable: 2 rows listed (#248)" in printed


def test_the_merge_command_refuses_a_partial_run_and_writes_nothing(tmp_path, monkeypatch):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows, skip={("confirm.2", 3)})
    monkeypatch.setattr(cli, "phase_d_rows",
                        lambda tranche, **kwargs: (rows[tranche], listed[tranche]))
    with pytest.raises(ValueError):
        cli.merge(_merge_args(tmp_path))
    assert _names(tmp_path) == []

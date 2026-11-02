"""Phase A's CLI glue (#263): a worker from a validated worklist, on stubs.

The modules below the CLI are covered by their own tests; this file covers the wiring the box
actually runs — ``python -m tadabur.tlog_mining align`` validating the parent's worklist,
fetching a shard's clips once, calling the teacher, and appending the records. The teacher's
forward pass and the clip decoder are stubbed (no GPU, no audio), the reference is a planted
two-word transcript, and the selection is stubbed to the fixture's rows, so the real worklist
checks, ``build_record``/``align_words`` and the parent's batch bound are exercised end to end.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from tadabur.phoneme_vocab import NUM_PHONEME_CLASSES, PHONEME_CHAR_TO_ID
from tadabur.tlog_mining import __main__ as cli
from tadabur.tlog_mining.give_way import GiveWayResult
from training.ctc_paths import ctc_best_path

SHARD = "clean-00001-of-00411"
OTHER_SHARD = "clean-00002-of-00411"
REFERENCE = "بت"
OFFSETS = [0, 1, 2]
FRAMES = 8


def _posteriors() -> np.ndarray:
    """Two planted phoneme runs in eight frames: ``ب`` at 1-2 and ``ت`` at 4-5."""
    labels = [PHONEME_CHAR_TO_ID[char] for char in REFERENCE]
    probabilities = np.full((FRAMES, NUM_PHONEME_CLASSES), 0.1 / (NUM_PHONEME_CLASSES - 1))
    probabilities[:, 0] = 0.9
    for label, (first, last) in zip(labels, [(1, 2), (4, 5)]):
        probabilities[first:last + 1] = 0.1 / (NUM_PHONEME_CLASSES - 1)
        probabilities[first:last + 1, label] = 0.9
    return np.log(probabilities / probabilities.sum(axis=1, keepdims=True))


class FakeDecoder:
    """The teacher's whole-clip forward, planted."""

    def span_log_posteriors(self, spans):
        return [_posteriors() for _ in spans]


def _row(number: int, payload: bytes, *, shard: str = SHARD) -> dict:
    return {"split": "clean", "shard_file": shard, "row": number,
            "audio_path": f"2_1_{number}.flac",
            "audio_sha256": hashlib.sha256(payload).hexdigest(),
            "filename_label": "2:1", "stratum": "gating"}


def _identity(fingerprint: dict | None = None) -> tuple:
    return FakeDecoder(), {"model_ref": "teacher"}, fingerprint or {"store": "spans"}


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A worklist of two cached clips, a stubbed teacher, reference and selection."""
    parts, clips = tmp_path / "parts", tmp_path / "clips"
    parts.mkdir()
    clips.mkdir()
    rows = []
    for number in (1, 2):
        payload = f"clip-{number}".encode()
        row = _row(number, payload)
        (clips / f"{row['audio_sha256']}.audio").write_bytes(payload)
        rows.append(row)
    worklist = parts / "worklist.look.0.jsonl"
    worklist.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(cli, "phase_a_rows", lambda role, **kwargs: [dict(row) for row in rows])
    monkeypatch.setattr(cli, "load_decoder", lambda: _identity())
    monkeypatch.setattr(cli, "default_audio", lambda data: np.zeros(FRAMES))
    monkeypatch.setattr(cli, "hafs_word_reference", lambda: (lambda words: (REFERENCE, OFFSETS)))
    monkeypatch.setattr(cli, "_canonical_words", lambda label: ["word-1", "word-2"])
    return {"parts": parts, "clips": clips, "worklist": worklist, "rows": rows,
            "shard_cache": tmp_path / "shards", "tranches": tmp_path / "tranches",
            "split": tmp_path / "split"}


def _align(box, monkeypatch, *extra: str, shard_index: int = 0, shard_count: int = 1) -> int:
    """Run the CLI's ``align``; returns 0 on success and 1 on a refusal."""
    monkeypatch.setattr(cli.sys, "argv", [
        "tlog_mining", "align", "--role", "look", "--shard-index", str(shard_index),
        "--shard-count", str(shard_count), "--parts-dir", str(box["parts"]),
        "--clip-dir", str(box["clips"]), "--shard-cache", str(box["shard_cache"]),
        "--worklist", str(box["worklist"]), "--budget-seconds", "60",
        "--tranches-dir", str(box["tranches"]), "--split-dir", str(box["split"]), *extra])
    with pytest.raises(SystemExit) as thrown:
        cli.main()
    code = thrown.value.code
    return 0 if code is None or code == 0 else 1


def _part_path(box) -> Path:
    return box["parts"] / f"word_times.{SHARD}.jsonl"


def _part(box) -> list[dict]:
    path = _part_path(box)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_worker_aligns_its_worklist_into_the_part(box, monkeypatch):
    assert _align(box, monkeypatch) == 0
    records = _part(box)
    assert [record["key"] for record in records] == [
        f"{row['audio_path']}#0" for row in box["rows"]]
    assert all(record["aligned"] is True for record in records)
    assert all(len(record["word_times"]) == 2 for record in records)
    assert records[0]["ll_per_frame"] < 0
    # The planted path is exact: word 0 is frames 1-2, word 1 frames 4-5, half-open at the end.
    path = ctc_best_path(_posteriors(), [PHONEME_CHAR_TO_ID[char] for char in REFERENCE])
    seconds = (FRAMES / 16000) / FRAMES
    assert records[0]["word_times"][0] == pytest.approx([path[0][0] * seconds,
                                                         (path[0][1] + 1) * seconds])
    assert records[0]["word_times"][1] == pytest.approx([path[1][0] * seconds,
                                                         (path[1][1] + 1) * seconds])
    assert _part_path(box).read_text().count("\n") == 2


def test_a_second_batch_writes_nothing_and_leaves_the_part_alone(box, monkeypatch):
    assert _align(box, monkeypatch) == 0
    before = _part_path(box).read_text(encoding="utf-8")
    assert _align(box, monkeypatch) == 0
    assert _part_path(box).read_text(encoding="utf-8") == before


# --- the worklist is not an authorization ---------------------------------------------------------


def test_a_worklist_row_outside_the_selection_is_refused(box, monkeypatch, capsys):
    foreign = {**_row(9, b"foreign"), "stratum": "reported:unclean"}
    box["worklist"].write_text(json.dumps(foreign) + "\n", encoding="utf-8")
    assert _align(box, monkeypatch) == 1
    assert "not a gating row" in capsys.readouterr().err
    assert _part(box) == []


def test_a_worklist_row_that_is_not_the_manifest_row_is_refused(box, monkeypatch, capsys):
    altered = {**box["rows"][0], "audio_sha256": "00" * 32}
    box["worklist"].write_text(json.dumps(altered) + "\n", encoding="utf-8")
    assert _align(box, monkeypatch) == 1
    assert "not the committed manifest row" in capsys.readouterr().err
    assert _part(box) == []


def test_a_worklist_row_twice_is_refused(box, monkeypatch, capsys):
    row = json.dumps(box["rows"][0])
    box["worklist"].write_text(row + "\n" + row + "\n", encoding="utf-8")
    assert _align(box, monkeypatch) == 1
    assert "appears twice" in capsys.readouterr().err


def test_a_worklist_row_of_another_workers_share_is_refused(box, monkeypatch, capsys):
    """Worker 0 of 2 owns one of the two shards; a row of the other is not its work."""
    first = box["rows"][0]
    other = _row(3, b"clip-3", shard=OTHER_SHARD)
    monkeypatch.setattr(cli, "phase_a_rows", lambda role, **kwargs: [dict(first), dict(other)])
    box["worklist"].write_text(json.dumps(other) + "\n", encoding="utf-8")
    assert _align(box, monkeypatch, shard_count=2) == 1
    assert "not a gating row" in capsys.readouterr().err


def test_a_sealed_tranche_is_not_a_role_the_worker_accepts(box, monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", [
        "tlog_mining", "align", "--role", "confirm.3", "--shard-index", "0", "--shard-count", "1",
        "--parts-dir", str(box["parts"]), "--clip-dir", str(box["clips"]),
        "--shard-cache", str(box["shard_cache"])])
    with pytest.raises(SystemExit) as thrown:
        cli.main()
    assert thrown.value.code != 0
    assert _part(box) == []


# --- the phase ------------------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["B", "C"])
def test_a_phase_other_than_a_is_refused_with_a_message(box, monkeypatch, capsys, phase):
    """Phases B and C are not runnable here: refused before the teacher, with no record."""
    assert _align(box, monkeypatch, "--phase", phase) == 1
    assert f"phase {phase}" in capsys.readouterr().err
    assert _part(box) == []


# --- the teacher record binds a resumed run -------------------------------------------------------


def test_a_changed_teacher_fingerprint_is_refused_before_a_row(box, monkeypatch, capsys):
    assert _align(box, monkeypatch) == 0
    before = _part_path(box).read_text(encoding="utf-8")
    monkeypatch.setattr(cli, "load_decoder",
                        lambda: _identity({"store": "spans", "dtype": "fp32"}))
    assert _align(box, monkeypatch) == 1
    assert "records other weights or settings" in capsys.readouterr().err
    assert _part_path(box).read_text(encoding="utf-8") == before


# --- the shard, and the batch's bound -------------------------------------------------------------


def test_a_shard_is_streamed_once_for_all_its_missing_clips(tmp_path, monkeypatch):
    """A per-row fetch would re-download the whole shard per clip (the blob is deleted on read)."""
    calls = []
    payloads = {number: f"clip-{number}".encode() for number in (1, 2, 3)}
    rows = [_row(number, payload) for number, payload in payloads.items()]

    def cache_shard(shard_file, wanted, clip_dir, *, stream):
        calls.append((shard_file, [row["row"] for row in wanted]))
        for row in wanted:
            Path(clip_dir, f"{row['audio_sha256']}.audio").write_bytes(payloads[row["row"]])

    monkeypatch.setattr(cli.ted, "cache_shard", cache_shard)
    args = Namespace(clip_dir=tmp_path / "clips", look_clip_cache=None,
                     shard_cache=tmp_path / "shards")
    args.clip_dir.mkdir()
    cli._fetch_shard(SHARD, rows, args)
    assert calls == [(SHARD, [1, 2, 3])]


def _worker_args(tmp_path) -> Namespace:
    return Namespace(parts_dir=tmp_path, clip_dir=tmp_path, shard_cache=tmp_path,
                     tranches_dir=tmp_path, split_dir=tmp_path, parallel=2,
                     look_clip_cache=None, phase="A")


def test_the_parent_stops_a_worker_that_overruns_the_batch(tmp_path, monkeypatch):
    """The worker's own budget is per row; the parent bounds the whole batch (and its startup)."""
    monkeypatch.setattr(cli, "BATCH_GRACE_SECONDS", 0.5)
    spawn = subprocess.Popen
    monkeypatch.setattr(cli.subprocess, "Popen",
                        lambda *a, **k: spawn([sys.executable, "-c",
                                               "import time; time.sleep(30)"]))
    started = time.monotonic()
    cli._run_workers("look", [0], 0.1, _worker_args(tmp_path))
    assert time.monotonic() - started < 10


# --- the card sizes the workers -------------------------------------------------------------------


@pytest.mark.parametrize("requested, free_mib, expected", [
    (2, 15_852, 1),   # the box's 15.5 GiB card with nothing else running: two do not fit
    (2, 19_000, 2),
    (2, 40_000, 2),
    (3, 40_000, 3),
    (1, 100, 1),      # a card below one worker's budget still runs one
])
def test_the_worker_count_is_what_the_free_memory_holds(requested, free_mib, expected):
    assert cli._worker_count(requested, free_mib) == expected


def test_the_driver_caps_parallel_before_writing_the_worklists(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "gpu_memory_mib", lambda field: 15_852)
    monkeypatch.setattr(cli, "phase_a_rows", lambda role, **kwargs: [])
    monkeypatch.setattr(cli, "_available_ram_gb", lambda: 64.0)
    monkeypatch.setattr(cli, "run_give_way",
                        lambda batch, **kwargs: GiveWayResult("done", 0, 0.0, 0.0))
    args = Namespace(phase="A", resume=True, parallel=2, parts_dir=tmp_path,
                     clip_dir=tmp_path, shard_cache=tmp_path, tranches_dir=tmp_path,
                     split_dir=tmp_path, pause_file=tmp_path / "PAUSE",
                     lock=tmp_path / "gpu.lock", batch_seconds=300, yield_seconds=30,
                     disk_reserve_gb=0.0, ram_floor_gb=0.0, look_clip_cache=None)
    assert cli.run(args) == 0
    assert args.parallel == 1
    assert sorted(path.name for path in tmp_path.glob("worklist.*.jsonl")) == sorted(
        f"worklist.{role}.0.jsonl" for role in cli.PHASE_A_TRANCHES)


def test_a_spawn_failure_reaps_the_worker_already_started(tmp_path, monkeypatch):
    """The caller releases the GPU lock next; no child may outlive the failed spawn."""
    monkeypatch.setattr(cli, "BATCH_GRACE_SECONDS", 0.5)
    spawn = subprocess.Popen
    children = []

    def fake(*args, **kwargs):
        if children:
            raise OSError("cannot fork")
        children.append(spawn([sys.executable, "-c", "import time; time.sleep(30)"]))
        return children[0]

    monkeypatch.setattr(cli.subprocess, "Popen", fake)
    with pytest.raises(OSError, match="cannot fork"):
        cli._run_workers("look", [0, 1], 60, _worker_args(tmp_path))
    assert children[0].poll() is not None

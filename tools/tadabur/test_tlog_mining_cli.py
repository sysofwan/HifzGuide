"""Phase A's CLI glue (#263): one worker's batch from a written worklist, on stubs.

The modules below the CLI are covered by their own tests; this file covers the wiring the box
actually runs — ``python -m tadabur.tlog_mining align`` reading the parent's worklist, fetching
a shard's clips once, calling the teacher, and appending the records. The teacher's forward pass
and the clip decoder are stubbed (no GPU, no audio), and the reference is a planted two-word
transcript, so the real ``build_record``/``align_words`` path is exercised end to end.
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
from training.ctc_paths import ctc_best_path

SHARD = "clean-00001-of-00411"
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


def _row(number: int, payload: bytes) -> dict:
    return {"split": "clean", "shard_file": SHARD, "row": number,
            "audio_path": f"2_1_{number}.flac", "audio_sha256": hashlib.sha256(payload).hexdigest(),
            "filename_label": "2:1", "stratum": "gating"}


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A worklist of two cached clips, a stubbed teacher and a planted reference."""
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
    monkeypatch.setattr(cli, "load_decoder",
                        lambda: (FakeDecoder(), {"model_ref": "teacher"},
                                 {"store": "spans"}))
    monkeypatch.setattr(cli, "default_audio", lambda data: np.zeros(FRAMES))
    monkeypatch.setattr(cli, "hafs_word_reference", lambda: (lambda words: (REFERENCE, OFFSETS)))
    monkeypatch.setattr(cli, "_canonical_words", lambda label: ["word-1", "word-2"])
    return {"parts": parts, "clips": clips, "worklist": worklist, "rows": rows}


def _align(box, tmp_path, monkeypatch, *extra: str) -> int:
    """Run the CLI's ``align``; returns its exit code."""
    monkeypatch.setattr(cli.sys, "argv", [
        "tlog_mining", "align", "--role", "look", "--shard-index", "0", "--shard-count", "1",
        "--parts-dir", str(box["parts"]), "--clip-dir", str(box["clips"]),
        "--shard-cache", str(tmp_path / "shards"), "--worklist", str(box["worklist"]),
        "--budget-seconds", "60", *extra])
    with pytest.raises(SystemExit) as thrown:
        cli.main()
    code = thrown.value.code
    return 0 if code is None or code == 0 else 1


def _part(box) -> list[dict]:
    path = box["parts"] / f"word_times.{SHARD}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_worker_aligns_its_worklist_into_the_part(box, tmp_path, monkeypatch):
    assert _align(box, tmp_path, monkeypatch) == 0
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
    assert (box["parts"] / f"word_times.{SHARD}.jsonl").read_text().count("\n") == 2


def test_a_second_batch_writes_nothing_and_leaves_the_part_alone(box, tmp_path, monkeypatch):
    assert _align(box, tmp_path, monkeypatch) == 0
    before = (box["parts"] / f"word_times.{SHARD}.jsonl").read_text(encoding="utf-8")
    assert _align(box, tmp_path, monkeypatch) == 0
    assert (box["parts"] / f"word_times.{SHARD}.jsonl").read_text(encoding="utf-8") == before


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


def test_the_parent_stops_a_worker_that_overruns_the_batch(tmp_path, monkeypatch):
    """The worker's own budget is per row; the parent bounds the whole batch (and its startup)."""
    monkeypatch.setattr(cli, "OVERDUE_SECONDS", 0.5)
    spawn = subprocess.Popen
    monkeypatch.setattr(cli.subprocess, "Popen",
                        lambda *a, **k: spawn([sys.executable, "-c",
                                               "import time; time.sleep(30)"]))
    args = Namespace(parts_dir=tmp_path, clip_dir=tmp_path, shard_cache=tmp_path,
                     tranches_dir=tmp_path, split_dir=tmp_path, parallel=1,
                     look_clip_cache=None, phase="A")
    started = time.monotonic()
    cli._run_workers("look", [0], 0.1, args)
    assert time.monotonic() - started < 10


def test_a_phase_other_than_a_is_refused_with_a_message(box, tmp_path, monkeypatch, capsys):
    assert _align(box, tmp_path, monkeypatch, "--phase", "B") == 1
    assert "phase B" in capsys.readouterr().err
    assert _part(box) == []

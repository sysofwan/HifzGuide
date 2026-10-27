"""The staged listening clips (``tadabur.tlog_audit.stage``).

The page serves ``local_audio`` and the worklist row carries the clip's recorded
``audio_sha256`` as its provenance, so staging verifies each streamed clip against that
checksum before writing the 16 kHz WAV and the worklist, and refuses a row whose bytes are not
the recorded clip.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
import soundfile as sf

from . import tlog_audit as run


def _wav(seconds: float = 0.5) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(16_000 * seconds), dtype=np.float32), 16_000,
             format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _row(data: bytes) -> dict:
    return {"population": "look", "role": "look", "split": "clean",
            "shard_file": "clean-00000-of-00411", "row": 1,
            "audio_path": "audio/clean-00000-of-00411.parquet",
            "audio_sha256": hashlib.sha256(data).hexdigest(), "duration_s": 12.5,
            "filename_label": "2:1", "assigned_label": "2:1", "class_label": "2:1",
            "label_source": "both", "id_floor_ok": True, "overlaps_filename": False,
            "blocking_reason": None, "strata": ("both",), "stratum": "both",
            "floor_sample": False}


def _stage(tmp_path, monkeypatch, stream: bytes, row: dict, *, previous: dict | None = None):
    """Stage one drawn row from a mocked shard whose second item is ``stream``."""
    monkeypatch.setattr(run, "shard_rows", lambda shard, cache: iter(
        [{"audio": {"bytes": b"another clip"}}, {"audio": {"bytes": stream}}]))
    worklist_dir = tmp_path / "audit"
    worklist_dir.mkdir()
    if previous is not None:
        run.write_jsonl([previous], worklist_dir / "worklist.jsonl", sort_keys=True)
    monkeypatch.setattr(run, "AUDIT_DIR", worklist_dir)
    audio_dir = tmp_path / "audio"
    run.stage(run.AuditDraw(items=(row,), strata={}), audio_dir=audio_dir,
              cache_dir=tmp_path / "cache")
    return audio_dir, worklist_dir


def test_stage_verifies_the_checksum_and_stages_the_drawn_clip(tmp_path, monkeypatch):
    data = _wav()
    row = _row(data)
    audio_dir, worklist_dir = _stage(tmp_path, monkeypatch, data, row)

    worklist = run.read_jsonl(worklist_dir / "worklist.jsonl")
    assert len(worklist) == 1
    item = worklist[0]
    assert (item["split"], item["shard_file"], item["row"]) == (
        row["split"], row["shard_file"], row["row"])
    assert item["audio_sha256"] == row["audio_sha256"]
    assert item["strata"] == ["both"]
    assert item["local_audio"].endswith(f"_{row['audio_sha256'][:8]}.wav")

    staged = audio_dir / item["local_audio"]
    samples, rate = sf.read(str(staged))
    assert rate == 16_000
    assert samples.shape == (8_000,)


def test_stage_deletes_the_previous_draws_orphan_audio(tmp_path, monkeypatch):
    data = _wav()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    orphan = audio_dir / "0000000000000001_00000000.wav"
    orphan.write_bytes(b"an earlier draw's clip")
    staged_dir, _ = _stage(tmp_path, monkeypatch, data, _row(data),
                           previous={"local_audio": orphan.name})

    assert not orphan.exists()
    assert len(list(staged_dir.glob("*.wav"))) == 1


def test_stage_refuses_bytes_that_are_not_the_recorded_clip(tmp_path, monkeypatch):
    row = _row(_wav())
    with pytest.raises(ValueError, match=f"not {row['audio_sha256']}"):
        _stage(tmp_path, monkeypatch, _wav(0.25), row)
    assert not (tmp_path / "audit" / "worklist.jsonl").exists()
    assert not list((tmp_path / "audio").glob("*.wav"))

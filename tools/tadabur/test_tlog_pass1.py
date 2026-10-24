"""Acceptance tests for the tlog pass-1 tracer (``tadabur.tlog_pass1``, #187).

The pass streams a tlog shard (download -> measure -> write -> delete the shard) and commits,
per row, only outcome-blind facts: ``audio_sha256``, ``codec_ok``, ``sample_rate``,
``duration_s``, ``speech_s`` and ``speech_ratio``. Fingerprints go to cuda-dev scratch, never
the committed rows; no embedding, teacher, transcript or label is computed. The silence floors
are the 0.5th percentiles of ``speech_ratio`` and ``speech_s`` over the Tarteel v1 look clips.

Everything here runs offline: the fake shard is a tiny parquet, VAD is a callable, and no GPU,
network or Silero import is involved.

Test command, from the repo root::

    python -m pytest -q tools/tadabur/test_tlog_pass1.py
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from tadabur.nonpro_sources import TARTEEL_V1, TLOG, tlog_split

#: Every fact a committed row carries, in no particular order.
FACT_KEYS = frozenset(
    {"audio_sha256", "codec_ok", "sample_rate", "duration_s", "speech_s", "speech_ratio"})
#: The identity every committed row carries beside its facts.
IDENTITY_KEYS = frozenset({"schema_version", "split", "shard_file", "row", "audio_path"})
#: The tracer's fixed rule, written before the run: clean shards 0 and 205 of 411, unclean 0.
TRACER_SHARDS = (
    "data/clean-00000-of-00411.parquet",
    "data/clean-00205-of-00411.parquet",
    "data/unclean-00000-of-00332.parquet",
)
CLEAN_SHARD = TRACER_SHARDS[0]
UNCLEAN_SHARD = TRACER_SHARDS[2]


@pytest.fixture
def tp():
    """The module under test, imported here so each test fails on its own criterion."""
    from tadabur import tlog_pass1

    return tlog_pass1


def wav(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """A mono PCM_16 WAV of silence, the bytes a shard row would carry."""
    buffer = io.BytesIO()
    frames = np.zeros(round(seconds * sample_rate), dtype=np.float32)
    sf.write(buffer, frames, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def empty_wav(sample_rate: int = 16000) -> bytes:
    """A valid WAV container with zero frames: the pre-registration's ``duration == 0``."""
    buffer = io.BytesIO()
    with sf.SoundFile(buffer, "w", samplerate=sample_rate, channels=1, format="WAV",
                      subtype="PCM_16"):
        pass
    return buffer.getvalue()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_stream(rows: list[dict]):
    """A ``shard_rows`` stand-in: the same rows for any shard, in order."""
    def stream(shard: str, cache_dir: Path):
        assert shard in TLOG.shards
        yield from rows

    return stream


# --- the row facts ---------------------------------------------------------------------------


def test_a_decodable_row_carries_the_stored_bytes_facts(tp):
    audio = wav(seconds=1.0, sample_rate=16000)
    seen: list[np.ndarray] = []

    def speech_seconds_of(waveform: np.ndarray) -> float:
        seen.append(waveform)
        return 0.5

    measured = tp.measure(audio, speech_seconds_of)

    assert set(measured.facts) == FACT_KEYS
    assert measured.facts == {
        "audio_sha256": sha(audio),
        "codec_ok": True,
        "sample_rate": 16000,
        "duration_s": 1.0,
        "speech_s": 0.5,
        "speech_ratio": 0.5,
    }
    assert len(seen) == 1 and seen[0].dtype == np.float32
    assert len(seen[0]) == 16000
    assert measured.waveform is not None and len(measured.waveform) == 16000


def test_a_clip_at_another_rate_records_its_source_rate_and_16k_duration(tp):
    audio = wav(seconds=1.0, sample_rate=8000)
    seen: list[np.ndarray] = []

    def speech_seconds_of(waveform: np.ndarray) -> float:
        seen.append(waveform)
        return 0.25

    measured = tp.measure(audio, speech_seconds_of)

    assert measured.facts["sample_rate"] == 8000
    assert measured.facts["duration_s"] == 1.0
    assert measured.facts["speech_s"] == 0.25
    assert measured.facts["speech_ratio"] == 0.25
    # VAD and the fingerprint see the 16 kHz mono signal, whatever the source rate was.
    assert len(seen[0]) == 16000


def test_an_undecodable_row_is_recorded_and_the_vad_is_not_called(tp):
    audio = b"not audio at all"
    called = []

    def speech_seconds_of(waveform):
        called.append(waveform)
        return 1.0

    measured = tp.measure(audio, speech_seconds_of)

    assert measured.facts == {
        "audio_sha256": sha(audio),
        "codec_ok": False,
        "sample_rate": None,
        "duration_s": None,
        "speech_s": None,
        "speech_ratio": None,
    }
    assert measured.waveform is None
    assert called == []


def test_a_zero_frame_row_is_a_codec_failure(tp):
    audio = empty_wav()
    called = []

    measured = tp.measure(audio, lambda waveform: called.append(waveform) or 0.0)

    assert measured.facts["codec_ok"] is False
    assert measured.facts["duration_s"] is None
    assert measured.facts["speech_ratio"] is None
    assert measured.waveform is None
    assert called == []


# --- the shard loop --------------------------------------------------------------------------


def test_process_shard_writes_one_row_per_shard_row_and_nothing_else(tp, tmp_path):
    rows = [
        {"audio": {"path": "100_10_1086409507.flac", "bytes": wav(1.0, 16000)}},
        {"audio": {"path": "101_11_2000000000.flac", "bytes": wav(0.5, 8000)}},
    ]
    records = tp.process_shard(CLEAN_SHARD, tmp_path / "cache",
                               speech_seconds_of=lambda waveform: 0.25,
                               stream=fake_stream(rows))

    assert len(records) == 2
    assert [record["row"] for record in records] == [0, 1]
    assert {record["split"] for record in records} == {"clean"}
    assert {record["shard_file"] for record in records} == {"clean-00000-of-00411"}
    assert [record["audio_path"] for record in records] == [
        "100_10_1086409507.flac", "101_11_2000000000.flac"]
    for record in records:
        assert set(record) == FACT_KEYS | IDENTITY_KEYS
        # Outcome-blind: no label, transcript, embedding or fingerprint in the committed row.
        assert set(record).isdisjoint(
            {"label", "filename_label", "transcript", "embedding", "fingerprint",
             "qud_fingerprint"})
    assert records[0]["speech_ratio"] == 0.25
    assert records[1]["duration_s"] == 0.5


def test_process_shard_refuses_a_shard_that_is_not_tlog(tp, tmp_path):
    bogus = "data/train-00000-of-00008.parquet"
    with pytest.raises(ValueError):
        tp.process_shard(bogus, tmp_path / "cache",
                         speech_seconds_of=lambda waveform: 0.0,
                         stream=fake_stream([]))


def test_the_fingerprint_is_written_only_to_the_scratch_directory(tp, tmp_path):
    path = "100_10_1086409507.flac"
    audio = wav(1.0, 16000)
    fp_dir = tmp_path / "fp"
    records = tp.process_shard(CLEAN_SHARD, tmp_path / "cache",
                               speech_seconds_of=lambda waveform: 0.0,
                               fp_dir=fp_dir,
                               stream=fake_stream([{"audio": {"path": path, "bytes": audio}}]))

    assert "fingerprint" not in records[0] and "qud_fingerprint" not in records[0]
    stored = np.load(fp_dir / "clean-00000-of-00411.npz")
    assert stored.files == [f"{path}|{sha(audio)}"]
    assert len(stored[stored.files[0]]) > 0


def test_shard_rows_deletes_the_downloaded_shard(tp, tmp_path, monkeypatch):
    import huggingface_hub

    parquet = tmp_path / "cached.parquet"
    table = pa.table({"audio": pa.array(
        [{"path": "100_10_1086409507.flac", "bytes": b"stored"}],
        type=pa.struct([("path", pa.string()), ("bytes", pa.binary())]))})
    pq.write_table(table, parquet)
    calls: list[tuple[str, str, object]] = []

    def fake_download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs.get("revision")))
        return str(parquet)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    rows = list(tp.shard_rows(CLEAN_SHARD, tmp_path / "cache"))

    assert [row["audio"]["path"] for row in rows] == ["100_10_1086409507.flac"]
    assert calls == [(TLOG.repo, CLEAN_SHARD, TLOG.revision)]
    assert not parquet.exists()


def test_shard_rows_deletes_the_shard_when_the_reader_stops_early(tp, tmp_path, monkeypatch):
    import huggingface_hub

    parquet = tmp_path / "cached.parquet"
    table = pa.table({"audio": pa.array(
        [{"path": f"{index}.flac", "bytes": b"stored"} for index in range(3)],
        type=pa.struct([("path", pa.string()), ("bytes", pa.binary())]))})
    pq.write_table(table, parquet)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, filename, **kwargs: str(parquet))

    rows = tp.shard_rows(CLEAN_SHARD, tmp_path / "cache")
    assert next(rows)["audio"]["path"] == "0.flac"
    rows.close()

    assert not parquet.exists()


# --- the committed output --------------------------------------------------------------------


def test_a_pass1_file_written_twice_is_the_same_bytes(tp, tmp_path):
    rows = tp.process_shard(UNCLEAN_SHARD, tmp_path / "cache",
                            speech_seconds_of=lambda waveform: 0.1,
                            stream=fake_stream([
                                {"audio": {"path": "a.wav", "bytes": wav(1.0, 48000)}},
                                {"audio": {"path": "b.wav", "bytes": b"broken"}}]))
    first, second = tmp_path / "first.jsonl.gz", tmp_path / "second.jsonl.gz"
    tp.write_pass1(rows, first)
    tp.write_pass1(rows, second)

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle] == rows


def test_run_writes_the_shard_and_the_meta(tp, tmp_path):
    shard = CLEAN_SHARD
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")
    rows = [{"audio": {"path": "a.flac", "bytes": wav(1.0, 16000)}},
            {"audio": {"path": "b.flac", "bytes": b"broken"}}]

    meta = tp.run([shard], out_dir=tmp_path / "pass1", cache_dir=tmp_path / "cache",
                  floors_path=floors, speech_seconds_of=lambda waveform: 0.5,
                  stream=fake_stream(rows), code_sha="0" * 40)

    output = tmp_path / "pass1" / "clean" / "clean-00000-of-00411.jsonl.gz"
    assert output.exists()
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        assert len([line for line in handle if line.strip()]) == 2
    assert meta["schema_version"] == tp.SCHEMA_VERSION
    assert meta["code_sha"] == "0" * 40
    assert meta["vad"] == tp.VAD_CONFIG
    assert meta["embeddings"] == "none (owner fallback 2026-10-24, #187)"
    assert meta["counts"]["rows"] == {"clean": 2, "unclean": 0, "total": 2}
    assert "throughput" in meta


def test_run_writes_meta_to_disk_and_a_fingerprint_store_when_asked(tp, tmp_path):
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")
    audio = wav(1.0, 16000)
    rows = [{"audio": {"path": "a.flac", "bytes": audio}}]
    out_dir, fp_dir = tmp_path / "pass1", tmp_path / "fp"

    meta = tp.run([CLEAN_SHARD], out_dir=out_dir, cache_dir=tmp_path / "cache",
                  floors_path=floors, speech_seconds_of=lambda waveform: 0.5,
                  stream=fake_stream(rows), fp_dir=fp_dir, code_sha="0" * 40)

    assert json.loads((out_dir / "meta.json").read_text(encoding="utf-8")) == meta
    stored = np.load(fp_dir / "clean-00000-of-00411.npz")
    assert stored.files == [f"a.flac|{sha(audio)}"]
    assert len(stored[stored.files[0]]) > 0


def test_run_tolerates_a_shard_with_no_fingerprintable_row(tp, tmp_path):
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")
    fp_dir = tmp_path / "fp"

    meta = tp.run([CLEAN_SHARD], out_dir=tmp_path / "pass1", cache_dir=tmp_path / "cache",
                  floors_path=floors, speech_seconds_of=lambda waveform: 0.0,
                  stream=fake_stream([{"audio": {"path": "a.flac", "bytes": b"broken"}}]),
                  fp_dir=fp_dir, code_sha="0" * 40)

    assert meta["counts"]["rows"]["total"] == 1
    assert not fp_dir.exists()


def test_run_refuses_a_vad_config_that_differs_from_the_floors(tp, tmp_path):
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": {**tp.VAD_CONFIG, "threshold": 0.4}}),
                      encoding="utf-8")

    with pytest.raises(ValueError, match="VAD"):
        tp.run([CLEAN_SHARD], out_dir=tmp_path / "pass1", cache_dir=tmp_path / "cache",
               floors_path=floors, speech_seconds_of=lambda waveform: 0.0,
               stream=fake_stream([]), code_sha="0" * 40)


# --- the silence floors ----------------------------------------------------------------------


def test_the_floors_are_the_type7_half_percentile_over_decodable_look_rows(tp):
    rows = [
        {"codec_ok": True, "duration_s": 1.0, "speech_s": 1.0, "speech_ratio": 0.1},
        {"codec_ok": True, "duration_s": 1.0, "speech_s": 2.0, "speech_ratio": 0.2},
        {"codec_ok": True, "duration_s": 1.0, "speech_s": 3.0, "speech_ratio": 0.3},
        {"codec_ok": True, "duration_s": 1.0, "speech_s": 4.0, "speech_ratio": 0.4},
        # Excluded: the pre-registration's floor population is codec_ok and duration > 0.
        {"codec_ok": False, "duration_s": None, "speech_s": None, "speech_ratio": None},
    ]

    floors = tp.silence_floors(rows, look_clips=5, decodes_sha256="ab" * 32)

    # Type-7 at q = 0.005 over four values interpolates 1.5% toward the second value.
    assert floors["f_s"] == 0.1015
    assert floors["d_s"] == 1.015
    assert floors["count"] == 4
    assert floors["source"] == {
        "dataset": "tarteel_v1", "split": "look", "clips": 5,
        "decodes_sha256": "ab" * 32}
    assert floors["percentile"] == {"method": "type-7", "q": 0.005}
    assert floors["vad"] == tp.VAD_CONFIG


def test_build_floors_measures_only_the_look_clips_and_deletes_the_shard(
        tp, tmp_path, monkeypatch):
    import huggingface_hub

    shard = TARTEEL_V1.shards[0]
    look, other = "100_10_1111111111.wav", "100_10_2222222222.wav"
    parquet = tmp_path / "tarteel.parquet"
    table = pa.table({"audio": pa.array(
        [{"path": look, "bytes": wav(1.0, 16000)},
         {"path": other, "bytes": wav(1.0, 16000)}],
        type=pa.struct([("path", pa.string()), ("bytes", pa.binary())]))})
    pq.write_table(table, parquet)
    calls: list[tuple[str, str, object]] = []

    def fake_download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs.get("revision")))
        return str(parquet)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    seen: list[np.ndarray] = []

    floors = tp.build_floors(
        [shard], look_ids={look}, cache_dir=tmp_path / "cache",
        speech_seconds_of=lambda waveform: seen.append(waveform) or 0.5)

    assert calls == [(TARTEEL_V1.repo, shard, TARTEEL_V1.revision)]
    assert len(seen) == 1  # the look clip, never the other row
    assert floors["count"] == 1
    assert floors["f_s"] == 0.5
    assert floors["d_s"] == 0.5
    assert floors["source"]["dataset"] == "tarteel_v1"
    assert floors["source"]["split"] == "look"
    assert floors["source"]["clips"] == 1
    assert len(floors["source"]["decodes_sha256"]) == 64
    assert floors["vad"] == tp.VAD_CONFIG
    assert not parquet.exists()


def test_the_floor_config_must_match_the_floors_file(tp):
    floors = {"vad": tp.VAD_CONFIG}
    assert tp.require_floor_config(floors) is None
    with pytest.raises(ValueError, match="VAD"):
        tp.require_floor_config({"vad": {**tp.VAD_CONFIG, "min_silence_duration_ms": 200}})


# --- the tracer rule -------------------------------------------------------------------------


def test_the_tracer_rule_is_two_clean_shards_and_one_unclean(tp):
    chosen = tp.tracer_shards()
    assert chosen == TRACER_SHARDS
    assert tuple(tlog_split(shard) for shard in chosen) == ("clean", "clean", "unclean")

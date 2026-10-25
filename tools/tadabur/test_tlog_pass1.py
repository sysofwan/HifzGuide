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
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from tadabur.nonpro_sources import TARTEEL_V1, TLOG, tlog_split

#: Every fact a committed row carries, in no particular order.
FACT_KEYS = frozenset(
    {"audio_sha256", "codec_ok", "sample_rate", "duration_s", "speech_s", "speech_ratio",
     "decoder"})
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


def mp4_aac(seconds: float = 0.5, sample_rate: int = 44100) -> bytes:
    """A mono AAC-in-MP4 clip: the container tlog's unclean split hides behind ``.wav`` paths."""
    import av

    frames = (0.2 * np.sin(2 * np.pi * 220 * np.arange(round(seconds * sample_rate))
                            / sample_rate)).astype(np.float32)[None, :]
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("aac", rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(frames, format="fltp", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
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
        "decoder": "soundfile",
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
        "decoder": None,
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
    assert measured.facts["decoder"] is None
    assert measured.waveform is None
    assert called == []


def test_an_mp4_aac_row_decodes_through_the_pyav_fallback(tp):
    pytest.importorskip("av")
    seen: list[np.ndarray] = []

    measured = tp.measure(mp4_aac(seconds=0.5), lambda waveform: seen.append(waveform) or 0.25)

    assert measured.facts["codec_ok"] is True
    assert measured.facts["decoder"] == "pyav"
    assert measured.facts["sample_rate"] == 44100
    assert measured.facts["duration_s"] == pytest.approx(0.5, abs=0.02)
    assert measured.facts["speech_ratio"] == pytest.approx(0.5, abs=0.02)
    # AAC priming/padding is codec-version specific (av 18.1 yields 8,174 samples for 0.5 s),
    # so the span is asserted loosely; duration_s above is the fact that matters.
    assert len(seen) == 1 and abs(len(seen[0]) - 8000) <= 320


def test_soundfile_remains_the_decoder_when_it_reads_the_bytes(tp):
    measured = tp.measure(wav(1.0, 16000), lambda waveform: 0.5)

    assert measured.facts["decoder"] == "soundfile"


def test_empty_bytes_are_a_codec_failure_even_with_the_fallback(tp):
    called: list[np.ndarray] = []

    measured = tp.measure(b"", lambda waveform: called.append(waveform) or 0.0)

    assert measured.facts["codec_ok"] is False
    assert measured.facts["decoder"] is None
    assert measured.waveform is None
    assert called == []


def test_an_allocation_failure_in_the_fallback_aborts_instead_of_excluding(tp, monkeypatch):
    pytest.importorskip("av")
    from tadabur import audio

    def out_of_memory(*args, **kwargs):
        raise MemoryError("out of memory")

    monkeypatch.setattr(audio.np, "concatenate", out_of_memory)

    with pytest.raises(MemoryError):
        tp.measure(mp4_aac(seconds=0.5), lambda waveform: 0.25)


def test_a_pyav_memory_error_aborts_instead_of_becoming_a_codec_exclusion(tp, monkeypatch):
    av = pytest.importorskip("av")
    from tadabur import audio

    def out_of_memory(data):
        raise av.error.MemoryError(12, "out of memory")

    monkeypatch.setattr(audio, "_decode_with_pyav", out_of_memory)

    with pytest.raises(MemoryError):
        tp.measure(mp4_aac(seconds=0.5), lambda waveform: 0.25)


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


def test_shard_rows_projects_only_the_audio_column(tp, tmp_path, monkeypatch):
    import huggingface_hub

    parquet = tmp_path / "cached.parquet"
    table = pa.table({
        "audio": pa.array([{"path": "a.flac", "bytes": b"stored"}],
                          type=pa.struct([("path", pa.string()), ("bytes", pa.binary())])),
        "label": pa.array(["an outcome the pass must never read"]),
        "is_clean": pa.array([True]),
    })
    pq.write_table(table, parquet)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, filename, **kwargs: str(parquet))
    requested: list[list[str] | None] = []
    original = pq.ParquetFile.iter_batches

    def tracked(self, *args, **kwargs):
        requested.append(kwargs.get("columns"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", tracked)

    rows = list(tp.shard_rows(CLEAN_SHARD, tmp_path / "cache"))

    assert requested == [("audio",)]
    assert [list(row) for row in rows] == [["audio"]]


def test_a_failed_delete_aborts_before_the_next_download(tp, tmp_path, monkeypatch):
    import os

    import huggingface_hub

    first, second = TLOG.shards[0], TLOG.shards[1]
    table = pa.table({"audio": pa.array(
        [{"path": "a.flac", "bytes": b"stored"}],
        type=pa.struct([("path", pa.string()), ("bytes", pa.binary())]))})
    cached = {}
    for shard in (first, second):
        path = tmp_path / f"{Path(shard).stem}.parquet"
        pq.write_table(table, path)
        cached[shard] = path
    downloaded: list[str] = []

    def fake_download(repo, filename, **kwargs):
        downloaded.append(filename)
        return str(cached[filename])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(os, "remove",
                        lambda path: (_ for _ in ()).throw(PermissionError("denied")))
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")

    with pytest.raises(PermissionError):
        tp.run([first, second], out_dir=tmp_path / "pass1", cache_dir=tmp_path / "cache",
               floors_path=floors, speech_seconds_of=lambda waveform: 0.0,
               code_sha="0" * 40)

    assert downloaded == [first]  # the second shard is never fetched
    assert cached[first].exists()
    assert not (tmp_path / "pass1" / "meta.json").exists()


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
    assert set(meta["decoders"]) == {"soundfile", "libsndfile", "pyav"}
    assert meta["decoders"]["soundfile"] is not None
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


def test_throughput_counts_each_stage_by_its_own_rows(tp, tmp_path):
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")
    rows = [{"audio": {"path": "ok.flac", "bytes": wav(1.0, 16000)}},
            {"audio": {"path": "bad.flac", "bytes": b"broken"}}]

    meta = tp.run([CLEAN_SHARD], out_dir=tmp_path / "pass1", cache_dir=tmp_path / "cache",
                  floors_path=floors, speech_seconds_of=lambda waveform: 0.5,
                  stream=fake_stream(rows), fp_dir=tmp_path / "fp", code_sha="0" * 40)

    # Decode runs on every row; VAD and the fingerprint only on the decodable one.
    assert meta["throughput"]["stage_rows"] == {"decode": 2, "vad": 1, "fingerprint": 1}
    assert all(rate is not None for rate in meta["throughput"]["rows_per_s"].values())


def test_extrapolation_scales_each_split_to_its_own_rows(tp):
    timings = tp._timings()
    by_split = {split: tp._timings() for split in tp.TLOG_SPLITS}
    # The tracer's real shard populations, at a constant 45 ms decode cost per row.
    for split, rows in (("clean", 970), ("unclean", 465)):
        by_split[split]["rows"] = rows
        by_split[split]["decode_seconds"] = 0.045 * rows
        timings["rows"] += rows
        timings["decode_seconds"] += 0.045 * rows

    throughput = tp._throughput(timings, by_split, 3)
    extrapolated = throughput["extrapolated"]

    # 0.045 s x 719,853 rows, not 0.045 s x 2,405 rows x 743 / 3 shards.
    assert extrapolated["cpu_hours"] == 8.998
    assert extrapolated["wall_hours"] == 8.998
    assert extrapolated["download_priced"] is False
    assert extrapolated["within_owner_envelope"] is False


def test_meta_records_the_pull_request_that_keeps_the_sha(tp, tmp_path):
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")

    meta = tp.run([CLEAN_SHARD], out_dir=tmp_path / "pass1", cache_dir=tmp_path / "cache",
                  floors_path=floors, speech_seconds_of=lambda waveform: 0.0,
                  stream=fake_stream([]), code_sha="a" * 40, pull_request=198)

    assert meta["code_sha"] == "a" * 40
    assert meta["pull_request"] == 198
    assert meta["pull_request_ref"] == "refs/pull/198/head"


def test_meta_records_a_repo_relative_floors_path(tp):
    assert tp._repo_relative(tp.FLOORS_PATH) == \
        "tools/tadabur/tlog/silence_floors.json"


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


def test_build_floors_refuses_a_look_clip_that_no_row_yields(tp, tmp_path):
    shard = TARTEEL_V1.shards[0]
    present, missing = "100_10_1111111111.wav", "100_10_2222222222.wav"
    rows = [{"audio": {"path": present, "bytes": wav(1.0, 16000)}}]

    def stream(shard_name, cache_dir):
        assert shard_name == shard
        yield from rows

    with pytest.raises(ValueError, match="look clips"):
        tp.build_floors([shard], look_ids={present, missing}, cache_dir=tmp_path / "cache",
                        speech_seconds_of=lambda waveform: 0.5, stream=stream)


def test_build_floors_counts_a_repeated_look_row_once(tp, tmp_path):
    shard = TARTEEL_V1.shards[0]
    look = "100_10_1111111111.wav"
    audio = wav(1.0, 16000)
    rows = [{"audio": {"path": look, "bytes": audio}},
            {"audio": {"path": look, "bytes": audio}}]

    def stream(shard_name, cache_dir):
        assert shard_name == shard
        yield from rows

    floors = tp.build_floors([shard], look_ids={look}, cache_dir=tmp_path / "cache",
                             speech_seconds_of=lambda waveform: 0.5, stream=stream)

    assert floors["count"] == 1
    assert floors["source"]["clips"] == 1
    # The split that selected the look ids is the provenance, not just the decodes file.
    assert floors["look_split"] == {
        "path": "tools/tadabur/nonpro_eval/tarteel_v1/split.json",
        "sha256": tp.sha256_file(tp.SPLIT_PATH)}


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


# --- the committed artifacts and the constants they imply ------------------------------------


def test_the_corpus_constants_follow_the_committed_index_meta(tp):
    """The S3b scale must move with #185's index instead of drifting as copied literals."""
    index = json.loads(tp.INDEX_META_PATH.read_text(encoding="utf-8"))

    assert tp.ROWS_BY_SPLIT == {split: index["counts"]["rows"][split]
                                for split in tp.TLOG_SPLITS}
    assert tp.TOTAL_ROWS == index["counts"]["rows"]["total"]
    assert tp.TOTAL_SHARDS == len(tp.TLOG.shards) == sum(index["source"]["splits"].values())


def test_the_decoder_and_vad_pins_match_the_training_requirements(tp):
    requirements = (Path(tp.__file__).resolve().parents[1]
                    / "requirements-train.txt").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^([A-Za-z0-9_-]+)(?:\[[^\]]*\])?==([^\s#]+)", requirements,
                           flags=re.MULTILINE))

    assert pins["av"] == tp.PYAV_PIN
    assert pins["silero-vad"] == tp.SILERO_VAD_VERSION


def test_the_committed_pass1_artifacts_agree_with_meta_and_the_floors(tp):
    meta = json.loads((tp.PASS1_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["floors"] == {"path": "tools/tadabur/tlog/silence_floors.json",
                              "sha256": tp.sha256_file(tp.FLOORS_PATH)}

    counted = {}
    for split in tp.TLOG_SPLITS:
        rows = []
        for path in sorted((tp.PASS1_DIR / split).glob("*.jsonl.gz")):
            stem = path.name[: -len(".jsonl.gz")]
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                content = [json.loads(line) for line in handle]
            assert {row["shard_file"] for row in content} == {stem}
            assert {row["split"] for row in content} == {split}
            assert {row["schema_version"] for row in content} == {tp.ROW_SCHEMA_VERSION}
            rows += content
        counted[split] = len(rows)

    assert counted == {split: meta["counts"]["rows"][split] for split in tp.TLOG_SPLITS}
    assert meta["counts"]["rows"]["total"] == sum(counted.values())

    floors = json.loads(tp.FLOORS_PATH.read_text(encoding="utf-8"))
    assert floors["source"]["clips"] == floors["count"]


# --- the provenance guard --------------------------------------------------------------------


def _committed_repo(path: Path) -> Path:
    """A fresh one-file git checkout at ``path``; returns it."""
    path.mkdir()
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    git = ["git", "-C", str(path), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "tracked.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    return path


def test_the_clean_tree_guard_passes_a_committed_checkout_and_refuses_any_drift(tmp_path):
    from tadabur.tlog_index import require_clean_tree

    repo = _committed_repo(tmp_path / "repo")
    assert require_clean_tree(repo) is None

    (repo / "untracked.txt").write_text("scratch\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        require_clean_tree(repo)

    (repo / "untracked.txt").unlink()
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked.txt"):
        require_clean_tree(repo)


def test_the_floors_command_stamps_the_commit_and_the_pull_request(tp, tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "require_clean_tree", lambda: None)
    monkeypatch.setattr(tp, "load_split", lambda: {"look": []})
    monkeypatch.setattr(tp, "silero_speech_seconds", lambda: (lambda waveform: 0.0))
    monkeypatch.setattr(tp, "build_floors",
                        lambda *args, **kwargs: {"f_s": 0.1, "d_s": 1.0, "count": 1, "run": {}})
    out = tmp_path / "silence_floors.json"

    tp._floors_command(SimpleNamespace(shards=[], cache_dir=tmp_path / "cache", out=out, pr=198))

    floors = json.loads(out.read_text(encoding="utf-8"))
    assert floors["code_sha"] == tp.head_sha()
    assert floors["pull_request"] == 198
    assert floors["pull_request_ref"] == "refs/pull/198/head"
    assert (floors["f_s"], floors["d_s"], floors["count"]) == (0.1, 1.0, 1)


@pytest.mark.parametrize("installed, match", [
    (lambda: (_ for _ in ()).throw(RuntimeError("PyAV is not installed")),
     "PyAV is not installed"),
    (lambda: "99.0.0", "pinned container fallback"),
])
@pytest.mark.parametrize("command", ["_floors_command", "_run_command"])
def test_both_commands_refuse_a_wrong_or_missing_pyav(
        tp, tmp_path, monkeypatch, installed, match, command):
    measured: list[tuple] = []
    monkeypatch.setattr(tp, "require_clean_tree", lambda: None)
    monkeypatch.setattr(tp, "require_pyav", installed)
    monkeypatch.setattr(tp, "run", lambda *args, **kwargs: measured.append(args))
    monkeypatch.setattr(tp, "build_floors", lambda *args, **kwargs: measured.append(args))
    args = SimpleNamespace(tracer=True, shards=[], out=tmp_path / "floors.json",
                           out_dir=tmp_path / "pass1", cache_dir=tmp_path / "cache",
                           floors=tmp_path / "floors.json", fp_dir=tmp_path / "fp",
                           code_sha=None, pr=198)

    with pytest.raises(RuntimeError, match=match):
        getattr(tp, command)(args)
    assert measured == []


@pytest.mark.parametrize("command", ["_floors_command", "_run_command"])
def test_a_measuring_command_refuses_a_dirty_tree_before_it_measures(
        tp, tmp_path, monkeypatch, command):
    measured: list[tuple] = []

    def refuse():
        raise RuntimeError("the checkout is dirty; commit or remove these paths first")

    monkeypatch.setattr(tp, "require_clean_tree", refuse)
    monkeypatch.setattr(tp, "build_floors", lambda *args, **kwargs: measured.append(args))
    monkeypatch.setattr(tp, "run", lambda *args, **kwargs: measured.append(args))
    args = SimpleNamespace(shards=[], cache_dir=tmp_path / "cache", out=tmp_path / "floors.json",
                           out_dir=tmp_path / "pass1", floors=tmp_path / "floors.json",
                           fp_dir=tmp_path / "fp", code_sha=None, pr=198, tracer=True)

    with pytest.raises(RuntimeError, match="dirty"):
        getattr(tp, command)(args)
    assert measured == []


# --- the VAD ---------------------------------------------------------------------------------


def test_the_pinned_vad_builds_and_finds_no_speech_in_a_second_of_silence():
    """The real loader, once silero is installed: 6.2.3's API is what the CLI actually calls."""
    pytest.importorskip("silero_vad")
    pytest.importorskip("onnxruntime")

    from tadabur.tlog_pass1 import TARGET_SAMPLE_RATE, silero_speech_seconds

    speech_seconds_of = silero_speech_seconds()
    assert speech_seconds_of(np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)) == 0.0


# --- the gate tranche and its projection -----------------------------------------------------


def test_the_gate_tranche_is_the_first_six_shards_of_each_split(tp):
    """The tranche rule (decision 3) is fixed before the run, not chosen after it."""
    assert tp.gate_shards() == tuple(
        [f"data/clean-{i:05d}-of-00411.parquet" for i in range(6)]
        + [f"data/unclean-{i:05d}-of-00332.parquet" for i in range(6)])
    assert len(tp.gate_shards()) == 12


def test_the_gate_projection_weights_the_remaining_rows_by_each_split_rate(tp):
    measured = {"clean": {"rows": 3880, "wall_seconds": 100.0},
                "unclean": {"rows": 970, "wall_seconds": 200.0}}
    projection = tp.gate_projection(
        measured, remaining_rows={"clean": 100_000, "unclean": 50_000}, workers=3)

    expected_seconds = 100_000 * 100.0 / 3880 + 50_000 * 200.0 / 970
    assert projection["workers"] == 3
    assert projection["rows_measured"] == {"clean": 3880, "unclean": 970}
    assert projection["rows_remaining"] == {"clean": 100_000, "unclean": 50_000}
    assert projection["rows_per_s"]["clean"] == pytest.approx(38.8)
    assert projection["rows_per_s"]["unclean"] == pytest.approx(4.85)
    assert projection["wall_hours"] == pytest.approx(expected_seconds / 3600)
    assert projection["gpu_hours"] == 0.0
    assert projection["within_owner_envelope"] is True


def test_a_split_with_no_measured_row_borrows_the_pooled_rate(tp):
    projection = tp.gate_projection({"clean": {"rows": 1000, "wall_seconds": 250.0}},
                                    remaining_rows={"clean": 1000, "unclean": 500},
                                    workers=3)

    assert projection["rows_per_s"]["clean"] == pytest.approx(4.0)
    assert projection["rows_per_s"]["unclean"] is None
    assert projection["wall_hours"] == pytest.approx(1500 / 4.0 / 3600)


def test_a_gate_projection_over_the_wall_envelope_fails_the_gate(tp):
    projection = tp.gate_projection({"clean": {"rows": 1, "wall_seconds": 10.0}},
                                    remaining_rows={"clean": 1_000_000, "unclean": 0},
                                    workers=3)

    assert projection["wall_hours"] > tp.WALL_ENVELOPE_HOURS
    assert projection["gpu_hours"] == 0.0
    assert projection["within_owner_envelope"] is False


def test_a_gate_projection_with_no_measured_row_is_refused(tp):
    with pytest.raises(ValueError, match="no measured row"):
        tp.gate_projection({}, remaining_rows={"clean": 1, "unclean": 1}, workers=3)


# --- the resume guard ------------------------------------------------------------------------


def _index_for(tp, tmp_path: Path, shard: str, paths: list[str]) -> Path:
    """A one-shard S1 index, written by the same deterministic JSONL-gzip writer."""
    split, stem = tlog_split(shard), Path(shard).stem
    rows = [{"schema_version": 1, "split": split, "shard_file": stem, "row": index,
             "audio_path": path, "filename_label": None, "recording_id": None,
             "id_group": path, "tarteel_v1_eval_overlap": False,
             "tarteel_v1_any_overlap": False} for index, path in enumerate(paths)]
    index_dir = tmp_path / "index"
    tp.write_pass1(rows, index_dir / split / f"{stem}.jsonl.gz")
    return index_dir


def _refuse_download(shard: str, cache_dir: Path):
    raise AssertionError(f"a resumed shard must not be downloaded: {shard}")
    yield  # pragma: no cover — makes this a generator like shard_rows


def test_a_resumed_shard_is_verified_and_not_downloaded(tp, tmp_path):
    shard, paths = CLEAN_SHARD, ["a.flac", "b.flac"]
    index_dir = _index_for(tp, tmp_path, shard, paths)
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")
    rows = [{"audio": {"path": path, "bytes": wav(1.0, 16000)}} for path in paths]
    out_dir = tmp_path / "pass1"

    first = tp.run([shard], out_dir=out_dir, cache_dir=tmp_path / "cache", floors_path=floors,
                   speech_seconds_of=lambda waveform: 0.5, stream=fake_stream(rows),
                   code_sha="0" * 40, index_dir=index_dir)
    second = tp.run([shard], out_dir=out_dir, cache_dir=tmp_path / "cache", floors_path=floors,
                    speech_seconds_of=lambda waveform: 0.5, stream=_refuse_download,
                    code_sha="0" * 40, index_dir=index_dir)

    assert first["throughput"]["shards_measured"] == 1
    assert first["throughput"]["shards_skipped"] == 0
    assert second["counts"]["rows"] == {"clean": 2, "unclean": 0, "total": 2}
    assert second["throughput"]["shards_skipped"] == 1
    assert second["throughput"]["shards_measured"] == 0
    # The skipped shard's rows are counted in the freeze but not in the gate's rates.
    assert second["throughput"]["rows"] == 0


def test_a_resumed_shard_with_a_different_row_count_is_refused(tp, tmp_path):
    shard, paths = CLEAN_SHARD, ["a.flac", "b.flac"]
    index_dir = _index_for(tp, tmp_path, shard, paths)
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")
    out_dir = tmp_path / "pass1"
    rows = [{"audio": {"path": path, "bytes": wav(1.0, 16000)}} for path in paths]
    tp.run([shard], out_dir=out_dir, cache_dir=tmp_path / "cache", floors_path=floors,
           speech_seconds_of=lambda waveform: 0.5, stream=fake_stream(rows),
           code_sha="0" * 40, index_dir=index_dir)

    output = out_dir / "clean" / "clean-00000-of-00411.jsonl.gz"
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        content = [json.loads(line) for line in handle]
    tp.write_pass1(content[:1], output)

    with pytest.raises(ValueError, match="row count"):
        tp.run([shard], out_dir=out_dir, cache_dir=tmp_path / "cache", floors_path=floors,
               speech_seconds_of=lambda waveform: 0.5, stream=_refuse_download,
               code_sha="0" * 40, index_dir=index_dir)


def test_a_resumed_shard_whose_paths_drift_from_the_index_is_refused(tp, tmp_path):
    shard, paths = CLEAN_SHARD, ["a.flac", "b.flac"]
    index_dir = _index_for(tp, tmp_path, shard, paths)
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")
    out_dir = tmp_path / "pass1"
    rows = [{"audio": {"path": path, "bytes": wav(1.0, 16000)}} for path in paths]
    tp.run([shard], out_dir=out_dir, cache_dir=tmp_path / "cache", floors_path=floors,
           speech_seconds_of=lambda waveform: 0.5, stream=fake_stream(rows),
           code_sha="0" * 40, index_dir=index_dir)

    output = out_dir / "clean" / "clean-00000-of-00411.jsonl.gz"
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        content = [json.loads(line) for line in handle]
    content[0]["audio_path"] = "z.flac"
    tp.write_pass1(content, output)

    with pytest.raises(ValueError, match="audio_path"):
        tp.run([shard], out_dir=out_dir, cache_dir=tmp_path / "cache", floors_path=floors,
               speech_seconds_of=lambda waveform: 0.5, stream=_refuse_download,
               code_sha="0" * 40, index_dir=index_dir)


def test_a_corrupt_resumed_shard_is_refused(tp, tmp_path):
    shard = CLEAN_SHARD
    index_dir = _index_for(tp, tmp_path, shard, ["a.flac"])
    floors = tmp_path / "silence_floors.json"
    floors.write_text(json.dumps({"vad": tp.VAD_CONFIG}), encoding="utf-8")
    out_dir = tmp_path / "pass1"
    output = out_dir / "clean" / "clean-00000-of-00411.jsonl.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"not gzip")

    with pytest.raises((OSError, ValueError)):
        tp.run([shard], out_dir=out_dir, cache_dir=tmp_path / "cache", floors_path=floors,
               speech_seconds_of=lambda waveform: 0.5, stream=_refuse_download,
               code_sha="0" * 40, index_dir=index_dir)


# --- the shared floors reader ----------------------------------------------------------------


@pytest.fixture
def floors_reading():
    return {"f_s": 0.4, "d_s": 1.5}


def test_a_codec_failure_is_excluded_before_its_null_facts_are_read(tp, floors_reading):
    row = {"codec_ok": False, "duration_s": None, "speech_s": None, "speech_ratio": None}

    assert tp.exclusion_reasons(row, floors_reading) == ("codec",)
    assert tp.usable(row, floors_reading) is False


def test_the_silence_floors_are_inclusive(tp, floors_reading):
    at_both_floors = {"codec_ok": True, "duration_s": 4.0, "speech_s": 1.5,
                      "speech_ratio": 0.4}

    assert tp.exclusion_reasons(at_both_floors, floors_reading) == ()
    assert tp.usable(at_both_floors, floors_reading) is True


@pytest.mark.parametrize("ratio, speech", [(0.39, 5.0), (0.9, 1.49)])
def test_a_short_or_sparse_clip_is_silence(tp, floors_reading, ratio, speech):
    row = {"codec_ok": True, "duration_s": 10.0, "speech_s": speech, "speech_ratio": ratio}

    assert tp.exclusion_reasons(row, floors_reading) == ("silence",)
    assert tp.usable(row, floors_reading) is False


# --- the full run's committed facts ----------------------------------------------------------


def test_every_index_shard_has_a_committed_pass1_file_with_the_same_rows(tp):
    index_files = sorted(tp.INDEX_META_PATH.parent.glob("*/*.jsonl.gz"))
    assert len(index_files) == tp.TOTAL_SHARDS == 743

    for index_path in index_files:
        split, stem = index_path.parent.name, index_path.name[: -len(".jsonl.gz")]
        with gzip.open(index_path, "rt", encoding="utf-8") as handle:
            index_rows = [json.loads(line) for line in handle]
        pass1_path = tp.PASS1_DIR / split / index_path.name
        assert pass1_path.exists(), f"pass 1 never committed {split}/{stem}"
        with gzip.open(pass1_path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        assert [row["row"] for row in rows] == list(range(len(index_rows)))
        assert [row["audio_path"] for row in rows] == [row["audio_path"] for row in index_rows]


def test_the_committed_pass1_counts_match_the_index_exactly(tp):
    index_meta = json.loads(tp.INDEX_META_PATH.read_text(encoding="utf-8"))
    meta = json.loads((tp.PASS1_DIR / "meta.json").read_text(encoding="utf-8"))

    assert meta["counts"]["rows"] == index_meta["counts"]["rows"]
    assert meta["counts"]["rows"]["total"] == tp.TOTAL_ROWS == 719_853


def test_the_committed_meta_names_each_shard_s_producing_sha_and_pr(tp):
    meta = json.loads((tp.PASS1_DIR / "meta.json").read_text(encoding="utf-8"))
    shards = meta["shards"]
    assert len(shards) == 743

    for shard in TRACER_SHARDS:
        split, stem = tlog_split(shard), Path(shard).stem
        entry = shards[f"{split}/{stem}"]
        assert entry["code_sha"].startswith("d2af0e3")
        assert entry["pull_request"] == 198
    for key, entry in shards.items():
        if key.split("/")[1] in {Path(shard).stem for shard in TRACER_SHARDS}:
            continue
        assert re.fullmatch(r"[0-9a-f]{40}", entry["code_sha"])
        assert entry["pull_request"] == meta["pull_request"]


def test_the_run_command_selects_the_gate_at_three_workers(tp, tmp_path, monkeypatch):
    captured = []

    def record(shards, **kwargs):
        captured.append((tuple(shards), kwargs))
        return {"throughput": {}}

    monkeypatch.setattr(tp, "require_clean_tree", lambda: None)
    monkeypatch.setattr(tp, "_require_pinned_pyav", lambda: None)
    monkeypatch.setattr(tp, "silero_speech_seconds", lambda: None)
    monkeypatch.setattr(tp, "run", record)
    args = SimpleNamespace(gate=True, tracer=False, all=False, shards=None,
                           out_dir=tmp_path / "pass1", cache_dir=tmp_path / "cache",
                           floors=tmp_path / "floors.json", fp_dir=tmp_path / "fp",
                           code_sha="0" * 40, pr=7, workers=3,
                           index_dir=tmp_path / "index")

    tp._run_command(args)

    shards, kwargs = captured[0]
    assert shards == tp.gate_shards()
    assert kwargs["workers"] == 3
    assert kwargs["index_dir"] == tmp_path / "index"


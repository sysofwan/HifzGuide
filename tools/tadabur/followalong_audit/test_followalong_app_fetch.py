"""The App-arm audio fetch (#309): one stream pass per shard, verified, resumable, WAVs by checksum.

``fetch_clips(rows, wav_dir, *, stream, load_audio)`` writes ``<audio_sha256>.wav`` (16 kHz mono
PCM_16, the one format the driver accepts) for every row, verifying each source clip against its
recorded ``audio_sha256`` first, and ``<wav_dir>/wav_sha256.json`` (the manifest
:func:`tadabur.followalong_audit.clips.verify_wav` reads). ``stream(name)`` yields the shard's
parquet rows, ``{"audio": {"bytes": ...}}``, for ``data/<shard_file>.parquet``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tadabur.followalong_audit.clips import verify_wav, wav_samples
from tadabur.followalong_audit.fetch import fetch_clips
from tadabur.tlog_index import sha256_file


def clip_bytes(name: str) -> bytes:
    return f"source-{name}".encode()


def row_of(shard: int, index: int) -> dict:
    name = f"{shard}-{index}"
    return {"split": "clean", "shard_file": f"clean-{shard:05d}-of-00411", "row": index,
            "audio_path": f"{name}.flac",
            "audio_sha256": hashlib.sha256(clip_bytes(name)).hexdigest()}


def load_audio(data: bytes) -> np.ndarray:
    """A fake decoder: a ramp whose length is the source's byte count."""
    return np.linspace(-0.5, 0.5, len(data) * 100, dtype=np.float32)


class FakeShards:
    """The row source: a shard is the list of its clips' bytes, in row order."""

    def __init__(self, shards: dict[int, int]) -> None:
        self.shards = shards
        self.calls: list[str] = []

    def __call__(self, name: str):
        self.calls.append(name)
        shard = int(name.split("clean-")[1][:5])
        return iter({"audio": {"bytes": clip_bytes(f"{shard}-{index}")}}
                    for index in range(self.shards[shard]))


def test_each_clip_becomes_a_16k_mono_wav_named_by_its_source_checksum(tmp_path):
    rows = [row_of(3, 1), row_of(3, 4)]
    manifest = fetch_clips(rows, tmp_path, stream=FakeShards({3: 6}), load_audio=load_audio)
    for row in rows:
        wav = tmp_path / f"{row['audio_sha256']}.wav"
        assert wav_samples(wav) == len(clip_bytes(row["audio_path"][:-5])) * 100
        assert manifest[row["audio_sha256"]] == {
            "wav_sha256": sha256_file(wav), "samples": wav_samples(wav)}
        assert verify_wav(wav, manifest[row["audio_sha256"]]) == sha256_file(wav)


def test_the_manifest_is_written_where_the_run_reads_it(tmp_path):
    manifest = fetch_clips([row_of(0, 2)], tmp_path, stream=FakeShards({0: 3}),
                           load_audio=load_audio)
    assert json.loads((tmp_path / "wav_sha256.json").read_text(encoding="utf-8")) == manifest


def test_a_shard_is_streamed_once_for_all_its_rows(tmp_path):
    shards = FakeShards({0: 5, 1: 5})
    fetch_clips([row_of(0, 1), row_of(0, 3), row_of(1, 2)], tmp_path, stream=shards,
                load_audio=load_audio)
    assert sorted(shards.calls) == ["data/clean-00000-of-00411.parquet",
                                    "data/clean-00001-of-00411.parquet"]


def test_a_source_clip_that_is_not_the_recorded_one_is_refused_and_leaves_no_wav(tmp_path):
    bad = {**row_of(0, 1), "audio_sha256": "0" * 64}
    with pytest.raises(ValueError):
        fetch_clips([bad], tmp_path, stream=FakeShards({0: 3}), load_audio=load_audio)
    assert not list(tmp_path.glob("*.wav"))


def test_a_wanted_row_the_shard_does_not_hold_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        fetch_clips([row_of(0, 9)], tmp_path, stream=FakeShards({0: 3}), load_audio=load_audio)


def test_a_rerun_reads_no_shard_for_clips_already_fetched(tmp_path):
    rows = [row_of(0, 1), row_of(1, 0)]
    first = fetch_clips(rows, tmp_path, stream=FakeShards({0: 2, 1: 2}), load_audio=load_audio)

    def no_stream(name):
        raise AssertionError(f"{name} was streamed again")

    assert fetch_clips(rows, tmp_path, stream=no_stream, load_audio=load_audio) == first


def test_fetching_more_clips_keeps_the_manifest_entries_already_there(tmp_path):
    first = fetch_clips([row_of(0, 1)], tmp_path, stream=FakeShards({0: 2}), load_audio=load_audio)
    both = fetch_clips([row_of(0, 1), row_of(1, 0)], tmp_path, stream=FakeShards({0: 2, 1: 2}),
                       load_audio=load_audio)
    assert set(both) == {row_of(0, 1)["audio_sha256"], row_of(1, 0)["audio_sha256"]}
    assert both[row_of(0, 1)["audio_sha256"]] == first[row_of(0, 1)["audio_sha256"]]
    assert set(json.loads((tmp_path / "wav_sha256.json").read_text(encoding="utf-8"))) == set(both)


def test_the_directory_holds_only_wavs_and_their_manifest(tmp_path):
    rows = [row_of(0, 0), row_of(0, 1)]
    fetch_clips(rows, tmp_path, stream=FakeShards({0: 2}), load_audio=load_audio)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [f"{row['audio_sha256']}.wav" for row in rows] + ["wav_sha256.json"])

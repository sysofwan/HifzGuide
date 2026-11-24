"""The fetch's shard contract: the canonical matcher, honest manifests, cleaned downloads (#309).

``shard_items`` matches a wanted row by position or, for a source that numbers rows differently,
by the clip's recorded checksum. A manifest entry whose WAV is gone is not evidence that the clip
was fetched, and a download that dies mid-shard must be closed (the stream deletes the parquet
once read) instead of left for the next run.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from tadabur.followalong_audit.fetch import fetch_clips


def clip_bytes(name: str) -> bytes:
    return f"source-{name}".encode()


def row_of(index: int) -> dict:
    name = f"0-{index}"
    return {"split": "clean", "shard_file": "clean-00000-of-00411", "row": index,
            "audio_path": f"{name}.flac",
            "audio_sha256": hashlib.sha256(clip_bytes(name)).hexdigest()}


def load_audio(data: bytes):
    import numpy as np

    return np.linspace(-0.5, 0.5, len(data) * 100, dtype=np.float32)


class Shifted:
    """A shard whose rows are not numbered as the manifest says; only the checksums agree."""

    def __init__(self, wanted: bytes) -> None:
        self.wanted = wanted

    def __call__(self, name: str):
        return iter([{"audio": {"bytes": self.wanted}},
                     {"audio": {"bytes": b"an unrelated row"}}])


def test_a_row_is_found_by_its_checksum_when_the_stream_numbers_it_differently(tmp_path):
    row = row_of(1)
    manifest = fetch_clips([row], tmp_path, stream=Shifted(clip_bytes("0-1")),
                           load_audio=load_audio)
    assert (tmp_path / f"{row['audio_sha256']}.wav").is_file()
    assert set(manifest) == {row["audio_sha256"]}


def test_a_manifest_entry_whose_wav_is_gone_is_fetched_again(tmp_path):
    row = row_of(0)
    fetch_clips([row], tmp_path, stream=Shifted(clip_bytes("0-0")), load_audio=load_audio)
    wav = tmp_path / f"{row['audio_sha256']}.wav"
    wav.unlink()

    manifest = fetch_clips([row], tmp_path, stream=Shifted(clip_bytes("0-0")),
                           load_audio=load_audio)
    assert wav.is_file() and row["audio_sha256"] in manifest


def test_a_failure_mid_shard_closes_the_download(tmp_path):
    closed: list[str] = []

    def stream(name: str):
        try:
            yield {"audio": {"bytes": clip_bytes("0-0")}}
            yield {"audio": {"bytes": b"not the recorded clip"}}
        finally:
            closed.append(name)

    with pytest.raises(ValueError, match="not the recorded"):
        fetch_clips([row_of(0), row_of(1)], tmp_path, stream=stream, load_audio=load_audio)
    assert closed, "the shard was left downloaded after the failure"
    assert not (tmp_path / "wav_sha256.json").exists()


def test_the_manifest_pruning_is_written_back(tmp_path):
    row = row_of(0)
    fetch_clips([row], tmp_path, stream=Shifted(clip_bytes("0-0")), load_audio=load_audio)
    (tmp_path / f"{row['audio_sha256']}.wav").unlink()
    fetch_clips([row], tmp_path, stream=Shifted(clip_bytes("0-0")), load_audio=load_audio)
    assert set(json.loads((tmp_path / "wav_sha256.json").read_text(encoding="utf-8"))) == {
        row["audio_sha256"]}

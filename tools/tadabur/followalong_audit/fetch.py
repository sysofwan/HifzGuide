"""The App-arm audio fetch (#309): one stream pass per shard, verified, resumable.

S1's WAVs were made by an uncommitted script; this is the committed fetch. Every clip of the
population is written to ``<wav_dir>/<audio_sha256>.wav`` — 16 kHz mono PCM_16, the one format
:func:`tadabur.followalong_audit.tracer.run_driver` accepts — and recorded in
``<wav_dir>/wav_sha256.json`` (``{audio_sha256: {wav_sha256, samples}}``), the manifest
:func:`tadabur.followalong_audit.clips.verify_wav` checks the run's WAVs against.

A shard is streamed **once** for all of its rows (the tlog stream downloads the shard and deletes
it once read: cuda-dev's disk is the binding constraint), through the canonical
:func:`tadabur.tlog_eval_decode.shard_items`, so a source that numbers rows differently from the
manifest is matched by the clip's recorded checksum and a wanted row the shard does not hold is
an error. The gain is the WAVs: about 1.6 GB for the 5,742-clip population. A clip whose manifest
entry names a WAV that is no longer on disk is fetched again, so an interrupted or cleaned
directory resumes honestly; a run that fetches more clips keeps the entries it had.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tadabur.followalong_audit.atomic import write_bytes_atomically
from tadabur.followalong_audit.clips import SAMPLE_RATE
from tadabur.nonpro_sources import audio_sha256
from tadabur.tlog_eval_decode import shard_items
from tadabur.tlog_index import sha256_file
from tadabur.tlog_mining.mined_decode import grouped_by_shard

#: The manifest's name in the WAV directory: what the run reads by default.
MANIFEST_NAME = "wav_sha256.json"


def write_wav(path: Path, samples: Sequence[float] | np.ndarray) -> int:
    """A mono 16 kHz PCM_16 WAV of ``samples`` (float in [-1, 1]); the frame count written."""
    import wave

    audio = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)
    return len(audio)


def write_manifest(path: Path, manifest: Mapping[str, Mapping]) -> None:
    """The digest manifest, sorted so the same fetch writes the same bytes."""
    write_bytes_atomically(Path(path),
                           (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_manifest(path: Path) -> dict[str, dict]:
    """The fetch manifest a directory holds, or an empty one before the first fetch."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def default_audio(data: bytes) -> np.ndarray:
    """The clip's mono 16 kHz samples, by the pinned decoder (``tadabur.tlog_pool``)."""
    from tadabur.tlog_pool import default_audio as decode

    return decode(data)


def fetch_clips(rows: Sequence[Mapping], wav_dir: Path, *,
                stream: Callable[[str], object] | None = None,
                load_audio: Callable[[bytes], object] | None = None,
                workers: int = 1, shard_cache: Path | None = None) -> dict[str, dict]:
    """Write every row's WAV and return the whole digest manifest; resume and verify per clip.

    ``stream(name)`` yields ``data/<shard_file>.parquet``'s rows (``{"audio": {"bytes": ...}}``),
    and is called once per shard. With ``workers > 1`` the shards are split over a process pool
    (one shard per worker, each streamed and deleted), and ``shard_cache`` names the directory
    the default stream downloads into: a custom ``stream`` is a single-process argument.
    """
    load_audio = load_audio or default_audio
    wav_dir = Path(wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = wav_dir / MANIFEST_NAME
    manifest = {key: value for key, value in read_manifest(manifest_path).items()
                if (wav_dir / f"{key}.wav").is_file()}
    wanted = [row for row in rows if row["audio_sha256"] not in manifest]
    by_shard = grouped_by_shard(wanted)
    if workers > 1 and stream is not None:
        raise ValueError("a custom stream cannot cross a process boundary; give shard_cache")
    if workers > 1 and len(by_shard) > 1:
        groups = [sorted(by_shard)[index::workers] for index in range(workers)]
        lock = _lock_path(wav_dir, shard_cache)
        jobs = [([row for shard in group for row in by_shard[shard]], str(wav_dir),
                 str(shard_cache) if shard_cache is not None else None, str(lock))
                for group in groups if group]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_fetch_group, jobs))
        manifest = read_manifest(manifest_path)
    else:
        source = stream if stream is not None else _default_stream(shard_cache)
        for shard_file, shard_rows in sorted(by_shard.items()):
            manifest.update(_fetch_shard(shard_file, shard_rows, wav_dir, source, load_audio))
            write_manifest(manifest_path, manifest)
    write_manifest(manifest_path,
                   {key: manifest[key] for key in sorted(manifest)})
    return manifest


def _default_stream(shard_cache: Path | None) -> Callable[[str], object]:
    from tadabur.tlog_eval_decode import shard_stream

    if shard_cache is None:
        raise ValueError("give shard_cache (the stream downloads each shard into it)")
    return shard_stream(Path(shard_cache))


def _lock_path(wav_dir: Path, shard_cache: Path | None) -> Path:
    """The cross-process manifest lock, kept out of the WAV directory (which holds only WAVs)."""
    directory = Path(shard_cache) if shard_cache is not None else Path(wav_dir).parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{Path(wav_dir).name}.manifest.lock"


def _fetch_group(job: tuple[list[dict], str, str | None, str]) -> None:
    """One worker's shard group, through its own stream; each finished shard is merged durably."""
    rows, wav_dir, shard_cache, lock = job
    source = _default_stream(Path(shard_cache) if shard_cache else None)
    manifest_path = Path(wav_dir) / MANIFEST_NAME
    for shard_file, shard_rows in sorted(grouped_by_shard(rows).items()):
        found = _fetch_shard(shard_file, shard_rows, Path(wav_dir), source, default_audio)
        _merge_manifest(manifest_path, found, Path(lock))


def _merge_manifest(path: Path, entries: Mapping[str, Mapping], lock_path: Path) -> None:
    """Add ``entries`` to the manifest under an exclusive lock, so a worker never drops another's."""
    import fcntl

    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            current = read_manifest(path)
            current.update(entries)
            write_manifest(path, {key: current[key] for key in sorted(current)})
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _fetch_shard(shard_file: str, rows: Sequence[Mapping], wav_dir: Path,
                 stream: Callable[[str], object],
                 load_audio: Callable[[bytes], object]) -> dict[str, dict]:
    """One stream pass of a shard, writing every wanted row's verified WAV.

    ``shard_items`` owns the one-pass row matching; closing it on any exception (a checksum that
    does not match, a disk that fills) drops the downloaded shard through the stream's own
    ``finally`` instead of leaving it for the next run to find.
    """
    found: dict[str, dict] = {}
    with contextlib.closing(shard_items(rows, stream=stream)) as items:
        for row, data in items:
            digest = audio_sha256(data)
            if digest != row["audio_sha256"]:
                raise ValueError(f"{row['audio_path']} is {digest}, not the recorded "
                                 f"{row['audio_sha256']}")
            wav = wav_dir / f"{row['audio_sha256']}.wav"
            samples = write_wav(wav, load_audio(data))
            found[row["audio_sha256"]] = {"wav_sha256": sha256_file(wav), "samples": samples}
    return found

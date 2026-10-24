"""The tlog pass-1 tracer (#187): outcome-blind facts for three shards, and the silence floors.

Pass 1 reads a shard once, records per-row facts that depend on nothing but the audio, writes
the shard's committed rows, and deletes the shard. Only the ``audio`` parquet column is
projected, so tlog's own outcome labels are never read, and a failed deletion aborts the run
before the next download: the disk budget is an invariant here. It runs **no teacher, student
or ASR model** and produces no transcript. The committed facts are ``audio_sha256``, ``codec_ok``,
``sample_rate``, ``duration_s``, ``speech_s``, ``speech_ratio`` and ``decoder`` (``soundfile`` or
``pyav``; ``null`` when neither reads the bytes). ``soundfile`` is tried first and PyAV
(:data:`PYAV_PIN`) only when it refuses the container, which the unclean split's MP4/AAC rows
need; the pins are recorded under ``decoders`` in ``meta.json``. The Haitsma-Kalker
fingerprint (:mod:`tadabur.qud_fingerprint`) is written to cuda-dev scratch only. The owner
accepted the clip-level fallback (2026-10-24), so no speaker embedding is computed at all.

``tools/tadabur/tlog/pass1/<split>/<shard>.jsonl.gz``
    one row per shard row: ``schema_version`` (2), ``split``, ``shard_file``, ``row``,
    ``audio_path`` and the seven facts. Keys sorted, ``mtime`` 0 and written atomically, so a
    rerun on the build interpreter reproduces every byte (the gzip OS header byte is
    platform-dependent, as in the index).
``tools/tadabur/tlog/pass1/meta.json``
    the code SHA with the PR number and ``refs/pull/<N>/head`` (the squash merge drops the
    branch commit from main's history), the VAD config, the installed decoder versions
    (``soundfile``, ``libsndfile``, ``pyav``), the floors file's SHA-256, the row counts and
    the measured throughput. ``"embeddings"`` records the fallback, not omission.
    Both commands refuse a checkout with uncommitted or untracked changes, so the cited SHA is
    the code that actually ran.
``tools/tadabur/tlog/silence_floors.json``
    the pre-registration's Exclusions floors: ``f_s`` and ``d_s`` are the type-7 0.5th
    percentiles of ``speech_ratio`` and ``speech_s`` over the Tarteel v1 look clips that decode.
    Silero VAD (:data:`SILERO_VAD_VERSION`, ONNX on CPU) supplies ``speech_s``; the padded
    Silero spans are summed. The floors commit lands **before** any pass-1 output commit, and
    :func:`run` refuses to start when its VAD config differs from the committed floors. Like
    ``meta.json`` it records ``code_sha``, ``pull_request`` and ``pull_request_ref``.

**Tracer rule (written before the run).** The tracer reads clean shard indices 0 and 205 (of
411) and unclean index 0 (of 332): :func:`tracer_shards`. It measures download, decode+resample,
VAD and fingerprint separately, and extrapolates the 743-shard cost from those numbers; the PR
and #173 carry the restatement. The branch is squash-merged, so a cited ``code_sha`` survives
only under ``refs/pull/<N>/head`` and is recorded with the PR number.

tlog audio is gated: the reads need a Hub token that accepted the dataset's terms and run on
cuda-dev, where the scratch directories live. Shards and their audio are never committed.

Usage (from ``tools/``, on cuda-dev). VAD and fingerprinting are CPU-only, so neither command
takes the GPU lock::

  python -m tadabur.tlog_pass1 floors --cache-dir /root/scratch/tlog/cache --pr 198
  python -m tadabur.tlog_pass1 run --tracer --cache-dir /root/scratch/tlog/cache --pr 198
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.tashkeel_eval import write_text_atomically

from .audio import (DecodeError, TARGET_SAMPLE_RATE, decode_to_mono_16k_with_rate_and_decoder,
                    decoder_versions, require_pyav)
from .nonpro_sources import (TARTEEL_V1, TLOG, TLOG_SPLITS, Dataset, audio_sha256, iter_rows,
                             tlog_split)
from .qud_fingerprint import fingerprint, save_fingerprints
from .tarteel_split import SPLIT_PATH, load_split
from .tlog_index import head_sha, require_clean_tree, sha256_file, write_shard

SCHEMA_VERSION = 1
#: Rows gained ``decoder`` after the first tracer run, so a row is schema 2 while the floors and
#: ``meta.json`` keep their (additively extended) schema 1.
ROW_SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Where the committed pass-1 output and the pre-registration's floors live.
PASS1_DIR = Path(__file__).parent / "tlog" / "pass1"
FLOORS_PATH = Path(__file__).parent / "tlog" / "silence_floors.json"
#: cuda-dev scratch: fingerprints only, never a committed row or the repo.
FP_DIR = Path("/root/scratch/tlog/fp")
#: The Tarteel v1 look clips' decodes: the floors' population, and the provenance hash.
DECODES_PATH = Path(__file__).parent / "nonpro_eval" / "tarteel_v1" / "decodes.base.jsonl.gz"

#: The pinned VAD. The exact config is recorded in the floors and in ``pass1/meta.json``, and a
#: run refuses a floors file whose config differs (the pre-registration requires one VAD).
SILERO_VAD_VERSION = "6.2.3"
VAD_CONFIG = {
    "model": "silero-vad",
    "version": SILERO_VAD_VERSION,
    "backend": "onnx-cpu",
    "sample_rate": TARGET_SAMPLE_RATE,
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
}

#: The container fallback's pin, in ``requirements-train.txt`` and ``meta.json``. The unclean
#: split's MP4/AAC rows decode through it, so a version bump can change their samples.
PYAV_PIN = "18.1.0"

#: The tracer rule, fixed before the run: clean shard indices 0 and 205, unclean 0.
TRACER_CLEAN = (0, 205)
TRACER_UNCLEAN = 0
#: The corpus' committed size, read from #185's index rather than copied: a re-pin of the index
#: would otherwise silently skew every S3b number. ``len(TLOG.shards)`` is 411 + 332.
INDEX_META_PATH = Path(__file__).parent / "tlog" / "index" / "meta.json"
ROWS_BY_SPLIT = {
    split: int(rows) for split, rows in
    json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))["counts"]["rows"].items()
    if split in TLOG_SPLITS}
TOTAL_SHARDS = len(TLOG.shards)
TOTAL_ROWS = sum(ROWS_BY_SPLIT.values())
#: The brief's corpus byte count (313 GiB), used to price downloads from the measured rate
#: instead of assuming the three tracer shards are byte-representative.
TOTAL_AUDIO_BYTES = 313 * 2**30
#: The owner's S3b envelope: wall hours, GPU hours and peak disk bytes.
WALL_ENVELOPE_HOURS = 8.0
GPU_ENVELOPE_HOURS = 3.0
DISK_ENVELOPE_BYTES = 5_000_000_000


@dataclass(frozen=True)
class Measured:
    """One row's fact record, and the 16 kHz signal the fingerprint is taken from."""

    facts: dict
    waveform: np.ndarray | None


def tracer_shards() -> tuple[str, str, str]:
    """The three tracer shards, by the rule in the module docstring."""
    clean = [shard for shard in TLOG.shards if tlog_split(shard) == "clean"]
    unclean = [shard for shard in TLOG.shards if tlog_split(shard) == "unclean"]
    return (clean[TRACER_CLEAN[0]], clean[TRACER_CLEAN[1]], unclean[TRACER_UNCLEAN])


def measure(audio: bytes, speech_seconds_of: Callable[[np.ndarray], float], *,
            timings: dict | None = None) -> Measured:
    """The outcome-blind facts of one row's stored bytes, and its 16 kHz waveform.

    Decode+resample and VAD are timed separately into ``timings``. A row that does not decode,
    or whose duration is 0, is a ``codec_ok: false`` fact (the pre-registration's ``codec``
    exclusion), not an exception; the VAD is never called for it. ``decoder`` names which of
    the two decoders read the bytes, so the fallback's share of the corpus stays visible.
    Only :class:`~tadabur.audio.DecodeError` is a row fact: allocation failures, a missing
    dependency and programming errors abort the run instead of becoming codec exclusions.
    """
    digest = audio_sha256(audio)
    started = time.perf_counter()
    try:
        waveform, sample_rate, decoder = decode_to_mono_16k_with_rate_and_decoder(audio)
    except DecodeError:
        waveform, sample_rate, decoder = None, None, None
    if timings is not None:
        timings["decode_seconds"] += time.perf_counter() - started
    duration_s = len(waveform) / TARGET_SAMPLE_RATE if waveform is not None else 0.0
    if waveform is None or duration_s <= 0:
        return Measured(
            {"audio_sha256": digest, "codec_ok": False, "sample_rate": None,
             "duration_s": None, "speech_s": None, "speech_ratio": None, "decoder": None},
            None)
    started = time.perf_counter()
    speech_s = float(speech_seconds_of(waveform))
    if timings is not None:
        timings["vad_seconds"] += time.perf_counter() - started
        timings["vad_rows"] += 1
    return Measured(
        {"audio_sha256": digest, "codec_ok": True, "sample_rate": sample_rate,
         "duration_s": duration_s, "speech_s": speech_s,
         "speech_ratio": speech_s / duration_s, "decoder": decoder},
        waveform)


def shard_rows(shard: str, cache_dir: Path, *, dataset: Dataset = TLOG,
               on_download: Callable[[str], None] | None = None,
               columns: Sequence[str] | None = ("audio",),
               strict_delete: bool = True) -> Iterator[dict]:
    """A shard's rows, downloaded and then deleted by :func:`nonpro_sources.iter_rows`.

    The dataset guard runs before the download: pass 1 reads tlog shards only, and the floors
    read Tarteel v1 only. Only the ``audio`` column is projected, so tlog's own outcome label
    is never materialized, and a failed deletion is an error: the disk budget is an invariant.
    """
    if shard not in dataset.shards:
        raise ValueError(f"not a {dataset.name} shard: {shard}")
    return iter_rows(dataset, shard, cache_dir, on_download=on_download, columns=columns,
                     strict_delete=strict_delete)


def process_shard(shard: str, cache_dir: Path, *,
                  speech_seconds_of: Callable[[np.ndarray], float],
                  fp_dir: Path | None = None, stream=None,
                  on_download: Callable[[str], None] | None = None,
                  timings: dict | None = None) -> list[dict]:
    """Measure every row of one shard, in row order, and write its fingerprints to scratch.

    ``stream`` is the row source (default :func:`shard_rows`); tests inject a fake shard.
    """
    split, stem = tlog_split(shard), Path(shard).stem
    timings = _timings() if timings is None else timings
    if stream is None:
        rows = shard_rows(shard, cache_dir, on_download=on_download)
    else:
        rows = stream(shard, cache_dir)
    records: list[dict] = []
    prints: dict[str, tuple[str, np.ndarray]] = {}
    for index, row in enumerate(rows):
        measured = measure(row["audio"]["bytes"], speech_seconds_of, timings=timings)
        records.append({
            "schema_version": ROW_SCHEMA_VERSION, "split": split, "shard_file": stem,
            "row": index, "audio_path": row["audio"]["path"], **measured.facts})
        if fp_dir is not None and measured.waveform is not None:
            started = time.perf_counter()
            prints[row["audio"]["path"]] = (measured.facts["audio_sha256"],
                                            fingerprint(measured.waveform))
            timings["fingerprint_seconds"] += time.perf_counter() - started
            timings["fingerprint_rows"] += 1
    if fp_dir is not None and prints:
        fp_dir = Path(fp_dir)
        fp_dir.mkdir(parents=True, exist_ok=True)
        path = fp_dir / f"{stem}.npz"
        save_fingerprints(path, prints)
        timings["fingerprint_bytes"] += path.stat().st_size
    timings["rows"] += len(records)
    return records


def write_pass1(rows: list[dict], path: Path) -> None:
    """Write one shard's rows by the index writer's deterministic JSONL-gzip contract."""
    write_shard(rows, path)


def silence_floors(rows: Iterable[dict], *, look_clips: int, decodes_sha256: str) -> dict:
    """The pre-registration's ``f_s`` and ``d_s`` from measured Tarteel look rows.

    The population is the look clips with ``codec_ok`` and ``duration_s > 0``; both floors are
    type-7 0.5th percentiles, rounded to 6 decimals so the committed file is stable.
    """
    usable = [row for row in rows if row["codec_ok"] and (row["duration_s"] or 0) > 0]
    if not usable:
        raise ValueError("no decodable Tarteel look clip: the silence floors are undefined")
    ratios = np.asarray([row["speech_ratio"] for row in usable], dtype=float)
    seconds = np.asarray([row["speech_s"] for row in usable], dtype=float)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"dataset": "tarteel_v1", "split": "look", "clips": look_clips,
                   "decodes_sha256": decodes_sha256},
        "look_split": {"path": _repo_relative(SPLIT_PATH),
                       "sha256": sha256_file(SPLIT_PATH)},
        "percentile": {"method": "type-7", "q": 0.005},
        "vad": VAD_CONFIG,
        "count": len(usable),
        "f_s": round(_quantile(ratios, 0.005), 6),
        "d_s": round(_quantile(seconds, 0.005), 6),
    }


def build_floors(shards: Iterable[str], *, look_ids: Iterable[str], cache_dir: Path,
                 speech_seconds_of: Callable[[np.ndarray], float],
                 stream=None) -> dict:
    """Measure only the Tarteel look clips among ``shards`` and return the floors record.

    Every other row is skipped without decoding. The shards are deleted once read, and the run
    block records the look shards' download size and the peak scratch disk they needed.
    """
    shards = list(shards)
    look = frozenset(look_ids)
    timings = _timings()
    begin, record = _download_meter(timings)
    facts: dict[str, dict] = {}
    for shard in shards:
        if stream is None:
            begin()
            rows = shard_rows(shard, cache_dir, dataset=TARTEEL_V1, on_download=record)
        else:
            rows = stream(shard, cache_dir)
        for row in rows:
            path = row["audio"]["path"]
            if path not in look:
                continue
            facts[path] = measure(row["audio"]["bytes"], speech_seconds_of,
                                  timings=timings).facts
        timings["peak_disk_bytes"] = max(timings["peak_disk_bytes"], timings["shard_bytes"])
    missing = sorted(look - facts.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(look)} Tarteel look clips were not read "
            f"(first: {missing[0]!r}): the floors need the whole frozen split")
    floors = silence_floors(facts.values(), look_clips=len(look),
                            decodes_sha256=sha256_file(DECODES_PATH))
    floors["run"] = {
        "shards": len(shards),
        "download_bytes": timings["download_bytes"],
        "download_seconds": round(timings["download_seconds"], 3),
        "peak_disk_bytes": timings["peak_disk_bytes"],
    }
    return floors


def require_floor_config(floors: dict, config: dict | None = None) -> None:
    """Refuse a run whose VAD config is not the one the committed floors measured."""
    expected = VAD_CONFIG if config is None else config
    if floors.get("vad") != expected:
        raise ValueError(
            f"VAD config mismatch: the floors record {floors.get('vad')!r}, "
            f"pass 1 would run {expected!r}")


def _provenance(code_sha: str | None, pull_request: int | None) -> dict:
    """The commit a measurement ran from, and the PR ref that keeps it after the squash merge."""
    return {"code_sha": code_sha or head_sha(), "pull_request": pull_request,
            "pull_request_ref": (f"refs/pull/{pull_request}/head"
                                 if pull_request is not None else None)}


def run(shards: Iterable[str], *, out_dir: Path, cache_dir: Path,
        speech_seconds_of: Callable[[np.ndarray], float], floors_path: Path = FLOORS_PATH,
        fp_dir: Path | None = None, stream=None, code_sha: str | None = None,
        pull_request: int | None = None) -> dict:
    """Process the shards in order, write each output, and return the ``meta.json`` record."""
    shards = list(shards)
    floors_path = Path(floors_path)
    require_floor_config(json.loads(floors_path.read_text(encoding="utf-8")))
    out_dir = Path(out_dir)
    timings = _timings()
    by_split = {split: _timings() for split in TLOG_SPLITS}
    begin, record = _download_meter(timings)
    counts: Counter = Counter()
    for shard in shards:
        split, stem = tlog_split(shard), Path(shard).stem
        shard_timings = _timings()
        if stream is None:
            begin()
            rows = process_shard(shard, cache_dir, speech_seconds_of=speech_seconds_of,
                                 fp_dir=fp_dir, on_download=record, timings=shard_timings)
        else:
            rows = process_shard(shard, cache_dir, speech_seconds_of=speech_seconds_of,
                                 fp_dir=fp_dir, stream=stream, timings=shard_timings)
        _accumulate(timings, shard_timings)
        _accumulate(by_split[split], shard_timings)
        path = out_dir / split / f"{stem}.jsonl.gz"
        write_pass1(rows, path)
        counts[split] += len(rows)
        fp_path = Path(fp_dir) / f"{stem}.npz" if fp_dir is not None else None
        fp_bytes = (fp_path.stat().st_size
                    if fp_path is not None and fp_path.exists() else 0)
        # The shard blob is deleted before the npz is written, so these never coexist.
        timings["peak_disk_bytes"] = max(timings["peak_disk_bytes"],
                                         timings["shard_bytes"], fp_bytes,
                                         path.stat().st_size)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repo": TLOG.repo, "revision": TLOG.revision, "licence": TLOG.licence,
            "splits": {split: sum(1 for shard in TLOG.shards if tlog_split(shard) == split)
                       for split in TLOG_SPLITS},
        },
        **_provenance(code_sha, pull_request),
        "vad": VAD_CONFIG,
        "decoders": decoder_versions(),
        "embeddings": "none (owner fallback 2026-10-24, #187)",
        "floors": {"path": _repo_relative(floors_path),
                   "sha256": sha256_file(floors_path)},
        "counts": {"rows": {**{split: counts[split] for split in TLOG_SPLITS},
                            "total": sum(counts.values())}},
        "throughput": _throughput(timings, by_split, len(shards)),
    }
    write_text_atomically(
        out_dir / "meta.json",
        json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return meta


def silero_speech_seconds() -> Callable[[np.ndarray], float]:
    """The pinned Silero VAD as a 16 kHz waveform -> speech-seconds callable (ONNX, CPU).

    The package is imported here so the offline tests need neither it nor torch. Its installed
    version must be :data:`SILERO_VAD_VERSION`: a different VAD would move the floors.
    """
    from importlib.metadata import PackageNotFoundError, version

    from silero_vad import get_speech_timestamps, load_silero_vad

    try:
        installed = version("silero-vad")
    except PackageNotFoundError:
        raise RuntimeError(
            f"silero-vad is not installed; install silero-vad=={SILERO_VAD_VERSION}") from None
    if installed != SILERO_VAD_VERSION:
        raise RuntimeError(
            f"silero-vad {installed} is installed; the pinned VAD is {SILERO_VAD_VERSION}")
    model = load_silero_vad(onnx=True)  # 6.x's ONNX wrapper is CPU-forced internally

    def speech_seconds_of(waveform: np.ndarray) -> float:
        import torch

        stamps = get_speech_timestamps(
            torch.from_numpy(np.asarray(waveform, dtype=np.float32)), model,
            sampling_rate=VAD_CONFIG["sample_rate"],
            threshold=VAD_CONFIG["threshold"],
            min_speech_duration_ms=VAD_CONFIG["min_speech_duration_ms"],
            min_silence_duration_ms=VAD_CONFIG["min_silence_duration_ms"],
            speech_pad_ms=VAD_CONFIG["speech_pad_ms"],
            return_seconds=False)
        return sum(stamp["end"] - stamp["start"] for stamp in stamps) / TARGET_SAMPLE_RATE

    return speech_seconds_of


def _repo_relative(path: Path) -> str:
    """``path`` inside the checkout as a repo-relative string; a file outside it unchanged."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _quantile(values: np.ndarray, q: float) -> float:
    from training.acceptance_stats import quantile_type7

    return float(quantile_type7(values, q))


def _timings() -> dict:
    return {"download_seconds": 0.0, "download_bytes": 0, "decode_seconds": 0.0,
            "vad_seconds": 0.0, "fingerprint_seconds": 0.0, "fingerprint_bytes": 0,
            "rows": 0, "vad_rows": 0, "fingerprint_rows": 0,
            "shard_bytes": 0, "peak_disk_bytes": 0}


#: The per-shard stage counters :func:`run` adds into the run and per-split timings. Download
#: and peak disk are tracked once for the run, never per shard.
_STAGE_KEYS = ("decode_seconds", "vad_seconds", "fingerprint_seconds", "fingerprint_bytes",
               "rows", "vad_rows", "fingerprint_rows")


def _accumulate(total: dict, part: dict) -> None:
    for key in _STAGE_KEYS:
        total[key] += part[key]


def _download_meter(timings: dict):
    """``(begin, record)``: ``begin`` starts a shard's download clock, ``record`` adds its time
    and on-disk size (the ``iter_rows`` hook), so no shard is downloaded twice."""
    state = {"start": 0.0}

    def begin() -> None:
        state["start"] = time.perf_counter()

    def record(path: str) -> None:
        timings["download_seconds"] += time.perf_counter() - state["start"]
        size = os.path.getsize(os.path.realpath(path))
        timings["download_bytes"] += size
        timings["shard_bytes"] = size

    return begin, record


def _throughput(timings: dict, by_split: dict[str, dict], shard_count: int) -> dict:
    """Measured stage rates, and the corpus extrapolation priced from the measurements.

    Compute is scaled by each split's own measured per-row cost to that split's known row
    population (:data:`ROWS_BY_SPLIT`), never by the tracer's shard count: the tracer's shards
    hold 2,405 rows, not 743 equal shards. A split with no measured row borrows the pooled
    per-row cost. Download is the corpus' 313 GiB at the measured rate, not a per-shard guess.
    """
    rows = timings["rows"]

    def rate(stage_rows: int, seconds: float) -> float | None:
        return round(stage_rows / seconds, 3) if seconds > 0 and stage_rows else None

    def stage_seconds(stage: dict) -> float:
        return (stage["decode_seconds"] + stage["vad_seconds"]
                + stage["fingerprint_seconds"])

    pooled_per_row = stage_seconds(timings) / rows if rows else 0.0
    cpu_hours = 0.0
    for split in TLOG_SPLITS:
        measured = by_split[split]
        split_rows = measured["rows"]
        per_row = (stage_seconds(measured) / split_rows if split_rows
                   else pooled_per_row)
        cpu_hours += per_row * ROWS_BY_SPLIT[split] / 3600
    download_rate = (timings["download_bytes"] / timings["download_seconds"]
                     if timings["download_seconds"] > 0 else None)
    download_hours = (TOTAL_AUDIO_BYTES / download_rate / 3600
                      if download_rate else None)
    wall_hours = cpu_hours + (download_hours or 0.0)
    # Pooled over every tracer row, unlike the per-split CPU scaling above: rows without audio
    # contribute 0 bytes and are still counted, so this is not a per-split weighted figure.
    store_bytes = (timings["fingerprint_bytes"] / rows * TOTAL_ROWS) if rows else 0
    within = (download_rate is not None and wall_hours <= WALL_ENVELOPE_HOURS
              and timings["peak_disk_bytes"] <= DISK_ENVELOPE_BYTES)
    return {
        "workers": 1,
        "shards_measured": shard_count,
        "rows": rows,
        "rows_by_split": {split: by_split[split]["rows"] for split in TLOG_SPLITS},
        "download_bytes": timings["download_bytes"],
        "download_seconds": round(timings["download_seconds"], 3),
        "download_mb_s": (round(timings["download_bytes"] / 1e6
                                / timings["download_seconds"], 3)
                          if timings["download_seconds"] > 0 else None),
        "decode_seconds": round(timings["decode_seconds"], 3),
        "vad_seconds": round(timings["vad_seconds"], 3),
        "fingerprint_seconds": round(timings["fingerprint_seconds"], 3),
        "stage_rows": {"decode": rows, "vad": timings["vad_rows"],
                       "fingerprint": timings["fingerprint_rows"]},
        "rows_per_s": {
            "decode": rate(rows, timings["decode_seconds"]),
            "vad": rate(timings["vad_rows"], timings["vad_seconds"]),
            "fingerprint": rate(timings["fingerprint_rows"],
                                timings["fingerprint_seconds"]),
        },
        "gpu_seconds_per_1000_rows": 0.0,
        "gpu_note": "VAD and fingerprint are CPU (owner decision 2026-10-24)",
        "peak_disk_bytes": timings["peak_disk_bytes"],
        "fingerprint_store_bytes": round(store_bytes),
        "extrapolated": {
            "shards": TOTAL_SHARDS,
            "rows": TOTAL_ROWS,
            "rows_by_split": dict(ROWS_BY_SPLIT),
            "cpu_hours": round(cpu_hours, 3),
            "download_hours": (round(download_hours, 3)
                               if download_hours is not None else None),
            "wall_hours": round(wall_hours, 3),
            "download_priced": download_rate is not None,
            "gpu_hours": 0.0,
            "peak_disk_bytes": timings["peak_disk_bytes"],
            "envelope": {"wall_hours_max": WALL_ENVELOPE_HOURS,
                         "gpu_hours_max": GPU_ENVELOPE_HOURS,
                         "peak_disk_bytes_max": DISK_ENVELOPE_BYTES},
            "within_owner_envelope": within,
        },
    }


# --- the commands ----------------------------------------------------------------------------


def _require_pinned_pyav() -> None:
    """The fallback's installed version must be the pin: a bump can move decoded samples."""
    installed = require_pyav()
    if installed != PYAV_PIN:
        raise RuntimeError(
            f"av {installed} is installed; the pinned container fallback is av=={PYAV_PIN}")


def _floors_command(args) -> None:
    require_clean_tree()
    _require_pinned_pyav()
    look_ids = set(load_split()["look"])
    floors = build_floors(args.shards, look_ids=look_ids, cache_dir=args.cache_dir,
                          speech_seconds_of=silero_speech_seconds())
    floors.update(_provenance(None, args.pr))
    write_text_atomically(
        Path(args.out),
        json.dumps(floors, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: floors[key] for key in ("f_s", "d_s", "count")}), flush=True)
    print(json.dumps(floors["run"], sort_keys=True), flush=True)


def _run_command(args) -> None:
    require_clean_tree()
    _require_pinned_pyav()
    shards = tracer_shards() if args.tracer else args.shards
    meta = run(shards, out_dir=args.out_dir, cache_dir=args.cache_dir,
               floors_path=args.floors, speech_seconds_of=silero_speech_seconds(),
               fp_dir=args.fp_dir, code_sha=args.code_sha, pull_request=args.pr)
    print(json.dumps(meta["throughput"], indent=1, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    floors_command = commands.add_parser(
        "floors", help="measure Tarteel v1 look clips and write silence_floors.json")
    floors_command.add_argument("--cache-dir", type=Path, required=True,
                                help="HF cache for the look shards (each is deleted after)")
    floors_command.add_argument("--out", type=Path, default=FLOORS_PATH)
    floors_command.add_argument("--shards", nargs="+", default=TARTEEL_V1.shards)
    floors_command.add_argument("--pr", type=int, required=True,
                                help="the PR number whose refs/pull/<N>/head keeps the run's "
                                     "code_sha reachable after the squash merge")
    floors_command.set_defaults(execute=_floors_command)

    run_command = commands.add_parser(
        "run", help="measure tlog shards and write their pass-1 rows and meta.json")
    run_command.add_argument("--cache-dir", type=Path, required=True,
                             help="HF cache for the tlog shards (each is deleted after)")
    run_command.add_argument("--out-dir", type=Path, default=PASS1_DIR)
    run_command.add_argument("--fp-dir", type=Path, default=FP_DIR,
                             help="scratch for the fingerprints (never committed)")
    run_command.add_argument("--floors", type=Path, default=FLOORS_PATH)
    run_command.add_argument("--code-sha", default=None)
    run_command.add_argument("--pr", type=int, required=True,
                             help="the PR number whose refs/pull/<N>/head keeps code_sha "
                                  "reachable after the squash merge")
    which = run_command.add_mutually_exclusive_group(required=True)
    which.add_argument("--tracer", action="store_true",
                       help="the three shards the module's tracer rule names")
    which.add_argument("--shards", nargs="+", help="explicit tlog shard file names")
    run_command.set_defaults(execute=_run_command)

    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()

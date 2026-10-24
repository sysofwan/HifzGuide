"""The tlog pass-1 tracer (#187): outcome-blind facts for three shards, and the silence floors.

Pass 1 reads a shard once, records per-row facts that depend on nothing but the audio, writes
the shard's committed rows, and deletes the shard. It runs **no teacher, student or ASR model**
and produces no transcript. The committed facts are ``audio_sha256``, ``codec_ok``,
``sample_rate``, ``duration_s``, ``speech_s`` and ``speech_ratio``; the Haitsma-Kalker
fingerprint (:mod:`tadabur.qud_fingerprint`) is written to cuda-dev scratch only. The owner
accepted the clip-level fallback (2026-10-24), so no speaker embedding is computed at all.

``tools/tadabur/tlog/pass1/<split>/<shard>.jsonl.gz``
    one row per shard row: ``schema_version``, ``split``, ``shard_file``, ``row``, ``audio_path``
    and the six facts. Keys sorted, ``mtime`` 0 and written atomically, so a rerun on the build
    interpreter reproduces every byte (the gzip OS header byte is platform-dependent, as in the
    index).
``tools/tadabur/tlog/pass1/meta.json``
    the code SHA, the VAD config, the floors file's SHA-256, the row counts and the measured
    throughput. ``"embeddings"`` records the fallback rather than omitting it.
``tools/tadabur/tlog/silence_floors.json``
    the pre-registration's Exclusions floors: ``f_s`` and ``d_s`` are the type-7 0.5th
    percentiles of ``speech_ratio`` and ``speech_s`` over the Tarteel v1 look clips that decode.
    Silero VAD (:data:`SILERO_VAD_VERSION`, ONNX on CPU) supplies ``speech_s``; the padded
    Silero spans are summed. The floors commit lands **before** any pass-1 output commit, and
    :func:`run` refuses to start when its VAD config differs from the committed floors.

**Tracer rule (written before the run).** The tracer reads clean shard indices 0 and 205 (of
411) and unclean index 0 (of 332): :func:`tracer_shards`. It measures download, decode+resample,
VAD and fingerprint separately, and extrapolates the 743-shard cost from those numbers; the PR
and #173 carry the restatement. The branch is squash-merged, so a cited ``code_sha`` survives
only under ``refs/pull/<N>/head`` and is recorded with the PR number.

tlog audio is gated: the reads need a Hub token that accepted the dataset's terms and run on
cuda-dev, where the scratch directories live. Shards and their audio are never committed.

Usage (from ``tools/``, on cuda-dev; under ``flock /root/scratch/gpu.lock``)::

  python -m tadabur.tlog_pass1 floors --cache-dir /root/scratch/tlog/cache
  python -m tadabur.tlog_pass1 run --tracer --cache-dir /root/scratch/tlog/cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from training.tashkeel_eval import write_text_atomically

from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k_with_rate
from .nonpro_sources import (TARTEEL_V1, TLOG, TLOG_SPLITS, Dataset, audio_sha256, iter_rows,
                             tlog_split)
from .qud_fingerprint import fingerprint, save_fingerprints
from .tarteel_split import SPLIT_PATH, load_split
from .tlog_index import head_sha, write_shard

SCHEMA_VERSION = 1
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

#: The tracer rule, fixed before the run: clean shard indices 0 and 205, unclean 0.
TRACER_CLEAN = (0, 205)
TRACER_UNCLEAN = 0
#: The corpus' committed size (#185's index): 411 clean + 332 unclean shards, 719,853 rows.
TOTAL_SHARDS = 743
TOTAL_ROWS = 719_853


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
    exclusion), not an exception; the VAD is never called for it.
    """
    digest = audio_sha256(audio)
    started = time.perf_counter()
    try:
        waveform, sample_rate = decode_to_mono_16k_with_rate(audio)
    except (sf.LibsndfileError, OSError, ValueError):
        waveform, sample_rate = None, None
    if timings is not None:
        timings["decode_seconds"] += time.perf_counter() - started
    duration_s = len(waveform) / TARGET_SAMPLE_RATE if waveform is not None else 0.0
    if waveform is None or duration_s <= 0:
        return Measured(
            {"audio_sha256": digest, "codec_ok": False, "sample_rate": None,
             "duration_s": None, "speech_s": None, "speech_ratio": None},
            None)
    started = time.perf_counter()
    speech_s = float(speech_seconds_of(waveform))
    if timings is not None:
        timings["vad_seconds"] += time.perf_counter() - started
        timings["vad_rows"] += 1
    return Measured(
        {"audio_sha256": digest, "codec_ok": True, "sample_rate": sample_rate,
         "duration_s": duration_s, "speech_s": speech_s,
         "speech_ratio": speech_s / duration_s},
        waveform)


def shard_rows(shard: str, cache_dir: Path, *, dataset: Dataset = TLOG,
               on_download: Callable[[str], None] | None = None) -> Iterator[dict]:
    """A shard's rows, downloaded and then deleted by :func:`nonpro_sources.iter_rows`.

    The dataset guard runs before the download: pass 1 reads tlog shards only, and the floors
    read Tarteel v1 only.
    """
    if shard not in dataset.shards:
        raise ValueError(f"not a {dataset.name} shard: {shard}")
    return iter_rows(dataset, shard, cache_dir, on_download=on_download)


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
            "schema_version": SCHEMA_VERSION, "split": split, "shard_file": stem,
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


def run(shards: Iterable[str], *, out_dir: Path, cache_dir: Path,
        speech_seconds_of: Callable[[np.ndarray], float], floors_path: Path = FLOORS_PATH,
        fp_dir: Path | None = None, stream=None, code_sha: str | None = None) -> dict:
    """Process the shards in order, write each output, and return the ``meta.json`` record."""
    shards = list(shards)
    floors_path = Path(floors_path)
    require_floor_config(json.loads(floors_path.read_text(encoding="utf-8")))
    out_dir = Path(out_dir)
    timings = _timings()
    begin, record = _download_meter(timings)
    counts: Counter = Counter()
    for shard in shards:
        split, stem = tlog_split(shard), Path(shard).stem
        if stream is None:
            begin()
            rows = process_shard(shard, cache_dir, speech_seconds_of=speech_seconds_of,
                                 fp_dir=fp_dir, on_download=record, timings=timings)
        else:
            rows = process_shard(shard, cache_dir, speech_seconds_of=speech_seconds_of,
                                 fp_dir=fp_dir, stream=stream, timings=timings)
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
        "code_sha": code_sha or head_sha(),
        "vad": VAD_CONFIG,
        "embeddings": "none (owner fallback 2026-10-24, #187)",
        "floors": {"path": _repo_relative(floors_path),
                   "sha256": sha256_file(floors_path)},
        "counts": {"rows": {**{split: counts[split] for split in TLOG_SPLITS},
                            "total": sum(counts.values())}},
        "throughput": _throughput(timings, len(shards)),
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
    model = load_silero_vad(onnx=True, force_onnx_cpu=True)

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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _throughput(timings: dict, shard_count: int) -> dict:
    """The measured stage rates, and the sequential extrapolation to (:data:`TOTAL_SHARDS`)."""
    rows = timings["rows"]

    def rate(stage_rows: int, seconds: float) -> float | None:
        return round(stage_rows / seconds, 3) if seconds > 0 and stage_rows else None

    cpu_seconds = (timings["decode_seconds"] + timings["vad_seconds"]
                   + timings["fingerprint_seconds"])
    scale = TOTAL_SHARDS / shard_count if shard_count else 0.0
    cpu_hours = cpu_seconds / 3600 * scale
    download_hours = timings["download_seconds"] / 3600 * scale
    store_bytes = (timings["fingerprint_bytes"] / rows * TOTAL_ROWS) if rows else 0
    return {
        "workers": 1,
        "shards_measured": shard_count,
        "rows": rows,
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
            "cpu_hours": round(cpu_hours, 3),
            "download_hours": round(download_hours, 3),
            "wall_hours": round(cpu_hours + download_hours, 3),
            "gpu_hours": 0.0,
        },
    }


# --- the commands ----------------------------------------------------------------------------


def _floors_command(args) -> None:
    look_ids = set(load_split()["look"])
    floors = build_floors(args.shards, look_ids=look_ids, cache_dir=args.cache_dir,
                          speech_seconds_of=silero_speech_seconds())
    write_text_atomically(
        Path(args.out),
        json.dumps(floors, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: floors[key] for key in ("f_s", "d_s", "count")}), flush=True)
    print(json.dumps(floors["run"], sort_keys=True), flush=True)


def _run_command(args) -> None:
    shards = tracer_shards() if args.tracer else args.shards
    meta = run(shards, out_dir=args.out_dir, cache_dir=args.cache_dir,
               floors_path=args.floors, speech_seconds_of=silero_speech_seconds(),
               fp_dir=args.fp_dir, code_sha=args.code_sha)
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
    which = run_command.add_mutually_exclusive_group(required=True)
    which.add_argument("--tracer", action="store_true",
                       help="the three shards the module's tracer rule names")
    which.add_argument("--shards", nargs="+", help="explicit tlog shard file names")
    run_command.set_defaults(execute=_run_command)

    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()

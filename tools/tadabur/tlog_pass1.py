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

**Gate rule (written before the run, S3b).** The gate names clean shard indices 0-5 (of 411)
and unclean 0-5 (of 332): :func:`gate_shards` with :data:`GATE_CLEAN` and :data:`GATE_UNCLEAN`.
Clean 0 and unclean 0 are already committed by the tracer, so the resume guard verifies and
skips them and they contribute nothing to the measured throughput. Each split is its own timed
phase of :data:`GATE_WORKERS` workers (clean, then unclean), so each split's ``wall_seconds`` is
real elapsed wall time at that concurrency and :func:`gate_projection` does not divide by the
worker count again. The five fresh shards on three workers leave a tail, which makes the
projected rate conservative. The projection weights each split's measured rows/s by the index's
row count for the shards this run did not process; the full run starts only if it projects <=
the owner's 8 wall hours with 0 GPU hours (:data:`WALL_ENVELOPE_HOURS`).

**Worker concurrency (S3b).** The full run uses :data:`GATE_WORKERS` processes, so up to three
tlog shards are on disk at once; each worker deletes its shard before the next download. The
peak-disk check uses the sum of the largest measured shard files plus the accumulated
fingerprint store, against :data:`DISK_ENVELOPE_BYTES`.

tlog audio is gated: the reads need a Hub token that accepted the dataset's terms and run on
cuda-dev, where the scratch directories live. Shards and their audio are never committed.

Usage (from ``tools/``, on cuda-dev). VAD and fingerprinting are CPU-only, so neither command
takes the GPU lock::

  python -m tadabur.tlog_pass1 floors --cache-dir /root/scratch/tlog/cache --pr 198
  python -m tadabur.tlog_pass1 run --tracer --cache-dir /root/scratch/tlog/cache --pr 198
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
from .tlog_index import (INDEX_DIR, head_sha, require_clean_tree, sha256_file, write_shard)

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
#: The gate rule, fixed before the run: the first six shards of each split, at three workers.
#: Clean 0 and unclean 0 are the tracer's committed outputs, so the resume guard skips them.
GATE_CLEAN = (0, 1, 2, 3, 4, 5)
GATE_UNCLEAN = (0, 1, 2, 3, 4, 5)
GATE_WORKERS = 3
#: The S3a tracer's committed rows were produced by this commit in PR 198; a resumed run
#: verifies them but must cite their real producer, not the S3b code that skipped them. The SHA
#: survives only under ``refs/pull/198/head`` because that branch was squash-merged.
S3A_CODE_SHA = "d2af0e3abefee2f331c731e435755ffbf422604d"
S3A_PULL_REQUEST = 198
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


def gate_shards() -> tuple[str, ...]:
    """The gate's 12 shards, by the rule in the module docstring (both splits, six each)."""
    clean = [shard for shard in TLOG.shards if tlog_split(shard) == "clean"]
    unclean = [shard for shard in TLOG.shards if tlog_split(shard) == "unclean"]
    return (tuple(clean[index] for index in GATE_CLEAN)
            + tuple(unclean[index] for index in GATE_UNCLEAN))


#: The three tracer shards' real producer, keyed as ``<split>/<stem>`` for a run's meta.json.
S3A_PROVENANCE = {
    f"{tlog_split(shard)}/{Path(shard).stem}":
        {"code_sha": S3A_CODE_SHA, "pull_request": S3A_PULL_REQUEST}
    for shard in tracer_shards()}


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


def exclusion_reasons(row: Mapping[str, object], floors: Mapping[str, float]) -> tuple[str, ...]:
    """The pre-registration's floor exclusions for one pass-1 row.

    A ``codec_ok: false`` row has null duration, speech and ratio; it is excluded as
    ``"codec"`` before any floor is compared, so no null arithmetic is attempted. Every other
    row is ``"silence"`` when its speech ratio is below ``f_s`` or its speech seconds below
    ``d_s``. The comparison is inclusive: a row exactly on both floors is usable. Later slices
    read pass-1 rows through this helper rather than re-deriving the floors.
    """
    if not row["codec_ok"]:
        return ("codec",)
    if (row["speech_ratio"] < floors["f_s"]) or (row["speech_s"] < floors["d_s"]):
        return ("silence",)
    return ()


def usable(row: Mapping[str, object], floors: Mapping[str, float]) -> bool:
    """Whether a pass-1 row survives the pre-registration's floors (see :func:`exclusion_reasons`)."""
    return not exclusion_reasons(row, floors)


def _provenance(code_sha: str | None, pull_request: int | None) -> dict:
    """The commit a measurement ran from, and the PR ref that keeps it after the squash merge."""
    return {"code_sha": code_sha or head_sha(), "pull_request": pull_request,
            "pull_request_ref": (f"refs/pull/{pull_request}/head"
                                 if pull_request is not None else None)}


def require_pass1_alignment(rows: Sequence[Mapping[str, object]],
                            index_rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Refuse pass-1 rows that are not exactly the index shard's rows.

    The row count, the row ordinals and the ``audio_path`` sequence must all match the S1
    index shard. Both the resume guard and the split freeze use this one rule, so a freeze can
    never be built from a shard that a resumed run would have refused.
    """
    if len(rows) != len(index_rows):
        raise ValueError(f"pass 1 output {path} has row count {len(rows)}; "
                         f"the index shard has {len(index_rows)}")
    if [row.get("row") for row in rows] != list(range(len(index_rows))):
        raise ValueError(f"pass 1 output {path} does not carry row ordinals 0..n-1")
    if [row.get("audio_path") for row in rows] != [row["audio_path"] for row in index_rows]:
        raise ValueError(f"pass 1 output {path} has an audio_path sequence that differs "
                         f"from the index shard")


def verify_pass1_shard(path: Path, index_path: Path) -> int:
    """Verify a committed or scratch pass-1 file against its S1 index shard, before skipping it.

    Reading the gzip proves its integrity; :func:`require_pass1_alignment` checks the row
    count, ordinals and ``audio_path`` sequence. Any drift raises ``ValueError``: a resumed run
    never re-downloads a shard whose facts are not the frozen index's rows. Returns the
    verified row count.
    """
    index_rows = read_jsonl_gz(index_path)
    rows = read_jsonl_gz(path)
    require_pass1_alignment(rows, index_rows, path)
    return len(rows)


def read_jsonl_gz(path: Path) -> list[dict]:
    """Every row of a committed JSONL-gzip file; a corrupt file is an error, never a skip."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable JSONL-gzip file: {exc}") from exc


def gate_projection(measured: Mapping[str, Mapping[str, float]], *,
                    remaining_rows: Mapping[str, int], workers: int) -> dict:
    """Project the unprocessed corpus from the gate's own timed, per-split phases.

    ``measured[split]`` holds ``rows`` and the phase's real ``wall_seconds`` at ``workers``
    concurrency (decision 7), so the rates already include the workers and are never divided by
    them again. A split with no measured row borrows the pooled per-row wall. The remaining
    rows are the index's counts for the shards this run did not process, weighted by each
    split's own rate. The 8 wall-hour gate passes only when the projection fits; the run is
    CPU-only, so GPU hours are zero by construction.
    """
    measured_rows = {split: int(measured.get(split, {}).get("rows", 0))
                     for split in TLOG_SPLITS}
    if not any(measured_rows.values()):
        raise ValueError("the gate has no measured row to project from")
    pooled = (sum(float(measured[split]["wall_seconds"]) for split in TLOG_SPLITS
                  if measured_rows[split])
              / sum(measured_rows.values()))
    per_row: dict[str, float] = {}
    rates: dict[str, float | None] = {}
    for split in TLOG_SPLITS:
        if measured_rows[split]:
            per_row[split] = float(measured[split]["wall_seconds"]) / measured_rows[split]
            rates[split] = round(1 / per_row[split], 3)
        else:
            per_row[split] = pooled
            rates[split] = None
    wall_hours = (sum(int(remaining_rows.get(split, 0)) * per_row[split]
                      for split in TLOG_SPLITS) / 3600)
    return {
        "workers": workers,
        "rows_measured": measured_rows,
        "rows_remaining": {split: int(remaining_rows.get(split, 0)) for split in TLOG_SPLITS},
        "rows_per_s": rates,
        "wall_hours": wall_hours,
        "gpu_hours": 0.0,
        "within_owner_envelope": wall_hours <= WALL_ENVELOPE_HOURS,
    }


def run(shards: Iterable[str], *, out_dir: Path, cache_dir: Path,
        speech_seconds_of: Callable[[np.ndarray], float], floors_path: Path = FLOORS_PATH,
        fp_dir: Path | None = None, stream=None, code_sha: str | None = None,
        pull_request: int | None = None, workers: int = 1,
        index_dir: Path = INDEX_DIR,
        shard_provenance: Mapping[str, Mapping[str, object]] | None = None) -> dict:
    """Process the shards split phase by split phase, verify or measure each, and write meta.

    Each split is one timed phase, so its ``wall_seconds`` is real elapsed time at ``workers``
    concurrency and the projection never re-divides by the worker count (decision 7). A shard
    whose committed or scratch output exists is verified against ``index_dir``'s shard and
    skipped without a download; its rows count in the freeze, not in the gate's rates. Fresh
    shards are measured (serially, or in ``workers`` processes) and written atomically.
    ``shard_provenance`` overrides a shard's cited producer, which is how the three S3a tracer
    shards keep ``d2af0e3``/``#198`` after S3b verifies instead of re-running them.
    """
    shards = list(shards)
    floors_path = Path(floors_path)
    require_floor_config(json.loads(floors_path.read_text(encoding="utf-8")))
    out_dir, index_dir = Path(out_dir), Path(index_dir)
    provenance = S3A_PROVENANCE if shard_provenance is None else shard_provenance
    timings = _timings()
    by_split = {split: _timings() for split in TLOG_SPLITS}
    counts: Counter = Counter()
    skipped = 0
    measured_shards = 0
    shard_sizes: list[int] = []
    for split in TLOG_SPLITS:
        phase = [shard for shard in shards if tlog_split(shard) == split]
        if not phase:
            continue
        started = time.perf_counter()
        fresh: list[str] = []
        for shard in phase:
            path = out_dir / split / f"{Path(shard).stem}.jsonl.gz"
            if path.exists():
                counts[split] += verify_pass1_shard(
                    path, index_dir / split / f"{Path(shard).stem}.jsonl.gz")
                skipped += 1
            else:
                fresh.append(shard)
        _measure_phase(fresh, out_dir=out_dir, cache_dir=cache_dir, fp_dir=fp_dir,
                       speech_seconds_of=speech_seconds_of, stream=stream, workers=workers,
                       timings=timings, by_split=by_split[split], counts=counts,
                       shard_sizes=shard_sizes)
        measured_shards += len(fresh)
        by_split[split]["wall_seconds"] = time.perf_counter() - started
        timings["wall_seconds"] += by_split[split]["wall_seconds"]

    measured_rows = sum(by_split[split]["rows"] for split in TLOG_SPLITS)
    remaining = {split: ROWS_BY_SPLIT[split] - counts[split] for split in TLOG_SPLITS}
    projection = (gate_projection(
        {split: {"rows": by_split[split]["rows"],
                 "wall_seconds": by_split[split]["wall_seconds"]}
         for split in TLOG_SPLITS},
        remaining_rows=remaining, workers=workers) if measured_rows else None)
    # Up to ``workers`` shard blobs coexist; the fingerprint store and the committed outputs
    # accumulate, so the bound adds them to the largest concurrent shard bytes.
    output_bytes = sum(
        (out_dir / tlog_split(shard) / f"{Path(shard).stem}.jsonl.gz").stat().st_size
        for shard in shards)
    concurrent = sum(sorted(shard_sizes, reverse=True)[:max(workers, 1)])
    timings["peak_disk_bytes"] = max(timings["peak_disk_bytes"],
                                     concurrent + timings["fingerprint_bytes"] + output_bytes)
    current = _provenance(code_sha, pull_request)
    shard_records = {
        f"{tlog_split(shard)}/{Path(shard).stem}": dict(provenance.get(
            f"{tlog_split(shard)}/{Path(shard).stem}", current))
        for shard in shards}
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
        "shards": shard_records,
        "throughput": {**_throughput(timings, by_split, measured_shards, workers=workers),
                       "shards_skipped": skipped,
                       "gate_projection": projection},
    }
    write_text_atomically(
        out_dir / "meta.json",
        json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return meta


_WORKER_SPEECH: Callable[[np.ndarray], float] | None = None


def _init_worker() -> None:
    """One Silero VAD per process, and one torch thread: workers share the box, not threads."""
    global _WORKER_SPEECH

    import torch

    torch.set_num_threads(1)
    _WORKER_SPEECH = silero_speech_seconds()


def _worker_shard(task: tuple[str, Path, Path | None]) -> tuple[list[dict], dict]:
    """Measure one shard in a worker process, with its own download clock and timings."""
    shard, cache_dir, fp_dir = task
    timings = _timings()
    begin, record = _download_meter(timings)
    begin()
    rows = process_shard(shard, cache_dir, speech_seconds_of=_WORKER_SPEECH,
                         fp_dir=fp_dir, on_download=record, timings=timings)
    return rows, timings


def _measure_phase(shards: list[str], *, out_dir: Path, cache_dir: Path,
                   fp_dir: Path | None, speech_seconds_of: Callable[[np.ndarray], float],
                   stream, workers: int, timings: dict, by_split: dict, counts: Counter,
                   shard_sizes: list[int]) -> None:
    """Measure one split's fresh shards, serially or in ``workers`` processes."""
    if not shards:
        return
    if stream is not None or workers <= 1:
        for shard in shards:
            shard_timings = _timings()
            if stream is None:
                begin, record = _download_meter(shard_timings)
                begin()
                rows = process_shard(shard, cache_dir, speech_seconds_of=speech_seconds_of,
                                     fp_dir=fp_dir, on_download=record,
                                     timings=shard_timings)
            else:
                rows = process_shard(shard, cache_dir, speech_seconds_of=speech_seconds_of,
                                     fp_dir=fp_dir, stream=stream, timings=shard_timings)
            _write_measured(shard, rows, shard_timings, out_dir=out_dir, timings=timings,
                            by_split=by_split, counts=counts, shard_sizes=shard_sizes)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        queue = iter(shards)
        pending: dict = {}

        def fill() -> None:
            while len(pending) < workers * 2:
                try:
                    shard = next(queue)
                except StopIteration:
                    return
                pending[pool.submit(_worker_shard, (shard, cache_dir, fp_dir))] = shard

        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                shard = pending.pop(future)
                rows, shard_timings = future.result()
                _write_measured(shard, rows, shard_timings, out_dir=out_dir, timings=timings,
                                by_split=by_split, counts=counts, shard_sizes=shard_sizes)
            fill()


def _write_measured(shard: str, rows: list[dict], shard_timings: dict, *, out_dir: Path,
                    timings: dict, by_split: dict, counts: Counter,
                    shard_sizes: list[int]) -> None:
    """Write one measured shard and add its stage, download and size counters to the run."""
    split, stem = tlog_split(shard), Path(shard).stem
    _accumulate(timings, shard_timings)
    _accumulate(by_split, shard_timings)
    timings["download_seconds"] += shard_timings["download_seconds"]
    timings["download_bytes"] += shard_timings["download_bytes"]
    write_pass1(rows, out_dir / split / f"{stem}.jsonl.gz")
    counts[split] += len(rows)
    shard_sizes.append(shard_timings["shard_bytes"])


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
            "shard_bytes": 0, "peak_disk_bytes": 0, "wall_seconds": 0.0}


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


def _throughput(timings: dict, by_split: dict[str, dict], shard_count: int, *,
                workers: int = 1) -> dict:
    """Measured stage rates, and the corpus extrapolation priced from the measurements.

    Compute is scaled by each split's own measured per-row cost to that split's known row
    population (:data:`ROWS_BY_SPLIT`), never by the tracer's shard count: the tracer's shards
    hold 2,405 rows, not 743 equal shards. A split with no measured row borrows the pooled
    per-row cost. Download is the corpus' 313 GiB at the measured rate, not a per-shard guess.
    The stage model is a single worker's view; the gate's decision uses
    :func:`gate_projection`, whose per-split ``wall_seconds`` already include ``workers``.
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
        "workers": workers,
        "shards_measured": shard_count,
        "rows": rows,
        "rows_by_split": {split: by_split[split]["rows"] for split in TLOG_SPLITS},
        "wall_seconds": {split: round(by_split[split].get("wall_seconds", 0.0), 3)
                         for split in TLOG_SPLITS},
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
    if getattr(args, "gate", False):
        shards = gate_shards()
    elif getattr(args, "tracer", False):
        shards = tracer_shards()
    elif getattr(args, "all", False):
        shards = TLOG.shards
    else:
        shards = args.shards
    meta = run(shards, out_dir=args.out_dir, cache_dir=args.cache_dir,
               floors_path=args.floors, speech_seconds_of=silero_speech_seconds(),
               fp_dir=args.fp_dir, code_sha=args.code_sha, pull_request=args.pr,
               workers=getattr(args, "workers", 1),
               index_dir=getattr(args, "index_dir", INDEX_DIR))
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
    run_command.add_argument("--index-dir", type=Path, default=INDEX_DIR,
                             help="the S1 index the resume guard verifies against")
    run_command.add_argument("--workers", type=int, default=1,
                             help="shard processes; the gate and full run use %d"
                                  % GATE_WORKERS)
    run_command.add_argument("--pr", type=int, required=True,
                             help="the PR number whose refs/pull/<N>/head keeps code_sha "
                                  "reachable after the squash merge")
    which = run_command.add_mutually_exclusive_group(required=True)
    which.add_argument("--tracer", action="store_true",
                       help="the three shards the module's tracer rule names")
    which.add_argument("--gate", action="store_true",
                       help="the twelve shards the module's gate rule names (S3b)")
    which.add_argument("--all", action="store_true", help="every tlog shard")
    which.add_argument("--shards", nargs="+", help="explicit tlog shard file names")
    run_command.set_defaults(execute=_run_command)

    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()

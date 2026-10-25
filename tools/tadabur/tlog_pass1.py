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
``<out-dir>/gate_phases.json``
    a gate run's settled per-split phase measurements: the shards each phase measured, its
    per-split timings and per-shard byte sizes, bound to the tranche, the worker count and the
    code SHA. Written after each phase, so an interrupted gate keeps the split it measured.
    Committed beside the pass-1 rows with ``gate_projection.json``: together they are the
    envelope evidence.
``<out-dir>/gate_projection.json``
    written by a gate run only: the projection for every row the gate did not process, the
    measured disk bound, the owner-envelope verdict (wall hours **and** peak disk) and the code
    SHA with its PR. It is its own file because the full run reuses the gate's scratch directory
    and would otherwise overwrite the gate's one measured number; the committed pass-1 tree
    carries it, so the envelope can be re-checked without the box.
``<out-dir>/gate_override.json``
    written by a full run the owner raised the wall limit for, before it measures the first
    shard: the limit in force, the comment that grants it, and the gate's measured hours, GPU
    hours and peak disk. The gate's own ``gate_projection.json`` keeps the pre-registered
    verdict, so the record still shows what was measured rather than what was allowed.
``<out-dir>/producers.json``
    the producing commit of every shard in the out-dir, written as each shard is published (and
    when a shard is seeded or resumed), so an interrupted run still attributes the rows it
    wrote and a resumed run never re-attributes them to its own commit.
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
Clean 0 and unclean 0 are already committed by the tracer: ``--seed-dir`` (the checkout's
``tools/tadabur/tlog/pass1``) verifies them against the S1 index and copies them into
``--out-dir``, so the resume guard skips them instead of downloading them again. The throughput
sample is therefore clean 1-5 and unclean 1-5 (:func:`gate_measured_shards`), each split in its
own timed phase of :data:`GATE_WORKERS` workers (clean, then unclean), so each split's
``wall_seconds`` is real elapsed wall time at that concurrency and :func:`gate_projection` does
not divide by the worker count again. The five fresh shards on three workers leave a tail, which
makes the projected rate conservative. Each phase is settled in ``gate_phases.json`` as soon as
it finishes: a retry reuses a stored phase only when it measured the same sample at the same
concurrency and code, and a split whose sample this attempt did not measure completely and no
stored phase covers makes the gate refuse — the pooled fallback would price the other split's
work. The projection weights each split's measured rows/s by the index's row count for the
shards this run did not process; the full run starts only if it projects <= the owner's 8 wall
hours with 0 GPU hours (:data:`WALL_ENVELOPE_HOURS`) **and** the measured peak disk is within
:data:`DISK_ENVELOPE_BYTES` (:func:`gate_verdict`). An owner may raise the wall limit for one
run by citing the comment that grants it (``--owner-wall-hours`` with ``--owner-decision``):
:func:`require_gate_pass` then allows the full run when the projected hours fit the raised
limit and the disk and GPU bounds still hold, and never rewrites the gate's verdict — the
raised limit and its citation are written into ``meta.json`` and ``gate_override.json``
instead.

**Resume guard (S3b).** A shard whose pass-1 file is already in ``--out-dir`` is verified
against the S1 index — gzip integrity, row count, ordinals, ``audio_path`` sequence — and
skipped, never re-downloaded. ``--seed-dir`` also seeds an empty scratch directory from a tree
of committed pass-1 files: each seed is verified the same way before it is copied into place,
so the gate measures only the shards neither the tracer nor an earlier run produced. A freshly
measured shard is verified against the index before it is written, so a drifted read fails at
that shard instead of at the freeze.

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
import math
import os
import shutil
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
#: The owner's S3b envelope: wall hours, GPU hours and peak disk bytes. The pass is CPU-only
#: (decode, VAD, fingerprint) and takes no GPU lock, so its GPU allowance is zero — the issue's
#: earlier "<= 3 GPU-h" priced the models S2 would have run, which the fallback removed. The
#: owner's 12 h grace raised the wall bound only; disk and GPU stay as they are.
WALL_ENVELOPE_HOURS = 8.0
GPU_ENVELOPE_HOURS = 0.0
DISK_ENVELOPE_BYTES = 5_000_000_000
#: The out-dir's own records: each published shard's producer, and the gate's settled phases.
PRODUCERS_NAME = "producers.json"
GATE_PHASES_NAME = "gate_phases.json"
GATE_PROJECTION_NAME = "gate_projection.json"
#: A full run the owner raised the wall limit for records the limit and the comment granting it.
GATE_OVERRIDE_NAME = "gate_override.json"


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


def gate_measured_shards() -> dict[str, tuple[str, ...]]:
    """The shards each split's throughput sample is measured on: clean 1-5, unclean 1-5.

    Decision 3 excludes the tracer's committed shards (clean 0, unclean 0) from the gate's
    rate: the seed supplies them and the run only verifies them. A gate whose phase measured
    anything other than exactly this sample has no measurement it may project from.
    """
    committed = set(tracer_shards())
    return {split: tuple(shard for shard in gate_shards()
                         if tlog_split(shard) == split and shard not in committed)
            for split in TLOG_SPLITS}


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

    These are the two **floor** exclusions only. The pre-registration's set also has
    ``eval_overlap:<set>`` and ``duplicate``: the first needs the S1 index row (the freeze
    reports its scope in ``split/meta.json``'s ``exclusions``) and the second needs S4's
    fingerprints, so neither is decided here. A training pool must apply both before it trains.
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


def _seed_pass1(seed: Path, target: Path, index_path: Path) -> None:
    """Copy a committed pass-1 shard into the run's scratch, so the guard skips it.

    The seed is verified against the S1 index before the copy, so a drifted committed file
    never becomes this run's resume point. An absent seed is not an error: the shard is simply
    fresh, and the caller measures it.
    """
    if not seed.exists():
        return
    verify_pass1_shard(seed, index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    shutil.copyfile(seed, partial)
    os.replace(partial, target)


def scratch_producers(out_dir: Path) -> dict[str, dict]:
    """The producing commit of every shard an out-dir has published, keyed ``<split>/<stem>``."""
    path = Path(out_dir) / PRODUCERS_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def committed_producers(seed_dir: Path | None) -> dict[str, dict]:
    """The producing commits a committed pass-1 tree's ``meta.json`` records (may be none)."""
    if seed_dir is None:
        return {}
    meta_path = Path(seed_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8")).get("shards", {})


def _record_producer(path: Path, key: str, producer: Mapping[str, object]) -> None:
    """Record one shard's producer durably, so an interrupted run still attributes its rows."""
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record[key] = dict(producer)
    write_text_atomically(path, json.dumps(record, indent=1, ensure_ascii=False,
                                           sort_keys=True) + "\n")


def _known_producer(key: str, path: Path, known: Mapping[str, Mapping[str, object]]) -> dict:
    """An existing pass-1 file's recorded producer; never this run, which did not write it."""
    try:
        return dict(known[key])
    except KeyError:
        raise ValueError(
            f"no producing record for {key} ({path}): neither the out-dir's {PRODUCERS_NAME} "
            f"nor the seed's meta.json names the commit that measured these rows, and this run "
            f"only verified them; delete the file so the shard is measured again, or seed it "
            f"from a committed pass-1 tree") from None


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


def gate_verdict(projection: Mapping[str, object], *, peak_disk_bytes: int,
                 phases: Mapping[str, Mapping[str, object]]) -> dict:
    """The gate's published verdict: the projection, the measured disk bound and both envelopes.

    The projection prices time only; the owner's envelope is wall hours **and** peak disk, so a
    fast gate over the disk budget does not pass. ``extrapolated.within_owner_envelope`` in
    ``meta.json`` stays the separate single-worker stage model and is not this verdict.
    ``phases`` names the shards whose timed phase each split's rate came from, so the artifact
    shows the sample it projected from.
    """
    within_wall = bool(projection["within_owner_envelope"])
    within_disk = int(peak_disk_bytes) <= DISK_ENVELOPE_BYTES
    return {
        **projection,
        "phases": {split: sorted(phase["shards"]) for split, phase in phases.items()},
        "peak_disk_bytes": int(peak_disk_bytes),
        "envelope": {"wall_hours_max": WALL_ENVELOPE_HOURS,
                     "gpu_hours_max": GPU_ENVELOPE_HOURS,
                     "peak_disk_bytes_max": DISK_ENVELOPE_BYTES},
        "within_wall_envelope": within_wall,
        "within_disk_envelope": within_disk,
        "within_owner_envelope": within_wall and within_disk,
    }


def require_gate_pass(verdict: Mapping[str, object], *, owner_wall_hours: float | None = None,
                      owner_decision: str | None = None) -> dict | None:
    """The full run's preconditions, and the override record when the owner raised the limit.

    ``verdict`` is the gate's published ``gate_projection.json``, never rewritten: a projection
    over the pre-registered :data:`WALL_ENVELOPE_HOURS` keeps ``within_owner_envelope: false``
    there even when the owner grants more wall time. The full run starts when the measured peak
    disk and GPU hours fit their envelopes and the projected wall hours fit the limit in force —
    the pre-registered one, or the owner's override, which is valid only with the comment that
    grants it. The returned record names the applied limit, the citation and the measurement it
    covered, and is written into ``meta.json`` and ``gate_override.json``; it is ``None`` when
    the pre-registered envelope already passes.
    """
    if (owner_wall_hours is None) != (owner_decision is None):
        raise SystemExit("an owner wall override needs both --owner-wall-hours and "
                         "--owner-decision (the comment that grants it)")
    wall_max = WALL_ENVELOPE_HOURS
    if owner_wall_hours is not None:
        if not math.isfinite(owner_wall_hours) or owner_wall_hours <= 0:
            raise SystemExit(f"--owner-wall-hours is a positive number of hours, not "
                             f"{owner_wall_hours}")
        wall_max = float(owner_wall_hours)
    missing = [key for key in ("wall_hours", "gpu_hours", "peak_disk_bytes",
                               "within_disk_envelope", "within_owner_envelope")
               if key not in verdict]
    if missing:
        raise SystemExit(f"the gate record names no {', '.join(missing)}: it is not a verdict "
                         f"this guard can read")
    if not verdict["within_disk_envelope"]:
        raise SystemExit(
            f"the gate measured {verdict['peak_disk_bytes']} bytes of peak disk, over the "
            f"{DISK_ENVELOPE_BYTES}-byte envelope; a wall override cannot relax the disk bound")
    gpu_hours = float(verdict["gpu_hours"])
    if gpu_hours > GPU_ENVELOPE_HOURS:
        raise SystemExit(
            f"the gate measured {gpu_hours:g} GPU hours, over the {GPU_ENVELOPE_HOURS:g} h "
            f"envelope; this pass is CPU-only and a wall override cannot relax the GPU bound")
    wall_hours = float(verdict["wall_hours"])
    if wall_hours > wall_max:
        raise SystemExit(
            f"the gate projects {wall_hours:.2f} h, over the {wall_max:g} h wall limit"
            + ("" if owner_wall_hours is not None else
               "; the owner's comment raises it: pass --owner-wall-hours with "
               "--owner-decision"))
    if owner_wall_hours is None:
        return None
    return {
        "wall_hours_max": wall_max,
        "decision": str(owner_decision),
        "measured": {"wall_hours": wall_hours, "gpu_hours": gpu_hours,
                     "peak_disk_bytes": int(verdict["peak_disk_bytes"]),
                     "within_owner_envelope": bool(verdict["within_owner_envelope"])},
    }


def gate_phases(path: Path, binding: Mapping[str, object]) -> dict[str, dict]:
    """The per-split gate phases an earlier attempt settled at this binding, else ``{}``.

    A record from another tranche, worker count or code SHA measured different rows (or the
    same rows at another concurrency), so it is not this gate's measurement and is ignored.
    """
    if not path.exists():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    return record.get("splits", {}) if record.get("binding") == binding else {}


def _settle_gate_phase(split: str, *, fresh: Sequence[str], required: Sequence[str],
                       recorded: Mapping[str, object] | None, elapsed: float,
                       measured: dict, timings: dict, shard_sizes: list[int],
                       sizes_before: int, phases: dict, path: Path,
                       binding: Mapping[str, object]) -> None:
    """Settle one split's gate phase: this attempt's complete measurement, or a stored one.

    The gate's throughput must come from one timed, complete phase per split. A retry that
    skipped part of the sample has no measurement of its own, so it reuses the stored phase of
    exactly that sample, and refuses to gate when no such phase exists: the pooled fallback
    would price this split at the other split's rate.

    ``shard_sizes`` is the run's own list: it gained this phase's bytes from ``sizes_before``
    onwards. A settled phase records exactly that slice, and a restored phase adds its stored
    sizes back to the run, so the final concurrency bound counts every measured shard's blob
    (a resumed gate must not read as an empty disk).
    """
    measured["wall_seconds"] = elapsed
    if set(fresh) == set(required):
        phases[split] = {"shards": sorted(fresh), "timings": dict(measured),
                         "shard_bytes": list(shard_sizes[sizes_before:])}
    elif recorded is not None and set(recorded.get("shards", ())) == set(required):
        measured.update(recorded["timings"])
        _accumulate(timings, recorded["timings"])
        timings["download_seconds"] += recorded["timings"]["download_seconds"]
        timings["download_bytes"] += recorded["timings"]["download_bytes"]
        shard_sizes.extend(recorded["shard_bytes"])
        phases[split] = dict(recorded)
    else:
        raise ValueError(
            f"the gate has no complete phase measurement for the {split} split: this attempt "
            f"measured {len(fresh)} of the {len(required)} shards the sample names, and "
            f"{path} holds no phase at this tranche, worker count and code that measured all "
            f"{len(required)}; remove the gate tranche's outputs and rerun the gate")
    timings["wall_seconds"] += measured["wall_seconds"]
    write_text_atomically(path, json.dumps(
        {"binding": dict(binding), "splits": phases},
        indent=1, ensure_ascii=False, sort_keys=True) + "\n")


def run(shards: Iterable[str], *, out_dir: Path, cache_dir: Path,
        speech_seconds_of: Callable[[np.ndarray], float], floors_path: Path = FLOORS_PATH,
        fp_dir: Path | None = None, stream=None, code_sha: str | None = None,
        pull_request: int | None = None, workers: int = 1,
        index_dir: Path = INDEX_DIR, seed_dir: Path | None = None, gate: bool = False,
        shard_provenance: Mapping[str, Mapping[str, object]] | None = None,
        gate_override: Mapping[str, object] | None = None) -> dict:
    """Process the shards split phase by split phase, verify or measure each, and write meta.

    Each split is one timed phase, so its ``wall_seconds`` is real elapsed time at ``workers``
    concurrency and the projection never re-divides by the worker count (decision 7). A shard
    whose committed or scratch output exists is verified against ``index_dir``'s shard and
    skipped without a download; ``seed_dir`` supplies the committed outputs an empty scratch
    directory starts from, and its shards are skipped the same way. A skipped shard's rows
    count in the freeze, not in the gate's rates.

    Every output's producer is recorded in ``producers.json`` when it is published or verified,
    so a later run at another commit never re-attributes rows it did not write, and an
    interrupted run still attributes the shards it finished; an existing file with no recorded
    producer is refused. A ``gate`` run settles each split's phase in ``gate_phases.json`` and
    writes its verdict — wall hours and peak disk — to ``gate_projection.json`` and into
    ``meta.json``; another run leaves both alone. ``shard_provenance`` names the producer of a
    **skipped** shard the records do not cover, which is how the three S3a tracer shards keep
    ``d2af0e3``/``#198``. ``gate_override`` is :func:`require_gate_pass`'s record of an owner's
    raised wall limit: it lands in ``gate_override.json`` before the first shard and in
    ``meta.json``, and the gate's own ``gate_projection.json`` is left exactly as measured.
    """
    shards = list(shards)
    if gate and set(shards) != set(gate_shards()):
        raise ValueError("a gate run measures all twelve gate shards and nothing else: the "
                         "sample and its two phases are fixed before the run")
    floors_path = Path(floors_path)
    require_floor_config(json.loads(floors_path.read_text(encoding="utf-8")))
    out_dir, index_dir = Path(out_dir), Path(index_dir)
    seed_dir = Path(seed_dir) if seed_dir is not None else None
    if gate_override is not None:
        # Before the first shard: an interrupted full run must still show the limit and the
        # comment it ran under, just as a published shard keeps its producer.
        write_text_atomically(
            out_dir / GATE_OVERRIDE_NAME,
            json.dumps(dict(gate_override), indent=1, ensure_ascii=False,
                       sort_keys=True) + "\n")
    producers_path = out_dir / PRODUCERS_NAME
    phases_path = out_dir / GATE_PHASES_NAME
    current = _provenance(code_sha, pull_request)
    known_producers = {**(S3A_PROVENANCE if shard_provenance is None else shard_provenance),
                       **committed_producers(seed_dir), **scratch_producers(out_dir)}
    measured_sample = gate_measured_shards() if gate else {}
    binding = ({"tranche": sorted(gate_shards()), "workers": workers,
                "code_sha": current["code_sha"]} if gate else {})
    settled = gate_phases(phases_path, binding) if gate else {}
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
        sizes_before = len(shard_sizes)
        fresh: list[str] = []
        for shard in phase:
            stem = Path(shard).stem
            key = f"{split}/{stem}"
            path = out_dir / split / f"{stem}.jsonl.gz"
            index_path = index_dir / split / f"{stem}.jsonl.gz"
            if not path.exists() and seed_dir is not None:
                _seed_pass1(seed_dir / split / path.name, path, index_path)
            if path.exists():
                counts[split] += verify_pass1_shard(path, index_path)
                _record_producer(producers_path, key,
                                 _known_producer(key, path, known_producers))
                skipped += 1
            else:
                fresh.append(shard)
        _measure_phase(fresh, out_dir=out_dir, cache_dir=cache_dir, fp_dir=fp_dir,
                       speech_seconds_of=speech_seconds_of, stream=stream, workers=workers,
                       timings=timings, by_split=by_split[split], counts=counts,
                       shard_sizes=shard_sizes, index_dir=index_dir, producer=current)
        measured_shards += len(fresh)
        elapsed = time.perf_counter() - started
        if gate:
            _settle_gate_phase(split, fresh=fresh, required=measured_sample[split],
                               recorded=settled.get(split), elapsed=elapsed,
                               measured=by_split[split], timings=timings,
                               shard_sizes=shard_sizes, sizes_before=sizes_before,
                               phases=settled, path=phases_path, binding=binding)
        else:
            by_split[split]["wall_seconds"] = elapsed
            timings["wall_seconds"] += elapsed

    measured_rows = sum(by_split[split]["rows"] for split in TLOG_SPLITS)
    remaining = {split: ROWS_BY_SPLIT[split] - counts[split] for split in TLOG_SPLITS}
    # Up to ``workers`` shard blobs coexist; the fingerprint store and the committed outputs
    # accumulate, so the bound adds them to the largest concurrent shard bytes.
    output_bytes = sum(
        (out_dir / tlog_split(shard) / f"{Path(shard).stem}.jsonl.gz").stat().st_size
        for shard in shards)
    concurrent = sum(sorted(shard_sizes, reverse=True)[:max(workers, 1)])
    timings["peak_disk_bytes"] = max(timings["peak_disk_bytes"],
                                     concurrent + timings["fingerprint_bytes"] + output_bytes)
    projection = (gate_projection(
        {split: {"rows": by_split[split]["rows"],
                 "wall_seconds": by_split[split]["wall_seconds"]}
         for split in TLOG_SPLITS},
        remaining_rows=remaining, workers=workers) if measured_rows else None)
    if gate:
        if projection is None:
            raise ValueError("the gate has no measured row to project from")
        projection = gate_verdict(projection, peak_disk_bytes=timings["peak_disk_bytes"],
                                  phases=settled)
        write_text_atomically(
            out_dir / GATE_PROJECTION_NAME,
            json.dumps({**projection, **current}, indent=1, ensure_ascii=False,
                       sort_keys=True) + "\n")
    settled_producers = scratch_producers(out_dir)
    shard_records = {key: settled_producers[key]
                     for key in (f"{tlog_split(shard)}/{Path(shard).stem}"
                                 for shard in shards)}
    throughput = {**_throughput(timings, by_split, measured_shards, workers=workers),
                  "shards_skipped": skipped}
    if gate:
        throughput["gate_projection"] = projection
    if gate_override is not None:
        throughput["gate_override"] = dict(gate_override)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repo": TLOG.repo, "revision": TLOG.revision, "licence": TLOG.licence,
            "splits": {split: sum(1 for shard in TLOG.shards if tlog_split(shard) == split)
                       for split in TLOG_SPLITS},
        },
        **current,
        "vad": VAD_CONFIG,
        "decoders": decoder_versions(),
        "embeddings": "none (owner fallback 2026-10-24, #187)",
        "floors": {"path": _repo_relative(floors_path),
                   "sha256": sha256_file(floors_path)},
        "counts": {"rows": {**{split: counts[split] for split in TLOG_SPLITS},
                            "total": sum(counts.values())}},
        "shards": shard_records,
        "throughput": throughput,
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
                   shard_sizes: list[int], index_dir: Path,
                   producer: Mapping[str, object]) -> None:
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
                            by_split=by_split, counts=counts, shard_sizes=shard_sizes,
                            index_dir=index_dir if stream is None else None,
                            producer=producer)
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
                                by_split=by_split, counts=counts, shard_sizes=shard_sizes,
                                index_dir=index_dir, producer=producer)
            fill()


def _write_measured(shard: str, rows: list[dict], shard_timings: dict, *, out_dir: Path,
                    timings: dict, by_split: dict, counts: Counter,
                    shard_sizes: list[int], index_dir: Path | None,
                    producer: Mapping[str, object]) -> None:
    """Write one measured shard, its producer, and add its counters to the run.

    ``index_dir`` is the S1 index the rows must be exactly: the same rule the resume guard
    applies, checked before the write, so a drifted read fails at that shard instead of at the
    freeze hours later. A synthetic ``stream`` read has no index, and passes ``None``. The
    producer record lands right after the rows, so an interrupted run still attributes them.
    """
    split, stem = tlog_split(shard), Path(shard).stem
    target = out_dir / split / f"{stem}.jsonl.gz"
    if index_dir is not None:
        require_pass1_alignment(rows, read_jsonl_gz(index_dir / split / f"{stem}.jsonl.gz"),
                                target)
    _accumulate(timings, shard_timings)
    _accumulate(by_split, shard_timings)
    timings["download_seconds"] += shard_timings["download_seconds"]
    timings["download_bytes"] += shard_timings["download_bytes"]
    write_pass1(rows, target)
    _record_producer(out_dir / PRODUCERS_NAME, f"{split}/{stem}", producer)
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
    workers = getattr(args, "workers", 1)
    gate = getattr(args, "gate", False)
    if gate and workers != GATE_WORKERS:
        raise SystemExit(f"the gate's tranche is measured at {GATE_WORKERS} workers, not "
                         f"{workers}: a rerun at another concurrency is not the pre-registered "
                         f"measurement")
    if gate:
        shards = gate_shards()
    elif getattr(args, "tracer", False):
        shards = tracer_shards()
    elif getattr(args, "all", False):
        shards = TLOG.shards
    else:
        shards = args.shards
    gate_override = None
    if getattr(args, "all", False):
        verdict_path = Path(args.out_dir) / GATE_PROJECTION_NAME
        if not verdict_path.exists():
            raise SystemExit(f"the full run needs the gate's verdict at {verdict_path}: run "
                             f"the gate into {args.out_dir} before the full corpus")
        gate_override = require_gate_pass(
            json.loads(verdict_path.read_text(encoding="utf-8")),
            owner_wall_hours=getattr(args, "owner_wall_hours", None),
            owner_decision=getattr(args, "owner_decision", None))
    meta = run(shards, out_dir=args.out_dir, cache_dir=args.cache_dir,
               floors_path=args.floors, speech_seconds_of=silero_speech_seconds(),
               fp_dir=args.fp_dir, code_sha=args.code_sha, pull_request=args.pr,
               workers=workers, index_dir=getattr(args, "index_dir", INDEX_DIR),
               seed_dir=getattr(args, "seed_dir", None), gate=gate,
               gate_override=gate_override)
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
    run_command.add_argument("--seed-dir", type=Path, default=PASS1_DIR,
                             help="committed pass-1 files to verify and copy into --out-dir "
                                  "instead of downloading those shards again")
    run_command.add_argument("--workers", type=int, default=1,
                             help="shard processes; the gate and full run use %d"
                                  % GATE_WORKERS)
    run_command.add_argument("--pr", type=int, required=True,
                             help="the PR number whose refs/pull/<N>/head keeps code_sha "
                                  "reachable after the squash merge")
    run_command.add_argument("--owner-wall-hours", type=float, default=None,
                             help="the wall limit an owner comment raised for this full run; "
                                  "it may only relax the wall bound, never the disk or GPU one "
                                  "(default: the pre-registered %g h)"
                                  % WALL_ENVELOPE_HOURS)
    run_command.add_argument("--owner-decision", default=None,
                             help="the comment URL that grants --owner-wall-hours; recorded in "
                                  "meta.json and gate_override.json")
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

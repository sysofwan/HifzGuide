"""Measure the peak GPU memory of the round-2 P3 decode setting (#216's open risk).

P3 decodes the tlog tranches with four models at several processes of one thread and shared
features. Round 1 measured that configuration's *time*, never its footprint, and nothing has
decoded four models at once on the 16 GB card. This runs :mod:`tadabur.tlog_eval_decode`'s
decode on the parity selection only — the first :data:`~tadabur.tlog_eval_decode.PARITY_SHARDS`
look gating shards — with ``base c95 candidate_ema h448``: ``h448`` stands in for round 2's
control, which does not exist yet, on the same preset as C95 and the candidate. Nothing is
graded, no confirm row is opened, and the decodes are scratch.

``nvidia-smi`` is sampled every second while the workers run, and the maximum is the setting's
peak. A setting passes at or under :data:`PASS_PEAK_MIB` of the card's 16,311 MiB (architect
r28's ~1.8 GiB headroom); a failing one is retried at the next count in
:data:`FALLBACK_PROCESSES`, largest first, and the first pass is what P3 runs at. The record
lands in :data:`VRAM_PATH` and names the peaks, the setting and the code it ran from. The parts
are the caller's to delete: no decode here is read.

Usage, from ``tools/`` on the GPU box, under ``flock /root/scratch/gpu.lock``::

    OMP_NUM_THREADS=1 python -m tadabur.tlog_vram_check \\
        --parts-root DIR --cache-dir DIR --clip-cache DIR --pr 213
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .tlog_eval_decode import MODELS, parity_shards
from .tlog_index import head_sha, provenance, read_jsonl, write_json

#: The models P3 decodes. ``h448`` stands in for the missing control: same preset as C95.
VRAM_MODELS = MODELS
#: The pass rule: peak at or under this many MiB of the card's 16,311 (~1.8 GiB headroom).
PASS_PEAK_MIB = 14500
#: The counts to measure, largest first: P3's 6, then the architect's 5 and 4 fallbacks.
FALLBACK_PROCESSES = (6, 5, 4)
#: One thread per process, the configuration P3 runs.
THREADS = 1
SAMPLE_SECONDS = 1.0
#: The committed record of the measurement.
VRAM_PATH = Path(__file__).parent / "consonant2_readout" / "vram_check.json"
#: The workers import ``tadabur`` and ``training`` from the tools directory.
TOOLS_ROOT = Path(__file__).resolve().parents[1]


def gpu_memory_mib(field: str = "memory.used") -> int:
    """One ``nvidia-smi`` reading of the first GPU, in MiB.

    The box has one card; on a multi-GPU host the first line is the one measured, rather than
    an aggregate that would hide which card ran the workers.
    """
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    return int(result.stdout.splitlines()[0].strip())


def _worker_command(index: int, processes: int, *, parts_dir: Path, cache_dir: Path,
                    clip_cache: Path, models: Sequence[str],
                    shards: Sequence[str]) -> list[str]:
    """One process's decode command: its share of the named parity shards, shared features."""
    return [sys.executable, "-u", "-m", "tadabur.tlog_eval_decode", "decode",
            "--parts-dir", str(parts_dir), "--cache-dir", str(cache_dir),
            "--clip-cache", str(clip_cache), "--models", *models,
            "--shard-files", *shards, "--shard-index", str(index),
            "--shard-count", str(processes)]


def measure(processes: int, *, parts_root: Path, cache_dir: Path, clip_cache: Path,
            shards: Sequence[str], models: Sequence[str] = VRAM_MODELS,
            threads: int = THREADS, sample_seconds: float = SAMPLE_SECONDS,
            memory: Callable[[str], int] = gpu_memory_mib) -> dict:
    """Decode the parity shards at ``processes`` workers and report their memory peak.

    The workers write under ``<parts-root>/p<processes>``, which must not exist, so a retry
    never decodes into an earlier setting's directory. The process group is sampled every
    ``sample_seconds`` until every worker has exited. A worker that fails makes the reading
    meaningless, so its code is raised rather than reported. The parts are never read; the
    caller deletes them.
    """
    parts_dir = Path(parts_root) / f"p{processes}"
    parts_dir.mkdir(parents=True)
    baseline = memory("memory.used")
    environment = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    workers = [subprocess.Popen(
        _worker_command(index, processes, parts_dir=parts_dir, cache_dir=cache_dir,
                        clip_cache=clip_cache, models=models, shards=shards),
        cwd=TOOLS_ROOT, env=environment) for index in range(processes)]
    started = time.monotonic()
    peaks = [baseline]
    while any(worker.poll() is None for worker in workers):
        peaks.append(memory("memory.used"))
        time.sleep(sample_seconds)
    codes = [worker.wait() for worker in workers]
    seconds = round(time.monotonic() - started, 1)
    if any(codes):
        raise RuntimeError(f"a decode worker at {processes} processes exited "
                           f"{sorted(set(codes))}; the measurement is not valid")
    timings = read_jsonl(parts_dir / "timing.jsonl")
    return {"processes": processes, "peak_mib": max(peaks), "baseline_mib": baseline,
            "samples": len(peaks), "seconds": seconds, "shards": len(timings),
            "items": sum(timing["items"] for timing in timings)}


def vram_check(*, parts_root: Path, cache_dir: Path, clip_cache: Path,
               out: Path = VRAM_PATH, pull_request: int | None = None,
               models: Sequence[str] = VRAM_MODELS,
               processes: Sequence[int] = FALLBACK_PROCESSES, threads: int = THREADS,
               pass_peak_mib: int = PASS_PEAK_MIB, sample_seconds: float = SAMPLE_SECONDS,
               memory: Callable[[str], int] = gpu_memory_mib) -> dict:
    """Measure each process count until one passes, and write the record of the run.

    The first passing count is the largest, because the counts are measured in the order given
    and the loop stops there. A run in which none passes still writes its record, with
    ``chosen_processes`` null: the peaks are the answer, and no file must go missing for them.
    """
    shards = parity_shards()
    total_mib = memory("memory.total")
    settings = []
    for count in processes:
        setting = measure(count, parts_root=parts_root, cache_dir=cache_dir,
                          clip_cache=clip_cache, shards=shards, models=models,
                          threads=threads, sample_seconds=sample_seconds,
                          memory=memory)
        setting["passed"] = setting["peak_mib"] <= pass_peak_mib
        settings.append(setting)
        if setting["passed"]:
            break
    record = {**provenance(head_sha(), pull_request),
              "models": list(models), "shards": shards, "threads": threads,
              "share_features": True, "total_mib": total_mib,
              "pass_peak_mib": pass_peak_mib, "settings": settings,
              "chosen_processes": (settings[-1]["processes"]
                                   if settings[-1]["passed"] else None)}
    write_json(record, out)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=VRAM_PATH)
    parser.add_argument("--models", nargs="+", default=list(VRAM_MODELS))
    parser.add_argument("--processes", nargs="+", type=int, default=list(FALLBACK_PROCESSES))
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--pass-peak-mib", type=int, default=PASS_PEAK_MIB)
    parser.add_argument("--sample-seconds", type=float, default=SAMPLE_SECONDS)
    parser.add_argument("--pr", type=int, default=None,
                        help="the PR whose refs/pull/<N>/head keeps the cited code SHA")
    args = parser.parse_args()
    record = vram_check(parts_root=args.parts_root, cache_dir=args.cache_dir,
                        clip_cache=args.clip_cache, out=args.out,
                        pull_request=args.pr, models=args.models,
                        processes=args.processes, threads=args.threads,
                        pass_peak_mib=args.pass_peak_mib,
                        sample_seconds=args.sample_seconds)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

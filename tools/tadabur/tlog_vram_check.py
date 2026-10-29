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
:data:`FALLBACK_PROCESSES`, largest first, and the first pass is what P3 runs at. The setting
that does not fit is the one a worker dies on, so a worker whose stderr shows a CUDA
out-of-memory fails that setting and the retry runs; a failure while sampling, or any other
worker exit, stops the workers and raises, so no measurement leaves decoders holding the card.
The record lands in :data:`VRAM_PATH` and names the peaks, the setting and the code it ran
from. The parts are the caller's to delete: no decode here is read.

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
#: What torch prints when the card, not the code, is what failed. A worker that exits
#: nonzero without this in its stderr aborts the measurement instead of failing the setting.
OUT_OF_MEMORY = "CUDA out of memory"
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
                    clip_cache: Path, models: Sequence[str], shards: Sequence[str],
                    share_features: bool) -> list[str]:
    """One process's decode command: its share of the named parity shards.

    ``share_features`` is explicit rather than left to the decoder's default, so the record
    can read the sharing it measured off the command instead of restating it.
    """
    command = [sys.executable, "-u", "-m", "tadabur.tlog_eval_decode", "decode",
               "--parts-dir", str(parts_dir), "--cache-dir", str(cache_dir),
               "--clip-cache", str(clip_cache), "--models", *models,
               "--shard-files", *shards, "--shard-index", str(index),
               "--shard-count", str(processes)]
    return command if share_features else [*command, "--no-share-features"]


def _worker_log(parts_dir: Path, index: int) -> Path:
    """Where worker ``index``'s stderr goes: the only place a CUDA OOM is visible."""
    return Path(parts_dir) / f"worker-{index}.log"


def _out_of_memory(parts_dir: Path, index: int) -> bool:
    """Whether worker ``index``'s stderr says the card, not the code, is what failed."""
    log = _worker_log(parts_dir, index)
    return log.exists() and OUT_OF_MEMORY in log.read_text(encoding="utf-8", errors="replace")


def _stop_workers(workers: Sequence[subprocess.Popen]) -> None:
    """Terminate and reap the workers of a measurement that is being abandoned.

    The runner's flock is released when this process exits, so an orphan would keep holding
    the card with no live job for :func:`refuse_if_live` to see.
    """
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
    for worker in workers:
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()


def measure(processes: int, *, parts_root: Path, cache_dir: Path, clip_cache: Path,
            shards: Sequence[str], models: Sequence[str] = VRAM_MODELS,
            threads: int = THREADS, share_features: bool = True,
            sample_seconds: float = SAMPLE_SECONDS,
            memory: Callable[[str], int] = gpu_memory_mib) -> dict:
    """Decode the parity shards at ``processes`` workers and report their memory peak.

    The workers write under ``<parts-root>/p<processes>``, which must not exist, so a retry
    never decodes into an earlier setting's directory. The process group is sampled every
    ``sample_seconds`` until every worker has exited.

    A worker killed by the card is the setting not fitting, not a broken run: its stderr is
    kept in the parts directory and, when every nonzero exit is a CUDA out-of-memory, the
    setting is returned with ``exit_codes`` so the caller can try the next count. Any other
    worker failure — a bad pin, network, disk — makes the reading meaningless and is raised,
    as is a failure while sampling; either way the workers are stopped first. The parts are
    never read as decodes; the caller deletes them.
    """
    parts_dir = Path(parts_root) / f"p{processes}"
    parts_dir.mkdir(parents=True)
    commands = [_worker_command(index, processes, parts_dir=parts_dir, cache_dir=cache_dir,
                                clip_cache=clip_cache, models=models, shards=shards,
                                share_features=share_features)
                for index in range(processes)]
    environment = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    baseline = memory("memory.used")
    started = time.monotonic()
    peaks = [baseline]
    workers = []
    try:
        for index, command in enumerate(commands):
            with open(_worker_log(parts_dir, index), "wb") as log:
                workers.append(subprocess.Popen(command, cwd=TOOLS_ROOT, env=environment,
                                                stderr=log))
        while any(worker.poll() is None for worker in workers):
            peaks.append(memory("memory.used"))
            time.sleep(sample_seconds)
        codes = [worker.wait() for worker in workers]
    except BaseException:
        _stop_workers(workers)
        raise
    seconds = round(time.monotonic() - started, 1)
    if any(codes) and not all(code == 0 or _out_of_memory(parts_dir, index)
                              for index, code in enumerate(codes)):
        raise RuntimeError(f"a decode worker at {processes} processes exited "
                           f"{sorted(set(codes))} (peak so far {max(peaks)} MiB); "
                           "the measurement is not valid")
    timing_path = parts_dir / "timing.jsonl"
    timings = read_jsonl(timing_path) if timing_path.exists() else []
    setting = {"processes": processes, "peak_mib": max(peaks), "baseline_mib": baseline,
               "samples": len(peaks), "seconds": seconds, "shards": len(timings),
               "items": sum(timing["items"] for timing in timings),
               "share_features": "--no-share-features" not in commands[0]}
    if any(codes):
        setting["exit_codes"] = codes  # present only for a setting that ran out of memory
    return setting


def vram_check(*, parts_root: Path, cache_dir: Path, clip_cache: Path,
               out: Path = VRAM_PATH, pull_request: int | None = None,
               models: Sequence[str] = VRAM_MODELS,
               processes: Sequence[int] = FALLBACK_PROCESSES, threads: int = THREADS,
               pass_peak_mib: int = PASS_PEAK_MIB, sample_seconds: float = SAMPLE_SECONDS,
               memory: Callable[[str], int] = gpu_memory_mib) -> dict:
    """Measure each process count until one passes, and write the record of the run.

    A setting passes only when every worker exited cleanly and its peak is at or under
    ``pass_peak_mib``; one that ran out of memory therefore retries at the next count. The
    first passing count is the largest, because the counts are measured in the order given.
    A run in which none passes still writes its record, with ``chosen_processes`` null: the
    peaks are the answer, and no file must go missing for them.
    """
    shards = parity_shards()
    total_mib = memory("memory.total")
    settings = []
    for count in processes:
        setting = measure(count, parts_root=parts_root, cache_dir=cache_dir,
                          clip_cache=clip_cache, shards=shards, models=models,
                          threads=threads, sample_seconds=sample_seconds,
                          memory=memory)
        setting["passed"] = ("exit_codes" not in setting
                             and setting["peak_mib"] <= pass_peak_mib)
        settings.append(setting)
        if setting["passed"]:
            break
    sharing = {setting.pop("share_features") for setting in settings}
    if len(sharing) != 1:
        raise RuntimeError(f"the settings disagree on sharing: {sorted(sharing)}")
    record = {**provenance(head_sha(), pull_request),
              "models": list(models), "shards": shards, "threads": threads,
              "share_features": sharing.pop(), "total_mib": total_mib,
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

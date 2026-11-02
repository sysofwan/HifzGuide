"""Phase A's CLI (#263), run on the GPU box by ``scripts/issue-263-mining-decode.sh``.

``align`` is one worker's batch: the shards ``sorted(shards)[shard_index::shard_count]`` of
``--role``'s gating rows, each clip run whole through the base teacher
(:func:`tadabur.tlog_ident_decode.load_decoder`, which pins the hub revision), aligned, and
appended to its shard's part (``batch.run_batch``). The clip is fetched into ``--clip-dir``
once, by hardlink from the round-2 look clip cache when it holds it and by a verified shard
stream otherwise; its SHA-256 is checked before the teacher sees it.

``run`` is the preemptible driver: :func:`tadabur.tlog_mining.give_way.run_give_way` takes the
GPU lock non-blocking for one batch at a time, and each batch spawns the workers that still
have rows to do. The PAUSE file and the disk/RAM floors stop it at a batch boundary. It exits 3
when a gate stopped it (a pause is deliberate — ``resume`` clears it and reruns) and 0 when no
row is left.

``merge`` gathers the parts into the committed files, then the launcher fetches them back.

Usage, from ``tools/`` on the box::

    python -m tadabur.tlog_mining align --role look --shard-index 0 --shard-count 2 \\
        --parts-dir DIR --clip-dir DIR --look-clip-cache DIR --shard-cache DIR
    python -m tadabur.tlog_mining run --parts-dir DIR --clip-dir DIR --pause-file PATH \\
        --parallel 2 --batch-seconds 300 --yield-seconds 30
    python -m tadabur.tlog_mining merge --parts-dir DIR --out-dir DIR --pull-request 263
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tadabur import tlog_eval_decode as ted
from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.tlog_ident_decode import load_decoder
from tadabur.tlog_index import head_sha, provenance, read_jsonl, sha256_file, write_jsonl
from tadabur.tlog_pool import default_audio
from tadabur.tlog_tranches import SPLIT_DIR, TRANCHES_DIR
from tadabur.waqf_segments import hafs_word_reference
from training.tlog_eval import _canonical_words

from .batch import pending, run_batch, worker_rows
from .give_way import (BATCH_SECONDS, DISK_RESERVE_GB, RAM_FLOOR_GB, YIELD_SECONDS, Gates,
                       GpuLock, run_give_way)
from .merge import PHASE_A_TRANCHES, merge_word_times
from .selection import PHASE, phase_a_rows, require_phase
from .word_times import build_record

#: The box's GPU lock, the file round 3's launcher holds.
GPU_LOCK = Path("/root/scratch/gpu.lock")
#: How long a worker may overrun the batch before the parent stops it: one in-flight row plus
#: its teardown. The worker's own budget is checked between rows, so it normally stops itself.
OVERDUE_SECONDS = 60


def align(args: argparse.Namespace) -> int:
    """One worker's batch over its shard share, appending to the parts and stopping on budget."""
    require_phase(args.phase)
    rows = _worker_rows(args)
    decoder, identity, fingerprint = load_decoder()
    _record_teacher(args.parts_dir, identity, fingerprint)
    result = run_batch(rows, args.parts_dir, align_row=_align_row(decoder, args),
                       budget_seconds=args.budget_seconds, clock=time.monotonic,
                       prepare_shard=lambda shard, pending: _fetch_shard(shard, pending, args))
    print(f"{args.role} worker {args.shard_index}: wrote {result.written}, done={result.done}",
          flush=True)
    return 0


def _worker_rows(args: argparse.Namespace) -> list[dict]:
    """The worker's rows: its written worklist, else its share of the tranche's gating rows."""
    if args.worklist is not None:
        return read_jsonl(args.worklist)
    return worker_rows(phase_a_rows(args.role, tranches_dir=args.tranches_dir,
                                    split_dir=args.split_dir),
                       shard_index=args.shard_index, shard_count=args.shard_count)


def run(args: argparse.Namespace) -> int:
    """The give-way loop: one lock-held batch at a time, the gates checked at every boundary."""
    require_phase(args.phase)
    if not args.resume and any(Path(args.parts_dir).glob("word_times.*.jsonl")):
        raise SystemExit(f"{args.parts_dir} already holds records: pass --resume to continue "
                         f"that run, or use a fresh --parts-dir")
    rows_by_role = {role: phase_a_rows(role, tranches_dir=args.tranches_dir,
                                       split_dir=args.split_dir)
                    for role in PHASE_A_TRANCHES}
    for directory in (args.parts_dir, args.clip_dir, args.shard_cache):
        Path(directory).mkdir(parents=True, exist_ok=True)
    _write_worklists(rows_by_role, args)
    gates = Gates(args.pause_file, free_disk_gb=lambda: _free_disk_gb(args.parts_dir),
                  available_ram_gb=_available_ram_gb,
                  disk_reserve_gb=args.disk_reserve_gb, ram_floor_gb=args.ram_floor_gb)
    result = run_give_way(_batch(rows_by_role, args), lock=GpuLock(args.lock), gates=gates,
                          batch_seconds=args.batch_seconds, yield_seconds=args.yield_seconds)
    print(f"stopped={result.stopped} batches={result.batches} "
          f"gpu_seconds={result.held_seconds:.0f} yielded_seconds={result.yielded_seconds:.0f}",
          flush=True)
    return 0 if result.stopped == "done" else 3


def merge(args: argparse.Namespace) -> int:
    """The parts into ``<out-dir>/<tranche>.jsonl.gz`` and ``meta.json``."""
    require_phase(args.phase)
    manifests = {role: phase_a_rows(role, tranches_dir=args.tranches_dir,
                                    split_dir=args.split_dir)
                 for role in PHASE_A_TRANCHES}
    meta = merge_word_times(args.parts_dir, args.out_dir, manifests=manifests,
                            provenance={**provenance(args.code_sha or head_sha(),
                                                     args.pull_request),
                                        **_teacher(args.parts_dir)})
    for tranche, counts in meta["tranches"].items():
        print(f"{tranche}: {counts['rows']} rows, {counts['aligned']} aligned, "
              f"{counts['fallback']} fallback ({counts['fallback_rate']:.2%})", flush=True)
    return 0


def _batch(rows_by_role: dict, args: argparse.Namespace):
    """The give-way loop's batch: spawn the workers of the first role with rows left."""
    def batch(budget: float) -> bool:
        for role in PHASE_A_TRANCHES:
            shares = [worker_rows(rows_by_role[role], shard_index=index,
                                  shard_count=args.parallel)
                      for index in range(args.parallel)]
            indexes = [index for index, share in enumerate(shares)
                       if pending(share, args.parts_dir)]
            if indexes:
                _run_workers(role, indexes, budget, args)
                return False
        return True
    return batch


def _write_worklists(rows_by_role: dict, args: argparse.Namespace) -> None:
    """Each worker's rows, written once at launch: a batch then costs no selection read."""
    for role in PHASE_A_TRANCHES:
        for index in range(args.parallel):
            share = worker_rows(rows_by_role[role], shard_index=index,
                                shard_count=args.parallel)
            write_jsonl(share, Path(args.parts_dir) / f"worklist.{role}.{index}.jsonl")


def _run_workers(role: str, indexes: list[int], budget: float, args: argparse.Namespace) -> None:
    """Spawn the worker indexes and bound the whole batch in the parent, not just their rows."""
    command = [sys.executable, "-m", "tadabur.tlog_mining", "align", "--role", role,
               "--parts-dir", str(args.parts_dir), "--clip-dir", str(args.clip_dir),
               "--shard-cache", str(args.shard_cache), "--shard-count", str(args.parallel),
               "--tranches-dir", str(args.tranches_dir), "--split-dir", str(args.split_dir),
               "--phase", args.phase]
    if args.look_clip_cache is not None:
        command += ["--look-clip-cache", str(args.look_clip_cache)]
    environment = {**os.environ, "OMP_NUM_THREADS": "1"}
    started = time.monotonic()
    processes = [(index, subprocess.Popen(
        [*command, "--shard-index", str(index), "--budget-seconds", str(budget),
         "--worklist", str(Path(args.parts_dir) / f"worklist.{role}.{index}.jsonl")],
        env=environment)) for index in indexes]
    failures = []
    for index, process in processes:
        remaining = max(0.0, budget - (time.monotonic() - started))
        try:
            status = process.wait(timeout=remaining + OVERDUE_SECONDS)
        except subprocess.TimeoutExpired:
            # Past the batch's deadline: the row in flight is lost, its part line is not.
            process.terminate()
            try:
                process.wait(timeout=OVERDUE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            continue
        if status != 0:
            failures.append((index, status))
    if failures:
        index, status = failures[0]
        raise RuntimeError(f"{role} worker {index} exited {status}")


def _align_row(decoder, args: argparse.Namespace):
    """The clip -> samples -> posteriors -> record path, with the clip cached by checksum."""
    reference = hafs_word_reference()

    def reference_for(label: str) -> tuple[str, list[int]]:
        return reference(_canonical_words(label))

    def align(row: dict) -> dict:
        path = _verified(ted.clip_path(args.clip_dir, row["audio_sha256"]), row)
        samples = default_audio(path.read_bytes())
        (log_posteriors,) = decoder.span_log_posteriors([samples])
        return build_record(row, reference_for=reference_for, log_posteriors=log_posteriors,
                            duration_s=len(samples) / TARGET_SAMPLE_RATE)
    return align


def _fetch_shard(shard_file: str, rows: list[dict], args: argparse.Namespace) -> None:
    """Put one shard's pending clips in ``--clip-dir`` with **one** stream pass over the shard.

    A clip already cached is re-verified; one the round-2 look cache holds is hardlinked; every
    other clip of the shard comes from the shard's single download (the stream deletes the
    blob once read, so a per-row fetch would re-download the whole shard per clip).
    """
    missing = []
    for row in rows:
        target = ted.clip_path(args.clip_dir, row["audio_sha256"])
        if target.is_file():
            _verified(target, row)
            continue
        source = (None if args.look_clip_cache is None
                  else ted.clip_path(args.look_clip_cache, row["audio_sha256"]))
        if source is not None and source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                target.write_bytes(source.read_bytes())
            _verified(target, row)
            continue
        missing.append(row)
    if not missing:
        return
    ted.cache_shard(shard_file, missing, args.clip_dir,
                    stream=ted.shard_stream(args.shard_cache))
    for row in missing:
        _verified(ted.clip_path(args.clip_dir, row["audio_sha256"]), row)


def _verified(path: Path, row: dict) -> Path:
    checksum = sha256_file(path)
    if checksum != row["audio_sha256"]:
        raise ValueError(f"{path} is {checksum}, not the recorded {row['audio_sha256']}")
    return path


def _record_teacher(parts_dir: Path, identity: dict, fingerprint: dict) -> None:
    """The run's teacher identity, once: ``merge`` refuses a run whose workers disagree.

    Workers of the first batch may race here; each writes its own pid-named file, and every
    later batch sees one and writes nothing.
    """
    directory = Path(parts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob("teacher-*.json")):
        return
    path = directory / f"teacher-{os.getpid()}.json"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"identity": identity, "fingerprint": fingerprint},
                                    ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _teacher(parts_dir: Path) -> dict:
    """The run's teacher record, from the workers' files; a disagreement is a refusal."""
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(Path(parts_dir).glob("teacher-*.json"))]
    if not records:
        return {}
    if any(record != records[0] for record in records[1:]):
        raise ValueError("the workers' teacher identities disagree")
    return {"teacher": records[0]}


def _free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _available_ram_gb() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    raise RuntimeError("/proc/meminfo has no MemAvailable line")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tadabur.tlog_mining", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--tranches-dir", type=Path, default=TRANCHES_DIR)
        command.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
        command.add_argument("--phase", default=PHASE,
                             help=f"the only runnable phase ({PHASE}); B and C are refused")

    align_parser = commands.add_parser("align", help="one worker's batch")
    align_parser.add_argument("--role", choices=list(PHASE_A_TRANCHES), required=True)
    align_parser.add_argument("--shard-index", type=int, required=True)
    align_parser.add_argument("--shard-count", type=int, required=True)
    align_parser.add_argument("--parts-dir", type=Path, required=True)
    align_parser.add_argument("--clip-dir", type=Path, required=True)
    align_parser.add_argument("--shard-cache", type=Path, required=True)
    align_parser.add_argument("--look-clip-cache", type=Path)
    align_parser.add_argument("--worklist", type=Path,
                              help="the rows `run` wrote for this worker; else select them here")
    align_parser.add_argument("--budget-seconds", type=float, default=BATCH_SECONDS)
    add_common(align_parser)
    align_parser.set_defaults(handler=align)

    run_parser = commands.add_parser("run", help="the give-way loop")
    run_parser.add_argument("--parts-dir", type=Path, required=True)
    run_parser.add_argument("--clip-dir", type=Path, required=True)
    run_parser.add_argument("--shard-cache", type=Path, required=True)
    run_parser.add_argument("--look-clip-cache", type=Path)
    run_parser.add_argument("--pause-file", type=Path, required=True)
    run_parser.add_argument("--lock", type=Path, default=GPU_LOCK)
    run_parser.add_argument("--parallel", type=int, default=2)
    run_parser.add_argument("--batch-seconds", type=float, default=BATCH_SECONDS)
    run_parser.add_argument("--yield-seconds", type=float, default=YIELD_SECONDS)
    run_parser.add_argument("--disk-reserve-gb", type=float, default=DISK_RESERVE_GB)
    run_parser.add_argument("--ram-floor-gb", type=float, default=RAM_FLOOR_GB)
    run_parser.add_argument("--resume", action="store_true",
                            help="keep the failed run's parts (the batch resumes by key)")
    add_common(run_parser)
    run_parser.set_defaults(handler=run)

    merge_parser = commands.add_parser("merge", help="the parts into the tranche files")
    merge_parser.add_argument("--parts-dir", type=Path, required=True)
    merge_parser.add_argument("--out-dir", type=Path, required=True)
    merge_parser.add_argument("--code-sha", help="defaults to the staged clone's HEAD")
    merge_parser.add_argument("--pull-request", type=int)
    add_common(merge_parser)
    merge_parser.set_defaults(handler=merge)

    args = parser.parse_args()
    try:
        sys.exit(args.handler(args))
    except ValueError as error:
        # A refusal (a wrong phase, a partial merge, a bad budget) is a message, not a traceback.
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

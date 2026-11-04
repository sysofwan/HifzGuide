"""#267's decode driver: CPU by default, and any GPU batch through #263's yield helper.

``decode_batches`` is torch- and NeMo-free: the model is a callable the caller supplies, so the
loop's gate behaviour is testable without the runtime. On a CPU (the default) the two stops
that still matter on a shared box, the owner's PAUSE file and the RAM floor, are honoured at
batch boundaries and the GPU lock is never touched. On a GPU a batch runs only under
``gpu.lock`` taken non-blockingly by :func:`tadabur.tlog_mining.give_way.run_give_way`, never
waited for; the driver is CPU by default because the sets are a few hours of audio.

:func:`preflight` is the refusal the run cannot skip: the set must be decodable, and the dated
erratum announcing the teacher's letter-level bars must be committed first.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from tadabur.fastconformer_screen.sets import (ERRATUM_PATH, decode_plan, require_decodable,
                                               require_erratum)
from tadabur.fastconformer_screen.verdict import Rates
from tadabur.tlog_mining.give_way import Gates, GpuLock, run_give_way

#: CPU is the default: the sets are small and the GPU belongs to round 3 (#263).
DEFAULT_DEVICE = "cpu"
BATCH_SIZE = 8


@dataclass
class DecodeRun:
    """Where the loop stopped (``done``, ``pause``, ``disk`` or ``ram``) and what it decoded."""

    stopped: str
    lines: list[dict] = field(default_factory=list)
    batches: int = 0
    held_seconds: float = 0.0
    yielded_seconds: float = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The run's command line: a set, a device (CPU unless asked) and an output directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=("ship-a", "edits", "qdat"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEFAULT_DEVICE)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args(argv)


def decode_batches(items: Iterable, decode: Callable[[object], dict], *,
                   device: str, lock: GpuLock, gates: Gates, batch_size: int,
                   sleep: Callable[[float], None] = time.sleep) -> DecodeRun:
    """Decode the items in order, one batch at a time, stopping at a batch boundary.

    ``decode`` turns one item into its record. On the CPU the lock is never touched; on CUDA
    every batch runs under ``run_give_way``'s non-blocking lock and its gates.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {device!r}: only cpu and cuda are supported")
    items = list(items)
    run = DecodeRun("done")

    if device == "cpu":
        for start in range(0, len(items), batch_size):
            reason = gates.stop_reason()
            if reason is not None:
                run.stopped = reason
                return run
            run.lines.extend(decode(item) for item in items[start:start + batch_size])
            run.batches += 1
        return run

    position = 0

    def batch(_seconds: float) -> bool:
        nonlocal position
        run.lines.extend(decode(item) for item in items[position:position + batch_size])
        position += batch_size
        return position >= len(items)

    result = run_give_way(batch, lock=lock, gates=gates, sleep=sleep)
    run.stopped, run.batches = result.stopped, result.batches
    run.held_seconds, run.yielded_seconds = result.held_seconds, result.yielded_seconds
    return run


def preflight(set_name: str, rates: Rates, *, erratum_path: Path = ERRATUM_PATH) -> list[str]:
    """Refuse a round-3 set or a missing/stale erratum, then return the set's decode plan."""
    require_decodable(set_name)
    require_erratum(rates, erratum_path)
    return decode_plan(set_name)

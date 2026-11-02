"""The mining launcher gives way to round 3 (#263): one batch at a time, never a block-wait.

Round 3 is the owner's priority on the box. Phase A therefore takes ``/root/scratch/gpu.lock``
with ``flock(LOCK_EX | LOCK_NB)`` for **one batch** (at most :data:`BATCH_SECONDS`), releases it,
sleeps at least :data:`YIELD_SECONDS`, and tries again. A held lock is waited out by sleeping,
never by blocking: :class:`GpuLock` offers nothing but ``try_acquire``/``release``, so a
blocking acquire is not expressible.

At every batch boundary three gates can stop the loop cleanly — after the batch in flight has
finished and its part flushed — so the box's state is never left half-applied:

* ``PAUSE``: the owner's file; ``pause`` touches it, ``resume`` removes it.
* free disk below :data:`DISK_RESERVE_GB` (bytes free, not the repository's size).
* available RAM below :data:`RAM_FLOOR_GB`.

The result reports where the loop stopped, how many batches it ran, and the time it held the
lock versus the time it yielded: the run note's GPU-hours and give-way totals.
"""

from __future__ import annotations

import fcntl
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: A batch's budget: short enough that round 3 waits minutes, not hours, for the lock.
BATCH_SECONDS = 300
#: The sleep between tries; the issue's floor, so a blocked round-3 launcher gets the lock.
YIELD_SECONDS = 30
#: Free-disk headroom below which a batch is not started.
DISK_RESERVE_GB = 10
#: Available-RAM headroom below which a batch is not started.
RAM_FLOOR_GB = 16


@dataclass(frozen=True)
class GiveWayResult:
    """Where the loop stopped (``done``, ``pause``, ``disk`` or ``ram``) and what it spent."""

    stopped: str
    batches: int
    held_seconds: float
    yielded_seconds: float


class GpuLock:
    """The box's one GPU lock: a non-blocking take and a release, and nothing else."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = None

    def try_acquire(self) -> bool:
        """Take the lock without waiting; ``False`` when another process holds it."""
        handle = open(self.path, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle, fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


class Gates:
    """The three batch-boundary stops, in precedence order: pause, then disk, then RAM.

    The two meters are zero-argument callables, so the caller owns how they are read
    (``shutil.disk_usage`` and ``/proc/meminfo`` on the box; constants in a test).
    """

    def __init__(self, pause_file: Path, *, free_disk_gb: Callable[[], float],
                 available_ram_gb: Callable[[], float],
                 disk_reserve_gb: float = DISK_RESERVE_GB,
                 ram_floor_gb: float = RAM_FLOOR_GB) -> None:
        self.pause_file = Path(pause_file)
        self.free_disk_gb = free_disk_gb
        self.available_ram_gb = available_ram_gb
        self.disk_reserve_gb = disk_reserve_gb
        self.ram_floor_gb = ram_floor_gb

    def stop_reason(self) -> str | None:
        if self.pause_file.exists():
            return "pause"
        if self.free_disk_gb() < self.disk_reserve_gb:
            return "disk"
        if self.available_ram_gb() < self.ram_floor_gb:
            return "ram"
        return None


def run_give_way(batch: Callable[[float], bool], *, lock: GpuLock, gates: Gates,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 batch_seconds: float = BATCH_SECONDS,
                 yield_seconds: float = YIELD_SECONDS) -> GiveWayResult:
    """Run ``batch(batch_seconds)`` under the lock, one batch per iteration, until it says done.

    ``batch`` returns ``True`` when no row is left; a batch that raises releases the lock and
    propagates (the failure is the run's). The gates are checked before every lock try and after
    every batch, so a stop never lands inside a batch. ``yield_seconds`` below
    :data:`YIELD_SECONDS` and a non-positive ``batch_seconds`` are refused up front.
    """
    if batch_seconds <= 0:
        raise ValueError(f"batch_seconds must be positive, not {batch_seconds}")
    if yield_seconds < YIELD_SECONDS:
        raise ValueError(f"yield_seconds must be at least {YIELD_SECONDS}, not {yield_seconds}")
    batches = 0
    held_seconds = yielded_seconds = 0.0
    while True:
        reason = gates.stop_reason()
        if reason is not None:
            return GiveWayResult(reason, batches, held_seconds, yielded_seconds)
        if not lock.try_acquire():
            sleep(yield_seconds)
            yielded_seconds += yield_seconds
            continue
        started = clock()
        try:
            finished = batch(batch_seconds)
        finally:
            lock.release()
            held_seconds += clock() - started
        batches += 1
        if finished:
            return GiveWayResult("done", batches, held_seconds, yielded_seconds)
        reason = gates.stop_reason()
        if reason is not None:
            return GiveWayResult(reason, batches, held_seconds, yielded_seconds)
        sleep(yield_seconds)
        yielded_seconds += yield_seconds

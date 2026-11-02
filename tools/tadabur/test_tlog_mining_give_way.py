"""The mining launcher gives way to round 3 (#263): its loop, torch-free, on fakes.

The interface, in ``tadabur.tlog_mining.give_way``:

* ``GpuLock(path)``: ``try_acquire() -> bool`` takes ``flock(LOCK_EX | LOCK_NB)`` and never
  waits; ``release()`` lets go. Nothing else is offered, so a block-wait is not expressible.
* ``Gates(pause_file, *, free_disk_gb, available_ram_gb, disk_reserve_gb=DISK_RESERVE_GB,
  ram_floor_gb=RAM_FLOOR_GB)``: the two meters are zero-argument callables.
  ``stop_reason()`` is ``"pause"`` when the pause file exists, else ``"disk"`` when free disk is
  *below* the reserve, else ``"ram"`` when available RAM is *below* the floor, else ``None``.
* ``run_give_way(batch, *, lock, gates, sleep, clock, batch_seconds, yield_seconds)
  -> GiveWayResult``. Each iteration checks the gates (a stop is returned at once); tries the
  lock once; if it is held elsewhere, sleeps ``yield_seconds`` and loops; otherwise calls
  ``batch(batch_seconds)`` with the lock held (it returns ``True`` when no work is left),
  releases the lock (also when the batch raises, which propagates), returns ``"done"`` if the
  work is finished, and else checks the gates again and sleeps ``yield_seconds``. The result has
  ``stopped`` (``"done"``, ``"pause"``, ``"disk"`` or ``"ram"``), ``batches``, ``held_seconds``
  (clock time inside batches) and ``yielded_seconds`` (the sleeps' total).
"""

from __future__ import annotations

import fcntl
import time

import pytest

from tadabur.tlog_mining import give_way as gw


class FakeLock:
    """Only the two calls the contract allows; anything else is an AttributeError."""

    def __init__(self, free_after: int = 0, events: list | None = None) -> None:
        self.busy_tries = free_after
        self.events = [] if events is None else events

    def try_acquire(self) -> bool:
        self.events.append("try")
        if self.busy_tries:
            self.busy_tries -= 1
            return False
        self.events.append("held")
        return True

    def release(self) -> None:
        self.events.append("release")


class World:
    """A fake clock that sleeps and batches advance, and the meters the gates read."""

    def __init__(self, tmp_path, *, disk=100.0, ram=64.0) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.budgets: list[float] = []
        self.disk, self.ram = disk, ram
        self.pause = tmp_path / "PAUSE"
        self.gates = gw.Gates(self.pause, free_disk_gb=lambda: self.disk,
                              available_ram_gb=lambda: self.ram)

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def batch(self, *, finishes_on: int = 10**9, seconds: float = 100.0, hook=None):
        def run(budget: float) -> bool:
            self.budgets.append(budget)
            self.now += seconds
            if hook:
                hook(len(self.budgets))
            return len(self.budgets) >= finishes_on
        return run

    def run(self, batch, lock, **kwargs) -> gw.GiveWayResult:
        return gw.run_give_way(batch, lock=lock, gates=self.gates, sleep=self.sleep,
                               clock=self.clock, **kwargs)


# --- the defaults ---------------------------------------------------------------------------------


def test_the_defaults_are_the_issues():
    assert gw.BATCH_SECONDS <= 300
    assert gw.YIELD_SECONDS >= 30
    assert gw.DISK_RESERVE_GB == 10 and gw.RAM_FLOOR_GB == 16


@pytest.mark.parametrize("yield_seconds", [0, 29.9, -1])
def test_a_yield_shorter_than_30_seconds_is_refused(tmp_path, yield_seconds):
    world = World(tmp_path)
    with pytest.raises(ValueError):
        world.run(world.batch(), FakeLock(), batch_seconds=300, yield_seconds=yield_seconds)


@pytest.mark.parametrize("batch_seconds", [0, -5])
def test_a_batch_without_a_budget_is_refused(tmp_path, batch_seconds):
    world = World(tmp_path)
    with pytest.raises(ValueError):
        world.run(world.batch(), FakeLock(), batch_seconds=batch_seconds, yield_seconds=30)


# --- the lock -------------------------------------------------------------------------------------


def test_the_lock_is_tried_non_blocking_held_for_one_batch_and_released_before_the_next(tmp_path):
    world, lock = World(tmp_path), FakeLock()
    result = world.run(world.batch(finishes_on=3), lock, batch_seconds=240, yield_seconds=45)
    assert lock.events == ["try", "held", "release"] * 3
    assert world.budgets == [240] * 3
    assert result.stopped == "done" and result.batches == 3


def test_a_lock_held_elsewhere_is_waited_out_by_sleeping_not_blocking(tmp_path):
    world, lock = World(tmp_path), FakeLock(free_after=3)
    result = world.run(world.batch(finishes_on=1), lock, batch_seconds=300, yield_seconds=30)
    assert lock.events == ["try", "try", "try", "try", "held", "release"]
    assert world.sleeps == [30, 30, 30]
    assert result.yielded_seconds == 90 and result.batches == 1


def test_every_sleep_between_tries_is_at_least_the_yield(tmp_path):
    world = World(tmp_path)
    world.run(world.batch(finishes_on=3), FakeLock(free_after=2), batch_seconds=300,
              yield_seconds=75)
    assert world.sleeps and all(seconds >= 75 for seconds in world.sleeps)


def test_the_loop_sleeps_between_batches_so_a_blocked_round_3_launcher_gets_the_lock(tmp_path):
    world, lock = World(tmp_path), FakeLock(events=[])

    def record_sleep(seconds):
        lock.events.append("sleep")
        World.sleep(world, seconds)

    result = gw.run_give_way(world.batch(finishes_on=2), lock=lock, gates=world.gates,
                             sleep=record_sleep, clock=world.clock, batch_seconds=300,
                             yield_seconds=30)
    assert lock.events == ["try", "held", "release", "sleep", "try", "held", "release"]
    assert result.stopped == "done"


def test_a_finished_run_does_not_sleep_after_its_last_batch(tmp_path):
    world = World(tmp_path)
    world.run(world.batch(finishes_on=1), FakeLock(), batch_seconds=300, yield_seconds=30)
    assert world.sleeps == []


def test_the_lock_is_released_when_a_batch_fails_and_the_failure_propagates(tmp_path):
    world, lock = World(tmp_path), FakeLock()

    def failing(budget):
        raise RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        world.run(failing, lock, batch_seconds=300, yield_seconds=30)
    assert lock.events == ["try", "held", "release"]


def test_held_and_yielded_time_are_measured(tmp_path):
    world = World(tmp_path)
    result = world.run(world.batch(finishes_on=3, seconds=120), FakeLock(free_after=1),
                       batch_seconds=300, yield_seconds=30)
    assert result.held_seconds == 360
    assert result.yielded_seconds == sum(world.sleeps) == 90


# --- the gates ------------------------------------------------------------------------------------


def test_a_pause_file_that_exists_before_the_start_stops_without_trying_the_lock(tmp_path):
    world, lock = World(tmp_path), FakeLock()
    world.pause.touch()
    result = world.run(world.batch(), lock, batch_seconds=300, yield_seconds=30)
    assert result.stopped == "pause" and result.batches == 0
    assert lock.events == [] and world.budgets == []


def test_a_pause_during_a_batch_stops_at_its_boundary_with_the_lock_released(tmp_path):
    world, lock = World(tmp_path), FakeLock()
    batch = world.batch(hook=lambda number: world.pause.touch() if number == 2 else None)
    result = world.run(batch, lock, batch_seconds=300, yield_seconds=30)
    assert result.stopped == "pause" and result.batches == 2
    assert lock.events == ["try", "held", "release"] * 2
    assert world.sleeps == [30]  # the one between batch 1 and 2, none after the pause


def test_a_pause_while_the_lock_is_busy_stops_the_wait(tmp_path):
    world = World(tmp_path)
    lock = FakeLock(free_after=100)

    def touch_after_two_sleeps(seconds):
        World.sleep(world, seconds)
        if len(world.sleeps) == 2:
            world.pause.touch()

    result = gw.run_give_way(world.batch(), lock=lock, gates=world.gates,
                             sleep=touch_after_two_sleeps, clock=world.clock,
                             batch_seconds=300, yield_seconds=30)
    assert result.stopped == "pause" and result.batches == 0
    assert world.budgets == []


@pytest.mark.parametrize("meter, value, reason", [("disk", 9.99, "disk"), ("ram", 15.99, "ram")])
def test_a_floor_crossed_during_a_batch_stops_at_its_boundary(tmp_path, meter, value, reason):
    world, lock = World(tmp_path), FakeLock()
    batch = world.batch(hook=lambda number: setattr(world, meter, value) if number == 1 else None)
    result = world.run(batch, lock, batch_seconds=300, yield_seconds=30)
    assert result.stopped == reason and result.batches == 1
    assert lock.events == ["try", "held", "release"]


@pytest.mark.parametrize("meter, value", [("disk", 10.0), ("ram", 16.0)])
def test_a_meter_exactly_at_its_floor_is_not_below_it(tmp_path, meter, value):
    world = World(tmp_path)
    setattr(world, meter, value)
    assert world.gates.stop_reason() is None


def test_the_floors_are_set_by_the_caller(tmp_path):
    gates = gw.Gates(tmp_path / "PAUSE", free_disk_gb=lambda: 20.0, available_ram_gb=lambda: 40.0,
                     disk_reserve_gb=25, ram_floor_gb=32)
    assert gates.stop_reason() == "disk"
    gates = gw.Gates(tmp_path / "PAUSE", free_disk_gb=lambda: 30.0, available_ram_gb=lambda: 31.0,
                     disk_reserve_gb=25, ram_floor_gb=32)
    assert gates.stop_reason() == "ram"


def test_the_pause_file_wins_over_the_floors(tmp_path):
    world = World(tmp_path, disk=1.0, ram=1.0)
    world.pause.touch()
    assert world.gates.stop_reason() == "pause"
    world.pause.unlink()
    assert world.gates.stop_reason() == "disk"


# --- the real lock --------------------------------------------------------------------------------


def test_the_real_lock_reports_a_held_lock_at_once_instead_of_waiting(tmp_path):
    path = tmp_path / "gpu.lock"
    with open(path, "w") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        started = time.monotonic()
        assert gw.GpuLock(path).try_acquire() is False
        assert time.monotonic() - started < 2
    lock = gw.GpuLock(path)
    assert lock.try_acquire() is True
    other = gw.GpuLock(path)
    assert other.try_acquire() is False
    lock.release()
    assert other.try_acquire() is True
    other.release()


def test_the_lock_offers_no_blocking_acquire():
    assert not hasattr(gw.GpuLock, "acquire") and not hasattr(gw.GpuLock, "wait")

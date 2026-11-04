"""Tests for #267's decode driver: CPU by default, and any GPU use through #263's yield helper.

``decode_batches`` is torch-free: the model is a function the caller supplies. On the CPU
(the default) the gate stops that still matter on a shared box, PAUSE and the RAM floor, are
honoured at batch boundaries; on a GPU the batch runs only under ``gpu.lock`` taken
non-blockingly by :func:`tadabur.tlog_mining.give_way.run_give_way`, never waited for.
"""

from __future__ import annotations

import pytest

from tadabur.fastconformer_screen.decode import (
    DEFAULT_DEVICE,
    decode_batches,
    parse_args,
)
from tadabur.tlog_mining.give_way import Gates, GpuLock


def gates(tmp_path, ram_gb: float = 64.0) -> Gates:
    return Gates(tmp_path / "PAUSE", free_disk_gb=lambda: 500.0, available_ram_gb=lambda: ram_gb)


class Calls:
    """A decode function that records what it was asked to decode."""

    def __init__(self, on_call=None):
        self.items, self.on_call = [], on_call

    def __call__(self, item):
        self.items.append(item)
        if self.on_call:
            self.on_call(item)
        return {"key": item}


class ForbiddenLock:
    """A lock that must never be touched (the CPU path does not use the GPU lock)."""

    def try_acquire(self):
        raise AssertionError("the CPU path must not take gpu.lock")

    def release(self):
        raise AssertionError("the CPU path must not touch gpu.lock")


def test_cpu_is_the_default_device():
    assert DEFAULT_DEVICE == "cpu"
    assert parse_args([]).device == "cpu"
    assert parse_args(["--device", "cuda"]).device == "cuda"


def test_an_unknown_device_is_refused():
    with pytest.raises(SystemExit):
        parse_args(["--device", "tpu"])


def test_the_cpu_path_decodes_everything_in_order_without_the_gpu_lock(tmp_path):
    calls = Calls()
    run = decode_batches(["a", "b", "c"], calls, device="cpu", lock=ForbiddenLock(),
                         gates=gates(tmp_path), batch_size=2, sleep=lambda _: None)
    assert run.lines == [{"key": "a"}, {"key": "b"}, {"key": "c"}]
    assert (run.stopped, calls.items) == ("done", ["a", "b", "c"])


def test_the_cpu_path_still_stops_at_a_batch_boundary_when_pause_appears(tmp_path):
    pause = tmp_path / "PAUSE"
    calls = Calls(on_call=lambda item: pause.touch() if item == "a" else None)
    run = decode_batches(["a", "b", "c", "d"], calls, device="cpu", lock=ForbiddenLock(),
                         gates=gates(tmp_path), batch_size=2, sleep=lambda _: None)
    assert run.stopped == "pause"
    assert calls.items == ["a", "b"]            # the batch in flight finished; the next never began


def test_the_cpu_path_stops_before_starting_below_the_ram_floor(tmp_path):
    calls = Calls()
    run = decode_batches(["a"], calls, device="cpu", lock=ForbiddenLock(),
                         gates=gates(tmp_path, ram_gb=1.0), batch_size=2, sleep=lambda _: None)
    assert (run.stopped, run.lines, calls.items) == ("ram", [], [])


def test_the_gpu_path_decodes_under_the_lock_and_releases_it(tmp_path):
    lock = GpuLock(tmp_path / "gpu.lock")
    calls = Calls()
    run = decode_batches(["a", "b", "c"], calls, device="cuda", lock=lock,
                         gates=gates(tmp_path), batch_size=2, sleep=lambda _: None)
    assert run.lines == [{"key": "a"}, {"key": "b"}, {"key": "c"}] and run.stopped == "done"
    other = GpuLock(tmp_path / "gpu.lock")
    assert other.try_acquire()                  # released between and after the batches
    other.release()


def test_the_gpu_path_never_waits_for_a_held_lock(tmp_path):
    holder = GpuLock(tmp_path / "gpu.lock")
    assert holder.try_acquire()                 # round 3 holds the GPU
    calls = Calls()
    # the only way out of the yield sleep is the owner's PAUSE: a blocking acquire would hang here
    sleeps = []
    run = decode_batches(["a", "b"], calls, device="cuda", lock=GpuLock(tmp_path / "gpu.lock"),
                         gates=gates(tmp_path), batch_size=1,
                         sleep=lambda seconds: (sleeps.append(seconds), (tmp_path / "PAUSE").touch()))
    holder.release()
    assert (run.stopped, run.lines, calls.items) == ("pause", [], [])
    assert len(sleeps) == 1


def test_the_gpu_path_stops_when_pause_exists_before_any_batch(tmp_path):
    (tmp_path / "PAUSE").touch()
    calls = Calls()
    run = decode_batches(["a"], calls, device="cuda", lock=GpuLock(tmp_path / "gpu.lock"),
                         gates=gates(tmp_path), batch_size=1, sleep=lambda _: None)
    assert (run.stopped, calls.items) == ("pause", [])


def test_a_device_other_than_cpu_or_cuda_is_refused(tmp_path):
    with pytest.raises(ValueError, match="tpu"):
        decode_batches(["a"], Calls(), device="tpu", lock=ForbiddenLock(),
                       gates=gates(tmp_path), batch_size=1, sleep=lambda _: None)

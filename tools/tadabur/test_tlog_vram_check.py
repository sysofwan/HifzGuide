"""Tests for the P3 VRAM check's record, its fallback rule and its failure paths (#216).

The committed ``vram_check.json`` is the measurement round 2 launches on: it must name the
parity selection, the four models, one thread and the peaks of every setting tried, and the
count P3 runs at must be the largest whose peak fits the card's budget. The real reading
cannot be re-derived by a fake, so the record is pinned as it was written.

The rest pins the paths the box run did not take: a setting killed by an out-of-memory worker
must fail and retry rather than abort, a broken environment must abort, a failed reading must
leave no decoders behind, and the launcher's runner must invoke the module where the record
lives and delete the scratch parts. Those are the failures that would otherwise only show up as
a launch that OOMs for no visible reason.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import pytest

from tadabur import tlog_eval_decode as ted
from tadabur import tlog_vram_check as tv
from tadabur.test_tlog_eval_decode import _launcher_call, _launcher_env, _stub_launcher_box


def test_the_committed_vram_check_records_a_passing_setting_of_the_parity_decode():
    record = json.loads(tv.VRAM_PATH.read_text(encoding="utf-8"))

    assert re.fullmatch(r"[0-9a-f]{40}", record["code_sha"])
    assert record["pull_request"] is None or re.fullmatch(
        rf"refs/pull/{record['pull_request']}/head", record["pull_request_ref"])
    assert record["models"] == list(ted.MODELS)
    assert (record["threads"], record["share_features"]) == (tv.THREADS, True)
    assert record["shards"] == ted.parity_shards() and len(record["shards"]) == 20
    assert record["total_mib"] == 16311
    assert (record["pass_peak_mib"], record["chosen_processes"]) == (14500, 6)
    settings = record["settings"]
    assert [setting["processes"] for setting in settings] == [6]  # one look decode passed
    for setting in settings:
        assert setting["items"] > 0 and setting["samples"] >= 2
        assert setting["peak_mib"] >= setting["baseline_mib"] > 0
        assert setting["peak_mib"] <= record["pass_peak_mib"]
        assert setting["passed"] is True


def _fake_measure(peaks, tried):
    def measure(processes, **kwargs):
        tried.append(processes)
        return {"processes": processes, "peak_mib": peaks[processes], "baseline_mib": 2,
                "samples": 3, "seconds": 1.0, "shards": 1, "items": 1,
                "share_features": True}
    return measure


def test_the_first_passing_setting_is_chosen_and_the_rest_are_not_run(monkeypatch, tmp_path):
    tried = []
    monkeypatch.setattr(tv, "parity_shards", lambda: ["clean-00000-of-00411"])
    monkeypatch.setattr(tv, "head_sha", lambda: "0" * 40)
    monkeypatch.setattr(tv, "measure",
                        _fake_measure({6: 15000, 5: 14000, 4: 13000}, tried))
    record = tv.vram_check(parts_root=tmp_path, cache_dir=tmp_path, clip_cache=tmp_path,
                           out=tmp_path / "vram.json", memory=lambda field: 16311)

    assert tried == [6, 5]
    assert [setting["passed"] for setting in record["settings"]] == [False, True]
    assert record["chosen_processes"] == 5
    assert record["share_features"] is True
    assert json.loads((tmp_path / "vram.json").read_text(encoding="utf-8")) == record


def test_a_setting_that_ran_out_of_memory_fails_and_retries(monkeypatch, tmp_path):
    def oom(index, processes, **kwargs):
        return [sys.executable, "-c",
                "import sys; sys.stderr.write('CUDA out of memory: tried to allocate 2.00 GiB');"
                " sys.exit(1)"]

    monkeypatch.setattr(tv, "_worker_command", oom)
    monkeypatch.setattr(tv, "parity_shards", lambda: ["clean-00000-of-00411"])
    monkeypatch.setattr(tv, "head_sha", lambda: "0" * 40)
    record = tv.vram_check(parts_root=tmp_path / "parts", cache_dir=tmp_path,
                           clip_cache=tmp_path, out=tmp_path / "vram.json",
                           processes=[2, 1], memory=lambda field: 16000,
                           sample_seconds=0.01)

    settings = record["settings"]
    assert [setting["processes"] for setting in settings] == [2, 1]
    assert all(setting["passed"] is False for setting in settings)
    assert all(set(setting["exit_codes"]) == {1} for setting in settings)
    assert record["chosen_processes"] is None


def test_a_worker_that_fails_without_the_card_aborts_the_measurement(monkeypatch, tmp_path):
    def broken(index, processes, **kwargs):
        return [sys.executable, "-c",
                "import sys; sys.stderr.write('ValueError: bad checkpoint'); sys.exit(1)"]

    monkeypatch.setattr(tv, "_worker_command", broken)
    monkeypatch.setattr(tv, "parity_shards", lambda: ["clean-00000-of-00411"])
    monkeypatch.setattr(tv, "head_sha", lambda: "0" * 40)
    with pytest.raises(RuntimeError, match=r"exited \[1\] \(peak so far 100 MiB\)"):
        tv.vram_check(parts_root=tmp_path / "parts", cache_dir=tmp_path,
                      clip_cache=tmp_path, out=tmp_path / "vram.json", processes=[1],
                      memory=lambda field: 100, sample_seconds=0.01)

    assert not (tmp_path / "vram.json").exists()


def test_a_failed_reading_stops_the_workers_it_started(monkeypatch, tmp_path):
    pidfile = tmp_path / "pid"

    def sleeper(index, processes, **kwargs):
        return [sys.executable, "-c",
                f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid()));"
                " time.sleep(30)"]

    def failing_memory(field):
        if field == "memory.used":
            used.append(field)
            if len(used) == 2:  # the baseline, then the first sample: the worker is live
                time.sleep(0.5)
                raise OSError("nvidia-smi failed")
        return 100

    used = []
    monkeypatch.setattr(tv, "_worker_command", sleeper)
    monkeypatch.setattr(tv, "parity_shards", lambda: ["clean-00000-of-00411"])
    monkeypatch.setattr(tv, "head_sha", lambda: "0" * 40)
    with pytest.raises(OSError, match="nvidia-smi failed"):
        tv.vram_check(parts_root=tmp_path / "parts", cache_dir=tmp_path,
                      clip_cache=tmp_path, out=tmp_path / "vram.json", processes=[1],
                      memory=failing_memory, sample_seconds=0.01)

    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text(encoding="utf-8")), 0)


def test_the_generated_vram_runner_measures_the_parity_decode_and_deletes_its_parts(tmp_path):
    parts = tmp_path / "root" / "parts-vram"
    env = _launcher_env(tmp_path, PR="217", VRAM_PARTS=str(parts))
    binaries, log = _stub_launcher_box(tmp_path, env)
    (tmp_path / "root" / "logs").mkdir()
    parts.mkdir()
    assert _launcher_call(tmp_path, "vram", "write_vram_runner", env).returncode == 0
    run = subprocess.run(["bash", str(tmp_path / "root" / "run-vram.sh")],
                         env={**env, "PATH": f"{binaries}:{env['PATH']}"},
                         capture_output=True, text=True, timeout=120)

    assert run.returncode == 0, run.stderr
    calls = [line for line in log.read_text(encoding="utf-8").splitlines()
             if "tadabur.tlog_vram_check" in line]
    assert len(calls) == 1
    record = tmp_path / "root" / "repo" / tv.VRAM_PATH.relative_to(tv.TOOLS_ROOT.parent)
    for argument in (f"--parts-root {parts}", f"--cache-dir {env['CACHE']}",
                     f"--clip-cache {env['CLIP_CACHE_DIR']}", f"--out {record}", "--pr 217"):
        assert argument in calls[0]
    assert not parts.exists()
    assert (tmp_path / "root" / "logs" / "vram.exit").read_text(encoding="utf-8").strip() == "0"

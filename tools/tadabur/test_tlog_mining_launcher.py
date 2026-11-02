"""The mining launcher (#263): ``scripts/issue-263-mining-decode.sh``, run dry.

Phase A decodes the look and confirm.1/2 gating rows with the base teacher, and gives way to
round 3 at every batch boundary (the loop is :mod:`tadabur.tlog_mining.give_way`; this file is
the shell around it). It is ``scripts/issue-230-confirm-decode.sh``'s shape, so the stand-ins are
that file's: ``ssh`` logs its remote command (``$2``, so the script calls ``ssh "$HOST" "<cmd>"``
with no flag before the host) and runs a ``nohup`` line locally; ``setsid`` and ``flock`` execute
their arguments.

* ``plan [PHASE]`` (default ``A``) prints, from the **committed** ``choice.json`` and the two
  committed confirm claims, and writes nothing::

      align look base at 2 processes
      align confirm.1 base at 2 processes
      align confirm.2 base at 2 processes
      merge word_times

  It refuses (non-zero exit, nothing on stdout, ``not committed`` on stderr) when one of them is
  not committed and unchanged at HEAD. ``PARALLEL`` (default 2) sets the processes; ``ROUND_DIR``
  and ``TRANCHES_DIR`` are as in the confirm launcher. Phase ``B`` and ``C`` are refused by every
  command that takes a phase (``plan``, ``launch``): ``phase B`` on stderr, nothing staged.
* ``launch [PHASE]``: the same refusals, then stages the repository and starts ``run`` detached
  with ``setsid nohup`` (never under a blocking ``flock`` of the GPU lock: the Python loop takes
  the lock non-blocking for one batch at a time), printing ``started: <pid>``.
* ``pause``: ``touch <ISSUE_ROOT>/PAUSE`` on the box; ``ISSUE_ROOT`` defaults to
  ``/root/scratch/tlog-mining``. The loop stops at its next batch boundary.
* ``resume``: removes that file (``rm -f``), restages and starts ``run --resume`` (the loop
  resumes by row key).
* ``status`` is read-only on the box (``run.done``/``run.exit`` and the wall and yielded-time
  log); the box's ``run`` flags are ``--pause-file --batch-seconds --yield-seconds
  --disk-reserve-gb --ram-floor-gb``.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from training import round2_verdict_fixtures as rv
from training.test_consonant2_confirm_launcher import _env, _mock, _round

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "issue-263-mining-decode.sh"
PLAN = ["align look base at 2 processes", "align confirm.1 base at 2 processes",
        "align confirm.2 base at 2 processes", "merge word_times"]


def _run(command: str, round_: rv.Round, tmp_path: Path, *args: str, **extra: str):
    return subprocess.run(["bash", str(SCRIPT), command, *args], cwd=round_.repo,
                          env=_env(tmp_path, **extra), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60)


def _box(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "mining"
    (root / "logs").mkdir(parents=True)
    (root / "repo" / "scripts").mkdir(parents=True)
    marker = tmp_path / "started"
    _mock(root / "repo" / "scripts" / SCRIPT.name,
          f"#!/bin/sh\nprintf 'ran\\n' > '{marker}'\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _mock(bindir / "ssh", '#!/bin/sh\nprintf "%s\\n" "$2" >> "$SSH_LOG"\n'
          'case "$2" in\n  *nohup*) exec sh -c "$2" ;;\n  *) cat >/dev/null ;;\nesac\n')
    for name in ("flock", "setsid"):
        _mock(bindir / name, '#!/bin/sh\n[ "$1" = -n ] && shift\nexec "$@"\n')
    return {"PATH": f"{bindir}:{_env(tmp_path)['PATH']}", "HOST": "box",
            "ISSUE_ROOT": str(root), "SSH_LOG": str(tmp_path / "ssh.log")}


def _ssh_log(tmp_path: Path) -> str:
    return (tmp_path / "ssh.log").read_text(encoding="utf-8")


def test_the_script_exists_parses_and_runs_the_mining_package():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    text = SCRIPT.read_text(encoding="utf-8")
    assert "tadabur.tlog_mining" in text
    assert "/root/scratch/tlog-mining" in text
    for flag in ("--pause-file", "--batch-seconds", "--yield-seconds", "--disk-reserve-gb",
                 "--ram-floor-gb"):
        assert flag in text


def test_the_script_never_waits_on_the_gpu_lock():
    """Any ``flock`` the script runs is non-blocking: round 3's launcher must get the lock."""
    code = [line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")]
    for line in code:
        assert not re.search(r"\bflock\b(?!\s+-\w*n\b)", line), line
        assert "flock -w" not in line and "--timeout" not in line


def test_the_plan_lists_phase_a(tmp_path):
    round_ = _round(tmp_path)
    done = _run("plan", round_, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == PLAN
    assert _run("plan", round_, tmp_path, "A").stdout.splitlines() == PLAN


@pytest.mark.parametrize("phase", ["B", "C"])
@pytest.mark.parametrize("command", ["plan", "launch"])
def test_phases_b_and_c_are_refused_before_anything_is_staged(tmp_path, command, phase):
    round_ = _round(tmp_path)
    done = _run(command, round_, tmp_path, phase, **_box(tmp_path))
    assert done.returncode != 0 and done.stdout == ""
    assert f"phase {phase}" in done.stderr
    assert not (tmp_path / "ssh.log").exists()


@pytest.mark.parametrize("name", ["readout/choice.json", "tranches/confirm.1.claim.json",
                                  "tranches/confirm.2.claim.json"])
def test_the_plan_refuses_a_file_that_is_not_committed_at_head(tmp_path, name):
    round_ = _round(tmp_path)
    (round_.repo / name).write_text('{"edited": true}\n', encoding="utf-8")
    done = _run("plan", round_, tmp_path)
    assert done.returncode != 0 and done.stdout == ""
    assert "not committed" in done.stderr


def test_the_launch_refuses_before_staging_anything_without_the_claims(tmp_path):
    round_ = rv.committed_round(tmp_path)
    done = _run("launch", round_, tmp_path, **_box(tmp_path))
    assert done.returncode != 0 and "not committed" in done.stderr
    assert not (tmp_path / "ssh.log").exists()


def test_the_launch_starts_the_detached_run_without_a_blocking_lock(tmp_path):
    round_ = _round(tmp_path)
    done = _run("launch", round_, tmp_path, **_box(tmp_path))
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"started: \d+", done.stdout.strip()), done.stdout
    log = _ssh_log(tmp_path)
    assert "setsid" in log and "nohup" in log and "PARALLEL=2" in log
    assert not re.search(r"\bflock\b(?!\s+-\w*n\b)", log)
    deadline = time.monotonic() + 10
    while not (tmp_path / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (tmp_path / "started").exists()


def test_the_pause_touches_the_pause_file_on_the_box_and_stages_nothing(tmp_path):
    round_ = _round(tmp_path)
    env = _box(tmp_path)
    done = _run("pause", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    log = _ssh_log(tmp_path)
    assert f"touch '{env['ISSUE_ROOT']}/PAUSE'" in log or f"touch {env['ISSUE_ROOT']}/PAUSE" in log
    assert "nohup" not in log and "bundle" not in log


def test_the_resume_clears_the_pause_and_restarts_a_resumed_run(tmp_path):
    round_ = _round(tmp_path)
    env = _box(tmp_path)
    done = _run("resume", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    log = _ssh_log(tmp_path)
    assert re.search(rf"rm -f '?{re.escape(env['ISSUE_ROOT'])}/PAUSE", log)
    assert "run --resume" in log and "setsid" in log
    assert log.index("PAUSE") < log.index("run --resume")

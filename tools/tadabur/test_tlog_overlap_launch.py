"""Launcher tests for the tlog S4 run (``scripts/issue-189-overlap.sh``, #189).

The documented ``external`` / ``build`` commands stage a clone **on cuda-dev** and then sweep
there: nothing under ``$ISSUE_ROOT`` exists on this Mac. These tests put a stub ``ssh`` on
``PATH`` — the same idiom ``training/test_consonant_baseline.py`` uses — whose "remote" root is
a tmp directory that the commands are rewritten to, so they prove the sweep checks the box's
checkout rather than a local one, and still refuses a box that was never staged.

Test command, from ``tools/``::

    python -m pytest -q tadabur/test_tlog_overlap_launch.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "issue-189-overlap.sh"


def _mock(path: Path, body: str) -> None:
    """An executable stand-in for ``ssh`` or ``setsid``."""
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _endpoint(tmp_path: Path, *, staged: bool) -> tuple[dict[str, str], Path, Path]:
    """A stub box: its own root, a stub ``ssh`` that rewrites the remote path onto it, and the
    environment pointing the launcher at a *local* root that does not exist."""
    box = tmp_path / "box"
    (box / "logs").mkdir(parents=True)
    if staged:
        (box / "repo").mkdir()
    absent = tmp_path / "nowhere" / "s4"
    bindir, log = tmp_path / "bin", tmp_path / "ssh.log"
    bindir.mkdir()
    _mock(bindir / "ssh", '#!/bin/sh\n'
          'printf "%s\\n" "$2" >> "$SSH_LOG"\n'
          'cmd=$(printf "%s" "$2" | sed "s|$FAKE_ROOT|$REAL_ROOT|g")\n'
          'sh -c "$cmd"\n')
    _mock(bindir / "setsid", '#!/bin/sh\nexit 0\n')
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "HOST": "box",
           "ISSUE_ROOT": str(absent), "SSH_LOG": str(log),
           "FAKE_ROOT": str(absent), "REAL_ROOT": str(box)}
    return env, log, absent


def _sweep(env: dict[str, str], stage: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), stage], env=env, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL, timeout=30)


def test_the_sweep_runs_against_the_boxes_checkout_not_a_local_one(tmp_path):
    """Only the remote checkout needs to exist: the launcher is driven from a Mac whose
    ``$ISSUE_ROOT`` is never created (R1.1)."""
    env, log, absent = _endpoint(tmp_path, staged=True)

    done = _sweep(env, "external")

    assert done.returncode == 0, done.stderr
    assert "started external: pid" in done.stdout
    assert not absent.exists(), "the sweep required a local checkout"
    assert "test -d" in log.read_text(encoding="utf-8")


def test_the_sweep_still_refuses_a_box_without_a_checkout(tmp_path):
    env, log, _ = _endpoint(tmp_path, staged=False)

    done = _sweep(env, "external")

    assert done.returncode != 0
    assert "run 'launch' first" in done.stderr
    assert "test -d" in log.read_text(encoding="utf-8")

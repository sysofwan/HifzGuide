"""The round-2 decode launcher (#230): ``scripts/issue-230-decode.sh``, run in dry-run.

The launcher is round 1's ``issue-179-decode.sh`` pattern for round 2: it stages the repository
as a bundle clone on cuda-dev and runs ``training.aug_readout_decode --spec consonant2`` for the
chosen candidate and control on the four non-tlog sets, then G6's held-out caches. The sub-
commands these tests drive need neither the box nor a GPU:

* ``plan``: prints, from the **committed** ``choice.json``, one line per decode job and per G6
  cache, and nothing else::

      decode <set> <candidate model>,<control model>      (tarteel_v1.confirm, qdat, qud_other,
                                                            tadabur_dev_unseen, in that order)
      held-out <model>                                     (candidate, then control)

  It refuses (non-zero exit, "not committed" or "already" on stderr, nothing on stdout) unless
  ``choice.json``, ``look.json`` and ``look.claim.json`` are committed and unchanged at HEAD, and
  once a decode of a chosen model is already committed in the round's ``decodes/`` (the sitting
  reads once). ``ROUND_DIR`` (default ``tools/tadabur/consonant2_readout``) names the round's
  directory relative to the repository root; the script runs in the repository it is invoked in.
* ``launch``: the same refusals, then stages the repository and starts ``run`` detached on the
  box under ``flock /root/scratch/gpu.lock``, printing ``started: <remote pid>``.

Round 1's ``--spec consonant1``, its confirm sets and its candidate-only models appear nowhere.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from training import round2_verdict_fixtures as rv

TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS.parent / "scripts" / "issue-230-decode.sh"
DECODE_LINES = [
    "decode tarteel_v1.confirm r2_candidate_ema,r2_control_ema",
    "decode qdat r2_candidate_ema,r2_control_ema",
    "decode qud_other r2_candidate_ema,r2_control_ema",
    "decode tadabur_dev_unseen r2_candidate_ema,r2_control_ema",
    "held-out r2_candidate_ema",
    "held-out r2_control_ema",
]


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    shim = tmp_path / "shim"
    shim.mkdir(exist_ok=True)
    python = shim / "python"
    if not python.exists():
        python.symlink_to(sys.executable)
    return {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}", "ROUND_DIR": "readout",
            **extra}


def _run(command: str, round_: rv.Round, tmp_path: Path, **extra: str):
    return subprocess.run(["bash", str(SCRIPT), command], cwd=round_.repo,
                          env=_env(tmp_path, **extra), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60)


def test_the_script_exists_parses_and_is_round_2_only():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--spec consonant2" in text and "held-out" in text
    assert "consonant1" not in text
    assert "--set tlog" not in text and "tlog.confirm" not in text


def test_the_plan_lists_the_chosen_weights_decodes_and_g6_caches(tmp_path):
    round_ = rv.committed_round(tmp_path)
    done = _run("plan", round_, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == DECODE_LINES


def test_the_plan_follows_the_committed_choice_not_a_fixed_weight(tmp_path):
    round_ = rv.committed_round(tmp_path, rv.MIXED_FLAGS)
    done = _run("plan", round_, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        line.replace("r2_candidate_ema", "r2_candidate_live") for line in DECODE_LINES]


@pytest.mark.parametrize("name", ["choice.json", "look.json", "look.claim.json"])
def test_the_plan_refuses_a_file_that_is_not_committed_at_head(tmp_path, name):
    round_ = rv.committed_round(tmp_path)
    (round_.layout.root / name).write_text('{"edited": true}\n', encoding="utf-8")
    done = _run("plan", round_, tmp_path)
    assert done.returncode != 0 and done.stdout == ""
    assert "not committed" in done.stderr


def test_the_plan_refuses_a_checkout_without_the_choice(tmp_path):
    from training import round2_fixtures as fx

    repo = fx.init_repo(tmp_path / "bare")
    done = subprocess.run(["bash", str(SCRIPT), "plan"], cwd=repo, env=_env(tmp_path),
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode != 0 and done.stdout == ""
    assert "not committed" in done.stderr


def test_the_plan_refuses_once_a_decode_of_a_chosen_model_is_committed(tmp_path):
    round_ = rv.committed_round(tmp_path)
    round_.write_commit(round_.layout.root / "decodes" / "qdat.r2_candidate_ema.jsonl.gz", "x",
                        "a decode")
    done = _run("plan", round_, tmp_path)
    assert done.returncode != 0 and done.stdout == ""
    assert "already" in done.stderr


def test_the_plan_ignores_the_look_decodes_of_the_same_models(tmp_path):
    """The round's look decodes name the arms too; only the sets this run decodes count."""
    round_ = rv.committed_round(tmp_path)
    for model in ("r2_candidate_ema", "r2_control_ema"):
        round_.write_commit(round_.layout.root / "decodes" / f"tlog.look.{model}.jsonl.gz",
                            "x", "a look decode")
    done = _run("plan", round_, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == DECODE_LINES


# --- launch ---------------------------------------------------------------------------------------


def _mock(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _box(tmp_path: Path) -> dict[str, str]:
    """``ssh`` and ``flock`` stand-ins: ssh logs its remote command, swallows stdin, and runs a
    ``nohup`` line locally; the box's stub script writes a marker."""
    root = tmp_path / "issue"
    (root / "logs").mkdir(parents=True)
    (root / "repo" / "scripts").mkdir(parents=True)
    marker = tmp_path / "started"
    _mock(root / "repo" / "scripts" / "issue-230-decode.sh",
          f"#!/bin/sh\nprintf 'ran\\n' > '{marker}'\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _mock(bindir / "ssh", '#!/bin/sh\nprintf "%s\\n" "$2" >> "$SSH_LOG"\n'
          'case "$2" in\n  *nohup*) exec sh -c "$2" ;;\n  *) cat >/dev/null ;;\nesac\n')
    _mock(bindir / "flock", '#!/bin/sh\nshift\nexec "$@"\n')
    return {"PATH": f"{bindir}:{_env(tmp_path)['PATH']}", "HOST": "box",
            "ISSUE_ROOT": str(root), "SSH_LOG": str(tmp_path / "ssh.log")}


def test_the_launch_refuses_before_staging_anything_without_a_committed_choice(tmp_path):
    from training import round2_fixtures as fx

    repo = fx.init_repo(tmp_path / "bare")
    done = subprocess.run(["bash", str(SCRIPT), "launch"], cwd=repo,
                          env=_env(tmp_path, **_box(tmp_path)), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode != 0 and "not committed" in done.stderr
    assert not (tmp_path / "ssh.log").exists()


def test_the_launch_starts_the_detached_run_on_the_box_under_the_gpu_lock(tmp_path):
    round_ = rv.committed_round(tmp_path)
    box = _box(tmp_path)
    done = _run("launch", round_, tmp_path, **box)
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"started: \d+", done.stdout.strip()), done.stdout
    assert "nohup flock /root/scratch/gpu.lock" in (tmp_path / "ssh.log").read_text(
        encoding="utf-8")
    deadline = time.monotonic() + 10
    while not (tmp_path / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (tmp_path / "started").exists()

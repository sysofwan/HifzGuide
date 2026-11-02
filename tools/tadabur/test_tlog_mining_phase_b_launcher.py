"""The mining launcher's phase B (#263), run dry: ``scripts/issue-263-mining-decode.sh``.

Phase A's contract is ``test_tlog_mining_launcher.py``; this file adds phase B's.

* ``plan B`` prints, from the committed choice, the two committed confirm claims and the
  committed amendment, one decode line per mined tranche and the merge, and writes nothing::

      decode confirm.1 base,c95,r2_candidate_ema at 2 processes
      decode confirm.2 base,c95,r2_candidate_ema at 2 processes
      merge mined

  It refuses (non-zero exit, nothing on stdout, ``amendment`` on stderr) unless
  ``docs/tlog-mining-amendment.md`` at HEAD is the amendment the code pins by SHA-256
  (``tadabur.tlog_mining.selection.MINING_AMENDMENT_SHA256``): missing, or committed with other
  bytes; the message names ``amendment`` and ``SHA-256``. A claim or the choice not committed refuses as in phase A (``not committed``).
* ``launch B`` and ``resume B``: the same refusals, then the phase A staging and a detached
  ``run`` whose command line names phase ``B``; never a blocking ``flock``.
* Phase ``C`` stays refused by ``plan`` and ``launch`` (``phase C`` and ``round 3`` on stderr,
  nothing staged), and a phase left out still means ``A``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from tadabur.test_tlog_mining_launcher import PLAN, _box, _run, _ssh_log
from training import round2_verdict_fixtures as rv
from training.test_consonant2_confirm_launcher import _round

REPO = Path(__file__).resolve().parents[2]
AMENDMENT = REPO / "docs" / "tlog-mining-amendment.md"
PLAN_B = ["decode confirm.1 base,c95,r2_candidate_ema at 2 processes",
          "decode confirm.2 base,c95,r2_candidate_ema at 2 processes", "merge mined"]


def _with_amendment(tmp_path: Path, content: bytes | None = None) -> rv.Round:
    """The confirm launcher's committed round plus the amendment, committed at HEAD."""
    round_ = _round(tmp_path)
    path = round_.repo / "docs" / "tlog-mining-amendment.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(AMENDMENT.read_bytes() if content is None else content)
    round_.commit(path, message="the amendment, as pinned" if content is None else "edited")
    return round_


def _nohup_line(tmp_path: Path) -> str:
    return next(line for line in _ssh_log(tmp_path).splitlines() if "nohup" in line)


def test_the_plan_lists_phase_b(tmp_path):
    round_ = _with_amendment(tmp_path)
    done = _run("plan", round_, tmp_path, "B")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == PLAN_B


def test_a_phase_left_out_is_still_phase_a(tmp_path):
    round_ = _with_amendment(tmp_path)
    assert _run("plan", round_, tmp_path).stdout.splitlines() == PLAN


@pytest.mark.parametrize("amendment", [None, b"not the pinned amendment\n",
                                       AMENDMENT.read_bytes() + b"\nA late edit.\n"],
                         ids=["missing", "other-bytes", "late-edit"])
@pytest.mark.parametrize("command", ["plan", "launch", "resume"])
def test_phase_b_is_refused_before_staging_unless_the_amendment_is_the_pinned_one(
        tmp_path, command, amendment):
    round_ = _round(tmp_path) if amendment is None else _with_amendment(tmp_path, amendment)
    done = _run(command, round_, tmp_path, "B", **_box(tmp_path))
    assert done.returncode != 0 and done.stdout == ""
    assert "amendment" in done.stderr and "sha-256" in done.stderr.lower()
    assert not (tmp_path / "ssh.log").exists()


@pytest.mark.parametrize("name", ["readout/choice.json", "tranches/confirm.1.claim.json",
                                  "tranches/confirm.2.claim.json"])
def test_phase_b_still_refuses_an_input_that_is_not_committed_at_head(tmp_path, name):
    round_ = _with_amendment(tmp_path)
    (round_.repo / name).write_text('{"edited": true}\n', encoding="utf-8")
    done = _run("plan", round_, tmp_path, "B")
    assert done.returncode != 0 and done.stdout == ""
    assert "not committed" in done.stderr


@pytest.mark.parametrize("command", ["plan", "launch"])
def test_phase_c_is_refused_and_names_round_3(tmp_path, command):
    round_ = _with_amendment(tmp_path)
    done = _run(command, round_, tmp_path, "C", **_box(tmp_path))
    assert done.returncode != 0 and done.stdout == ""
    assert "phase C" in done.stderr and "round 3" in done.stderr
    assert "#262" not in done.stderr
    assert not (tmp_path / "ssh.log").exists()


def test_the_phase_b_launch_starts_a_detached_run_that_names_phase_b(tmp_path):
    round_ = _with_amendment(tmp_path)
    done = _run("launch", round_, tmp_path, "B", **_box(tmp_path))
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"started: \d+", done.stdout.strip()), done.stdout
    line = _nohup_line(tmp_path)
    assert "setsid" in line and re.search(r"decode\.sh'? run\b[^\n]*\bB\b", line), line
    assert not re.search(r"\bflock\b(?!\s+-\w*n\b)", _ssh_log(tmp_path))
    deadline = time.monotonic() + 10
    while not (tmp_path / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (tmp_path / "started").exists()


def test_the_phase_b_resume_clears_the_pause_and_restarts_a_resumed_phase_b_run(tmp_path):
    round_ = _with_amendment(tmp_path)
    env = _box(tmp_path)
    done = _run("resume", round_, tmp_path, "B", **env)
    assert done.returncode == 0, done.stderr
    log = _ssh_log(tmp_path)
    assert re.search(rf"rm -f '?{re.escape(env['ISSUE_ROOT'])}/PAUSE", log)
    line = _nohup_line(tmp_path)
    assert "--resume" in line and re.search(r"decode\.sh'? run\b[^\n]*\bB\b", line), line
    assert log.index("PAUSE") < log.index("--resume")


def test_plan_b_refuses_a_committed_choice_that_names_another_candidate(tmp_path):
    """Phase B's producer set is frozen at round 2's EMA: the dry run must catch it, not a worker."""
    round_ = _with_amendment(tmp_path)
    choice = json.loads(round_.layout.choice.read_text(encoding="utf-8"))
    choice["models"]["candidate"] = "r2_candidate_live"
    round_.write_commit(round_.layout.choice, json.dumps(choice), message="another candidate")
    done = _run("plan", round_, tmp_path, "B")
    assert done.returncode != 0 and done.stdout == ""
    assert "candidate" in done.stderr and "r2_candidate_ema" in done.stderr


def test_launch_refuses_while_the_other_phases_loop_is_live(tmp_path):
    """The clone is shared: staging under a live phase A loop breaks that loop's next batch."""
    round_ = _with_amendment(tmp_path)
    env = _box(tmp_path)
    ssh = Path(env["PATH"].split(":")[0]) / "ssh"
    ssh.write_text('#!/bin/sh\nprintf "%s\\n" "$2" >> "$SSH_LOG"\n'
                   'case "$2" in\n'
                   '  *"/B/"*) ;;\n'
                   '  *logs/run.pid*) echo 4242 ;;\n'
                   'esac\n'
                   'cat >/dev/null\n', encoding="utf-8")
    done = _run("launch", round_, tmp_path, "B", **env)
    assert done.returncode != 0 and "still live" in done.stderr and "4242" in done.stderr
    assert "bundle" not in _ssh_log(tmp_path)

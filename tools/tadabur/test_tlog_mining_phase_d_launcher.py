"""The mining launcher's phase D (#263), run dry: ``scripts/issue-263-mining-decode.sh``.

Phases A, B and C are ``test_tlog_mining_launcher.py``, ``test_tlog_mining_phase_b_launcher.py``
and ``test_tlog_mining_phase_c_launcher.py``; this file adds phase D's: the base teacher's word
times over phase B's mined rows.

* Phase D is accepted by ``plan``, ``launch`` and ``resume`` only under the pinned amendment
  (``amendment`` and ``SHA-256`` on stderr), and refused before anything is staged (non-zero
  exit, nothing on stdout, nothing sent over ``ssh``) otherwise. Round 3's choice and verdict are
  **not** needed. The two confirm claims and the committed ``choice.json`` must be committed, as
  for every phase (``not committed``). Phase D aligns with the teacher alone, so the candidate the
  choice names is irrelevant to it.
* ``plan D`` prints one align line per mined tranche and the merge, and writes nothing::

      align confirm.1 base at 2 processes
      align confirm.2 base at 2 processes
      merge mined word_times

* ``launch D`` / ``resume D``: the phase A staging and a detached ``run`` whose command line names
  phase ``D``, logging to ``<ISSUE_ROOT>/D/logs/run.out``. Phase D's parts, clips, shards, logs
  and out directories are all under ``<ISSUE_ROOT>/D``, and a live loop of any phase blocks a
  restage (the clone is shared).
* ``status D`` reads ``<ISSUE_ROOT>/D/logs`` only.
* ``fetch D`` refuses unless the box holds a complete merge in ``<ISSUE_ROOT>/D/out`` (``run.exit``
  0 and all of ``mined.confirm.1.jsonl.gz``, ``mined.confirm.2.jsonl.gz`` and
  ``meta.mined.json``), then copies the out directory into ``tools/tadabur/tlog_mining/
  word_times`` (``FETCH_DIR`` overrides) **without touching** phase A's ``look.jsonl.gz``,
  ``confirm.1.jsonl.gz``, ``confirm.2.jsonl.gz`` or ``meta.json``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from tadabur.test_tlog_mining_launcher import PLAN, SCRIPT, _box, _run, _ssh_log
from tadabur.test_tlog_mining_phase_b_launcher import AMENDMENT, _nohup_line, _with_amendment
from tadabur.test_tlog_mining_phase_c_launcher import PLAN_C, _phase_c_round
from training.test_consonant2_confirm_launcher import _env, _mock, _round

PLAN_D = ["align confirm.1 base at 2 processes", "align confirm.2 base at 2 processes",
          "merge mined word_times"]
#: Phase D's merged files, as ``fetch`` checks them.
PHASE_D_NAMES = ["mined.confirm.1.jsonl.gz", "mined.confirm.2.jsonl.gz", "meta.mined.json"]
PHASE_A_FILES = ["look.jsonl.gz", "confirm.1.jsonl.gz", "confirm.2.jsonl.gz", "meta.json"]


def _d(command: str, round_, tmp_path: Path, *args: str, **extra: str):
    return _run(command, round_, tmp_path, "D", *args, **extra)


def test_the_plan_lists_phase_d_and_writes_nothing(tmp_path):
    round_ = _with_amendment(tmp_path)
    done = _d("plan", round_, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == PLAN_D
    assert not (tmp_path / "ssh.log").exists()


def test_phase_d_needs_no_round_3_choice_or_verdict(tmp_path):
    round_ = _with_amendment(tmp_path)
    done = _d("plan", round_, tmp_path, ROUND3_DIR="no-round-3")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == PLAN_D


def test_the_other_phases_plans_are_unchanged(tmp_path):
    round_ = _phase_c_round(tmp_path)
    assert _run("plan", round_, tmp_path).stdout.splitlines() == PLAN
    assert _run("plan", round_, tmp_path, "C", ROUND3_DIR="round3").stdout.splitlines() == PLAN_C


@pytest.mark.parametrize("amendment", [None, b"not the pinned amendment\n",
                                       AMENDMENT.read_bytes() + b"\nA late edit.\n"],
                         ids=["missing", "other-bytes", "late-edit"])
@pytest.mark.parametrize("command", ["plan", "launch", "resume"])
def test_phase_d_is_refused_before_staging_unless_the_amendment_is_the_pinned_one(
        tmp_path, command, amendment):
    round_ = _round(tmp_path) if amendment is None else _with_amendment(tmp_path, amendment)
    done = _d(command, round_, tmp_path, **_box(tmp_path))
    assert done.returncode != 0 and done.stdout == ""
    assert "amendment" in done.stderr and "sha-256" in done.stderr.lower()
    assert not (tmp_path / "ssh.log").exists()


@pytest.mark.parametrize("name", ["readout/choice.json", "tranches/confirm.1.claim.json",
                                  "tranches/confirm.2.claim.json"])
def test_phase_d_still_refuses_an_input_that_is_not_committed_at_head(tmp_path, name):
    round_ = _with_amendment(tmp_path)
    (round_.repo / name).write_text('{"edited": true}\n', encoding="utf-8")
    done = _d("plan", round_, tmp_path)
    assert done.returncode != 0 and done.stdout == ""
    assert "not committed" in done.stderr


def test_the_phase_d_launch_starts_a_detached_run_that_names_phase_d_and_logs_under_d(tmp_path):
    round_ = _with_amendment(tmp_path)
    env = _box(tmp_path)
    done = _d("launch", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"started: \d+", done.stdout.strip()), done.stdout
    line = _nohup_line(tmp_path)
    assert "setsid" in line and re.search(r"decode\.sh'? run\b[^\n]*\bD\b", line), line
    root = re.escape(env["ISSUE_ROOT"])
    assert re.search(rf">> '?{root}/D/logs/run\.out", line), line
    assert not re.search(rf"{root}/((B|C)/)?logs/run\.out", line)
    assert not re.search(r"\bflock\b(?!\s+-\w*n\b)", _ssh_log(tmp_path))
    deadline = time.monotonic() + 10
    while not (tmp_path / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (tmp_path / "started").exists()


def test_the_phase_d_resume_clears_the_pause_and_restarts_a_resumed_phase_d_run(tmp_path):
    round_ = _with_amendment(tmp_path)
    env = _box(tmp_path)
    done = _d("resume", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    log = _ssh_log(tmp_path)
    assert re.search(rf"rm -f '?{re.escape(env['ISSUE_ROOT'])}/PAUSE", log)
    line = _nohup_line(tmp_path)
    assert "--resume" in line and re.search(r"decode\.sh'? run\b[^\n]*\bD\b", line), line
    assert log.index("PAUSE") < log.index("--resume")


def _live_box(tmp_path: Path, live: str) -> dict[str, str]:
    """A box whose ``run.pid`` check answers 4242 for the logs directory containing ``live``."""
    env = _box(tmp_path)
    ssh = Path(env["PATH"].split(":")[0]) / "ssh"
    _mock(ssh, '#!/bin/sh\nprintf "%s\\n" "$2" >> "$SSH_LOG"\n'
          'case "$2" in\n'
          f'  *"{live}"*logs/run.pid*) echo 4242 ;;\n'
          'esac\ncat >/dev/null\n')
    return env


@pytest.mark.parametrize("live, phase", [("/D/", "A"), ("/D/", "B"), ("/D/", "C"),
                                         ("/B/", "D"), ("/C/", "D")])
def test_a_live_loop_of_another_phase_blocks_a_restage(tmp_path, live, phase):
    """The clone is shared by all four phases: restaging under a live loop breaks its batch."""
    round_ = _phase_c_round(tmp_path)
    done = _run("launch", round_, tmp_path, phase, ROUND3_DIR="round3",
                **_live_box(tmp_path, live))
    assert done.returncode != 0 and "still live" in done.stderr and "4242" in done.stderr
    assert "bundle" not in _ssh_log(tmp_path)


def test_a_live_phase_a_loop_blocks_a_phase_d_launch(tmp_path):
    round_ = _with_amendment(tmp_path)
    env = _box(tmp_path)
    ssh = Path(env["PATH"].split(":")[0]) / "ssh"
    _mock(ssh, '#!/bin/sh\nprintf "%s\\n" "$2" >> "$SSH_LOG"\n'
          'case "$2" in\n  *"/B/"*|*"/C/"*|*"/D/"*) ;;\n  *logs/run.pid*) echo 4242 ;;\nesac\n'
          'cat >/dev/null\n')
    done = _d("launch", round_, tmp_path, **env)
    assert done.returncode != 0 and "still live" in done.stderr
    assert "bundle" not in _ssh_log(tmp_path)


# --- status and fetch read phase D's own directories -------------------------------------------


def _merge_box(tmp_path: Path, names: list[str], *, exit_code: str = "0\n") -> dict[str, str]:
    """A box with a complete phase D merge under ``D/`` and a *different* finished run in every
    other phase's directories. ``ssh`` executes the remote script it is handed and ``rsync``
    copies what it is pointed at, so the generated checks are run, not merely logged."""
    root = tmp_path / "mining"
    for phase, phase_root in (("A", root), ("B", root / "B"), ("C", root / "C"),
                              ("D", root / "D")):
        out, logs = phase_root / "out", phase_root / "logs"
        out.mkdir(parents=True)
        logs.mkdir(parents=True)
        (logs / "run.exit").write_text("0\n" if phase == "D" else "1\n", encoding="utf-8")
        (logs / "run.done").write_text(f"phase {phase}\n", encoding="utf-8")
        (out / f"foreign-{phase}.txt").write_text(f"not phase {phase}'s\n", encoding="utf-8")
    (root / "D" / "logs" / "run.exit").write_text(exit_code, encoding="utf-8")
    for name in names:
        (root / "D" / "out" / name).write_text("", encoding="utf-8")
    bindir = tmp_path / "fetch-bin"
    bindir.mkdir()
    _mock(bindir / "ssh", '#!/bin/sh\nprintf "%s\\n" "$*" >> "$SSH_LOG"\nshift\n'
          'if [ $# -ge 2 ]; then exec "$@"; else exec sh -c "$1"; fi\n')
    _mock(bindir / "rsync", '#!/bin/sh\nprintf "%s\\n" "$*" >> "$RSYNC_LOG"\n'
          'mkdir -p "$3"\ncp -a "${2#*:}." "$3"\n')
    return {"PATH": f"{bindir}:{_env(tmp_path)['PATH']}", "HOST": "box",
            "ISSUE_ROOT": str(root), "SSH_LOG": str(tmp_path / "ssh.log"),
            "RSYNC_LOG": str(tmp_path / "rsync.log"), "FETCH_DIR": str(tmp_path / "fetched")}


def test_the_status_reads_phase_ds_logs_only(tmp_path):
    round_ = _with_amendment(tmp_path)
    env = _merge_box(tmp_path, PHASE_D_NAMES)
    done = _d("status", round_, tmp_path, **env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.startswith("done: exit 0")


def test_a_pending_phase_d_is_not_mistaken_for_another_phases_finished_run(tmp_path):
    round_ = _with_amendment(tmp_path)
    env = _merge_box(tmp_path, PHASE_D_NAMES)
    (tmp_path / "mining" / "D" / "logs" / "run.done").unlink()
    done = _d("status", round_, tmp_path, **env)
    assert done.returncode == 3 and done.stdout.startswith("pending")
    assert f"{env['ISSUE_ROOT']}/D/logs/run.done" in done.stdout


def test_the_phase_d_fetch_copies_a_complete_merge_from_ds_out_dir_only(tmp_path):
    round_ = _with_amendment(tmp_path)
    env = _merge_box(tmp_path, PHASE_D_NAMES)
    done = _d("fetch", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    copied = sorted(path.name for path in Path(env["FETCH_DIR"]).iterdir())
    assert copied == sorted([*PHASE_D_NAMES, "foreign-D.txt"])
    assert f"box:{env['ISSUE_ROOT']}/D/out/" in (tmp_path / "rsync.log").read_text(
        encoding="utf-8")


def test_the_phase_d_fetch_goes_to_the_word_times_directory_by_default(tmp_path):
    round_ = _with_amendment(tmp_path)
    env = _merge_box(tmp_path, PHASE_D_NAMES)
    del env["FETCH_DIR"]
    done = _d("fetch", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    assert (round_.repo / "tools" / "tadabur" / "tlog_mining" / "word_times"
            / "meta.mined.json").is_file()


def test_the_phase_d_fetch_leaves_phase_as_committed_files_byte_identical(tmp_path):
    round_ = _with_amendment(tmp_path)
    env = _merge_box(tmp_path, PHASE_D_NAMES)
    fetched = Path(env["FETCH_DIR"])
    fetched.mkdir()
    for name in PHASE_A_FILES:
        (fetched / name).write_bytes(f"phase A's {name}".encode())
    done = _d("fetch", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    for name in PHASE_A_FILES:
        assert (fetched / name).read_bytes() == f"phase A's {name}".encode()
    assert not set(PHASE_A_FILES) & {path.name for path in
                                     (tmp_path / "mining" / "D" / "out").iterdir()}


@pytest.mark.parametrize("missing", PHASE_D_NAMES)
def test_the_phase_d_fetch_refuses_a_merge_missing_one_of_its_files(tmp_path, missing):
    round_ = _with_amendment(tmp_path)
    env = _merge_box(tmp_path, [name for name in PHASE_D_NAMES if name != missing])
    done = _d("fetch", round_, tmp_path, **env)
    assert done.returncode != 0 and "no complete merge" in done.stderr
    assert not (tmp_path / "rsync.log").exists()


def test_the_phase_d_fetch_does_not_take_phase_as_files_for_its_own(tmp_path):
    """A box whose D out holds phase A's names (but not D's) is not a complete D merge."""
    round_ = _with_amendment(tmp_path)
    env = _merge_box(tmp_path, [*PHASE_A_FILES])
    done = _d("fetch", round_, tmp_path, **env)
    assert done.returncode != 0 and "no complete merge" in done.stderr
    assert not (tmp_path / "rsync.log").exists()


@pytest.mark.parametrize("exit_code", ["3\n", "1\n"])
def test_the_phase_d_fetch_refuses_a_run_that_did_not_exit_zero(tmp_path, exit_code):
    round_ = _with_amendment(tmp_path)
    env = _merge_box(tmp_path, PHASE_D_NAMES, exit_code=exit_code)
    done = _d("fetch", round_, tmp_path, **env)
    assert done.returncode != 0 and "no complete merge" in done.stderr
    assert not (tmp_path / "rsync.log").exists()


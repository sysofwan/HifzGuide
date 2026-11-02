"""Acceptance tests for round 3's decode launcher, ``scripts/issue-256-decode.sh`` (#256), written
before the script and **locked**: later turns may add tests but never weaken or rewrite these.

One script, three phases, modelled on current main's ``issue-230-decode.sh`` and
``issue-230-confirm-decode.sh`` (copied from them, which stay untouched)::

    scripts/issue-256-decode.sh <phase> <command>        phase: look | gates | confirm
                                                          command: plan | launch | resume | check | fetch
                                                          (box side: run [--resume], decode-phase)

Like ``test_consonant2_launcher`` and ``test_consonant2_confirm_launcher`` these drive the real
script's sub-commands against a temporary git repository and stand-ins for ``ssh``, ``flock``,
``rsync``, ``timeout`` and ``python``; nothing touches a box or a GPU.

The interface the tests pin (``ROUND_DIR`` default ``tools/tadabur/consonant3_readout``,
``TRANCHES_DIR`` default ``tools/tadabur/tlog/tranches``, both relative to the repository the
script runs in; ``DECODES`` is ``$ROUND_DIR/decodes``):

* **Box layout**: each phase owns ``$ISSUE_ROOT/<phase>/{logs,parts,out,decodes}`` (default
  ``ISSUE_ROOT=/root/scratch/issue-256``); the staged clone is shared at ``$ISSUE_ROOT/repo``.
  A run writes ``logs/run.exit`` first, then ``logs/run.done``, and a worker line
  ``<ts> <role> worker <index> exit <rc>`` (look, confirm) or ``<ts> <set> shard <i> exit <rc>``
  (gates) to ``logs/events``. Two phases never share a done/exit file. Sourced rather than
  run, the script takes the phase from ``$PHASE`` and dispatches nothing, so ``decode`` and
  ``record_run_result`` are callable.
* **look** decodes ``r3_candidate_live,r3_candidate_ema`` on the look gating rows with
  ``tadabur.tlog_eval_decode decode --spec consonant3 --role look`` (``PARALLEL`` disjoint
  workers, default 4), then ``merge --spec consonant3``. It reads no choice. The teacher's, C95's
  and round 2's look decodes are reused, so no other model is decoded.
  ``plan`` prints ``decode look <models> at <P> processes`` and ``merge <models>``.
* **gates** decodes the committed choice's candidate (``models.candidate``; round 3 has no
  control) on ``tarteel_v1.confirm``, ``qdat`` (whose stream is ``qdat_bench``), ``qud_other`` and
  ``tadabur_dev_unseen`` with ``training.aug_readout_decode --spec consonant3``, then Ship A's
  held-out cache (``training.consonant_baseline held-out --spec consonant3``).
  ``plan`` prints ``decode <set> <model>`` per set, then ``held-out <model>``.
* **confirm** decodes ``base,c95,<candidate>`` on ``confirm.3`` then ``confirm.4`` (every worker
  of the first tranche finishes before the second starts) and then runs
  ``merge-confirm --spec consonant3``. ``plan`` prints ``decode <role> <models> at <P> processes``
  per tranche and ``merge-confirm <models>``.
* ``plan`` refuses (non-zero exit, ``not committed`` or ``already`` on stderr, nothing on stdout)
  unless the phase's committed inputs are in HEAD and unchanged: gates need ``choice.json``,
  ``look.json`` and ``look.claim.json``; confirm needs ``choice.json`` and **both**
  ``confirm.3.claim.json`` and ``confirm.4.claim.json``; look needs none. A decode of one of the
  phase's models already committed in ``DECODES`` blocks it (the sitting reads once): the look's
  ``tlog.look.<model>``, the gates' ``<set>.<model>``, the confirm's ``tlog.confirm.<model>``;
  another phase's files never block it.
* ``launch`` and ``resume`` repeat the refusals before staging anything, then start
  ``run <phase>`` (``run <phase> --resume``) detached under ``flock /root/scratch/gpu.lock`` at
  ``PARALLEL=4`` (``RESUME_PARALLEL=2`` for a resume: #230's host-RAM OOM at 6).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from training import round2_fixtures as fx

TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS.parent / "scripts" / "issue-256-decode.sh"
PHASES = ("look", "gates", "confirm")
CANDIDATE = "r3_candidate_ema"
LOOK_MODELS = "r3_candidate_live,r3_candidate_ema"
CONFIRM_MODELS = f"base,c95,{CANDIDATE}"
ROLES = ["confirm.3", "confirm.4"]
SETS = ["tarteel_v1.confirm", "qdat", "qud_other", "tadabur_dev_unseen"]
MODELS_OF = {"look": LOOK_MODELS, "gates": CANDIDATE, "confirm": CONFIRM_MODELS}
CHOICE_JSON = json.dumps({"models": {"candidate": CANDIDATE}})
DECODES = "readout/decodes"
PLAN = {
    "look": [f"decode look {LOOK_MODELS} at 4 processes", f"merge {LOOK_MODELS}"],
    "gates": [*(f"decode {name} {CANDIDATE}" for name in SETS), f"held-out {CANDIDATE}"],
    "confirm": [*(f"decode {role} {CONFIRM_MODELS} at 4 processes" for role in ROLES),
                f"merge-confirm {CONFIRM_MODELS}"],
}


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    shim = tmp_path / "shim"
    shim.mkdir(exist_ok=True)
    python = shim / "python"
    if not python.exists():
        python.symlink_to(sys.executable)
    return {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}", "ROUND_DIR": "readout",
            "TRANCHES_DIR": "tranches", **extra}


def _write(repo: Path, name: str, text: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str = "inputs") -> None:
    fx.git(repo, "add", "-A")
    fx.git(repo, "commit", "-q", "--allow-empty", "-m", message)


def _repo(tmp_path: Path, *, choice: str | None = CHOICE_JSON,
          claims: tuple[str, ...] = tuple(ROLES)) -> Path:
    """A repository whose HEAD holds the round's committed choice, look and claims."""
    repo = fx.init_repo(tmp_path / "repo")
    if choice is not None:
        _write(repo, "readout/choice.json", choice)
        _write(repo, "readout/look.json", "{}\n")
        _write(repo, "readout/look.claim.json", "{}\n")
    for role in claims:
        _write(repo, f"tranches/{role}.claim.json", json.dumps({"role": role}))
    _commit(repo)
    return repo


def _run(phase: str, command: str, repo: Path, tmp_path: Path, **extra: str):
    return subprocess.run(["bash", str(SCRIPT), phase, command], cwd=repo,
                          env=_env(tmp_path, **extra), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60)


def _mock(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _refused(done: subprocess.CompletedProcess, *words: str) -> None:
    assert done.returncode != 0 and done.stdout == ""
    assert any(word in done.stderr for word in words), done.stderr


# --- the script ------------------------------------------------------------------------------------


def test_the_script_exists_parses_and_is_round_3_only():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    text = SCRIPT.read_text(encoding="utf-8")
    assert "tadabur.tlog_eval_decode" in text and "training.aug_readout_decode" in text
    assert "merge-confirm" in text and "consonant_baseline" in text
    assert "tools/tadabur/consonant3_readout" in text
    assert "--spec consonant2" not in text and "--spec consonant1" not in text


@pytest.mark.parametrize("args", [(), ("look",), ("midpoint", "plan"), ("look", "fly"),
                                  ("plan",), ("confirm", "run-everything")])
def test_an_unknown_phase_or_command_prints_the_usage(tmp_path, args):
    done = subprocess.run(["bash", str(SCRIPT), *args], cwd=_repo(tmp_path),
                          env=_env(tmp_path), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode == 2 and done.stdout == ""
    assert "usage" in done.stderr


# --- plan: the jobs, from committed inputs ---------------------------------------------------------


@pytest.mark.parametrize("phase", PHASES)
def test_the_plan_lists_the_phase_s_jobs(tmp_path, phase):
    done = _run(phase, "plan", _repo(tmp_path), tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == PLAN[phase]


@pytest.mark.parametrize("phase", ["gates", "confirm"])
def test_the_plan_follows_the_committed_choice_not_a_fixed_weight(tmp_path, phase):
    live = json.dumps({"models": {"candidate": "r3_candidate_live"}})
    done = _run(phase, "plan", _repo(tmp_path, choice=live), tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [line.replace(CANDIDATE, "r3_candidate_live")
                                        for line in PLAN[phase]]


@pytest.mark.parametrize("phase", ["gates", "confirm"])
def test_a_choice_that_still_names_a_control_decodes_no_control(tmp_path, phase):
    both = json.dumps({"models": {"candidate": CANDIDATE, "control": "r2_control_ema"}})
    done = _run(phase, "plan", _repo(tmp_path, choice=both), tmp_path)
    assert done.returncode == 0, done.stderr
    assert "control" not in done.stdout and done.stdout.splitlines() == PLAN[phase]


def test_the_look_plan_needs_no_committed_choice_and_follows_the_parallelism(tmp_path):
    done = _run("look", "plan", _repo(tmp_path, choice=None, claims=()), tmp_path, PARALLEL="6")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == f"decode look {LOOK_MODELS} at 6 processes"


@pytest.mark.parametrize("name", ["readout/choice.json", "readout/look.json",
                                  "readout/look.claim.json"])
def test_the_gates_plan_refuses_a_file_that_is_not_committed_at_head(tmp_path, name):
    repo = _repo(tmp_path)
    (repo / name).write_text('{"edited": true}\n', encoding="utf-8")
    _refused(_run("gates", "plan", repo, tmp_path), "not committed")


@pytest.mark.parametrize("name", ["readout/choice.json", "tranches/confirm.3.claim.json",
                                  "tranches/confirm.4.claim.json"])
def test_the_confirm_plan_refuses_a_file_that_is_not_committed_at_head(tmp_path, name):
    repo = _repo(tmp_path)
    (repo / name).write_text('{"edited": true}\n', encoding="utf-8")
    _refused(_run("confirm", "plan", repo, tmp_path), "not committed")


@pytest.mark.parametrize("claims", [(), ("confirm.3",), ("confirm.4",),
                                    ("confirm.1", "confirm.2")])
def test_the_confirm_plan_needs_both_round_3_claims_committed(tmp_path, claims):
    _refused(_run("confirm", "plan", _repo(tmp_path, claims=claims), tmp_path), "not committed")


def test_the_confirm_plan_refuses_a_claim_that_is_written_but_not_committed(tmp_path):
    repo = _repo(tmp_path, claims=("confirm.3",))
    _write(repo, "tranches/confirm.4.claim.json", "{}\n")
    _refused(_run("confirm", "plan", repo, tmp_path), "not committed")


@pytest.mark.parametrize("phase", ["gates", "confirm"])
def test_the_plan_refuses_a_checkout_without_the_choice(tmp_path, phase):
    _refused(_run(phase, "plan", _repo(tmp_path, choice=None), tmp_path), "not committed")


@pytest.mark.parametrize("phase, decode", [
    ("look", "tlog.look.r3_candidate_live"), ("look", "tlog.look.r3_candidate_ema"),
    ("gates", "qdat.r3_candidate_ema"), ("gates", "tarteel_v1.confirm.r3_candidate_ema"),
    ("gates", "tadabur_dev_unseen.r3_candidate_ema"),
    ("confirm", "tlog.confirm.base"), ("confirm", "tlog.confirm.c95"),
    ("confirm", "tlog.confirm.r3_candidate_ema")])
def test_the_plan_refuses_once_a_decode_of_a_planned_model_is_committed(
        tmp_path, phase, decode):
    repo = _repo(tmp_path)
    _write(repo, f"{DECODES}/{decode}.jsonl.gz", "x")
    _commit(repo, "a decode")
    _refused(_run(phase, "plan", repo, tmp_path), "already")


@pytest.mark.parametrize("phase, other", [
    ("gates", "tlog.look.r3_candidate_ema"), ("gates", "tlog.confirm.r3_candidate_ema"),
    ("confirm", "tlog.look.r3_candidate_ema"), ("confirm", "qdat.r3_candidate_ema"),
    ("look", "tlog.confirm.r3_candidate_ema"), ("look", "qdat.r3_candidate_ema"),
    ("look", "tlog.look.base"), ("look", "tlog.look.c95"), ("look", "tlog.look.r2_candidate_ema")])
def test_the_plan_ignores_decodes_that_are_not_its_own(tmp_path, phase, other):
    """Another phase's files do not block a phase, and the look's reused decodes (the teacher,
    C95, round 2's candidate) are not decodes of the look phase's models."""
    repo = _repo(tmp_path)
    _write(repo, f"{DECODES}/{other}.jsonl.gz", "x")
    _commit(repo, "a decode")
    done = _run(phase, "plan", repo, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == PLAN[phase]


# --- launch and resume ------------------------------------------------------------------------------


def _box(tmp_path: Path, phase: str) -> dict[str, str]:
    """``ssh`` and ``flock`` stand-ins: ssh logs its remote command, swallows stdin, and runs a
    ``nohup`` line locally; the box's stub script writes a marker."""
    root = tmp_path / "issue"
    (root / phase / "logs").mkdir(parents=True)
    (root / "repo" / "scripts").mkdir(parents=True)
    _mock(root / "repo" / "scripts" / "issue-256-decode.sh",
          f"#!/bin/sh\nprintf '%s\\n' \"$*\" > '{tmp_path / 'started'}'\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _mock(bindir / "ssh", '#!/bin/sh\nprintf "%s\\n" "$2" >> "$SSH_LOG"\n'
          'case "$2" in\n  *nohup*) exec sh -c "$2" ;;\n  *) cat >/dev/null ;;\nesac\n')
    _mock(bindir / "flock", '#!/bin/sh\nshift\nexec "$@"\n')
    return {"PATH": f"{bindir}:{_env(tmp_path)['PATH']}", "HOST": "box",
            "ISSUE_ROOT": str(root), "SSH_LOG": str(tmp_path / "ssh.log")}


def _started(tmp_path: Path) -> str:
    deadline = time.monotonic() + 10
    while not (tmp_path / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return (tmp_path / "started").read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("phase", PHASES)
def test_the_launch_starts_the_detached_run_on_the_box_under_the_gpu_lock(tmp_path, phase):
    done = _run(phase, "launch", _repo(tmp_path), tmp_path, **_box(tmp_path, phase))
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"started: \d+", done.stdout.strip()), done.stdout
    log = (tmp_path / "ssh.log").read_text(encoding="utf-8")
    assert "nohup flock /root/scratch/gpu.lock" in log
    assert "PARALLEL=4" in log
    assert _started(tmp_path) == f"run {phase}"


@pytest.mark.parametrize("phase", PHASES)
def test_the_resume_restages_and_starts_a_resumed_run_at_the_low_parallelism(tmp_path, phase):
    done = _run(phase, "resume", _repo(tmp_path), tmp_path, **_box(tmp_path, phase))
    assert done.returncode == 0, done.stderr
    log = (tmp_path / "ssh.log").read_text(encoding="utf-8")
    assert "nohup flock /root/scratch/gpu.lock" in log and "PARALLEL=2" in log
    assert _started(tmp_path) == f"run {phase} --resume"


def test_the_resume_parallelism_follows_resume_parallel(tmp_path):
    done = _run("look", "resume", _repo(tmp_path), tmp_path, **_box(tmp_path, "look"),
                RESUME_PARALLEL="3")
    assert done.returncode == 0, done.stderr
    assert "PARALLEL=3" in (tmp_path / "ssh.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("command", ["launch", "resume"])
@pytest.mark.parametrize("phase, repo_args", [
    ("gates", {"choice": None}), ("confirm", {"choice": None}),
    ("confirm", {"claims": ("confirm.3",)}), ("confirm", {"claims": ()})])
def test_a_launch_refuses_before_staging_anything_without_its_inputs(
        tmp_path, phase, repo_args, command):
    done = _run(phase, command, _repo(tmp_path, **repo_args), tmp_path, **_box(tmp_path, phase))
    assert done.returncode != 0 and "not committed" in done.stderr
    assert not (tmp_path / "ssh.log").exists()


@pytest.mark.parametrize("command", ["launch", "resume"])
def test_a_launch_refuses_before_staging_when_the_decode_is_already_committed(
        tmp_path, command):
    repo = _repo(tmp_path)
    _write(repo, f"{DECODES}/tlog.confirm.c95.jsonl.gz", "x")
    _commit(repo, "a decode")
    done = _run("confirm", command, repo, tmp_path, **_box(tmp_path, "confirm"))
    assert done.returncode != 0 and "already" in done.stderr
    assert not (tmp_path / "ssh.log").exists()


# --- the box's run ------------------------------------------------------------------------------------


def _box_run_env(tmp_path: Path, choice: str | None, *, timeout_body: str = "exit 0",
                 conda: bool = True) -> dict[str, str]:
    """The box's shell shape for ``run``: no ``python`` on PATH until the conda env is
    activated, and a ``timeout`` stub in place of the decode phase (nothing decodes here)."""
    root = tmp_path / "issue"
    (root / "repo" / "readout").mkdir(parents=True, exist_ok=True)
    if choice is not None:
        (root / "repo" / "readout" / "choice.json").write_text(choice, encoding="utf-8")
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    _mock(prefix / "bin" / "python",
          f'#!/bin/sh\ntouch "{tmp_path / "activated"}"\nexec "{sys.executable}" "$@"\n')
    conda_script = tmp_path / "conda.sh"
    if conda:
        conda_script.write_text(f'conda() {{ PATH="{prefix}/bin:$PATH"; export PATH; }}\n',
                                encoding="utf-8")
    bindir = tmp_path / "run-bin"
    bindir.mkdir(parents=True, exist_ok=True)
    _mock(bindir / "timeout",
          f'#!/bin/sh\nprintf "%s" "$MODELS" > "{tmp_path / "models"}"\n{timeout_body}\n')
    return {**os.environ, "PATH": f"{bindir}:/usr/bin:/bin", "CONDA": str(conda_script),
            "ISSUE_ROOT": str(root), "ROUND_DIR": "readout"}


def _run_on_the_box(tmp_path: Path, phase: str, choice: str | None = CHOICE_JSON, *,
                    args: tuple[str, ...] = (), **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), "run", phase, *args],
                          env=_box_run_env(tmp_path, choice, **kwargs),
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)


def _logs(tmp_path: Path, phase: str) -> Path:
    return tmp_path / "issue" / phase / "logs"


@pytest.mark.parametrize("phase", PHASES)
def test_the_run_exports_the_phase_s_models_from_the_activated_env(tmp_path, phase):
    done = _run_on_the_box(tmp_path, phase)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "models").read_text(encoding="utf-8") == MODELS_OF[phase]
    if phase != "look":
        assert (tmp_path / "activated").exists()
    logs = _logs(tmp_path, phase)
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "0"
    assert "exit 0" in (logs / "run.done").read_text(encoding="utf-8")


def test_the_look_run_reads_no_choice(tmp_path):
    done = _run_on_the_box(tmp_path, "look", choice=None)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "models").read_text(encoding="utf-8") == LOOK_MODELS


@pytest.mark.parametrize("phase", ["gates", "confirm"])
def test_the_run_ignores_a_control_in_the_choice(tmp_path, phase):
    both = json.dumps({"models": {"candidate": CANDIDATE, "control": "r2_control_ema"}})
    assert _run_on_the_box(tmp_path, phase, both).returncode == 0
    assert (tmp_path / "models").read_text(encoding="utf-8") == MODELS_OF[phase]


@pytest.mark.parametrize("phase", ["gates", "confirm"])
def test_a_run_that_dies_before_the_decode_phase_still_records_the_failure(tmp_path, phase):
    done = _run_on_the_box(tmp_path, phase, "{edited")
    logs = _logs(tmp_path, phase)
    assert done.returncode != 0
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == str(done.returncode)
    assert (logs / "run.done").exists()


@pytest.mark.parametrize("phase", PHASES)
def test_a_missing_conda_script_is_a_recorded_failure_not_a_pass(tmp_path, phase):
    done = _run_on_the_box(tmp_path, phase, conda=False)
    logs = _logs(tmp_path, phase)
    assert done.returncode != 0
    assert "no conda" in done.stderr
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == str(done.returncode)
    assert (logs / "run.done").exists()


@pytest.mark.parametrize("phase", PHASES)
def test_a_failed_decode_phase_is_recorded_for_check(tmp_path, phase):
    done = _run_on_the_box(tmp_path, phase, timeout_body="exit 7")
    logs = _logs(tmp_path, phase)
    assert done.returncode == 7
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "7"
    assert "exit 7" in (logs / "run.done").read_text(encoding="utf-8")


def test_a_budget_expiry_is_recorded_as_the_budget(tmp_path):
    """GNU timeout's own status is 124: that, and only that, is the budget stop."""
    done = _run_on_the_box(tmp_path, "confirm", timeout_body="exit 124")
    logs = _logs(tmp_path, "confirm")
    assert done.returncode == 2
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "2"
    assert "budget-exceeded" in (logs / "run.done").read_text(encoding="utf-8")


def test_a_timeout_escalation_is_recorded_as_an_ambiguous_sigkill(tmp_path):
    """``timeout --kill-after`` escalates a budget expiry to SIGKILL, whose status is 137; the
    record must not call that "not the budget": it cannot tell escalation from an external kill
    or the OOM killer."""
    done = _run_on_the_box(tmp_path, "look", timeout_body="exit 137")
    logs = _logs(tmp_path, "look")
    assert done.returncode == 137
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "137"
    recorded = (logs / "run.done").read_text(encoding="utf-8")
    assert "timeout escalation or external kill/OOM" in recorded
    assert "budget-exceeded" not in recorded and "not the budget" not in recorded


@pytest.mark.parametrize("phase", PHASES)
def test_an_externally_killed_worker_is_recorded_with_the_same_ambiguity(tmp_path, phase):
    root = tmp_path / "issue"
    (root / phase / "logs").mkdir(parents=True)
    done = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; record_run_result 137'],
                          env={**os.environ, "ISSUE_ROOT": str(root), "PHASE": phase},
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode == 137
    logs = root / phase / "logs"
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "137"
    assert "timeout escalation or external kill/OOM" in (logs / "run.done").read_text(
        encoding="utf-8")


def test_a_terminated_run_records_its_signal_instead_of_a_pass(tmp_path):
    started = tmp_path / "phase-started"
    env = _box_run_env(tmp_path, CHOICE_JSON, timeout_body=f'touch "{started}"; sleep 30')
    proc = subprocess.Popen(["bash", str(SCRIPT), "run", "confirm"], env=env,
                            start_new_session=True, stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert started.exists()
    os.killpg(proc.pid, signal.SIGTERM)
    assert proc.wait(timeout=30) == 143
    assert (_logs(tmp_path, "confirm") / "run.exit").read_text(
        encoding="utf-8").strip() == "143"


def test_a_resumed_run_keeps_the_failed_runs_events_and_a_fresh_run_starts_them_clean(
        tmp_path):
    events = _logs(tmp_path, "confirm") / "events"
    events.parent.mkdir(parents=True)
    events.write_text("2026-11-02T00:00:00Z confirm.3 worker 0 exit 0\n", encoding="utf-8")
    assert _run_on_the_box(tmp_path, "confirm", args=("--resume",)).returncode == 0
    assert events.read_text(encoding="utf-8") == "2026-11-02T00:00:00Z confirm.3 worker 0 exit 0\n"
    assert _run_on_the_box(tmp_path, "confirm").returncode == 0
    assert events.read_text(encoding="utf-8") == ""


def test_the_phases_never_share_a_done_or_exit_file(tmp_path):
    assert _run_on_the_box(tmp_path, "look", timeout_body="exit 7").returncode == 7
    assert _run_on_the_box(tmp_path, "confirm").returncode == 0
    assert (_logs(tmp_path, "look") / "run.exit").read_text(encoding="utf-8").strip() == "7"
    assert (_logs(tmp_path, "confirm") / "run.exit").read_text(encoding="utf-8").strip() == "0"
    assert not (_logs(tmp_path, "gates") / "run.exit").exists()


def test_an_unknown_run_argument_is_refused(tmp_path):
    done = _run_on_the_box(tmp_path, "look", args=("--force",))
    assert done.returncode != 0 and "unknown run argument" in done.stderr


# --- the decode phase: the tlog workers (look, confirm) ------------------------------------------------

_TLOG_STUB = '''import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["PY_LOG"]).open("a", encoding="utf-8") as log:
    log.write(" ".join(args) + "\\n")
module = next((arg for arg in args if "tlog_eval_decode" in arg), None)
if module is not None:
    sub = args[args.index(module) + 1]
    models = []
    if "--models" in args:
        for arg in args[args.index("--models") + 1:]:
            if arg.startswith("--"):
                break
            models.append(arg)
    if sub == "decode":
        role = args[args.index("--role") + 1]
        index = args[args.index("--shard-index") + 1]
        if os.environ.get("FAIL_WORKER") == role + " " + index:
            raise SystemExit(1)
        parts = Path(args[args.index("--parts-dir") + 1])
        for model in models:
            (parts / ("tlog.%s.shard%s.%s.jsonl" % (role, index, model))).write_text(
                "decoded\\n", encoding="utf-8")
    elif sub in ("merge", "merge-confirm"):
        out = Path(args[args.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / ("meta.json" if sub == "merge" else "meta.confirm.json")).write_text(
            "{}\\n", encoding="utf-8")
'''

WORKER_ROLES = {"look": ["look"], "confirm": ROLES}


def _decode_env(tmp_path: Path, phase: str, stub: str, **extra: str) -> dict[str, str]:
    """``decode`` without a GPU: ``python`` stands in for the decoders and records its args."""
    root = tmp_path / "issue"
    for name in ("logs", "parts", "out", "decodes", "clips"):
        (root / phase / name).mkdir(parents=True)
    bindir = tmp_path / "decode-bin"
    bindir.mkdir()
    _mock(bindir / "python", f"#!{sys.executable}\n" + stub)
    return {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "ISSUE_ROOT": str(root),
            "PHASE": phase, "MODELS": MODELS_OF[phase], "PARALLEL": "2",
            "PY_LOG": str(tmp_path / "python.log"), **extra}


def _decode(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source "{SCRIPT}"; decode'], env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)


def _python_log(tmp_path: Path) -> list[list[str]]:
    return [line.split() for line in
            (tmp_path / "python.log").read_text(encoding="utf-8").splitlines()]


def _workers(tmp_path: Path) -> list[list[str]]:
    return [args for args in _python_log(tmp_path)
            if any("tlog_eval_decode" in a for a in args) and "decode" in args]


def _models_of(args: list[str]) -> list[str]:
    models = []
    for arg in args[args.index("--models") + 1:]:
        if arg.startswith("--"):
            break
        models.append(arg)
    return models


@pytest.mark.parametrize("phase", ["look", "confirm"])
def test_each_tranche_is_decoded_by_disjoint_workers_with_the_round_3_spec(tmp_path, phase):
    done = _decode(_decode_env(tmp_path, phase, _TLOG_STUB))
    assert done.returncode == 0, done.stderr
    workers = _workers(tmp_path)
    roles = WORKER_ROLES[phase]
    assert sorted((a[a.index("--role") + 1], int(a[a.index("--shard-index") + 1]))
                  for a in workers) == sorted((role, index) for role in roles
                                              for index in range(2))
    for args in workers:
        assert args[args.index("--spec") + 1] == "consonant3"
        assert args[args.index("--shard-count") + 1] == "2"
        assert _models_of(args) == MODELS_OF[phase].split(",")
    events = (tmp_path / "issue" / phase / "logs" / "events").read_text(encoding="utf-8")
    assert {line.split(" ", 1)[1] for line in events.splitlines()} == {
        f"{role} worker {index} exit 0" for role in roles for index in range(2)}


def test_confirm_3_is_decoded_to_the_end_before_confirm_4_starts(tmp_path):
    assert _decode(_decode_env(tmp_path, "confirm", _TLOG_STUB)).returncode == 0
    order = [args[args.index("--role") + 1] for args in _workers(tmp_path)]
    assert order[:2] == ["confirm.3", "confirm.3"] and order[2:] == ["confirm.4", "confirm.4"]


def test_the_confirm_decode_is_pooled_once_for_round_3(tmp_path):
    assert _decode(_decode_env(tmp_path, "confirm", _TLOG_STUB)).returncode == 0
    merges = [args for args in _python_log(tmp_path) if "merge-confirm" in args]
    assert len(merges) == 1
    assert merges[0][merges[0].index("--spec") + 1] == "consonant3"
    assert _models_of(merges[0]) == CONFIRM_MODELS.split(",")
    assert (tmp_path / "issue" / "confirm" / "out" / "meta.confirm.json").exists()
    assert str(tmp_path / "issue" / "confirm" / "out") in " ".join(merges[0])


def test_the_look_decode_is_merged_once_for_round_3(tmp_path):
    assert _decode(_decode_env(tmp_path, "look", _TLOG_STUB)).returncode == 0
    merges = [args for args in _python_log(tmp_path)
              if any("tlog_eval_decode" in a for a in args) and "merge" in args]
    assert len(merges) == 1
    assert merges[0][merges[0].index("--spec") + 1] == "consonant3"
    assert _models_of(merges[0]) == LOOK_MODELS.split(",")
    assert (tmp_path / "issue" / "look" / "out" / "meta.json").exists()


@pytest.mark.parametrize("phase, failing", [("look", "look 1"), ("confirm", "confirm.4 1")])
def test_a_failed_worker_stops_the_run_before_the_merge(tmp_path, phase, failing):
    done = _decode(_decode_env(tmp_path, phase, _TLOG_STUB, FAIL_WORKER=failing))
    assert done.returncode != 0
    events = (tmp_path / "issue" / phase / "logs" / "events").read_text(encoding="utf-8")
    assert f"{failing.split()[0]} worker 1 exit 1" in events
    merges = [args for args in _python_log(tmp_path)
              if any("tlog_eval_decode" in a for a in args)
              and {"merge", "merge-confirm"} & set(args)]
    assert merges == []
    assert not list((tmp_path / "issue" / phase / "out").iterdir())


def test_a_failed_confirm_3_worker_never_starts_confirm_4(tmp_path):
    done = _decode(_decode_env(tmp_path, "confirm", _TLOG_STUB, FAIL_WORKER="confirm.3 0"))
    assert done.returncode != 0
    assert {args[args.index("--role") + 1] for args in _workers(tmp_path)} == {"confirm.3"}


# --- the decode phase: the guard sets (gates) -------------------------------------------------------------

#: Two shards per set: the queue the decode phase plans from ``--list-shards``.
SHARDS = 2
PAIRS = [(set_name, shard) for set_name in SETS for shard in range(SHARDS)]

_GATES_STUB = '''import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["PY_LOG"]).open("a", encoding="utf-8") as log:
    log.write(" ".join(args) + "\\n")
if any("aug_readout_decode" in arg for arg in args):
    if "--list-shards" in args:
        for i in range(int(os.environ["SHARDS"])):
            print("%d shard%d" % (i, i))
    else:
        set_name = args[args.index("--set") + 1]
        shard = args[args.index("--shard") + 1]
        if not set_name or not shard:  # argparse refuses an empty --set choice
            raise SystemExit(2)
        if os.environ.get("FAIL_SHARD") == set_name + " " + shard:
            raise SystemExit(1)
        out = Path(args[args.index("--out-dir") + 1])
        for model in args[args.index("--models") + 1].split(","):
            (out / ("%s.%s.%s.jsonl" % (set_name, shard, model))).write_text(
                "decoded\\n", encoding="utf-8")
elif any("consonant_baseline" in arg for arg in args):
    with Path(os.environ["CACHE_LOG"]).open("a", encoding="utf-8") as log:
        log.write(args[args.index("--model") + 1] + "\\n")
'''

_XARGS_STUB = f'''#!/bin/sh
# GNU xargs runs its command once when stdin holds nothing (no --no-run-if-empty); BSD xargs
# does not, so this stub keeps an empty-queue bug visible on a Mac. Non-empty input goes to the
# real xargs unchanged.
input=$(cat)
if [ -n "$input" ]; then
  printf '%s\\n' "$input" | exec "{shutil.which("xargs")}" "$@"
  exit $?
fi
while [ $# -gt 0 ]; do
  case "$1" in
    -P|-n|-I|-d|-s|-L) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
exec "$@"
'''


def _gates_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = _decode_env(tmp_path, "gates", _GATES_STUB, SHARDS=str(SHARDS),
                      CACHE_LOG=str(tmp_path / "cache.log"), PARALLEL="1", **extra)
    _mock(tmp_path / "decode-bin" / "xargs", _XARGS_STUB)
    return env


def _decoded_pairs(tmp_path: Path) -> list[tuple[str, int]]:
    return [(a[a.index("--set") + 1], int(a[a.index("--shard") + 1]))
            for a in _python_log(tmp_path)
            if "--list-shards" not in a and any("aug_readout_decode" in x for x in a)]


def _gate_events(tmp_path: Path) -> set[str]:
    events = (tmp_path / "issue" / "gates" / "logs" / "events").read_text(encoding="utf-8")
    return {line.split(" ", 1)[1] for line in events.splitlines()}


def test_a_first_gates_decode_queues_every_shard_of_every_set_for_the_candidate_alone(tmp_path):
    done = _decode(_gates_env(tmp_path))
    assert done.returncode == 0, done.stderr
    assert _decoded_pairs(tmp_path) == PAIRS
    for args in _python_log(tmp_path):
        if any("aug_readout_decode" in a for a in args) and "--shard" in args:
            assert args[args.index("--spec") + 1] == "consonant3"
            assert args[args.index("--models") + 1] == CANDIDATE
    assert (tmp_path / "cache.log").read_text(encoding="utf-8") == f"{CANDIDATE}\n"
    held_out = [a for a in _python_log(tmp_path) if any("consonant_baseline" in x for x in a)]
    assert len(held_out) == 1 and held_out[0][held_out[0].index("--spec") + 1] == "consonant3"
    assert _gate_events(tmp_path) == {f"{s} shard {i} exit 0" for s, i in PAIRS}


def test_a_resumed_gates_decode_retries_only_the_shards_without_an_exit_0_event(tmp_path):
    env = _gates_env(tmp_path)
    root = tmp_path / "issue" / "gates"
    lines = []
    for set_name, shard in PAIRS:
        if (set_name, shard) == ("qdat", 0):
            continue
        status = 137 if (set_name, shard) == ("tadabur_dev_unseen", 1) else 0
        lines.append(f"2026-11-02T00:00:00Z {set_name} shard {shard} exit {status}")
    (root / "logs" / "events").write_text("\n".join(lines) + "\n", encoding="utf-8")
    kept = root / "decodes" / f"qdat.1.{CANDIDATE}.jsonl"
    kept.write_text("first sitting\n", encoding="utf-8")
    done = _decode(env)
    assert done.returncode == 0, done.stderr
    assert _decoded_pairs(tmp_path) == [("qdat", 0), ("tadabur_dev_unseen", 1)]
    assert kept.read_text(encoding="utf-8") == "first sitting\n"
    assert _gate_events(tmp_path) == (
        {f"{s} shard {i} exit 0" for s, i in PAIRS} | {"tadabur_dev_unseen shard 1 exit 137"})
    assert (tmp_path / "cache.log").read_text(encoding="utf-8") == f"{CANDIDATE}\n"


def test_a_still_failing_gates_shard_does_not_leave_the_held_out_cache_unwritten(tmp_path):
    done = _decode(_gates_env(tmp_path, FAIL_SHARD="qdat 0"))
    assert done.returncode != 0
    assert "qdat shard 0 exit 1" in _gate_events(tmp_path)
    assert (tmp_path / "cache.log").read_text(encoding="utf-8") == f"{CANDIDATE}\n"


def test_a_held_out_only_resume_runs_no_shard_and_still_writes_the_cache(tmp_path):
    """Every shard already carries an exit-0 event: the retry queue is empty, and the empty queue
    must call the decoder zero times without failing the run."""
    env = _gates_env(tmp_path)
    root = tmp_path / "issue" / "gates"
    events = "".join(f"2026-11-02T00:00:00Z {s} shard {i} exit 0\n" for s, i in PAIRS)
    (root / "logs" / "events").write_text(events, encoding="utf-8")
    done = _decode(env)
    assert done.returncode == 0, done.stderr
    assert _decoded_pairs(tmp_path) == []
    assert (root / "logs" / "events").read_text(encoding="utf-8") == events
    assert (tmp_path / "cache.log").read_text(encoding="utf-8") == f"{CANDIDATE}\n"


# --- check and fetch ---------------------------------------------------------------------------------------


def _local_box(tmp_path: Path) -> dict[str, str]:
    """``ssh`` runs its command here, so the remote paths in these tests are local tmp dirs;
    ``rsync`` only logs the source it was given."""
    bindir = tmp_path / "local-bin"
    bindir.mkdir()
    _mock(bindir / "ssh", '#!/bin/sh\nshift\nexec sh -c "$*"\n')
    _mock(bindir / "rsync", f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{tmp_path / "rsync.log"}"\n')
    return {"PATH": f"{bindir}:{_env(tmp_path)['PATH']}", "HOST": "box",
            "ISSUE_ROOT": str(tmp_path / "issue"), "FETCH_DIR": str(tmp_path / "fetched")}


def _finished(tmp_path: Path, phase: str, *, exit_code: str = "0") -> Path:
    logs = tmp_path / "issue" / phase / "logs"
    logs.mkdir(parents=True)
    (logs / "run.exit").write_text(f"{exit_code}\n", encoding="utf-8")
    (logs / "run.done").write_text(f"2026-11-02T00:00:00Z exit {exit_code}\n", encoding="utf-8")
    return logs


def _completed_run(tmp_path: Path, phase: str) -> None:
    """The box-side output of a finished run of ``phase``: exit 0, every job's exit-0 event, and
    the phase's merged (look, confirm) or per-shard (gates) files and caches."""
    logs = _finished(tmp_path, phase)
    root = tmp_path / "issue" / phase
    if phase == "gates":
        (root / "decodes").mkdir()
        jobs, events = [], []
        for set_name in SETS:
            jobs.append(f"{set_name} 0")
            events.append(f"2026-11-02T00:00:00Z {set_name} shard 0 exit 0")
            (root / "decodes" / f"{set_name}.0.{CANDIDATE}.jsonl").write_text(
                "{}\n", encoding="utf-8")
        (logs / "jobs.txt").write_text("\n".join(jobs) + "\n", encoding="utf-8")
        (logs / "events").write_text("\n".join(events) + "\n", encoding="utf-8")
        for kind in ("decodes", "checks"):
            cache = (tmp_path / "issue" / "repo" / "tools" / "tadabur" / "truth_baseline" / kind
                     / f"{CANDIDATE}.json")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("{}\n", encoding="utf-8")
        return
    stem, meta = ("look", "meta.json") if phase == "look" else ("confirm", "meta.confirm.json")
    (logs / "events").write_text("".join(
        f"2026-11-02T00:00:00Z {role} worker {index} exit 0\n"
        for role in WORKER_ROLES[phase] for index in range(2)), encoding="utf-8")
    out = root / "out"
    out.mkdir()
    (out / meta).write_text("{}\n", encoding="utf-8")
    for model in MODELS_OF[phase].split(","):
        (out / f"tlog.{stem}.{model}.jsonl.gz").write_text("x\n", encoding="utf-8")


def _victim(tmp_path: Path, phase: str) -> Path:
    root = tmp_path / "issue" / phase
    if phase == "gates":
        return root / "decodes" / f"qdat.0.{CANDIDATE}.jsonl"
    return root / "out" / f"tlog.{phase}.{MODELS_OF[phase].split(',')[-1]}.jsonl.gz"


@pytest.mark.parametrize("phase", PHASES)
def test_check_reports_a_run_that_has_not_finished(tmp_path, phase):
    (tmp_path / "issue" / phase / "logs").mkdir(parents=True)
    done = _run(phase, "check", _repo(tmp_path), tmp_path, **_local_box(tmp_path))
    assert done.returncode == 3 and "pending" in done.stdout


@pytest.mark.parametrize("phase", PHASES)
def test_check_reports_a_done_marker_that_has_no_exit_marker(tmp_path, phase):
    logs = _finished(tmp_path, phase)
    (logs / "run.exit").unlink()
    done = _run(phase, "check", _repo(tmp_path), tmp_path, **_local_box(tmp_path))
    assert done.returncode == 1
    assert "failed" in done.stdout and "run.exit" in done.stdout


@pytest.mark.parametrize("phase", PHASES)
def test_check_reads_the_phase_s_own_logs(tmp_path, phase):
    _completed_run(tmp_path, phase)
    other = next(name for name in PHASES if name != phase)
    _finished(tmp_path, other, exit_code="7")
    done = _run(phase, "check", _repo(tmp_path), tmp_path, **_local_box(tmp_path))
    assert done.returncode == 0 and "done: exit 0" in done.stdout


@pytest.mark.parametrize("phase", PHASES)
def test_check_reports_a_budget_stop(tmp_path, phase):
    _finished(tmp_path, phase, exit_code="2")
    (tmp_path / "issue" / phase / "logs" / "events").write_text("", encoding="utf-8")
    done = _run(phase, "check", _repo(tmp_path), tmp_path, **_local_box(tmp_path))
    assert done.returncode == 2 and "budget exceeded" in done.stdout


@pytest.mark.parametrize("phase", PHASES)
def test_fetch_refuses_a_run_that_did_not_finish_every_decode(tmp_path, phase):
    _completed_run(tmp_path, phase)
    _victim(tmp_path, phase).unlink()
    done = _run(phase, "fetch", _repo(tmp_path), tmp_path, **_local_box(tmp_path))
    assert done.returncode != 0 and "complete" in done.stderr
    assert not (tmp_path / "rsync.log").exists()


@pytest.mark.parametrize("phase", PHASES)
def test_fetch_refuses_a_run_that_exited_non_zero(tmp_path, phase):
    _completed_run(tmp_path, phase)
    (tmp_path / "issue" / phase / "logs" / "run.exit").write_text("2\n", encoding="utf-8")
    done = _run(phase, "fetch", _repo(tmp_path), tmp_path, **_local_box(tmp_path))
    assert done.returncode != 0 and "complete" in done.stderr
    assert not (tmp_path / "rsync.log").exists()


@pytest.mark.parametrize("phase", PHASES)
def test_fetch_copies_a_complete_run(tmp_path, phase):
    _completed_run(tmp_path, phase)
    done = _run(phase, "fetch", _repo(tmp_path), tmp_path, **_local_box(tmp_path))
    assert done.returncode == 0, done.stderr
    copied = (tmp_path / "rsync.log").read_text(encoding="utf-8")
    assert f"{phase}/{'decodes' if phase == 'gates' else 'out'}" in copied


@pytest.mark.parametrize("phase", ["gates", "confirm"])
def test_fetch_refuses_without_the_phase_s_committed_inputs(tmp_path, phase):
    _completed_run(tmp_path, phase)
    done = _run(phase, "fetch", _repo(tmp_path, choice=None), tmp_path, **_local_box(tmp_path))
    assert done.returncode != 0 and "not committed" in done.stderr
    assert not (tmp_path / "rsync.log").exists()

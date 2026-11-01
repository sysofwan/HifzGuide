"""The round-2 confirm decode launcher (#230): ``scripts/issue-230-confirm-decode.sh``, run dry.

Round 2's P3 decode reads confirm.1 and confirm.2 once for the four frozen producers: the
teacher, C95 and the committed choice's candidate and control. The launcher is round 2's
``issue-230-decode.sh`` pattern applied to the tlog decoder (``tadabur.tlog_eval_decode``), which
owns the sealed-tranche gates and the pooled merge; it is a separate script because the non-tlog
launcher's own tests keep that one scoped to its four sets.

The sub-commands these tests drive need neither the box nor a GPU:

* ``plan``: prints, from the **committed** ``choice.json`` and the two committed claims, one line
  per decode (one per tranche) and one for the pooled merge::

      decode confirm.1 base,c95,<candidate>,<control> at <parallel> processes
      decode confirm.2 base,c95,<candidate>,<control> at <parallel> processes
      merge-confirm base,c95,<candidate>,<control>

  It refuses (non-zero exit, "not committed" or "already" on stderr, nothing on stdout) unless
  ``choice.json`` and both claims are committed and unchanged at HEAD, and once a confirm decode
  of a chosen model is already committed in the round's ``decodes/`` (the sitting reads once).
  ``ROUND_DIR`` (default ``tools/tadabur/consonant2_readout``) names the round's directory and
  ``TRANCHES_DIR`` (default ``tools/tadabur/tlog/tranches``) the claims' directory, both relative
  to the repository root; the script runs in the repository it is invoked in.
* ``launch``: the same refusals, then stages the repository and starts ``run`` detached on the
  box under ``flock /root/scratch/gpu.lock``, printing ``started: <remote pid>``.
* ``resume``: the same refusals and staging, starting ``run --resume`` at ``RESUME_PARALLEL``.
* ``check`` and ``fetch``: the box's ``run.done``/``run.exit`` and a complete-run gate before
  the pooled files are copied.
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

from training import round2_verdict_fixtures as rv

TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS.parent / "scripts" / "issue-230-confirm-decode.sh"
ROLES = ["confirm.1", "confirm.2"]
MODELS = "base,c95,r2_candidate_ema,r2_control_ema"
DECODE_LINES = [f"decode {role} {MODELS} at 4 processes" for role in ROLES] + [
    f"merge-confirm {MODELS}"]


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    shim = tmp_path / "shim"
    shim.mkdir(exist_ok=True)
    python = shim / "python"
    if not python.exists():
        python.symlink_to(sys.executable)
    return {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}", "ROUND_DIR": "readout",
            "TRANCHES_DIR": "tranches", **extra}


def _claims(round_: rv.Round) -> None:
    """Both tranches' opening claims, committed in HEAD."""
    paths = []
    for role in ROLES:
        path = round_.tranches / f"{role}.claim.json"
        path.write_text(json.dumps({"role": role, "issue": round_.spec.issue,
                                    "commit": rv.CLAIM_COMMIT, "date": "2026-10-31"}),
                        encoding="utf-8")
        paths.append(path)
    round_.commit(*paths, message="the confirm claims")


def _round(tmp_path: Path, flags: dict[str, int] | None = None) -> rv.Round:
    round_ = rv.committed_round(tmp_path, flags)
    _claims(round_)
    return round_


def _run(command: str, round_: rv.Round, tmp_path: Path, **extra: str):
    return subprocess.run(["bash", str(SCRIPT), command], cwd=round_.repo,
                          env=_env(tmp_path, **extra), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60)


def test_the_script_exists_parses_and_is_the_tlog_decoder():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    text = SCRIPT.read_text(encoding="utf-8")
    assert "tadabur.tlog_eval_decode" in text and "merge-confirm" in text
    assert "consonant1" not in text
    assert "tarteel_v1.confirm" not in text and "held-out" not in text


def test_the_plan_lists_both_tranches_and_the_pooled_merge(tmp_path):
    round_ = _round(tmp_path)
    done = _run("plan", round_, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == DECODE_LINES


def test_the_plan_follows_the_committed_choice_not_a_fixed_weight(tmp_path):
    round_ = _round(tmp_path, rv.MIXED_FLAGS)
    done = _run("plan", round_, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        line.replace("r2_candidate_ema", "r2_candidate_live") for line in DECODE_LINES]


@pytest.mark.parametrize("name", ["readout/choice.json",
                                  "tranches/confirm.1.claim.json",
                                  "tranches/confirm.2.claim.json"])
def test_the_plan_refuses_a_file_that_is_not_committed_at_head(tmp_path, name):
    round_ = _round(tmp_path)
    (round_.repo / name).write_text('{"edited": true}\n', encoding="utf-8")
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


def test_the_plan_refuses_once_a_confirm_decode_of_a_chosen_model_is_committed(tmp_path):
    round_ = _round(tmp_path)
    round_.write_commit(round_.layout.root / "decodes" / "tlog.confirm.r2_candidate_ema.jsonl.gz",
                        "x", "a confirm decode")
    done = _run("plan", round_, tmp_path)
    assert done.returncode != 0 and done.stdout == ""
    assert "already" in done.stderr


def test_the_plan_ignores_the_look_decodes_of_the_same_models(tmp_path):
    """The round's look decodes name the same models; only the pooled confirm files count."""
    round_ = _round(tmp_path)
    for model in ("base", "c95", "r2_candidate_ema"):
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
    _mock(root / "repo" / "scripts" / "issue-230-confirm-decode.sh",
          f"#!/bin/sh\nprintf 'ran\\n' > '{marker}'\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _mock(bindir / "ssh", '#!/bin/sh\nprintf "%s\\n" "$2" >> "$SSH_LOG"\n'
          'case "$2" in\n  *nohup*) exec sh -c "$2" ;;\n  *) cat >/dev/null ;;\nesac\n')
    _mock(bindir / "flock", '#!/bin/sh\nshift\nexec "$@"\n')
    return {"PATH": f"{bindir}:{_env(tmp_path)['PATH']}", "HOST": "box",
            "ISSUE_ROOT": str(root), "SSH_LOG": str(tmp_path / "ssh.log")}


def test_the_launch_refuses_before_staging_anything_without_the_claims(tmp_path):
    round_ = rv.committed_round(tmp_path)
    done = _run("launch", round_, tmp_path, **_box(tmp_path))
    assert done.returncode != 0 and "not committed" in done.stderr
    assert not (tmp_path / "ssh.log").exists()


def test_the_launch_starts_the_detached_run_on_the_box_under_the_gpu_lock(tmp_path):
    round_ = _round(tmp_path)
    done = _run("launch", round_, tmp_path, **_box(tmp_path))
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"started: \d+", done.stdout.strip()), done.stdout
    log = (tmp_path / "ssh.log").read_text(encoding="utf-8")
    assert "nohup flock /root/scratch/gpu.lock" in log
    assert "PARALLEL=4" in log
    deadline = time.monotonic() + 10
    while not (tmp_path / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (tmp_path / "started").exists()


def test_the_resume_restages_and_starts_a_resumed_run_at_the_low_parallelism(tmp_path):
    round_ = _round(tmp_path)
    done = _run("resume", round_, tmp_path, **_box(tmp_path))
    assert done.returncode == 0, done.stderr
    log = (tmp_path / "ssh.log").read_text(encoding="utf-8")
    assert "nohup flock /root/scratch/gpu.lock" in log
    assert "run --resume" in log
    assert "PARALLEL=2" in log


# --- the box's run --------------------------------------------------------------------------------


CHOICE_JSON = '{"models": {"candidate": "r2_candidate_ema", "control": "r2_control_ema"}}'


def _box_run_env(tmp_path: Path, choice: str, *, timeout_body: str = "exit 0",
                 conda: bool = True) -> dict[str, str]:
    """The box's shell shape for ``run``: no ``python`` on PATH until the conda env is
    activated, and a ``timeout`` stub in place of the decode phase (nothing decodes here)."""
    root = tmp_path / "issue"
    (root / "repo" / "readout").mkdir(parents=True)
    (root / "repo" / "readout" / "choice.json").write_text(choice, encoding="utf-8")
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    _mock(prefix / "bin" / "python",
          f'#!/bin/sh\ntouch "{tmp_path / "activated"}"\nexec "{sys.executable}" "$@"\n')
    conda_script = tmp_path / "conda.sh"
    if conda:
        conda_script.write_text(f'conda() {{ PATH="{prefix}/bin:$PATH"; export PATH; }}\n',
                                encoding="utf-8")
    bindir = tmp_path / "run-bin"
    bindir.mkdir()
    _mock(bindir / "timeout",
          f'#!/bin/sh\nprintf "%s" "$MODELS" > "{tmp_path / "models"}"\n{timeout_body}\n')
    return {**os.environ, "PATH": f"{bindir}:/usr/bin:/bin", "CONDA": str(conda_script),
            "ISSUE_ROOT": str(root), "ROUND_DIR": "readout"}


def _run_on_the_box(tmp_path: Path, choice: str, *, args: tuple[str, ...] = (),
                    **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), "run", *args],
                          env=_box_run_env(tmp_path, choice, **kwargs),
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)


def test_the_run_reads_the_choice_with_the_activated_env_python(tmp_path):
    done = _run_on_the_box(tmp_path, CHOICE_JSON)
    logs = tmp_path / "issue" / "logs"
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "activated").exists()
    assert (tmp_path / "models").read_text(encoding="utf-8") == MODELS
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "0"
    assert "exit 0" in (logs / "run.done").read_text(encoding="utf-8")


def test_a_run_that_dies_before_the_decode_phase_still_records_the_failure(tmp_path):
    done = _run_on_the_box(tmp_path, "{edited")
    logs = tmp_path / "issue" / "logs"
    assert done.returncode != 0
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == str(done.returncode)
    assert (logs / "run.done").exists()


def test_a_missing_conda_script_is_a_recorded_failure_not_a_pass(tmp_path):
    done = _run_on_the_box(tmp_path, CHOICE_JSON, conda=False)
    logs = tmp_path / "issue" / "logs"
    assert done.returncode != 0
    assert "no conda" in done.stderr
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == str(done.returncode)
    assert (logs / "run.done").exists()


def test_a_failed_decode_phase_is_recorded_for_check(tmp_path):
    done = _run_on_the_box(tmp_path, CHOICE_JSON, timeout_body="exit 7")
    logs = tmp_path / "issue" / "logs"
    assert done.returncode == 7
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "7"
    assert "exit 7" in (logs / "run.done").read_text(encoding="utf-8")


def test_a_budget_expiry_is_recorded_as_the_budget(tmp_path):
    """GNU timeout's own status is 124: that, and only that, is the budget stop."""
    done = _run_on_the_box(tmp_path, CHOICE_JSON, timeout_body="exit 124")
    logs = tmp_path / "issue" / "logs"
    assert done.returncode == 2
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "2"
    assert "budget-exceeded" in (logs / "run.done").read_text(encoding="utf-8")


def test_a_timeout_escalation_is_recorded_as_an_ambiguous_sigkill(tmp_path):
    """``timeout --kill-after`` escalates a budget expiry to SIGKILL, whose status is 137;
    the record must not call that "not the budget" — it cannot tell escalation from an
    external kill."""
    done = _run_on_the_box(tmp_path, CHOICE_JSON, timeout_body="exit 137")
    logs = tmp_path / "issue" / "logs"
    assert done.returncode == 137
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "137"
    recorded = (logs / "run.done").read_text(encoding="utf-8")
    assert "timeout escalation or external kill/OOM" in recorded
    assert "budget-exceeded" not in recorded and "not the budget" not in recorded


def test_an_externally_killed_worker_is_recorded_with_the_same_ambiguity(tmp_path):
    """A worker killed from outside reaches ``record_run_result 137`` directly, with no
    timeout on the stack: the same honest message, never a guess at the cause."""
    root = tmp_path / "issue"
    (root / "logs").mkdir(parents=True)
    done = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; record_run_result 137'],
                          env={**os.environ, "ISSUE_ROOT": str(root)},
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode == 137
    assert (root / "logs" / "run.exit").read_text(encoding="utf-8").strip() == "137"
    recorded = (root / "logs" / "run.done").read_text(encoding="utf-8")
    assert "timeout escalation or external kill/OOM" in recorded


def test_a_terminated_run_records_its_signal_instead_of_a_pass(tmp_path):
    started = tmp_path / "phase-started"
    env = _box_run_env(tmp_path, CHOICE_JSON,
                       timeout_body=f'touch "{started}"; sleep 30')
    proc = subprocess.Popen(["bash", str(SCRIPT), "run"], env=env, start_new_session=True,
                            stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert started.exists()
    os.killpg(proc.pid, signal.SIGTERM)
    assert proc.wait(timeout=30) == 143
    assert (tmp_path / "issue" / "logs" / "run.exit").read_text(encoding="utf-8").strip() == "143"


# --- the decode phase: both tranches, then the pooled merge ---------------------------------------

_STUB_BODY = '''import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["PY_LOG"]).open("a", encoding="utf-8") as log:
    log.write(" ".join(args) + "\\n")
if "decode" in args and any("tlog_eval_decode" in arg for arg in args):
    role = args[args.index("--role") + 1]
    index = args[args.index("--shard-index") + 1]
    if os.environ.get("FAIL_WORKER") == role + " " + index:
        raise SystemExit(1)
    out = Path(args[args.index("--parts-dir") + 1])
    for model in args[args.index("--models") + 1].split(","):
        (out / ("tlog.%s.shard%s.%s.jsonl" % (role, index, model))).write_text(
            "decoded\\n", encoding="utf-8")
elif "merge-confirm" in args and any("tlog_eval_decode" in arg for arg in args):
    out = Path(args[args.index("--out-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.confirm.json").write_text("{}\\n", encoding="utf-8")
'''


def _decode_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """``decode`` without a GPU: ``python`` stands in for the tlog decoder and records its args."""
    root = tmp_path / "issue"
    for name in ("logs", "parts", "out"):
        (root / name).mkdir(parents=True)
    bindir = tmp_path / "decode-bin"
    bindir.mkdir()
    _mock(bindir / "python", f"#!{sys.executable}\n" + _STUB_BODY)
    return {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "ISSUE_ROOT": str(root),
            "MODELS": MODELS, "PARALLEL": "2", "PY_LOG": str(tmp_path / "python.log"),
            **extra}


def _decode(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source "{SCRIPT}"; decode'], env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)


def _calls(tmp_path: Path) -> list[tuple[str, int]]:
    pairs = []
    for line in (tmp_path / "python.log").read_text(encoding="utf-8").splitlines():
        args = line.split()
        if "decode" in args and any("tlog_eval_decode" in a for a in args):
            pairs.append((args[args.index("--role") + 1],
                          int(args[args.index("--shard-index") + 1])))
    return pairs


def test_both_tranches_are_decoded_by_disjoint_workers_then_pooled(tmp_path):
    env = _decode_env(tmp_path)
    done = _decode(env)
    assert done.returncode == 0, done.stderr
    assert sorted(_calls(tmp_path)) == sorted(
        (role, index) for role in ROLES for index in range(2))
    log = (tmp_path / "python.log").read_text(encoding="utf-8").splitlines()
    merge = [line for line in log if "merge-confirm" in line]
    assert len(merge) == 1 and "--models base c95 r2_candidate_ema r2_control_ema" in merge[0]
    events = (tmp_path / "issue" / "logs" / "events").read_text(encoding="utf-8")
    assert {line.split(" ", 1)[1] for line in events.splitlines()} == {
        f"{role} worker {index} exit 0" for role in ROLES for index in range(2)}
    assert (tmp_path / "issue" / "out" / "meta.confirm.json").exists()


def test_a_failed_worker_stops_the_run_before_the_merge(tmp_path):
    env = _decode_env(tmp_path, FAIL_WORKER="confirm.2 1")
    done = _decode(env)
    assert done.returncode != 0
    events = (tmp_path / "issue" / "logs" / "events").read_text(encoding="utf-8")
    assert "confirm.2 worker 1 exit 1" in events
    assert "merge-confirm" not in (tmp_path / "python.log").read_text(encoding="utf-8")
    assert not (tmp_path / "issue" / "out" / "meta.confirm.json").exists()


# --- check and fetch ------------------------------------------------------------------------------


def _local_box(tmp_path: Path) -> dict[str, str]:
    """``ssh`` runs its command here, so the remote paths in these tests are local tmp dirs;
    ``rsync`` only logs the source it was given."""
    bindir = tmp_path / "local-bin"
    bindir.mkdir()
    _mock(bindir / "ssh", '#!/bin/sh\nshift\nexec sh -c "$*"\n')
    _mock(bindir / "rsync", f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{tmp_path / "rsync.log"}"\n')
    return {"PATH": f"{bindir}:{_env(tmp_path)['PATH']}", "HOST": "box",
            "ISSUE_ROOT": str(tmp_path / "issue")}


def _completed_run(tmp_path: Path) -> None:
    """The box-side output of a finished run: exit 0, both tranches' worker lines, and the
    pooled merge's four files plus its meta."""
    root = tmp_path / "issue"
    logs = root / "logs"
    logs.mkdir(parents=True)
    (logs / "run.exit").write_text("0\n", encoding="utf-8")
    (logs / "run.done").write_text("2026-01-01T00:00:00Z exit 0\n", encoding="utf-8")
    events = [f"2026-01-01T00:00:00Z {role} worker {index} exit 0"
              for role in ROLES for index in range(2)]
    (logs / "events").write_text("\n".join(events) + "\n", encoding="utf-8")
    out = root / "out"
    out.mkdir()
    (out / "meta.confirm.json").write_text("{}\n", encoding="utf-8")
    for model in MODELS.split(","):
        (out / f"tlog.confirm.{model}.jsonl.gz").write_text("x\n", encoding="utf-8")


def test_fetch_refuses_a_run_that_did_not_finish_every_decode(tmp_path):
    round_ = _round(tmp_path)
    _completed_run(tmp_path)
    (tmp_path / "issue" / "out" / "tlog.confirm.r2_control_ema.jsonl.gz").unlink()
    done = _run("fetch", round_, tmp_path, **_local_box(tmp_path),
                FETCH_DIR=str(tmp_path / "fetched"))
    assert done.returncode != 0
    assert "complete" in done.stderr
    assert not (tmp_path / "rsync.log").exists()


def test_fetch_copies_a_complete_run(tmp_path):
    round_ = _round(tmp_path)
    _completed_run(tmp_path)
    done = _run("fetch", round_, tmp_path, **_local_box(tmp_path),
                FETCH_DIR=str(tmp_path / "fetched"))
    assert done.returncode == 0, done.stderr
    assert "out" in (tmp_path / "rsync.log").read_text(encoding="utf-8")


def test_check_reports_a_done_marker_that_has_no_exit_marker(tmp_path):
    logs = tmp_path / "issue" / "logs"
    logs.mkdir(parents=True)
    (logs / "run.done").write_text("2026-01-01T00:00:00Z exit 0\n", encoding="utf-8")
    done = subprocess.run(["bash", str(SCRIPT), "check"],
                          env=_env(tmp_path, **_local_box(tmp_path)),
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode == 1
    assert "failed" in done.stdout and "run.exit" in done.stdout

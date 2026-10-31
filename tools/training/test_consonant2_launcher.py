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
* ``resume``: the same refusals and staging, starting ``run --resume`` at ``RESUME_PARALLEL``
  processes (default 2): a run the box stopped early retries only the shards with no exit-0
  event, keeps the outputs the failed run wrote, and then writes the G6 caches.

Round 1's ``--spec consonant1``, its confirm sets and its candidate-only models appear nowhere.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from training import round2_verdict_fixtures as rv

TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS.parent / "scripts" / "issue-230-decode.sh"
#: The sets the launcher decodes for both arms, and the two chosen models.
SETS = ["tarteel_v1.confirm", "qdat", "qud_other", "tadabur_dev_unseen"]
MODELS = ["r2_candidate_ema", "r2_control_ema"]
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


def test_the_resume_restages_and_starts_a_resumed_run_at_the_low_parallelism(tmp_path):
    round_ = rv.committed_round(tmp_path)
    box = _box(tmp_path)
    done = _run("resume", round_, tmp_path, **box)
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"started: \d+", done.stdout.strip()), done.stdout
    log = (tmp_path / "ssh.log").read_text(encoding="utf-8")
    assert "nohup flock /root/scratch/gpu.lock" in log
    assert "run --resume" in log
    assert "PARALLEL=2" in log
    deadline = time.monotonic() + 10
    while not (tmp_path / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (tmp_path / "started").exists()


def test_the_resume_parallelism_follows_resume_parallel(tmp_path):
    round_ = rv.committed_round(tmp_path)
    done = _run("resume", round_, tmp_path, RESUME_PARALLEL="1", **_box(tmp_path))
    assert done.returncode == 0, done.stderr
    log = (tmp_path / "ssh.log").read_text(encoding="utf-8")
    assert "PARALLEL=1" in log and "PARALLEL=2" not in log


# --- the box's run --------------------------------------------------------------------------------


CHOICE_JSON = '{"models": {"candidate": "r2_candidate_ema", "control": "r2_control_ema"}}'


def _box_run_env(tmp_path: Path, choice: str, *, timeout_body: str = "exit 0",
                 conda: bool = True) -> dict[str, str]:
    """The box's shell shape for ``run``: no ``python`` on PATH until the conda env is
    activated, and a ``timeout`` stub in place of the decode phase (nothing decodes here).

    The env's ``python`` touches ``activated`` and the stub records ``$MODELS``, so a test can
    see which interpreter read the choice and what it read."""
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
    assert (tmp_path / "models").read_text(encoding="utf-8") == "r2_candidate_ema,r2_control_ema"
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "0"
    assert "exit 0" in (logs / "run.done").read_text(encoding="utf-8")


def test_a_run_that_dies_before_the_decode_phase_still_records_the_failure(tmp_path):
    done = _run_on_the_box(tmp_path, "{edited")
    logs = tmp_path / "issue" / "logs"
    assert done.returncode != 0
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == str(done.returncode)
    assert (logs / "run.done").exists()


def test_a_run_that_cannot_make_its_directories_still_records_the_failure(tmp_path):
    (tmp_path / "issue" / "logs").mkdir(parents=True)
    (tmp_path / "issue" / "decodes").write_text("a stale file\n", encoding="utf-8")
    done = _run_on_the_box(tmp_path, CHOICE_JSON)
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


FAILED_EVENT = "2026-10-31T08:22:30Z tarteel_v1.confirm shard 5 exit 137\n"


def test_a_resumed_run_keeps_the_failed_runs_events(tmp_path):
    logs = tmp_path / "issue" / "logs"
    logs.mkdir(parents=True)
    (logs / "events").write_text(FAILED_EVENT, encoding="utf-8")
    done = _run_on_the_box(tmp_path, CHOICE_JSON, args=("--resume",))
    assert done.returncode == 0, done.stderr
    assert (logs / "events").read_text(encoding="utf-8") == FAILED_EVENT
    assert (logs / "run.exit").read_text(encoding="utf-8").strip() == "0"
    assert "exit 0" in (logs / "run.done").read_text(encoding="utf-8")


def test_a_fresh_run_starts_the_events_clean(tmp_path):
    """A stale exit-0 line would let ``fetch`` take a shard this sitting never decoded."""
    logs = tmp_path / "issue" / "logs"
    logs.mkdir(parents=True)
    (logs / "events").write_text(FAILED_EVENT, encoding="utf-8")
    done = _run_on_the_box(tmp_path, CHOICE_JSON)
    assert done.returncode == 0, done.stderr
    assert (logs / "events").read_text(encoding="utf-8") == ""


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


def test_check_reports_a_done_marker_that_has_no_exit_marker(tmp_path):
    logs = tmp_path / "issue" / "logs"
    logs.mkdir(parents=True)
    (logs / "run.done").write_text("2026-01-01T00:00:00Z exit 0\n", encoding="utf-8")
    done = subprocess.run(["bash", str(SCRIPT), "check"], env=_env(tmp_path, **_local_box(tmp_path)),
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode == 1
    assert "failed" in done.stdout and "run.exit" in done.stdout


def _completed_run(tmp_path: Path) -> None:
    """The box-side output of a finished run: every queued job exited 0, both models' shard
    files exist, and both arms' G6 caches exist."""
    root = tmp_path / "issue"
    logs = root / "logs"
    logs.mkdir(parents=True)
    (logs / "run.exit").write_text("0\n", encoding="utf-8")
    (logs / "run.done").write_text("2026-01-01T00:00:00Z exit 0\n", encoding="utf-8")
    jobs, events = [], []
    (root / "decodes").mkdir()
    for set_name in SETS:
        jobs.append(f"{set_name} 0")
        events.append(f"2026-01-01T00:00:00Z {set_name} shard 0 exit 0")
        for model in MODELS:
            (root / "decodes" / f"{set_name}.0.{model}.jsonl").write_text("{}\n", encoding="utf-8")
    (logs / "jobs.txt").write_text("\n".join(jobs) + "\n", encoding="utf-8")
    (logs / "events").write_text("\n".join(events) + "\n", encoding="utf-8")
    for model in MODELS:
        for kind in ("decodes", "checks"):
            cache = root / "repo" / "tools" / "tadabur" / "truth_baseline" / kind / f"{model}.json"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("{}\n", encoding="utf-8")


def test_fetch_refuses_a_run_that_did_not_finish_every_decode(tmp_path):
    round_ = rv.committed_round(tmp_path)
    _completed_run(tmp_path)
    (tmp_path / "issue" / "decodes" / f"qdat.0.{MODELS[0]}.jsonl").unlink()
    done = _run("fetch", round_, tmp_path, **_local_box(tmp_path),
                FETCH_DIR=str(tmp_path / "fetched"))
    assert done.returncode != 0
    assert "complete" in done.stderr
    assert not (tmp_path / "rsync.log").exists()


def test_fetch_copies_a_complete_run(tmp_path):
    round_ = rv.committed_round(tmp_path)
    _completed_run(tmp_path)
    done = _run("fetch", round_, tmp_path, **_local_box(tmp_path),
                FETCH_DIR=str(tmp_path / "fetched"))
    assert done.returncode == 0, done.stderr
    assert "decodes" in (tmp_path / "rsync.log").read_text(encoding="utf-8")


# --- the box's decode: retrying only what the failed run left undone -------------------------------

#: Two shards per set: the queue the decode phase plans from ``--list-shards``.
SHARDS = 2
PAIRS = [(set_name, shard) for set_name in SETS for shard in range(SHARDS)]

_STUB_BODY = '''import os
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


def _decode_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """``decode`` without a GPU: ``python`` stands in for both tools, listing shards, writing
    each shard's outputs, or logging the G6 cache model it was asked for."""
    root = tmp_path / "issue"
    for name in ("logs", "decodes", "clips"):
        (root / name).mkdir(parents=True)
    bindir = tmp_path / "decode-bin"
    bindir.mkdir()
    _mock(bindir / "python", f"#!{sys.executable}\n" + _STUB_BODY)
    return {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "ISSUE_ROOT": str(root),
            "MODELS": ",".join(MODELS), "PARALLEL": "1", "SHARDS": str(SHARDS),
            "PY_LOG": str(tmp_path / "python.log"), "CACHE_LOG": str(tmp_path / "cache.log"),
            **extra}


def _decode(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source "{SCRIPT}"; decode'], env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)


def _decoded_pairs(tmp_path: Path) -> list[tuple[str, int]]:
    pairs = []
    for line in (tmp_path / "python.log").read_text(encoding="utf-8").splitlines():
        args = line.split()
        if "--list-shards" in args or not any("aug_readout_decode" in a for a in args):
            continue
        pairs.append((args[args.index("--set") + 1], int(args[args.index("--shard") + 1])))
    return pairs


def _event_suffixes(tmp_path: Path) -> set[str]:
    events = (tmp_path / "issue" / "logs" / "events").read_text(encoding="utf-8")
    return {line.split(" ", 1)[1] for line in events.splitlines()}


def test_a_resumed_decode_retries_only_the_shards_without_an_exit_0_event(tmp_path):
    env = _decode_env(tmp_path)
    root = tmp_path / "issue"
    lines = []
    for set_name, shard in PAIRS:
        if (set_name, shard) == ("qdat", 0):
            continue
        status = 137 if (set_name, shard) == ("tadabur_dev_unseen", 1) else 0
        lines.append(f"2026-10-31T00:00:00Z {set_name} shard {shard} exit {status}")
    (root / "logs" / "events").write_text("\n".join(lines) + "\n", encoding="utf-8")
    kept = root / "decodes" / f"qdat.1.{MODELS[0]}.jsonl"
    kept.write_text("first sitting\n", encoding="utf-8")
    done = _decode(env)
    assert done.returncode == 0, done.stderr
    assert _decoded_pairs(tmp_path) == [("qdat", 0), ("tadabur_dev_unseen", 1)]
    # The failed run's outputs and its history stay; the retried shards record their own events.
    assert kept.read_text(encoding="utf-8") == "first sitting\n"
    assert (root / "decodes" / f"qdat.0.{MODELS[0]}.jsonl").read_text(
        encoding="utf-8") == "decoded\n"
    expected = {f"{s} shard {i} exit 0" for s, i in PAIRS}
    expected |= {"tadabur_dev_unseen shard 1 exit 137"}
    assert _event_suffixes(tmp_path) == expected
    # G6's caches run after the retries, for both arms.
    assert (tmp_path / "cache.log").read_text(encoding="utf-8") == "\n".join(MODELS) + "\n"
    log = (tmp_path / "python.log").read_text(encoding="utf-8").splitlines()
    assert all("consonant_baseline" in line for line in log[-2:])


def test_a_first_decode_queues_every_shard(tmp_path):
    env = _decode_env(tmp_path)
    done = _decode(env)
    assert done.returncode == 0, done.stderr
    assert _decoded_pairs(tmp_path) == PAIRS
    assert (tmp_path / "cache.log").read_text(encoding="utf-8") == "\n".join(MODELS) + "\n"


def test_a_still_failing_shard_does_not_leave_the_g6_caches_unwritten(tmp_path):
    env = _decode_env(tmp_path, FAIL_SHARD="qdat 0")
    done = _decode(env)
    assert done.returncode != 0
    assert {"qdat shard 0 exit 1"} <= _event_suffixes(tmp_path)
    assert (tmp_path / "cache.log").read_text(encoding="utf-8") == "\n".join(MODELS) + "\n"

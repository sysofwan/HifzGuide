"""One ui per scratch root, reserved atomically (#311 PR3, reviewer r8 R8.1).

The answers live in ``$ROOT/verdicts.jsonl`` and a second server would answer from its own
snapshot of it, while staging replaces the clone the first serves from — so the reservation has
to exclude a second launch *before* anything destructive, including one racing it rather than
following it. The reservation is ``$ROOT/ui.lock``, created with ``mkdir`` on the box: the
single atomic step. This file pins the four states a launch can find it in (free, a running
runner, a launcher mid-staging, a dead runner), the runner's claim-and-release, and the race
itself — six launches of one root at once, exactly one of which may reserve it.

``_launch`` runs the real script against an ``ssh`` shim; the race runs the real
``reserve_root_ui`` against a local stand-in for the box, where ``mkdir`` arbitrates exactly as
it does on cuda-dev.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "issue-311-audit.sh"
LIBRARY = SCRIPT.parent / "lib" / "audit-launcher.sh"
HOST = "root@box"
LISTEN = 'LISTEN 0 5 0.0.0.0:8000 0.0.0.0:* users:(("python",pid=4242,fd=3))'
#: What the shim answers: free (default), a running runner, a launcher still staging, a dead one.
LOCKED = ("live", "starting", "dead")


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    """A one-commit repository for the launcher to bundle."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    (repo / "README").write_text("staged\n", encoding="utf-8")
    subprocess.run([*git, "add", "README"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "stage"], check=True)
    return repo


def _launch(tmp_path: Path, repo: Path, *args: str, bound: str = "", lock: str = "",
            start_fails: bool = False, **env: str) -> subprocess.CompletedProcess:
    """Run the script with ``ssh`` shimmed (every remote command in ``ssh.log``).

    ``lock`` models `$ROOT/ui.lock` as the box holds it: ``live`` (a runner of this root is
    there), ``starting`` (a launcher reserved it and has not started a runner), ``dead`` (the
    runner it recorded is gone); empty is a free root. The stale swap and the retried ``mkdir``
    succeed iff the lock was ``dead``, so a takeover is observable in the log."""
    marker = tmp_path / "launched"
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    (shim / "ssh").write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> '{tmp_path}/ssh.log'
cat > /dev/null
cmd="$*"
if [ {int(start_fails)} = 1 ]; then
  case "$cmd" in *'setsid'*) exit 1 ;; esac
fi
case "$cmd" in *'run-ui.sh'*) : > '{marker}'; exit 0 ;; esac
case "$cmd" in *'test -f '*'ui.exit'*) exit 1 ;; esac
case "$cmd" in
  *'mv '*) if [ "{lock}" = dead ]; then : > '{tmp_path}/swapped'; exit 0; fi; exit 1 ;;
  *'rm -rf '*'ui.lock.stale'*) exit 0 ;;
  *'ui.lock/token'*)
    case "$cmd" in
      *mkdir*)
        if [ -z "{lock}" ]; then exit 0; fi
        if [ "{lock}" = dead ] && [ -f '{tmp_path}/swapped' ]; then exit 0; fi
        exit 1 ;;
      *) exit 0 ;;
    esac ;;
  *'ui.lock/pid'*) case "{lock}" in live) printf 4242 ;; dead) printf 999 ;; esac; exit 0 ;;
  *'ps -p'*) case "{lock}" in live) printf '%s\\n' "/usr/bin/bash {tmp_path}/scratch/run-ui.sh" ;; esac; exit 0 ;;
esac
if [ -f '{marker}' ]; then
  case "$cmd" in *'ss -ltnp'*) printf '%s\\n' '{LISTEN}' ; exit 0 ;; esac
elif [ -n "{bound}" ]; then
  case "$cmd" in *":{bound} "*) printf '%s\\n' '{LISTEN}'; exit 0 ;; esac
fi
case "$cmd" in *'pgrep -P'*) echo 4242 ;; esac
exit 0
""", encoding="utf-8")
    (shim / "ssh").chmod(0o755)
    base = {key: value for key, value in os.environ.items()
            if key not in {"ISSUE_ROOT", "PORT", "WITH_MAIN", "HOST", "REPO_DIR"}}
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env={**base, "HOST": HOST, "REPO_DIR": str(repo), "ISSUE_ROOT": str(tmp_path / "scratch"),
             "PATH": f"{shim}:{os.environ['PATH']}", **env},
        capture_output=True, text=True, stdin=subprocess.DEVNULL)


def _remote(tmp_path: Path) -> str:
    log = tmp_path / "ssh.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _text(done: subprocess.CompletedProcess) -> str:
    return done.stdout + done.stderr


def _nothing_done(done: subprocess.CompletedProcess, tmp_path: Path) -> None:
    assert "job started" not in done.stdout
    remote = _remote(tmp_path)
    assert "git clone" not in remote and "setsid" not in remote


def test_a_root_that_already_serves_a_ui_refuses_a_second_launch_before_staging(tmp_path, stage):
    """The pilot is running; `WITH_MAIN=1 ui launch` must not stage over it.

    The refusal is the root's, not the port's: it fires although 8002 is free and the fallback
    would take it."""
    root = tmp_path / "scratch"
    done = _launch(tmp_path, stage, "ui", "launch", bound="8000", lock="live")
    assert done.returncode == 1, _text(done)
    assert f"{root} on root@box already serves a ui (pid 4242)" in done.stderr
    assert "kill -- -4242" in done.stderr
    _nothing_done(done, tmp_path)
    assert ":8002 " not in _remote(tmp_path) and "trying" not in done.stderr


def test_a_reservation_mid_launch_is_not_stolen(tmp_path, stage):
    """The holder has reserved the root and is staging: no pid yet, and not stale."""
    done = _launch(tmp_path, stage, "ui", "launch", lock="starting")
    assert done.returncode == 1, _text(done)
    assert "already being served or launched" in done.stderr
    _nothing_done(done, tmp_path)


def test_a_reservation_whose_runner_is_gone_is_taken_over(tmp_path, stage):
    """A runner that died holds nothing: the next launch takes the reservation and serves."""
    done = _launch(tmp_path, stage, "ui", "launch", bound="8000", lock="dead")
    assert done.returncode == 0, _text(done)
    assert "  url:  http://box:8002/" in done.stdout       # the foreign 8000 still falls back
    remote = _remote(tmp_path)
    assert "mv '{}".format(tmp_path / "scratch" / "ui.lock") in remote


def test_the_runner_claims_the_reservation_and_releases_it_on_the_way_out(tmp_path, stage):
    done = _launch(tmp_path, stage, "ui", "launch")
    assert done.returncode == 0, _text(done)
    runner = next(line for line in _remote(tmp_path).splitlines() if "claim_reservation" in line)
    assert 'trap release_reservation EXIT' in _remote(tmp_path)
    assert f'{tmp_path}/scratch/ui.lock/token' in runner and 'echo $$ >' in runner


def test_a_failed_start_hands_the_reservation_back(tmp_path, stage):
    """A start that never happened must not lock the root out until the 15-minute takeover."""
    done = _launch(tmp_path, stage, "ui", "launch", start_fails=True)
    assert done.returncode == 1, _text(done)
    dropped = [line for line in _remote(tmp_path).splitlines()
               if "ui.lock/token" in line and "rm -rf" in line]
    assert dropped, _remote(tmp_path)


RACE = """#!/usr/bin/env bash
# The real reservation against a local stand-in for the box: `remote` runs the command here, so
# `mkdir` arbitrates exactly as it does on cuda-dev. All processes wait for one start file.
set -uo pipefail
ROOT=$1; HOST=box; job=ui; NAME=ui; LOGS=$ROOT/logs
. "$2"
remote() { bash -c "$1"; }
while [ ! -f "$3" ]; do sleep 0.01; done
reserve_root_ui
echo "reserved:$?"
"""


def test_two_launches_of_one_root_reserve_it_once(tmp_path):
    """Six launches of one root released together: exactly one wins, and it is the same root."""
    root = tmp_path / "scratch"
    go = tmp_path / "go"
    harness = tmp_path / "reserve.sh"
    harness.write_text(RACE, encoding="utf-8")
    processes = [subprocess.Popen(["bash", str(harness), str(root), str(LIBRARY), str(go)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(6)]
    go.write_text("", encoding="utf-8")
    results = [process.communicate(timeout=60) for process in processes]
    assert sum(1 for out, _ in results if out.strip() == "reserved:0") == 1, results
    assert sum(1 for _, err in results if "already being served or launched" in err) == 5, results

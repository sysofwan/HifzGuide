"""The S5 launcher's port decision (#311 PR3, self-check r3 S3.3): 8000, then 8002 once.

The locked launcher test pins the module, the default port and the fetch/push contract by
reading the script. This one runs the script against an ``ssh`` shim and pins the decision the
brief made: the owner reads on 10.0.1.195:8000, so an occupied 8000 retries once on 8002 — the
runner, the probe, the printed URL and the check hint all follow the port actually used — while
an explicit ``PORT`` refuses rather than guessing. The box is never touched; the staging target
is a throwaway one-commit repository, because the real one is large and slow.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "issue-311-audit.sh"
HOST = "root@box"
#: What `ss -ltnp` prints for a listener: `serves_from` matches the pid, the delimiter included.
LISTEN = 'LISTEN 0 5 0.0.0.0:8000 0.0.0.0:* users:(("python",pid=4242,fd=3))'


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


def _launch(tmp_path: Path, repo: Path, *args: str, bound: str = "",
            **env: str) -> subprocess.CompletedProcess:
    """Run the script with ``ssh`` shimmed; every remote command is appended to ``ssh.log``.

    Before the launch starts, only ``bound``'s port answers ``ss`` (the owner's other server);
    writing the runner creates the marker, after which the job's own listener answers on
    whatever port it asks about and ``pgrep -P`` reports its child. ``ui.pid`` then reads
    ``UI_PID``, and ``ps -p`` answers ``UI_PS`` — together, a ui already running on this root.
    The launch pipeline itself runs unchanged against those answers."""
    marker = tmp_path / "launched"
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    (shim / "ssh").write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> '{tmp_path}/ssh.log'
cat > /dev/null
cmd="$*"
case "$cmd" in *'run-ui.sh'*) : > '{marker}'; exit 0 ;; esac
case "$cmd" in *'test -f '*'ui.exit'*) exit 1 ;; esac
case "$cmd" in *"cat '"*'ui.pid'*) printf '%s' "${{UI_PID:-}}"; exit 0 ;; esac
case "$cmd" in *'ps -p'*) printf '%s\\n' "${{UI_PS:-}}"; exit 0 ;; esac
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


def test_an_occupied_8000_retries_once_on_8002_and_the_whole_launch_follows(tmp_path, stage):
    done = _launch(tmp_path, stage, "ui", "launch", bound="8000")
    assert done.returncode == 0, _text(done)
    assert "port 8000 on root@box is already bound; trying 8002" in done.stderr
    remote = _remote(tmp_path)
    runner = next(line for line in remote.splitlines() if "--port" in line)
    assert "tadabur.followalong_audit.s5_ui" in runner and "--port 8002" in runner
    assert "base = 'http://127.0.0.1:8002'" in remote          # the probe read the new port
    assert "  url:  http://box:8002/" in done.stdout
    assert "PORT=8002" in done.stdout                          # the hint carries the choice


def test_an_explicit_port_refuses_an_occupied_one_and_stages_nothing(tmp_path, stage):
    done = _launch(tmp_path, stage, "ui", "launch", bound="8000", PORT="8000")
    assert done.returncode == 1
    assert "port 8000 on root@box is already bound; nothing was staged or started" in done.stderr
    assert "job started" not in done.stdout
    remote = _remote(tmp_path)
    assert "git clone" not in remote and "run-ui.sh" not in remote


def test_a_free_8000_is_served_without_a_second_look(tmp_path, stage):
    done = _launch(tmp_path, stage, "ui", "launch")
    assert done.returncode == 0, _text(done)
    assert "  url:  http://box:8000/" in done.stdout and "trying" not in done.stderr
    remote = _remote(tmp_path)
    assert "--port 8000" in remote and ":8002 " not in remote


def test_a_root_that_already_serves_a_ui_refuses_a_second_launch_before_staging(tmp_path, stage):
    """The pilot is running on this root; `WITH_MAIN=1 ui launch` must not stage over it.

    The same scratch root means the same clone, runner, log and verdicts file — the second ui
    would answer from its own snapshot of that file. The refusal is the root's, not the port's:
    it fires even though 8002 is free and the fallback would happily take it. An occupied 8000
    with no ui of this root (the owner's other server) still falls back — see the test above."""
    root = tmp_path / "scratch"
    done = _launch(tmp_path, stage, "ui", "launch", bound="8000", UI_PID="4242",
                   UI_PS=f"/usr/bin/bash {root}/run-ui.sh")
    assert done.returncode == 1, _text(done)
    assert f"{root} on root@box already serves a ui (pid 4242)" in done.stderr
    assert "kill -- -4242" in done.stderr and "job started" not in done.stdout
    remote = _remote(tmp_path)
    assert "git clone" not in remote and "setsid" not in remote    # nothing staged or started
    assert ":8002 " not in remote and "trying" not in done.stderr  # no port was consulted


def test_a_pid_file_whose_runner_is_gone_does_not_block_the_launch(tmp_path, stage):
    """A run left a pid file behind (killed, rebooted); its pid now names no runner of ours."""
    done = _launch(tmp_path, stage, "ui", "launch", bound="8000", UI_PID="999", UI_PS="")
    assert done.returncode == 0, _text(done)
    assert "port 8000 on root@box is already bound; trying 8002" in done.stderr
    assert "  url:  http://box:8002/" in done.stdout

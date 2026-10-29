"""The catch-loss audit's launcher fetch (#225): a transport failure must fail loudly.

``scripts/issue-225-audit.sh ui fetch`` copies the owner's human data off cuda-dev. A file that
is not there yet — the review list before the first "review later", the backup directory before
its first run — is not an error; an unreachable box is, and must neither print a successful
fetch nor leave a previous sitting's local files looking current. The real script runs with
``ssh`` and ``rsync`` shims, so no box is touched.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS.parent / "scripts" / "issue-225-audit.sh"
HOST = "root@box"


def _shims(tmp_path: Path, *, ssh_status: int = 0, absent: str = "") -> Path:
    """``ssh`` exits ``ssh_status``, or 1 for a remote command naming ``absent``; ``rsync``
    records that it ran and succeeds."""
    shim = tmp_path / "bin"
    shim.mkdir(parents=True, exist_ok=True)
    ssh = shim / "ssh"
    ssh.write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> '{tmp_path}/ssh.log'
if [ -n '{absent}' ]; then
  case "$*" in *'{absent}'*) exit 1 ;; esac
fi
exit {ssh_status}
""", encoding="utf-8")
    rsync = shim / "rsync"
    rsync.write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> '{tmp_path}/rsync.log'
exit 0
""", encoding="utf-8")
    for path in (ssh, rsync):
        path.chmod(0o755)
    return shim


def _run(tmp_path: Path, job: str = "ui", **shims) -> subprocess.CompletedProcess:
    env = {**os.environ, "ISSUE_ROOT": str(tmp_path), "HOST": HOST,
           "PATH": f"{_shims(tmp_path, **shims)}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(SCRIPT), job, "fetch", str(tmp_path / "out")],
                          env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def _text(done: subprocess.CompletedProcess) -> str:
    return done.stdout + done.stderr


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_a_fetch_copies_the_verdicts_and_the_review_list(tmp_path):
    done = _run(tmp_path)

    assert done.returncode == 0, _text(done)
    assert "fetched ui into" in done.stdout
    assert len(_lines(tmp_path / "rsync.log")) == 2


def test_an_unreachable_box_fails_the_fetch_and_copies_nothing(tmp_path):
    done = _run(tmp_path, ssh_status=255)

    assert done.returncode == 255, _text(done)
    assert "fetched" not in done.stdout
    assert _lines(tmp_path / "rsync.log") == []
    assert "ssh" in done.stderr and "255" in done.stderr


def test_an_absent_review_list_is_not_a_failure(tmp_path):
    done = _run(tmp_path, absent="review.jsonl")

    assert done.returncode == 0, _text(done)
    assert "review.jsonl is absent" in done.stdout
    assert len(_lines(tmp_path / "rsync.log")) == 1


def test_an_unreachable_box_fails_the_backup_fetch_and_copies_nothing(tmp_path):
    done = _run(tmp_path, job="backup", ssh_status=255)

    assert done.returncode == 255, _text(done)
    assert "fetched" not in done.stdout
    assert _lines(tmp_path / "rsync.log") == []


def test_an_absent_backup_directory_is_not_a_failure(tmp_path):
    done = _run(tmp_path, job="backup", absent="backup")

    assert done.returncode == 0, _text(done)
    assert "backup is absent" in done.stdout
    assert _lines(tmp_path / "rsync.log") == []

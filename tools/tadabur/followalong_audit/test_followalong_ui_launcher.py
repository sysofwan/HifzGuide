"""The S5 listening UI's launcher on cuda-dev (#311 PR3, scout Q5).

Committed before ``scripts/issue-311-audit.sh`` exists and **locked**: later turns may add tests but
never weaken or rewrite these. The script is the issue-225 launcher's sibling (stage the branch,
run the UI detached under ``setsid``, keep the owner's verdicts in the scratch root, pull them back
to commit); the real script runs here with ``ssh`` and ``rsync`` shims, so no box is touched. Deploying
to cuda-dev is not part of PR3's acceptance.

Contract this file pins (``ISSUE_ROOT`` and ``HOST`` override the scratch root and the box)
-------------------------------------------------------------------------------------------
* ``ui fetch DIR``: one ``rsync`` of ``$ROOT/verdicts.jsonl`` into ``DIR``, then ``fetched ui into
  DIR``. A verdicts file not there yet is "``verdicts.jsonl is absent``", not a failure; an
  unreachable box exits with ssh's status, copies nothing and never prints a fetch.
* ``backup fetch DIR``: pulls the timestamped copies; an unreachable box fails loudly the same way,
  an absent backup directory is not a failure.
* ``wavs push [WAV_DIR]``: one ``rsync`` with ``--files-from`` naming exactly the ``<audio_sha256>.wav``
  of the committed worklist's clips (498 of the Mac's 5,742), to ``$HOST:$ROOT/wav/``.
* The launch runs ``python -m tadabur.followalong_audit.s5_ui`` on ``0.0.0.0``, default port 8000, with
  ``--wav-dir`` and ``--verdicts`` in the scratch root; the main sitting only behind ``WITH_MAIN``.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "issue-311-audit.sh"
WORKLIST = Path(__file__).resolve().parent / "s5_draw" / "worklist.jsonl"
HOST = "root@box"


def _shims(tmp_path: Path, *, ssh_status: int = 0, absent: str = "") -> Path:
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
prev=
for arg in "$@"; do
  case "$arg" in --files-from=*) cp "${{arg#--files-from=}}" '{tmp_path}/files-from.txt' ;; esac
  if [ "$prev" = --files-from ]; then cp "$arg" '{tmp_path}/files-from.txt'; fi
  prev=$arg
done
exit 0
""", encoding="utf-8")
    for path in (ssh, rsync):
        path.chmod(0o755)
    return shim


def _run(tmp_path: Path, *args: str, **shims) -> subprocess.CompletedProcess:
    env = {**os.environ, "ISSUE_ROOT": str(tmp_path / "scratch"), "HOST": HOST,
           "PATH": f"{_shims(tmp_path, **shims)}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _text(done: subprocess.CompletedProcess) -> str:
    return done.stdout + done.stderr


def test_the_script_exists_is_executable_and_is_valid_bash():
    assert SCRIPT.is_file() and SCRIPT.stat().st_mode & stat.S_IXUSR
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0


def test_the_launch_runs_the_s5_ui_on_all_interfaces_at_8000_with_the_scratch_files():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "tadabur.followalong_audit.s5_ui" in text
    assert "--host 0.0.0.0" in text and "--wav-dir" in text and "--verdicts" in text
    assert re.search(r"PORT:-8000\b", text)
    assert "setsid" in text


def test_the_main_sitting_is_started_only_behind_an_explicit_setting():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "WITH_MAIN" in text and "--with-main" in text


def test_a_ui_fetch_copies_the_verdicts_file_once(tmp_path):
    done = _run(tmp_path, "ui", "fetch", str(tmp_path / "out"))
    assert done.returncode == 0, _text(done)
    assert "fetched ui into" in done.stdout
    rsync = _lines(tmp_path / "rsync.log")
    assert len(rsync) == 1 and "verdicts.jsonl" in rsync[0]


def test_an_absent_verdicts_file_is_not_a_failure(tmp_path):
    done = _run(tmp_path, "ui", "fetch", str(tmp_path / "out"), absent="verdicts.jsonl")
    assert done.returncode == 0, _text(done)
    assert "verdicts.jsonl is absent" in done.stdout
    assert _lines(tmp_path / "rsync.log") == []


def test_an_unreachable_box_fails_the_fetch_and_copies_nothing(tmp_path):
    done = _run(tmp_path, "ui", "fetch", str(tmp_path / "out"), ssh_status=255)
    assert done.returncode == 255, _text(done)
    assert "fetched" not in done.stdout and _lines(tmp_path / "rsync.log") == []
    assert "ssh" in done.stderr and "255" in done.stderr


def test_an_unreachable_box_fails_the_backup_fetch_and_an_absent_backup_is_fine(tmp_path):
    failed = _run(tmp_path, "backup", "fetch", str(tmp_path / "out"), ssh_status=255)
    assert failed.returncode == 255 and _lines(tmp_path / "rsync.log") == []
    absent = _run(tmp_path, "backup", "fetch", str(tmp_path / "out"), absent="backup")
    assert absent.returncode == 0, _text(absent)


def test_the_wav_push_sends_exactly_the_drawn_clips_wavs(tmp_path):
    done = _run(tmp_path, "wavs", "push", str(tmp_path / "wavs"))
    assert done.returncode == 0, _text(done)
    rsync = _lines(tmp_path / "rsync.log")
    assert len(rsync) == 1 and f"{HOST}:{tmp_path / 'scratch'}/wav" in rsync[0]
    drawn = {json.loads(line)["audio_sha256"] + ".wav"
             for line in WORKLIST.read_text(encoding="utf-8").splitlines() if line.strip()}
    sent = _lines(tmp_path / "files-from.txt")
    assert len(drawn) == 498 and set(sent) == drawn and len(sent) == len(drawn)


def test_an_unreachable_box_fails_the_wav_push(tmp_path):
    done = _run(tmp_path, "wavs", "push", str(tmp_path / "wavs"), ssh_status=255)
    assert done.returncode == 255, _text(done)
    assert _lines(tmp_path / "rsync.log") == []

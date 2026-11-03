"""The catch-loss audit's launcher is spec-aware (#257): round 3 beside round 2's, untouched.

``scripts/issue-225-audit.sh`` was written for round 2 and hardcoded its readout directory,
scratch root and port. ``SPEC=consonant3`` points the same ``draw`` / ``ui`` / ``backup`` jobs
at round 3's committed worklist, its own scratch root (so its verdicts never share a file with
round 2's) and its own port (round 2's UI keeps serving on 8000, untouched); no ``SPEC`` is
round 2 exactly as before. The real script runs against an ``ssh`` shim that records every
remote command, including the runner script ``launch`` writes on the box, and a throwaway
git repository to stage, so no box is touched.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "issue-225-audit.sh"
HOST = "root@box"
CLIPS = "/root/scratch/tlog/r2/clips"

R2_ROOT = "/root/scratch/catch_loss"
R3_ROOT = "/root/scratch/catch_loss_r3"
R2_AUDIT = "tools/tadabur/consonant2_readout/catch_loss"
R3_AUDIT = "tools/tadabur/consonant3_readout/catch_loss"
TRANCHE = "tools/tadabur/tlog/tranches/look.jsonl"


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    """A one-commit repository for the launcher to bundle (the real one is large and slow)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    (repo / "README").write_text("staged\n", encoding="utf-8")
    subprocess.run([*git, "add", "README"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "stage"], check=True)
    return repo


def _launch(tmp_path: Path, repo: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    """Run the script with ``ssh`` (every remote command appended to ``ssh.log``) and ``rsync``
    shimmed; ``ISSUE_ROOT`` is left to the script unless a test sets it."""
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    (shim / "ssh").write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> '{tmp_path}/ssh.log'
cat > /dev/null
exit 0
""", encoding="utf-8")
    (shim / "rsync").write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> '{tmp_path}/rsync.log'
exit 0
""", encoding="utf-8")
    for tool in ("ssh", "rsync"):
        (shim / tool).chmod(0o755)
    base = {key: value for key, value in os.environ.items()
            if key not in {"SPEC", "ISSUE_ROOT", "PORT", "CLIPS"}}
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env={**base, "HOST": HOST, "REPO_DIR": str(repo),
             "PATH": f"{shim}:{os.environ['PATH']}", **env},
        capture_output=True, text=True, stdin=subprocess.DEVNULL)


def _remote(tmp_path: Path) -> str:
    """Everything the script asked the box to do, runner scripts included."""
    log = tmp_path / "ssh.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _fetched(tmp_path: Path) -> str:
    log = tmp_path / "rsync.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _text(done: subprocess.CompletedProcess) -> str:
    return done.stdout + done.stderr


def _ui_command(root: str, audit: str, port: int) -> str:
    return (f"python -u -m tadabur.tashkeel_audit_ui --catch-loss "
            f"'{root}/repo/{audit}/worklist.jsonl' --clip-cache '{CLIPS}' "
            f"--verdicts '{root}/verdicts.jsonl' --review '{root}/review.jsonl' "
            f"--port {port} --host 0.0.0.0")


# --- round 2 is unchanged -----------------------------------------------------------------------


@pytest.mark.parametrize("env", [{}, {"SPEC": "consonant2"}], ids=["no-spec", "consonant2"])
def test_round_two_serves_its_worklist_on_8000_exactly_as_before(tmp_path, stage, env):
    done = _launch(tmp_path, stage, "ui", "launch", **env)

    assert done.returncode == 0, _text(done)
    assert _ui_command(R2_ROOT, R2_AUDIT, 8000) in _remote(tmp_path)
    assert "consonant3" not in _remote(tmp_path)


@pytest.mark.parametrize("env", [{}, {"SPEC": "consonant2"}], ids=["no-spec", "consonant2"])
def test_round_two_draws_into_its_scratch_root_with_no_spec_argument(tmp_path, stage, env):
    done = _launch(tmp_path, stage, "draw", "launch", **env)

    assert done.returncode == 0, _text(done)
    remote = _remote(tmp_path)
    assert f"python -u -m tadabur.catch_loss_audit draw --tranche '{R2_ROOT}/repo/{TRANCHE}'" in remote
    assert f"cp '{R2_ROOT}/repo/{R2_AUDIT}/worklist.jsonl'" in remote
    assert "consonant3" not in remote


# --- round 3 ------------------------------------------------------------------------------------


def test_round_three_serves_its_committed_worklist_on_8001(tmp_path, stage):
    done = _launch(tmp_path, stage, "ui", "launch", SPEC="consonant3")

    assert done.returncode == 0, _text(done)
    assert _ui_command(R3_ROOT, R3_AUDIT, 8001) in _remote(tmp_path)


def test_round_three_stages_and_logs_under_its_own_scratch_root(tmp_path, stage):
    done = _launch(tmp_path, stage, "ui", "launch", SPEC="consonant3")

    assert done.returncode == 0, _text(done)
    remote = _remote(tmp_path)
    assert f"mkdir -p '{R3_ROOT}' '{R3_ROOT}/logs'" in remote
    assert f"'{R3_ROOT}/logs/ui.exit'" in remote
    assert f"{R3_ROOT}/run-ui.sh" in remote


def test_round_three_never_names_round_twos_files(tmp_path, stage):
    """Round 2's verdicts, review list, clone and logs live under ``/root/scratch/catch_loss``;
    a relaunch of round 3 re-clones its own root and must not reach them."""
    for job in ("draw", "ui", "backup"):
        assert _launch(tmp_path, stage, job, "launch", SPEC="consonant3").returncode == 0

    remote = _remote(tmp_path)
    assert f"{R2_ROOT}/" not in remote and f"{R2_ROOT}'" not in remote
    assert "consonant2" not in remote


def test_round_three_draws_with_its_spec_from_the_look_tranche(tmp_path, stage):
    done = _launch(tmp_path, stage, "draw", "launch", SPEC="consonant3")

    assert done.returncode == 0, _text(done)
    remote = _remote(tmp_path)
    assert ("python -u -m tadabur.catch_loss_audit draw --spec consonant3 "
            f"--tranche '{R3_ROOT}/repo/{TRANCHE}'") in remote
    for name in ("worklist.jsonl", "key.jsonl", "summary.json"):
        assert f"'{R3_ROOT}/repo/{R3_AUDIT}/{name}'" in remote
    assert f"'{R3_ROOT}/'" in remote  # the copies land in the job's scratch root


def test_round_three_backs_up_its_own_verdicts_and_review_list(tmp_path, stage):
    done = _launch(tmp_path, stage, "backup", "launch", SPEC="consonant3")

    assert done.returncode == 0, _text(done)
    remote = _remote(tmp_path)
    # The substitution must sit outside the destination quotes, or the loop writes one file
    # literally named `verdicts.$(date +%s).jsonl` and the retention never trims anything.
    assert (f"cp '{R3_ROOT}/verdicts.jsonl' '{R3_ROOT}/backup/verdicts.'$(date +%s)'.jsonl'"
            in remote)
    assert (f"cp '{R3_ROOT}/review.jsonl' '{R3_ROOT}/backup/review.'$(date +%s)'.jsonl'"
            in remote)


def test_the_clip_cache_is_still_overridable_for_round_three(tmp_path, stage):
    done = _launch(tmp_path, stage, "ui", "launch", SPEC="consonant3", CLIPS="/elsewhere/clips")

    assert done.returncode == 0, _text(done)
    assert "--clip-cache '/elsewhere/clips'" in _remote(tmp_path)


def test_round_three_fetches_its_verdicts_review_list_and_draw_from_its_root(tmp_path, stage):
    for job in ("ui", "draw", "backup"):
        done = _launch(tmp_path, stage, job, "fetch", str(tmp_path / "out"), SPEC="consonant3")
        assert done.returncode == 0, _text(done)

    fetched = _fetched(tmp_path)
    assert f"{HOST}:{R3_ROOT}/verdicts.jsonl" in fetched
    assert f"{HOST}:{R3_ROOT}/review.jsonl" in fetched
    assert f"{HOST}:{R3_ROOT}/ " in fetched  # the draw's three files, from the scratch root
    assert f"{HOST}:{R3_ROOT}/backup/" in fetched
    assert R2_ROOT + "/" not in fetched and R2_ROOT + "'" not in fetched


# --- the launcher never reaches what it does not own --------------------------------------------


@pytest.mark.parametrize("spec", ["consonant2", "consonant3"])
@pytest.mark.parametrize("job", ["draw", "ui", "backup"])
def test_a_launch_stops_nothing_and_leaves_the_gpu_alone(tmp_path, stage, spec, job):
    """Round 2's UI holds 8000 and mining phase B holds the GPU: no launch may kill a process
    or touch ``gpu.lock``, the mining directory or its PAUSE file."""
    done = _launch(tmp_path, stage, job, "launch", SPEC=spec)

    assert done.returncode == 0, _text(done)
    remote = _remote(tmp_path)
    for forbidden in ("kill", "gpu.lock", "tlog-mining", "PAUSE", "flock", "nvidia"):
        assert forbidden not in remote


# --- the spec selector --------------------------------------------------------------------------


def test_an_unknown_spec_is_refused_before_anything_reaches_the_box(tmp_path, stage):
    done = _launch(tmp_path, stage, "ui", "launch", SPEC="consonant9")

    assert done.returncode == 2, _text(done)
    assert "consonant9" in done.stderr
    assert _remote(tmp_path) == ""
    assert _fetched(tmp_path) == ""


def test_an_unknown_spec_is_refused_for_a_fetch_too(tmp_path, stage):
    done = _launch(tmp_path, stage, "ui", "fetch", str(tmp_path / "out"), SPEC="consonant9")

    assert done.returncode == 2, _text(done)
    assert _remote(tmp_path) == "" and _fetched(tmp_path) == ""

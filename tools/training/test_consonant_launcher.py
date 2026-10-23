"""The launcher's data assertion must be able to stop a launch.

`run` calls `assert_ready || return 1`, and bash disables errexit for the whole body of a
function called that way -- so every step inside the assertion has to carry its own
`|| return 1`. A swallowed failure here is worse than no check: the run note tells the
architect that a drifted shard list would refuse to launch, and the run launches anyway.

These execute the real script (its `verify` subcommand is the assertion alone, without the
trainer or the GPU) in a temporary `ISSUE_ROOT`, with a `python` shim on PATH so the host's
interpreter name does not matter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS.parent / "scripts" / "issue-178-train.sh"

#: The head of the seed-9494 permutation, as #94's log reports it.
OBSERVED_HEAD = (169, 354, 207, 312, 227)


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    python = shim / "python"
    if not python.exists():
        python.symlink_to(sys.executable)
    env = {**os.environ, "ISSUE_ROOT": str(tmp_path), "PATH": f"{shim}:{os.environ['PATH']}"}
    env.update(overrides)
    return env


def _verify(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), "verify"],
        cwd=TOOLS,
        env=_env(tmp_path, **overrides),
        capture_output=True,
        text=True,
    )


def test_the_data_assertion_fails_when_the_log_yields_no_prefix(tmp_path):
    """An empty log must stop the launch, not select one consumed shard and carry on."""
    log = tmp_path / "empty.log"
    log.write_text("", encoding="utf-8")
    done = _verify(tmp_path, SOURCE_LOG=str(log))
    assert done.returncode != 0
    assert "no shard-completion" in done.stdout + done.stderr
    assert not (tmp_path / "logs" / "spec.txt").exists()


def test_the_data_assertion_fails_when_the_registration_has_drifted(tmp_path):
    """The committed shard use is compared with a fresh derivation; a mismatch must exit.

    The index here covers nothing, which is how a drifted or re-derived spec looks to the
    check: every shard the derivation names is missing from it.
    """
    log = tmp_path / "source.log"
    log.write_text(
        "".join(
            f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n"
            for shard in OBSERVED_HEAD
        ),
        encoding="utf-8",
    )
    index = tmp_path / "index.jsonl"
    index.write_text("", encoding="utf-8")
    done = _verify(tmp_path, SOURCE_LOG=str(log), INDEX=str(index))
    assert done.returncode != 0
    assert "does not index" in done.stdout + done.stderr
    assert not (tmp_path / "logs" / "spec.txt").exists()

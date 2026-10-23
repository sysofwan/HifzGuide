"""The launcher's data assertion must be able to stop a launch.

`run` calls `assert_ready || return 1`, and bash disables errexit for the whole body of a
function called that way -- so every step inside the assertion has to carry its own
`|| return 1`. A swallowed failure here is worse than no check: the run note tells the
architect that a drifted shard list would refuse to launch, and the run launches anyway.

These execute the real script (its `verify`, `gate` and `monitor` subcommands are the
assertion, the smoke gate and the post-run monitoring guard alone, without the trainer or
the GPU) in a temporary `ISSUE_ROOT`. The `monitor` tests run with the box's shell shape —
`/usr/bin:/bin` and a conda stub that supplies `python` — so a subcommand that forgets to
activate its environment fails here as it did on cuda-dev.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS.parent / "scripts" / "issue-178-train.sh"

#: The head of the seed-9494 permutation, as #94's log reports it.
OBSERVED_HEAD = (169, 354, 207, 312, 227)

#: #178's seed: the continuation's shard order (`--monitor-seed`).
RUN_SEED = 178


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    python = shim / "python"
    if not python.exists():
        python.symlink_to(sys.executable)
    env = {**os.environ, "ISSUE_ROOT": str(tmp_path), "PATH": f"{shim}:{os.environ['PATH']}"}
    env.update(overrides)
    return env


def _fresh_shell_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    """The box as `report` sees it: a fresh SSH shell with no `python` on PATH.

    The only interpreter is the one the conda environment provides, so a subcommand that
    does not activate the environment fails here exactly as it did on cuda-dev (exit 127).
    """
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    python = prefix / "bin" / "python"
    if not python.exists():
        python.symlink_to(sys.executable)
    conda = tmp_path / "conda.sh"
    conda.write_text(
        f'conda() {{ PATH="{prefix}/bin:$PATH"; export PATH; }}\n', encoding="utf-8"
    )
    env = {
        **os.environ,
        "ISSUE_ROOT": str(tmp_path),
        "PATH": "/usr/bin:/bin",
        "CONDA": str(conda),
    }
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


# --- the smoke gate ------------------------------------------------------------------


def _gate(tmp_path: Path, metrics: list[dict], log_text: str) -> subprocess.CompletedProcess:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in metrics), encoding="utf-8"
    )
    log = tmp_path / "smoke.log"
    log.write_text(log_text, encoding="utf-8")
    return subprocess.run(
        ["bash", str(SCRIPT), "gate", str(metrics_path), str(log)],
        cwd=TOOLS,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
    )


#: Two training records at ~0.9 steps/s: 300 steps in 330 s.
TRAINED = [
    {"step": 0, "elapsed_s": 0.0, "total": 2.0},
    {"step": 275, "elapsed_s": 305.0, "total": 1.5},
]
#: One evaluation of the staged clips, as `evaluate` logs it.
VALIDATED = {"step": 0, "split": "val", "agreement": 0.41, "target_prob": 0.2}
WARM_START_EMA = (
    "[init] warm-started from checkpoint.pt (its step 10000, weights=ema); "
    "fresh optimiser and schedule\n"
)


def test_the_smoke_gate_refuses_a_smoke_that_never_validated(tmp_path):
    """`--eval-every` must reach the staged clips in the smoke, not 40 min into the full run."""
    done = _gate(tmp_path, TRAINED, WARM_START_EMA)
    assert done.returncode != 0
    assert "no evaluation of the staged clips" in done.stdout + done.stderr


def test_the_smoke_gate_refuses_an_empty_validation_split(tmp_path):
    """A val split that resolved to no clip logs `{}` and would otherwise pass silently."""
    done = _gate(tmp_path, [*TRAINED, {"step": 0, "split": "val"}], WARM_START_EMA)
    assert done.returncode != 0
    assert "val split was empty" in done.stdout + done.stderr


def test_the_smoke_gate_refuses_a_warm_start_that_is_not_the_averaged_weights(tmp_path):
    """The averaged-weights warm start is the candidate's defining ingredient."""
    live = WARM_START_EMA.replace("weights=ema", "weights=live")
    done = _gate(tmp_path, [*TRAINED, VALIDATED], live)
    assert done.returncode != 0
    assert "not the averaged weights" in done.stdout + done.stderr


def test_the_smoke_gate_accepts_a_smoke_that_validated_and_warm_started_in_ema(tmp_path):
    done = _gate(tmp_path, [*TRAINED, VALIDATED], WARM_START_EMA)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "steps/s" in done.stdout
    assert "[gate] 1 val evaluations" in done.stdout
    assert "weights=ema" in done.stdout


# --- the post-run monitoring guard ----------------------------------------------------


def _derived(tmp_path: Path) -> tuple[Path, list[int], int]:
    """The source log of the `verify` tests, the shards left, and the monitor shard's position."""
    from training.consumed_prefix import monitoring_shard, remaining_from_log

    log = tmp_path / "source.log"
    log.write_text(
        "".join(
            f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n"
            for shard in OBSERVED_HEAD
        ),
        encoding="utf-8",
    )
    remaining = remaining_from_log(log)
    return log, remaining, monitoring_shard(remaining, RUN_SEED)[1]


def _monitor(
    tmp_path: Path, completions: int, **overrides: str
) -> subprocess.CompletedProcess:
    from training.consumed_prefix import shard_permutation

    log, remaining, _ = _derived(tmp_path)
    streamed = tmp_path / "train.log"
    streamed.write_text(
        "".join(
            f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n"
            for shard in shard_permutation(remaining, RUN_SEED)[:completions]
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(SCRIPT), "monitor", str(streamed)],
        cwd=TOOLS,
        env=_fresh_shell_env(tmp_path, SOURCE_LOG=str(log), **overrides),
        capture_output=True,
        text=True,
    )


def test_the_monitoring_guard_refuses_a_run_that_entered_the_validation_shard(tmp_path):
    """10,000 steps stop ~65 shards in, but the guard must not depend on that staying true."""
    _, _, position = _derived(tmp_path)
    done = _monitor(tmp_path, position)
    assert done.returncode != 0
    assert "entered monitor shard" in done.stdout + done.stderr


def test_the_monitoring_guard_accepts_a_run_that_stopped_short(tmp_path):
    _, _, position = _derived(tmp_path)
    done = _monitor(tmp_path, position - 1)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "never streamed" in done.stdout


def test_the_monitor_hook_finds_python_only_through_the_environment(tmp_path):
    """R1.2: `report` runs `monitor` from a fresh shell, where only conda provides `python`."""
    assert shutil.which("python", path="/usr/bin:/bin") is None  # the shell has no python
    _, _, position = _derived(tmp_path)
    done = _monitor(tmp_path, position - 1)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "never streamed" in done.stdout


def test_the_monitor_hook_fails_when_the_environment_cannot_be_activated(tmp_path):
    """The control: activation is what supplies the interpreter, not the host's PATH."""
    _, _, position = _derived(tmp_path)
    done = _monitor(tmp_path, position - 1, CONDA=str(tmp_path / "missing-conda.sh"))
    assert done.returncode != 0
    assert "never streamed" not in done.stdout

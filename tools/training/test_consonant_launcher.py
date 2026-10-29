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
#: The round-2 control (#223): #178's recipe once more, on Tadabur alone. Its `verify`, `gate`,
#: `monitor` and `ready` subcommands are the same cut points, so these tests drive it too.
CONTROL = TOOLS.parent / "scripts" / "issue-223-control.sh"

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


# --- the round-2 control launcher (#223) ----------------------------------------------


#: The control's seed: its stream's shard order and its monitor shard. The candidate's, so
#: that candidate minus control is the tlog data alone.
CONTROL_SEED = 2026
#: #178's full-run checkpoint -- live and averaged weights in one file -- is the warm start.
CONTROL_INIT_SHA = "057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2"


def _completion_lines(shards) -> str:
    return "".join(
        f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n" for shard in shards
    )


def _control_sources(tmp_path: Path) -> tuple[Path, Path, list[int], list[int]]:
    """Stub logs of #94 and #178, the shards #94 left, and the shards both left.

    #94's log is the head of the seed-9494 permutation of `h448`'s 345 shards; #178's is the
    head of the seed-178 permutation of what that leaves, as the two runs on the box
    streamed. The control's corpus is what remains once both prefixes are off.
    """
    from tadabur.shard_reader import parse_shard_spec
    from training.consumed_prefix import (
        AUG94_SEED,
        consumed_prefix,
        remaining_shards,
        shard_permutation,
    )
    from training.decode_evalset import training_shard_spec

    source = parse_shard_spec(training_shard_spec())
    after_94 = remaining_shards(source, consumed_prefix(source, OBSERVED_HEAD, AUG94_SEED))
    streamed_178 = shard_permutation(after_94, RUN_SEED)[:4]
    after_both = remaining_shards(
        after_94, consumed_prefix(after_94, streamed_178, RUN_SEED)
    )
    log_94 = tmp_path / "aug94.log"
    log_94.write_text(_completion_lines(OBSERVED_HEAD), encoding="utf-8")
    log_178 = tmp_path / "train178.log"
    log_178.write_text(_completion_lines(streamed_178), encoding="utf-8")
    return log_94, log_178, after_94, after_both


def _control_verify(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(CONTROL), "verify"],
        cwd=TOOLS,
        env=_env(tmp_path, **overrides),
        capture_output=True,
        text=True,
    )


def _stub_index(path: Path, shards: list[int]) -> None:
    """One indexed row per shard, so the registration check gets past its coverage test."""
    path.write_text(
        "".join(
            json.dumps(
                {
                    "audio_filename": f"{shard}_0000.wav",
                    "shard": shard,
                    "row_index": 0,
                    "reciter_id": 1,
                    "surah_id": 1,
                    "ayah_id": 1,
                    "ayah_duration_s": 1.0,
                }
            )
            + "\n"
            for shard in shards
        ),
        encoding="utf-8",
    )


def test_the_control_excludes_the_shards_both_earlier_runs_streamed(tmp_path):
    """Candidate minus control is the tlog data alone only if neither #94's nor #178's audio
    is in the control's corpus."""
    from tadabur.shard_reader import shard_spec
    from training.consumed_prefix import shard_permutation

    log_94, log_178, after_94, after_both = _control_sources(tmp_path)
    index = tmp_path / "index.jsonl"
    index.write_text("", encoding="utf-8")  # the check stops here, after the derivation
    done = _control_verify(
        tmp_path, SOURCE_LOG_94=str(log_94), SOURCE_LOG_178=str(log_178), INDEX=str(index)
    )
    printed = [line for line in done.stdout.splitlines() if "stream shards:" in line]
    assert printed, done.stdout + done.stderr
    assert printed[-1].endswith(shard_spec(after_both))
    # #178's first shard is inside #94's remainder, so only the chained derivation takes it
    # off: this is the second run's exclusion.
    assert shard_permutation(after_94, RUN_SEED)[0] not in after_both
    assert not (tmp_path / "logs" / "spec.txt").exists()


def test_the_controls_seed_names_the_monitor_shard_it_never_streams(tmp_path):
    """The monitor shard is the last of the run's own seed permutation, so the seed is
    observable without reading the trainer's arguments."""
    from training.consumed_prefix import monitoring_shard

    log_94, log_178, _, after_both = _control_sources(tmp_path)
    index = tmp_path / "index.jsonl"
    index.write_text("", encoding="utf-8")
    done = _control_verify(
        tmp_path, SOURCE_LOG_94=str(log_94), SOURCE_LOG_178=str(log_178), INDEX=str(index)
    )
    monitor, position = monitoring_shard(after_both, CONTROL_SEED)
    assert f"monitor shard {monitor}:" in done.stdout
    assert f"after {position} completions" in done.stdout
    assert monitor != monitoring_shard(after_both, RUN_SEED)[0]


def test_the_control_refuses_a_second_log_with_no_completions(tmp_path):
    """A truncated #178 log would leave its whole prefix back in the corpus."""
    log_94, log_178, _, _ = _control_sources(tmp_path)
    log_178.write_text("", encoding="utf-8")
    done = _control_verify(
        tmp_path, SOURCE_LOG_94=str(log_94), SOURCE_LOG_178=str(log_178)
    )
    assert done.returncode != 0
    assert "no shard-completion" in done.stdout + done.stderr
    assert not (tmp_path / "logs" / "spec.txt").exists()


def test_the_control_refuses_a_warm_start_that_is_not_the_178_checkpoint(tmp_path):
    """The averaged #178 weights are the arm's defining ingredient; the launcher pins the
    file's SHA-256, so a truncated or re-run file cannot pass as them."""
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    sha = shim / "sha256sum"
    sha.write_text(
        '#!/bin/sh\necho "0000000000000000000000000000000000000000000000000000000000000000  $1"\n',
        encoding="utf-8",
    )
    sha.chmod(0o755)
    init = tmp_path / "checkpoint.pt"
    init.write_bytes(b"not the averaged weights")
    done = subprocess.run(
        ["bash", str(CONTROL), "ready"],
        cwd=TOOLS,
        env=_env(tmp_path, INIT=str(init)),
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0
    assert "init checkpoint is 0" in done.stdout + done.stderr
    assert CONTROL_INIT_SHA in done.stdout + done.stderr


def test_the_control_registers_its_shard_use_as_the_round_2_control(tmp_path):
    """`--check` proves the registered shard set at launch; it must be the control's own use,
    never #178's registration reused for a corpus #178 has since streamed."""
    log_94, log_178, _, after_both = _control_sources(tmp_path)
    index = tmp_path / "index.jsonl"
    _stub_index(index, after_both)
    done = _control_verify(
        tmp_path, SOURCE_LOG_94=str(log_94), SOURCE_LOG_178=str(log_178), INDEX=str(index)
    )
    assert done.returncode != 0
    assert "consonant2.control.training" in done.stdout + done.stderr
    assert not (tmp_path / "logs" / "spec.txt").exists()


def _control_launch(
    tmp_path: Path, *args: str, run_dir: bool = False
) -> subprocess.CompletedProcess:
    """`launch` against a stand-in box: every ssh is answered by a shim and logged."""
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    log = tmp_path / "ssh.log"
    ssh = shim / "ssh"
    ssh.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                f'printf \'%s\\n\' "$*" >> "{log}"',
                'case "$*" in',
                '  *"tar -x"*) cat > /dev/null ;;',  # the archived tree arrives on stdin
                '  *"test -e"*) [ "%d" = 1 ] && echo exists ;;' % run_dir,
                "esac",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    return subprocess.run(
        ["bash", str(CONTROL), "launch", *args],
        cwd=TOOLS.parent,
        env=_env(tmp_path, HOST="box"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )


def test_the_launch_refuses_a_previous_run_dir_so_a_retry_keeps_its_checkpoints(tmp_path):
    """#221: the #178 launch re-extracted the repo over the run directories. This arm's
    checkpoints are its only artifact, so an existing run dir is refused -- and nothing is
    archived or started first."""
    done = _control_launch(tmp_path, run_dir=True)
    assert done.returncode == 4
    assert "refusing" in done.stderr
    issued = (tmp_path / "ssh.log").read_text(encoding="utf-8")
    assert "tar -x" not in issued
    assert "setsid" not in issued


def test_launch_resume_keeps_the_run_dir_and_starts_detached(tmp_path):
    """#221: `nohup`'s inherited streams kept the launching ssh session open. The run starts
    under `setsid` with stdin from /dev/null, and a resume keeps the checkpoints' directory."""
    done = _control_launch(tmp_path, "--resume", run_dir=True)
    assert done.returncode == 0, done.stdout + done.stderr
    issued = (tmp_path / "ssh.log").read_text(encoding="utf-8")
    started = [line for line in issued.splitlines() if "setsid" in line]
    assert started, issued
    assert "< /dev/null" in started[-1]
    assert "RESUME=1" in started[-1]


def _control_manifest(tmp_path: Path) -> subprocess.CompletedProcess:
    """`manifest` reads the spec the data assertion wrote, as `chain` reaches it."""
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "spec.txt").write_text("21-23,40-41,90,383\n", encoding="utf-8")
    return subprocess.run(
        ["bash", str(CONTROL), "manifest"],
        cwd=TOOLS,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
    )


def test_the_launch_records_the_frame_the_candidate_must_replay(tmp_path):
    """Same seed is not enough: the shard order and the window shuffle draw from different
    seeds, so the candidate's all-Tadabur substream needs the frame, both seeds and the
    recipe record to reproduce this arm's input."""
    from tadabur.shard_reader import parse_shard_spec
    from training.consumed_prefix import shard_permutation
    from training.distill_stream import window_shuffle_seed
    from training.frame_manifest import shards_digest

    done = _control_manifest(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = json.loads(
        (tmp_path / "runs" / "consonant2_control" / "frame_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    shards = parse_shard_spec("21-23,40-41,90,383")
    assert manifest["shards"] == shards
    assert manifest["shards_sha256"] == shards_digest(shards)
    assert manifest["stream_order"] == shard_permutation(shards, CONTROL_SEED)
    assert manifest["shard_order_seed"] == CONTROL_SEED
    assert manifest["window_shuffle_seed"] == window_shuffle_seed(CONTROL_SEED) == 2027
    assert manifest["augmentation"]["version"] == "aug94-mixture-v2"
    assert manifest["exposure_use"] == "consonant2.control.training"

"""Acceptance tests for the round-2 candidate's launcher, ``scripts/issue-228-train.sh`` (#228),
written before the script.

Locked: later turns may add tests but never weaken or rewrite these.

Like ``test_consonant_launcher``, these run the real script's subcommands in a temporary
``ISSUE_ROOT`` without a GPU. The script is ``issue-223-control.sh``'s sibling: the same box,
warm start, seed and schedule, with the tlog stream added. Contract:

* ``frame``: prints ``monitor=``, ``position=`` and ``stream=`` (a shard spec) for the frame the
  candidate replays -- the control's 211 shards, read from the committed
  ``consonant2.control.training`` use -- with the monitor shard the last of its seed-2026 order.
* ``manifest``: writes ``$ISSUE_ROOT/runs/consonant2_candidate/frame_manifest.json`` for the
  spec in ``logs/spec.txt`` and the use ``consonant2.training``.
* ``ready``: refuses a warm start whose SHA-256 is not #178's.
* ``verify``: the data assertion. It must run ``tadabur.exposure training-stream --use
  consonant2.training --membership exact --check`` and ``tadabur.exposure
  tlog-training-disjoint`` (never ``check_disjoint``), stop on either failing, and only then write
  ``logs/spec.txt``.
* ``gate METRICS SMOKE_LOG SHARDS_LOG CONSUMED_KEYS``: the smoke gate (throughput, finite KL from
  both sources, every batch 24 + 8, both splits consumed, at most 8 tlog shards on disk, the
  ``weights=ema`` line).
* ``monitor LOG``: the post-run guard on the frame's monitor shard.
* ``freeze-check``: run in a git checkout; refuses unless the pre-registration, the freeze record
  (``$FREEZE_RECORD``, a path in the repo), ``tlog/training_schedule.json`` and
  ``tlog/training_prefix.json`` are committed in HEAD.
* ``launch [--smoke]``: a full launch runs ``freeze-check`` first and touches the box only if it
  passes; ``--smoke`` skips it and starts the box's run with ``SMOKE_ONLY=1``. A held
  ``gpu.lock`` exits 3. ``--resume`` is refused: the mixed stream cannot replay its consumed-keys
  log. HEAD is staged as a bundle clone (a checkout), not a ``git archive`` tree -- the tlog
  reader checks the overlay commit out of the staged tree's own history.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS.parent / "scripts" / "issue-228-train.sh"
CONTROL_SHARDS_PATH = TOOLS / "tadabur" / "exposure" / "consonant2.control.training.shards.json"
INIT_SHA = "057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2"
SEED = 2026
FREEZE_RECORD = "docs/consonant2-freeze.json"
FROZEN = ("docs/consonant-round2-preregistration.md", FREEZE_RECORD,
          "tools/tadabur/tlog/training_schedule.json", "tools/tadabur/tlog/training_prefix.json")


def _shards() -> list[int]:
    return json.loads(CONTROL_SHARDS_PATH.read_text(encoding="utf-8"))["shards"]


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
    """The box as `report` sees it: no `python` on PATH until the conda env is activated."""
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    python = prefix / "bin" / "python"
    if not python.exists():
        python.symlink_to(sys.executable)
    conda = tmp_path / "conda.sh"
    conda.write_text(f'conda() {{ PATH="{prefix}/bin:$PATH"; export PATH; }}\n', encoding="utf-8")
    env = {**os.environ, "ISSUE_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin", "CONDA": str(conda)}
    env.update(overrides)
    return env


def _run(env: dict[str, str], *args: str, cwd: Path = TOOLS) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL)


def _text(done: subprocess.CompletedProcess) -> str:
    return done.stdout + done.stderr


# --- the frame ------------------------------------------------------------------------------


def test_the_frame_is_the_controls_with_its_monitor_shard_last_in_the_seed_2026_order(tmp_path):
    from tadabur.shard_reader import parse_shard_spec
    from training.consumed_prefix import monitoring_shard

    done = _run(_env(tmp_path), "frame")
    assert done.returncode == 0, _text(done)
    printed = dict(line.split("=", 1) for line in done.stdout.splitlines() if "=" in line)
    assert parse_shard_spec(printed["stream"]) == _shards()
    monitor, position = monitoring_shard(_shards(), SEED)
    assert (int(printed["monitor"]), int(printed["position"])) == (monitor, position)


def test_the_manifest_records_the_frame_the_control_ran(tmp_path):
    from tadabur.shard_reader import shard_spec

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "spec.txt").write_text(shard_spec(_shards()) + "\n", encoding="utf-8")
    done = _run(_env(tmp_path), "manifest")
    assert done.returncode == 0, _text(done)
    manifest = json.loads((tmp_path / "runs" / "consonant2_candidate" / "frame_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["shards"] == _shards()
    assert manifest["shards_sha256"] == (
        "4bd00c2ba5d0979bcc017546e3beeea70807dad443b125fa293adfe7f5f5d7d8")
    assert manifest["stream_order_sha256"] == (
        "7f0f1533618d0fc4e32e1fa34e20c4d1d22f4995779ad816a58e9d54193674fe")
    assert (manifest["shard_order_seed"], manifest["window_shuffle_seed"]) == (SEED, SEED + 1)
    assert manifest["exposure_use"] == "consonant2.training"
    assert manifest["augmentation"]["version"] == "aug94-mixture-v2"


def test_a_warm_start_that_is_not_the_178_checkpoint_refuses_the_launch(tmp_path):
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    sha = shim / "sha256sum"
    sha.write_text('#!/bin/sh\necho "' + "0" * 64 + '  $1"\n', encoding="utf-8")
    sha.chmod(0o755)
    init = tmp_path / "checkpoint.pt"
    init.write_bytes(b"not the averaged weights")
    done = _run(_env(tmp_path, INIT=str(init)), "ready")
    assert done.returncode != 0
    assert "init checkpoint is 0" in _text(done)
    assert INIT_SHA in _text(done)


# --- the data assertion ----------------------------------------------------------------------


def _python_shim(tmp_path: Path) -> Path:
    """A `python` that answers the two exposure commands the launcher must run, recording every
    call, and passes everything else to the real interpreter."""
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    python = shim / "python"
    python.unlink(missing_ok=True)
    python.write_text("\n".join([
        "#!/bin/sh",
        f'printf \'%s\\n\' "$*" >> "{tmp_path}/python.log"',
        'case "$*" in',
        '  *"-m tadabur.exposure tlog-training-disjoint"*) exit "${DISJOINT_EXIT:-0}" ;;',
        '  *"-m tadabur.exposure training-stream"*) exit "${STREAM_EXIT:-0}" ;;',
        "esac",
        f'exec "{sys.executable}" "$@"',
        "",
    ]), encoding="utf-8")
    python.chmod(0o755)
    return shim


def _verify(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess:
    _python_shim(tmp_path)
    return _run(_env(tmp_path, INDEX=str(tmp_path / "index.jsonl"), **overrides), "verify")


def _python_calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "python.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def test_the_data_assertion_checks_the_candidates_shard_use_and_the_tlog_rows(tmp_path):
    done = _verify(tmp_path)
    assert done.returncode == 0, _text(done)
    calls = _python_calls(tmp_path)
    stream = [call for call in calls if "-m tadabur.exposure training-stream" in call]
    assert len(stream) == 1
    assert "--use consonant2.training" in stream[0]
    assert "--membership exact" in stream[0]
    assert "--check" in stream[0]
    assert sum("-m tadabur.exposure tlog-training-disjoint" in call for call in calls) == 1
    assert not any("check_disjoint" in call for call in calls)
    from tadabur.shard_reader import parse_shard_spec

    spec = (tmp_path / "logs" / "spec.txt").read_text(encoding="utf-8").strip()
    assert parse_shard_spec(spec) == _shards()


@pytest.mark.parametrize("failing", ["DISJOINT_EXIT", "STREAM_EXIT"])
def test_the_data_assertion_stops_when_either_check_fails_and_writes_no_spec(tmp_path, failing):
    """`chain` reaches the assertion as `assert_ready || return 1`, where bash disables errexit,
    so each check has to carry its own `|| return 1`."""
    done = _verify(tmp_path, **{failing: "7"})
    assert done.returncode != 0
    assert not (tmp_path / "logs" / "spec.txt").exists()


def test_the_script_never_calls_the_check_that_refuses_tlog():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "tlog-training-disjoint" in text
    assert "check_disjoint" not in text


def test_the_script_runs_the_pre_registered_recipe_from_the_178_weights():
    text = SCRIPT.read_text(encoding="utf-8")
    for literal in (
        "tlog=24,tadabur=8", "--init-weights ema", f"INIT_SHA={INIT_SHA}",
        "TEACHER_REVISION=01a1ef9fbe40d144ef845101e89ff924aed3fef5", "SEED=2026", "STEPS=10000",
        "SMOKE_STEPS=700", "--batch-size 32", "--learning-rate 5e-5", "--warmup-steps 500",
        "--cooldown-steps 3000", "--ema-decay 0.999", "--save-every 2000", "--num-workers 1",
        "BUDGET=${BUDGET:-21600}", "/root/scratch/issue-178/repo/tools/runs/consonant1_candidate/"
        "checkpoint.pt", "--tlog-schedule", "--frame-manifest",
    ):
        assert literal in text, literal


# --- the smoke gate ---------------------------------------------------------------------------


WARM = "[init] warm-started from /box/checkpoint.pt (its step 10000, weights=ema); fresh optimiser"


def _record(step: int, rate: float, **overrides) -> dict:
    record = {"step": step, "elapsed_s": round(step / rate, 3), "total": 0.4, "kl_tlog": 0.41,
              "kl_tadabur": 0.39, "batch_tlog": 24, "batch_tadabur": 8}
    record.update(overrides)
    return record


def _gate(tmp_path: Path, *, rate: float = 0.8, mutate=None, warm: str = WARM,
          shards=(3, 8, 8, 5), splits=("clean", "unclean")) -> subprocess.CompletedProcess:
    records = [_record(step, rate) for step in range(25, 301, 25)]
    if mutate is not None:
        mutate(records)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("".join(json.dumps(record) + "\n" for record in records)
                       + json.dumps({"step": 150, "split": "val", "overall_agreement": 0.5})
                       + "\n", encoding="utf-8")
    log = tmp_path / "smoke.log"
    log.write_text("[setup] preset h448\n" + (warm + "\n" if warm else ""), encoding="utf-8")
    on_disk = tmp_path / "shards.log"
    on_disk.write_text("".join(f"2026-10-29T00:00:{index:02d}Z {count}\n"
                               for index, count in enumerate(shards)), encoding="utf-8")
    consumed = tmp_path / "tlog_consumed.jsonl.gz"
    with gzip.open(consumed, "wt", encoding="utf-8") as handle:
        for index, split in enumerate(splits * 3):
            handle.write(json.dumps({"split": split, "shard_file": f"{split}-0000{index}",
                                     "row": index}) + "\n")
    return _run(_env(tmp_path), "gate", str(metrics), str(log), str(on_disk), str(consumed))


def test_the_smoke_gate_accepts_a_smoke_that_meets_every_criterion(tmp_path):
    done = _gate(tmp_path)
    assert done.returncode == 0, _text(done)


def test_the_smoke_gate_refuses_a_slow_smoke(tmp_path):
    assert _gate(tmp_path, rate=0.5).returncode != 0


@pytest.mark.parametrize("change", [
    {"kl_tlog": float("nan")}, {"kl_tadabur": float("inf")}, {"kl_tlog": None},
])
def test_the_smoke_gate_refuses_a_source_whose_kl_is_not_finite(tmp_path, change):
    assert _gate(tmp_path, mutate=lambda records: records[3].update(change)).returncode != 0


def test_the_smoke_gate_refuses_a_log_that_never_reports_a_sources_kl(tmp_path):
    assert _gate(tmp_path, mutate=lambda records: [r.pop("kl_tadabur") for r in records]
                 ).returncode != 0


@pytest.mark.parametrize("counts", [(23, 9), (32, 0), (0, 32), (24, 9)])
def test_the_smoke_gate_refuses_any_batch_that_is_not_24_and_8(tmp_path, counts):
    change = {"batch_tlog": counts[0], "batch_tadabur": counts[1]}
    assert _gate(tmp_path, mutate=lambda records: records[5].update(change)).returncode != 0


def test_the_smoke_gate_refuses_a_log_that_never_counts_its_batches(tmp_path):
    assert _gate(tmp_path, mutate=lambda records: [r.pop("batch_tlog") for r in records]
                 ).returncode != 0


def test_the_smoke_gate_refuses_a_smoke_that_consumed_only_one_split(tmp_path):
    assert _gate(tmp_path, splits=("clean",)).returncode != 0
    assert _gate(tmp_path, splits=("unclean",)).returncode != 0


def test_the_smoke_gate_takes_the_most_tlog_shards_ever_on_disk(tmp_path):
    assert _gate(tmp_path, shards=(1, 8, 2)).returncode == 0
    assert _gate(tmp_path, shards=(1, 2, 9, 2)).returncode != 0


def test_the_smoke_gate_refuses_a_missing_shard_sampler_log(tmp_path):
    assert _gate(tmp_path, shards=()).returncode != 0


@pytest.mark.parametrize("warm", [
    "", "[init] warm-started from /box/checkpoint.pt (its step 10000, weights=live); fresh",
])
def test_the_smoke_gate_refuses_a_warm_start_that_is_not_the_averaged_weights(tmp_path, warm):
    assert _gate(tmp_path, warm=warm).returncode != 0


# --- the monitoring guard ---------------------------------------------------------------------


def _monitor(tmp_path: Path, completions: int) -> subprocess.CompletedProcess:
    from training.consumed_prefix import shard_permutation

    log = tmp_path / "train.log"
    log.write_text("".join(f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n"
                           for shard in shard_permutation(_shards(), SEED)[:completions]),
                   encoding="utf-8")
    return _run(_fresh_shell_env(tmp_path), "monitor", str(log))


def test_the_monitoring_guard_accepts_a_run_that_stopped_short_of_the_monitor_shard(tmp_path):
    done = _monitor(tmp_path, 16)  # 8 Tadabur windows a step: ~16 of the 211 shards in 10k steps
    assert done.returncode == 0, _text(done)


def test_the_monitoring_guard_refuses_a_run_that_entered_the_monitor_shard(tmp_path):
    done = _monitor(tmp_path, 210)
    assert done.returncode != 0


# --- the freeze check and the launch ------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path: Path, committed: tuple[str, ...] = FROZEN) -> Path:
    """A checkout whose HEAD holds ``committed`` and nothing else but the launcher."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    for name in ("README.md", *committed):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "freeze")
    return repo


def _freeze_check(tmp_path: Path, repo: Path) -> subprocess.CompletedProcess:
    return _run(_env(tmp_path, FREEZE_RECORD=FREEZE_RECORD), "freeze-check", cwd=repo)


def test_the_freeze_check_passes_when_every_artefact_is_committed_in_head(tmp_path):
    done = _freeze_check(tmp_path, _repo(tmp_path))
    assert done.returncode == 0, _text(done)


@pytest.mark.parametrize("missing", FROZEN)
def test_the_freeze_check_refuses_when_an_artefact_is_not_in_head(tmp_path, missing):
    repo = _repo(tmp_path, tuple(name for name in FROZEN if name != missing))
    done = _freeze_check(tmp_path, repo)
    assert done.returncode != 0
    assert missing in _text(done)


def test_the_freeze_check_refuses_an_artefact_that_is_on_disk_but_not_committed(tmp_path):
    prereg = FROZEN[0]
    repo = _repo(tmp_path, FROZEN[1:])
    (repo / prereg).parent.mkdir(parents=True, exist_ok=True)
    (repo / prereg).write_text("a draft\n", encoding="utf-8")
    done = _freeze_check(tmp_path, repo)
    assert done.returncode != 0
    assert prereg in _text(done)


def _launch(tmp_path: Path, repo: Path, *args: str, lock_held: bool = False,
            checkpoint: str = "absent") -> subprocess.CompletedProcess:
    """`launch` against a stand-in box: every ssh is answered by a shim and logged.

    ``checkpoint`` is the box's answer to the checkpoint probe: ``absent`` or ``exists``; the
    ``fail`` and ``garbage`` states model a dropped connection (exit 255, no answer) and an
    unparseable answer."""
    probe = {
        "absent": "printf 'absent\\n'",
        "exists": "printf 'exists\\n'",
        "fail": "exit 255",
        "garbage": "printf 'ok\\n'",
    }[checkpoint]
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    ssh = shim / "ssh"
    ssh.write_text("\n".join([
        "#!/bin/sh",
        f'printf \'%s\\n\' "$*" >> "{tmp_path}/ssh.log"',
        'case "$*" in',
        '  *"tar -x"*) cat > /dev/null ;;',
        f'  *"checkpoint.pt"*) {probe} ;;',
        f'  *"flock -n"*) [ "{int(lock_held)}" = 1 ] && exit 1 ;;',
        f'  *"df -B1"*) echo {400 * 1024 ** 3} ;;',
        "esac",
        "exit 0",
        "",
    ]), encoding="utf-8")
    ssh.chmod(0o755)
    return _run(_env(tmp_path, HOST="box", FREEZE_RECORD=FREEZE_RECORD), "launch", *args, cwd=repo)


def _ssh_log(tmp_path: Path) -> str:
    log = tmp_path / "ssh.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def test_a_full_launch_without_the_preregistration_in_head_never_touches_the_box(tmp_path):
    repo = _repo(tmp_path, FROZEN[1:])
    done = _launch(tmp_path, repo)
    assert done.returncode != 0
    assert "consonant-round2-preregistration.md" in _text(done)
    assert _ssh_log(tmp_path) == ""


def test_a_full_launch_with_the_freeze_committed_stages_head_and_starts_the_run(tmp_path):
    done = _launch(tmp_path, _repo(tmp_path))
    assert done.returncode == 0, _text(done)
    issued = _ssh_log(tmp_path)
    assert "git clone" in issued  # a checkout, so the tlog overlay commit is in its history
    assert "tar -x" not in issued
    started = [line for line in issued.splitlines() if "setsid" in line]
    assert started and "SMOKE_ONLY" not in started[-1]


def test_a_smoke_launch_does_not_need_the_freeze(tmp_path):
    done = _launch(tmp_path, _repo(tmp_path, ()), "--smoke")
    assert done.returncode == 0, _text(done)
    started = [line for line in _ssh_log(tmp_path).splitlines() if "setsid" in line]
    assert started and "SMOKE_ONLY=1" in started[-1]


@pytest.mark.parametrize("args", [(), ("--smoke",)])
def test_a_held_gpu_lock_refuses_the_launch_before_anything_is_staged(tmp_path, args):
    done = _launch(tmp_path, _repo(tmp_path), *args, lock_held=True)
    assert done.returncode == 3
    assert "gpu.lock is held" in _text(done)
    issued = _ssh_log(tmp_path)
    assert "tar -x" not in issued
    assert "git clone" not in issued
    assert "mkdir" not in issued


def test_a_previous_attempts_checkpoint_refuses_the_launch_and_is_kept(tmp_path):
    done = _launch(tmp_path, _repo(tmp_path), checkpoint="exists")
    assert done.returncode == 4
    assert "previous attempt's checkpoint" in _text(done)
    issued = _ssh_log(tmp_path)
    assert "rm -rf" not in issued
    assert "git clone" not in issued


@pytest.mark.parametrize("answer", ["fail", "garbage"])
def test_an_unclear_checkpoint_probe_neither_removes_nor_stages_the_run(tmp_path, answer):
    """R1.1: a dropped SSH connection (or an answer that is neither `exists` nor `absent`)
    is not an absent checkpoint, so it must not authorize deleting the run directory."""
    done = _launch(tmp_path, _repo(tmp_path), "--smoke", checkpoint=answer)
    assert done.returncode != 0
    assert "nothing was removed" in _text(done)
    issued = _ssh_log(tmp_path)
    assert "rm -rf" not in issued
    assert "git clone" not in issued
    assert "mkdir" in issued  # only the logs directory, before the probe


# --- the smoke's evidence is tied to the launch commit ---------------------------------------


def test_a_smoke_is_evidence_only_for_the_launch_commit_that_ran_it(tmp_path):
    """A code change between `launch --smoke` and the full launch must smoke again: the gate
    result belongs to the archived tree, not to the arm's name."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "smoke.done").write_text("a" * 40 + "\n", encoding="utf-8")
    assert _run(_env(tmp_path), "smoke-valid", "a" * 40).returncode == 0
    assert _run(_env(tmp_path), "smoke-valid", "b" * 40).returncode != 0


@pytest.mark.parametrize("done", [None, "", "\n"])
def test_an_absent_or_empty_smoke_done_is_not_evidence(tmp_path, done):
    logs = tmp_path / "logs"
    logs.mkdir()
    if done is not None:
        (logs / "smoke.done").write_text(done, encoding="utf-8")
    assert _run(_env(tmp_path), "smoke-valid", "a" * 40).returncode != 0


# --- the staged tree must be a checkout, not an archive --------------------------------------

_OVERLAY_CHECK = (
    "from tadabur import tlog_pool\n"
    "from tadabur.exposure import TLOG_TRAINING, load_registry\n"
    "use = load_registry().row_sets[TLOG_TRAINING]\n"
    "tlog_pool.require_overlay_at(use.overlay_commit)\n"
)


def _require_overlay(tree: Path) -> subprocess.CompletedProcess:
    """The tlog reader's own staging check, in ``tree``: its overlay commit must be checked
    out of the tree's own history (``load_training_rows`` calls exactly this)."""
    return subprocess.run([sys.executable, "-c", _OVERLAY_CHECK], cwd=tree / "tools",
                          capture_output=True, text=True)


def test_the_staged_tree_is_a_checkout_the_tlog_reader_can_verify(tmp_path):
    """S1.1: `launch` stages a bundle clone, not a `git archive` tree. The tlog reader
    re-materialises `tlog.training` by checking the overlay commit out of the staged tree's
    history, so an archive tree refuses in the data assertion and again in the mixed stream."""
    repo = TOOLS.parent
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    bundle = tmp_path / "repo.bundle"
    subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle), branch],
                   check=True, capture_output=True)
    staged = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", "-b", branch, str(bundle), str(staged)],
                   check=True, capture_output=True)
    assert subprocess.run(["git", "-C", str(staged), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip() == head
    checks_out = _require_overlay(staged)
    assert checks_out.returncode == 0, checks_out.stdout + checks_out.stderr

    archive = tmp_path / "archive"
    archive.mkdir()
    subprocess.run(f"git -C '{repo}' archive HEAD | tar -x -C '{archive}'", shell=True,
                   check=True)
    old = _require_overlay(archive)
    assert old.returncode != 0
    assert "no overlay commit" in old.stdout + old.stderr

    text = SCRIPT.read_text(encoding="utf-8")
    assert "bundle create" in text and "git clone" in text
    assert 'git -C "$repo_root" archive' not in text

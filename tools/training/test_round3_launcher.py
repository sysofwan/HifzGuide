"""Acceptance tests for the round-3 candidate's launcher, ``scripts/issue-252-train.sh`` (#252),
written before the script.

Locked: later turns may add tests but never weaken or rewrite these.

The script is ``issue-228-train.sh``'s sibling (copied from it, which stays untouched): the same
mixed 24 + 8 recipe, with round 3's values. Like ``test_round2_launcher`` these run the real
script's subcommands in a temporary ``ISSUE_ROOT`` without a GPU. What differs from round 2:

* the warm start is round 2's candidate checkpoint, averaged weights
  (``INIT_SHA=89352371...ea58``, ``--init-weights ema``), under ``/root/scratch/issue-228``;
* ``SEED=2030`` over the 194 shards of ``consonant3.training`` (``EXPOSURE_USE`` and
  ``FRAME_USE`` are both that use), so ``frame`` prints the last shard of the seed-2030 order as the
  monitor shard -- not round 2's shard 72, which round 3 streams;
* the tlog walk starts after round 2's rows: ``--tlog-after`` names the committed
  ``tools/tadabur/tlog/consumed/consonant2.jsonl.gz`` (staged under ``$ROOT/repo``, overridable
  through ``$TLOG_AFTER``), whose SHA-256 the launcher pins and ``ready`` asserts;
* ``FROZEN`` is the round-3 pre-registration, ``training_schedule.json``,
  ``training_prefix.round3.json`` and the consumed-keys file, plus the freeze record
  (``$FREEZE_RECORD``, a repo path); the full run also refuses unless that record, as committed
  in HEAD, has ``power_rule.satisfied == true``. A smoke needs none of it;
* ``OUT=$ROOT/runs/consonant3_candidate``, ``ROOT`` defaults to ``/root/scratch/issue-252``;
* ``gate`` keeps its arguments and gains two criteria, read against ``$TLOG_AFTER``: the smoke's
  first consumed tlog key is the frozen walk's row 73,991 (the first row after round 2's
  73,990), and none of its consumed keys is in round 2's committed set. Its stdout reports both
  (a line naming ``73,991``).

``launch``, ``check``, ``freeze-check``, ``verify``, ``manifest``, ``monitor``, ``smoke-valid``
and ``ready`` keep round 2's contract (``test_round2_launcher``): ``--resume`` exits 2, a held
``gpu.lock`` exits 3, a previous attempt's checkpoint exits 4, and HEAD is staged as a bundle
clone.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from itertools import islice
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
REPO = TOOLS.parent
SCRIPT = REPO / "scripts" / "issue-252-train.sh"
ROUND2_SCRIPT = REPO / "scripts" / "issue-228-train.sh"
CONSUMED = TOOLS / "tadabur" / "tlog" / "consumed" / "consonant2.jsonl.gz"
CONSUMED_SHA = "5e95c3c06f498768323781fc04b4394093233c2a7717ae193f80feda87929238"
INIT_SHA = "893523710186cb16854dd62a258331644e2284abbee8b2cb54e8ab9d32dbea58"
ROUND2_INIT_SHA = "057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2"
POOL_SHA = "d2978d0ff3fcc14b08b21b9360d0933202e1a9f7b92fae5de9376d6ba2b3fd43"
SEED = 2030
ROUND2_CONSUMED_ROWS = 73_990
FREEZE_RECORD = "tools/tadabur/consonant3_readout/freeze.json"
FROZEN = ("docs/consonant-round3-preregistration.md", FREEZE_RECORD,
          "tools/tadabur/tlog/training_schedule.json",
          "tools/tadabur/tlog/training_prefix.round3.json",
          "tools/tadabur/tlog/consumed/consonant2.jsonl.gz")
POWER_MET = json.dumps({"power_rule": {"satisfied": True}})


def _shards() -> list[int]:
    from tadabur.exposure import CONSONANT3_TRAINING, load_registry

    return list(load_registry().shard_uses[CONSONANT3_TRAINING].shards)


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
    # A missing script exits non-zero like a refusal would; fail loudly so no refusal test can
    # pass for the want of the launcher.
    assert SCRIPT.is_file(), f"{SCRIPT} does not exist"
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL)


def _text(done: subprocess.CompletedProcess) -> str:
    return done.stdout + done.stderr


# --- the frame ------------------------------------------------------------------------------


def test_the_frame_is_the_194_shards_with_its_monitor_shard_last_in_the_seed_2030_order(tmp_path):
    from tadabur.shard_reader import parse_shard_spec
    from training.consumed_prefix import monitoring_shard

    done = _run(_env(tmp_path), "frame")
    assert done.returncode == 0, _text(done)
    printed = dict(line.split("=", 1) for line in done.stdout.splitlines() if "=" in line)
    assert parse_shard_spec(printed["stream"]) == _shards()
    assert len(_shards()) == 194
    monitor, position = monitoring_shard(_shards(), SEED)
    assert (int(printed["monitor"]), int(printed["position"])) == (monitor, position)
    assert int(printed["monitor"]) != 72  # round 2's monitor shard is round 3's training audio


def test_the_manifest_records_the_round_3_frame(tmp_path):
    from tadabur.shard_reader import shard_spec

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "spec.txt").write_text(shard_spec(_shards()) + "\n", encoding="utf-8")
    done = _run(_env(tmp_path), "manifest")
    assert done.returncode == 0, _text(done)
    manifest = json.loads((tmp_path / "runs" / "consonant3_candidate" / "frame_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["shards"] == _shards()
    assert (manifest["shard_order_seed"], manifest["window_shuffle_seed"]) == (SEED, SEED + 1)
    assert manifest["exposure_use"] == "consonant3.training"
    assert manifest["augmentation"]["version"] == "aug94-mixture-v2"


# --- the warm start and the tlog offset --------------------------------------------------------


def _sha_shim(tmp_path: Path, **known: Path) -> None:
    """A `sha256sum` that reports the stated digest for the named files and the real one for any
    other, so `ready` can be driven without the box's multi-GiB checkpoint."""
    table = {str(path): digest for path, digest in known.items()}
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    script = shim / "sha256sum"
    script.write_text("\n".join([
        f"#!{sys.executable}",
        "import hashlib, sys",
        f"table = {table!r}",
        "for name in sys.argv[1:]:",
        "    digest = table.get(name) or hashlib.sha256(open(name, 'rb').read()).hexdigest()",
        "    print(f'{digest}  {name}')",
        "",
    ]), encoding="utf-8")
    script.chmod(0o755)


def _stand_in_files(tmp_path: Path) -> tuple[Path, Path]:
    init = tmp_path / "checkpoint.pt"
    init.write_bytes(b"the averaged weights")
    pool = tmp_path / "noise_pool.npz"
    pool.write_bytes(b"the noise pool")
    return init, pool


def _ready(tmp_path: Path, *, init_digest: str = INIT_SHA, **overrides: str):
    _python_shim(tmp_path)
    init, pool = _stand_in_files(tmp_path)
    _sha_shim(tmp_path, **{str(init): init_digest, str(pool): POOL_SHA})
    env = _env(tmp_path, INIT=str(init), POOL=str(pool), INDEX=str(tmp_path / "index.jsonl"),
               **overrides)
    env.setdefault("TLOG_AFTER", str(CONSUMED))
    return _run(env, "ready")


def test_a_warm_start_that_is_not_round_2s_candidate_refuses_the_launch(tmp_path):
    done = _ready(tmp_path, init_digest=ROUND2_INIT_SHA)  # #178's checkpoint, round 2's warm start
    assert done.returncode != 0
    assert f"init checkpoint is {ROUND2_INIT_SHA}" in _text(done)
    assert INIT_SHA in _text(done)


def test_the_right_warm_start_and_consumed_keys_pass_the_readiness_check(tmp_path):
    done = _ready(tmp_path)
    assert done.returncode == 0, _text(done)


def test_a_consumed_keys_file_that_is_not_round_2s_committed_log_refuses_the_launch(tmp_path):
    other = tmp_path / "consonant2.jsonl.gz"
    with gzip.open(other, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"split": "clean", "shard_file": "clean-00000-of-00411",
                                 "row": 0}) + "\n")
    done = _ready(tmp_path, TLOG_AFTER=str(other))
    assert done.returncode != 0
    assert CONSUMED_SHA in _text(done)


def test_the_committed_consumed_keys_are_the_file_the_launcher_pins():
    assert hashlib.sha256(CONSUMED.read_bytes()).hexdigest() == CONSUMED_SHA


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


def test_the_data_assertion_checks_the_round_3_shard_use_and_the_tlog_rows(tmp_path):
    done = _verify(tmp_path)
    assert done.returncode == 0, _text(done)
    calls = _python_calls(tmp_path)
    stream = [call for call in calls if "-m tadabur.exposure training-stream" in call]
    assert len(stream) == 1
    assert "--use consonant3.training" in stream[0]
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


def test_the_script_runs_the_round_2_recipe_from_round_2s_averaged_weights():
    text = SCRIPT.read_text(encoding="utf-8")
    for literal in (
        "tlog=24,tadabur=8", "--init-weights ema", f"INIT_SHA={INIT_SHA}",
        "TEACHER_REVISION=01a1ef9fbe40d144ef845101e89ff924aed3fef5", f"SEED={SEED}", "STEPS=10000",
        "SMOKE_STEPS=700", "--batch-size 32", "--learning-rate 5e-5", "--warmup-steps 500",
        "--cooldown-steps 3000", "--ema-decay 0.999", "--save-every 2000", "--num-workers 1",
        "BUDGET=${BUDGET:-21600}", "MIN_FREE_GB=${MIN_FREE_GB:-16}",
        "/root/scratch/issue-228/runs/consonant2_candidate/checkpoint.pt",
        "--tlog-schedule", "--frame-manifest", "--augment", "--noise-pool",
        "EXPOSURE_USE=consonant3.training", "FRAME_USE=consonant3.training",
        "ROOT=${ISSUE_ROOT:-/root/scratch/issue-252}", "$ROOT/runs/consonant3_candidate",
    ):
        assert literal in text, literal
    assert ROUND2_INIT_SHA not in text


def test_the_script_starts_the_tlog_walk_after_round_2s_committed_keys():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--tlog-after" in text
    assert "tools/tadabur/tlog/consumed/consonant2.jsonl.gz" in text
    assert CONSUMED_SHA in text


def test_the_script_is_a_sibling_not_a_second_name_for_round_2s():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "issue-228-train.sh" not in text  # it never invokes, or tells a reader to run, round 2's
    assert "consonant2.control.training" not in text
    assert "CONSONANT2" not in text
    assert ROUND2_SCRIPT.exists()


# --- the smoke gate ---------------------------------------------------------------------------


WARM = "[init] warm-started from /box/checkpoint.pt (its step 10000, weights=ema); fresh optimiser"
SMOKE_ROWS = 3_000  # enough walk rows after round 2's prefix to meet both splits


@pytest.fixture(scope="module")
def walk() -> list[tuple[str, str, int]]:
    """The frozen schedule's walk keys, round 2's 73,990 and then ``SMOKE_ROWS`` more."""
    from tadabur.tlog_pool import load_training_rows, row_key
    from tadabur.tlog_schedule import interleave_order, read_schedule

    rows = islice(interleave_order(load_training_rows(), read_schedule()),
                  ROUND2_CONSUMED_ROWS + SMOKE_ROWS)
    keys = [row_key(row) for row in rows]
    assert len(keys) == ROUND2_CONSUMED_ROWS + SMOKE_ROWS
    return keys


def _record(step: int, rate: float, **overrides) -> dict:
    record = {"step": step, "elapsed_s": round(step / rate, 3), "total": 0.4, "kl_tlog": 0.41,
              "kl_tadabur": 0.39, "batch_tlog": 24, "batch_tadabur": 8}
    record.update(overrides)
    return record


def _gate(tmp_path: Path, keys, *, rate: float = 0.8, mutate=None,
          warm: str = WARM) -> subprocess.CompletedProcess:
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
    on_disk.write_text("".join(f"2026-11-02T00:00:{index:02d}Z {count}\n"
                               for index, count in enumerate((3, 8, 8, 5))), encoding="utf-8")
    consumed = tmp_path / "tlog_consumed.jsonl.gz"
    with gzip.open(consumed, "wt", encoding="utf-8") as handle:
        for split, shard_file, row in keys:
            handle.write(json.dumps({"split": split, "shard_file": shard_file, "row": row}) + "\n")
    env = _env(tmp_path, TLOG_AFTER=str(CONSUMED))
    return _run(env, "gate", str(metrics), str(log), str(on_disk), str(consumed))


def test_the_smoke_gate_accepts_a_smoke_that_starts_at_walk_row_73_991(tmp_path, walk):
    keys = walk[ROUND2_CONSUMED_ROWS:]
    assert {split for split, _, _ in keys} == {"clean", "unclean"}
    done = _gate(tmp_path, keys)
    assert done.returncode == 0, _text(done)
    assert any("73,991" in line for line in done.stdout.splitlines() if "[gate]" in line)


def test_the_first_key_of_the_walk_after_round_2_is_clean_shard_85_row_375(walk):
    """The pin the gate's first line stands on: walk row 73,991 (1-based) is index 73,990."""
    assert walk[ROUND2_CONSUMED_ROWS] == ("clean", "clean-00085-of-00411", 375)


@pytest.mark.parametrize("start", [
    0,                          # a smoke that began at the top of the walk, ignoring --tlog-after
    ROUND2_CONSUMED_ROWS - 1,   # one row early: round 2's last key
    ROUND2_CONSUMED_ROWS + 1,   # one row late
], ids=["from-the-top", "one-row-early", "one-row-late"])
def test_the_smoke_gate_refuses_a_smoke_whose_first_key_is_not_walk_row_73_991(
        tmp_path, walk, start):
    keys = walk[start:start + SMOKE_ROWS]
    assert _gate(tmp_path, keys).returncode != 0


def test_the_smoke_gate_refuses_a_smoke_that_consumed_a_round_2_key_later_on(tmp_path, walk):
    from tadabur.tlog_schedule import read_consumed_keys

    keys = list(walk[ROUND2_CONSUMED_ROWS:])
    keys[1_500] = read_consumed_keys(CONSUMED)[100]  # the first key is right; one other is not
    done = _gate(tmp_path, keys)
    assert done.returncode != 0
    assert "round 2" in _text(done).lower() or "consumed" in _text(done).lower()


def test_the_smoke_gate_keeps_round_2s_other_refusals(tmp_path, walk):
    keys = walk[ROUND2_CONSUMED_ROWS:]
    assert _gate(tmp_path, keys, rate=0.5).returncode != 0
    assert _gate(tmp_path, keys, mutate=lambda records: records[5].update(
        {"batch_tlog": 23, "batch_tadabur": 9})).returncode != 0
    assert _gate(tmp_path, keys, warm="[init] warm-started from /box/checkpoint.pt (its step "
                 "10000, weights=live); fresh").returncode != 0


# --- the monitoring guard ---------------------------------------------------------------------


def _monitor(tmp_path: Path, completions: int) -> subprocess.CompletedProcess:
    from training.consumed_prefix import shard_permutation

    log = tmp_path / "train.log"
    log.write_text("".join(f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n"
                           for shard in shard_permutation(_shards(), SEED)[:completions]),
                   encoding="utf-8")
    return _run(_fresh_shell_env(tmp_path), "monitor", str(log))


def test_the_monitoring_guard_accepts_a_run_that_stopped_short_of_the_monitor_shard(tmp_path):
    done = _monitor(tmp_path, 16)  # 8 Tadabur windows a step: ~16 shards in 10k steps
    assert done.returncode == 0, _text(done)


def test_the_monitoring_guard_refuses_a_run_that_entered_the_monitor_shard(tmp_path):
    done = _monitor(tmp_path, 193)
    assert done.returncode != 0


# --- the freeze check and the launch ------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path: Path, committed: tuple[str, ...] = FROZEN, *,
          record: str = POWER_MET) -> Path:
    """A checkout whose HEAD holds ``committed`` and nothing else but the launcher. The freeze
    record's committed content is ``record``."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    for name in ("README.md", *committed):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record + "\n" if name == FREEZE_RECORD else f"{name}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "freeze")
    return repo


def _freeze_check(tmp_path: Path, repo: Path) -> subprocess.CompletedProcess:
    return _run(_env(tmp_path, FREEZE_RECORD=FREEZE_RECORD), "freeze-check", cwd=repo)


def test_the_freeze_check_passes_when_every_artefact_is_committed_and_the_power_rule_holds(
        tmp_path):
    done = _freeze_check(tmp_path, _repo(tmp_path))
    assert done.returncode == 0, _text(done)
    assert "5 artefacts committed" in done.stdout


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


@pytest.mark.parametrize("record", [
    json.dumps({"power_rule": {"satisfied": False}}),
    json.dumps({"power_rule": {}}),
    json.dumps({"power_rule": {"satisfied": "true"}}),
    json.dumps({"look": "frozen"}),
    "not json at all",
], ids=["unsatisfied", "no-verdict", "truthy-string", "no-power-rule", "not-json"])
def test_the_freeze_check_refuses_a_record_whose_power_rule_is_not_satisfied(tmp_path, record):
    done = _freeze_check(tmp_path, _repo(tmp_path, record=record))
    assert done.returncode != 0
    assert "power_rule" in _text(done)


def test_the_power_rule_is_read_from_head_not_from_the_working_tree(tmp_path):
    repo = _repo(tmp_path, record=json.dumps({"power_rule": {"satisfied": False}}))
    (repo / FREEZE_RECORD).write_text(POWER_MET + "\n", encoding="utf-8")  # uncommitted edit
    done = _freeze_check(tmp_path, repo)
    assert done.returncode != 0
    assert "power_rule" in _text(done)


def _has_consonant3_spec() -> bool:
    from training import aug_readout_spec

    try:
        aug_readout_spec.spec("consonant3")
    except KeyError:
        return False
    return True


@pytest.mark.skipif(not _has_consonant3_spec(),
                    reason="the round-3 spec (#251) is not merged yet; its readout root is "
                           "what the launcher resolves the freeze record through")
def test_the_freeze_check_resolves_the_rounds_record_from_the_consonant3_spec(tmp_path):
    from training.aug_readout_spec import spec
    from training.consonant_readout import freeze_path

    resolved = str(freeze_path(spec("consonant3")).resolve().relative_to(REPO.resolve()))
    assert resolved == FREEZE_RECORD
    repo = _repo(tmp_path)
    done = _run(_env(tmp_path), "freeze-check", cwd=repo)  # no FREEZE_RECORD override
    assert done.returncode == 0, _text(done)


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
    assert "consonant-round3-preregistration.md" in _text(done)
    assert _ssh_log(tmp_path) == ""


def test_a_full_launch_without_the_freeze_record_in_head_never_touches_the_box(tmp_path):
    repo = _repo(tmp_path, tuple(name for name in FROZEN if name != FREEZE_RECORD))
    done = _launch(tmp_path, repo)
    assert done.returncode != 0
    assert FREEZE_RECORD in _text(done)
    assert _ssh_log(tmp_path) == ""


def test_a_full_launch_with_an_unsatisfied_power_rule_never_touches_the_box(tmp_path):
    repo = _repo(tmp_path, record=json.dumps({"power_rule": {"satisfied": False}}))
    done = _launch(tmp_path, repo)
    assert done.returncode != 0
    assert "power_rule" in _text(done)
    assert _ssh_log(tmp_path) == ""


def test_a_full_launch_with_the_freeze_committed_stages_head_and_starts_the_run(tmp_path):
    done = _launch(tmp_path, _repo(tmp_path))
    assert done.returncode == 0, _text(done)
    issued = _ssh_log(tmp_path)
    assert "git clone" in issued  # a checkout, so the tlog overlay commit is in its history
    assert "tar -x" not in issued
    assert "issue-252-train.sh" in issued
    started = [line for line in issued.splitlines() if "setsid" in line]
    assert started and "SMOKE_ONLY" not in started[-1]


def test_a_smoke_launch_needs_neither_the_freeze_nor_the_power_rule(tmp_path):
    done = _launch(tmp_path, _repo(tmp_path, (), record=""), "--smoke")
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


@pytest.mark.parametrize("args", [("--resume",), ("--resume", "--smoke")])
def test_resume_is_refused_and_the_box_is_never_touched(tmp_path, args):
    done = _launch(tmp_path, _repo(tmp_path), *args)
    assert done.returncode == 2
    assert "--resume" in _text(done)
    assert _ssh_log(tmp_path) == ""


def test_a_previous_attempts_checkpoint_refuses_the_launch_and_is_kept(tmp_path):
    done = _launch(tmp_path, _repo(tmp_path), checkpoint="exists")
    assert done.returncode == 4
    assert "previous attempt's checkpoint" in _text(done)
    issued = _ssh_log(tmp_path)
    assert "rm -rf" not in issued
    assert "git clone" not in issued


@pytest.mark.parametrize("answer", ["fail", "garbage"])
def test_an_unclear_checkpoint_probe_neither_removes_nor_stages_the_run(tmp_path, answer):
    done = _launch(tmp_path, _repo(tmp_path), "--smoke", checkpoint=answer)
    assert done.returncode != 0
    assert "nothing was removed" in _text(done)
    issued = _ssh_log(tmp_path)
    assert "rm -rf" not in issued
    assert "git clone" not in issued


# --- the on-box execution boundary ------------------------------------------------------------


def _staged_root(tmp_path: Path) -> Path:
    """The box's ``$ISSUE_ROOT`` after a permitted pre-freeze smoke: a checkout, ``launch.txt``
    and a ``smoke.done`` that belongs to that commit."""
    logs = tmp_path / "logs"
    logs.mkdir()
    commit = "a" * 40
    (logs / "launch.txt").write_text(f"{commit} 2026-11-02T00:00:00Z\n", encoding="utf-8")
    (logs / "smoke.done").write_text(commit + "\n", encoding="utf-8")
    (tmp_path / "repo" / "tools").mkdir(parents=True)
    return logs


def test_the_on_box_run_refuses_the_full_arm_without_the_committed_freeze(tmp_path):
    """`run` is the documented on-box restart path: a current smoke behind it must not let a bare
    `run` spend the round's 10,000-step attempt on a checkout whose HEAD lacks the freeze."""
    logs = _staged_root(tmp_path)
    done = _run(_fresh_shell_env(tmp_path), "run")
    assert done.returncode != 0
    assert "refusing the full run" in (logs / "chain.out").read_text(encoding="utf-8")
    assert "full start" not in (logs / "events").read_text(encoding="utf-8")
    assert (logs / "train.exit").read_text(encoding="utf-8").strip() == "1"
    assert "chain exit" in (logs / "train.done").read_text(encoding="utf-8")


def test_a_smoke_only_run_is_exempt_from_the_freeze(tmp_path):
    logs = _staged_root(tmp_path)
    done = _run(_fresh_shell_env(tmp_path, SMOKE_ONLY="1"), "run")
    assert done.returncode != 0  # the warm start is absent here, so it stops there
    chain = (logs / "chain.out").read_text(encoding="utf-8")
    assert "refusing the full run" not in chain
    assert "[assert] warm start" in chain


# --- the smoke's evidence is tied to the launch commit ---------------------------------------


def test_a_smoke_is_evidence_only_for_the_launch_commit_that_ran_it(tmp_path):
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


def test_the_launch_stages_a_bundle_clone_not_an_archive():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "bundle create" in text and "git clone" in text
    assert 'git -C "$repo_root" archive' not in text

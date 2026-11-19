"""The mining launcher's phase C (#263), run dry: ``scripts/issue-263-mining-decode.sh``.

Phases A and B are ``test_tlog_mining_launcher.py`` and ``test_tlog_mining_phase_b_launcher.py``;
this file adds phase C's: the round-3 chosen weights over the mined pool.

* The round-3 readout directory is ``ROUND3_DIR`` (default ``tools/tadabur/consonant3_readout``;
  relative to the repository like ``ROUND_DIR``). Phase C is accepted by ``plan``, ``launch`` and
  ``resume`` only when **all** of these hold at HEAD, and refused before anything is staged
  (non-zero exit, nothing on stdout, nothing sent over ``ssh``) otherwise:

  - ``$ROUND3_DIR/choice.json`` is committed and unchanged (``not committed`` on stderr), names
    candidate ``r3_candidate_ema`` (``r3_candidate_ema`` on stderr) and its checkpoint SHA-256 is
    ``training.aug_readout_spec.R3_CANDIDATE_SHA256`` (``checkpoint`` on stderr). The pin is read
    from the Python constant: the script text never contains the hash.
  - ``$ROUND3_DIR/verdict.json`` is committed and unchanged and its ``verdict`` is ``pass``
    (``verdict`` on stderr).
  - ``docs/tlog-mining-amendment.md`` is the pinned amendment (``amendment`` and ``SHA-256``).
  - The two confirm claims and phase A/B's ``choice.json`` are committed, as for every phase.

* ``plan C`` prints one decode line per mined tranche for the round-3 weights alone, and the
  merge, and writes nothing::

      decode confirm.1 r3_candidate_ema at 2 processes
      decode confirm.2 r3_candidate_ema at 2 processes
      merge mined

* ``launch C`` / ``resume C``: the phase A staging and a detached ``run`` whose command line names
  phase ``C``, logging to ``<ISSUE_ROOT>/C/logs/run.out``. Phase C's parts, clips, shards, logs and
  out directories are all under ``<ISSUE_ROOT>/C``: no part, done or exit file of A, B or C is
  ever read as another's, and a live loop of any phase blocks a restage.
* ``status C`` reads ``<ISSUE_ROOT>/C/logs`` only.
* ``fetch C`` refuses unless the box holds a complete merge in ``<ISSUE_ROOT>/C/out`` (``run.exit``
  0 and all of ``tlog.mined.r3_candidate_ema.jsonl.gz``, ``tlog.gating.r3_candidate_ema.jsonl.gz``
  and ``meta.r3_candidate_ema.json``), then copies the out directory into
  ``tools/tadabur/tlog_mining/decodes`` (``FETCH_DIR`` overrides) **without touching** phase B's
  ``meta.json`` or ``tlog.mined.{base,c95,r2_candidate_ema}.jsonl.gz``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from tadabur.test_tlog_mining_launcher import (PLAN, SCRIPT, _box, _run, _ssh_log)
from tadabur.test_tlog_mining_phase_b_launcher import _nohup_line, _with_amendment
from training import round2_verdict_fixtures as rv
from training.aug_readout_spec import R3_CANDIDATE_SHA256
from training.test_consonant2_confirm_launcher import _env, _mock

REPO = Path(__file__).resolve().parents[2]
ROUND3 = REPO / "tools" / "tadabur" / "consonant3_readout"
MODEL = "r3_candidate_ema"
PLAN_C = [f"decode confirm.1 {MODEL} at 2 processes", f"decode confirm.2 {MODEL} at 2 processes",
          "merge mined"]
#: Phase C's merged files, as ``fetch`` checks them: the mined rows, the gating rows, its own meta.
PHASE_C_NAMES = [f"tlog.mined.{MODEL}.jsonl.gz", f"tlog.gating.{MODEL}.jsonl.gz",
                 f"meta.{MODEL}.json"]
PHASE_B_FILES = ["meta.json", *(f"tlog.mined.{model}.jsonl.gz"
                                for model in ("base", "c95", "r2_candidate_ema"))]


def _choice(edit=None) -> str:
    document = json.loads((ROUND3 / "choice.json").read_text(encoding="utf-8"))
    if edit is not None:
        edit(document)
    return json.dumps(document)


def _phase_c_round(tmp_path: Path, *, choice: str | None = "", verdict: str | None = "pass",
                   amendment: bytes | None = None) -> rv.Round:
    """The phase B round (claims, choice, amendment) plus round 3's committed choice and verdict.

    ``choice=None`` / ``verdict=None`` leave that file out; ``""`` is the real choice.
    """
    round_ = _with_amendment(tmp_path, amendment)
    directory = round_.repo / "round3"
    if choice is not None:
        round_.write_commit(directory / "choice.json", choice or _choice(), message="r3 choice")
    if verdict is not None:
        round_.write_commit(directory / "verdict.json", json.dumps({"verdict": verdict}),
                            message="r3 verdict")
    return round_


def _c(command: str, round_: rv.Round, tmp_path: Path, *args: str, **extra: str):
    return _run(command, round_, tmp_path, "C", *args, ROUND3_DIR="round3", **extra)


def test_the_plan_lists_phase_c_and_writes_nothing(tmp_path):
    round_ = _phase_c_round(tmp_path)
    done = _c("plan", round_, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == PLAN_C
    assert not (tmp_path / "ssh.log").exists()


def test_a_phase_left_out_is_still_phase_a_with_round_3_pinned(tmp_path):
    round_ = _phase_c_round(tmp_path)
    done = _run("plan", round_, tmp_path, ROUND3_DIR="round3")
    assert done.stdout.splitlines() == PLAN


def test_the_script_does_not_carry_the_checkpoint_pin():
    """One place for the pin: ``training.aug_readout_spec.R3_CANDIDATE_SHA256``."""
    assert R3_CANDIDATE_SHA256 not in SCRIPT.read_text(encoding="utf-8")


#: Each refusal: how the round is built, and what stderr must say.
REFUSALS = {
    "choice-missing": (dict(choice=None), r"not committed"),
    "verdict-missing": (dict(verdict=None), r"not committed"),
    "other-candidate": (dict(choice=_choice(lambda c: c["models"].update(
        candidate="r3_candidate_live"))), r"candidate.*r3_candidate_ema"),
    "other-checkpoint": (dict(choice=_choice(lambda c: c["identities"][MODEL].update(
        checkpoint_sha256="0" * 64))), r"checkpoint"),
    "verdict-fail": (dict(verdict="fail"), r"verdict.*pass"),
    "amendment-other-bytes": (dict(amendment=b"not the pinned amendment\n"), r"amendment"),
}


@pytest.mark.parametrize("command", ["plan", "launch", "resume"])
@pytest.mark.parametrize("name", REFUSALS)
def test_phase_c_is_refused_before_staging_unless_every_pin_holds(tmp_path, command, name):
    build, pattern = REFUSALS[name]
    round_ = _phase_c_round(tmp_path, **build)
    done = _c(command, round_, tmp_path, **_box(tmp_path))
    assert done.returncode != 0 and done.stdout == ""
    assert re.search(pattern, done.stderr, re.IGNORECASE | re.DOTALL), done.stderr
    assert not (tmp_path / "ssh.log").exists()


@pytest.mark.parametrize("name", ["choice.json", "verdict.json"])
def test_phase_c_refuses_a_round_3_file_that_changed_since_its_commit(tmp_path, name):
    round_ = _phase_c_round(tmp_path)
    (round_.repo / "round3" / name).write_text(
        _choice() if name == "choice.json" else '{"verdict": "pass", "edited": true}',
        encoding="utf-8")
    done = _c("plan", round_, tmp_path)
    assert done.returncode != 0 and done.stdout == ""
    assert "not committed" in done.stderr


@pytest.mark.parametrize("name", ["readout/choice.json", "tranches/confirm.1.claim.json",
                                  "tranches/confirm.2.claim.json"])
def test_phase_c_still_refuses_an_input_that_is_not_committed_at_head(tmp_path, name):
    round_ = _phase_c_round(tmp_path)
    (round_.repo / name).write_text('{"edited": true}\n', encoding="utf-8")
    done = _c("plan", round_, tmp_path)
    assert done.returncode != 0 and done.stdout == ""
    assert "not committed" in done.stderr


def test_the_phase_c_launch_starts_a_detached_run_that_names_phase_c_and_logs_under_c(tmp_path):
    round_ = _phase_c_round(tmp_path)
    env = _box(tmp_path)
    done = _c("launch", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"started: \d+", done.stdout.strip()), done.stdout
    line = _nohup_line(tmp_path)
    assert "setsid" in line and re.search(r"decode\.sh'? run\b[^\n]*\bC\b", line), line
    root = re.escape(env["ISSUE_ROOT"])
    assert re.search(rf">> '?{root}/C/logs/run\.out", line), line
    assert not re.search(rf"{root}/(B/)?logs/run\.out", line)
    assert not re.search(r"\bflock\b(?!\s+-\w*n\b)", _ssh_log(tmp_path))
    deadline = time.monotonic() + 10
    while not (tmp_path / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (tmp_path / "started").exists()


def test_the_phase_c_resume_clears_the_pause_and_restarts_a_resumed_phase_c_run(tmp_path):
    round_ = _phase_c_round(tmp_path)
    env = _box(tmp_path)
    done = _c("resume", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    log = _ssh_log(tmp_path)
    assert re.search(rf"rm -f '?{re.escape(env['ISSUE_ROOT'])}/PAUSE", log)
    line = _nohup_line(tmp_path)
    assert "--resume" in line and re.search(r"decode\.sh'? run\b[^\n]*\bC\b", line), line
    assert log.index("PAUSE") < log.index("--resume")


def _live_box(tmp_path: Path, live: str) -> dict[str, str]:
    """A box whose ``run.pid`` check answers 4242 for the logs directory containing ``live``."""
    env = _box(tmp_path)
    ssh = Path(env["PATH"].split(":")[0]) / "ssh"
    _mock(ssh, '#!/bin/sh\nprintf "%s\\n" "$2" >> "$SSH_LOG"\n'
          'case "$2" in\n'
          f'  *"{live}"*logs/run.pid*) echo 4242 ;;\n'
          'esac\ncat >/dev/null\n')
    return env


@pytest.mark.parametrize("live, phase", [("/C/", "A"), ("/C/", "B"), ("/B/", "C")])
def test_a_live_loop_of_another_phase_blocks_a_restage(tmp_path, live, phase):
    """The clone is shared by all three phases: restaging under a live loop breaks its batch."""
    round_ = _phase_c_round(tmp_path)
    done = _run("launch", round_, tmp_path, phase, ROUND3_DIR="round3",
                **_live_box(tmp_path, live))
    assert done.returncode != 0 and "still live" in done.stderr and "4242" in done.stderr
    assert "bundle" not in _ssh_log(tmp_path)


def test_a_live_phase_a_loop_blocks_a_phase_c_launch(tmp_path):
    round_ = _phase_c_round(tmp_path)
    env = _box(tmp_path)
    ssh = Path(env["PATH"].split(":")[0]) / "ssh"
    _mock(ssh, '#!/bin/sh\nprintf "%s\\n" "$2" >> "$SSH_LOG"\n'
          'case "$2" in\n  *"/B/"*|*"/C/"*) ;;\n  *logs/run.pid*) echo 4242 ;;\nesac\n'
          'cat >/dev/null\n')
    done = _c("launch", round_, tmp_path, **env)
    assert done.returncode != 0 and "still live" in done.stderr
    assert "bundle" not in _ssh_log(tmp_path)


# --- status and fetch read phase C's own directories -------------------------------------------


def _merge_box(tmp_path: Path, names: list[str], *, exit_code: str = "0\n") -> dict[str, str]:
    """A box with a complete phase C merge under ``C/`` and a *different* finished run in A and
    B's directories. ``ssh`` executes the remote script it is handed and ``rsync`` copies what it
    is pointed at, so the generated checks are run, not merely logged."""
    root = tmp_path / "mining"
    for phase, phase_root in (("A", root), ("B", root / "B"), ("C", root / "C")):
        out, logs = phase_root / "out", phase_root / "logs"
        out.mkdir(parents=True)
        logs.mkdir(parents=True)
        (logs / "run.exit").write_text("0\n" if phase == "C" else "1\n", encoding="utf-8")
        (logs / "run.done").write_text(f"phase {phase}\n", encoding="utf-8")
        (out / f"foreign-{phase}.txt").write_text("not phase C's\n", encoding="utf-8")
    (root / "C" / "logs" / "run.exit").write_text(exit_code, encoding="utf-8")
    for name in names:
        (root / "C" / "out" / name).write_text("", encoding="utf-8")
    bindir = tmp_path / "fetch-bin"
    bindir.mkdir()
    _mock(bindir / "ssh", '#!/bin/sh\nprintf "%s\\n" "$*" >> "$SSH_LOG"\nshift\n'
          'if [ $# -ge 2 ]; then exec "$@"; else exec sh -c "$1"; fi\n')
    _mock(bindir / "rsync", '#!/bin/sh\nprintf "%s\\n" "$*" >> "$RSYNC_LOG"\n'
          'mkdir -p "$3"\ncp -a "${2#*:}." "$3"\n')
    return {"PATH": f"{bindir}:{_env(tmp_path)['PATH']}", "HOST": "box",
            "ISSUE_ROOT": str(root), "SSH_LOG": str(tmp_path / "ssh.log"),
            "RSYNC_LOG": str(tmp_path / "rsync.log"), "FETCH_DIR": str(tmp_path / "fetched")}


def test_the_status_reads_phase_cs_logs_only(tmp_path):
    round_ = _phase_c_round(tmp_path)
    env = _merge_box(tmp_path, PHASE_C_NAMES)
    done = _c("status", round_, tmp_path, **env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.startswith("done: exit 0")


def test_a_pending_phase_c_is_not_mistaken_for_another_phases_finished_run(tmp_path):
    round_ = _phase_c_round(tmp_path)
    env = _merge_box(tmp_path, PHASE_C_NAMES)
    (tmp_path / "mining" / "C" / "logs" / "run.done").unlink()
    done = _c("status", round_, tmp_path, **env)
    assert done.returncode == 3 and done.stdout.startswith("pending")
    assert f"{env['ISSUE_ROOT']}/C/logs/run.done" in done.stdout


def test_the_phase_c_fetch_copies_a_complete_merge_from_cs_out_dir_only(tmp_path):
    round_ = _phase_c_round(tmp_path)
    env = _merge_box(tmp_path, PHASE_C_NAMES)
    done = _c("fetch", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    copied = sorted(path.name for path in Path(env["FETCH_DIR"]).iterdir())
    assert copied == sorted([*PHASE_C_NAMES, "foreign-C.txt"])
    assert f"box:{env['ISSUE_ROOT']}/C/out/" in (tmp_path / "rsync.log").read_text(
        encoding="utf-8")


def test_the_phase_c_fetch_goes_to_the_mining_decodes_directory_by_default(tmp_path):
    round_ = _phase_c_round(tmp_path)
    env = _merge_box(tmp_path, PHASE_C_NAMES)
    del env["FETCH_DIR"]
    done = _c("fetch", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    assert (round_.repo / "tools" / "tadabur" / "tlog_mining" / "decodes"
            / f"meta.{MODEL}.json").is_file()


def test_the_phase_c_fetch_leaves_phase_bs_committed_files_byte_identical(tmp_path):
    round_ = _phase_c_round(tmp_path)
    env = _merge_box(tmp_path, PHASE_C_NAMES)
    fetched = Path(env["FETCH_DIR"])
    fetched.mkdir()
    for name in PHASE_B_FILES:
        (fetched / name).write_bytes(f"phase B's {name}".encode())
    done = _c("fetch", round_, tmp_path, **env)
    assert done.returncode == 0, done.stderr
    for name in PHASE_B_FILES:
        assert (fetched / name).read_bytes() == f"phase B's {name}".encode()
    assert not any(name in {path.name for path in (tmp_path / "mining" / "C" / "out").iterdir()}
                   for name in PHASE_B_FILES)


@pytest.mark.parametrize("missing", PHASE_C_NAMES)
def test_the_phase_c_fetch_refuses_a_merge_missing_one_of_its_files(tmp_path, missing):
    round_ = _phase_c_round(tmp_path)
    env = _merge_box(tmp_path, [name for name in PHASE_C_NAMES if name != missing])
    done = _c("fetch", round_, tmp_path, **env)
    assert done.returncode != 0 and "no complete merge" in done.stderr
    assert not (tmp_path / "rsync.log").exists()


def test_the_phase_c_fetch_does_not_take_phase_bs_meta_for_its_own(tmp_path):
    """A box whose C out holds phase B's names (but not C's meta) is not a complete C merge."""
    round_ = _phase_c_round(tmp_path)
    env = _merge_box(tmp_path, [*PHASE_C_NAMES[:2], "meta.json"])
    done = _c("fetch", round_, tmp_path, **env)
    assert done.returncode != 0 and "no complete merge" in done.stderr
    assert not (tmp_path / "rsync.log").exists()


@pytest.mark.parametrize("exit_code", ["3\n", "1\n"])
def test_the_phase_c_fetch_refuses_a_run_that_did_not_exit_zero(tmp_path, exit_code):
    round_ = _phase_c_round(tmp_path)
    env = _merge_box(tmp_path, PHASE_C_NAMES, exit_code=exit_code)
    done = _c("fetch", round_, tmp_path, **env)
    assert done.returncode != 0 and "no complete merge" in done.stderr
    assert not (tmp_path / "rsync.log").exists()

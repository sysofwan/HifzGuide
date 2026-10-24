"""#179's round-specific behaviour: the #178 candidate's choice, decode gates and reproduction.

The pipeline itself is #95's; a round spec may only add its own candidate and its own record.
These tests pin both sides: the consonant round names its own issue and decodes only its own
pin, and #95's choice and decode resolution stay exactly what they were. Harness-free and
GPU-free: every source is synthetic.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tadabur.qud_text import ItemReference
from tadabur.tarteel_split import LOOK, half
from training import aug_readout as ar
from training import aug_readout_decode as ard
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import consonant_baseline as cb
from training import consonant_readout as cr
from training.external_corpus import WordGrade
from training.muraja_policy import SHIPPED_DEFAULT, SiteGrade
from training.nonpro_eval import ClipItem
from training.word_flags import Flag

TOOLS = Path(__file__).resolve().parents[1]
LAUNCHER = TOOLS.parent / "scripts" / "issue-179-decode.sh"
SHIPPED = SHIPPED_DEFAULT.name
REFERENCE = "قَالَ سَططَ"
CLEAN = REFERENCE.replace(" ", "")
SWAPPED = "قَالَشَططَ"  # the second word's س heard as ش
OFFSETS = (0, 6)
FLAG = Flag(("other_consonant",), (), "minor", (), consonants=(("other_consonant", "س→ش"),))
#: #178's checkpoint and the pin the round must hold (``docs/consonant-round-run.md``).
CANDIDATE_REF = "/root/scratch/issue-178/repo/tools/runs/consonant1_candidate/checkpoint.pt"
CANDIDATE_RECIPE = "aug94-mixture-v2"
CANDIDATE_SHA256 = "057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2"


# --- the round's choice -------------------------------------------------------------------------


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def _repo(tmp_path) -> ag.Layout:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "init", "-q")
    (tmp_path / "README").write_text("x")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "-qm", "init")
    return ag.Layout(tmp_path)


def _commit(layout: ag.Layout) -> None:
    _git(layout.root, "add", layout.look.name, layout.choice.name, layout.claim(LOOK).name)
    _git(layout.root, "commit", "-qm", "the look")


def _candidate_identity(weights: str) -> dict:
    return {"model_ref": CANDIDATE_REF, "weights": weights, "step": 10_000,
            "recipe": CANDIDATE_RECIPE, "checkpoint_sha256": CANDIDATE_SHA256}


def _look_pins() -> dict:
    return {f"candidate_{w}": _candidate_identity(w) for w in ("live", "ema")}


def _sitting(identities: dict) -> dict:
    return {"phase": LOOK, "identities": identities, "inputs": {"look.input": "0" * 64},
            "harness": {"binary_sha256": "a" * 64}}


def _round_choice(layout: ag.Layout, weights: str = "ema", identities: dict | None = None,
                  models: dict | None = None) -> dict:
    """A committed consonant look and choice, as ``run_look`` writes them."""
    sitting = _sitting(identities or _look_pins())
    look_text = ag.canonical({
        "issue": asp.CONSONANT1.issue, "half": LOOK, "items": 4, "inputs": {},
        "rule": ar.CHOICE_RULE, "choice": weights, "control": ar.CONTROL_ON_LOOK,
        "identities": sitting["identities"], "input_sha256": sitting["inputs"],
        "harness": sitting["harness"]})
    layout.look.write_text(look_text, encoding="utf-8")
    layout.claim(LOOK).write_text(ag.canonical(sitting), encoding="utf-8")
    choice = ag.make_choice(weights, look_text, sitting, spec=asp.CONSONANT1)
    if models is not None:
        choice["models"] = models
    layout.choice.write_text(ag.canonical(choice), encoding="utf-8")
    _commit(layout)
    return choice


def test_the_round_s_choice_names_its_issue_and_only_its_candidate():
    sitting = _sitting(_look_pins())
    choice = ag.make_choice("ema", "the look's text", sitting, spec=asp.CONSONANT1)
    assert choice == {"issue": asp.CONSONANT1.issue, "weights": "ema",
                      "models": {"candidate": "candidate_ema"},
                      "identities": _look_pins(),
                      "look_sha256": ag.sha256_bytes("the look's text".encode("utf-8")),
                      "look_inputs_sha256": ag.sha256_bytes(
                          ag.canonical(sitting["inputs"]).encode("utf-8"))}


def test_95_s_choice_is_exactly_what_it_was():
    """The round adds a spec, never a change to #95's committed artifacts."""
    sitting = _sitting({"candidate_live": {"a": 1}, "candidate_ema": {"a": 2}})
    choice = ag.make_choice("ema", "x", sitting)
    assert choice == {
        "issue": 95, "weights": "ema",
        "models": {"candidate": "candidate_ema", "control": "control_ema"},
        "identities": sitting["identities"],
        "look_sha256": ag.sha256_bytes(b"x"),
        "look_inputs_sha256": ag.sha256_bytes(
            ag.canonical(sitting["inputs"]).encode("utf-8"))}


def test_a_committed_round_choice_is_validated_against_the_178_pin(tmp_path):
    layout = _repo(tmp_path / "good")
    choice = _round_choice(layout)
    found = ag.committed_choice(layout, spec=asp.CONSONANT1)
    assert found.choice == choice and found.choice["issue"] == asp.CONSONANT1.issue
    # A well-formed SHA that is not the pin is not this round's candidate.
    bad = _look_pins()
    bad["candidate_ema"] = {**bad["candidate_ema"], "checkpoint_sha256": "b" * 64}
    layout = _repo(tmp_path / "bad")
    _round_choice(layout, identities=bad)
    with pytest.raises(ValueError, match="pinned checkpoint"):
        ag.committed_choice(layout, spec=asp.CONSONANT1)


def test_a_round_choice_that_names_a_control_is_refused(tmp_path):
    layout = _repo(tmp_path)
    _round_choice(layout, models={"candidate": "candidate_ema", "control": "control_ema"})
    with pytest.raises(SystemExit, match="other models"):
        ag.committed_choice(layout, spec=asp.CONSONANT1)


# --- the look sitting ---------------------------------------------------------------------------


def _item(n: int) -> ClipItem:
    name = f"{n:03d}_1_{n}.wav"
    return ClipItem(f"{name}#0", "tarteel_v1", name, 1, 1, 0, 16000, "e" * 64,
                    ItemReference(("قَالَ", "سَططَ"), (7, 8), REFERENCE, OFFSETS),
                    f"r{n % 7}", n % 7, {"final_label": "correct"})


def _line(item: ClipItem, text: str) -> dict:
    return {"key": item.key, "audio_filename": item.audio_filename, "audio_sha256": "e" * 64,
            "start_sample": 0, "end_sample": 16000,
            "checks": [{"hop": text, "overlap": "", "flush": True}], "stream_b0": text}


def _grade(model: str, item: ClipItem, position: int, flagged: bool) -> WordGrade:
    return WordGrade(model, SHIPPED, item, position, 2, position + 1, 7 + position,
                     SiteGrade("flagged" if flagged else "not_flagged"),
                     FLAG if flagged else None)


class _LookSources(ar.Sources):
    """The look half only, synthetic: the two candidate weights and the teacher."""

    def __init__(self, files: Path):
        self.files = files
        files.mkdir(parents=True, exist_ok=True)
        (files / "look.input").write_text("x", encoding="utf-8")
        self.items = [item for item in map(_item, range(40))
                      if half(item.audio_filename) == LOOK][:12]
        assert self.items

    def input_files(self, phase, weights):
        return [self.files / "look.input"]

    def tarteel(self, which_half, models):
        items = [i for i in self.items if half(i.audio_filename) == which_half]
        return ar.SetData(items, {m: {i.key: _line(i, CLEAN if m == "base" else SWAPPED)
                                      for i in items} for m in models})

    def grade(self, items, decodes, models):
        return [_grade(m, i, p, m == "candidate_ema" and p == 1)
                for m in models for i in items for p in (0, 1)]

    harness_record = {"build": {"muraja_commit": "test"}, "binary_sha256": "a" * 64}

    def harness_identity(self):
        return dict(self.harness_record)


def _record(model: str) -> dict:
    return {"model": model, "fingerprints": ag.expected_fingerprints(CANDIDATE_REF),
            "identity": _candidate_identity(model.rsplit("_", 1)[1])}


def test_the_look_sitting_writes_the_round_s_choice(tmp_path):
    layout = _repo(tmp_path / "readout")
    layout.models.write_text(json.dumps(
        {f"tarteel_v1.look/{m}": _record(m) for m in ("candidate_live", "candidate_ema")}),
        encoding="utf-8")
    sources = _LookSources(tmp_path / "files")
    choice = ar.run_look(sources, layout, spec=asp.CONSONANT1)
    look = json.loads(layout.look.read_text(encoding="utf-8"))
    assert choice["issue"] == asp.CONSONANT1.issue
    assert choice["models"] == {"candidate": f"candidate_{choice['weights']}"}
    assert choice["identities"] == _look_pins()
    assert look["issue"] == asp.CONSONANT1.issue
    assert look["input_sha256"] == ag.digests(sources.input_files(LOOK, None))
    assert choice["look_inputs_sha256"] == ag.sha256_bytes(
        ag.canonical(look["input_sha256"]).encode("utf-8"))


# --- the decode run -----------------------------------------------------------------------------


def test_the_decoder_gates_the_round_s_students_and_its_checkpoint(tmp_path):
    layout = _repo(tmp_path)
    _round_choice(layout)
    ard.check_models(ag.TARTEEL_CONFIRM, ["candidate_ema", "c95"], layout,
                     checkpoint_sha256=lambda ref: CANDIDATE_SHA256, spec=asp.CONSONANT1)
    with pytest.raises(SystemExit, match="only for"):
        ard.check_models(ag.TARTEEL_CONFIRM, ["control_ema"], layout, spec=asp.CONSONANT1)
    with pytest.raises(SystemExit, match="not the one the choice read"):
        ard.check_models(ag.TARTEEL_CONFIRM, ["candidate_ema"], layout,
                         checkpoint_sha256=lambda ref: "0" * 64, spec=asp.CONSONANT1)


def test_a_gated_decode_resolves_the_round_s_layout(tmp_path):
    """A gated run with no layout reads this round's committed choice, never #95's (S1.1)."""
    layout = _repo(tmp_path)
    _round_choice(layout)
    spec = dataclasses.replace(asp.CONSONANT1, root=layout.root)
    ard.check_models(ag.QUD_SET, ["candidate_ema"], spec=spec,
                     checkpoint_sha256=lambda ref: CANDIDATE_SHA256)
    elsewhere = dataclasses.replace(spec, root=tmp_path / "none")
    with pytest.raises(SystemExit, match="not committed"):
        ard.check_models(ag.QUD_SET, ["candidate_ema"], spec=elsewhere)


def test_the_decode_run_binds_the_round_s_checkpoint(tmp_path, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(ard, "check_models", lambda *a, **k: None)
    monkeypatch.setattr(ard, "shards", lambda set_name: ["one"])
    monkeypatch.setattr(ard, "recordings", lambda *a: iter(()))
    monkeypatch.setattr(ard, "load_decoders",
                        lambda identity_path, models: captured.update(models)
                        or {m: object() for m in models})
    monkeypatch.setattr(ard, "decode_recordings", lambda *a, **k: {
        "items": 0, "audio_seconds": 0.0, "wall_seconds": 0.0, "decode_seconds": {}})
    monkeypatch.setattr(ard, "append_timing", lambda *a: None)
    ard.run(ag.TARTEEL_LOOK, 0, ["candidate_ema"], tmp_path, tmp_path, None,
            spec=asp.CONSONANT1)
    assert captured["candidate_ema"].ref == asp.CONSONANT1.models["candidate_ema"].ref
    assert captured["candidate_ema"].use_ema is True
    captured.clear()
    ard.run(ag.TARTEEL_LOOK, 0, ["candidate_ema"], tmp_path, tmp_path, None)  # #95 unchanged
    assert captured["candidate_ema"].ref == ag.ARMS["candidate"]


def test_a_candidate_only_merge_completes_the_confirm_manifest(tmp_path, monkeypatch):
    """R1.1: the merge inherits the reused baselines' identities from their own manifest."""
    layout = _repo(tmp_path / "round")
    # The committed round manifest already holds C95's records, qud_other included.
    layout.models.write_text(cr.MODELS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    _round_choice(layout)
    gpu = tmp_path / "gpu"
    gpu.mkdir()
    for set_name in ("tarteel_v1.confirm", "qdat", "qud_other", "qud_half",
                     "tadabur_dev_unseen"):
        (gpu / f"{set_name}.candidate_ema.identity.json").write_text(
            json.dumps(_record("candidate_ema")), encoding="utf-8")
    monkeypatch.setattr("training.external_corpus.merge_decode_files", lambda *a: None)
    ar.merge(gpu, layout, spec=asp.CONSONANT1)
    manifest = json.loads(layout.models.read_text(encoding="utf-8"))
    identities = ag.require_manifest(manifest, ag.needs("confirm", "ema", asp.CONSONANT1),
                                     ag.committed_choice(layout, spec=asp.CONSONANT1).choice,
                                     spec=asp.CONSONANT1)
    assert {"h448", "base"} <= set(identities)
    assert {"tadabur_dev_unseen/h448", "tadabur_dev_unseen/base"} <= set(manifest)


def test_a_candidate_guard_decode_waits_for_the_committed_choice(tmp_path, monkeypatch):
    """R1.3: no held-out clip is decoded for the candidate before the choice is committed."""
    monkeypatch.setattr(cb, "model_caches",
                        lambda *a, **k: pytest.fail("decoded the guard sites"))
    monkeypatch.setattr(cb, "CONSONANT1",
                        dataclasses.replace(asp.CONSONANT1, root=tmp_path / "absent"))
    with pytest.raises(SystemExit, match="not committed"):
        cb.update_held_out(tmp_path, "candidate_ema")
    layout = _repo(tmp_path / "round")
    layout.look.write_text("{}", encoding="utf-8")
    layout.claim(LOOK).write_text("{}", encoding="utf-8")
    layout.choice.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cb, "CONSONANT1",
                        dataclasses.replace(asp.CONSONANT1, root=layout.root))
    with pytest.raises(SystemExit, match="not committed"):
        cb.update_held_out(tmp_path, "candidate_ema")


def test_a_guard_decode_refuses_weights_the_choice_did_not_pick(tmp_path, monkeypatch):
    """R1.3: the guard caches read only the selector the committed choice named."""
    layout = _repo(tmp_path / "round")
    _round_choice(layout, weights="ema")
    monkeypatch.setattr(cb, "CONSONANT1",
                        dataclasses.replace(asp.CONSONANT1, root=layout.root))
    monkeypatch.setattr(cb, "model_caches",
                        lambda *a, **k: pytest.fail("decoded the guard sites"))
    with pytest.raises(SystemExit, match="committed choice reads"):
        cb.update_held_out(tmp_path, "candidate_live")


def test_the_decode_cli_takes_the_round_spec():
    out = subprocess.run([sys.executable, "-m", "training.aug_readout_decode", "--set",
                          ag.TARTEEL_LOOK, "--list-shards", "--spec", "consonant1"],
                         cwd=TOOLS, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert len(out.stdout.splitlines()) == 8


# --- the reproduction check ---------------------------------------------------------------------


class _ConfirmSources:
    def input_files(self, phase, weights):
        return []

    def harness_identity(self):
        return {"binary_sha256": "a" * 64}


def _dry_confirm(monkeypatch, counts) -> None:
    monkeypatch.setattr(ar, "absent_sets", lambda *a: ())
    monkeypatch.setattr(ar, "require_manifest", lambda *a, **k: {})
    monkeypatch.setattr(ar, "check_pins", lambda *a, **k: None)
    monkeypatch.setattr(ar, "baseline_identities", lambda: {})
    monkeypatch.setattr(ar, "load_confirm", lambda *a, **k: object())
    monkeypatch.setattr(ar, "confirm_report",
                        lambda *a, **k: {"verdict": {"verdict": "cannot certify"}})
    monkeypatch.setattr(ar, "reproduction", lambda sources: counts)


def test_the_confirm_sitting_records_the_reproduction_it_passed(tmp_path, monkeypatch):
    _dry_confirm(monkeypatch, ar.REPRODUCTION)
    layout = _repo(tmp_path)
    _round_choice(layout)
    report = ar.run_confirm(_ConfirmSources(), layout, spec=asp.CONSONANT1)
    assert report["reproduction"] == {"passed": True, "counts": ar.REPRODUCTION,
                                      "expected": ar.REPRODUCTION}
    written = json.loads(layout.report.read_text(encoding="utf-8"))
    assert written["reproduction"]["passed"] is True
    assert written["reproduction"]["counts"] == ar.REPRODUCTION


def test_a_failed_reproduction_writes_nothing(tmp_path, monkeypatch):
    perturbed = {"tarteel_v1_confirm": {"flagged": 0, "graded": 41_572},
                 "qdat": {"flagged": 670, "graded": 11_502}}
    _dry_confirm(monkeypatch, perturbed)
    layout = _repo(tmp_path)
    _round_choice(layout)
    with pytest.raises(SystemExit, match="reproduction check failed"):
        ar.run_confirm(_ConfirmSources(), layout, spec=asp.CONSONANT1)
    assert not layout.report.exists() and not layout.verdict.exists()


# --- the launcher -------------------------------------------------------------------------------


def test_the_round_s_launcher_parses():
    assert LAUNCHER.is_file(), f"{LAUNCHER} must exist (commit A ships it)"
    parsed = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert parsed.returncode == 0, parsed.stderr


def test_the_launcher_bundles_a_detached_head(tmp_path):
    """The harness's worktrees are detached: the bundle still carries a real ref (S1.3)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README").write_text("x")
    _git(repo, "add", "README")
    _git(repo, "commit", "-qm", "one")
    _git(repo, "checkout", "-q", "--detach")
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    bundle = tmp_path / "repo.bundle"
    subprocess.run(["bash", "-c", 'source "$1"; bundle_repo "$2" "$3"', "_",
                    str(LAUNCHER), str(repo), str(bundle)], check=True,
                   capture_output=True, text=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(bundle), str(clone)], check=True,
                   capture_output=True, text=True)
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == sha

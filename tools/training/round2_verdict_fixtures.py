"""Synthetic rounds for the round-2 verdict tests (#230): a git repository holding the committed
pre-registration, look, claim and choice, built by running the real look sitting on fake grades.

``committed_round`` is the state the owner's audit starts from: both arms chose their weights on
tlog look gating (EMA for both unless ``flags`` says otherwise) and ``choice.json`` is committed
beside ``look.json``, its claim and the per-word export. Everything lives in one temporary
repository so the claim, preclaim and decode gates, which read ``git`` for "committed in HEAD",
run for real. No harness, no GPU, no corpus.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import round2_fixtures as fx

FLAG = fx.consonant_flag("س→ش")
ITEMS = 12
#: How many of the 12 look items flag their second word, per arm weight: EMA wins for both arms.
EMA_FLAGS = {"r2_candidate_live": 6, "r2_candidate_ema": 2,
             "r2_control_live": 5, "r2_control_ema": 1}
#: The same look with the candidate's live weights and the control's EMA weights winning.
MIXED_FLAGS = {"r2_candidate_live": 1, "r2_candidate_ema": 4,
               "r2_control_live": 5, "r2_control_ema": 2}
CLAIM_COMMIT = "a" * 40


@dataclass
class Round:
    repo: Path
    spec: asp.RoundSpec
    layout: ag.Layout
    tranches: Path

    def commit(self, *paths: Path, message: str = "commit") -> str:
        """Stage and commit ``paths``; returns the new HEAD."""
        fx.git(self.repo, "add", *map(str, paths))
        fx.git(self.repo, "commit", "-qm", message)
        return self.head()

    def head(self) -> str:
        import subprocess

        return subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()

    def write_commit(self, path: Path, text: str, message: str = "commit") -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return self.commit(path, message=message)

    @property
    def preclaim(self) -> Path:
        return self.layout.root / "preclaim.json"

    @property
    def catch_loss(self) -> Path:
        return self.layout.root / "catch_loss"


def arm_identities(directory: Path) -> dict[str, asp.ModelIdentity]:
    """Each arm's checkpoint as a real file under ``directory``, hashed: the arms' live and EMA
    weights, so a decode or cache step that reads the file finds the bytes it must hold."""
    directory.mkdir(parents=True, exist_ok=True)
    identities = {}
    for arm in ("candidate", "control"):
        path = directory / f"{arm}.pt"
        path.write_bytes(f"checkpoint of the {arm}".encode())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for weights in ("live", "ema"):
            identities[f"r2_{arm}_{weights}"] = asp.ModelIdentity(
                str(path), weights, 10_000, fx.RECIPE, sha)
    return identities


def committed_round(tmp_path: Path, flags: dict[str, int] | None = None, *,
                    checkpoints: bool = False) -> Round:
    """A repository with the pre-registration, the look sitting and its choice committed."""
    repo = fx.init_repo(tmp_path / "repo")
    prereg = fx.commit_prereg(repo)
    spec = fx.filled_spec(root=repo / "readout", prereg=prereg)
    if checkpoints:
        spec = dataclasses.replace(
            spec, models={**spec.models, **arm_identities(tmp_path / "checkpoints")})
    spec = dataclasses.replace(spec, doc_path=repo / "docs" / "consonant-round2-readout.md")
    layout = ag.Layout(spec.root)
    layout.root.mkdir()
    layout.models.write_text(json.dumps(fx.look_manifest(spec)), encoding="utf-8")
    flagged = flags or EMA_FLAGS
    items = [fx.tlog_item(n, 1 + n % 5) for n in range(ITEMS)]
    sources = fx.TlogSources(
        tmp_path / "files", items,
        lambda model, i, p: FLAG if p == 1 and i < flagged.get(model, 0) else None, spec=spec)
    ar.run_look(sources, layout, spec=spec)
    tranches = repo / "tranches"
    tranches.mkdir()
    (tranches / "meta.json").write_text("{}", encoding="utf-8")
    round_ = Round(repo, spec, layout, tranches)
    round_.commit(layout.look, layout.choice, layout.claim("look"), layout.models,
                  layout.root / ar.LOOK_GRADES, tranches / "meta.json", message="the look")
    return round_


def plant_audit(round_: Round, real: int, correct: int, unclear: int = 0, *,
                commit: bool = True) -> dict:
    """The owner's catch-loss sitting as the real scorer writes it (``result.json``), over a
    removed stratum of ``real`` mistakes, ``correct`` and ``unclear`` verdicts."""
    from tadabur import catch_loss_audit as audit
    from tadabur.test_catch_loss_audit_score import _plant, _removed

    round_.catch_loss.mkdir(exist_ok=True)
    _plant(round_.catch_loss, _removed(real, correct, unclear))
    result = audit.score(round_.catch_loss, spec=round_.spec)
    if commit:
        round_.commit(round_.catch_loss / audit.RESULT, message="the audit's result")
    return result


CONTROL_COLUMNS = {"qdat": {"all_cause_per_1000": 61.5}}


def clean_report(control: dict | None = None) -> dict:
    """Round 1's synthetic confirm report, every gate clear of its bar, as the pre-claim phase
    builds it (plus ``control``, the control's non-tlog columns). ``_report()`` shares one table
    object between the three bench kinds and between several mechanism tables, so an edit to one
    would move them all: every table is copied apart here."""
    from training.test_consonant_readout_verdict import _report

    report = _report()
    for holder in (report["guards"]["qdat_bench"], report["mechanism"]):
        for key in holder:
            holder[key] = copy.deepcopy(holder[key])
    report["control"] = CONTROL_COLUMNS if control is None else control
    return report

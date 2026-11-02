"""Synthetic rounds for the round-3 readout tests (#255): ``round2_verdict_fixtures``' sibling for
a round with **no control arm** and the claimable tranches ``confirm.3`` and ``confirm.4``.

``committed_round`` is the state the owner's audit starts from, as round 2's is: the look sitting
ran for the round's one arm (``r3_candidate_{live,ema}``, EMA chosen) beside C95 and the teacher,
and ``choice.json`` is committed. The repository's ``tranches`` directory already holds the
committed ``confirm.1`` and ``confirm.2`` claims, as HEAD does (round 2's), so the claim order
that opens ``confirm.3`` is real. The pre-registration committed in the repository is the real
round-3 text, so a doc that quotes it can be compared with the file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import round2_fixtures as fx
from training import round2_verdict_fixtures as rv
from training.truth_baseline import DEFAULT_MODELS

R3_MODELS = ("r3_candidate_live", "r3_candidate_ema")
#: How many of the 12 look items flag their second word, per weight selector: EMA wins.
EMA_FLAGS = {"r3_candidate_live": 6, "r3_candidate_ema": 2}
PREREG_NAME = "consonant-round3-preregistration.md"
REAL_PREREG = asp.CONSONANT3_PREREG_PATH
DOC_NAME = "consonant-round3-readout.md"


def r3_identities() -> dict[str, asp.ModelIdentity]:
    """The round-3 candidate's checkpoint at its live and averaged weights, as the run issue
    will pin them."""
    sha = hashlib.sha256(b"round 3 candidate").hexdigest()
    return {f"r3_candidate_{weights}": asp.ModelIdentity(
        "/root/scratch/round3/candidate/checkpoint.pt", weights, 10_000, fx.RECIPE, sha)
        for weights in ("live", "ema")}


def filled_spec(*, root: Path | None = None, prereg: tuple[Path, str] | None = None
                ) -> asp.RoundSpec:
    """``CONSONANT3`` with the candidate's identities filled and, optionally, its
    pre-registration pointed at a test repository's committed copy, pinned to its SHA-256."""
    spec = dataclasses.replace(asp.CONSONANT3,
                               models={**asp.CONSONANT3.models, **r3_identities()})
    if root is not None:
        spec = dataclasses.replace(spec, root=root)
    if prereg is not None:
        path, sha = prereg
        spec = dataclasses.replace(spec, prereg_path=path,
                                   pins=dataclasses.replace(spec.pins, prereg_sha256=sha))
    return spec


def look_manifest(spec: asp.RoundSpec) -> dict:
    """``models.json`` for the look sitting: the candidate's two weights, C95 and the teacher."""
    records = {f"tlog.look/{m}": fx.model_record(spec, m) for m in (*R3_MODELS, asp.C95)}
    records["tlog.look/base"] = {
        "model": "base", "fingerprints": ag.expected_fingerprints(DEFAULT_MODELS["base"]),
        "identity": ag.baseline_identities()["base"]}
    return records


def confirm_manifest(spec: asp.RoundSpec, weights: str = "ema") -> dict:
    """``models.json`` as the round's merges leave it: the look's records plus the identities
    of every set the pre-claim and confirm sittings read."""
    from tadabur.tarteel_split import CONFIRM
    from training.consonant2_readout import PRECLAIM, sitting_roster

    records = look_manifest(spec)
    for phase in (PRECLAIM, CONFIRM):
        for set_name, models in sitting_roster(spec, phase, weights).items():
            for model in models:
                records[f"{set_name}/{model}"] = fx._identity_record(spec, model)
    return records


def plant_claim(round_: rv.Round, role: str, *, commit: bool = True) -> Path:
    """A valid claim record for ``role`` in the round's tranches directory, committed unless
    ``commit`` is false."""
    path = round_.tranches / f"{role}.claim.json"
    path.write_text(json.dumps({"role": role, "issue": 251, "commit": rv.CLAIM_COMMIT,
                                "date": "2026-11-02"}), encoding="utf-8")
    if commit:
        round_.commit(path, message=f"claim {role}")
    return path


def committed_round(tmp_path: Path) -> rv.Round:
    """A repository with the round-3 pre-registration, the look sitting and its choice
    committed, and round 2's two claims in HEAD's tranches directory."""
    repo = fx.init_repo(tmp_path / "repo")
    prereg_path = repo / "docs" / PREREG_NAME
    prereg_path.parent.mkdir(exist_ok=True)
    text = REAL_PREREG.read_text(encoding="utf-8")
    prereg_path.write_text(text, encoding="utf-8")
    fx.git(repo, "add", f"docs/{PREREG_NAME}")
    fx.git(repo, "commit", "-qm", "freeze the pre-registration")
    spec = filled_spec(root=repo / "readout",
                       prereg=(prereg_path, hashlib.sha256(text.encode("utf-8")).hexdigest()))
    spec = dataclasses.replace(spec, doc_path=repo / "docs" / DOC_NAME)
    layout = ag.Layout(spec.root)
    layout.root.mkdir()
    layout.models.write_text(json.dumps(confirm_manifest(spec)), encoding="utf-8")
    items = [fx.tlog_item(n, 1 + n % 5) for n in range(rv.ITEMS)]
    sources = fx.TlogSources(
        tmp_path / "files", items,
        lambda model, i, p: rv.FLAG if p == 1 and i < EMA_FLAGS.get(model, 0) else None,
        spec=spec)
    ar.run_look(sources, layout, spec=spec)
    tranches = repo / "tranches"
    tranches.mkdir()
    (tranches / "meta.json").write_text("{}", encoding="utf-8")
    round_ = rv.Round(repo, spec, layout, tranches)
    round_.commit(layout.look, layout.choice, layout.claim("look"), layout.models,
                  layout.root / ar.LOOK_GRADES, tranches / "meta.json", message="the look")
    for role in ("confirm.1", "confirm.2"):
        plant_claim(round_, role)
    return round_


def clean_report() -> dict:
    """The pre-claim report as a round with no control builds it: every gate clear of its bar,
    and no ``control`` columns."""
    report = rv.clean_report()
    del report["control"]
    return report

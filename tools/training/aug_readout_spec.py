"""The rounds the readout runs: one pipeline, two specs (#175).

#95's augmentation experiment and the consonant round (#77) share every phase — look, choice,
confirm, verdict, doc — and differ only in what a spec declares: the models, which one is the
paired baseline, the sets, the bars and the pins. The bars themselves live in
``docs/consonant-round-preregistration.md`` (#174) and are quoted, never reinterpreted, by
:mod:`training.aug_readout_verdict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping

from training.aug_readout_metrics import C95, CANDIDATE, CONTROL, H448
from training.external_decode import ModelSpec

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
PREREG_PATH = DOCS_DIR / "consonant-round-preregistration.md"
AUG95_DOC_PATH = DOCS_DIR / "augmentation-readout.md"
CONSONANT_DOC_PATH = DOCS_DIR / "consonant-round-readout.md"

MURAJA_COMMIT = "99c326f3fb7c6c53ff525568976c7e2442ccde3c"
HARNESS_SHA256 = "cfcb99ba583fd2324f7ede0e5ab896379c54fa52783460d583e44d103eb68d8b"
#: The pre-registration's committed text (#174), owner-approved 2026-10-22.
PREREG_SHA256 = "daa611ab401f1350ef920b6801c9c439262579d4aced3f7be062a240cf7ed9d7"
#: C95's checkpoint: the consonant round's paired baseline (#174).
C95_REF = "/root/scratch/issue-94/repo-fix/tools/runs/aug94_candidate_v2/checkpoint.pt"
C95_SHA256 = "8490b63d0664827d5a2e2e3121eb4af7b22069273db39268c7638cc7d8839cd6"
EMA = "ema"


@dataclass(frozen=True)
class Pins:
    """What the round's readout is fixed to, refused if any of it moves (#174)."""

    muraja_commit: str
    harness_sha256: str
    prereg_sha256: str


@dataclass(frozen=True)
class ModelIdentity:
    """The immutable identity of one model the spec reads: its checkpoint and decode facts."""

    ref: str
    weights: str
    step: int
    recipe: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class RoundSpec:
    """Everything one round fixes before its readout: the sets, the baseline, the bars, the pins."""

    name: str
    issue: int
    baseline: str
    sets: tuple[str, ...]
    doc_path: Path
    title: str
    comparisons: tuple[tuple[str, str], ...]
    #: The roles a gated set reads (the round's students), as roles: an arm's role becomes its
    #: chosen weights, a pinned model or a baseline keeps its name (``#176`` decodes C95).
    students: tuple[str, ...] = (CANDIDATE, CONTROL, H448)
    bars: Mapping[str, float] = field(default_factory=dict)
    models: Mapping[str, ModelIdentity] = field(default_factory=dict)
    #: Role -> the committed decode file's model name (C95's decodes are #95's candidate's).
    decode_names: Mapping[str, str] = field(default_factory=dict)
    pins: Pins | None = None


AUG95 = RoundSpec(
    name="aug95", issue=95, baseline=H448,
    sets=("tarteel_v1.look", "tarteel_v1.confirm", "qdat", "retasy", "qud_half",
          "tadabur_dev_unseen"),
    doc_path=AUG95_DOC_PATH, title="The augmentation experiment's readout (#95)",
    comparisons=((CANDIDATE, H448), (CANDIDATE, CONTROL)))

CONSONANT1 = RoundSpec(
    name="consonant1", issue=175, baseline=C95,
    sets=("tarteel_v1.look", "tarteel_v1.confirm", "qdat", "qud_other", "tadabur_dev_unseen",
          "ship_a_verified"),
    doc_path=CONSONANT_DOC_PATH, title="The consonant round's readout (#175)",
    #: The round has no control (#174): C95 is its paired baseline.
    students=(CANDIDATE, C95, H448),
    #: Bar 2's 4.7 is 10 % of C95's published 47.41 flags per 1,000 (#174).
    bars={"all_cause_upper_below": 0.0, "reduction_at_least": 4.7,
          "consonant_upper_below": 0.0, "agreement_lower_above": 0.0,
          "departs_lower_at_least": -0.5, "qud_upper_at_most": 1.0,
          "tadabur_lower_at_least": -1.0, "bench_lower_at_least": -2.0,
          "tashkeel_upper_at_most": 1.0, "catch_at_least": -2.0},
    models={C95: ModelIdentity(C95_REF, EMA, 10_000, "aug94-mixture-v2", C95_SHA256)},
    decode_names={C95: "candidate_ema"},
    comparisons=((CANDIDATE, C95),),
    pins=Pins(MURAJA_COMMIT, HARNESS_SHA256, PREREG_SHA256))

SPECS = {AUG95.name: AUG95, CONSONANT1.name: CONSONANT1}


def spec(name: str) -> RoundSpec:
    """One round's spec by name; an unknown name is a KeyError."""
    return SPECS[name]


def model_spec(specification: RoundSpec, name: str) -> ModelSpec | None:
    """The decode spec of a model the round pins, or ``None`` for one it does not."""
    if name not in specification.models:
        return None
    identity = specification.models[name]
    return ModelSpec(identity.ref, identity.weights == EMA, identity.step, identity.recipe)


def check_pins(specification: RoundSpec, harness: Mapping, prereg_sha256: str) -> None:
    """Refuse a sitting whose Muraja commit, harness binary or pre-registration differs from
    the round's pins (#174). ``harness`` is :func:`training.aug_readout_gate.harness_identity`."""
    pins = specification.pins
    if pins is None:
        return
    observed = {"Muraja pin": (harness.get("muraja_commit"), pins.muraja_commit),
                "harness pin": (harness.get("binary_sha256"), pins.harness_sha256),
                "pre-registration pin": (prereg_sha256, pins.prereg_sha256)}
    for name, (found, pinned) in observed.items():
        if found != pinned:
            raise SystemExit(f"{name} differs: {found!r} is not the pinned {pinned!r}")

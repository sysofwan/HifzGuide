"""The rounds the readout runs: one pipeline, three specs (#175, #227).

#95's augmentation experiment and the two consonant rounds (#77) share every phase — look,
choice, confirm, verdict, doc — and differ only in what a spec declares: the models, which one
is the paired baseline, the sets, the bars and the pins. The bars themselves live in each
round's pre-registration (``docs/consonant-round-preregistration.md`` for #174) and are quoted,
never reinterpreted, by :mod:`training.aug_readout_verdict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping

from tadabur.tarteel_split import CONFIRM, LOOK
from training.aug_readout_metrics import BASE, C95, CANDIDATE, CONTROL, H448
from training.external_decode import ModelSpec

AUG95_ROOT = Path(__file__).parent.parent / "tadabur" / "aug_readout"
CONSONANT_ROOT = Path(__file__).parent.parent / "tadabur" / "consonant_readout"
CONSONANT2_ROOT = Path(__file__).parent.parent / "tadabur" / "consonant2_readout"
DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
PREREG_PATH = DOCS_DIR / "consonant-round-preregistration.md"
CONSONANT2_PREREG_PATH = DOCS_DIR / "consonant-round2-preregistration.md"
AUG95_DOC_PATH = DOCS_DIR / "augmentation-readout.md"
CONSONANT_DOC_PATH = DOCS_DIR / "consonant-round-readout.md"
CONSONANT2_DOC_PATH = DOCS_DIR / "consonant-round2-readout.md"
CONSONANT_FREEZE_DOC_PATH = DOCS_DIR / "consonant-round-freeze.md"
CONSONANT2_FREEZE_DOC_PATH = DOCS_DIR / "consonant-round2-freeze.md"

MURAJA_COMMIT = "99c326f3fb7c6c53ff525568976c7e2442ccde3c"
HARNESS_SHA256 = "cfcb99ba583fd2324f7ede0e5ab896379c54fa52783460d583e44d103eb68d8b"
#: The pre-registration's committed text (#174), owner-approved 2026-10-22.
PREREG_SHA256 = "daa611ab401f1350ef920b6801c9c439262579d4aced3f7be062a240cf7ed9d7"
#: C95's checkpoint: the consonant round's paired baseline (#174).
C95_REF = "/root/scratch/issue-94/repo-fix/tools/runs/aug94_candidate_v2/checkpoint.pt"
C95_SHA256 = "8490b63d0664827d5a2e2e3121eb4af7b22069273db39268c7638cc7d8839cd6"
#: #178's candidate: C95 continued for 10,000 steps under ``aug94-mixture-v2``. One file holds
#: both weight selectors (docs/consonant-round-run.md).
CANDIDATE_REF = "/root/scratch/issue-178/repo/tools/runs/consonant1_candidate/checkpoint.pt"
CANDIDATE_SHA256 = "057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2"
CANDIDATE_RECIPE = "aug94-mixture-v2"
LIVE = "live"
EMA = "ema"
#: A pin the freeze issue has not filled in yet: no observed value equals it, so
#: :func:`check_pins` refuses every sitting of a round whose pin is still unfilled (#227).
UNFILLED_PIN = None


@dataclass(frozen=True)
class Pins:
    """What the round's readout is fixed to, refused if any of it moves (#174)."""

    muraja_commit: str
    harness_sha256: str
    prereg_sha256: str | None


@dataclass(frozen=True)
class ModelIdentity:
    """The immutable identity of one model the spec reads: its checkpoint and decode facts."""

    ref: str
    weights: str
    step: int
    recipe: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class Decodes:
    """Where one model's decode files of a set are committed: a readout root and the file name."""

    root: Path
    model: str


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
    #: The roles whose **readout decodes** a gated set reads, as roles: an arm's role becomes its
    #: chosen weights, a pinned model (``#176`` decodes C95) keeps its name. The reused baselines
    #: (``h448``, the teacher) are committed elsewhere and added on Tadabur, as #95 does.
    students: tuple[str, ...] = (CANDIDATE, CONTROL)
    bars: Mapping[str, float] = field(default_factory=dict)
    models: Mapping[str, ModelIdentity] = field(default_factory=dict)
    #: This round's own readout directory: its decodes, manifests, choice and results live here,
    #: never in another round's, whose committed decodes stay immutable (#175 review R1.1).
    root: Path = AUG95_ROOT
    #: ``(set, role)`` -> the decode files another round already committed for it, e.g. #95's
    #: immutable candidate for C95. Every other set and role reads this round's own root.
    decode_sources: Mapping[tuple[str, str], Decodes] = field(default_factory=dict)
    pins: Pins | None = None
    #: The document the round's pre-registration is committed at; ``pins.prereg_sha256`` is its
    #: SHA-256 once the freeze issue has filled that in.
    prereg_path: Path = PREREG_PATH
    #: The set the look sitting reads, whose first grade is the freeze (#174, #227).
    look_set: str = f"tarteel_v1.{LOOK}"
    #: The roles decoded on the look set, each at both weight selectors. One role is #95's and
    #: the consonant round's candidate alone (the control is never decoded on the look half);
    #: round 2 decodes both arms, so each chooses its own weights.
    look_roles: tuple[str, ...] = (CANDIDATE,)
    #: The prefix of an arm's decode name: #95's and the consonant round's arms are #94's
    #: checkpoints, named ``<arm>_<weights>``; round 2's four arms are its own checkpoints,
    #: named ``r2_<arm>_<weights>`` and pinned in ``models``.
    arm_prefix: str = ""
    #: The qdat_bench letter kinds G3 gates (#174 gates all three; round 2 reports shaddah
    #: only, owner 2026-10-28).
    g3_kinds: tuple[str, ...] = ("consonant", "haraka", "shaddah")
    #: The doc the round's freeze renders to.
    freeze_doc_path: Path = CONSONANT_FREEZE_DOC_PATH


AUG95 = RoundSpec(
    name="aug95", issue=95, baseline=H448,
    sets=("tarteel_v1.look", "tarteel_v1.confirm", "qdat", "retasy", "qud_half",
          "tadabur_dev_unseen"),
    doc_path=AUG95_DOC_PATH, title="The augmentation experiment's readout (#95)",
    comparisons=((CANDIDATE, H448), (CANDIDATE, CONTROL)))

#: The sets #95 committed decodes of and the round reads, so the round reuses those files
#: unchanged instead of decoding its own models on them again. ``qud_other`` is the round's own
#: set (no #95 file exists); RetaSy stays #171's descriptive-only subtree.
REUSED_SETS = tuple(name for name in AUG95.sets if name != "retasy")


CONSONANT1 = RoundSpec(
    name="consonant1", issue=175, baseline=C95,
    #: ``qud_half`` is #95's half, reported not gating: it is still declared, so its identities,
    #: merge counts and input digests are validated like every other set the sitting reads.
    sets=("tarteel_v1.look", "tarteel_v1.confirm", "qdat", "qud_other", "qud_half",
          "tadabur_dev_unseen", "ship_a_verified"),
    doc_path=CONSONANT_DOC_PATH, title="The consonant round's readout (#175)",
    #: The round has no control (#174): C95 is its paired baseline and its one extra student.
    students=(CANDIDATE, C95),
    #: Bar 2's 4.7 is 10 % of C95's published 47.41 flags per 1,000 (#174).
    bars={"all_cause_upper_below": 0.0, "reduction_at_least": 4.7,
          "consonant_upper_below": 0.0, "agreement_lower_above": 0.0,
          "departs_lower_at_least": -0.5, "qud_upper_at_most": 1.0,
          "tadabur_lower_at_least": -1.0, "bench_lower_at_least": -2.0,
          "tashkeel_upper_at_most": 1.0, "catch_at_least": -2.0},
    models={
        C95: ModelIdentity(C95_REF, EMA, 10_000, "aug94-mixture-v2", C95_SHA256),
        # The candidate's two selectors of the one #178 checkpoint: the round decodes both on
        # the look half and refuses any other checkpoint, step or recipe (#178, #179).
        "candidate_live": ModelIdentity(CANDIDATE_REF, LIVE, 10_000, CANDIDATE_RECIPE,
                                        CANDIDATE_SHA256),
        "candidate_ema": ModelIdentity(CANDIDATE_REF, EMA, 10_000, CANDIDATE_RECIPE,
                                       CANDIDATE_SHA256),
    },
    root=CONSONANT_ROOT,
    #: C95's decodes are #95's committed candidate's, for the sets #95 fixed; the baselines'
    #: are #95's too. The round's candidate decodes land in :data:`CONSONANT_ROOT`, so merging
    #: it can never overwrite #95's immutable inputs.
    decode_sources={
        **{(name, C95): Decodes(AUG95_ROOT, "candidate_ema") for name in REUSED_SETS},
        **{(name, model): Decodes(AUG95_ROOT, model)
           for name in REUSED_SETS for model in (H448, BASE)}},
    comparisons=((CANDIDATE, C95),),
    pins=Pins(MURAJA_COMMIT, HARNESS_SHA256, PREREG_SHA256))

#: #178's candidate, round 2's context column, under the round's own role name.
C178 = "c178"
#: The prefix of the tlog tranche sets (#227).
TLOG_PREFIX = "tlog."
TLOG_LOOK, TLOG_CONFIRM = f"tlog.{LOOK}", f"tlog.{CONFIRM}"

CONSONANT2 = RoundSpec(
    name="consonant2", issue=227, baseline=C95,
    #: The tlog look and confirm sets (#227); the non-tlog sets are round 1's, read unchanged.
    sets=(TLOG_LOOK, TLOG_CONFIRM, "qdat", "qdat_bench", "qud_other", "tadabur_dev_unseen",
          "ship_a_verified", f"tarteel_v1.{LOOK}", f"tarteel_v1.{CONFIRM}"),
    doc_path=CONSONANT2_DOC_PATH, title="The tlog consonant round's readout (#227)",
    #: The round reads both arms and its pinned baseline (owner 2026-10-28): the control is
    #: reported, never in the verdict.
    students=(CANDIDATE, CONTROL, C95),
    #: #174's bars unchanged, plus the catch-loss audit the owner added for this round.
    bars={**CONSONANT1.bars, "catch_loss_real_at_most": 0.20,
          "catch_loss_decisive_at_least": 80},
    models={
        C95: CONSONANT1.models[C95],
        # #178's one checkpoint at its averaged weights, the round's context column.
        C178: CONSONANT1.models["candidate_ema"],
    },
    root=CONSONANT2_ROOT,
    #: #212's committed look decodes keep their file names: the file called ``candidate_ema`` in
    #: the round's root is #178, so ``c178`` reads it. The round's four arms land beside them,
    #: and every non-tlog set reads the decodes round 1 committed for C95 and #178.
    decode_sources={
        (TLOG_LOOK, C178): Decodes(CONSONANT2_ROOT, "candidate_ema"),
        **{(name, C95): Decodes(AUG95_ROOT, "candidate_ema") for name in REUSED_SETS},
        **{(name, C178): Decodes(CONSONANT_ROOT, "candidate_ema") for name in REUSED_SETS},
        # ``qud_other`` is round 1's own set: its C95 and #178 decodes are round 1's too.
        ("qud_other", C95): Decodes(CONSONANT_ROOT, "c95"),
        ("qud_other", C178): Decodes(CONSONANT_ROOT, "candidate_ema"),
        **{(name, model): Decodes(AUG95_ROOT, model)
           for name in REUSED_SETS for model in (H448, BASE)}},
    comparisons=((CANDIDATE, C95),),
    pins=Pins(MURAJA_COMMIT, HARNESS_SHA256, UNFILLED_PIN),
    prereg_path=CONSONANT2_PREREG_PATH,
    look_set=TLOG_LOOK,
    look_roles=(CANDIDATE, CONTROL),
    arm_prefix="r2_",
    #: G3's shaddah part is reported only (owner 2026-10-28): the catch side of a geminate is
    #: no longer gated; flags the app shows stay in the all-cause bars.
    g3_kinds=("consonant", "haraka"),
    freeze_doc_path=CONSONANT2_FREEZE_DOC_PATH)

SPECS = {AUG95.name: AUG95, CONSONANT1.name: CONSONANT1, CONSONANT2.name: CONSONANT2}


def spec(name: str) -> RoundSpec:
    """One round's spec by name; an unknown name is a KeyError."""
    return SPECS[name]


def tlog_round(specification: RoundSpec) -> bool:
    """Whether the round reads the tlog tranches (round 2): its decodes are graded only after
    the pre-registration is committed, its look sitting exports the audit's word grades, and
    its pooled confirm read is part 2 (#227)."""
    return any(name.startswith(TLOG_PREFIX) for name in specification.sets)


def model_spec(specification: RoundSpec, name: str) -> ModelSpec | None:
    """The decode spec of a model the round pins, or ``None`` for one it does not."""
    if name not in specification.models:
        return None
    identity = specification.models[name]
    return ModelSpec(identity.ref, identity.weights == EMA, identity.step, identity.recipe)


def check_pins(specification: RoundSpec, harness: Mapping, prereg_sha256: str) -> None:
    """Refuse a sitting whose Muraja commit, harness binary or pre-registration differs from
    the round's pins (#174), and every sitting of a round whose pre-registration pin is still
    unfilled (#227). ``harness`` is
    :func:`training.aug_readout_gate.harness_identity`."""
    pins = specification.pins
    if pins is None:
        return
    if pins.prereg_sha256 is None:
        raise SystemExit(f"the pre-registration pin is unfilled: "
                         f"{specification.prereg_path.name} has no committed SHA-256 to check "
                         f"against; the freeze issue fills it in")
    observed = {"Muraja pin": (harness.get("muraja_commit"), pins.muraja_commit),
                "harness pin": (harness.get("binary_sha256"), pins.harness_sha256),
                "pre-registration pin": (prereg_sha256, pins.prereg_sha256)}
    for name, (found, pinned) in observed.items():
        if found != pinned:
            raise SystemExit(f"{name} differs: {found!r} is not the pinned {pinned!r}")

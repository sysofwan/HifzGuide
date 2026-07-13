"""ADR-0004 ablation ladder — the (1)→(2)→(3) go/no-go orchestration (issue #33, P7.E).

The joint detached-waqf run (rung 3) shares Muaalem's phoneme path with the whole-clip
phoneme-only run (rung 2), which sits on the segmented ADR-0001/0003 baseline (rung 1). The
isolation claim ADR-0004 rests on is *verified, not assumed*, with a three-rung ladder:

  (1) segmented phoneme-only        — the ADR-0001/0003 baseline (scored elsewhere)
  (2) whole-clip phoneme-only       — :mod:`training.whole_clip_phoneme`
  (3) whole-clip phoneme + detached waqf — :mod:`training.joint_waqf`

This module owns the ladder itself — three concerns, one per ADR-0004 go/no-go:

* **The deterministic (2)↔(3) phoneme identity check.** :func:`phoneme_identity_report`
  runs rung (2)'s phoneme forward and rung (3)'s joint model on the *same* backbone and batch
  and asserts the phoneme **logits** and the **backbone gradients** they induce are identical.
  This is the direct, cheap, deterministic form of ADR-0004's (2)→(3) go/no-go: the waqf head
  may only *add* a detached branch, never reshape the phoneme path. It complements the
  waqf-KL-alone isolation check in :mod:`training.joint_waqf`; together they pin the isolation
  from both sides (adding waqf changes nothing; waqf alone touches nothing).

* **The should-accept / should-reject deltas across all three rungs.** :class:`AblationLadder`
  reads the three #7 :class:`~tadabur.eval_report.EvalReport` JSONs and reports the recall /
  discrimination deltas at each transition — the (1)→(2) *whole-clip move* and the (2)→(3)
  *waqf-head addition* — the numbers the ladder's verdict turns on.

* **The whole-clip-move regression → LoRA-native lever, an owned action.** (1)→(2) measures
  the whole-clip move itself and must independently clear the should-reject bar. When it
  regresses, ADR-0004's owned response is **LoRA-native** — lower rank/alpha, then L2-SP on the
  adapters — *not* reattaching the sifat heads (that is :mod:`training.joint_waqf`'s last
  resort). :func:`recommend_lora_lever` selects the concrete next lever
  (:class:`LoRALever` → a re-runnable rung-(2) config).

Pure orchestration: it consumes the rungs' eval outputs and a tiny forward pass, so it runs on
CPU in tests and needs no CUDA. See ADR-0004 and issues #7 / #29 / #31.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from tadabur.inference import PHONEME_LEVEL
from training.waqf_head import WaqfJointModel, phoneme_ctc_loss, phoneme_forward
from training.whole_clip_phoneme import LoRASettings

# Any measured drop in should-reject discrimination across the whole-clip move (1)→(2) is a
# regression: should-reject is the safety property (a genuinely-wrong recitation slipping
# through), and the harness metric is an exact rational, so equality is exact and 0.0 needs no
# float tolerance. Callers may loosen it, but the default treats any erosion as actionable.
DEFAULT_DISCRIMINATION_REGRESSION_TOLERANCE = 0.0

# The L2-SP anchor weight the second LoRA-native lever pulls the adapters toward their starting
# point with (consumed by training.whole_clip_phoneme.TrainConfig.l2_sp). A knob, not a law.
DEFAULT_L2_SP = 1e-3


# --- (2)↔(3) phoneme identity check (the go/no-go) ---------------------------


@dataclass
class PhonemeIdentityReport:
    """Whether rung (2) and rung (3) share the phoneme path bit-for-bit on one batch."""

    logits_match: bool
    max_logit_diff: float
    grads_match: bool
    max_grad_diff: float

    @property
    def identical(self) -> bool:
        return self.logits_match and self.grads_match


def _phoneme_backbone_grads(base: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot every backbone parameter's gradient by name, then clear it."""
    grads = {
        name: param.grad.detach().clone()
        for name, param in base.named_parameters()
        if param.grad is not None
    }
    base.zero_grad(set_to_none=True)
    return grads


def phoneme_identity_report(
    base: torch.nn.Module,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> PhonemeIdentityReport:
    """Prove rung (2) and rung (3) compute an identical phoneme path on ``base``.

    Runs, on the **same** backbone and batch and in eval mode (dropout off, so the check is
    deterministic): rung (2)'s bare :func:`phoneme_forward` and rung (3)'s
    :class:`WaqfJointModel`, backpropagating the *same* phoneme CTC loss from each. Asserts
    the phoneme logits are equal and every backbone gradient is equal — the direct (2)→(3)
    identity ADR-0004's go/no-go requires. Any difference means the waqf head has reshaped the
    phoneme path (broken isolation), so a regression could not be attributed to it.
    """
    joint = WaqfJointModel(base, phoneme_level=PHONEME_LEVEL)
    joint.eval()

    base.zero_grad(set_to_none=True)
    rung2 = phoneme_forward(base, PHONEME_LEVEL, input_features, attention_mask)
    phoneme_ctc_loss(rung2.phoneme_logits, labels, rung2.student_lengths, base.config).backward()
    grads_2 = _phoneme_backbone_grads(base)

    rung3 = joint(input_features, attention_mask)
    phoneme_ctc_loss(rung3.phoneme_logits, labels, rung3.student_lengths, base.config).backward()
    grads_3 = _phoneme_backbone_grads(base)

    max_logit_diff = float(
        (rung2.phoneme_logits - rung3.phoneme_logits).detach().abs().max()
    )
    max_grad_diff = max(
        (float((grads_2[name] - grads_3[name]).abs().max()) for name in grads_2),
        default=0.0,
    )
    return PhonemeIdentityReport(
        logits_match=max_logit_diff == 0.0 and grads_2.keys() == grads_3.keys(),
        max_logit_diff=max_logit_diff,
        grads_match=grads_2.keys() == grads_3.keys() and max_grad_diff == 0.0,
        max_grad_diff=max_grad_diff,
    )


def assert_phoneme_identity(
    base: torch.nn.Module,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> PhonemeIdentityReport:
    """Raise unless rung (2) and rung (3) share the phoneme path bit-for-bit."""
    report = phoneme_identity_report(base, input_features, attention_mask, labels)
    if not report.identical:
        raise AssertionError(
            "rung (2) and rung (3) diverge on the phoneme path "
            f"(max logit diff {report.max_logit_diff:.3e}, max grad diff "
            f"{report.max_grad_diff:.3e}) — the waqf head reshaped the phoneme path, so the "
            "(2)→(3) isolation ADR-0004 requires does not hold."
        )
    return report


# --- three-rung ladder deltas (#7 harness numbers) ---------------------------


@dataclass(frozen=True)
class RungMetrics:
    """One rung's headline #7 numbers: should-accept recall + should-reject discrimination."""

    name: str
    should_accept_recall: Optional[float]
    should_reject_discrimination: Optional[float]

    @classmethod
    def from_eval_json(cls, name: str, report: dict) -> "RungMetrics":
        """Read a rung's numbers from a :meth:`EvalReport.to_json_dict` mapping."""
        return cls(
            name=name,
            should_accept_recall=report["should_accept"]["recall"],
            should_reject_discrimination=report["should_reject"]["discrimination"],
        )


@dataclass(frozen=True)
class LadderTransition:
    """The recall / discrimination deltas across one rung transition (higher is better)."""

    from_rung: str
    to_rung: str
    recall_delta: Optional[float]
    discrimination_delta: Optional[float]


def _delta(later: Optional[float], earlier: Optional[float]) -> Optional[float]:
    if later is None or earlier is None:
        return None
    return later - earlier


def _transition(earlier: RungMetrics, later: RungMetrics) -> LadderTransition:
    return LadderTransition(
        from_rung=earlier.name,
        to_rung=later.name,
        recall_delta=_delta(later.should_accept_recall, earlier.should_accept_recall),
        discrimination_delta=_delta(
            later.should_reject_discrimination, earlier.should_reject_discrimination
        ),
    )


@dataclass(frozen=True)
class AblationLadder:
    """The full (1)→(2)→(3) ladder: three rungs' numbers and the two transitions between them."""

    segmented: RungMetrics
    whole_clip: RungMetrics
    joint: RungMetrics

    @classmethod
    def from_reports(
        cls, segmented: dict, whole_clip: dict, joint: dict
    ) -> "AblationLadder":
        """Build the ladder from the three #7 eval-report JSON mappings (rung 1/2/3)."""
        return cls(
            segmented=RungMetrics.from_eval_json("segmented_phoneme_only", segmented),
            whole_clip=RungMetrics.from_eval_json("whole_clip_phoneme_only", whole_clip),
            joint=RungMetrics.from_eval_json("whole_clip_joint_waqf", joint),
        )

    @property
    def whole_clip_move(self) -> LadderTransition:
        """(1)→(2): the whole-clip move, which must independently clear the should-reject bar."""
        return _transition(self.segmented, self.whole_clip)

    @property
    def waqf_head_addition(self) -> LadderTransition:
        """(2)→(3): adding the detached waqf head, which the identity check pins to ~zero."""
        return _transition(self.whole_clip, self.joint)

    def to_json_dict(self) -> dict:
        return {
            "rungs": {
                rung.name: {
                    "should_accept_recall": rung.should_accept_recall,
                    "should_reject_discrimination": rung.should_reject_discrimination,
                }
                for rung in (self.segmented, self.whole_clip, self.joint)
            },
            "transitions": {
                "whole_clip_move": asdict(self.whole_clip_move),
                "waqf_head_addition": asdict(self.waqf_head_addition),
            },
        }


# --- whole-clip-move regression → LoRA-native lever (the owned action) --------


@dataclass(frozen=True)
class LoRALever:
    """One LoRA-native response to a whole-clip-move should-reject regression (ADR-0004).

    Each lever is a re-runnable rung-(2) config: ``lora`` (rank/alpha, the first lever) and
    ``l2_sp`` (the adapter anchor, the second) feed
    :class:`training.whole_clip_phoneme.TrainConfig` directly. Reattaching sifat is *not* on
    this ladder — that is :mod:`training.joint_waqf`'s last resort, deliberately kept out of
    the whole-clip-move response.
    """

    name: str
    description: str
    lora: LoRASettings
    l2_sp: float


# The ordered LoRA-native levers ADR-0004 names for a should-reject regression on the
# whole-clip move: "lower rank/alpha, or L2-SP on the adapters". The first entry is the
# rung-(2) default (the config that regressed); the responses follow in order.
LORA_LEVER_LADDER: tuple[LoRALever, ...] = (
    LoRALever(
        name="baseline",
        description="Rung-(2) default LoRA (rank 16 / alpha 32), no anchor — the run that regressed.",
        lora=LoRASettings(),
        l2_sp=0.0,
    ),
    LoRALever(
        name="lower_rank_alpha",
        description="Lower LoRA rank/alpha (8 / 16) — tighten the bounded backbone drift.",
        lora=LoRASettings(rank=8, alpha=16),
        l2_sp=0.0,
    ),
    LoRALever(
        name="lower_rank_alpha_l2_sp",
        description="Lower rank/alpha + L2-SP anchor on the adapters — the harder drift bound.",
        lora=LoRASettings(rank=8, alpha=16),
        l2_sp=DEFAULT_L2_SP,
    ),
)

_LEVER_BY_NAME = {lever.name: lever for lever in LORA_LEVER_LADDER}


def lever_by_name(name: str) -> LoRALever:
    try:
        return _LEVER_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown LoRA lever {name!r}; the ladder is {[l.name for l in LORA_LEVER_LADDER]}."
        ) from exc


def next_lora_lever(current: LoRALever) -> Optional[LoRALever]:
    """The next LoRA-native lever after ``current``, or ``None`` at the end of the ladder."""
    following = LORA_LEVER_LADDER[LORA_LEVER_LADDER.index(current) + 1 :]
    return following[0] if following else None


def whole_clip_move_regressed(
    ladder: AblationLadder,
    tolerance: float = DEFAULT_DISCRIMINATION_REGRESSION_TOLERANCE,
) -> bool:
    """True when (1)→(2) eroded should-reject discrimination beyond ``tolerance``.

    ADR-0004 makes the whole-clip move's should-reject regression the trigger for the
    LoRA-native levers. A ``None`` delta (a rung missing that side) is treated as not a
    regression — the ladder cannot judge what it cannot measure.
    """
    delta = ladder.whole_clip_move.discrimination_delta
    return delta is not None and delta < -tolerance


def recommend_lora_lever(
    ladder: AblationLadder,
    current: LoRALever = LORA_LEVER_LADDER[0],
    tolerance: float = DEFAULT_DISCRIMINATION_REGRESSION_TOLERANCE,
) -> Optional[LoRALever]:
    """The next LoRA-native lever to run if the whole-clip move regressed, else ``None``.

    This is ADR-0004's owned action for a (1)→(2) should-reject regression: escalate one rung
    down the LoRA-native ladder (rank/alpha → L2-SP), never reattach sifat.
    """
    if not whole_clip_move_regressed(ladder, tolerance):
        return None
    return next_lora_lever(current)


# --- CLI ---------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cmd_report(args: argparse.Namespace) -> None:
    ladder = AblationLadder.from_reports(
        _load_json(args.segmented), _load_json(args.whole_clip), _load_json(args.joint)
    )
    lever = recommend_lora_lever(ladder, tolerance=args.tolerance)
    payload = {
        "ladder": ladder.to_json_dict(),
        "whole_clip_move_regressed": whole_clip_move_regressed(ladder, args.tolerance),
        "recommended_lora_lever": (
            {
                "name": lever.name,
                "description": lever.description,
                "lora_rank": lever.lora.rank,
                "lora_alpha": lever.lora.alpha,
                "l2_sp": lever.l2_sp,
            }
            if lever is not None
            else None
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser(
        "report", help="compute the (1)→(2)→(3) deltas and the owned LoRA-native lever"
    )
    rep.add_argument("--segmented", type=Path, required=True,
                     help="rung (1) segmented phoneme-only #7 eval-report JSON.")
    rep.add_argument("--whole-clip", type=Path, required=True,
                     help="rung (2) whole-clip phoneme-only #7 eval-report JSON.")
    rep.add_argument("--joint", type=Path, required=True,
                     help="rung (3) whole-clip joint detached-waqf #7 eval-report JSON.")
    rep.add_argument("--tolerance", type=float,
                     default=DEFAULT_DISCRIMINATION_REGRESSION_TOLERANCE,
                     help="should-reject discrimination drop tolerated on the whole-clip move.")
    rep.add_argument("--out", type=Path, default=None,
                     help="optional path to also write the ladder report JSON.")
    rep.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Distillation losses and diagnostics for the thin Muaalem student.

The objective is **behavioural cloning of the teacher's phoneme head**, not accuracy: the
student is correct exactly insofar as it reproduces what ``obadx/muaalem-model-v3_2``
already produces, because that is the behaviour Muraja's scorer, fixtures and thresholds
are all tuned against. Everything here therefore measures student-vs-teacher, never
student-vs-reference.

**The objective is a single term: frame-weighted KL on the logits.** It arrived there by
subtraction, and the removals are the useful part of the story.

A tapped hidden-state matching term and a CTC anchor against the teacher's decoded sequence
were both tried and both removed. :func:`feature_matching_loss` and :func:`ctc_anchor_loss`
survive as flags so the ablations stay reproducible; neither should be switched on again
without new evidence. See ADR-0010 for the measurements, and :class:`DistillLossConfig` for
the short version.

Two weightings shape the remaining term, and both exist because an unweighted mean over the
125-timestep lattice optimises the wrong thing.

**Non-blank frames are up-weighted.** CTC output is blank-dominated -- per Muraja's own
``CTCStats``, blank holds 30-60% of timesteps during speech and 85-98% during silence. A
flat KL is therefore mostly a lesson in predicting blank, and a student can drive the loss
down while its decoded string degrades. ``training.waqf_head`` hit the mirror image of this
(silence frames swamped by the speech majority) and answered it with ``pause_frame_weights``
plus a collapse diagnostic; :func:`frame_weights` and :func:`agreement_stats` are the same
two answers pointed at blank.

**The confirmed region is up-weighted.** Only the first ``CONFIRM_TIMESTEPS`` of each
window ever reach the user. ``MuaalemInference.predictSplit`` splits on
``seg.midpoint < splitPoint`` with ``splitPoint = confirmTimeSteps = 25``, so a phoneme is
committed to the transcript when its window position is *oldest* -- having accumulated the
full 4 s of right context. The remaining 100 timesteps only ever drive the provisional
overlap display and the hallucination gate. They still matter, so they keep weight 1.0
rather than being masked out, but the region that becomes the transcript is worth more.

One caution carried over from the removals: **frame agreement and decoded agreement are not
the same objective.** Under a hard-label variant, frame ``confirmed_agreement`` rose while
decoded char accuracy fell. After the ``scanCTC`` collapse, *where* an error lands matters
more than how many there are, so any future objective work should be scored on the decoded
stream from the start.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Blank/pad class of the phoneme head (``tadabur.phoneme_vocab.PHONEME_PAD_ID``).
BLANK_ID = 0

# Timesteps of each 125-frame window that become the user-visible transcript
# (``MuaalemInference.predictSplit``; ``hopTimeSteps = outputTimeSteps / 5``).
CONFIRM_TIMESTEPS = 25

# Defaults chosen so neither weighting dominates: with ~50% non-blank during speech, a 3x
# non-blank boost moves the effective mass to roughly 3:1 toward the frames that carry
# phonemes, and the 2x confirmed boost leaves the other 100 timesteps clearly present.
DEFAULT_NONBLANK_WEIGHT = 3.0
DEFAULT_CONFIRM_WEIGHT = 2.0
DEFAULT_TEMPERATURE = 2.0

# Encoder depths tapped for feature matching, as indices into ``hidden_states`` (which is
# 25 long: the post-projection embedding followed by all 24 layers). Every fourth layer,
# ending at the final one the CTC head actually reads.
DEFAULT_TAP_LAYERS = (4, 8, 12, 16, 20, 24)


def frame_weights(
    teacher_logits: torch.Tensor,
    nonblank_weight: float = DEFAULT_NONBLANK_WEIGHT,
    confirm_timesteps: int = CONFIRM_TIMESTEPS,
    confirm_weight: float = DEFAULT_CONFIRM_WEIGHT,
) -> torch.Tensor:
    """Per-frame loss weights, shaped ``(B, T)``.

    The two boosts multiply: a non-blank frame inside the confirmed region is worth
    ``nonblank_weight * confirm_weight``. Derived from the **teacher's** argmax, so the
    weighting is a fixed property of the target and cannot be gamed by the student
    predicting blank everywhere to shrink its own loss.
    """
    if teacher_logits.dim() != 3:
        raise ValueError(f"expected (B, T, C) teacher logits, got {tuple(teacher_logits.shape)}")

    batch, frames, _ = teacher_logits.shape
    weights = torch.ones(
        (batch, frames), device=teacher_logits.device, dtype=torch.float32
    )

    is_nonblank = teacher_logits.argmax(dim=-1) != BLANK_ID
    weights = torch.where(is_nonblank, weights * nonblank_weight, weights)

    if confirm_timesteps > 0 and confirm_weight != 1.0:
        span = min(confirm_timesteps, frames)
        weights[:, :span] = weights[:, :span] * confirm_weight

    return weights


def weighted_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Frame-weighted ``KL(teacher || student)``, scaled by ``T^2``.

    Computed in float32 regardless of the autocast dtype: a 43-way softmax KL in bfloat16
    loses enough mantissa on the small-probability tail to make the gradient noisy, and the
    tensor is tiny, so there is no reason to economise here.

    The ``T^2`` factor keeps the gradient magnitude comparable across temperatures (Hinton
    et al.), so temperature can be tuned without re-tuning the loss weights.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_logits.shape)} vs "
            f"teacher {tuple(teacher_logits.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    student = student_logits.float()
    teacher = teacher_logits.float()

    teacher_log_probs = F.log_softmax(teacher / temperature, dim=-1)
    student_log_probs = F.log_softmax(student / temperature, dim=-1)
    per_frame = (
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)

    if weights is None:
        loss = per_frame.mean()
    else:
        if weights.shape != per_frame.shape:
            raise ValueError(
                f"weights {tuple(weights.shape)} do not match frames {tuple(per_frame.shape)}"
            )
        weights = weights.float()
        loss = (per_frame * weights).sum() / weights.sum().clamp_min(1e-8)

    return loss * (temperature**2)


class FeatureProjector(nn.Module):
    """Per-tap linear maps from student width to teacher width.

    One projection per tapped depth rather than a shared one: different depths carry
    different representations, and a single shared map would force them into a common
    basis the student has no reason to have. The projections are training-only scaffolding
    -- they are not exported, so their parameter cost never reaches the device.
    """

    def __init__(self, student_dim: int, teacher_dim: int, num_taps: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(student_dim, teacher_dim) for _ in range(num_taps)
        )

    def forward(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.projections[index](hidden)


def feature_matching_loss(
    student_hidden: tuple[torch.Tensor, ...],
    teacher_hidden: tuple[torch.Tensor, ...],
    projector: FeatureProjector,
    tap_layers: tuple[int, ...] = DEFAULT_TAP_LAYERS,
) -> tuple[torch.Tensor, float]:
    """MSE between projected student hidden states and the teacher's, at tapped depths.

    Returns ``(loss, mean_cosine_similarity)``. Cosine is reported separately as a
    scale-free diagnostic: MSE alone cannot distinguish a student that has the right
    representation at the wrong magnitude from one that has the wrong representation, and
    the first is a much better place to be mid-training.

    Both stacks are at 250 frames pre-adapter, so taps align frame-for-frame with no
    pooling -- unlike ``training.waqf_distill``, whose 20 ms VAD teacher needed 2:1 pooling
    onto the 40 ms lattice.
    """
    if not tap_layers:
        raise ValueError("tap_layers must not be empty")

    total = torch.zeros((), device=student_hidden[0].device, dtype=torch.float32)
    cosine_total = 0.0

    for tap_index, layer in enumerate(tap_layers):
        if layer >= len(student_hidden) or layer >= len(teacher_hidden):
            raise IndexError(
                f"tap layer {layer} out of range for student "
                f"({len(student_hidden)}) / teacher ({len(teacher_hidden)}) hidden states"
            )
        projected = projector(tap_index, student_hidden[layer]).float()
        target = teacher_hidden[layer].float()
        total = total + F.mse_loss(projected, target)
        cosine_total += F.cosine_similarity(projected, target, dim=-1).mean().item()

    return total / len(tap_layers), cosine_total / len(tap_layers)


@dataclass(frozen=True)
class AgreementStats:
    """Student-vs-teacher agreement on one batch. All fractions in [0, 1].

    ``confirmed_agreement`` is the headline number -- it is measured on exactly the
    timesteps that become the user-visible transcript, so it is the one that predicts what
    Muraja will do. ``blank_ratio_*`` exist to catch the failure mode the weighting is
    there to prevent: a student that collapses to all-blank scores high on overall
    agreement (because blank is the majority class) while decoding to nothing.
    """

    overall_agreement: float
    confirmed_agreement: float
    teacher_blank_ratio: float
    student_blank_ratio: float
    nonblank_agreement: float

    @property
    def blank_collapse_margin(self) -> float:
        """How much blankier the student is than the teacher. Near 0 is healthy."""
        return self.student_blank_ratio - self.teacher_blank_ratio

    def is_collapsing(self, threshold: float = 0.15) -> bool:
        """True when the student is predicting substantially more blank than the teacher."""
        return self.blank_collapse_margin > threshold

    def as_dict(self) -> dict:
        return {
            "overall_agreement": round(self.overall_agreement, 4),
            "confirmed_agreement": round(self.confirmed_agreement, 4),
            "nonblank_agreement": round(self.nonblank_agreement, 4),
            "teacher_blank_ratio": round(self.teacher_blank_ratio, 4),
            "student_blank_ratio": round(self.student_blank_ratio, 4),
            "blank_collapse_margin": round(self.blank_collapse_margin, 4),
        }


@torch.no_grad()
def agreement_stats(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    confirm_timesteps: int = CONFIRM_TIMESTEPS,
) -> AgreementStats:
    """Per-frame argmax agreement plus the blank-collapse diagnostic.

    Frame-level, not decoded-string-level: it is cheap enough to run every step and moves
    smoothly, which makes it usable as a training signal. The decoded-string agreement that
    actually gates a release is a separate, more expensive measurement over the deployed
    sliding-window protocol.
    """
    student_ids = student_logits.argmax(dim=-1)
    teacher_ids = teacher_logits.argmax(dim=-1)
    match = student_ids == teacher_ids

    span = min(confirm_timesteps, student_ids.shape[1])
    teacher_nonblank = teacher_ids != BLANK_ID
    nonblank_count = teacher_nonblank.sum()

    return AgreementStats(
        overall_agreement=match.float().mean().item(),
        confirmed_agreement=match[:, :span].float().mean().item(),
        teacher_blank_ratio=(teacher_ids == BLANK_ID).float().mean().item(),
        student_blank_ratio=(student_ids == BLANK_ID).float().mean().item(),
        nonblank_agreement=(
            (match & teacher_nonblank).sum().float() / nonblank_count.clamp_min(1)
        ).item(),
    )


@torch.no_grad()
def teacher_target_sequences(
    teacher_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy-decode the teacher into CTC target sequences: ``(flat_targets, lengths)``.

    Standard CTC collapse of the teacher's argmax -- merge runs of the same token, drop
    blanks -- which is exactly what ``MuaalemInference.scanCTC`` does on device, so the
    targets are the tokens the teacher would actually have put on screen.

    Returned flat (concatenated) rather than padded because that is the form
    ``F.ctc_loss`` takes without needing a padding value that could collide with a real
    class id.
    """
    ids = teacher_logits.argmax(dim=-1)
    sequences = []
    for row in ids:
        previous = -1
        tokens = []
        for token in row.tolist():
            if token != previous and token != BLANK_ID:
                tokens.append(token)
            previous = token
        sequences.append(tokens)

    lengths = torch.tensor(
        [len(s) for s in sequences], dtype=torch.long, device=teacher_logits.device
    )
    flat = torch.tensor(
        [t for s in sequences for t in s], dtype=torch.long, device=teacher_logits.device
    )
    return flat, lengths


def ctc_anchor_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """CTC loss against the teacher's decoded sequence -- the term that breaks blank collapse.

    Frame-weighted KL alone has no mechanism to escape the all-blank basin. It is a
    *per-frame* objective: it can tell the student that frame 61 should have been a qaf,
    but if the student's representation does not yet align phonemes to frames, every
    individual frame is cheapest to answer with blank, and there is no gradient pressure
    toward emitting the token *somewhere*. All-blank is a stable fixed point.

    CTC loss is the standard answer because it is a *sequence* objective. Its
    forward-backward sums over every valid alignment of the target to the input, so the
    student is rewarded for emitting the right tokens in the right order regardless of
    which frames it picks -- and the all-blank output has probability zero under any
    alignment of a non-empty target, so the basin is not a fixed point at all.

    Uses ``zero_infinity=True``: a window whose teacher decode is longer than the 125-frame
    lattice can accommodate has infinite loss, and one such window would otherwise destroy
    the batch's gradient.

    **Reference scale, measured on 32 windows of the Tadabur corpus** (teacher targets
    average 29 tokens over the 125-frame lattice, none empty):

    ==================================  ==========
    student                             ctc loss
    ==================================  ==========
    *saturated* blank (one-hot)          **17.76**
    argmax blank, but soft elsewhere      **~3.3**
    matching the teacher                  **0.19**
    ==================================  ==========

    The middle row is the one that matters and the one the table used to omit, which made
    the other two actively misleading.

    The number is not read the way it looks: 3.3 is much nearer the 0.19 floor than the
    17.76 ceiling, so a student sitting there looks 80% converged while decoding to nothing.
    CTC marginalises over every alignment, so spreading a little probability onto the target
    tokens satisfies it without ever winning an argmax -- the classic non-peaky regime.

    In this work that descent never completed, and pushing on it was the wrong response:
    the term was removed instead. See :class:`DistillLossConfig`.
    """
    log_probs = F.log_softmax(student_logits.float(), dim=-1).transpose(0, 1)
    targets, target_lengths = teacher_target_sequences(teacher_logits)

    batch, frames, _ = student_logits.shape
    input_lengths = torch.full(
        (batch,), frames, dtype=torch.long, device=student_logits.device
    )

    if targets.numel() == 0:
        return torch.zeros((), device=student_logits.device, dtype=torch.float32)

    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=BLANK_ID,
        reduction="mean",
        zero_infinity=True,
    )


def hard_label_ce(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Frame cross-entropy against the teacher's **argmax**, not its distribution.

    Exists because of a measured mismatch between what we optimise and what we score. The
    release gate is decoded-stream agreement, which depends only on the student's argmax;
    the KL term instead asks it to reproduce a temperature-2 *distribution*, spending
    capacity on the probability tail that no metric reads.

    The evidence that this matters: at the end of the first full run the teacher's class sat
    at mean rank **1.25** with top-5 agreement **0.992**, yet frame agreement was 0.882.
    The correct class is almost always present and merely loses the argmax, which is a
    margin failure rather than a representation failure -- and a hard-label objective
    optimises exactly that margin.

    Takes the same frame weighting as :func:`weighted_kl` so the non-blank and
    confirmed-region emphasis carries over unchanged.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_logits.shape)} vs "
            f"teacher {tuple(teacher_logits.shape)}"
        )

    targets = teacher_logits.argmax(dim=-1)
    per_frame = F.cross_entropy(
        student_logits.float().flatten(0, 1),
        targets.flatten(0, 1),
        reduction="none",
    ).view(targets.shape)

    if weights is None:
        return per_frame.mean()
    weights = weights.float()
    return (per_frame * weights).sum() / weights.sum().clamp_min(1e-8)


@dataclass(frozen=True)
class BreakoutStats:
    """How far a blank-collapsed student is from emitting phonemes at all.

    Early in distillation the student parks in the all-blank basin, where
    :class:`AgreementStats` reports ``nonblank_agreement == 0`` and stays there for
    thousands of steps whether the run is healthy or genuinely stuck. Argmax is a step
    function; it cannot distinguish "the right class holds 40% and is about to overtake
    blank" from "the right class holds 0.1%". These are the continuous quantities that can,
    all measured **on teacher-non-blank frames only** -- the frames the student is failing.

    ``target_prob`` is the probability the student puts on the class the teacher chose, and
    ``target_rank`` is where that class sits in the student's own ranking (1 = argmax). A
    run with rank near 2 and target_prob climbing is converging normally; rank in the tens
    with flat target_prob is stuck, and no amount of further patience will fix it.
    """

    target_prob: float          # student P(teacher's argmax class)
    target_rank: float          # 1 = student already agrees
    blank_prob: float           # student P(blank) on those same frames
    top5_agreement: float       # teacher's class within the student's top 5

    @property
    def prob_margin(self) -> float:
        """How much more probability blank holds than the correct class. <=0 means escaped."""
        return self.blank_prob - self.target_prob

    def as_dict(self) -> dict:
        return {
            "target_prob": round(self.target_prob, 4),
            "target_rank": round(self.target_rank, 2),
            "blank_prob": round(self.blank_prob, 4),
            "top5_agreement": round(self.top5_agreement, 4),
            "prob_margin": round(self.prob_margin, 4),
        }


@torch.no_grad()
def breakout_stats(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> BreakoutStats:
    """Continuous distance-from-breakout, over teacher-non-blank frames only."""
    teacher_ids = teacher_logits.argmax(dim=-1)
    nonblank = teacher_ids != BLANK_ID
    if not bool(nonblank.any()):
        return BreakoutStats(0.0, 0.0, 0.0, 0.0)

    probs = F.softmax(student_logits.float(), dim=-1)[nonblank]
    targets = teacher_ids[nonblank]

    target_prob = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # Rank of the target class: 1 + how many classes the student scores strictly higher.
    rank = (probs > target_prob.unsqueeze(1)).sum(dim=1) + 1
    top5 = probs.topk(min(5, probs.shape[1]), dim=-1).indices
    in_top5 = (top5 == targets.unsqueeze(1)).any(dim=1)

    return BreakoutStats(
        target_prob=target_prob.mean().item(),
        target_rank=rank.float().mean().item(),
        blank_prob=probs[:, BLANK_ID].mean().item(),
        top5_agreement=in_top5.float().mean().item(),
    )


@dataclass(frozen=True)
class DistillLossConfig:
    """Weights and knobs for :func:`distillation_loss`."""

    logit_weight: float = 1.0
    # OFF by default. Measured on the full corpus it earned 0.5% of the gradient and scored
    # 0.886x the predict-the-mean baseline -- it was never doing work. Kept as a flag so the
    # ablation stays reproducible, not because the default should ever be 1.0 again.
    feature_weight: float = 0.0
    # Cross-entropy on the teacher's argmax. Off by default: it optimises the metric
    # directly but carries no distributional information, so it is a *finishing* objective
    # for a student that has already escaped the blank basin, not a from-scratch one.
    hard_weight: float = 0.0
    # OFF by default, and the reason is the most expensive finding in this work. This term
    # was added to escape a blank collapse at step 2000, when the real cause was the
    # SpecAugment/dropout determinism bug fixed later. It was never re-examined, and
    # measurement then showed it taking most of the gradient and destabilising training as
    # the data diversified: at 1024 fixed windows, KL-only reached decoded 0.847 while
    # KL+CTC collapsed to 0.000. Removing it moved the full run 84.58% -> 89.37% decoded and
    # 81.0% -> 91.5% gate agreement. See ADR-0010.
    ctc_weight: float = 0.0
    temperature: float = DEFAULT_TEMPERATURE
    nonblank_weight: float = DEFAULT_NONBLANK_WEIGHT
    confirm_weight: float = DEFAULT_CONFIRM_WEIGHT
    confirm_timesteps: int = CONFIRM_TIMESTEPS
    tap_layers: tuple[int, ...] = DEFAULT_TAP_LAYERS

    def __post_init__(self) -> None:
        # Every weight zero makes `total` a constant, and the failure surfaces three
        # minutes later inside backward() as "element 0 of tensors does not require
        # grad" -- after the teacher has loaded and the batches have been fit. It is
        # always a config mistake, so say so before any of that work happens. This
        # bit an ablation script that passed --logit-weight 0 --ctc-weight 0 and
        # relied on feature_weight still defaulting to 1.0, which it no longer does.
        if not any((self.logit_weight, self.feature_weight,
                    self.hard_weight, self.ctc_weight)):
            raise ValueError(
                "every loss weight is zero, so there is nothing to optimise. "
                "Enable at least one of logit_weight, feature_weight, hard_weight, "
                "ctc_weight -- note that only logit_weight defaults to 1.0."
            )


def _round_or_none(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


@dataclass
class DistillLossOutput:
    """The scalar to backprop plus every component, for logging."""

    total: torch.Tensor
    logit_loss: float | None
    feature_loss: float | None
    hard_loss: float | None
    ctc_loss: float | None
    feature_cosine: float | None
    stats: AgreementStats

    def as_dict(self) -> dict:
        return {
            "total": float(self.total.detach()),
            # None, not 0.0, for a term that was not computed. For MSE and CTC zero IS the
            # optimum, so a disabled term logged as 0.0000 reads as a perfect match, and a
            # disabled cosine reads as orthogonal rather than absent.
            "logit_loss": _round_or_none(self.logit_loss, 5),
            "feature_loss": _round_or_none(self.feature_loss, 5),
            "hard_loss": _round_or_none(self.hard_loss, 5),
            "ctc_loss": _round_or_none(self.ctc_loss, 5),
            "feature_cosine": _round_or_none(self.feature_cosine, 4),
            **self.stats.as_dict(),
        }


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_hidden: tuple[torch.Tensor, ...],
    teacher_hidden: tuple[torch.Tensor, ...],
    projector: FeatureProjector,
    config: DistillLossConfig = DistillLossConfig(),
) -> DistillLossOutput:
    """The objective: frame-weighted logit KL, plus any ablation term switched on.

    Only the KL is enabled by default. Terms at weight 0 are skipped rather than multiplied
    by zero -- each costs a real forward -- and report ``None`` rather than 0.0.
    """
    weights = frame_weights(
        teacher_logits,
        nonblank_weight=config.nonblank_weight,
        confirm_timesteps=config.confirm_timesteps,
        confirm_weight=config.confirm_weight,
    )
    # Zero-weighted terms are skipped rather than multiplied by 0. Each still costs a real
    # forward (a 43-way softmax KL, and six projector matmuls for the feature term), which
    # is ~10% of a step that does not need them -- and a finishing run deliberately turns
    # both off. They are still *reported* when computed, because a rising KL under a
    # hard-label objective is informative: it shows the student trading distributional
    # fidelity for the argmax the release gate actually reads.
    if config.logit_weight:
        logit_loss = weighted_kl(
            student_logits, teacher_logits, weights, temperature=config.temperature
        )
        logit_value = float(logit_loss.detach())
    else:
        logit_loss = torch.zeros((), device=student_logits.device)
        logit_value = None

    if config.feature_weight:
        feature_loss, cosine = feature_matching_loss(
            student_hidden, teacher_hidden, projector, config.tap_layers
        )
        feature_value = float(feature_loss.detach())
    else:
        feature_loss = torch.zeros((), device=student_logits.device)
        feature_value, cosine = None, None

    total = config.logit_weight * logit_loss + config.feature_weight * feature_loss

    if config.hard_weight:
        hard = hard_label_ce(student_logits, teacher_logits, weights)
        total = total + config.hard_weight * hard
        hard_value = float(hard.detach())
    else:
        hard_value = None

    if config.ctc_weight:
        ctc = ctc_anchor_loss(student_logits, teacher_logits)
        total = total + config.ctc_weight * ctc
        ctc_value = float(ctc.detach())
    else:
        ctc_value = None

    return DistillLossOutput(
        total=total,
        logit_loss=logit_value,
        feature_loss=feature_value,
        hard_loss=hard_value,
        ctc_loss=ctc_value,
        feature_cosine=cosine,
        stats=agreement_stats(student_logits, teacher_logits, config.confirm_timesteps),
    )

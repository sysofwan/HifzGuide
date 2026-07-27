"""Waqf frame-classification head + detached joint loss (ADR-0004, amends #9).

The Muaalem fine-tune (ADR-0001/0003: LoRA on the phoneme head) is extended with a
per-frame **silence** head distilled from the Recitation VAD. This module owns the
torch side of that head and the joint training loss; the torch-free teacher pooling,
windowing, and the persisted soft-label store live in :mod:`training.waqf_distill`.

Three things are pinned here, matching ADR-0004:

* **The head rides the 40 ms post-adapter lattice — the same place as the phoneme head.**
  :class:`WaqfJointModel` runs the backbone once and reads the post-adapter encoder
  output that the phoneme CTC head reads. The **sifat heads are dropped from the graph**:
  they are never invoked, so they cost no compute and take no gradient.

* **The waqf branch is stop-gradient.** :class:`WaqfHead` detaches its input, so the
  waqf loss contributes **zero** backbone gradient — the isolation claim ADR-0004's
  ablation ladder verifies. The head still exploits the phoneme-tuned sukun / madd cues;
  it just cannot reshape them.

* **The distillation KL is pause-weighted.** Frame KL over the 40 ms lattice is dominated
  by the speech-frame majority, so a head can score high frame accuracy while missing the
  rare silence/boundary frames that *are* the waqf signal. :func:`pause_frame_weights`
  up-weights silence — and, because the teacher posteriors ramp continuously across a
  boundary, boundary frames too — and :func:`pause_collapse_diagnostic` catches a head
  that has collapsed to predicting all-speech.

Runs on Linux + CUDA (see ``tools/environment.yml``); this is the training path, not the
torch-free offline stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# Silence is a small minority of the 40 ms frames, so an un-weighted KL is dominated by
# speech frames. Weighting a fully-silent teacher frame this many times a speech frame
# (linearly interpolated for the soft boundary posteriors) keeps the rare pause/boundary
# frames in the objective. Tunable; the eval slice only tunes the *inference* threshold.
DEFAULT_PAUSE_WEIGHT = 5.0

# The waqf KL is added to the phoneme CTC at this weight. The waqf gradient is detached
# from the backbone regardless, so this only balances how fast the head itself learns.
DEFAULT_WAQF_LOSS_WEIGHT = 1.0

# no-pause-collapse diagnostic: flag the head as collapsed when the teacher has a real
# amount of silence but the head predicts silence on less than this fraction of it.
DEFAULT_COLLAPSE_RATIO = 0.1
DEFAULT_MIN_TARGET_SILENCE_RATE = 0.02


class WaqfHead(nn.Module):
    """Per-frame binary silence classifier on the **detached** 40 ms lattice.

    Emits one silence logit per 40 ms student frame from the post-adapter backbone
    features. The input is detached (stop-gradient), so training this head produces zero
    backbone gradient — the deliberate isolation of ADR-0004. Default is a single linear
    layer (ADR-0004: "a linear binary head on detached Muaalem features"); ``hidden_dim``
    swaps in the small-MLP fallback for when the silence cues are not linearly present.
    """

    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            self.classifier: nn.Module = nn.Linear(feature_dim, 1)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def classify(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``(B, T, feature_dim)`` post-adapter states → ``(B, T)`` silence logits.

        Pure classification, no gradient control — this is the path the exported
        ChunkF traces (inference wants the logits, not the isolation). ``forward``
        wraps it with the training-time stop-gradient. The states are cast to the
        classifier's own dtype first: the backbone runs in bf16 while this head is a
        handful of parameters that stay in fp32 for optimizer precision, so without the
        cast a real joint run dies in ``F.linear`` on mismatched dtypes.
        """
        return self.classifier(hidden_states.to(self._param_dtype())).squeeze(-1)

    def _param_dtype(self) -> torch.dtype:
        return next(self.classifier.parameters()).dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Training path: classify the **detached** states so no gradient reaches
        the backbone — the deliberate isolation of ADR-0004's ablation ladder."""
        return self.classify(hidden_states.detach())


@dataclass
class PhonemeForward:
    """The backbone→phoneme-head result shared by the phoneme-only and joint runs.

    ``phoneme_logits`` ``(B, T, V)`` are the 40 ms lattice logits; ``hidden_states``
    ``(B, T, feature_dim)`` is the **pre-dropout** post-adapter output the waqf head rides;
    ``student_lengths`` ``(B,)`` is each example's valid 40 ms frame count.
    """

    phoneme_logits: torch.Tensor
    hidden_states: torch.Tensor
    student_lengths: torch.Tensor


def phoneme_forward(
    muaalem: nn.Module,
    phoneme_level: str,
    input_features: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> PhonemeForward:
    """One backbone pass → phoneme head, the path both ablation rungs share.

    ADR-0004's ablation ladder needs rung (2) *whole-clip phoneme-only* and rung (3)
    *whole-clip phoneme + detached waqf* to be **bit-identical on the phoneme path** so a
    regression is attributable to the waqf head, not to incidental code drift. Both the
    phoneme-only model (:mod:`training.whole_clip_phoneme`) and :class:`WaqfJointModel`
    compute their phoneme logits here, so that identity is structural, not asserted: the
    waqf head only *adds* a detached branch on ``hidden_states``, it cannot change them.
    """
    if attention_mask is None:
        attention_mask = torch.ones(
            input_features.shape[:2], device=input_features.device, dtype=torch.long
        )
    hidden_states = muaalem.wav2vec2_bert(
        input_features, attention_mask=attention_mask, return_dict=True
    )[0]
    phoneme_logits = muaalem.level_to_lm_head[phoneme_level](muaalem.dropout(hidden_states))
    student_lengths = muaalem._get_feat_extract_output_lengths(
        attention_mask.sum(-1)
    ).to(torch.long)
    return PhonemeForward(phoneme_logits, hidden_states, student_lengths)


@dataclass
class WaqfJointOutput:
    """One backbone forward pass, phoneme + waqf heads, sifat dropped.

    ``phoneme_logits`` ``(B, T, V)`` and ``silence_logits`` ``(B, T)`` share the 40 ms
    lattice. ``student_lengths`` ``(B,)`` is each example's valid 40 ms frame count
    (from the attention mask), so a padded batch masks its CTC and KL to the real frames.
    """

    phoneme_logits: torch.Tensor
    silence_logits: torch.Tensor
    student_lengths: torch.Tensor


class WaqfJointModel(nn.Module):
    """Muaalem backbone + phoneme CTC head + detached waqf head, one forward pass.

    Wraps a :class:`~tadabur.muaalem.modeling_multi_level_ctc.Wav2Vec2BertForMultilevelCTC`
    and runs the backbone once, keeping only the phoneme head and the waqf head. The
    **sifat heads are not invoked** — dropped from the training graph and the export
    target, exactly as ADR-0004 requires; the exported ChunkF ships phoneme + waqf only.
    """

    def __init__(
        self,
        muaalem: nn.Module,
        phoneme_level: str = "phonemes",
        waqf_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if phoneme_level not in muaalem.level_to_lm_head:
            raise ValueError(
                f"phoneme level {phoneme_level!r} not in model heads "
                f"{sorted(muaalem.level_to_lm_head)}"
            )
        self.muaalem = muaalem
        self.phoneme_level = phoneme_level
        feature_dim = muaalem.level_to_lm_head[phoneme_level].in_features
        self.waqf_head = WaqfHead(feature_dim, hidden_dim=waqf_hidden_dim)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> WaqfJointOutput:
        if attention_mask is None:
            attention_mask = torch.ones(
                input_features.shape[:2], device=input_features.device, dtype=torch.long
            )

        # The phoneme path is factored into phoneme_forward so it is bit-identical to the
        # phoneme-only rung (2) run (ADR-0004 isolation). The waqf head only adds a
        # detached branch on the pre-dropout hidden states — the sifat heads never enter
        # the graph.
        forward = phoneme_forward(
            self.muaalem, self.phoneme_level, input_features, attention_mask
        )
        silence_logits = self.waqf_head(forward.hidden_states)

        return WaqfJointOutput(
            phoneme_logits=forward.phoneme_logits,
            silence_logits=silence_logits,
            student_lengths=forward.student_lengths,
        )


def frame_mask_from_lengths(lengths: torch.Tensor, num_frames: int) -> torch.Tensor:
    """``(B,)`` valid-frame counts → ``(B, num_frames)`` boolean validity mask."""
    positions = torch.arange(num_frames, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def pause_frame_weights(
    target_silence: torch.Tensor, pause_weight: float = DEFAULT_PAUSE_WEIGHT
) -> torch.Tensor:
    """Per-frame KL weight, interpolated ``1 → pause_weight`` by silence content.

    A speech frame (``target_silence == 0``) keeps weight 1; a fully-silent frame gets
    ``pause_weight``. Because the pooled teacher posteriors ramp continuously through a
    speech↔silence boundary, boundary frames land between the two — the silence *and*
    boundary weighting ADR-0004 asks for, from a single continuous rule.
    """
    if pause_weight < 1.0:
        raise ValueError(f"pause_weight must be >= 1, got {pause_weight}")
    return 1.0 + (pause_weight - 1.0) * target_silence


def _binary_kl(silence_logits: torch.Tensor, target_silence: torch.Tensor) -> torch.Tensor:
    """Per-frame ``KL(target || student)`` for the binary silence distribution.

    ``KL = CE(target, student) - H(target)``; the entropy term keeps the value a proper
    KL (``>= 0``, ``0`` at a perfect match) for the diagnostic while the gradient matches
    the numerically stable BCE-with-logits path. ``xlogy`` handles the hard ``0``/``1``
    teacher frames (``0 * log 0 == 0``).
    """
    cross_entropy = F.binary_cross_entropy_with_logits(
        silence_logits, target_silence, reduction="none"
    )
    entropy = -(
        torch.special.xlogy(target_silence, target_silence)
        + torch.special.xlogy(1.0 - target_silence, 1.0 - target_silence)
    )
    return (cross_entropy - entropy).clamp_min(0.0)


def waqf_distillation_loss(
    silence_logits: torch.Tensor,
    target_silence: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
    pause_weight: float = DEFAULT_PAUSE_WEIGHT,
) -> torch.Tensor:
    """Pause-weighted distillation KL of the waqf head against the pooled VAD target.

    ``silence_logits`` / ``target_silence`` are ``(B, T)`` on the 40 ms lattice;
    ``target_silence`` is the teacher ``P(silence)`` already pooled 2:1 by
    :mod:`training.waqf_distill`. ``frame_mask`` (``(B, T)`` bool) excludes padded frames.
    Returns the weight-normalised mean KL — a scalar in the same units regardless of how
    much silence a batch happens to contain.
    """
    kl = _binary_kl(silence_logits, target_silence.to(silence_logits.dtype))
    weights = pause_frame_weights(target_silence, pause_weight).to(silence_logits.dtype)
    if frame_mask is not None:
        weights = weights * frame_mask.to(silence_logits.dtype)
    total_weight = weights.sum()
    if total_weight <= 0:
        return silence_logits.new_zeros(())
    return (weights * kl).sum() / total_weight


@dataclass
class PauseCollapseDiagnostic:
    """No-pause-collapse check: is the head still predicting silence at all?

    ``collapsed`` fires when the teacher carries a real amount of silence
    (``target_silence_rate >= min_target_silence_rate``) but the head predicts silence on
    less than ``collapse_ratio`` of that — the failure a speech-dominated KL invites. It
    is a training-health signal, not a product metric (ADR-0004: frame-F1 is never the
    gate); event-level waqf metrics live in the eval slice.
    """

    predicted_silence_rate: float
    target_silence_rate: float
    collapsed: bool


def pause_collapse_diagnostic(
    silence_logits: torch.Tensor,
    target_silence: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
    collapse_ratio: float = DEFAULT_COLLAPSE_RATIO,
    min_target_silence_rate: float = DEFAULT_MIN_TARGET_SILENCE_RATE,
) -> PauseCollapseDiagnostic:
    """Compare predicted vs teacher silence rate over the valid frames."""
    if frame_mask is None:
        frame_mask = torch.ones_like(target_silence, dtype=torch.bool)
    valid = frame_mask.to(torch.bool)
    denom = valid.sum().clamp_min(1)

    predicted_silent = (silence_logits > 0) & valid  # sigmoid(logit) > 0.5
    target_silent = (target_silence >= 0.5) & valid
    predicted_rate = (predicted_silent.sum() / denom).item()
    target_rate = (target_silent.sum() / denom).item()

    collapsed = (
        target_rate >= min_target_silence_rate
        and predicted_rate < collapse_ratio * target_rate
    )
    return PauseCollapseDiagnostic(
        predicted_silence_rate=predicted_rate,
        target_silence_rate=target_rate,
        collapsed=collapsed,
    )


def phoneme_ctc_loss(
    phoneme_logits: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    config,
) -> torch.Tensor:
    """CTC loss on the phoneme head — the ADR-0001/#9 objective, unchanged.

    ``labels`` ``(B, L)`` uses ``-100`` for padding (ignored). Mirrors the multi-level
    model's own CTC call (blank ``= pad_token_id``, ``config`` reduction / zero-infinity)
    so the joint run's phoneme objective is bit-for-bit the phoneme-only baseline's.
    """
    labels_mask = labels >= 0
    target_lengths = labels_mask.sum(-1)
    flattened_targets = labels.masked_select(labels_mask)

    log_probs = F.log_softmax(phoneme_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
    with torch.backends.cudnn.flags(enabled=False):
        return F.ctc_loss(
            log_probs,
            flattened_targets,
            input_lengths,
            target_lengths,
            blank=config.pad_token_id,
            reduction=config.ctc_loss_reduction,
            zero_infinity=config.ctc_zero_infinity,
        )


@dataclass
class JointLoss:
    """The joint training objective and its parts, for logging and go/no-go checks."""

    total: torch.Tensor
    phoneme_ctc: torch.Tensor
    waqf_kl: torch.Tensor
    collapse: PauseCollapseDiagnostic


def joint_loss(
    output: WaqfJointOutput,
    phoneme_labels: torch.Tensor,
    target_silence: torch.Tensor,
    config,
    pause_weight: float = DEFAULT_PAUSE_WEIGHT,
    waqf_loss_weight: float = DEFAULT_WAQF_LOSS_WEIGHT,
) -> JointLoss:
    """Joint loss = phoneme CTC + pause-weighted waqf KL (waqf detached from backbone).

    One forward pass, one data pipeline (ADR-0004): ``output`` carries both heads' logits
    from a single backbone run. ``target_silence`` ``(B, T)`` is the pooled VAD teacher.
    The waqf term is stop-gradient at the backbone by construction (:class:`WaqfHead`), so
    its weight only tunes head learning speed, not the phoneme objective.
    """
    num_frames = output.silence_logits.shape[1]
    frame_mask = frame_mask_from_lengths(output.student_lengths, num_frames)

    phoneme_ctc = phoneme_ctc_loss(
        output.phoneme_logits, phoneme_labels, output.student_lengths, config
    )
    waqf_kl = waqf_distillation_loss(
        output.silence_logits, target_silence, frame_mask, pause_weight
    )
    collapse = pause_collapse_diagnostic(
        output.silence_logits, target_silence, frame_mask
    )
    return JointLoss(
        total=phoneme_ctc + waqf_loss_weight * waqf_kl,
        phoneme_ctc=phoneme_ctc,
        waqf_kl=waqf_kl,
        collapse=collapse,
    )


__all__ = [
    "WaqfHead",
    "WaqfJointModel",
    "WaqfJointOutput",
    "PhonemeForward",
    "phoneme_forward",
    "JointLoss",
    "PauseCollapseDiagnostic",
    "frame_mask_from_lengths",
    "pause_frame_weights",
    "waqf_distillation_loss",
    "pause_collapse_diagnostic",
    "phoneme_ctc_loss",
    "joint_loss",
]

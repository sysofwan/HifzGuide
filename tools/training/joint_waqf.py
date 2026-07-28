"""Whole-clip phoneme + **detached** waqf joint fine-tune — ADR-0004 ablation rung (3).

This is the run rung (3) of ADR-0004's ablation ladder sits on: **whole-clip phoneme +
detached waqf**, on top of the rung-(2) phoneme-only baseline
(:mod:`training.whole_clip_phoneme`). One backbone forward pass feeds two heads — the
phoneme CTC head (the ADR-0001 objective, unchanged) and a per-frame silence head distilled
from the Recitation VAD — and the loss is ``phoneme CTC + pause-weighted waqf KL``. The waqf
head reads **detached** backbone features (:class:`training.waqf_head.WaqfHead`), so the waqf
term contributes **zero** backbone gradient: the phoneme path stays bit-identical to rung (2)
and a regression is attributable to the waqf head, not to code drift.

What this module owns (ADR-0004 D3):

* **The joint run itself.** LoRA on the backbone + a trainable phoneme head (reused from
  rung (2), so the isolation is structural) plus the waqf head, trained on the joint windowed
  batch (:class:`training.windowed_batch.JointWindowedBatch`) — the phoneme window paired with
  its 2:1-pooled VAD silence teacher on the shared window grid.

* **The asserted gradient isolation.** :func:`backbone_isolation_report` backprops the waqf
  KL *alone* and proves every backbone parameter (LoRA adapters + frozen base + phoneme head)
  takes zero gradient from it — the go/no-go (2)→(3) check, asserted at the start of every
  head-only run so a broken detach fails the run rather than silently confounding the eval.

* **The held-out distillation floor.** :func:`evaluate_distillation` scores the waqf head on
  held-out windows (silence-frame F1 against the pooled teacher + the no-pause-collapse
  diagnostic). This is a *distillation* sanity floor, never the product gate — event-level and
  integration evals live in the F/H slices (#34/#35). :func:`train` **fails fast when the
  held-out split has no windows**: without a floor verdict a run could never trigger the
  fallback ladder and would report a false pass, so every successful run must score the floor.

* **The capacity-fallback gate (:mod:`the ladder below`).** A linear head on detached features
  may not reproduce the VAD if the silence cues are not linearly present. When the floor is
  missed, remedies fire **in order** — small MLP head → pause-weighted retune → partial
  backbone unfreeze → blank-run — and any **partial unfreeze re-confounds the isolation**, so
  it re-runs the ablation ladder (#33, E) and the integration eval (#35, H). The ladder is a
  governed policy (:data:`FALLBACK_LADDER`), not a hope: this run refuses to silently execute
  an isolation-breaking stage.

Runs on Linux + CUDA (RTX 5060 Ti, 16 GB, sm_120 — cu128 torch; see ``tools/README.md``).

Usage:
  # verify one real joint windowed batch fits 16 GB
  python -m training.joint_waqf preflight \\
      --audio-dir audit_run/segment_audio_v2 --sample-clip <clip.wav>

  # run the joint detached-waqf fine-tune, emit the #7 phoneme eval + distillation floor
  python -m training.joint_waqf train \\
      --labels windowed_labels.jsonl --soft-labels runs/soft_labels \\
      --audio-dir audit_run/segment_audio_v2 --out-dir runs/rung3 \\
      --eval-segment-manifest audit_run/segment_manifest_v2.jsonl \\
      --eval-audio-dir audit_run/segment_audio_v2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from tadabur.inference import PHONEME_LEVEL
from tadabur.muaalem import Wav2Vec2BertForMultilevelCTC
from training.waqf_distill import DEPLOYED_WINDOW_FEATURE_FRAMES, muaalem_lattice_length
from training.waqf_head import (
    WaqfJointModel,
    frame_mask_from_lengths,
    joint_loss,
    waqf_distillation_loss,
    DEFAULT_COLLAPSE_RATIO,
    DEFAULT_MIN_TARGET_SILENCE_RATE,
    DEFAULT_PAUSE_WEIGHT,
    DEFAULT_WAQF_LOSS_WEIGHT,
)
from training.windowed_batch import (
    JointWindowedBatch,
    JointWindowedCollator,
    JointWindowedExample,
    WindowedCtcCollator,
    WindowedCtcExample,
    length_bucketed_batches,
    load_joint_examples,
)
from training.whole_clip_phoneme import (
    DEFAULT_VRAM_BUDGET_GIB,
    WORST_CASE_WINDOW_FEATURE_FRAMES,
    LoRASettings,
    attach_phoneme_lora,
    base_of,
    enable_gradient_checkpointing,
    load_feature_extractor,
    load_muaalem,
    merge_checkpoint,
    set_seed,
)
from training.windowed_labels import read_labels

# The small-MLP fallback's hidden width (ADR-0004 "small MLP head"). Only used when the
# linear head misses the distillation floor; a knob, not a constant of the linear default.
DEFAULT_MLP_HIDDEN_DIM = 128

# The pause-weight the retune stage raises to (from the linear default) when up-weighting
# silence/boundary frames harder is the next lever after the MLP head (ADR-0004 order).
RETUNE_PAUSE_WEIGHT = 10.0

# Held-out silence-frame F1 the detached head must clear to pass the distillation floor.
# This is a distillation sanity floor (ADR-0004: frame metrics are never the product gate);
# the event-level / integration gates live in #34 / #35.
DEFAULT_MIN_SILENCE_F1 = 0.5


# --- capacity-fallback ladder (the governed gate) ----------------------------


@dataclass(frozen=True)
class FallbackStage:
    """One rung of ADR-0004's capacity-fallback ladder for the detached waqf head.

    A linear head on *detached* Muaalem features may not reproduce the VAD if the silence
    cues are not linearly present, so the floor miss escalates through these stages **in
    order**. ``waqf_hidden_dim`` / ``pause_weight`` parameterize the head-only remedies this
    run can execute while isolation still holds; ``backbone_unfreeze_layers`` > 0 is the
    deliberate second backbone objective that **breaks** isolation (``requires_reablation``);
    ``is_blank_run`` is the terminal drop-the-head reference baseline (ADR-0002/0004).
    """

    name: str
    description: str
    waqf_hidden_dim: Optional[int]
    pause_weight: float
    backbone_unfreeze_layers: int
    is_blank_run: bool

    @property
    def breaks_isolation(self) -> bool:
        return self.backbone_unfreeze_layers > 0

    @property
    def requires_reablation(self) -> bool:
        """A partial unfreeze re-confounds the eval → re-run E (#33) and H (#35)."""
        return self.breaks_isolation

    @property
    def is_head_only(self) -> bool:
        """Head-only stages keep the detach; this run executes them and asserts isolation."""
        return not self.breaks_isolation and not self.is_blank_run


FALLBACK_LADDER: tuple[FallbackStage, ...] = (
    FallbackStage(
        name="linear",
        description="Linear silence head on detached 40 ms features (ADR-0004 default).",
        waqf_hidden_dim=None,
        pause_weight=DEFAULT_PAUSE_WEIGHT,
        backbone_unfreeze_layers=0,
        is_blank_run=False,
    ),
    FallbackStage(
        name="mlp_head",
        description="Small MLP silence head, still detached — the first floor-miss remedy.",
        waqf_hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
        pause_weight=DEFAULT_PAUSE_WEIGHT,
        backbone_unfreeze_layers=0,
        is_blank_run=False,
    ),
    FallbackStage(
        name="pause_weight_retune",
        description="MLP head + harder silence/boundary up-weighting (pause-weighted retune).",
        waqf_hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
        pause_weight=RETUNE_PAUSE_WEIGHT,
        backbone_unfreeze_layers=0,
        is_blank_run=False,
    ),
    FallbackStage(
        name="partial_unfreeze",
        description=(
            "Partial backbone unfreeze — a deliberate second backbone objective that BREAKS "
            "isolation; re-runs the ablation ladder (#33) and the integration eval (#35)."
        ),
        waqf_hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
        pause_weight=RETUNE_PAUSE_WEIGHT,
        backbone_unfreeze_layers=2,
        is_blank_run=False,
    ),
    FallbackStage(
        name="blank_run",
        description="Terminal: drop the head, fall back to the CTC blank-run reference.",
        waqf_hidden_dim=None,
        pause_weight=DEFAULT_PAUSE_WEIGHT,
        backbone_unfreeze_layers=0,
        is_blank_run=True,
    ),
)

_LADDER_BY_NAME = {stage.name: stage for stage in FALLBACK_LADDER}


def stage_by_name(name: str) -> FallbackStage:
    try:
        return _LADDER_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown fallback stage {name!r}; the ladder is "
            f"{[s.name for s in FALLBACK_LADDER]}."
        ) from exc


def next_fallback(current: FallbackStage) -> Optional[FallbackStage]:
    """The next rung after ``current`` when the floor is missed, or ``None`` at the end."""
    index = FALLBACK_LADDER.index(current)
    following = FALLBACK_LADDER[index + 1 :]
    return following[0] if following else None


# --- config ------------------------------------------------------------------


@dataclass
class JointTrainConfig:
    """Joint run hyperparameters: the rung-(2) budget plus the waqf head + its floor."""

    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_accum_steps: int = 8
    max_frames_per_batch: int = 1000
    max_windows_per_batch: int = 8
    seed: int = 0
    waqf_loss_weight: float = DEFAULT_WAQF_LOSS_WEIGHT
    min_silence_f1: float = DEFAULT_MIN_SILENCE_F1
    stage_name: str = "linear"
    held_out_split: str = "val"
    lora: LoRASettings = field(default_factory=LoRASettings)

    @property
    def stage(self) -> FallbackStage:
        return stage_by_name(self.stage_name)


# --- joint model assembly ----------------------------------------------------


def freeze_adapter_layerdrop(base: Wav2Vec2BertForMultilevelCTC) -> None:
    """Force the backbone adapter's layerdrop to 0 — the frozen windowing contract.

    In training mode the Wav2Vec2-BERT adapter stochastically drops its (single) conformer
    layer with probability ``config.layerdrop``; a dropped layer skips the 2:1 subsampling, so
    the lattice length jumps from the frozen ``muaalem_lattice_length`` to the un-downsampled
    frame count. ADR-0004's frozen windowing contract requires that lattice length to be
    deterministic: the pooled VAD silence target is pre-aligned to it, and a length mismatch
    would silently corrupt the joint loss. Since rung (3) trains with the backbone **frozen**
    (LoRA + heads only), layerdrop is meaningless regularization here anyway. The production
    ``muaalem-model-v3_2`` checkpoint already ships ``layerdrop=0.0``; this makes the guarantee
    explicit and independent of the loaded config.
    """
    base.config.layerdrop = 0.0
    adapter = getattr(base.wav2vec2_bert, "adapter", None)
    if adapter is not None:
        adapter.layerdrop = 0.0


def build_joint_model(peft_model, stage: FallbackStage) -> WaqfJointModel:
    """Wrap the LoRA-adapted base model with the phoneme + detached waqf heads.

    Runs the shared phoneme forward on the *same* LoRA-adapted base the phoneme-only rung
    uses (:func:`training.whole_clip_phoneme.base_of`) and adds the stage's waqf head — linear
    by default, the small MLP for the fallback. The sifat heads are never invoked. The waqf
    head's parameters are trainable; the backbone stays frozen except the LoRA adapters and
    the phoneme head, so the only gradient reaching the backbone is the phoneme CTC's.

    Adapter layerdrop is forced off (:func:`freeze_adapter_layerdrop`) so the lattice length is
    deterministic under the frozen windowing contract.
    """
    base = base_of(peft_model)
    freeze_adapter_layerdrop(base)
    return WaqfJointModel(base, waqf_hidden_dim=stage.waqf_hidden_dim)


def _backbone_parameters(joint_model: WaqfJointModel):
    """Every joint-model parameter except the waqf head's — the isolation target set."""
    waqf_ids = {id(p) for p in joint_model.waqf_head.parameters()}
    for _, param in joint_model.named_parameters():
        if id(param) not in waqf_ids:
            yield param


# --- gradient-isolation assertion (the go/no-go (2)→(3) check) ----------------


@dataclass
class IsolationReport:
    """Whether the waqf KL alone leaves the backbone gradient untouched."""

    isolated: bool
    max_backbone_grad: float
    num_backbone_params_with_grad: int


def backbone_isolation_report(
    joint_model: WaqfJointModel,
    batch: JointWindowedBatch,
    pause_weight: float,
    waqf_loss_weight: float = DEFAULT_WAQF_LOSS_WEIGHT,
) -> IsolationReport:
    """Backprop the waqf KL **alone** and measure the backbone gradient it induces.

    ADR-0004's isolation claim is that the detached waqf head contributes zero backbone
    gradient, so rung (3) shares the phoneme path with rung (2). This proves it directly:
    with the phoneme CTC term removed, a correctly-detached head must leave every backbone
    parameter (LoRA adapters, frozen base, phoneme head) with no gradient. Any non-zero
    backbone gradient means the detach is broken and the isolation claim is false.
    """
    joint_model.zero_grad(set_to_none=True)
    output = joint_model(batch.phoneme.input_features, batch.phoneme.attention_mask)
    frame_mask = frame_mask_from_lengths(
        output.student_lengths, output.silence_logits.shape[1]
    )
    kl = waqf_distillation_loss(
        output.silence_logits, batch.target_silence, frame_mask, pause_weight
    )
    (waqf_loss_weight * kl).backward()

    max_grad = 0.0
    num_with_grad = 0
    for param in _backbone_parameters(joint_model):
        if param.grad is not None:
            grad_max = float(param.grad.abs().max())
            if grad_max > 0.0:
                num_with_grad += 1
                max_grad = max(max_grad, grad_max)
    joint_model.zero_grad(set_to_none=True)
    return IsolationReport(
        isolated=num_with_grad == 0,
        max_backbone_grad=max_grad,
        num_backbone_params_with_grad=num_with_grad,
    )


def assert_backbone_isolation(
    joint_model: WaqfJointModel,
    batch: JointWindowedBatch,
    pause_weight: float,
    waqf_loss_weight: float = DEFAULT_WAQF_LOSS_WEIGHT,
) -> IsolationReport:
    """Raise unless the waqf KL leaves the backbone gradient untouched (head-only stages)."""
    report = backbone_isolation_report(
        joint_model, batch, pause_weight, waqf_loss_weight
    )
    if not report.isolated:
        raise AssertionError(
            f"waqf KL induced backbone gradient (max {report.max_backbone_grad:.3e} over "
            f"{report.num_backbone_params_with_grad} params) — the detach is broken, so the "
            "rung (2)↔(3) isolation ADR-0004 requires does not hold. Fix WaqfHead.forward."
        )
    return report


# --- held-out distillation floor ---------------------------------------------


@dataclass
class DistillationReport:
    """Held-out distillation health of the waqf head (a floor, never the product gate)."""

    silence_f1: float
    mean_kl: float
    predicted_silence_rate: float
    target_silence_rate: float
    collapsed: bool
    min_silence_f1: float
    meets_floor: bool


@dataclass(frozen=True)
class SilenceCounts:
    """Aggregated per-frame silence confusion + weighted KL over a held-out set."""

    true_positive: float
    predicted_silent: float
    target_silent: float
    valid_frames: float
    kl_sum: float
    kl_weight: float


def _f1_from_counts(
    true_positive: float, predicted_positive: float, actual_positive: float
) -> float:
    """F1 from confusion counts; 1.0 iff both predicted and actual silence are empty."""
    if predicted_positive == 0.0 or actual_positive == 0.0:
        return 1.0 if predicted_positive == 0.0 and actual_positive == 0.0 else 0.0
    precision = true_positive / predicted_positive
    recall = true_positive / actual_positive
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def distillation_verdict(
    counts: SilenceCounts, min_silence_f1: float = DEFAULT_MIN_SILENCE_F1
) -> DistillationReport:
    """Turn aggregated silence counts into the floor verdict — the pure gate logic.

    Silence-frame F1 must clear ``min_silence_f1`` **and** the head must not have collapsed
    to all-speech (:mod:`training.waqf_head`'s canonical rate rule). Kept pure of any tensor
    or model so the floor logic is unit-testable without a GPU (ADR-0004: the floor is a
    distillation sanity check, never the product gate).
    """
    f1 = _f1_from_counts(
        counts.true_positive, counts.predicted_silent, counts.target_silent
    )
    predicted_rate = counts.predicted_silent / max(counts.valid_frames, 1.0)
    target_rate = counts.target_silent / max(counts.valid_frames, 1.0)
    collapsed = (
        target_rate >= DEFAULT_MIN_TARGET_SILENCE_RATE
        and predicted_rate < DEFAULT_COLLAPSE_RATIO * target_rate
    )
    return DistillationReport(
        silence_f1=round(f1, 4),
        mean_kl=round(counts.kl_sum / max(counts.kl_weight, 1.0), 6),
        predicted_silence_rate=round(predicted_rate, 4),
        target_silence_rate=round(target_rate, 4),
        collapsed=collapsed,
        min_silence_f1=min_silence_f1,
        meets_floor=f1 >= min_silence_f1 and not collapsed,
    )


@torch.no_grad()
def evaluate_distillation(
    joint_model: WaqfJointModel,
    batches: list[list[JointWindowedExample]],
    collate: JointWindowedCollator,
    device: torch.device,
    dtype: torch.dtype,
    pause_weight: float,
    min_silence_f1: float = DEFAULT_MIN_SILENCE_F1,
) -> DistillationReport:
    """Score the waqf head against the pooled teacher on held-out windows → the floor check.

    Aggregates the silence confusion counts and the weighted KL across ``batches`` (frame-mask
    so padded frames never score), then delegates the F1 / no-collapse / floor verdict to
    :func:`distillation_verdict`. This is a distillation sanity check on the *frame* signal —
    the event-level and integration gates are #34 / #35, never this.
    """
    joint_model.eval()
    true_positive = predicted_silent = target_silent = valid_frames = 0.0
    kl_sum = kl_weight = 0.0
    for examples in batches:
        batch = collate(examples).to(device, dtype)
        output = joint_model(batch.phoneme.input_features, batch.phoneme.attention_mask)
        frame_mask = frame_mask_from_lengths(
            output.student_lengths, output.silence_logits.shape[1]
        )
        valid = frame_mask.to(torch.bool)
        predicted = (output.silence_logits > 0) & valid
        target = (batch.target_silence >= 0.5) & valid
        true_positive += float((predicted & target).sum())
        predicted_silent += float(predicted.sum())
        target_silent += float(target.sum())
        valid_frames += float(valid.sum())

        kl = waqf_distillation_loss(
            output.silence_logits, batch.target_silence, frame_mask, pause_weight
        )
        kl_sum += float(kl) * float(valid.sum())
        kl_weight += float(valid.sum())

    joint_model.train()
    return distillation_verdict(
        SilenceCounts(
            true_positive=true_positive,
            predicted_silent=predicted_silent,
            target_silent=target_silent,
            valid_frames=valid_frames,
            kl_sum=kl_sum,
            kl_weight=kl_weight,
        ),
        min_silence_f1,
    )


# --- memory preflight (joint forward + backward) -----------------------------


@dataclass
class JointPreflightReport:
    """Peak VRAM of one real worst-case joint forward/backward vs the 16 GB budget."""

    num_windows: int
    window_feature_frames: int
    peak_allocated_gib: float
    peak_reserved_gib: float
    budget_gib: float
    fits: bool


def _worst_case_joint_batch(
    feature_extractor,
    sample_waveform: np.ndarray,
    num_windows: int,
    window_feature_frames: int,
) -> list[JointWindowedExample]:
    """A batch of ``num_windows`` maximal-length joint windows — the memory worst case.

    Each window is ``window_feature_frames`` of the (real) ``sample_waveform`` at a maximal
    CTC target and a full-length (all-speech) silence teacher on the 40 ms lattice, so the
    batch reproduces the largest phoneme *and* waqf activation shapes a real batch can hit.
    """
    from tadabur.audio import TARGET_SAMPLE_RATE
    from tadabur.phoneme_vocab import NUM_PHONEME_CLASSES

    samples = window_feature_frames * (TARGET_SAMPLE_RATE * 20 // 1000)
    wave = np.asarray(sample_waveform, dtype=np.float32)
    if len(wave) < samples:
        wave = np.tile(wave, int(np.ceil(samples / max(len(wave), 1))))
    window_audio = wave[:samples]
    logit_frames = muaalem_lattice_length(window_feature_frames)
    label_ids = tuple(int(i % NUM_PHONEME_CLASSES) or 1 for i in range(logit_frames // 2))
    return [
        JointWindowedExample(
            ctc=WindowedCtcExample(
                key=("__preflight__", i),
                audio=window_audio,
                label_ids=label_ids,
                start_sample=0,
                num_samples=samples,
                feature_frames=window_feature_frames,
                logit_frames=logit_frames,
            ),
            target_silence=np.zeros(logit_frames, dtype=np.float32),
        )
        for i in range(num_windows)
    ]


def preflight_joint_batch_memory(
    peft_model,
    feature_extractor,
    sample_waveform: np.ndarray,
    num_windows: int,
    device: torch.device,
    stage: FallbackStage,
    dtype: torch.dtype = torch.bfloat16,
    window_feature_frames: int = WORST_CASE_WINDOW_FEATURE_FRAMES,
    budget_gib: float = DEFAULT_VRAM_BUDGET_GIB,
    pause_weight: Optional[float] = None,
) -> JointPreflightReport:
    """Run one real worst-case **joint** forward+backward and measure peak VRAM.

    ADR-0004 requires verifying a real batch fits before committing, and the joint batch is
    the binding case (both heads live in one pass). Builds the largest joint batch the
    contract allows, runs the same bf16 + gradient-checkpointed joint forward and
    ``phoneme CTC + waqf KL`` backward the training step uses, and reports peak VRAM against
    ``budget_gib``. Raises only off a CUDA device (the measurement is meaningless there).
    """
    if device.type != "cuda":
        raise RuntimeError("preflight measures CUDA memory; run it on the GPU.")

    base = base_of(peft_model)
    enable_gradient_checkpointing(base)
    joint_model = build_joint_model(peft_model, stage).to(device)
    joint_model.train()

    collate = JointWindowedCollator(feature_extractor)
    batch = collate(
        _worst_case_joint_batch(
            feature_extractor, sample_waveform, num_windows, window_feature_frames
        )
    ).to(device, dtype)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    output = joint_model(batch.phoneme.input_features, batch.phoneme.attention_mask)
    loss = joint_loss(
        output,
        batch.phoneme.labels,
        batch.target_silence,
        base.config,
        pause_weight=pause_weight if pause_weight is not None else stage.pause_weight,
        waqf_loss_weight=DEFAULT_WAQF_LOSS_WEIGHT,
    )
    loss.total.backward()
    joint_model.zero_grad(set_to_none=True)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    return JointPreflightReport(
        num_windows=num_windows,
        window_feature_frames=window_feature_frames,
        peak_allocated_gib=round(peak_allocated, 3),
        peak_reserved_gib=round(peak_reserved, 3),
        budget_gib=budget_gib,
        fits=peak_reserved <= budget_gib,
    )


# --- training loop -----------------------------------------------------------


@dataclass
class EpochStats:
    """One epoch's mean total / phoneme-CTC / waqf-KL loss — the joint convergence log."""

    epoch: int
    total_loss: float
    phoneme_ctc: float
    waqf_kl: float


@dataclass
class JointRunReport:
    """The rung-(3) run outcome: convergence, isolation, and the floor/fallback verdict."""

    stage_name: str
    trace: list[EpochStats]
    isolation: IsolationReport
    distillation: DistillationReport
    recommended_fallback: Optional[str]
    requires_reablation: bool


def _has_split(labels_path: Path, split: str) -> bool:
    return bool(read_labels(labels_path).get(split))


def train(
    labels_path: Path,
    audio_dir: Path,
    soft_label_root: Path,
    out_dir: Path,
    config: JointTrainConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> JointRunReport:
    """Run the joint detached-waqf fine-tune; return convergence + the floor/fallback verdict.

    Trains the LoRA adapters, phoneme head, and the detached waqf head under the joint loss,
    asserts the backbone gradient isolation up front (a head-only stage must keep the detach),
    scores the held-out distillation floor, and — if the floor is missed — recommends the next
    fallback rung. Saves the LoRA adapters, the waqf head, the loss trace, and the run report
    under ``out_dir``. Deterministic for a given ``config.seed``.

    The issue acceptance requires every successful run to produce a held-out floor verdict, so
    a missing/empty ``config.held_out_split`` **fails fast**: without held-out windows the run
    could never trigger the fallback ladder and would report a false pass. Point the labels
    build at a split (``config.held_out_split``, default ``"val"``) that actually carries
    windows.
    """
    stage = config.stage
    if not stage.is_head_only:
        raise ValueError(
            f"stage {stage.name!r} is not head-only ({stage.description}); this run executes "
            "only the detached head-only rungs. An unfreeze must go through the re-ablation "
            "path (#33/#35) and the blank-run is a scorer-side reference, not a training run."
        )
    if not _has_split(labels_path, config.held_out_split):
        raise ValueError(
            f"labels {labels_path} has no {config.held_out_split!r} split, so the detached "
            "joint run has no held-out windows to score the waqf distillation floor on. The "
            "issue acceptance requires every successful run to produce a floor verdict (and "
            "trigger the fallback ladder on failure) — regenerate the labels with a held-out "
            f"split or pass --held-out-split naming one (have "
            f"{sorted(read_labels(labels_path))})."
        )
    set_seed(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = load_feature_extractor()
    collate = JointWindowedCollator(feature_extractor)
    train_examples = load_joint_examples(labels_path, audio_dir, soft_label_root, "train")
    val_examples = load_joint_examples(
        labels_path, audio_dir, soft_label_root, config.held_out_split
    )

    peft_model = attach_phoneme_lora(load_muaalem(dtype=dtype), config.lora).to(device)
    base = base_of(peft_model)
    enable_gradient_checkpointing(base)
    joint_model = build_joint_model(peft_model, stage).to(device)
    joint_model.train()

    optimizer = torch.optim.AdamW(
        (p for p in joint_model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    val_batches = length_bucketed_batches(
        val_examples, config.max_frames_per_batch, config.max_windows_per_batch, config.seed
    )
    if not val_batches:
        raise ValueError(
            f"the {config.held_out_split!r} split produced no held-out batches, so the waqf "
            "distillation floor cannot be scored — the run would falsely pass. Check the "
            "labels/soft-labels/audio inputs for that split."
        )

    isolation: Optional[IsolationReport] = None
    trace: list[EpochStats] = []
    for epoch in range(config.epochs):
        batches = length_bucketed_batches(
            train_examples,
            config.max_frames_per_batch,
            config.max_windows_per_batch,
            config.seed + epoch,
        )
        totals = {"total": 0.0, "phoneme_ctc": 0.0, "waqf_kl": 0.0}
        windows = 0
        optimizer.zero_grad(set_to_none=True)
        # See :mod:`training.whole_clip_phoneme` — a corpus epoch runs for hours, so the
        # bar carries the rate/ETA the single per-epoch summary line cannot.
        progress = tqdm(
            batches, desc=f"epoch {epoch}/{config.epochs - 1}", unit="batch", leave=False
        )
        for step, examples in enumerate(progress):
            batch = collate(examples).to(device, dtype)
            if isolation is None:
                # Prove the detach before the first optimizer step touches the backbone.
                isolation = assert_backbone_isolation(
                    joint_model, batch, stage.pause_weight, config.waqf_loss_weight
                )
            output = joint_model(batch.phoneme.input_features, batch.phoneme.attention_mask)
            loss = joint_loss(
                output,
                batch.phoneme.labels,
                batch.target_silence,
                base.config,
                pause_weight=stage.pause_weight,
                waqf_loss_weight=config.waqf_loss_weight,
            )
            (loss.total / config.grad_accum_steps).backward()
            totals["total"] += float(loss.total.item())
            totals["phoneme_ctc"] += float(loss.phoneme_ctc.item())
            totals["waqf_kl"] += float(loss.waqf_kl.item())
            windows += len(examples)
            if (step + 1) % config.grad_accum_steps == 0 or step + 1 == len(batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(
                ctc=f"{totals['phoneme_ctc'] / max(windows, 1):.4f}",
                kl=f"{totals['waqf_kl'] / max(windows, 1):.4f}",
            )
        progress.close()

        stats = EpochStats(
            epoch,
            totals["total"] / max(windows, 1),
            totals["phoneme_ctc"] / max(windows, 1),
            totals["waqf_kl"] / max(windows, 1),
        )
        trace.append(stats)
        print(
            f"epoch {epoch}: total {stats.total_loss:.4f}  ctc {stats.phoneme_ctc:.4f}  "
            f"waqf_kl {stats.waqf_kl:.4f}",
            flush=True,
        )

    if isolation is None:
        raise ValueError("no training batches — check the labels/soft-labels/audio inputs.")

    distillation = evaluate_distillation(
        joint_model, val_batches, collate, device, dtype,
        stage.pause_weight, config.min_silence_f1,
    )
    fallback = next_fallback(stage) if not distillation.meets_floor else None

    peft_model.save_pretrained(out_dir / "lora_adapter")
    torch.save(joint_model.waqf_head.state_dict(), out_dir / "waqf_head.pt")
    report = JointRunReport(
        stage_name=stage.name,
        trace=trace,
        isolation=isolation,
        distillation=distillation,
        recommended_fallback=fallback.name if fallback else None,
        requires_reablation=bool(fallback and fallback.requires_reablation),
    )
    _write_report(out_dir, report)
    if fallback is not None:
        print(
            f"distillation floor missed (F1 {distillation.silence_f1} < "
            f"{config.min_silence_f1}); next fallback rung: {fallback.name} — "
            f"{fallback.description}"
        )
    return report


def _write_report(out_dir: Path, report: JointRunReport) -> None:
    payload = {
        "stage_name": report.stage_name,
        "trace": [asdict(s) for s in report.trace],
        "isolation": asdict(report.isolation),
        "distillation": asdict(report.distillation),
        "recommended_fallback": report.recommended_fallback,
        "requires_reablation": report.requires_reablation,
    }
    (out_dir / "joint_run_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def emit_eval_report(
    merged_dir: Path,
    segment_manifest: Path,
    eval_audio_dir: Path,
    out_path: Path,
) -> None:
    """Score the merged rung-(3) phoneme checkpoint with the two-sided #7 harness.

    The waqf head does not enter the #7 harness (it is phoneme should-accept / should-reject);
    the waqf head's floor is the distillation report above and its product gate is #34 / #35.
    This scores the merged phoneme path — the rung-(3) numbers the ablation ladder (#33)
    compares against rung (2).
    """
    from tadabur.eval_harness import run_eval

    report = run_eval(segment_manifest, eval_audio_dir, model_id=str(merged_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote rung-(3) phoneme eval report to {out_path}")


# --- CLI ---------------------------------------------------------------------


def _sample_waveform(audio_dir: Path, sample_clip: Optional[str]) -> np.ndarray:
    from tadabur.audio import decode_to_mono_16k

    if sample_clip:
        path = audio_dir / sample_clip
    else:
        clips = sorted(audio_dir.glob("*.wav"))
        if not clips:
            raise FileNotFoundError(f"no .wav clips under {audio_dir} for the preflight.")
        path = clips[0]
    return decode_to_mono_16k(path.read_bytes())


def _cmd_preflight(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    stage = stage_by_name(args.stage)
    peft_model = attach_phoneme_lora(load_muaalem(), LoRASettings()).to(device)
    report = preflight_joint_batch_memory(
        peft_model,
        load_feature_extractor(),
        _sample_waveform(args.audio_dir, args.sample_clip),
        num_windows=args.num_windows,
        device=device,
        stage=stage,
        budget_gib=args.budget_gib,
    )
    print(json.dumps(asdict(report), indent=2))
    if not report.fits:
        raise SystemExit(
            f"joint batch of {report.num_windows} windows peaks at "
            f"{report.peak_reserved_gib} GiB, over the {report.budget_gib} GiB budget — "
            "lower --num-windows or add checkpointing."
        )


def _cmd_train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = JointTrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        grad_accum_steps=args.grad_accum_steps,
        max_frames_per_batch=args.max_frames_per_batch,
        max_windows_per_batch=args.max_windows_per_batch,
        seed=args.seed,
        waqf_loss_weight=args.waqf_loss_weight,
        min_silence_f1=args.min_silence_f1,
        stage_name=args.stage,
        held_out_split=args.held_out_split,
        lora=LoRASettings(rank=args.lora_rank, alpha=args.lora_alpha),
    )
    train(args.labels, args.audio_dir, args.soft_labels, args.out_dir, config, device)
    if args.eval_segment_manifest and args.eval_audio_dir:
        merged_dir = merge_checkpoint(args.out_dir)
        emit_eval_report(
            merged_dir,
            args.eval_segment_manifest,
            args.eval_audio_dir,
            args.out_dir / "eval_rung3.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight", help="verify one real joint windowed batch fits 16 GB")
    pre.add_argument("--audio-dir", type=Path, required=True)
    pre.add_argument("--sample-clip", type=str, default=None,
                     help="clip filename under --audio-dir; defaults to the first .wav.")
    pre.add_argument("--num-windows", type=int, default=JointTrainConfig.max_windows_per_batch)
    pre.add_argument("--budget-gib", type=float, default=DEFAULT_VRAM_BUDGET_GIB)
    pre.add_argument("--stage", type=str, default="linear",
                     help=f"fallback rung: {[s.name for s in FALLBACK_LADDER]}.")
    pre.set_defaults(func=_cmd_preflight)

    tr = sub.add_parser("train", help="run the joint detached-waqf fine-tune")
    tr.add_argument("--labels", type=Path, required=True,
                    help="windowed CTC labels JSONL (training.windowed_labels).")
    tr.add_argument("--soft-labels", type=Path, required=True,
                    help="soft-label store root (training.waqf_distill, recitation grid).")
    tr.add_argument("--audio-dir", type=Path, required=True)
    tr.add_argument("--out-dir", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=JointTrainConfig.epochs)
    tr.add_argument("--learning-rate", type=float, default=JointTrainConfig.learning_rate)
    tr.add_argument("--grad-accum-steps", type=int, default=JointTrainConfig.grad_accum_steps)
    tr.add_argument("--max-frames-per-batch", type=int,
                    default=JointTrainConfig.max_frames_per_batch)
    tr.add_argument("--max-windows-per-batch", type=int,
                    default=JointTrainConfig.max_windows_per_batch)
    tr.add_argument("--seed", type=int, default=JointTrainConfig.seed)
    tr.add_argument("--waqf-loss-weight", type=float, default=DEFAULT_WAQF_LOSS_WEIGHT)
    tr.add_argument("--min-silence-f1", type=float, default=DEFAULT_MIN_SILENCE_F1)
    tr.add_argument("--held-out-split", type=str, default=JointTrainConfig.held_out_split,
                    help="labels split scored for the distillation floor; must carry windows.")
    tr.add_argument("--stage", type=str, default="linear",
                    help=f"head-only fallback rung: {[s.name for s in FALLBACK_LADDER if s.is_head_only]}.")
    tr.add_argument("--lora-rank", type=int, default=LoRASettings.rank,
                    help="LoRA rank; must match the rung-(2) run this rung is compared against "
                         "in the #33 ablation ladder.")
    tr.add_argument("--lora-alpha", type=int, default=LoRASettings.alpha,
                    help="LoRA alpha; see --lora-rank.")
    tr.add_argument("--eval-segment-manifest", type=Path, default=None,
                    help="segment manifest for the #7 phoneme eval (with --eval-audio-dir).")
    tr.add_argument("--eval-audio-dir", type=Path, default=None)
    tr.set_defaults(func=_cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

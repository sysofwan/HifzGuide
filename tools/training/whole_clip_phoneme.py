"""Whole-clip phoneme-only LoRA fine-tune — ADR-0004 ablation rung (2), the ADR gate.

ADR-0004 extends the Muaalem phoneme fine-tune (ADR-0001/0003) with a waqf head, but moving
from the segmented ADR-0001 fine-tune to **fixed 5 s windows over the whole recitation**
already changes batch shapes, padding, loss normalization and RNG order. So the ablation
ladder inserts this rung between them: **whole-clip phoneme-only**, no waqf head. It must land
*before* the waqf head so a later regression is attributable to the whole-clip move (this rung
vs the ADR-0001 baseline) rather than to the waqf head (rung (3) vs this rung). This module is
that run.

What it pins:

* **LoRA on the backbone, phoneme head trainable, sifat heads dropped.** ADR-0004: "LoRA on
  the phoneme head … backbone base weights frozen, drift bounded by construction." LoRA
  adapters ride the encoder attention projections (the bounded backbone drift), the phoneme
  head trains in full, and the sifat heads take no gradient — :func:`attach_phoneme_lora`.
  The exported model ships phoneme(+waqf) only.

* **The phoneme forward is shared with the joint rung.** Training runs
  :func:`training.waqf_head.phoneme_forward`, the exact path :class:`WaqfJointModel` uses, so
  rung (2) and rung (3) are bit-identical on the phoneme path *by construction* — the
  isolation ADR-0004's go/no-go (#33) verifies.

* **The 16 GB budget is verified, not assumed.** :func:`preflight_batch_memory` builds one
  real worst-case windowed batch, runs a bf16 + gradient-checkpointed forward/backward, and
  asserts peak VRAM stays under budget before any run commits — ADR-0004's OOM mitigation.

* **Eval is the two-sided #7 harness.** After training the LoRA adapters are merged into a
  full checkpoint and scored by :mod:`tadabur.eval_harness` (should-accept recall /
  should-reject discrimination) — the rung-(2) numbers the ladder (#33) compares.

Runs on Linux + CUDA (RTX 5060 Ti, 16 GB, sm_120 — cu128 torch; see ``tools/README.md``).

Usage:
  # verify one real windowed batch fits 16 GB
  python -m training.whole_clip_phoneme preflight \\
      --audio-dir audit_run/segment_audio_v2 --sample-clip <clip.wav>

  # run the whole-clip phoneme-only fine-tune and emit the #7 eval report
  python -m training.whole_clip_phoneme train \\
      --labels windowed_labels.jsonl --audio-dir audit_run/segment_audio_v2 \\
      --out-dir runs/rung2 \\
      --eval-segment-manifest audit_run/segment_manifest_v2.jsonl \\
      --eval-audio-dir audit_run/segment_audio_v2
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import SeamlessM4TFeatureExtractor

from tadabur.audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from tadabur.inference import MODEL_ID, PHONEME_LEVEL
from tadabur.muaalem import (
    Wav2Vec2BertForMultilevelCTC,
    Wav2Vec2BertForMultilevelCTCConfig,
)
from tadabur.phoneme_vocab import NUM_PHONEME_CLASSES
from training.waqf_head import phoneme_forward, phoneme_ctc_loss
from training.windowed_batch import (
    WindowedCtcBatch,
    WindowedCtcCollator,
    WindowedCtcExample,
    length_bucketed_batches,
    load_examples,
)
from training.waqf_distill import DEPLOYED_WINDOW_FEATURE_FRAMES

# 16 GB card, headroom for the CUDA context / allocator fragmentation. A batch whose peak
# stays under this is safe to commit (ADR-0004 "verify one real batch fits before committing").
DEFAULT_VRAM_BUDGET_GIB = 15.0

# The frozen 5 s window is the worst-case per-window length (250 feature frames); the batch
# memory bound is a *batch* of these, so the preflight sizes its batch from this.
WORST_CASE_WINDOW_FEATURE_FRAMES = DEPLOYED_WINDOW_FEATURE_FRAMES  # 250


@dataclass(frozen=True)
class LoRASettings:
    """LoRA adapter geometry for the bounded backbone drift (ADR-0004).

    The adapters ride the encoder self-attention projections — the backbone drift ADR-0004
    bounds by construction — while the phoneme head trains in full (``modules_to_save``).
    ``rank`` / ``alpha`` are the first should-reject-regression lever ADR-0004 names ("lower
    rank/alpha"), so they are knobs, not constants.
    """

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("linear_q", "linear_k", "linear_v", "linear_out")
    # The phoneme head is trained in full (not low-rank) — it is the ADR-0001 objective.
    phoneme_head_module: str = PHONEME_LEVEL


@dataclass
class TrainConfig:
    """Whole-clip phoneme-only run hyperparameters and the 16 GB batch budget."""

    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_accum_steps: int = 8
    max_frames_per_batch: int = 1000
    max_windows_per_batch: int = 8
    seed: int = 0
    lora: LoRASettings = field(default_factory=LoRASettings)


def set_seed(seed: int) -> None:
    """Seed Python / NumPy / torch for a reproducible run (thermo-nuclear #8)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_extractor(model_id: str = MODEL_ID) -> SeamlessM4TFeatureExtractor:
    """The model's own 16 kHz feature extractor (train/inference parity)."""
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_id)
    if feature_extractor.sampling_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"{model_id} feature extractor expects {feature_extractor.sampling_rate} Hz, "
            f"not {TARGET_SAMPLE_RATE} Hz."
        )
    return feature_extractor


def load_muaalem(
    model_id: str = MODEL_ID, dtype: torch.dtype = torch.bfloat16
) -> Wav2Vec2BertForMultilevelCTC:
    """Load the base Muaalem model, asserting the phoneme head is the expected 43 classes."""
    config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(model_id)
    phoneme_classes = config.level_to_vocab_size[PHONEME_LEVEL]
    if phoneme_classes != NUM_PHONEME_CLASSES:
        raise ValueError(
            f"{model_id} phoneme head has {phoneme_classes} classes, expected "
            f"{NUM_PHONEME_CLASSES} — label mapping would be corrupt."
        )
    return Wav2Vec2BertForMultilevelCTC.from_pretrained(model_id, config=config, dtype=dtype)


def attach_phoneme_lora(
    muaalem: Wav2Vec2BertForMultilevelCTC, settings: LoRASettings
):
    """Freeze the backbone, add LoRA on the attention projections, train the phoneme head.

    Returns the PEFT-wrapped model. All base weights are frozen; the LoRA adapters (bounded
    backbone drift) and the phoneme head (``modules_to_save``, the full ADR-0001 objective)
    are the only trainable parameters, so the **sifat heads take no gradient** — the
    phoneme-only isolation ADR-0004 requires. Enable checkpointing separately with
    :func:`enable_gradient_checkpointing`.
    """
    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        modules_to_save=[settings.phoneme_head_module],
        bias="none",
    )
    return get_peft_model(muaalem, config)


def enable_gradient_checkpointing(base: Wav2Vec2BertForMultilevelCTC) -> None:
    """Turn on **non-reentrant** activation checkpointing on the encoder (ADR-0004 OOM lever).

    Non-reentrant checkpointing recomputes each block under grad even though the backbone
    base weights are frozen, so the LoRA adapters inside a checkpointed block still receive
    gradient — the reentrant path would need an input tensor to require grad (this audio
    model exposes no input embeddings) and would silently starve the adapters.
    """
    base.wav2vec2_bert.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def base_of(peft_model) -> Wav2Vec2BertForMultilevelCTC:
    """The LoRA-mutated Muaalem the shared phoneme forward runs on (adapters active in place)."""
    return peft_model.get_base_model()


# --- memory preflight --------------------------------------------------------


@dataclass
class PreflightReport:
    """Peak VRAM of one real worst-case windowed forward/backward vs the budget."""

    num_windows: int
    window_feature_frames: int
    peak_allocated_gib: float
    peak_reserved_gib: float
    budget_gib: float
    fits: bool


def _worst_case_batch(
    feature_extractor: SeamlessM4TFeatureExtractor,
    sample_waveform: np.ndarray,
    num_windows: int,
    window_feature_frames: int,
) -> WindowedCtcBatch:
    """A full batch of ``num_windows`` maximal-length windows — the memory worst case.

    Each window is ``window_feature_frames`` of the (real) ``sample_waveform`` at a maximal
    (frame-length) CTC target, so the batch reproduces the largest activation and CTC shapes
    a real batch can hit under the frozen contract. The waveform is tiled/trimmed to the exact
    window sample length.
    """
    samples = window_feature_frames * (TARGET_SAMPLE_RATE * 20 // 1000)  # 20 ms/frame
    wave = np.asarray(sample_waveform, dtype=np.float32)
    if len(wave) < samples:
        wave = np.tile(wave, int(np.ceil(samples / max(len(wave), 1))))
    window_audio = wave[:samples]
    # Maximal CTC target: one label id per two logit frames (a phoneme cannot occupy fewer
    # than ~2 frames after the CTC blank), the longest feasible target for this window.
    logit_frames = window_feature_frames // 2
    label_ids = tuple(int(i % NUM_PHONEME_CLASSES) or 1 for i in range(logit_frames // 2))
    examples = [
        WindowedCtcExample(
            key=("__preflight__", i),
            audio=window_audio,
            label_ids=label_ids,
            start_sample=0,
            num_samples=samples,
            feature_frames=window_feature_frames,
            logit_frames=logit_frames,
        )
        for i in range(num_windows)
    ]
    return WindowedCtcCollator(feature_extractor)(examples)


def preflight_batch_memory(
    peft_model,
    feature_extractor: SeamlessM4TFeatureExtractor,
    sample_waveform: np.ndarray,
    num_windows: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    window_feature_frames: int = WORST_CASE_WINDOW_FEATURE_FRAMES,
    budget_gib: float = DEFAULT_VRAM_BUDGET_GIB,
) -> PreflightReport:
    """Run one real worst-case windowed forward+backward and measure peak VRAM.

    ADR-0004's OOM mitigation demands verifying a real batch fits *before* committing a run.
    This builds the largest batch the contract allows (``num_windows`` full 5 s windows),
    runs the same bf16 + gradient-checkpointed phoneme forward and CTC backward the training
    step uses, and reports peak allocated/reserved VRAM against ``budget_gib``. It never
    raises on a fit/no-fit outcome (that is the caller's go/no-go); it raises only if run off
    a CUDA device, where the measurement would be meaningless.
    """
    if device.type != "cuda":
        raise RuntimeError("preflight measures CUDA memory; run it on the GPU.")

    base = base_of(peft_model)
    enable_gradient_checkpointing(base)
    peft_model.train()

    batch = _worst_case_batch(
        feature_extractor, sample_waveform, num_windows, window_feature_frames
    ).to(device, dtype)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    forward = phoneme_forward(base, PHONEME_LEVEL, batch.input_features, batch.attention_mask)
    loss = phoneme_ctc_loss(
        forward.phoneme_logits, batch.labels, forward.student_lengths, base.config
    )
    loss.backward()
    peft_model.zero_grad(set_to_none=True)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    return PreflightReport(
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
    """One epoch's mean phoneme CTC loss on train and (if present) val — the convergence log."""

    epoch: int
    train_loss: float
    val_loss: float | None


def _step_loss(base, batch: WindowedCtcBatch) -> torch.Tensor:
    """Phoneme CTC loss for one collated batch through the shared phoneme forward."""
    forward = phoneme_forward(base, PHONEME_LEVEL, batch.input_features, batch.attention_mask)
    return phoneme_ctc_loss(
        forward.phoneme_logits, batch.labels, forward.student_lengths, base.config
    )


@torch.no_grad()
def evaluate_ctc_loss(
    peft_model,
    base,
    batches: list[list[WindowedCtcExample]],
    collate: WindowedCtcCollator,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Mean per-window CTC loss over ``batches`` — the val convergence signal.

    The model's CTC reduction is ``sum``, so each batch's loss is divided by its window
    count and averaged, giving a per-window number comparable across differently-sized
    batches.
    """
    peft_model.eval()
    total, windows = 0.0, 0
    for examples in batches:
        batch = collate(examples).to(device, dtype)
        loss = _step_loss(base, batch)
        total += float(loss.item())
        windows += len(examples)
    peft_model.train()
    return total / max(windows, 1)


def train(
    labels_path: Path,
    audio_dir: Path,
    out_dir: Path,
    config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> list[EpochStats]:
    """Run the whole-clip phoneme-only LoRA fine-tune; return the per-epoch loss trace.

    Saves the LoRA adapters and the per-epoch loss trace under ``out_dir``. Grad-accum,
    length bucketing and bf16 + gradient checkpointing keep it inside the 16 GB budget the
    preflight verifies. Deterministic for a given ``config.seed``.
    """
    set_seed(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = load_feature_extractor()
    collate = WindowedCtcCollator(feature_extractor)
    train_examples = load_examples(labels_path, audio_dir, "train")
    val_by_split = load_examples(labels_path, audio_dir, "val") if _has_val(labels_path) else []

    peft_model = attach_phoneme_lora(load_muaalem(dtype=dtype), config.lora).to(device)
    base = base_of(peft_model)
    enable_gradient_checkpointing(base)
    peft_model.train()

    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    val_batches = length_bucketed_batches(
        val_by_split, config.max_frames_per_batch, config.max_windows_per_batch, config.seed
    ) if val_by_split else []

    trace: list[EpochStats] = []
    for epoch in range(config.epochs):
        batches = length_bucketed_batches(
            train_examples,
            config.max_frames_per_batch,
            config.max_windows_per_batch,
            config.seed + epoch,
        )
        epoch_total, epoch_windows = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, examples in enumerate(batches):
            batch = collate(examples).to(device, dtype)
            loss = _step_loss(base, batch)
            (loss / config.grad_accum_steps).backward()
            epoch_total += float(loss.item())
            epoch_windows += len(examples)
            if (step + 1) % config.grad_accum_steps == 0 or step + 1 == len(batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_loss = (
            evaluate_ctc_loss(peft_model, base, val_batches, collate, device, dtype)
            if val_batches
            else None
        )
        stats = EpochStats(epoch, epoch_total / max(epoch_windows, 1), val_loss)
        trace.append(stats)
        print(f"epoch {epoch}: train {stats.train_loss:.4f}"
              + (f"  val {val_loss:.4f}" if val_loss is not None else ""))

    peft_model.save_pretrained(out_dir / "lora_adapter")
    (out_dir / "loss_trace.json").write_text(
        json.dumps([asdict(s) for s in trace], indent=2), encoding="utf-8"
    )
    return trace


def _has_val(labels_path: Path) -> bool:
    from training.windowed_labels import read_labels

    return bool(read_labels(labels_path).get("val"))


def merge_checkpoint(out_dir: Path, dtype: torch.dtype = torch.bfloat16) -> Path:
    """Merge the trained LoRA adapters into a full Muaalem checkpoint for eval/export.

    Loads the base model + saved adapters, merges the low-rank updates into the backbone
    (and the trained phoneme head), and saves a standalone checkpoint plus the feature
    extractor so :mod:`tadabur.eval_harness` / :mod:`tadabur.inference` can load it by path.
    """
    from peft import PeftModel

    merged_dir = out_dir / "merged"
    base = load_muaalem(dtype=dtype)
    peft_model = PeftModel.from_pretrained(base, out_dir / "lora_adapter")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    load_feature_extractor().save_pretrained(merged_dir)
    return merged_dir


def emit_eval_report(
    merged_dir: Path,
    segment_manifest: Path,
    eval_audio_dir: Path,
    out_path: Path,
) -> None:
    """Score the merged rung-(2) checkpoint with the two-sided #7 harness and write it.

    Runs :func:`tadabur.eval_harness.run_eval` (should-accept recall / should-reject
    discrimination + the soft-pair/shadda confusion matrix) on the merged checkpoint — the
    rung-(2) eval outputs ADR-0004's ablation ladder (#33) compares against the ADR-0001
    baseline and the joint rung (3).
    """
    from tadabur.eval_harness import run_eval

    report = run_eval(segment_manifest, eval_audio_dir, model_id=str(merged_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote rung-(2) eval report to {out_path}")


# --- CLI ---------------------------------------------------------------------


def _sample_waveform(audio_dir: Path, sample_clip: str | None) -> np.ndarray:
    """A real staged clip waveform for the preflight (named clip, else the first in the dir)."""
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
    peft_model = attach_phoneme_lora(load_muaalem(), LoRASettings()).to(device)
    report = preflight_batch_memory(
        peft_model,
        load_feature_extractor(),
        _sample_waveform(args.audio_dir, args.sample_clip),
        num_windows=args.num_windows,
        device=device,
        budget_gib=args.budget_gib,
    )
    print(json.dumps(asdict(report), indent=2))
    if not report.fits:
        raise SystemExit(
            f"batch of {report.num_windows} windows peaks at {report.peak_reserved_gib} GiB, "
            f"over the {report.budget_gib} GiB budget — lower --num-windows or add checkpointing."
        )


def _cmd_train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        grad_accum_steps=args.grad_accum_steps,
        max_frames_per_batch=args.max_frames_per_batch,
        max_windows_per_batch=args.max_windows_per_batch,
        seed=args.seed,
    )
    train(args.labels, args.audio_dir, args.out_dir, config, device)
    if args.eval_segment_manifest and args.eval_audio_dir:
        merged_dir = merge_checkpoint(args.out_dir)
        emit_eval_report(
            merged_dir,
            args.eval_segment_manifest,
            args.eval_audio_dir,
            args.out_dir / "eval_rung2.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight", help="verify one real windowed batch fits 16 GB")
    pre.add_argument("--audio-dir", type=Path, required=True)
    pre.add_argument("--sample-clip", type=str, default=None,
                     help="clip filename under --audio-dir; defaults to the first .wav.")
    pre.add_argument("--num-windows", type=int, default=TrainConfig.max_windows_per_batch)
    pre.add_argument("--budget-gib", type=float, default=DEFAULT_VRAM_BUDGET_GIB)
    pre.set_defaults(func=_cmd_preflight)

    tr = sub.add_parser("train", help="run the whole-clip phoneme-only fine-tune")
    tr.add_argument("--labels", type=Path, required=True,
                    help="windowed CTC labels JSONL (training.windowed_labels).")
    tr.add_argument("--audio-dir", type=Path, required=True)
    tr.add_argument("--out-dir", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    tr.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    tr.add_argument("--grad-accum-steps", type=int, default=TrainConfig.grad_accum_steps)
    tr.add_argument("--max-frames-per-batch", type=int, default=TrainConfig.max_frames_per_batch)
    tr.add_argument("--max-windows-per-batch", type=int, default=TrainConfig.max_windows_per_batch)
    tr.add_argument("--seed", type=int, default=TrainConfig.seed)
    tr.add_argument("--eval-segment-manifest", type=Path, default=None,
                    help="segment manifest for the #7 eval (with --eval-audio-dir).")
    tr.add_argument("--eval-audio-dir", type=Path, default=None)
    tr.set_defaults(func=_cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Size-distillation training: thin Muaalem student from the 578M teacher.

Trains one of ``training.distill_student``'s width presets to reproduce
``obadx/muaalem-model-v3_2``'s phoneme head, so the deployed model can drop from six ANE
chunks to one. The objective and its rationale live in ``training.distill_loss``; this
module owns the run: the frozen teacher, the optimiser, the VRAM preflight, checkpointing
and the metrics log.

**The teacher runs online rather than from a cache.** Caching only its 43-way logits would
be cheap (~11 KB/window) but would forfeit the feature-matching term, which is what carries
a 7x width cut; caching its hidden states instead would cost ~3 MB/window, i.e. terabytes
over the corpus. Running it live costs one bf16 forward per step with no gradient, no
optimiser state and no stored activations -- about 1.2 GB resident -- which the 16 GB card
absorbs comfortably. It also keeps teacher and student pointwise aligned on identical
input by construction, with no cache-staleness failure mode.

**Preflight before commit.** ``training.whole_clip_phoneme`` established the pattern this
follows: run one real worst-case forward/backward, measure peak VRAM against a budget, and
refuse to start a multi-hour run that would OOM an hour in. ``--preflight-only`` runs just
that check.

Usage::

    # Check the step fits before committing to a run
    python -m training.distill_train --preset h384 \\
        --audio-root ../tadabur/audit_run/clips_v2 --preflight-only

    # Train
    python -m training.distill_train --preset h384 \\
        --audio-root ../tadabur/audit_run/clips_v2 \\
        --out-dir runs/h384 --batch-size 16 --steps 60000

    # Resume an interrupted run
    python -m training.distill_train --preset h384 \\
        --audio-root ../tadabur/audit_run/clips_v2 --out-dir runs/h384 --resume

Linux + CUDA only (see ``tools/environment.yml``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.distill_data import (
    DistillWindowDataset,
    build_window_index,
    discover_clips,
    split_clips,
)
from training.distill_loss import (
    DEFAULT_TAP_LAYERS,
    DistillLossConfig,
    FeatureProjector,
    agreement_stats,
    breakout_stats,
    distillation_loss,
)
from training.distill_student import (
    PRESETS,
    TEACHER_HIDDEN_SIZE,
    TEACHER_MODEL_ID,
    build_student,
    count_parameters,
)

# Matches ``training.whole_clip_phoneme``'s budget: the box is a 16 GB card and a run must
# leave headroom for allocator fragmentation rather than sitting exactly at the ceiling.
DEFAULT_VRAM_BUDGET_GIB = 15.0

DEFAULT_SEED = 1234
METRICS_FILENAME = "metrics.jsonl"
CHECKPOINT_FILENAME = "checkpoint.pt"
CONFIG_FILENAME = "run_config.json"


@dataclass
class TrainConfig:
    """Everything that defines a run, persisted so a resume cannot silently diverge."""

    preset: str
    audio_root: str
    out_dir: str
    steps: int = 60_000
    batch_size: int = 16
    grad_accum: int = 1
    # 1e-4, not 3e-4. At 3e-4 the run improved through warmup and then regressed the
    # moment the rate reached peak (non-blank agreement 0.002 -> 0.000, rank 8.19 -> 8.35
    # between steps 2000 and 4000). Even at 1e-4 the pre-clip gradient norm runs 3-7 and
    # hits the 5.0 clip about half the time, so 3e-4 was clipped on essentially every step:
    # training far below its nominal rate while the loss curve looked converged. See
    # ADR-0010.
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2_000
    max_grad_norm: float = 5.0
    hop_seconds: float = 2.5
    val_fraction: float = 0.02
    # When set, training windows are streamed from Tadabur parquet shards instead of read
    # from --audio-root, which stays the (held-out) validation source. Peak disk is one
    # shard per worker; see training.distill_stream.
    stream_shards: str = ""
    num_workers: int = 8
    log_every: int = 50
    eval_every: int = 2_000
    save_every: int = 2_000
    eval_batches: int = 40
    seed: int = DEFAULT_SEED
    logit_weight: float = 1.0
    # Both OFF by default -- see DistillLossConfig for the measurements. The shipped
    # objective is pure weighted KL.
    feature_weight: float = 0.0
    ctc_weight: float = 0.0
    hard_weight: float = 0.0
    temperature: float = 2.0
    nonblank_weight: float = 3.0
    confirm_weight: float = 2.0

    def loss_config(self) -> DistillLossConfig:
        return DistillLossConfig(
            logit_weight=self.logit_weight,
            feature_weight=self.feature_weight,
            ctc_weight=self.ctc_weight,
            hard_weight=self.hard_weight,
            temperature=self.temperature,
            nonblank_weight=self.nonblank_weight,
            confirm_weight=self.confirm_weight,
            tap_layers=DEFAULT_TAP_LAYERS,
        )


# Fields a resume must reproduce exactly. Changing any of them mid-run makes the metrics
# log describe two different experiments spliced together -- and the first four would
# silently corrupt the run outright, since the optimiser and LR schedule being restored
# were fitted under them. Deliberately excludes the operational knobs (num_workers,
# log_every, eval_every, save_every, eval_batches), which a resume may legitimately change.
RESUME_CRITICAL_FIELDS = (
    "preset",
    "steps",
    "batch_size",
    "grad_accum",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "max_grad_norm",
    "hop_seconds",
    "val_fraction",
    "stream_shards",
    "seed",
    "audio_root",
    "logit_weight",
    "feature_weight",
    "ctc_weight",
    "hard_weight",
    "temperature",
    "nonblank_weight",
    "confirm_weight",
)


def check_resume_compatible(saved: dict, current: TrainConfig) -> None:
    """Fail fast when a resume would splice two different experiments together.

    The same guarantee ``training.waqf_distill``'s ``SoftLabelStore`` gives its generation
    contract, for the same reason: a silently divergent resume produces an artifact that
    looks fine and is not. A run resumed at a different ``ctc_weight`` or ``learning_rate``
    carries an optimiser state and LR schedule fitted under the old values, and its metrics
    log reads as one continuous curve.
    """
    current_values = asdict(current)
    mismatches = [
        f"  {field}: checkpoint={saved[field]!r} current={current_values[field]!r}"
        for field in RESUME_CRITICAL_FIELDS
        if field in saved and saved[field] != current_values[field]
    ]
    if mismatches:
        raise SystemExit(
            "refusing to resume: this run's config differs from the checkpoint's on "
            "fields the restored optimiser and schedule depend on.\n"
            + "\n".join(mismatches)
            + "\n\nEither match them, or start a fresh --out-dir."
        )


def _init_worker(worker_id: int) -> None:
    """Pin each DataLoader worker to a single compute thread.

    PyTorch gives every process intra-op threads up to the core count, so N workers on a
    12-core box spawn ~14 threads each and oversubscribe it by an order of magnitude --
    measured here: 8 workers, 112 threads, load average 68, and step rate dropping from
    0.95/s to 0.69/s as the contention built. A worker's job is feature extraction on one
    window at a time, which does not benefit from intra-op parallelism; the parallelism
    that matters is across workers.

    Set ``OMP_NUM_THREADS=1`` in the launching environment as well: this covers torch, but
    numpy and the feature extractor read the env var directly.
    """
    torch.set_num_threads(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_teacher(device: torch.device, model_id: str = TEACHER_MODEL_ID):
    """The frozen teacher in bf16, eval mode, gradients disabled.

    Loaded through the vendored modeling code (not ``AutoModel``) for the same reason
    ``tadabur.inference`` does: the multi-level CTC class is not registered with
    transformers, and pinning the vendored copy keeps train and filter on identical weights.
    """
    from tadabur.muaalem import (
        Wav2Vec2BertForMultilevelCTC,
        Wav2Vec2BertForMultilevelCTCConfig,
    )

    config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(model_id)
    teacher = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_id, config=config)
    teacher = teacher.to(device=device, dtype=torch.bfloat16)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Train and validation loaders over disjoint *clips*."""
    audio_root = Path(config.audio_root)
    if not audio_root.is_dir():
        raise SystemExit(f"not a directory: {audio_root}")

    clips = discover_clips(audio_root)
    if not clips:
        raise SystemExit(f"no .wav files under {audio_root}")
    train_clips, val_clips = split_clips(clips, config.val_fraction)

    val_refs = build_window_index(val_clips, config.hop_seconds)

    if config.stream_shards:
        # Streaming: the staged corpus is validation only, and the shards it was built
        # from are refused by StreamingWindowDataset so the split cannot leak.
        from tadabur.shard_reader import parse_shard_spec
        from training.distill_stream import StreamingWindowDataset

        train_dataset = StreamingWindowDataset(
            parse_shard_spec(config.stream_shards),
            hop_seconds=config.hop_seconds,
            seed=config.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=config.num_workers > 0,
            worker_init_fn=_init_worker,
        )
    else:
        train_refs = build_window_index(train_clips, config.hop_seconds)
        if not train_refs:
            raise SystemExit("no training windows -- check --audio-root and --hop-seconds")
        train_loader = DataLoader(
            DistillWindowDataset(train_refs),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=config.num_workers > 0,
            worker_init_fn=_init_worker,
        )
    val_loader = DataLoader(
        DistillWindowDataset(val_refs),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(2, config.num_workers // 2),
        pin_memory=True,
        drop_last=False,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_init_worker,
    )
    return train_loader, val_loader


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 1% of peak."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def forward_step(
    features: torch.Tensor,
    teacher,
    student,
    projector: FeatureProjector,
    loss_config: DistillLossConfig,
):
    """One teacher+student forward and the loss over them."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher_out = teacher(features, output_hidden_states=True, return_dict=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        student_out = student(features, output_hidden_states=True, return_dict=True)

    return distillation_loss(
        student_logits=student_out["logits"]["phonemes"],
        teacher_logits=teacher_out["logits"]["phonemes"],
        student_hidden=student_out["hidden_states"],
        teacher_hidden=teacher_out["hidden_states"],
        projector=projector,
        config=loss_config,
    )


@torch.no_grad()
def evaluate(
    val_loader: DataLoader,
    teacher,
    student,
    device: torch.device,
    max_batches: int,
) -> dict:
    """Mean student-vs-teacher agreement over held-out clips.

    Reports frame-level agreement only -- the decoded-string agreement over the deployed
    sliding-window protocol is a separate, more expensive measurement (see
    ``training.distill_eval``). This one is cheap enough to run during training.
    """
    student.eval()
    totals: dict[str, float] = {}
    count = 0

    for index, features in enumerate(val_loader):
        if index >= max_batches:
            break
        features = features.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_logits = teacher(features, return_dict=True)["logits"]["phonemes"]
            student_logits = student(features, return_dict=True)["logits"]["phonemes"]
        # Agreement plus the continuous distance-from-breakout. While the student sits in
        # the all-blank basin, argmax agreement is a flat 0 whether the run is converging
        # or stuck; target_prob / target_rank are what move.
        stats = agreement_stats(student_logits, teacher_logits).as_dict()
        stats.update(breakout_stats(student_logits, teacher_logits).as_dict())
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1

    student.train()
    if count == 0:
        return {}
    return {key: round(value / count, 4) for key, value in totals.items()}


def preflight(
    config: TrainConfig, device: torch.device, budget_gib: float
) -> tuple[float, bool]:
    """One real worst-case forward/backward; returns (peak GiB, fits).

    Uses a full batch of the configured size at the static 250-frame window -- which *is*
    the worst case, since the ANE shape contract means every window is the same size. There
    is no longer-sequence tail to worry about, unlike the variable-length training paths.
    """
    spec = PRESETS[config.preset]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    teacher = load_teacher(device)
    student = build_student(spec).to(device)
    student.train()
    projector = FeatureProjector(
        spec.hidden_size, TEACHER_HIDDEN_SIZE, len(DEFAULT_TAP_LAYERS)
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=config.learning_rate,
    )

    features = torch.randn(config.batch_size, 250, 160, device=device)
    try:
        output = forward_step(features, teacher, student, projector, config.loss_config())
        output.total.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        fits = peak_gib <= budget_gib
        del output
    except torch.cuda.OutOfMemoryError:
        # An OOM *is* the preflight's answer, not a crash: report the batch as not fitting
        # so the caller can say so plainly. Peak is whatever we reached before failing, so
        # it is a lower bound on the real requirement.
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        fits = False

    del teacher, student, projector, optimizer, features
    torch.cuda.empty_cache()

    return peak_gib, fits


def save_checkpoint(
    path: Path, step: int, student, projector, optimizer, scheduler, config: TrainConfig
) -> None:
    """Atomic checkpoint write -- temp file then rename, so a kill mid-save cannot
    leave a truncated checkpoint that fails to load on resume."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "student": student.state_dict(),
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": asdict(config),
        },
        tmp,
    )
    tmp.replace(path)


def load_student_weights(checkpoint_path: Path, student, projector, device) -> int:
    """Warm-start weights from another run, without its optimiser or schedule.

    Distinct from ``--resume`` on purpose. A resume continues one experiment and therefore
    demands an identical config (:func:`check_resume_compatible`); this deliberately starts
    a *new* experiment -- different objective, fresh learning-rate schedule -- from another
    run's learned weights. Using resume for that would either be refused by the guard or,
    without it, silently splice two objectives onto one optimiser state.

    Returns the step the source checkpoint reached, for the log only.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student.load_state_dict(state["student"])
    if "projector" in state:
        projector.load_state_dict(state["projector"])
    return state.get("step", 0)


def train(config: TrainConfig, resume: bool = False, init_from: Path | None = None) -> None:
    device = torch.device("cuda")
    seed_everything(config.seed)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CONFIG_FILENAME).write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    spec = PRESETS[config.preset]
    print(f"[setup] preset {config.preset}: {spec}")

    teacher = load_teacher(device)
    student = build_student(spec).to(device)
    student.train()
    projector = FeatureProjector(
        spec.hidden_size, TEACHER_HIDDEN_SIZE, len(DEFAULT_TAP_LAYERS)
    ).to(device)

    print(f"[setup] teacher {count_parameters(teacher) / 1e6:.1f}M (frozen, bf16)")
    print(f"[setup] student {count_parameters(student) / 1e6:.1f}M trainable")
    print(f"[setup] projector {count_parameters(projector) / 1e6:.1f}M (training only)")

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, config.warmup_steps, config.steps)
    )

    start_step = 0
    if init_from is not None:
        source_step = load_student_weights(init_from, student, projector, device)
        print(f"[init] warm-started from {init_from} (its step {source_step}); "
              f"fresh optimiser and schedule")

    checkpoint_path = out_dir / CHECKPOINT_FILENAME
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        check_resume_compatible(state.get("config", {}), config)
        student.load_state_dict(state["student"])
        projector.load_state_dict(state["projector"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        print(f"[resume] restored from step {start_step}")

    train_loader, val_loader = build_dataloaders(config)
    if config.stream_shards:
        print(
            f"[data] streaming shards {config.stream_shards} "
            f"(one shard per worker on disk), "
            f"{len(val_loader.dataset):,} held-out val windows"
        )
    else:
        print(
            f"[data] {len(train_loader.dataset):,} train windows, "
            f"{len(val_loader.dataset):,} val windows"
        )

    loss_config = config.loss_config()
    metrics_path = out_dir / METRICS_FILENAME
    step = start_step
    grad_norm = 0.0
    started = time.time()

    def log(record: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    while step < config.steps:
        for features in train_loader:
            if step >= config.steps:
                break
            features = features.to(device, non_blocking=True)

            output = forward_step(features, teacher, student, projector, loss_config)
            (output.total / config.grad_accum).backward()

            if (step + 1) % config.grad_accum == 0:
                # Keep the pre-clip norm: it is the direct evidence for an unstable
                # learning rate, and without it a run that is being clipped every step --
                # i.e. training far below its nominal rate, or diverging and being reined
                # in -- is indistinguishable from one that has converged.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(projector.parameters()),
                    config.max_grad_norm,
                ).item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % config.log_every == 0:
                record = {
                    "step": step,
                    "lr": round(scheduler.get_last_lr()[0], 7),
                    "grad_norm_preclip": round(grad_norm, 3),
                    "clipped": grad_norm > config.max_grad_norm,
                    "elapsed_s": round(time.time() - started, 1),
                    **output.as_dict(),
                }
                log(record)
                print(
                    f"[{step:>6}/{config.steps}] loss {record['total']:.4f} "
                    f"kl {record['logit_loss']:.4f} feat {record['feature_loss']:.4f} "
                    f"ctc {record['ctc_loss']:.4f} hard {record['hard_loss']:.4f} "
                    f"cos {record['feature_cosine']:.3f} "
                    f"conf-agree {record['confirmed_agreement']:.3f} "
                    f"blank-margin {record['blank_collapse_margin']:+.3f} "
                    f"|g| {record['grad_norm_preclip']:.1f}"
                    f"{'*' if record['clipped'] else ''}",
                    flush=True,
                )
                if output.stats.is_collapsing():
                    print(
                        f"[warn] blank collapse: student blank "
                        f"{output.stats.student_blank_ratio:.3f} vs teacher "
                        f"{output.stats.teacher_blank_ratio:.3f}",
                        flush=True,
                    )

            if step % config.eval_every == 0:
                val = evaluate(val_loader, teacher, student, device, config.eval_batches)
                log({"step": step, "split": "val", **val})
                print(f"[eval {step}] {val}", flush=True)

            if step % config.save_every == 0:
                save_checkpoint(
                    checkpoint_path, step, student, projector, optimizer, scheduler, config
                )

    save_checkpoint(
        checkpoint_path, step, student, projector, optimizer, scheduler, config
    )
    val = evaluate(val_loader, teacher, student, device, config.eval_batches)
    log({"step": step, "split": "val", "final": True, **val})
    print(f"[done] {step} steps in {(time.time() - started) / 3600:.2f}h -- final {val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distil a thin Muaalem student")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="h384")
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/distill"))
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--hop-seconds", type=float, default=2.5)
    parser.add_argument(
        "--stream-shards",
        default="",
        help='stream training windows from Tadabur shards, e.g. "20-384", instead of '
        "reading --audio-root (which then serves only the held-out validation clips). "
        "Peak disk is one ~2.5 GB shard per worker however many shards the run covers.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--feature-weight",
        type=float,
        default=0.0,
        help="tapped hidden-state matching. Off: it earned 0.5%% of the gradient and scored "
        "0.886x the predict-the-mean baseline. Kept for ablation only.",
    )
    parser.add_argument(
        "--logit-weight",
        type=float,
        default=1.0,
        help="weight on the temperature-scaled KL. Set to 0 with --hard-weight 1 to train "
        "purely on the teacher's argmax, which is what the release gate measures.",
    )
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=0.0,
        help="CTC anchor against the teacher's decoded sequence. OFF by default: it "
        "destabilises training as the data diversifies (at 1024 fixed windows KL-only "
        "reached decoded 0.847, KL+CTC collapsed to 0.000). Set >0 only to reproduce the "
        "ablation.",
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--nonblank-weight",
        type=float,
        default=3.0,
        help="loss multiplier on frames the teacher decodes as non-blank; raise it if the "
        "student parks in the all-blank basin (watch blank_collapse_margin)",
    )
    parser.add_argument("--confirm-weight", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-from",
        type=Path,
        help="warm-start student/projector weights from another run's checkpoint with a "
        "fresh optimiser and schedule. Use this, not --resume, when changing the objective.",
    )
    parser.add_argument(
        "--hard-weight",
        type=float,
        default=0.0,
        help="weight on cross-entropy against the teacher's argmax. Optimises the decoded "
        "metric directly; intended as a finishing objective once the student is past the "
        "blank basin.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--budget-gib", type=float, default=DEFAULT_VRAM_BUDGET_GIB)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    config = TrainConfig(
        preset=args.preset,
        audio_root=str(args.audio_root),
        out_dir=str(args.out_dir),
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        hop_seconds=args.hop_seconds,
        stream_shards=args.stream_shards,
        num_workers=args.num_workers,
        log_every=args.log_every,
        eval_batches=args.eval_batches,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        logit_weight=args.logit_weight,
        feature_weight=args.feature_weight,
        ctc_weight=args.ctc_weight,
        hard_weight=args.hard_weight,
        temperature=args.temperature,
        nonblank_weight=args.nonblank_weight,
        confirm_weight=args.confirm_weight,
    )

    device = torch.device("cuda")
    peak, fits = preflight(config, device, args.budget_gib)
    if fits:
        verdict = "fits"
    elif peak > args.budget_gib:
        verdict = "EXCEEDS BUDGET"
    else:
        # Allocation failed below the stated budget: the card has less usable memory than
        # the budget claims (other processes, or fragmentation).
        verdict = "OUT OF MEMORY below budget"
    print(
        f"[preflight] peak {peak:.2f} GiB vs {args.budget_gib:.1f} GiB budget "
        f"at batch {config.batch_size} -- {verdict}"
    )
    if not fits:
        raise SystemExit("reduce --batch-size or raise --budget-gib")
    if args.preflight_only:
        return

    train(config, resume=args.resume, init_from=args.init_from)


if __name__ == "__main__":
    main()

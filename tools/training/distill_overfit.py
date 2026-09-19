"""Can the student fit a handful of windows at all? The first question to ask a stuck run.

When a distillation run plateaus, the expensive mistake is to keep tuning the loss on the
full corpus. The cheap and decisive test is whether the model can **overfit a tiny fixed
batch**: a network that cannot drive the loss toward zero on 32 examples it sees over and
over has a structural problem -- optimisation, gradient flow, scaling -- that no amount of
data or patience will fix. One that *can* overfit is telling you the architecture and the
objective are sound, and the problem is elsewhere.

It also instruments the thing a training loop usually hides: the **pre-clip gradient norm**.
``distill_train`` clips to ``max_grad_norm`` and never reports what it clipped from, so a
run whose true norm is 100 against a limit of 5 is quietly training at a twentieth of its
nominal learning rate and looks, from the loss curve, exactly like a model that has
converged. That failure is invisible without this number.

Reference points for reading the output, measured on this corpus and teacher:

* ``ctc``: **17.8** for an all-blank student, **0.19** for one matching the teacher.
* ``feat``: the best *constant* predictor (per-dimension mean of the teacher's hidden
  states) scores **0.203** averaged over the six tapped layers. A feature loss near that is
  not learning a representation; it is predicting the mean.

Usage::

    python -m training.distill_overfit --preset h384 \\
        --audio-root ../tadabur/audit_run/clips_v2 --steps 400

    # sweep learning rates to separate "cannot learn" from "learning rate is wrong"
    python -m training.distill_overfit --preset h384 --audio-root <dir> \\
        --steps 300 --lr-sweep 3e-4 1e-4 3e-5

Linux + CUDA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.distill_data import DistillWindowDataset, build_window_index, discover_clips
from training.distill_loss import (
    DEFAULT_TAP_LAYERS,
    CONFIRM_TIMESTEPS,
    DistillLossConfig,
    FeatureProjector,
    breakout_stats,
    ctc_anchor_loss,
    distillation_loss,
    feature_matching_loss,
    frame_weights,
    hard_label_ce,
    weighted_kl,
)
from training.distill_student import PRESETS, TEACHER_HIDDEN_SIZE, build_student
from training.distill_train import load_teacher, seed_everything

# The MSE a constant predictor achieves on the teacher's tapped hidden states, averaged
# over DEFAULT_TAP_LAYERS. Measured on this corpus; see the module docstring.
PREDICT_MEAN_FEATURE_MSE = 0.203


def fixed_batches(
    audio_root: Path, num_windows: int, batch_size: int, device: torch.device
) -> list[torch.Tensor]:
    """A fixed, repeatedly-cycled set of real windows, loaded once.

    ``num_windows`` controls the *difficulty* of the probe and is the whole point of it. A
    model that fits 32 windows perfectly but stalls at 2048 has an optimisation or capacity
    problem that reveals itself with scale, not a data shortage -- and the size at which it
    breaks localises the failure far better than any single-point test.
    """
    clips = discover_clips(audio_root)
    if not clips:
        raise SystemExit(f"no .wav files under {audio_root}")
    refs = build_window_index(clips)[:num_windows]
    loader = DataLoader(
        DistillWindowDataset(refs), batch_size=batch_size, num_workers=8, drop_last=True
    )
    return [batch.to(device) for batch in loader]


def term_gradient_shares(
    student_logits,
    teacher_logits,
    student_hidden,
    teacher_hidden,
    projector,
    parameters,
    config: DistillLossConfig,
) -> dict:
    """Gradient norm each loss term contributes on its own, as a share of the total.

    Loss *magnitudes* say nothing about influence -- a term can read 0.18 against another's
    6.5 and still dominate the update. This backwards each term separately (retaining the
    graph) and measures the norm it actually puts on the parameters.

    The specific suspicion it exists to test: the feature term is an **unnormalised** MSE
    against teacher hidden states whose scale varies several-fold across tapped layers
    (measured on this teacher: std 0.15 at layer 24, 0.66 at layer 12). Under a plain MSE
    the large-scale layers dominate, so the term can quietly consume the update budget while
    its printed value looks negligible.
    """
    weights = frame_weights(
        teacher_logits,
        nonblank_weight=config.nonblank_weight,
        confirm_timesteps=config.confirm_timesteps,
        confirm_weight=config.confirm_weight,
    )
    terms = {
        "kl": config.logit_weight
        * weighted_kl(student_logits, teacher_logits, weights, config.temperature),
        "feature": config.feature_weight
        * feature_matching_loss(
            student_hidden, teacher_hidden, projector, config.tap_layers
        )[0],
        "ctc": config.ctc_weight * ctc_anchor_loss(student_logits, teacher_logits),
        "hard": config.hard_weight
        * hard_label_ce(student_logits, teacher_logits, weights),
    }

    norms: dict[str, float] = {}
    for name, term in terms.items():
        if not float(term.detach()):
            norms[name] = 0.0
            continue
        for parameter in parameters:
            parameter.grad = None
        term.backward(retain_graph=True)
        total = torch.sqrt(
            sum(
                (p.grad.detach() ** 2).sum()
                for p in parameters
                if p.grad is not None
            )
        )
        norms[name] = float(total)

    for parameter in parameters:
        parameter.grad = None

    denominator = sum(norms.values()) or 1.0
    return {f"grad_share_{k}": round(v / denominator, 4) for k, v in norms.items()}


@torch.no_grad()
def decoded_agreement(student_logits, teacher_logits) -> float:
    """Character agreement of the CONFIRMED decode on the overfit batch itself.

    Frame agreement can look healthy while the decoded stream does not, so the overfit probe
    has to score what the release gate scores. If the model cannot reach near-1.0 here --
    on data it is being shown repeatedly -- no amount of data or capacity is the answer.
    """
    from training.distill_eval import confirmed_tokens, levenshtein

    student_ids = student_logits.argmax(dim=-1).cpu().numpy()
    teacher_ids = teacher_logits.argmax(dim=-1).cpu().numpy()

    edits = tokens = 0
    for student_row, teacher_row in zip(student_ids, teacher_ids):
        reference = confirmed_tokens(teacher_row, CONFIRM_TIMESTEPS)
        hypothesis = confirmed_tokens(student_row, CONFIRM_TIMESTEPS)
        edits += levenshtein(reference, hypothesis)
        tokens += len(reference)
    return 1.0 - edits / max(1, tokens)


def overfit(
    preset: str,
    batches: list[torch.Tensor],
    teacher,
    device: torch.device,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    log_every: int,
    loss_config: DistillLossConfig,
) -> dict:
    """Train on one fixed batch and report whether the loss actually moves."""
    seed_everything(1234)
    spec = PRESETS[preset]
    student = build_student(spec).to(device)
    student.train()
    projector = FeatureProjector(
        spec.hidden_size, TEACHER_HIDDEN_SIZE, len(DEFAULT_TAP_LAYERS)
    ).to(device)
    parameters = list(student.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

    # Teacher outputs are recomputed each step rather than cached. Caching looks obvious --
    # the probe shows the same windows repeatedly -- but the hidden states are 25 layers x
    # B x 250 x d, ~150 MB per batch of 32, so caching even 64 batches needs ~10 GB and
    # OOMs the card. Recomputing costs one no-grad forward, which the real training pays
    # anyway.
    history = []
    for step in range(1, steps + 1):
        features = batches[(step - 1) % len(batches)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_out = teacher(features, output_hidden_states=True, return_dict=True)
        teacher_logits = teacher_out["logits"]["phonemes"]
        teacher_hidden = teacher_out["hidden_states"]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_out = student(features, output_hidden_states=True, return_dict=True)

        output = distillation_loss(
            student_logits=student_out["logits"]["phonemes"],
            teacher_logits=teacher_logits,
            student_hidden=student_out["hidden_states"],
            teacher_hidden=teacher_hidden,
            projector=projector,
            config=loss_config,
        )
        if step == 1:
            # Must run BEFORE the main backward: each term backwards with retain_graph so
            # the graph survives for the real step, but the main backward frees it.
            print(
                "  gradient share by term: "
                f"{term_gradient_shares(student_out['logits']['phonemes'], teacher_logits, student_out['hidden_states'], teacher_hidden, projector, parameters, loss_config)}",
                flush=True,
            )

        optimizer.zero_grad(set_to_none=True)
        output.total.backward()

        # The number distill_train never reports. clip_grad_norm_ returns the norm it saw
        # *before* clipping, so this is the true gradient scale.
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm).item()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            breakout = breakout_stats(
                student_out["logits"]["phonemes"], teacher_logits
            )
            record = {
                "step": step,
                "grad_norm_preclip": round(grad_norm, 2),
                "clipped": grad_norm > max_grad_norm,
                "decoded_agreement": round(
                    decoded_agreement(
                        student_out["logits"]["phonemes"], teacher_logits
                    ),
                    4,
                ),
                **output.as_dict(),
                **breakout.as_dict(),
            }
            history.append(record)
            print(
                f"  [{step:>4}] loss {record['total']:.3f} kl {record['logit_loss']:.3f} "
                f"ctc {record['ctc_loss']:.3f} feat {record['feature_loss']:.4f} "
                f"|g| {grad_norm:>8.1f}{'*' if record['clipped'] else ' '} "
                f"nb {record['nonblank_agreement']:.3f} "
                f"DECODED {record['decoded_agreement']:.3f}",
                flush=True,
            )

    first, last = history[0], history[-1]
    return {
        "learning_rate": learning_rate,
        "steps": steps,
        "first": first,
        "last": last,
        "total_dropped": round(first["total"] - last["total"], 3),
        "feature_vs_predict_mean": round(
            last["feature_loss"] / PREDICT_MEAN_FEATURE_MSE, 3
        ),
        "escaped_blank": last["nonblank_agreement"] > 0.0,
        "clipped_fraction": round(
            sum(1 for h in history if h["clipped"]) / len(history), 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overfit a fixed batch to tell a broken run from a slow one"
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="h384")
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-windows",
        type=int,
        default=32,
        help="how many distinct windows to fit. The probe's difficulty knob: sweep it to "
        "find where the model stops being able to reproduce the teacher.",
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument(
        "--lr-sweep",
        type=float,
        nargs="*",
        default=None,
        help="learning rates to try in turn; separates 'cannot learn' from 'lr is wrong'",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ctc-weight", type=float, default=0.0)
    parser.add_argument("--feature-weight", type=float, default=0.0)
    parser.add_argument(
        "--logit-weight",
        type=float,
        default=1.0,
        help="set terms to 0 to isolate one: a single term that still plateaus points at "
        "the model or the optimiser, whereas terms that only plateau together point at a "
        "conflict between them",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    batches = fixed_batches(
        args.audio_root, max(args.num_windows, args.batch_size), args.batch_size, device
    )
    print(f"fitting {len(batches) * args.batch_size} windows in {len(batches)} batches")
    teacher = load_teacher(device)
    loss_config = DistillLossConfig(
        logit_weight=args.logit_weight,
        ctc_weight=args.ctc_weight,
        feature_weight=args.feature_weight,
    )
    print(
        f"terms: logit={args.logit_weight:g} ctc={args.ctc_weight:g} "
        f"feature={args.feature_weight:g}"
    )

    rates = args.lr_sweep if args.lr_sweep else [args.learning_rate]
    results = []
    for rate in rates:
        print(f"\n=== lr {rate:g} ===", flush=True)
        results.append(
            overfit(
                args.preset,
                batches,
                teacher,
                device,
                args.steps,
                rate,
                args.max_grad_norm,
                args.log_every,
                loss_config,
            )
        )
        torch.cuda.empty_cache()

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"\n{'lr':>9} {'loss drop':>10} {'feat/mean':>10} {'clipped':>8} {'escaped':>8}")
    print("-" * 50)
    for result in results:
        print(
            f"{result['learning_rate']:>9.1e} {result['total_dropped']:>10.3f} "
            f"{result['feature_vs_predict_mean']:>10.3f} "
            f"{result['clipped_fraction']:>8.2f} {str(result['escaped_blank']):>8}"
        )
    print(
        "\nfeat/mean < 1.0 means better than predicting the teacher's mean; "
        "near 1.0 means no representation was learned."
    )


if __name__ == "__main__":
    main()

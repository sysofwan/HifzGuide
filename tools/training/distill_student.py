"""Thin Muaalem student architectures for size distillation, and their ANE sizing.

The deployed model is the 578M-parameter teacher (``obadx/muaalem-model-v3_2``, 24
layers, ``hidden_size=1024``) split into **six** 6-bit CoreML chunks because a single
model of that size exceeds the on-device ANE compiler budget. The split is what keeps
every chunk on the ANE, but it costs six CoreML dispatches and five FP16->FP32 boundary
conversions per window, and the per-window cost is the binding constraint: Muraja issues
**~6 inferences per audio-second** (one full-window pass plus ~5 throttled previews,
``RealtimeTranscriber``'s ``previewMinNewSamples`` = 200 ms), and every one of them pays
the full fixed ``(1, 250, 160)`` cost because the ANE input shape is static. This module
defines the thin students that distillation targets so that cost comes down.

**Thin, not shallow.** ``ml-model-transformation.md`` section 6 records that training-free
depth reduction destroyed this backbone (24 -> 12 layers gave 99.4% CER; even -4 layers
broke it). The speech-distillation literature points the same way: reducing *width* and
keeping depth preserves fine phonetic discrimination far better than reducing depth.
Every preset here therefore keeps all **24 layers** and shrinks ``hidden_size``. Depth
reduction is deliberately not offered as a preset -- it is the riskier axis and the width
presets already reach the latency budget.

**The student carries one head.** The teacher computes 11 CTC heads (phonemes + 10 sifat);
Muraja consumes only ``phonemes`` (43 classes). The students are built with
``level_to_vocab_size={"phonemes": 43}`` alone, so the sifat heads cost no parameters, no
compute, and take no gradient -- the same simplification ``training.waqf_head`` already
makes when it drops them from the graph.

**The adapter is preserved.** The teacher's single stride-2 adapter conv is what maps the
250-frame 20 ms feature lattice to the 125-frame 40 ms CTC lattice. Muraja's decode, the
hop/overlap split, and every downstream fixture assume 125 timesteps, so students keep
``add_adapter=True`` with the teacher's ``num_adapter_layers=1``, ``adapter_kernel_size=3``,
``adapter_stride=2``. A student is a drop-in only if it is exactly
``(1, 250, 160) -> (1, 125, 43)``; :func:`build_student` is the only place that shape
contract is constructed, and :func:`verify_shape_contract` asserts it.

Usage::

    # Print the sizing table (analytic, no torch needed)
    python -m training.distill_student

    # Instantiate each preset and report *measured* parameter counts + shape contract
    python -m training.distill_student --verify

Runs on Linux + CUDA for ``--verify`` (see ``tools/environment.yml``); the analytic
sizing path is torch-free so it can be unit-tested without a GPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace

# The teacher this student is distilled from. Its shape contract -- feature dim, adapter
# geometry, head size -- is what every student must reproduce.
TEACHER_MODEL_ID = "obadx/muaalem-model-v3_2"
TEACHER_HIDDEN_SIZE = 1024
TEACHER_NUM_LAYERS = 24
TEACHER_INTERMEDIATE_SIZE = 4096
TEACHER_NUM_HEADS = 16

# Phoneme head only: 43 classes (``tadabur.phoneme_vocab.NUM_PHONEME_CLASSES``). Duplicated
# as a literal rather than imported because this module must stay importable without the
# tadabur package (and therefore without torch/soundfile) for the analytic sizing path.
PHONEME_LEVEL = "phonemes"
NUM_PHONEME_CLASSES = 43

# The frozen deployed shape contract (``ml-model-transformation.md``, ``convert_to_coreml.py``).
# Fixed by the ANE, which requires fully static shapes.
FEATURE_INPUT_DIM = 160          # stride-2 concatenated 80-bin mel
DEPLOYED_FEATURE_FRAMES = 250    # 5 s at a 20 ms lattice
DEPLOYED_LOGIT_FRAMES = 125      # after the stride-2 adapter -- the 40 ms CTC lattice

# The teacher's adapter geometry, reproduced verbatim so the 250 -> 125 mapping is identical.
ADAPTER_NUM_LAYERS = 1
ADAPTER_KERNEL_SIZE = 3
ADAPTER_STRIDE = 2

# The teacher uses ``relative_key``, which the CoreML trace materialises as one
# ``(250, 64, 250)`` fp16 constant **per layer** -- 4M values each, 96M across 24 layers.
# That cost is fixed by sequence length and head dim, not by model width, so it barely
# registers on the 586M teacher (16%) and dominates a thin student: measured, it was 53% of
# an ``h384`` export's graph and pushed the 6-bit package to 130.3 MB against a predicted
# 71.1 MB. Students are trained from random init, so they are under no obligation to copy
# the teacher's positional scheme; ``rotary`` computes its embedding instead of storing it.
# See :func:`estimate_position_constant_params` and ADR-0010.
DEFAULT_POSITION_EMBEDDINGS = "rotary"
TEACHER_POSITION_EMBEDDINGS = "relative_key"

# Per-layer constant emitted by a traced ``relative_key`` attention: (T, head_dim, T).
RELATIVE_KEY_CONSTANT_SHAPE = (DEPLOYED_FEATURE_FRAMES, 64, DEPLOYED_FEATURE_FRAMES)

# --- On-device size model, over GRAPH CONSTANTS rather than parameters ---
#
# The quantity that determines package size is the number of constant values in the
# converted CoreML graph, which is **not** the model's parameter count: a traced
# ``relative_key`` attention bakes one ``(250, 64, 250)`` constant per layer into the graph
# on top of the weights. Sizing on parameters alone under-predicts badly for a thin model.
#
# Measured directly from two ``h384`` exports:
#   * relative_key: 181.30M graph values (85.3M weights + 96.0M position constants)
#                   -> 130.3 MB at 6-bit  => 0.753 bytes/value
#   * rotary:        85.55M graph values (no position constants)
#                   -> 61.7 MB at 6-bit   => 0.756 bytes/value
# Palettization hits its nominal 6/8 = 0.75 in both cases with no measurable surcharge; the
# 2.1x package-size difference between them is entirely the position constants.
#
# Cross-checking the teacher: 586.4M params + 96.0M position constants = 682.4M values,
# predicting 488 MB against the 504 MB actually measured across its six packages (3% under,
# the gap being per-package metadata paid six times over).
PALETTIZED_6BIT_RATIO = 0.75
BYTES_PER_GRAPH_VALUE_6BIT = PALETTIZED_6BIT_RATIO

# The largest single chunk we have actually compiled onto the iPhone 13 ANE. Chunks A-E are
# 81 MB at 6-bit and chunk F is 99 MB, and all six load (section 3.3). 99 MB is therefore a
# *demonstrated* ceiling, not a specification -- Apple publishes no budget, and section 9
# speculates that ~162 MB merged chunks "may fit". We size against the demonstrated number
# and treat anything above it as needing a split, because a chunk that overflows the budget
# fails **silently** to CPU (section 2.5) rather than raising.
PROVEN_MAX_CHUNK_MB = 99.0


@dataclass(frozen=True)
class StudentSpec:
    """One thin-student architecture, named by its width.

    ``num_hidden_layers`` defaults to the teacher's 24 for every preset: see the module
    docstring on why depth is the axis we do not cut. ``intermediate_size`` holds the
    teacher's 4x FFN ratio so the students differ along exactly one axis, which is what
    makes the agreement-vs-size curve interpretable.
    """

    name: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int = TEACHER_NUM_LAYERS
    position_embeddings_type: str = DEFAULT_POSITION_EMBEDDINGS

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"{self.name}: hidden_size {self.hidden_size} is not divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        if self.num_hidden_layers <= 0 or self.hidden_size <= 0:
            raise ValueError(f"{self.name}: layers and width must be positive")

    @property
    def width_ratio(self) -> float:
        """Student width as a fraction of the teacher's."""
        return self.hidden_size / TEACHER_HIDDEN_SIZE

    @property
    def flop_ratio(self) -> float:
        """Approximate per-window compute vs the teacher.

        Attention projections, both FFN blocks and the conv module all scale with
        ``hidden_size`` squared at fixed sequence length, so at equal depth the ratio is
        ``(h_student / h_teacher)^2``. This ignores the attention score matmul (which
        scales linearly in width, so the true saving is slightly *larger* than this) and
        the chunk-dispatch and FP16->FP32 boundary overhead a single-chunk student
        removes outright. Deliberately an under-estimate of the win.
        """
        depth_ratio = self.num_hidden_layers / TEACHER_NUM_LAYERS
        return (self.width_ratio**2) * depth_ratio


# The sizing ladder. Trained from one shared set of teacher targets so the marginal cost of
# each extra rung is training time only, which is what makes "smallest without significant
# tradeoff" an empirical question rather than a guess.
PRESETS: dict[str, StudentSpec] = {
    "h512": StudentSpec("h512", hidden_size=512, intermediate_size=2048, num_attention_heads=8),
    "h384": StudentSpec("h384", hidden_size=384, intermediate_size=1536, num_attention_heads=6),
    "h256": StudentSpec("h256", hidden_size=256, intermediate_size=1024, num_attention_heads=4),
}

DEFAULT_PRESET = "h384"


def _conformer_layer_params(hidden: int, intermediate: int, conv_kernel: int = 31) -> int:
    """Parameters in one Wav2Vec2-BERT (Conformer) encoder layer.

    The layer is FFN1 -> self-attention -> conv module -> FFN2, each with a layer norm.
    Bias and layer-norm terms are O(hidden) and folded in as the trailing term. This is an
    estimate for the sizing table only; ``--verify`` reports the real count from an
    instantiated model, and the unit tests pin the two against each other.
    """
    ffn = 2 * (2 * hidden * intermediate)      # two macaron FFN blocks, each up+down
    attention = 4 * hidden * hidden            # q, k, v, out projections
    conv_module = 3 * hidden * hidden + conv_kernel * hidden  # pointwise x2 + depthwise
    norms_and_biases = 12 * hidden
    return ffn + attention + conv_module + norms_and_biases


def estimate_params(spec: StudentSpec) -> int:
    """Analytic parameter count for a student, including projection, adapter and head."""
    encoder = spec.num_hidden_layers * _conformer_layer_params(
        spec.hidden_size, spec.intermediate_size
    )
    feature_projection = FEATURE_INPUT_DIM * spec.hidden_size + 2 * FEATURE_INPUT_DIM
    adapter = ADAPTER_NUM_LAYERS * (
        ADAPTER_KERNEL_SIZE * spec.hidden_size * (2 * spec.hidden_size)
    )
    head = spec.hidden_size * NUM_PHONEME_CLASSES + NUM_PHONEME_CLASSES
    return encoder + feature_projection + adapter + head


def estimate_position_constant_params(spec: StudentSpec) -> int:
    """Constants a traced ``relative_key`` attention bakes into the graph.

    One ``(T, head_dim, T)`` fp16 tensor per layer. Note what this does *not* depend on:
    ``hidden_size``. Head dim is 64 for both the teacher (1024/16) and every preset here, so
    the cost is a flat ~4M values per layer whatever the width -- which is why it is
    invisible on the teacher and decisive on a thin student. ``rotary`` computes its
    positional term instead of storing it, so it contributes nothing.
    """
    if spec.position_embeddings_type != TEACHER_POSITION_EMBEDDINGS:
        return 0
    frames, head_dim, _ = RELATIVE_KEY_CONSTANT_SHAPE
    return spec.num_hidden_layers * frames * head_dim * frames


def estimate_graph_constants(spec: StudentSpec, num_params: int | None = None) -> int:
    """Total constant values in the converted graph -- what package size is actually made of."""
    params = estimate_params(spec) if num_params is None else num_params
    return params + estimate_position_constant_params(spec)


def palettized_6bit_mb(num_graph_values: int) -> float:
    """On-device size at 6-bit palettization, per the measured model above."""
    return num_graph_values * BYTES_PER_GRAPH_VALUE_6BIT / (1024 * 1024)


def estimated_chunks(num_graph_values: int, max_chunk_mb: float = PROVEN_MAX_CHUNK_MB) -> int:
    """How many ANE chunks a 6-bit student of this size needs.

    Ceiling division against the largest chunk we have demonstrably compiled. This is a
    planning estimate: the real answer comes from ``compile_models.sh``, which is why
    ``export_student_coreml.py`` exists and why the plan front-loads a throwaway export.
    """
    size_mb = palettized_6bit_mb(num_graph_values)
    return max(1, -(-int(size_mb * 100) // int(max_chunk_mb * 100)))


@dataclass(frozen=True)
class Sizing:
    """The full planning view of one student: parameters, on-device size, chunk count."""

    spec: StudentSpec
    num_params: int
    graph_constants: int
    size_6bit_mb: float
    chunks: int
    flop_ratio: float

    @property
    def speedup(self) -> float:
        """Inverse of the FLOP ratio -- the per-window compute factor vs the teacher."""
        return 1.0 / self.flop_ratio if self.flop_ratio else float("inf")

    def as_dict(self) -> dict:
        return {
            "name": self.spec.name,
            "hidden_size": self.spec.hidden_size,
            "intermediate_size": self.spec.intermediate_size,
            "num_attention_heads": self.spec.num_attention_heads,
            "num_hidden_layers": self.spec.num_hidden_layers,
            "position_embeddings_type": self.spec.position_embeddings_type,
            "num_params": self.num_params,
            "graph_constants": self.graph_constants,
            "size_6bit_mb": round(self.size_6bit_mb, 1),
            "chunks": self.chunks,
            "flop_ratio": round(self.flop_ratio, 4),
            "speedup_vs_teacher": round(self.speedup, 2),
        }


def size_student(spec: StudentSpec, num_params: int | None = None) -> Sizing:
    """Sizing for one student; pass ``num_params`` to size a *measured* count."""
    params = estimate_params(spec) if num_params is None else num_params
    constants = estimate_graph_constants(spec, params)
    return Sizing(
        spec=spec,
        num_params=params,
        graph_constants=constants,
        size_6bit_mb=palettized_6bit_mb(constants),
        chunks=estimated_chunks(constants),
        flop_ratio=spec.flop_ratio,
    )


def teacher_sizing() -> Sizing:
    """The teacher as a row in the same table, for reference."""
    spec = StudentSpec(
        name="teacher",
        hidden_size=TEACHER_HIDDEN_SIZE,
        intermediate_size=TEACHER_INTERMEDIATE_SIZE,
        num_attention_heads=TEACHER_NUM_HEADS,
        num_hidden_layers=TEACHER_NUM_LAYERS,
        position_embeddings_type=TEACHER_POSITION_EMBEDDINGS,
    )
    return size_student(spec)


def build_student_config(spec: StudentSpec):
    """The vendored Muaalem config for a student -- phoneme head only, teacher adapter.

    Imported lazily so the analytic sizing path stays torch-free.
    """
    from tadabur.muaalem import Wav2Vec2BertForMultilevelCTCConfig

    return Wav2Vec2BertForMultilevelCTCConfig(
        level_to_vocab_size={PHONEME_LEVEL: NUM_PHONEME_CLASSES},
        level_to_loss_weight={PHONEME_LEVEL: 1.0},
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        intermediate_size=spec.intermediate_size,
        feature_projection_input_dim=FEATURE_INPUT_DIM,
        position_embeddings_type=spec.position_embeddings_type,
        # The 250 -> 125 adapter, verbatim from the teacher.
        add_adapter=True,
        num_adapter_layers=ADAPTER_NUM_LAYERS,
        adapter_kernel_size=ADAPTER_KERNEL_SIZE,
        adapter_stride=ADAPTER_STRIDE,
        output_hidden_size=spec.hidden_size,
        # Distillation supplies the targets, so the student must see exactly the input the
        # teacher saw. Every source of train-time stochasticity is therefore off.
        hidden_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        # These two default to 0.1 and are easy to miss -- neither shares the "dropout"
        # prefix of the ones above in the teacher's config JSON, so zeroing the obvious
        # four leaves them on. ``final_dropout`` sits directly on the CTC head's input and
        # ``conformer_conv_dropout`` fires in all 24 layers, so together they inject noise
        # on every step against a teacher that is deterministic in eval().
        final_dropout=0.0,
        conformer_conv_dropout=0.0,
        # SpecAugment must be disabled with THIS flag, not by zeroing the probabilities.
        # ``mask_time_prob=0.0`` alone does not disable it: transformers computes
        # ``num_masked_span = max(num_masked_span, min_masks)``, and the teacher config
        # carries ``mask_time_min_masks=2``, so a zero probability still masks two spans of
        # ``mask_time_length=10`` per sequence in train() mode. That is 20 of 250 frames
        # randomised every step, against a teacher running in eval() on clean input -- an
        # unlearnable target at the masked positions, and a different input every step.
        # Measured: with it on, two identical train-mode forwards differed by 1.70, and the
        # student could not overfit 32 fixed windows. The probabilities are zeroed as well
        # so the intent survives anyone flipping this flag back on.
        apply_spec_augment=False,
        mask_time_prob=0.0,
        mask_time_min_masks=0,
        mask_feature_prob=0.0,
        mask_feature_min_masks=0,
    )


def build_student(spec: StudentSpec):
    """Instantiate a randomly-initialised thin student. Lazily imports torch."""
    from tadabur.muaalem import Wav2Vec2BertForMultilevelCTC

    return Wav2Vec2BertForMultilevelCTC(build_student_config(spec))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def verify_shape_contract(model) -> tuple[int, int]:
    """Run one dummy window and assert the student is a drop-in for the deployed shape.

    Returns the observed ``(frames, classes)``. Raises if the student does not map
    ``(1, 250, 160) -> (1, 125, 43)``: anything else cannot replace the teacher without
    changing Muraja's decode and every fixture built on the 40 ms lattice.
    """
    import torch

    model.eval()
    dummy = torch.zeros(1, DEPLOYED_FEATURE_FRAMES, FEATURE_INPUT_DIM)
    with torch.no_grad():
        out = model(dummy, return_dict=True)
    logits = out["logits"][PHONEME_LEVEL]
    _, frames, classes = logits.shape
    if (frames, classes) != (DEPLOYED_LOGIT_FRAMES, NUM_PHONEME_CLASSES):
        raise ValueError(
            f"shape contract violated: got ({frames}, {classes}), "
            f"expected ({DEPLOYED_LOGIT_FRAMES}, {NUM_PHONEME_CLASSES})"
        )
    return frames, classes


def _format_table(rows: list[Sizing]) -> str:
    header = (
        f"{'name':<9} {'pos-emb':<12} {'hidden':>7} {'params':>9} {'graph':>9} "
        f"{'6-bit MB':>9} {'chunks':>7} {'vs teacher':>11}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        speed = "1.0x (ref)" if row.spec.name == "teacher" else f"{row.speedup:.1f}x less"
        lines.append(
            f"{row.spec.name:<9} {row.spec.position_embeddings_type:<12} "
            f"{row.spec.hidden_size:>7} {row.num_params / 1e6:>8.1f}M "
            f"{row.graph_constants / 1e6:>8.1f}M {row.size_6bit_mb:>9.1f} "
            f"{row.chunks:>7} {speed:>11}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thin Muaalem student architectures and their ANE sizing"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="instantiate each preset, report measured parameter counts, and assert the "
        "(1, 250, 160) -> (1, 125, 43) shape contract (needs torch)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="restrict to one preset (default: all)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    specs = [PRESETS[args.preset]] if args.preset else [PRESETS[k] for k in PRESETS]

    measured: dict[str, int] = {}
    if args.verify:
        for spec in specs:
            model = build_student(spec)
            measured[spec.name] = count_parameters(model)
            frames, classes = verify_shape_contract(model)
            print(
                f"[{spec.name}] instantiated: {measured[spec.name]:,} params, "
                f"output ({frames}, {classes}) OK"
            )
            estimated = estimate_params(spec)
            drift = abs(measured[spec.name] - estimated) / measured[spec.name]
            print(f"[{spec.name}] analytic estimate {estimated:,} ({drift:+.1%} drift)")
            del model
        print()

    rows = [teacher_sizing()] + [
        size_student(spec, measured.get(spec.name)) for spec in specs
    ]

    if args.json:
        print(json.dumps([r.as_dict() for r in rows], indent=2, ensure_ascii=False))
        return

    label = "measured" if args.verify else "analytic"
    print(f"Muaalem distillation sizing ladder ({label} parameter counts)")
    print(
        "Size is over GRAPH CONSTANTS, not parameters: a traced relative_key attention "
        "bakes\n~4M values per layer into the graph regardless of width (96M over 24 "
        "layers), which\nis 16% of the teacher and would be 53% of an h384 student. "
        "rotary stores nothing."
    )
    print(f"Chunk count assumes the demonstrated {PROVEN_MAX_CHUNK_MB:.0f} MB ANE ceiling.\n")
    print(_format_table(rows))


if __name__ == "__main__":
    main()

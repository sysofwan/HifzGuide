"""Export a distilled thin student to a single CoreML model, and size it.

The point of size distillation (ADR-0010) is to replace the six-chunk 504 MB pipeline with
**one** model, so this export deliberately produces a single `.mlpackage` rather than the
chunked pipeline `convert_to_coreml.py` builds. It reuses that script's two hard-won ANE
fixes verbatim -- the custom `new_ones`/`new_zeros` op converters and the adapter
attention-mask monkey-patch -- because the student is the same architecture family and hits
exactly the same conversion walls.

**This can be run on an untrained student.** The first question ADR-0010 raises is whether
an `h384` student is one ANE chunk at all, and that depends on the graph and the palettized
size, not on the weights. `--random-init` therefore skips the checkpoint and exports a
freshly-initialised student purely to settle size and conversion -- the one-day de-risk that
should precede a full training run.

**What this can and cannot prove on a Mac.** Conversion, op cleanliness and the palettized
byte count are all settled here. Whether the iPhone 13 ANE compiler accepts the model as a
single chunk is **not**: that budget is enforced at `MLModel` load on the device, and
`ml-model-transformation.md` section 2.5 warns it fails *silently* to CPU rather than
raising. The size reported here against the 99 MB largest-chunk-we-have-shipped is strong
evidence, not proof; the device test is the proof.

**The waqf head is not exported here.** The shipped ChunkF emits `phoneme_logits` *and*
`waqf_logits` (ADR-0004), but the waqf head is a separate track against the same teacher and
adds only ``hidden_size`` parameters per frame. A student that ships will need it grafted on;
for sizing it is noise.

Usage::

    # De-risk: is an untrained h384 one chunk?
    python export_student_coreml.py --preset h384 --random-init --output-dir student_coreml

    # Export a trained student
    python export_student_coreml.py --checkpoint runs/h384/checkpoint.pt \\
        --output-dir student_coreml

macOS only (coremltools targets Apple Silicon; ``--compile`` needs Xcode's
``coremlcompiler``).
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

from training.distill_student import (
    DEPLOYED_FEATURE_FRAMES,
    FEATURE_INPUT_DIM,
    PRESETS,
    PROVEN_MAX_CHUNK_MB,
    build_student,
    count_parameters,
)

PHONEME_LEVEL = "phonemes"


class PhonemesOnlyWrapper(nn.Module):
    """Emit just the phoneme CTC logits, with the attention mask constant-folded away.

    The all-ones mask is a registered buffer so the trace folds every mask-dependent
    branch: the dynamic mask path produces ``gather_nd`` / ``logical_and`` / ``cast``, which
    break ANE compilation. Valid because the deployed pipeline always feeds a full,
    zero-padded 250-frame window -- there is no ragged batch on device.
    """

    def __init__(self, model, seq_len: int) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("fixed_mask", torch.ones(1, seq_len, dtype=torch.long))

    def forward(self, input_features):
        outputs = self.model.wav2vec2_bert(
            input_features, attention_mask=self.fixed_mask, return_dict=True
        )
        hidden_states = outputs[0]
        return self.model.level_to_lm_head[PHONEME_LEVEL](
            self.model.dropout(hidden_states)
        )


def load_student(checkpoint: Path | None, preset: str | None):
    """The student to export, from a checkpoint or freshly initialised."""
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        name = state["config"]["preset"]
        model = build_student(PRESETS[name])
        model.load_state_dict(state["student"])
        print(f"Loaded {name} from {checkpoint} @ step {state['step']}")
        return model, name, state["step"]

    if preset is None:
        raise SystemExit("pass --checkpoint or --preset with --random-init")
    print(f"Building randomly-initialised {preset} (sizing de-risk only)")
    return build_student(PRESETS[preset]), preset, 0


def trace_student(model, traced_path: Path) -> Path:
    """Trace to TorchScript with the adapter's mask computation patched out."""
    from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
        Wav2Vec2BertAdapter,
    )

    model.eval()

    original_forward = Wav2Vec2BertAdapter.forward

    def _adapter_no_mask(self, hidden_states, attention_mask=None):
        return original_forward(self, hidden_states, attention_mask=None)

    Wav2Vec2BertAdapter.forward = _adapter_no_mask
    try:
        wrapper = PhonemesOnlyWrapper(model, DEPLOYED_FEATURE_FRAMES)
        wrapper.eval()
        example = torch.randn(1, DEPLOYED_FEATURE_FRAMES, FEATURE_INPUT_DIM)

        # Warm up before tracing. ``Wav2Vec2BertRotaryPositionalEmbedding`` caches its
        # cos/sin table keyed on sequence length, so the very first forward builds the
        # table and later ones reuse it -- two structurally different graphs. torch.jit's
        # check_trace runs the model twice and would (correctly) report "Graphs differed
        # across invocations". One warmup call settles the cache; the deployed shape is
        # static, so the cache never invalidates afterwards.
        with torch.no_grad():
            wrapper(example)

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (example,))
            reference = wrapper(example)
            got = traced(example)
            drift = (reference - got).abs().max().item()
        print(f"Trace verification -- max abs diff: {drift:.2e}")
        if drift > 1e-3:
            raise SystemExit(f"trace diverged from eager by {drift:.2e}")

        traced.save(str(traced_path))
    finally:
        Wav2Vec2BertAdapter.forward = original_forward

    del traced, wrapper, example
    gc.collect()
    return traced_path


def convert_to_mlpackage(traced_path: Path, output_path: Path):
    """TorchScript -> CoreML mlprogram at FP16 compute precision.

    FP16 is mandatory, not an optimisation: ``ml-model-transformation.md`` section 2.3
    records that FLOAT32 compiles on a Mac and then crashes on device in the Metal graph
    compiler. The ANE is natively FP16.
    """
    import coremltools as ct

    from convert_to_coreml import _register_custom_ops

    _register_custom_ops()

    traced = torch.jit.load(str(traced_path))
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_features",
                shape=(1, DEPLOYED_FEATURE_FRAMES, FEATURE_INPUT_DIM),
            )
        ],
        outputs=[ct.TensorType(name="phoneme_logits")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    model.save(str(output_path))
    del traced
    gc.collect()
    return model


def palettize(mlpackage_path: Path, output_path: Path, nbits: int = 6):
    """K-means palettization, matching ``palettize_chunks.py``'s settings."""
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    model = ct.models.MLModel(str(mlpackage_path))
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=nbits)
    )
    compressed = cto.palettize_weights(model, config)
    compressed.save(str(output_path))
    return compressed


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def compile_model(mlpackage_path: Path, output_dir: Path) -> Path | None:
    """Run Xcode's ``coremlcompiler``, as ``compile_models.sh`` does for the chunks."""
    if shutil.which("xcrun") is None:
        print("[compile] xcrun not found -- skipping")
        return None
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(mlpackage_path), str(output_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[compile] FAILED: {result.stderr.strip()[:400]}")
        return None
    compiled = output_dir / (mlpackage_path.stem + ".mlmodelc")
    return compiled if compiled.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a distilled student to a single CoreML model"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="export an untrained student to settle size/conversion only",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("student_coreml"))
    parser.add_argument("--nbits", type=int, default=6)
    parser.add_argument("--compile", action="store_true", help="also run coremlcompiler")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.random_init and args.checkpoint is None:
        raise SystemExit("pass --checkpoint, or --preset with --random-init")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, preset, step = load_student(
        None if args.random_init else args.checkpoint, args.preset
    )
    num_params = count_parameters(model)
    print(f"Student {preset}: {num_params:,} parameters")

    traced_path = args.output_dir / f"student_{preset}_traced.pt"
    trace_student(model, traced_path)
    del model
    gc.collect()

    fp16_path = args.output_dir / f"MuaalemStudent_{preset}_FP16.mlpackage"
    print("Converting to CoreML...")
    convert_to_mlpackage(traced_path, fp16_path)
    fp16_mb = directory_size_mb(fp16_path)
    print(f"  FP16 mlpackage: {fp16_mb:.1f} MB")

    palettized_path = args.output_dir / f"MuaalemStudent_{preset}_{args.nbits}BIT.mlpackage"
    print(f"Palettizing to {args.nbits}-bit...")
    palettize(fp16_path, palettized_path, args.nbits)
    palettized_mb = directory_size_mb(palettized_path)
    print(f"  {args.nbits}-bit mlpackage: {palettized_mb:.1f} MB")

    compiled_mb = None
    if args.compile:
        compiled = compile_model(palettized_path, args.output_dir)
        if compiled is not None:
            compiled_mb = directory_size_mb(compiled)
            print(f"  compiled .mlmodelc: {compiled_mb:.1f} MB")

    fits = palettized_mb <= PROVEN_MAX_CHUNK_MB
    report = {
        "preset": preset,
        "step": step,
        "random_init": args.random_init,
        "num_params": num_params,
        "fp16_mb": round(fp16_mb, 1),
        f"palettized_{args.nbits}bit_mb": round(palettized_mb, 1),
        "compiled_mb": round(compiled_mb, 1) if compiled_mb else None,
        "proven_max_chunk_mb": PROVEN_MAX_CHUNK_MB,
        "within_proven_chunk_budget": fits,
    }
    (args.output_dir / f"sizing_{preset}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print()
    print(
        f"{preset} at {args.nbits}-bit is {palettized_mb:.1f} MB vs the "
        f"{PROVEN_MAX_CHUNK_MB:.0f} MB largest chunk we have shipped: "
        f"{'WITHIN budget' if fits else 'OVER budget -- expect a split'}"
    )
    print(
        "Note: the iPhone ANE compiler budget is enforced at MLModel load on device and "
        "fails silently to CPU. This size check is evidence, not proof -- confirm on device."
    )


if __name__ == "__main__":
    main()

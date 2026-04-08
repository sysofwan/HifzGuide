"""
Convert muaalem-v3_2 model to CoreML for iPhone 13 (ANE-optimized).

This script:
1. Downloads the original HuggingFace model (obadx/muaalem-model-v3_2)
2. Wraps it to extract only the phonemes CTC head
3. Monkey-patches the adapter to skip dynamic attention mask computation
   (which uses gather_nd, logical_and, cast etc. that break ANE/Metal)
4. Traces with a fixed input shape (100 frames = 2s sliding window)
5. Converts to CoreML (.mlpackage) and creates INT8 quantized version

The resulting model has only ANE-friendly ops: linear, conv, matmul,
layer_norm, softmax, silu, sigmoid, add, mul, reshape, transpose.

No attention_mask input — the model always assumes all frames are valid.
The app feeds fixed 2s windows of audio, so no padding is needed.

Usage:
    python convert_to_coreml.py [--output-dir OUTPUT_DIR] [--skip-quantization]
"""

import argparse
import gc
import os

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Custom op converters for coremltools
# ---------------------------------------------------------------------------

def _register_custom_ops():
    """Register converters for PyTorch ops not yet supported by coremltools."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import (
        _get_inputs,
        _make_fill_op,
        register_torch_op,
    )
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
    )

    _TORCH_DTYPE_TO_STR = {
        0: "uint8", 1: "int8", 2: "int16", 3: "int32", 4: "int32",
        5: "fp16", 6: "fp32", 7: "fp64", 11: "bool",
    }

    def _fill_with_dtype(context, node, fill_value):
        inputs = _get_inputs(context, node)
        size, dtype = inputs[1], inputs[2]
        result = _make_fill_op(size, fill_value, node.name + "_fill")
        if dtype is not None and hasattr(dtype, "val") and dtype.val in _TORCH_DTYPE_TO_STR:
            result = mb.cast(
                x=result, dtype=_TORCH_DTYPE_TO_STR[dtype.val], name=node.name
            )
        context.add(result, node.name)

    if "new_ones" not in _TORCH_OPS_REGISTRY:
        @register_torch_op
        def new_ones(context, node):
            _fill_with_dtype(context, node, 1.0)

    if "new_zeros" not in _TORCH_OPS_REGISTRY:
        @register_torch_op
        def new_zeros(context, node):
            _fill_with_dtype(context, node, 0.0)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class PhonemesOnlyWrapper(nn.Module):
    """Wraps the multi-head CTC model to output only phoneme logits.
    
    Hardcodes attention_mask to all-ones (registered buffer) so the trace
    constant-folds all mask-dependent branches. The adapter's mask computation
    is also bypassed via monkey-patching (see trace_and_save).
    
    Single input: input_features (1, T, 160).
    """

    def __init__(self, hf_model, seq_len):
        super().__init__()
        self.model = hf_model
        self.register_buffer('fixed_mask', torch.ones(1, seq_len, dtype=torch.long))

    def forward(self, input_features):
        output = self.model(
            input_features=input_features, attention_mask=self.fixed_mask
        )
        return output.logits["phonemes"]


# ---------------------------------------------------------------------------
# Step 1: Trace and convert
# ---------------------------------------------------------------------------

def trace_and_save(traced_path="muaalem_phonemes_traced.pt", pruned_model_path=None):
    """Load HF model, wrap for phonemes-only, trace, and save TorchScript.
    
    Monkey-patches the adapter to skip attention mask computation. The adapter
    normally calls _compute_new_attention_mask + create_bidirectional_mask which
    produce gather_nd/logical_and/cast ops that break Metal and ANE compilation.
    Since all frames are always valid (no padding), we can safely skip this.
    
    Args:
        traced_path: Path to save the traced TorchScript model.
        pruned_model_path: If set, load a pruned model from this local directory
                          instead of the full model from HuggingFace.
    """
    from quran_muaalem.modeling.modeling_multi_level_ctc import (
        Wav2Vec2BertForMultilevelCTC,
        Wav2Vec2BertForMultilevelCTCConfig,
    )
    from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
        Wav2Vec2BertAdapter,
    )

    FIXED_SEQ_LEN = 250  # ~5s sliding window

    if pruned_model_path:
        print(f"Loading pruned model from {pruned_model_path}...")
        config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(pruned_model_path)
        model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
            pruned_model_path, config=config
        )
        print(f"Pruned model: {config.num_hidden_layers} encoder layers")
    else:
        print("Loading model from HuggingFace...")
        config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(
            "obadx/muaalem-model-v3_2"
        )
        model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
            "obadx/muaalem-model-v3_2", config=config
        )
    model.eval()

    # Monkey-patch: skip attention mask in adapter (all frames always valid)
    _orig_adapter_fwd = Wav2Vec2BertAdapter.forward
    def _adapter_no_mask(self, hidden_states, attention_mask=None):
        return _orig_adapter_fwd(self, hidden_states, attention_mask=None)
    Wav2Vec2BertAdapter.forward = _adapter_no_mask

    wrapper = PhonemesOnlyWrapper(model, FIXED_SEQ_LEN)
    wrapper.eval()

    example_features = torch.randn(1, FIXED_SEQ_LEN, 160)

    print(f"Tracing model (fixed T={FIXED_SEQ_LEN}, no mask input)...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_features,))
        ref_out = wrapper(example_features)
        traced_out = traced(example_features)
        diff = (ref_out - traced_out).abs().max().item()
        print(f"Trace verification — max abs diff: {diff:.2e}")

    traced.save(traced_path)
    print(f"Traced model saved: {os.path.getsize(traced_path) / 1e6:.0f} MB")

    # Free all PyTorch memory before CoreML conversion
    del model, wrapper, traced, ref_out, traced_out, example_features
    gc.collect()

    return traced_path


def convert_traced_to_coreml(traced_path, output_dir):
    """Convert saved TorchScript to CoreML .mlpackage (FP32 weights).
    
    Uses a fixed input shape so the model can run on the Neural Engine (ANE).
    The ANE requires static shapes — dynamic/flexible shapes fall back to CPU/GPU.
    Fixed at 250 frames (~5s sliding window). Single input (no attention_mask).
    """
    import coremltools as ct

    _register_custom_ops()

    print("Loading traced model...")
    traced = torch.jit.load(traced_path, map_location="cpu")

    FIXED_SEQ_LEN = 250

    print(f"Converting to CoreML (fixed shape: T={FIXED_SEQ_LEN}, single input)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_features",
                shape=(1, FIXED_SEQ_LEN, 160),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="phoneme_logits")],
        minimum_deployment_target=ct.target.iOS17,
        # FP16 compute precision is critical for ANE — ANE natively operates in
        # FP16 and the on-device MLIR compiler fails on FP32 graphs.
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )

    del traced
    gc.collect()

    mlmodel.author = "Converted from obadx/muaalem-model-v3_2"
    mlmodel.short_description = (
        f"Quran phoneme recognition (ANE-optimized). "
        f"Input: (1, {FIXED_SEQ_LEN}, 160) mel features. "
        f"Output: (1, {FIXED_SEQ_LEN // 2}, 43) phoneme logits. "
        f"Fixed {FIXED_SEQ_LEN}-frame window, no attention_mask input."
    )
    mlmodel.version = "3.2"

    os.makedirs(output_dir, exist_ok=True)
    fp32_path = os.path.join(output_dir, "MuaalemPhonemes_FP32.mlpackage")
    print(f"Saving to {fp32_path}...")
    mlmodel.save(fp32_path)
    print(f"FP32 model size: {_dir_size_mb(fp32_path):.1f} MB")

    return fp32_path


# ---------------------------------------------------------------------------
# Step 2: Quantize
# ---------------------------------------------------------------------------

def quantize_model(fp32_path, output_dir):
    """Create INT8 and 4-bit quantized variants from the FP32 CoreML model."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OpPalettizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
        palettize_weights,
    )

    print("\nLoading FP32 model for quantization...")
    mlmodel = ct.models.MLModel(fp32_path)

    # --- INT8 (linear symmetric, weight-only) ---
    print("Applying INT8 linear quantization...")
    int8_config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    int8_model = linear_quantize_weights(mlmodel, config=int8_config)
    int8_path = os.path.join(output_dir, "MuaalemPhonemes_INT8.mlpackage")
    int8_model.save(int8_path)
    print(f"INT8 model size: {_dir_size_mb(int8_path):.1f} MB")
    del int8_model
    gc.collect()

    # --- 4-bit (k-means palettization) ---
    print("Applying 4-bit palettization (this may take 30-60 min)...")
    int4_config = OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=4, mode="kmeans")
    )
    int4_model = palettize_weights(mlmodel, config=int4_config)
    int4_path = os.path.join(output_dir, "MuaalemPhonemes_4BIT.mlpackage")
    int4_model.save(int4_path)
    print(f"4-bit model size: {_dir_size_mb(int4_path):.1f} MB")
    del int4_model, mlmodel
    gc.collect()

    return int8_path, int4_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size_mb(path):
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path) for f in fns
    )
    return total / 1e6


def _print_comparison(output_dir):
    variants = [
        ("FP32", "MuaalemPhonemes_FP32.mlpackage"),
        ("INT8", "MuaalemPhonemes_INT8.mlpackage"),
        ("4-bit", "MuaalemPhonemes_4BIT.mlpackage"),
    ]
    print("\n" + "=" * 50)
    print("Model Size Comparison")
    print("=" * 50)
    print(f"{'Variant':<10} {'Size (MB)':>10} {'vs FP32':>12}")
    print("-" * 34)
    baseline = None
    for name, fname in variants:
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            mb = _dir_size_mb(p)
            if baseline is None:
                baseline = mb
                print(f"{name:<10} {mb:>10.1f} {'baseline':>12}")
            else:
                print(f"{name:<10} {mb:>10.1f} {(1 - mb / baseline) * 100:>11.1f}%")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert muaalem model to CoreML")
    parser.add_argument(
        "--output-dir", default="./coreml_models",
        help="Directory to save CoreML models (default: ./coreml_models)",
    )
    parser.add_argument(
        "--skip-quantization", action="store_true",
        help="Skip INT8 and 4-bit quantization steps",
    )
    parser.add_argument(
        "--pruned-model", default=None,
        help="Path to a pruned model directory (from prune_model.py)",
    )
    args = parser.parse_args()

    # Step 1: Trace -> CoreML FP32
    traced_path = trace_and_save(pruned_model_path=args.pruned_model)
    fp32_path = convert_traced_to_coreml(traced_path, args.output_dir)

    # Clean up traced model
    if os.path.exists(traced_path):
        os.remove(traced_path)

    # Step 2: Quantize
    if not args.skip_quantization:
        quantize_model(fp32_path, args.output_dir)
        _print_comparison(args.output_dir)

    print(f"\nAll models saved to: {args.output_dir}/")
    print("Run verify_coreml.py to compare accuracy against the original model.")


if __name__ == "__main__":
    main()

"""
Verify CoreML model accuracy against the original PyTorch model.

Runs the same input through both models and compares outputs numerically.

Usage:
    python verify_coreml.py [--model-dir ./coreml_models] [--variant FP32]
"""

import argparse
import os
import sys

import numpy as np
import torch


def load_pytorch_model():
    """Load the original HuggingFace model."""
    from quran_muaalem.modeling.modeling_multi_level_ctc import (
        Wav2Vec2BertForMultilevelCTC,
        Wav2Vec2BertForMultilevelCTCConfig,
    )

    print("Loading PyTorch model...")
    config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(
        "obadx/muaalem-model-v3_2"
    )
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        "obadx/muaalem-model-v3_2", config=config
    )
    model.eval()
    return model


def load_coreml_model(model_path, force_cpu=False):
    """Load a CoreML model."""
    import coremltools as ct

    print(f"Loading CoreML model from {model_path}...")
    if force_cpu:
        print("  Using CPU_ONLY compute units")
        return ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    return ct.models.MLModel(model_path)


def generate_test_input(n_frames=499, seed=42):
    """Generate deterministic test input (mel-spectrogram features)."""
    rng = np.random.RandomState(seed)
    features = rng.randn(1, n_frames, 160).astype(np.float32)
    mask = np.ones((1, n_frames), dtype=np.int32)
    return features, mask


def run_pytorch(model, features, mask):
    """Run inference with PyTorch model."""
    with torch.no_grad():
        feat_t = torch.from_numpy(features)
        mask_t = torch.from_numpy(mask).long()
        output = model(input_features=feat_t, attention_mask=mask_t)
        return output.logits["phonemes"].numpy()


def run_coreml(model, features, mask):
    """Run inference with CoreML model."""
    prediction = model.predict({
        "input_features": features,
        "attention_mask": mask,
    })
    return prediction["phoneme_logits"]


def compare_outputs(pytorch_out, coreml_out, label=""):
    """Compare two output arrays and print metrics."""
    # Ensure same shape
    assert pytorch_out.shape == coreml_out.shape, (
        f"Shape mismatch: PyTorch {pytorch_out.shape} vs CoreML {coreml_out.shape}"
    )

    diff = np.abs(pytorch_out - coreml_out)
    max_abs_err = diff.max()
    mean_abs_err = diff.mean()

    # Cosine similarity (flatten)
    a = pytorch_out.flatten()
    b = coreml_out.flatten()
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    # Argmax agreement (CTC decoding would use argmax)
    pt_argmax = pytorch_out.argmax(axis=-1)
    cm_argmax = coreml_out.argmax(axis=-1)
    argmax_agreement = (pt_argmax == cm_argmax).mean() * 100

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Max absolute error:  {max_abs_err:.6f}")
    print(f"{prefix}Mean absolute error: {mean_abs_err:.6f}")
    print(f"{prefix}Cosine similarity:   {cos_sim:.8f}")
    print(f"{prefix}Argmax agreement:    {argmax_agreement:.2f}%")
    print()

    return {
        "max_abs_err": float(max_abs_err),
        "mean_abs_err": float(mean_abs_err),
        "cosine_similarity": float(cos_sim),
        "argmax_agreement": float(argmax_agreement),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify CoreML model accuracy")
    parser.add_argument(
        "--model-dir", default="./coreml_models",
        help="Directory containing CoreML models",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Specific variant to test (FP32, INT8, 4BIT). Tests all if omitted.",
    )
    args = parser.parse_args()

    # Variants to test
    variants = {
        "FP32": "MuaalemPhonemes_FP32.mlpackage",
        "INT8": "MuaalemPhonemes_INT8.mlpackage",
        "4BIT": "MuaalemPhonemes_4BIT.mlpackage",
    }

    if args.variant:
        if args.variant.upper() not in variants:
            print(f"Unknown variant: {args.variant}. Choose from: {list(variants.keys())}")
            sys.exit(1)
        variants = {args.variant.upper(): variants[args.variant.upper()]}

    # Filter to existing models
    variants = {
        k: v for k, v in variants.items()
        if os.path.exists(os.path.join(args.model_dir, v))
    }

    if not variants:
        print(f"No CoreML models found in {args.model_dir}/")
        sys.exit(1)

    # Load PyTorch model
    pt_model = load_pytorch_model()

    # Test with multiple sequence lengths
    test_cases = [
        ("5s audio", 249),
        ("10s audio", 499),
        ("30s audio", 1499),
    ]

    print("\n" + "=" * 60)
    print("CoreML Verification Results")
    print("=" * 60)

    for variant_name, variant_file in variants.items():
        coreml_path = os.path.join(args.model_dir, variant_file)
        # Quantized models may crash GPU/ANE on macOS; use CPU for verification
        force_cpu = variant_name in ("INT8", "4BIT")
        cm_model = load_coreml_model(coreml_path, force_cpu=force_cpu)

        print(f"\n--- {variant_name} ---")
        for test_name, n_frames in test_cases:
            features, mask = generate_test_input(n_frames)
            pt_out = run_pytorch(pt_model, features, mask)
            cm_out = run_coreml(cm_model, features, mask)
            compare_outputs(pt_out, cm_out, label=f"{variant_name}/{test_name}")

        del cm_model

    print("Verification complete.")


if __name__ == "__main__":
    main()

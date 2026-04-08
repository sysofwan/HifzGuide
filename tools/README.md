# Tools

Scripts for model conversion, data processing, and testing the Muaalem inference pipeline.

## Python Scripts

### `convert_to_coreml.py`

Converts the Wav2Vec2-BERT TorchScript model (`obadx/muaalem-model-v3_2`) to CoreML format optimized for Apple Neural Engine. Traces the model with a fixed input shape `(1, 250, 160)`, exports to FP32 `.mlpackage`, and optionally creates INT8 and 4-bit compressed variants.

```bash
python convert_to_coreml.py [--output-dir ./coreml_models] [--skip-quantization] [--pruned-model path/]
```

### `extract_phonemes.py`

Downloads the `obadx/muaalem-annotated-v3` dataset from Hugging Face and reconstructs per-ayah phoneme reference strings. Merges overlapping segments, deduplicates, and selects the most complete variant per ayah.

```bash
python extract_phonemes.py
# Output: ayah_phonemes.json
```

### `palettize_chunks.py`

Applies 6-bit palettization to the six chunked FP32 CoreML model packages using `coremltools`. Skips chunks that already have palettized output.

```bash
python palettize_chunks.py [nbits]  # default: 6
# Input:  coreml_models_chunked/*_FP32.mlpackage
# Output: coreml_models_chunked/*_6BIT.mlpackage
```

### `verify_coreml.py`

Runs identical random inputs through both the PyTorch model and exported CoreML model(s), comparing outputs numerically (max/mean absolute error, cosine similarity, argmax agreement).

```bash
python verify_coreml.py [--model-dir ./coreml_models] [--variant FP32|INT8|4BIT]
```

### `generate_quran_db.py`

Builds the consolidated `quran.db` SQLite database from multiple Quran data sources. Creates tables for surahs, ayahs, words, word-phoneme mapping, mushaf page layout, and ligature mappings.

```bash
python generate_quran_db.py
# Input:  ../data/*.json, ../data/*.db
# Output: ../ios/Muraja/Resources/quran.db
```

### `requirements.txt`

Python dependencies: `torch`, `coremltools`, `transformers`, `huggingface_hub`, `numpy`, `soundfile`, `librosa`.

## Swift Test Scripts

Standalone Swift scripts for testing and debugging the CoreML inference pipeline. Compile with `swiftc -framework CoreML -framework Accelerate`.

| Script | Description |
|--------|-------------|
| `test_features.swift` | Validates mel-spectrogram extraction against Python reference output |
| `test_fixed.swift` | Runs precomputed features through the 6-chunk CoreML pipeline, checks `MLMultiArray` stride correctness |
| `test_decode.swift` | Debugs `MLMultiArray` indexing and CTC decoding behavior |
| `test_e2e.swift` | End-to-end: raw audio → mel features → chunked CoreML → CTC decode |
| `test_e2e_v2.swift` | Same as above with improved float16/contiguity handling |
| `test_full_pipeline.swift` | Full pipeline timing test from audio to decoded phonemes |
| `simulate_v2.swift` | Simulates streaming inference with RMS-based silence detection over al-Fatihah audio |

---

## Deploying Wav2Vec2-BERT to iPhone — Deep Dive

This section documents the end-to-end process of deploying a 24-layer Wav2Vec2-BERT model (~2.4GB FP32) for real-time Quran phoneme recognition on an iPhone 13 (A15 Bionic). The model is from [obadx/muaalem-model-v3_2](https://huggingface.co/obadx/muaalem-model-v3_2), a multi-level CTC model with 11 output heads, of which we use only the phoneme head (43 classes).

The final working solution: a **6-chunk, 6-bit palettized CoreML pipeline** (504MB total) running on a mix of ANE and CPU/GPU, with a fully custom Kaldi-compatible mel spectrogram in Swift using Accelerate.framework.

### 1. Model Conversion (PyTorch → CoreML)

#### 1.1 Initial Challenges

**TorchScript `dictconstruct` error:** The model's forward method returns a dictionary of outputs (`{'phonemes': ..., 'ghonna': ..., ...}`), which coremltools cannot convert directly. We wrapped the model in `PhonemesOnlyWrapper` that extracts only the phoneme logits tensor.

**Unsupported ops (`new_ones`, `new_zeros`):** The attention mask creation uses these ops, which coremltools didn't support. We wrote custom op converters:

```python
@register_torch_op
def new_ones(context, node):
    _fill_with_dtype(context, node, 1.0)
```

**OOM during conversion:** The 2.4GB FP32 model exhausted memory during `ct.convert()`. Solved with a two-step approach: trace → save to disk → free PyTorch memory → load TorchScript → convert to CoreML.

#### 1.2 Final Conversion Pipeline

```python
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input_features", shape=(1, 250, 160))],
    outputs=[ct.TensorType(name="phoneme_logits")],
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,  # Critical for ANE
    convert_to="mlprogram",
)
```

#### 1.3 Quantization Results

| Variant | Size | Exact Match | Char Accuracy |
|---------|------|-------------|---------------|
| FP32 | 2,424 MB | 100% (reference) | 100% |
| INT8 (linear symmetric) | 611 MB | 95.5% | 98.98% |
| 6-bit (k-means palettization) | 504 MB* | 100% vs INT8 | ~same as INT8 |
| 4-bit (k-means palettization) | 306 MB | 87.2% | 96.96% |

\*6-bit was applied to chunked models; total across 6 chunks.

**Recommendation:** 6-bit palettization offers the best size/accuracy tradeoff. It produces identical argmax predictions to INT8 while being 25% smaller.

### 2. ANE Compatibility

#### 2.1 Fixed Shapes (Non-Negotiable)

The ANE requires **completely static tensor shapes**. Any dynamic dimension causes the on-device MIL compiler to bail out with:

```
MIL program has non-constant (dynamic) shapes
```

We fixed the input to `(1, 250, 160)` — 250 stride-2 mel frames = 5 seconds of audio at 16kHz.

#### 2.2 Eliminating Dynamic Ops

The original model had **39 op types** including ANE-hostile operations (`gather_nd`, `logical_and`, `cast`, `select`). These came from:

1. **Encoder attention mask:** Hardcoded an all-ones mask as a `register_buffer` since we always feed full 250-frame windows.

2. **Adapter attention mask:** Monkey-patched `Wav2Vec2BertAdapter.forward` to skip mask computation.

After fixes: **14 clean MIL op types** (linear, conv, matmul, layer_norm, softmax, silu, sigmoid, add, mul, reshape, transpose, split, pad, relu).

#### 2.3 FP16 Compute Precision is Mandatory

Using `compute_precision=ct.precision.FLOAT32` compiles on Mac but **crashes on device**:

```
MPSGraphExecutable.mm:5036: failed assertion 'Error: MLIR pass manager failed'
```

The ANE natively operates in FP16. You **must** use `ct.precision.FLOAT16`.

#### 2.4 The ANE Memory Budget

Even after op-level fixes, the full model (~611MB INT8) exceeds the iPhone 13 ANE compiler memory budget:

```
ANECCompile() FAILED
Error=Couldn't communicate with a helper application
```

#### 2.5 Silent ANE Failures

**This is the most dangerous ANE behavior.** When ANE compilation fails:
- `MLModel(contentsOf:configuration:)` does **not** throw
- The model loads and runs — on **CPU**, not ANE
- There is no public API to query which compute unit is actually being used

You can only detect this via Xcode Instruments (CoreML instrument) or by observing performance.

### 3. Model Chunking Strategy

#### 3.1 Final Pipeline Architecture

```
Input Features (1, 250, 160) float32
    │
    ▼
ChunkA: feature_projection + encoder layers 0-3    →  (1, 250, 1024) float16  [ANE]
    │ copyToFresh (FP16→FP32)
    ▼
ChunkB: encoder layers 4-7                         →  (1, 250, 1024) float16  [ANE]
    │ copyToFresh (FP16→FP32)
    ▼
ChunkC: encoder layers 8-11                        →  (1, 250, 1024) float16  [ANE]
    │ copyToFresh (FP16→FP32)
    ▼
ChunkD: encoder layers 12-15                       →  (1, 250, 1024) float16  [ANE]
    │ copyToFresh (FP16→FP32)
    ▼
ChunkE: encoder layers 16-19                       →  (1, 250, 1024) float16  [ANE]
    │ copyToFresh (FP16→FP32)
    ▼
ChunkF: encoder layers 20-23 + adapter + CTC head  →  (1, 125, 43) float16   [ANE→CPU/GPU fallback]
    │ copyToFresh (FP16→FP32)
    ▼
CTC Greedy Decode → Phoneme IDs → Arabic Characters
```

#### 3.2 Chunk Sizes

| Chunk | FP32 | INT8 | 6-bit |
|-------|------|------|-------|
| A (feat_proj + L0-3) | 215 MB | 108 MB | 81 MB |
| B (L4-7) | 215 MB | 108 MB | 81 MB |
| C (L8-11) | 215 MB | 108 MB | 81 MB |
| D (L12-15) | 215 MB | 108 MB | 81 MB |
| E (L16-19) | 215 MB | 108 MB | 81 MB |
| F (L20-23 + adapter + head) | 263 MB | 132 MB | 99 MB |
| **Total** | **1,338 MB** | **672 MB** | **504 MB** |

On low-storage iPhones, the ANE compiler writes temporary compilation caches (~100-200MB per chunk). We observed `errno = 28` (ENOSPC) errors when device storage was low.

### 4. Feature Extraction — Matching SeamlessM4TFeatureExtractor

#### 4.1 The 9 Critical Parameters

| Parameter | Wrong Initial Value | Correct Value |
|-----------|-------------------|---------------|
| Kaldi scaling | None | `audio × 32768` (simulate 16-bit int) |
| FFT size | 400 | **512** (zero-padded) |
| Preemphasis | None | 0.97 (applied after DC removal) |
| DC offset removal | No | Yes (subtract frame mean) |
| Window function | Standard Hann | **Exact Python periodic Hann** (exported as binary) |
| Mel filterbank | Hand-computed | **Exact Python filterbank** (exported as binary) |
| Mel floor | 1e-10 | **1.192092955078125e-07** (FP32 epsilon) |
| Normalization | None | Per-mel-bin zero-mean unit-variance (ddof=1) |
| Stride-2 concat | Before normalization | **After normalization** |

#### 4.2 Why Export Binary Files?

Rather than reimplementing the Python feature extractor (and risking subtle differences), we exported the exact arrays:

- `mel_filters.bin`: 80×257 float32 matrix (82KB)
- `window.bin`: 400 float32 samples (1.6KB)

These are bundled in the iOS app and loaded at runtime.

#### 4.3 vDSP FFT Scaling

Apple's `vDSP_fft_zrip` returns values **scaled by 2×** compared to the standard DFT definition. The power spectrum is 4× the true value. Fix: divide the power spectrum by 4.

### 5. The FP16 Data Type Bug

#### 5.1 The Symptom

Completely wrong phoneme sequences on device, despite correct mel spectrogram, correct weights, and correct vocabulary.

#### 5.2 The Root Cause

`compute_precision=FLOAT16` causes all CoreML model **outputs** to be FP16. Our `copyToFresh()` used raw `memcpy` treating FP16 bytes as FP32, completely corrupting data. Python's coremltools auto-converts to float32, masking this issue.

#### 5.3 The Fix

```swift
if source.dataType == .float16 {
    for i in 0..<count { dst[i] = source[i].floatValue }
} else {
    memcpy(dst, source.dataPointer, count * MemoryLayout<Float>.size)
}
```

#### 5.4 Lessons Learned

- **Never assume `MLMultiArray.dataPointer` is Float32.** Always check `.dataType`.
- Python's coremltools hides FP16→FP32 conversion, so Python verification may pass while Swift fails.
- The subscript accessor (`array[index].floatValue`) is safe but slower than pointer access.

### 6. Failed Approaches

**Layer Pruning (24 → 12 Layers):** Tried uniform, first-12, and last-12 strategies. All produced garbage output (99.4% CER). Even removing 4 layers destroyed quality. The encoder layers are deeply co-dependent.

**Single Large Model on ANE:** The full 611MB INT8 model consistently exceeds the iPhone 13 ANE compilation budget.

### 7. Performance

| Metric | Value |
|--------|-------|
| Model size (on-device) | 504 MB (6×6-bit chunked) |
| Feature extraction | ~10ms (5s audio → 250×160 features) |
| Inference (6-chunk pipeline) | ~150-200ms per 5s window on iPhone 13 |
| CTC decoding | <1ms |
| Total latency | ~160-210ms per window |

### 8. Key Takeaways

1. **ANE is powerful but unforgiving.** Fixed shapes, FP16 precision, no dynamic ops. And when it fails, it fails silently.
2. **Model chunking works** but introduces complexity. Each chunk boundary is a potential failure point for data type mismatches.
3. **Feature extraction must be exact.** Export reference arrays from Python rather than reimplementing.
4. **FP16 is a hidden landmine.** Python's coremltools auto-converts to float32, masking the issue. Always check `.dataType` in Swift.
5. **Test the full pipeline end-to-end in Swift**, not just in Python.
6. **Pruning pre-trained transformers without fine-tuning doesn't work** for this model.
7. **Palettization (6-bit) gives surprisingly good results** — identical argmax to INT8 with 25% size reduction.

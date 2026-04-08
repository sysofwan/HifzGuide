# ML Model Transformation

Deploying a 24-layer Wav2Vec2-BERT model (~2.4GB FP32) for real-time Quran phoneme recognition on iPhone 13 (A15 Bionic). The model is [obadx/muaalem-model-v3_2](https://huggingface.co/obadx/muaalem-model-v3_2), a multi-level CTC model with 11 output heads, of which we use only the phoneme head (43 classes).

**Final solution:** a 6-chunk, 6-bit palettized CoreML pipeline (504MB total) running on ANE + CPU/GPU, with a custom Kaldi-compatible mel spectrogram in Swift using Accelerate.framework.

## 1. Model Conversion (PyTorch → CoreML)

### 1.1 Initial Challenges

**TorchScript `dictconstruct` error:** The model returns a dictionary of outputs, which coremltools cannot convert. We wrapped it in `PhonemesOnlyWrapper` that extracts only the phoneme logits tensor.

**Unsupported ops (`new_ones`, `new_zeros`):** The attention mask uses these ops. We wrote custom op converters:

```python
@register_torch_op
def new_ones(context, node):
    _fill_with_dtype(context, node, 1.0)
```

**OOM during conversion:** Solved with a two-step approach: trace → save to disk → free PyTorch memory → load TorchScript → convert to CoreML.

### 1.2 Final Conversion Pipeline

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

### 1.3 Quantization Results

| Variant | Size | Exact Match | Char Accuracy |
|---------|------|-------------|---------------|
| FP32 | 2,424 MB | 100% (reference) | 100% |
| INT8 (linear symmetric) | 611 MB | 95.5% | 98.98% |
| 6-bit (k-means palettization) | 504 MB* | 100% vs INT8 | ~same as INT8 |
| 4-bit (k-means palettization) | 306 MB | 87.2% | 96.96% |

**Recommendation:** 6-bit palettization offers the best size/accuracy tradeoff — identical argmax to INT8 while being 25% smaller.

## 2. ANE Compatibility

### 2.1 Fixed Shapes (Non-Negotiable)

The ANE requires **completely static tensor shapes**. Any dynamic dimension causes:

```
MIL program has non-constant (dynamic) shapes
```

Fixed input to `(1, 250, 160)` — 250 stride-2 mel frames = 5 seconds of audio at 16kHz.

### 2.2 Eliminating Dynamic Ops

Original model had **39 op types** including ANE-hostile operations (`gather_nd`, `logical_and`, `cast`, `select`). Sources:

1. **Encoder attention mask:** Hardcoded an all-ones mask as `register_buffer` since we always feed full 250-frame windows.

2. **Adapter attention mask:** Monkey-patched `Wav2Vec2BertAdapter.forward` to skip mask computation.

After fixes: **14 clean MIL op types** (linear, conv, matmul, layer_norm, softmax, silu, sigmoid, add, mul, reshape, transpose, split, pad, relu).

### 2.3 FP16 Compute Precision is Mandatory

Using `compute_precision=FLOAT32` compiles on Mac but **crashes on device**:

```
MPSGraphExecutable.mm:5036: failed assertion 'Error: MLIR pass manager failed'
```

The ANE natively operates in FP16. You **must** use `ct.precision.FLOAT16`.

### 2.4 The ANE Memory Budget

The full model (~611MB INT8) exceeds the iPhone 13 ANE compiler memory budget:

```
ANECCompile() FAILED
Error=Couldn't communicate with a helper application
```

### 2.5 Silent ANE Failures

**Most dangerous ANE behavior:** When compilation fails:
- `MLModel(contentsOf:configuration:)` does **not** throw
- Model loads and runs — on **CPU**, not ANE
- No public API to query actual compute unit
- Only detectable via Xcode Instruments (CoreML instrument) or performance observation

## 3. Model Chunking Strategy

### 3.1 Evolution

| Attempt | Chunks | Result |
|---------|--------|--------|
| Full model | 1 (611MB) | ANE compilation fails |
| 3 chunks (8 layers each) | 3 (~215MB each) | A+B on ANE, C fails |
| 4 chunks (isolating adapter) | 4 | C still fails after A+B consume budget |
| 6 chunks (4 layers each) | 6 (~81MB 6-bit) | A-E on ANE, F falls back to CPU/GPU |

### 3.2 Final Pipeline

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

### 3.3 Chunk Sizes

| Chunk | FP32 | INT8 | 6-bit |
|-------|------|------|-------|
| A (feat_proj + L0-3) | 215 MB | 108 MB | 81 MB |
| B (L4-7) | 215 MB | 108 MB | 81 MB |
| C (L8-11) | 215 MB | 108 MB | 81 MB |
| D (L12-15) | 215 MB | 108 MB | 81 MB |
| E (L16-19) | 215 MB | 108 MB | 81 MB |
| F (L20-23 + adapter + head) | 263 MB | 132 MB | 99 MB |
| **Total** | **1,338 MB** | **672 MB** | **504 MB** |

On low-storage devices, ANE compiler temp caches (~100-200MB per chunk) can cause `errno = 28` (ENOSPC).

## 4. Feature Extraction — Matching SeamlessM4TFeatureExtractor

### 4.1 The 9 Critical Parameters

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

### 4.2 Why Export Binary Files?

Rather than reimplementing (and risking subtle differences), we exported exact arrays from Python:

- `mel_filters.bin`: 80×257 float32 matrix (82KB)
- `window.bin`: 400 float32 samples (1.6KB)

Bundled in the iOS app and loaded at runtime.

### 4.3 vDSP FFT Scaling

Apple's `vDSP_fft_zrip` returns values **scaled by 2×** vs standard DFT. Power spectrum is 4× the true value. Fix: divide by 4.

## 5. The FP16 Data Type Bug

### 5.1 Symptom

Completely wrong phoneme sequences on device, despite correct mel spectrogram, correct weights, and correct vocabulary.

### 5.2 Root Cause

`compute_precision=FLOAT16` causes all CoreML **outputs** to be FP16. Our `copyToFresh()` used raw `memcpy` treating FP16 bytes as FP32, corrupting data. Python's coremltools auto-converts to float32, masking this.

### 5.3 Fix

```swift
if source.dataType == .float16 {
    for i in 0..<count { dst[i] = source[i].floatValue }
} else {
    memcpy(dst, source.dataPointer, count * MemoryLayout<Float>.size)
}
```

### 5.4 Lessons

- **Never assume `MLMultiArray.dataPointer` is Float32.** Always check `.dataType`.
- Python's coremltools hides FP16→FP32 conversion — Python verification passes while Swift fails.
- The subscript accessor is safe but slower. For our tensor sizes (256K values), overhead is acceptable.

## 6. Failed Approaches

**Layer Pruning (24 → 12 layers):** Tried uniform, first-12, and last-12 strategies. All produced garbage (99.4% CER). Even removing 4 layers destroyed quality — encoder layers are deeply co-dependent.

**Single Large Model on ANE:** Full 611MB INT8 model consistently exceeds iPhone 13 ANE compilation budget.

## 7. Performance

| Metric | Value |
|--------|-------|
| Model size (on-device) | 504 MB (6×6-bit chunked) |
| Feature extraction | ~10ms (5s audio → 250×160 features) |
| Inference (6-chunk pipeline) | ~150-200ms per 5s window on iPhone 13 |
| CTC decoding | <1ms |
| Total latency | ~160-210ms per window |

## 8. Key Takeaways

1. **ANE is powerful but unforgiving.** Fixed shapes, FP16 precision, no dynamic ops. Fails silently.
2. **Model chunking works** but introduces data type mismatch risks at chunk boundaries.
3. **Feature extraction must be exact.** Export reference arrays from Python rather than reimplementing.
4. **FP16 is a hidden landmine.** Python auto-converts; Swift gives raw bytes. Always check `.dataType`.
5. **Test the full pipeline end-to-end in Swift**, not just in Python.
6. **Pruning without fine-tuning doesn't work** for this model.
7. **6-bit palettization gives surprisingly good results** — identical argmax to INT8 with 25% size reduction.

## 9. Possible Future Optimizations

- **Reduce chunk count:** ~162MB merged chunks may fit ANE budget
- **4-bit palettization:** ~306MB total, ~8% accuracy loss (may be acceptable for preview)
- **Faster FP16→FP32:** Use `vImageConvert_Planar16FtoPlanarF` from Accelerate
- **Knowledge distillation:** Fine-tuned 12-layer student model
- **FP16 end-to-end:** Avoid all FP16↔FP32 conversions
- **Newer devices:** A17 Pro+ may compile full model as single chunk

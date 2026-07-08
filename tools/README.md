# Tools

Scripts for model conversion, data processing, and Quran database generation.

## Environments

Two separate environments, because the training and export paths target different platforms:

### Linux + CUDA — filtering & fine-tuning (`tools/tadabur/`, `tools/training/`)

Verified on an NVIDIA RTX 5060 Ti (16 GB, **Blackwell / sm_120**). Blackwell requires
CUDA 12.8 PyTorch wheels — a plain PyPI/conda torch will not run on this GPU.

```bash
conda env create -f tools/environment.yml
conda activate hifzguide
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r tools/requirements-train.txt
```

### macOS — CoreML export (`convert_to_coreml.py`, `palettize_chunks.py`, `verify_coreml.py`, `compile_models.sh`)

`compile_models.sh` uses Xcode's `coremlcompiler`, so this path runs on macOS / Apple Silicon.

```bash
pip install -r tools/requirements.txt
```

## Python Scripts

### `tadabur.smoke_decode` (Linux + CUDA)

Walking-skeleton smoke test for the Tadabur filter's PyTorch inference→decode path
(PRD #1, Phase 0). Streams one clip from `FaisaI/tadabur` (no full-corpus download),
resamples it to 16 kHz mono, loads Muaalem (`obadx/muaalem-model-v3_2`, vendored
`Wav2Vec2BertForMultilevelCTC`) in bf16 on the GPU, runs one variable-length forward
pass, and greedy-CTC-decodes the phoneme head to a sanity phoneme string (recording
VRAM footprint, ~1.2 GB backbone).

```bash
cd tools
python -m tadabur.smoke_decode --config-name preview   # small row groups → fast
python -m tadabur.smoke_decode                          # default config (2.4 GB shards)
```

The default config's shards are a single ~2.4 GB / 1000-row Parquet row group, so
streaming even one clip pulls that whole group; `--config-name preview` uses the
dataset's small-row-group preview config for a quick check. The 43-class phoneme
vocabulary (`tadabur.phoneme_vocab`) is asserted to match the live model in
`tadabur/test_phoneme_vocab.py`.

### `convert_to_coreml.py`

Converts the Wav2Vec2-BERT TorchScript model (`obadx/muaalem-model-v3_2`) to CoreML format optimized for Apple Neural Engine. Traces the model with a fixed input shape `(1, 250, 160)`, exports to FP32 `.mlpackage`, and optionally creates INT8 and 4-bit compressed variants.

```bash
python convert_to_coreml.py [--output-dir ./coreml_models] [--skip-quantization] [--pruned-model path/]
```

### `generate_phonemes.py`

Downloads the `obadx/muaalem-annotated-v3` dataset from Hugging Face and reconstructs per-ayah phoneme reference strings. Merges overlapping segments, deduplicates, and selects the most complete variant per ayah.

```bash
python generate_phonemes.py
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

### `compile_models.sh`

Compiles palettized `.mlpackage` models to `.mlmodelc` bundles using Xcode's `coremlcompiler`. The compiled models can be loaded directly on-device without runtime compilation. Creates a `models.zip` for uploading to a GitHub Release.

```bash
bash compile_models.sh [input_dir] [output_dir]
# Default input:  coreml_models_chunked/
# Default output: compiled_models/
# Output zip:     models.zip
```

### `generate_quran_db.py`

Builds the consolidated `quran.db` SQLite database from multiple Quran data sources. Creates tables for surahs, ayahs, words, word-phoneme mapping, mushaf page layout, and ligature mappings.

```bash
python generate_quran_db.py
# Input:  ../data/*.json, ../data/*.db
# Output: quran.db
```

### `requirements.txt`

Python dependencies: `torch`, `coremltools`, `transformers`, `huggingface_hub`, `numpy`, `soundfile`, `librosa`.

---

For a detailed deep dive on the model conversion and ANE deployment process, see [ML Model Transformation](../ml-model-transformation.md).

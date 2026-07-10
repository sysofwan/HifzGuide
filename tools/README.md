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

### `tadabur.filter` (Linux + CUDA)

The Phase 3 filtering pipeline (PRD #1, ADR-0001): the passing-subset generator.
Streams `FaisaI/tadabur` once, resamples each clip to 16 kHz mono, runs **batched**
bf16 GPU inference in one variable-length full-ayah pass (no 250-frame windowing),
greedy-CTC-decodes the phoneme head, and scores each decoded string against the
cached `quran-transcript` reference for its `surah:ayah` (`tadabur.reference_phonemes`)
with the ported `.balanced` gate (`tadabur.scorer`). Passers are appended to a JSONL
manifest — `audio_filename`, `surah:ayah`, `match_ratio`, `ayah_duration_s`, `reciter_id`.

```bash
cd tools
python -m tadabur.filter --manifest passing_subset.jsonl --batch-size 64
python -m tadabur.filter --manifest passing_subset.jsonl --config-name preview --limit 200
```

Filtering is light on VRAM (~1.5 GB), so use a large `--batch-size` for throughput
over the 365k+ clips. The run is **resumable and idempotent**: a sibling
`<manifest>.progress.json` checkpoints how many clips have been scored, so a restart
skips them (rejected clips leave no manifest line but are still skipped), and a
per-`audio_filename` seen-set keeps the manifest duplicate-free if the last in-flight
batch is replayed after a crash.

### `tadabur.waqf_segments` (Linux/macOS — no GPU)

Waqf-aware reference labelling (PRD #1, ADR-0002): turns the passing-subset manifest
into an **offsets manifest** whose per-segment label matches what the reciter *actually*
recited. It needs no model and no GPU — it reads Tadabur's shipped forced alignment
(`metadata.word_alignments`), detects intra-ayah **waqf pauses** as inter-word gaps
(`word[i+1].start - word[i].end ≥ --pause-threshold`, default `0.25 s`), splits each clip
into contiguous **waqf segments**, and phonetizes each segment's Uthmani text on its own so
`quran_phonetizer`'s CleanEnd lands the terminal word in **waqf** form and the interior words
in **wasl**. Segments are lightweight `(start_s, end_s)` views — no per-segment audio and no
derived HF dataset; whole passing clips are kept locally as 16 kHz mono WAV, and the full
Tadabur source is streamed, never landed.

```bash
cd tools
python -m tadabur.waqf_segments --passing passing_subset.jsonl \
    --out segments.jsonl --audio-dir clips/ --config-name preview
```

The output manifest is **deterministic and idempotent** (records sorted by
`(audio_filename, segment_index)` and rewritten atomically). Clips whose alignment word count
disagrees with their Uthmani word count (the vocative `يا` is a separate simple-text word but
merged in Uthmani) or that hit the phonetizer's 8-ayah gap are **skipped and tallied**, never
silently mislabeled. After the build it prints a before/after report on the audit's
shadda-contrast bucket — how many phantom pre-waqf gemination mismatches the realized labels
remove. Feeds P4 data-prep (#8): the manifest is the label source, the reciter split is
computed over the post-segmentation units, and the collator slices audio by these offsets.

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

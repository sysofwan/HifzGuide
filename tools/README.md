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

### `tadabur.waqf_segments` (Linux/macOS — no GPU) — clip staging

Waqf-aware reference labelling (PRD #1, ADR-0002) splits each admitted clip at its intra-ayah
**waqf pauses** and labels each segment in the form the reciter *actually* recited (terminal word
in **waqf** form, interior words in **wasl**), removing phantom pre-waqf gemination mismatches from
the fine-tune data. The work is split across two stages because pause detection needs the model
(the shipped forced alignment *absorbs* pauses into word spans — see ADR-0002).

`tadabur.waqf_segments` is the torch-free half: it **stages** each passing clip as a whole 16 kHz
mono WAV on local disk (the full Tadabur source is streamed, never landed) and owns the shared
realized-reference vocabulary (`SegmentRecord`, `hafs_phonetizer`, `hafs_word_reference`). No model,
no GPU.

```bash
cd tools
python -m tadabur.waqf_segments --passing passing_subset.jsonl \
    --audio-dir clips/ --config-name preview
```

A full build that cannot locate a passing clip in the stream **fails loudly** (a partial clip set
is a data-integrity failure); a `--limit` smoke run instead tallies the unreached clips as
`missing_due_to_limit`.

### `tadabur.segment_score` (Linux — GPU) — model waqf pass + scoring

Owns the model pass end to end. It first runs a dedicated VAD (`obadx/recitation-segmenter-v2`
via `tadabur.vad`) over all clips to find the **waqf pauses** — the interior silences between
speech spans, with silences < `--min-silence-ms` merged away and speech < `--min-speech-ms`
dropped (both from the VAD's training labels). Then, freeing the VAD, it decodes each staged clip's
**whole** waveform once to per-frame phoneme ids and hands them, with the VAD pauses, to
`tadabur.waqf_detect` — along with the ayah's per-word phoneme boundaries (`hafs_word_reference`
derives these from the phonetizer's char `mappings`, robust to wasl word-merges) — which maps each
pause to a word boundary via Smith-Waterman, splitting at a word edge (waqf) but not mid-word (a
stop-consonant closure). Each resulting segment is then decoded again and scored against its
realized reference with the `.balanced` gate (same normalization / Smith-Waterman / contrast
attribution as the full-ayah filter, per segment). Output is one scored segment manifest (carrying
per-segment offsets — the P4 label source) plus each segment's sliced audio for the audit UI,
feeding the audit sampler + UI.

```bash
python -m tadabur.segment_score --passing passing_subset.jsonl --clips-dir clips/ \
    --out-manifest segment_manifest.jsonl --audio-out segment_audio/ \
    [--min-silence-ms 300] [--min-speech-ms 700] [--boundary-tol 3] [--vad-dtype bfloat16]
```

A clip that cannot be segmented safely (`repeated_recitation` / `low_alignment`) is kept whole (one
whole-ayah segment) and tallied; the 8 phonetizer-gap ayat are skipped (`phonetizer_unsupported`).
Feeds P4 data-prep (#8): the manifest is the label source, the reciter split is computed over the
post-segmentation units, and the collator slices audio by these offsets.

### `training.waqf_distill` (Linux — GPU teacher, CPU pooling) — waqf soft labels

The teacher half of the waqf-head distillation (ADR-0004). Runs the same Recitation VAD
(`obadx/recitation-segmenter-v2` via `tadabur.vad`) over the staged clips, but keeps its raw
**per-20 ms silence posteriors** (`P(silence)`, not the cleaned intervals), then **pools them 2:1 to
Muaalem's 40 ms CTC lattice** by a pinned rule: student frame `i` owns teacher frames `2i`/`2i+1`
and is silent iff *both* are (min-pool silence / max-pool speech), left-anchored so a ±few-frame
feature-extractor drift is absorbed at the tail, never by shifting an interior boundary. Because the
deployed model runs **fixed 5 s windows** (250 feature → 125 student frames), the soft targets are
emitted **per training window**: each clip's posteriors are sliced into windows and pooled, keyed to
the passing-subset manifest by `(audio_filename, window_index)`. An even window start lands on the
clip's 40 ms lattice (`start // 2`), so the per-window mapping is independent of window *spacing* —
which is exactly what lets this run before the inference-window contract (#24, overlap/edge/stitch)
is frozen; the window length is the deployed 5 s and the spacing defaults to a **provisional
non-overlapping tiling** (`--hop-feature-frames`). Output goes into a deterministic, idempotent
`SoftLabelStore` (per-window `.npy` arrays + a `soft_labels.jsonl` index, one line per clip listing
its windows). Generation **streams one clip at a time** and fsyncs each clip before the next, so a
crash mid-run keeps every clip already written and a resumed run skips them — the whole manifest is
never held in memory. The pooling/windowing/alignment is torch-free and covered by golden fixtures
(`training/test_waqf_distill.py`); only the VAD forward pass needs the GPU. The windowed collator
(#8) consumes these per-window targets against the phoneme lattice.

```bash
cd tools
python -m training.waqf_distill --manifest passing_subset.jsonl --clips-dir clips/ \
    --out-dir waqf_soft_labels/ [--window-feature-frames 250] [--hop-feature-frames 250] \
    [--device cuda] [--dtype bfloat16] [--batch-size 8]
```

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

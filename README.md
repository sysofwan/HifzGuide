# HifzGuide

Assets and data generation tools for [Muraja](https://github.com/sysofwan/Muraja) — a real-time Quran follow-along reading checker for iOS.

## Contents

- **`tools/`** — Python scripts for data generation, model conversion, filtering, and fine-tuning
  - **`tools/tadabur/`** — Tadabur quality-filtering pipeline (Linux + CUDA)
  - **`tools/training/`** — LoRA fine-tuning of the Muaalem phoneme head + evaluation harness (Linux + CUDA)
- **`data/`** — Source data files (Quran text, phonemes, mushaf layout, ligatures, fonts)
- **`docs/adr/`** — Architecture Decision Records
- **`CONTEXT.md`** — Canonical domain glossary
- **`ml-model-transformation.md`**, **`quran-database.md`** — Pipeline write-ups

## Release Assets

Pre-built assets are published as [GitHub Releases](https://github.com/sysofwan/HifzGuide/releases). Each release contains:

| Asset | Description | Size |
|-------|-------------|------|
| `models.zip` | 6-chunk Wav2Vec2-BERT CoreML models (6-bit palettized) | ~443 MB |
| `fonts.zip` | 604 QCF2 page fonts + decorative fonts | ~130 MB |
| `quran.db` | SQLite database with Quran text, phonemes, word mappings | ~48 MB |
| `mel_filters.bin` | Mel spectrogram filter bank | 80 KB |
| `window.bin` | Audio windowing function | 1.6 KB |
| `manifest.json` | Version metadata with SHA256 checksums | ~1 KB |

The Muraja iOS app downloads these assets on first launch.

## Environments

This repo has two separate Python environments because the workstreams target different platforms.

### Linux + CUDA — filtering & fine-tuning (`tools/tadabur/`, `tools/training/`)

Verified on an NVIDIA RTX 5060 Ti (16 GB, **Blackwell / sm_120**). Blackwell requires
CUDA 12.8 PyTorch wheels — a plain PyPI/conda `torch` will not run on this GPU, so PyTorch is
installed separately from the `cu128` index.

```bash
conda env create -f tools/environment.yml
conda activate hifzguide
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r tools/requirements-train.txt
```

### macOS — CoreML export (`convert_to_coreml.py`, `palettize_chunks.py`, `verify_coreml.py`, `compile_models.sh`)

`compile_models.sh` uses Xcode's `coremlcompiler`, so the export path runs on macOS / Apple Silicon.

```bash
pip install -r tools/requirements.txt
```

## Regenerating Assets

`generate_quran_db.py` requires `quran-transcript` (bundled in the Linux + CUDA env; without it
the phoneme tables are written empty):

```bash
conda activate hifzguide          # or: pip install quran-transcript
python tools/generate_quran_db.py # builds quran.db from data/ sources
```

CoreML export (macOS) is a multi-step pipeline — trace → convert → **chunk** → palettize →
compile — not a two-command flow. See [ml-model-transformation.md](ml-model-transformation.md)
for the full sequence and `tools/README.md` for each script's inputs/outputs.

## Acknowledgements

- **[Muaalem Model](https://huggingface.co/obadx/muaalem-model-v3_2)** — Arabic phoneme recognition model by [obadx](https://huggingface.co/obadx), fine-tuned from Meta's [Wav2Vec2-BERT](https://huggingface.co/facebook/w2v-bert-2.0) for Quranic recitation analysis
- **[Quranic Universal Library (QUL)](https://qul.tarteel.ai/)** by [Tarteel AI](https://tarteel.ai/) — QPC V2 page fonts (QCF2), word-by-word glyph mappings, mushaf layout data, and Quran text resources

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

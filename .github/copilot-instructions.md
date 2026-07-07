# Copilot Instructions for HifzGuide

## Project Overview

HifzGuide holds the **assets and data-generation tools** for the HifzGuide / Muraja Quran
recitation checker. It is a **Python + data** repo — there is no application UI here. The iOS app
that consumes these assets lives in the separate (closed-source) `sysofwan/Muraja` repo.

- **`tools/`** — Python scripts and shell harnesses for data extraction, DB generation, and ML
  model conversion (PyTorch → CoreML → palettized chunks).
- **`data/`** — source data (Quran text, phonemes, mushaf layout, ligatures, fonts).
- **Root docs** — `README.md`, `ml-model-transformation.md`, `quran-database.md`.

Built assets are published as [GitHub Releases](https://github.com/sysofwan/HifzGuide/releases)
(`quran.db`, `models.zip`, `fonts.zip`, `mel_filters.bin`, `window.bin`, `manifest.json`).

## Domain Language

See [`CONTEXT.md`](../CONTEXT.md) for the full glossary. Key rules:

- Use **"surah"** not "chapter", **"ayah"** not "verse", **"page"** always means mushaf page (1–604).
- **"Phoneme"** means a model output token (Arabic character class, 43 CTC classes), not a
  linguistic phoneme.
- **"Muaalem"** is the upstream Wav2Vec2-BERT model; this repo uses only its **phoneme head**.
- **"Phonetizer"** = `quran_phonetizer` from `quran-transcript`, which turns Uthmani text into the
  phonetic reference labels.
- **"Chunk"** / **"palettization"** refer to the CoreML model-conversion pipeline (see
  `ml-model-transformation.md`).
- Architecture decisions are documented in [`docs/adr/`](../docs/adr/) (created lazily).

## Repository Structure

```
tools/                 # Python generators & model-conversion scripts
  generate_quran_db.py   # Build quran.db from data/ sources
  generate_phonemes.py   # Phonetize Uthmani text → ayah phonemes
  convert_to_coreml.py   # PyTorch Muaalem → CoreML .mlpackage
  palettize_chunks.py    # 6-bit k-means quantization of model chunks
  compile_models.sh      # Compile .mlpackage → .mlmodelc, zip → models.zip
  verify_coreml.py       # Validate converted models
  requirements.txt
data/                  # Source data (Quran text, phonemes, mushaf layout, ligatures, fonts)
CONTEXT.md             # Domain glossary — canonical terminology
docs/adr/              # Architecture Decision Records
ml-model-transformation.md  # PyTorch → CoreML pipeline write-up
quran-database.md      # quran.db schema & generation
```

## Setup & Regenerating Assets

```bash
cd tools && pip install -r requirements.txt

# Regenerate quran.db from source data
python generate_quran_db.py

# Regenerate ayah phonemes (requires: pip install quran-transcript)
python generate_phonemes.py

# Convert ML model to CoreML, then palettize
python convert_to_coreml.py
python palettize_chunks.py
```

Publishing a new asset release is handled by the **`hifzguide-asset-release`** skill.

## Conventions

### Python (`tools/`)
- Target Python 3; prefer the standard library plus the pinned deps in `tools/requirements.txt`.
- Scripts are self-contained CLIs with a `main()` and a module docstring describing usage.
- Use `pathlib.Path`, `argparse`, and `json`/`sqlite3` from the stdlib.
- Resolve paths relative to the script (`Path(__file__).parent`), not the CWD.
- Write UTF-8 explicitly (`encoding="utf-8"`, `ensure_ascii=False`) — the data is Arabic text.
- Keep generators deterministic and idempotent; re-running should reproduce identical output.

### Data & assets
- `data/` holds source-of-truth inputs; generated artifacts (`quran.db`, models) are release
  assets, not hand-edited.
- Never edit `quran.db` by hand — regenerate it via `generate_quran_db.py`.
- Any asset change must go through a new release with a freshly generated `manifest.json`
  (checksums must match the actual files).

### Git
- Include the standard co-author trailer on commits:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`

## Agent Skills

Agent skills live in `.github/skills/`. See [`AGENTS.md`](../AGENTS.md) for the index and the
issue-tracker / triage-label / domain-doc conventions in `docs/agents/`.

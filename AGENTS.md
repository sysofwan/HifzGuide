# Agent Instructions for HifzGuide

> **Canonical agent instructions.** This file is the single source of truth for
> both Claude Code and GitHub Copilot. Copilot loads `AGENTS.md` natively;
> Claude Code loads it via the `@AGENTS.md` import in `CLAUDE.md`. Shared skills
> live in `.claude/skills/` — the only skills directory both harnesses read.
> All paths below are relative to the repository root.

## Project Overview

HifzGuide holds the **assets and data-generation tools** for the HifzGuide / Muraja Quran
recitation checker. It is a **Python + data** repo — there is no application UI here. The iOS app
that consumes these assets lives in the separate (closed-source) `sysofwan/Muraja` repo, so
nothing iOS-specific belongs here.

- **`tools/`** — Python scripts and shell harnesses for data extraction, DB generation, quality
  filtering, fine-tuning, and ML model conversion (PyTorch → CoreML → palettized chunks).
- **`data/`** — source data (Quran text, phonemes, mushaf layout, ligatures, fonts).
- **Root docs** — `README.md`, `ml-model-transformation.md`, `quran-database.md`.

Built assets are published as [GitHub Releases](https://github.com/sysofwan/HifzGuide/releases)
(`quran.db`, `models.zip`, `fonts.zip`, `mel_filters.bin`, `window.bin`, `manifest.json`).

## Domain Language

See [`CONTEXT.md`](CONTEXT.md) for the full glossary — read it before naming domain concepts.
See [`docs/agents/domain.md`](docs/agents/domain.md) for how skills should consume domain docs
and ADRs. Key rules:

- Use **"surah"** not "chapter", **"ayah"** not "verse", **"page"** always means mushaf page (1–604).
- **"Phoneme"** means a model output token (Arabic character class, 43 CTC classes), not a
  linguistic phoneme.
- **"Muaalem"** is the upstream Wav2Vec2-BERT model; this repo uses only its **phoneme head**.
- **"Phonetizer"** = `quran_phonetizer` from `quran-transcript`, which turns Uthmani text into the
  phonetic reference labels.
- **"Chunk"** / **"palettization"** refer to the CoreML model-conversion pipeline (see
  `ml-model-transformation.md`).
- Architecture decisions are documented in [`docs/adr/`](docs/adr/) (created lazily).

## Repository Structure

```
tools/                    # Python generators, filtering & training tools
  tadabur/                # Tadabur quality-filtering pipeline (Linux + CUDA)
  training/               # LoRA fine-tune of the Muaalem phoneme head + eval (Linux + CUDA)
  environment.yml         # conda env for Linux + CUDA
  requirements-train.txt  # Linux + CUDA filtering/training deps
  requirements.txt        # macOS-only CoreML export deps
  generate_quran_db.py generate_phonemes.py convert_to_coreml.py
  palettize_chunks.py verify_coreml.py compile_models.sh
data/                     # Source data (Quran text, phonemes, mushaf layout, ligatures, fonts)
docs/adr/                 # Architecture Decision Records
docs/agents/              # Issue-tracker, triage-label & domain-doc conventions
.claude/skills/           # Agent skills — read by Claude Code and Copilot CLI
CONTEXT.md                # Canonical domain glossary
ml-model-transformation.md  # PyTorch → CoreML pipeline write-up
quran-database.md         # quran.db schema & generation
```

## Environments

Two environments target different platforms (see `tools/README.md` for full setup):

- **Linux + CUDA** — filtering & fine-tuning. GPU is Blackwell (**sm_120**), so PyTorch **must**
  come from the CUDA 12.8 (`cu128`) wheel index; a default `torch` will not run. Use the conda
  env `hifzguide` (Python 3.11) via `tools/environment.yml` + `tools/requirements-train.txt`.
- **macOS** — CoreML export (`compile_models.sh` needs Xcode's `coremlcompiler`). Use
  `tools/requirements.txt`.

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

- Include the co-author trailer for the harness that made the commit:
  - Claude Code: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
  - Copilot: `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`

## Agent Skills

Agent skills live in `.claude/skills/` — the one skills directory both Claude Code and the
Copilot CLI read. Each has a `SKILL.md` (and optional reference files).

### Planning & documentation

- **`grill-with-docs`** — stress-test a plan against the project's domain model and documented
  decisions; sharpen terminology and update `CONTEXT.md` / ADRs inline as decisions crystallise.
- **`to-prd`** — turn the current conversation context into a PRD and publish it to the issue tracker.
- **`to-issues`** — break a plan, spec, or PRD into independently-grabbable issues using
  tracer-bullet vertical slices.
- **`handoff`** — compact the current conversation into a handoff document for another agent.

### Issue workflow

- **`triage`** — move issues through the triage state machine (see triage-label vocabulary).

### Code quality & architecture

- **`improve-codebase-architecture`** — find deepening opportunities and propose refactors that
  turn shallow modules into deep ones (testability, AI-navigability).
- **`thermo-nuclear-code-quality`** — write code that meets a very strict maintainability bar.
- **`thermo-nuclear-code-quality-review`** — run that strict maintainability review.

### Release

- **`hifzguide-asset-release`** — publish a new GitHub asset release (`quran.db`, models, fonts,
  `manifest.json`) with correct checksums.

### Tooling

- **`launch-audit-ui`** — serve the Tadabur poison-audit web UI (`tadabur.audit_ui`) so it is
  reachable from other machines on the LAN (`--host 0.0.0.0`, fully detached).

## Agent Workflows

### Issue tracker

Issues and PRDs live as GitHub Issues on `sysofwan/HifzGuide`. Use the `gh` CLI for all
operations. See [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md).

### Triage labels

Canonical triage roles map to concrete repo labels in
[`docs/agents/triage-labels.md`](docs/agents/triage-labels.md).

### AFK dual-agent loop

`scripts/ralph-loop.sh` picks up `ready-for-agent` issues and runs a code→review loop. Claude Code
runs every role; the Copilot CLI runs only the GPT-5.6 Sol reviewer, so the review pair stays
cross-vendor.

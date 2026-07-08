## Agent skills

Agent skills live in `.github/skills/`. Each has a `SKILL.md` (and optional reference files).

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

## Conventions

### Domain language

Read `CONTEXT.md` at the repo root — the canonical glossary — before naming domain concepts.
See `docs/agents/domain.md` for how skills should consume domain docs and ADRs.

### Issue tracker

Issues and PRDs live as GitHub Issues on `sysofwan/HifzGuide`. Use the `gh` CLI for all
operations. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical triage roles map to concrete repo labels in `docs/agents/triage-labels.md`.

## Repo notes

HifzGuide is a **Python + data** repo (asset generation, no application UI). The consuming iOS
app lives in the separate `sysofwan/Muraja` repo — nothing iOS-specific belongs here. See
`.github/copilot-instructions.md` for Python conventions.

### Structure

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
CONTEXT.md                # Canonical domain glossary
```

### Environments

Two environments target different platforms (see `tools/README.md` for full setup):

- **Linux + CUDA** — filtering & fine-tuning. GPU is Blackwell (**sm_120**), so PyTorch **must**
  come from the CUDA 12.8 (`cu128`) wheel index; a default `torch` will not run. Use the conda
  env `hifzguide` (Python 3.11) via `tools/environment.yml` + `tools/requirements-train.txt`.
- **macOS** — CoreML export (`compile_models.sh` needs Xcode's `coremlcompiler`). Use
  `tools/requirements.txt`.

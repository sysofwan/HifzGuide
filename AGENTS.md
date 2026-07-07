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

HifzGuide is a **Python + data** repo (asset generation, no application UI). See
`.github/copilot-instructions.md` for setup, structure, and Python conventions. The consuming iOS
app lives in the separate `sysofwan/Muraja` repo — nothing iOS-specific belongs here.

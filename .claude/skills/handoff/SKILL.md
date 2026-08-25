---
name: handoff
description: Compact the current conversation into a handoff document for another agent to pick up. Use when ending a session and wanting to preserve context for the next agent.
argument-hint: "What will the next session be used for?"
---

# Handoff Skill

Write a handoff document summarising the current conversation so a fresh agent can continue the work. Save to the temporary directory of the user's OS — **not** the current workspace.

## Behaviour

1. **Summarise** the current session: what was discussed, what was decided, what was implemented, and what remains.
2. **Save** the document to the OS temp directory:
   - macOS: `$TMPDIR` (e.g., `/var/folders/.../T/`)
   - Linux: `/tmp/`
   - Windows: `%TEMP%`
   - Filename format: `handoff-YYYY-MM-DD-HHMMSS.md`
3. **Include a "Suggested Skills" section** listing skills the next agent should invoke (from `.claude/skills/`), with a one-line rationale for each.
4. **Do not duplicate** content already captured in other artifacts (PRDs, plans, ADRs in `docs/adr/`, issues, commits, diffs). Reference them by path or URL instead.
5. **Redact** any sensitive information such as API keys, passwords, tokens, or personally identifiable information.
6. If the user passed **arguments**, treat them as a description of what the next session will focus on and tailor the document accordingly — emphasise relevant context and suppress unrelated details.

## Document Structure

```markdown
# Handoff — [brief title]

**Date:** YYYY-MM-DD HH:MM
**Previous session focus:** [one-line summary]
**Next session focus:** [from user args, or "Continue current work"]

## Context

[2-4 sentences of essential background. Link to CONTEXT.md for domain terms.]

## What Was Done

- [Completed item, with file paths or commit refs]
- ...

## What Remains

- [ ] [Outstanding task — include enough detail to act on]
- ...

## Key Decisions

- [Decision made, with ADR path if applicable]
- ...

## Relevant Files

| File | Why it matters |
|------|---------------|
| `path/to/file` | Brief reason |

## Suggested Skills

| Skill | Rationale |
|-------|-----------|
| `skill-name` | Why the next session should use it |

## Notes

[Anything else the next agent needs to know — gotchas, environment state, blocked items.]
```

## Guidelines

- Keep the document **concise** — aim for under 200 lines. A fresh agent needs orientation, not a transcript.
- Use **relative paths** from the repo root for file references.
- For commits and PRs, include the short SHA or PR number as a clickable reference.
- If nothing meaningful was accomplished (e.g., just exploration), say so honestly — don't pad the document.
- Always print the full path to the saved file so the user can find it.

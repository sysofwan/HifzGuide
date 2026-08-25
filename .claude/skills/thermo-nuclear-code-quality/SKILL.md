---
name: thermo-nuclear-code-quality
description: Write code that passes the thermo-nuclear code quality review bar — pragmatic structural quality plus ML-correctness (reproducibility, data integrity, numerics). Applies the review standards proactively during implementation.
disable-model-invocation: true
---

# Thermo-Nuclear Code Quality — Implementation Standards

These are the coding standards enforced by the review agent. Write code that meets this bar on the first pass.

This is a Python + data / ML repo (data-generation, filtering, model fine-tuning, CoreML export). Hold code to a **demanding but pragmatic** bar: be ambitious about structure, but respect legitimate ML idioms and put the highest priority on the correctness issues that actually break models.

## ML-code calibration (read first)

These are **not** automatically smells — do not contort the code to avoid them:

- **Long, linear training/data scripts are often fine.** A top-to-bottom `main()` that loads → transforms → trains → evaluates is readable *because* it is linear. Don't shatter it into tiny helpers or a class hierarchy to satisfy taste.
- **Hyperparameters and constants are data, not magic.** Learning rates, thresholds, layer targets, batch/vocab sizes are expected literals — prefer a config/argparse/constants block, but their presence is not a defect.
- **Some duplication is acceptable** when it keeps two experiments or pipeline stages independently legible. Extract shared logic to remove real drift risk, not merely to satisfy DRY.
- **Vectorization vs. legibility is a real trade-off.** Judge by clarity and correctness, not dogma.

## Mindset

Before writing code, ask: **"Is there a code-judo move that makes this dramatically simpler?"**

- Look for restructurings that preserve behavior while making the implementation dramatically simpler, smaller, more direct, and more elegant.
- Prefer the solution that makes the code feel inevitable in hindsight.
- Delete complexity rather than rearrange it.
- If you find yourself adding branching, conditionals, or helper layers, step back and ask if the design itself should change.

## Non-Negotiable Standards

0. **Be ambitious about structural simplification.**
   - Do not settle for "it works" if a cleaner structure is visible.
   - If whole branches, helpers, modes, conditionals, or layers can disappear through a better design, pursue that.
   - The best implementation is the one where complexity never existed in the first place.

1. **Watch file growth, but treat size as advisory — not a hard gate.**
   - A file growing large is a signal to *ask* whether it should be decomposed, not an automatic threshold.
   - If a change mixes several unrelated concerns, extract helpers, subcomponents, or modules.
   - A long but cohesive, top-to-bottom pipeline/training script can legitimately run long — judge by whether a reader can still follow it, not by a line count.

2. **Do not add spaghetti to existing code.**
   - No ad-hoc conditionals scattered in unrelated flows.
   - No "weird if statements in random places."
   - If your feature needs a conditional in an existing path, ask whether it belongs behind its own abstraction, helper, state machine, or policy object instead.

3. **Prefer cleaning the design over shipping messy working code.**
   - If behavior can stay the same while the structure becomes meaningfully cleaner, do the cleaner version.
   - Strongly prefer simplifications that remove moving pieces altogether over refactors that merely spread the same complexity around.

4. **Write direct, boring, maintainable code.**
   - No brittle, ad-hoc, or "magic" behavior.
   - No generic mechanisms that hide simple data-shape assumptions.
   - No thin abstractions, identity wrappers, or pass-through helpers that add indirection without buying clarity.
   - If the straightforward approach works, use it.

5. **Keep types and boundaries clean.**
   - Avoid unnecessary optionality or cast-heavy code when a clearer type boundary exists.
   - Prefer explicit typed models or shared contracts over loosely-shaped ad-hoc objects.
   - Make invariants explicit rather than papering over them with silent fallbacks.

6. **Keep logic in the canonical layer. Reuse existing helpers.**
   - Before writing a new helper, check if one already exists.
   - Feature logic belongs behind its own abstraction, not scattered through shared paths.
   - Implementation details must not leak through APIs.
   - Put code in the package/module/layer that already owns the concept.

7. **Avoid unnecessary sequential orchestration.**
   - If independent work can run in parallel without adding complexity, let it.
   - If related updates can leave state half-applied, restructure for atomicity.

## ML-Specific Standards (highest priority for this repo)

These prevent the bugs that silently ruin models and datasets. Get them right first.

8. **Reproducibility and determinism.**
   - Set and thread seeds (Python, NumPy, torch) where determinism is expected.
   - Keep generators idempotent — re-running should reproduce identical output. Pin ordering (sorted keys, stable iteration); don't rely on set/dict insertion accidents.

9. **Data integrity — no leakage, no silent label corruption.**
   - Split train/val/test so the same source unit (e.g. reciter/ayah) can't straddle splits.
   - Never let val/test statistics leak into training (fit normalizers on train only).
   - Treat reference labels/phonemes as source-of-truth; validate transforms rather than mutating them in place.

10. **Device / dtype / numerical correctness.**
    - Be explicit about device and dtype; don't move tensors across devices inside hot loops by accident.
    - Handle NaN/Inf, padding, and ignore indices (`-100`) deliberately.
    - Respect the cu128/Blackwell (sm_120) constraint — never reintroduce a default-index torch path.

11. **Resource and failure honesty on long runs.**
    - Fail fast and loudly on bad data/config; don't swallow exceptions in training/eval loops.
    - Checkpoint and log enough to resume and diagnose. Stay within the VRAM budget.

12. **Correctness parity with references.**
    - When porting logic (e.g. the Swift scorer), validate against fixtures before trusting output.
    - Preserve train-vs-inference preprocessing parity.

## Self-Check Before Committing

Before you commit, verify:

- [ ] Is there a simpler design I missed? Could I delete whole categories of complexity?
- [ ] Did I add branching where a better abstraction should exist?
- [ ] Is each piece of logic living in the right file and layer?
- [ ] Did I enlarge a file to the point it mixes unrelated concerns? (Size alone is fine if it stays one clear linear flow.)
- [ ] Are there repeated conditionals that signal a missing model or helper?
- [ ] Is the implementation direct and legible, or does it rely on special cases?
- [ ] Did I introduce unnecessary optionality or ad-hoc object shapes?
- [ ] Am I reusing canonical helpers, or did I create a near-duplicate?
- [ ] Is feature logic properly isolated, not leaking into shared paths?
- [ ] Would a reader understand this code without excessive context?

For ML / data changes:

- [ ] Is the run reproducible — seeds set, ordering pinned, generation idempotent?
- [ ] Could any split leak the same source unit across train/val/test, or corrupt reference labels?
- [ ] Are device, dtype, NaN/Inf, and padding/`-100` handled deliberately (and no non-cu128 torch path)?
- [ ] Did I validate any ported logic against fixtures and preserve train-vs-inference parity?
- [ ] Does this respect the documented decisions (ADR-0001, VRAM budget)?

## What NOT To Do

- Do not add one-off booleans, nullable modes, or flags that complicate existing control flow.
- Do not bolt new conditionals onto unrelated code paths.
- Do not create thin wrappers or identity abstractions that add indirection without simplifying.
- Do not copy-paste logic — extract a helper.
- Do not implement narrow edge-case handling in the middle of an already busy function.
- Do not create bespoke helpers where the codebase already has a canonical utility.
- Do not add logic in the wrong layer when there is a clear canonical home.
- Do not serialize independent work for no good reason.

## When In Doubt

If you're unsure between two approaches:

1. Pick the one with fewer concepts a reader must hold in their head.
2. Pick the one that deletes complexity rather than moving it.
3. Pick the one that makes the surrounding code simpler, not just the new code.
4. If both are roughly equal, pick the more boring/direct one.

---
name: thermo-nuclear-code-quality
description: Write code that passes the thermo-nuclear code quality review bar. Applies the same structural quality standards proactively during implementation rather than reactively during review.
disable-model-invocation: true
---

# Thermo-Nuclear Code Quality — Implementation Standards

These are the coding standards enforced by the review agent. Write code that meets this bar on the first pass.

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

1. **Do not push a file from under 1k lines to over 1k lines.**
   - If your change would cross this threshold, extract helpers, subcomponents, or modules first.
   - Keep files focused on a single cohesion boundary.
   - Use `// MARK: -` sections to organize code within files.

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

## Self-Check Before Committing

Before you commit, verify:

- [ ] Is there a simpler design I missed? Could I delete whole categories of complexity?
- [ ] Did I add branching where a better abstraction should exist?
- [ ] Is each piece of logic living in the right file and layer?
- [ ] Did I enlarge a file past a healthy size boundary? Should I extract?
- [ ] Are there repeated conditionals that signal a missing model or helper?
- [ ] Is the implementation direct and legible, or does it rely on special cases?
- [ ] Did I introduce unnecessary optionality or ad-hoc object shapes?
- [ ] Am I reusing canonical helpers, or did I create a near-duplicate?
- [ ] Is feature logic properly isolated, not leaking into shared paths?
- [ ] Would a reader understand this code without excessive context?

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

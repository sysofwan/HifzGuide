---
name: thermo-nuclear-code-quality-review
description: Run a pragmatic maintainability + ML-correctness review for abstraction quality, spaghetti growth, and the reproducibility/data/numerical issues that matter most in ML code. Use for a thermo-nuclear code quality review, thermonuclear review, deep code quality audit, or ML code review.
disable-model-invocation: true
source: https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/thermo-nuclear-code-quality-review/SKILL.md
---

# Thermo-Nuclear Code Quality Review

Use this skill for a demanding but **pragmatic** review focused on implementation quality, maintainability, abstraction quality, and — for ML code — correctness of the things that actually break models: reproducibility, data handling, and numerics.

This skill should push the reviewer to be **ambitious** about code structure: look for "code judo" moves that preserve behavior while making the implementation dramatically simpler, smaller, and more direct. But it must also be **calibrated to the domain**. This is a Python + data / ML repo (data-generation, filtering, model fine-tuning, CoreML export). ML code has legitimate idioms — linear training scripts, hyperparameter constants, config dictionaries, occasional duplication for experiment clarity — that are NOT automatically smells. Reserve the harshest escalation for genuine structural regressions and for the ML-correctness issues in the section below.

## ML-code calibration (read first)

Before flagging, distinguish "this violates general software taste" from "this will actually cause a bug or a maintenance problem here":

- **A long, linear training/data script is often fine.** A top-to-bottom `main()` that loads → transforms → trains → evaluates is readable *because* it is linear. Do not demand it be shattered into many tiny helpers or a class hierarchy.
- **Hyperparameters and constants are data, not magic.** Learning rates, thresholds, layer targets, batch sizes, and vocab sizes are expected literals. Prefer they live in a config/argparse/constants block, but their presence is not a "magic number" defect.
- **Some duplication is acceptable** when it keeps two experiments or two pipeline stages independently legible. Extract shared logic when it removes real risk of drift, not merely to satisfy DRY.
- **Vectorization vs. legibility is a real trade-off.** Do not force clever one-liners; do not force loops either. Judge by clarity and correctness, not dogma.
- **Notebooks/throwaway scripts** are held to a lower structural bar than library code under `tools/tadabur/` or `tools/training/`.

## Core Prompt

Start from this baseline:

> Perform a deep code quality audit of the current branch's changes.
> Rethink how to structure / implement the changes to meaningfully improve code quality without impacting behavior.
> Work to improve abstractions, modularity, reduce Spaghetti code, improve succinctness and legibility.
> Be ambitious, if there is a clear path to improving the implementation that involves restructuring some of the codebase, go for it.
> Be extremely thorough and rigorous. Measure twice, cut once.

## Non-Negotiable Additional Standards

Apply the baseline prompt above, plus these explicit review rules:

0. **Be ambitious about structural simplification.**
   - Do not stop at "this could be a bit cleaner."
   - Look for opportunities to reframe the change so that whole branches, helpers, modes, conditionals, or layers disappear entirely.
   - Prefer the solution that makes the code feel inevitable in hindsight.
   - Assume there is often a "code judo" move available: a re-organization that uses the existing architecture more effectively and makes the change dramatically simpler and more elegant.
   - If you see a path to delete complexity rather than rearrange it, push hard for that path.

1. **Watch file growth, but treat size as advisory — not a hard gate.**
   - A file growing large is a signal to *ask* whether it should be decomposed, not an automatic blocker.
   - Prefer extracting helpers, subcomponents, or modules when a file mixes several unrelated concerns.
   - A long but cohesive, top-to-bottom pipeline/training script can legitimately exceed a few hundred lines; judge by whether a reader can still follow it, not by a line count.
   - Only escalate size when the length reflects tangled responsibilities, not when it reflects one clear linear flow.

2. **Do not allow random spaghetti growth in existing code.**
   - Be highly suspicious of new ad-hoc conditionals, scattered special cases, or one-off branches inserted into unrelated flows.
   - If a change adds "weird if statements in random places", treat that as a design problem, not a stylistic nit.
   - Prefer pushing the logic into a dedicated abstraction, helper, state machine, policy object, or separate module instead of tangling an existing path.
   - Call out changes that make the surrounding code harder to reason about, even if they technically work.

3. **Bias toward cleaning the design, not just accepting working code.**
   - If behavior can stay the same while the structure becomes meaningfully cleaner, push for the cleaner version.
   - Do not rubber-stamp "it works" implementations that leave the codebase messier.
   - Strongly prefer simplifications that remove moving pieces altogether over refactors that merely spread the same complexity around.

4. **Prefer direct, boring, maintainable code over hacky or magical code.**
   - Treat brittle, ad-hoc, or "magic" behavior as a code-quality problem.
   - Be skeptical of generic mechanisms that hide simple data-shape assumptions.
   - Flag thin abstractions, identity wrappers, or pass-through helpers that add indirection without buying clarity.

5. **Push hard on type and boundary cleanliness when they affect maintainability.**
   - Question unnecessary optionality, `unknown`, `any`, or cast-heavy code when a clearer type boundary could exist.
   - Prefer explicit typed models or shared contracts over loosely-shaped ad-hoc objects.
   - If a branch relies on silent fallback to paper over an unclear invariant, ask whether the boundary should be made explicit instead.

6. **Keep logic in the canonical layer and reuse existing helpers.**
   - Call out feature logic leaking into shared paths or implementation details leaking through APIs.
   - Prefer existing canonical utilities/helpers over bespoke one-offs.
   - Push code toward the right package, service, or module instead of normalizing architectural drift.

7. **Treat unnecessary sequential orchestration and non-atomic updates as design smells when the cleaner structure is obvious.**
   - If independent work is serialized for no good reason, ask whether the flow should run in parallel instead.
   - If related updates can leave state half-applied, push for a more atomic structure.
   - Do not over-index on micro-optimizations, but do flag avoidable orchestration complexity that makes the implementation more brittle.

## ML-Specific Standards (highest priority for this repo)

For ML / data code these correctness concerns often matter **more** than structural taste. Flag them aggressively — a clean abstraction that silently corrupts training data or breaks on the GPU is worse than a slightly messy one that is correct.

8. **Reproducibility and determinism.**
   - Seeds set for `random`, `numpy`, and `torch` (and `torch.cuda`) where results must be repeatable.
   - No dependence on undefined ordering (e.g. iterating a set/dict for label assignment) that changes vocab/index mappings across runs.
   - Data generation stays deterministic and idempotent — re-running reproduces identical output (repo convention).

9. **Data integrity — no leakage, no silent label corruption.**
   - Train/val/test splits do not leak (e.g. same reciter or same ayah across splits when the design says split by reciter).
   - Reference/label construction matches the agreed source (e.g. `quran_phonetizer` Hafs config, the ported `.balanced` scorer) — a wrong label pipeline poisons the model.
   - Filtering/scoring thresholds and denominators match the documented decision (see ADR-0001); flag silent deviations.

10. **Device / dtype / numerical correctness.**
    - Tensors are on the expected device; no accidental CPU⇄GPU round-trips in hot loops.
    - dtype is intentional (bf16 autocast for compute, fp32 where precision matters, e.g. loss/label accumulation); no silent dtype mismatch.
    - Guard against NaN/Inf where it can arise (log/exp, division, empty sequences); no `-100`/padding label mishandling in CTC loss.
    - Blackwell/`cu128` assumptions are respected — do not reintroduce a plain `torch` install path that cannot run on sm_120.

11. **Resource and failure honesty on long runs.**
    - Streaming datasets are not silently materialized in full (memory blowups); batching respects the stated VRAM budget.
    - Long jobs are resumable/checkpointed where the issue calls for it; failures surface rather than being swallowed by bare `except`.
    - Config/hyperparameters are logged so a run is auditable and repeatable.

12. **Correctness parity with references.**
    - Ports (e.g. the Muraja scorer → Python) are validated against fixtures, not eyeballed.
    - Feature extraction / preprocessing matches train-vs-inference parity (same processor, same normalization).

## Primary Review Questions

For every meaningful change, ask:

- Is there a "code judo" move that would make this dramatically simpler?
- Can this change be reframed so fewer concepts, branches, or helper layers are needed?
- Does this improve or worsen the local architecture?
- Did the diff add branching complexity where a better abstraction should exist?
- Did a previously cohesive module become more coupled, more stateful, or harder to scan?
- Is this logic living in the right file and layer?
- Did this change enlarge a file or component past a healthy size boundary?
- Are there repeated conditionals that signal a missing model or missing helper?
- Is the implementation direct and legible, or does it rely on special cases and incidental control flow?
- Is this abstraction actually earning its keep, or is it just a wrapper?
- Did the diff introduce casts, optionality, or ad-hoc object shapes that obscure the real invariant?
- Is this logic living in the canonical layer, or did the diff leak details across a boundary?
- Is this orchestration more sequential or less atomic than it needs to be?

And for ML / data changes specifically:

- Could this leak data between train/val/test, or corrupt labels/references?
- Is the run reproducible (seeds, deterministic ordering, idempotent generation)?
- Are device, dtype, and numerical edge cases (NaN/Inf, padding/`-100`) handled correctly?
- Does preprocessing preserve train-vs-inference parity, and are ports validated against fixtures?
- Does this respect the documented decisions (ADR-0001, the cu128/Blackwell setup, VRAM budget)?

## What to Flag Aggressively

Escalate findings when you see:

- A complicated implementation where a cleaner reframing could delete whole categories of complexity.
- Refactors that move code around but fail to reduce the number of concepts a reader must hold in their head.
- A file crossing a healthy size boundary due to the PR *when the added code mixes unrelated concerns* and could be split out.
- New conditionals bolted onto unrelated code paths.
- One-off booleans, nullable modes, or flags that complicate existing control flow.
- Feature-specific logic leaking into general-purpose modules.
- Generic "magic" handling that hides simple structure and makes the code harder to reason about.
- Thin wrappers or identity abstractions that add indirection without simplifying anything.
- Unnecessary casts, `any`, `unknown`, or optional params that muddy the real contract.
- Copy-pasted logic instead of extracted helpers.
- Narrow edge-case handling implemented in the middle of an already busy function.
- Refactors that technically pass tests but make the code less modular or less readable.
- "Temporary" branching that is likely to become permanent debt.
- Bespoke helpers where the codebase already has a canonical utility for the job.
- Logic added in the wrong layer/package when it should live somewhere more central.
- Sequential async flow where obviously independent work could stay simpler and clearer with parallel execution.
- Partial-update logic that leaves state less atomic than necessary.

## Preferred Remedies

When you identify a code-quality problem, prefer suggestions like:

- Delete a whole layer of indirection rather than polishing it.
- Reframe the state model so conditionals disappear instead of getting centralized.
- Change the ownership boundary so the feature becomes a natural extension of an existing abstraction.
- Turn special-case logic into a simpler default flow with fewer exceptions.
- Extract a helper or pure function.
- Split a large file into smaller focused modules.
- Move feature-specific logic behind a dedicated abstraction.
- Replace condition chains with a typed model or explicit dispatcher.
- Separate orchestration from business logic.
- Collapse duplicate branches into a single clearer flow.
- Delete wrappers that do not meaningfully clarify the API.
- Reuse the existing canonical helper instead of introducing a near-duplicate.
- Make type boundaries more explicit so the control flow gets simpler.
- Move the logic to the package/module/layer that already owns the concept.
- Parallelize independent work when that also simplifies the orchestration.
- Restructure related updates into a more atomic flow when partial state would be harder to reason about.

Do not be satisfied with "maybe rename this" feedback when the real issue is structural.
Do not be satisfied with a merely cleaner version of the same messy idea if there is a plausible path to a much simpler idea.

## Review Tone

Be direct, serious, and demanding about quality, but proportionate. Do not soften major correctness or maintainability issues into mild suggestions — and equally, do not inflate domain-appropriate ML idioms (linear scripts, hyperparameter literals, pragmatic duplication) into blockers. Lead with the highest-impact findings; skip cosmetic nits when the code is correct and readable.
If the code is making the codebase messier, or risks a reproducibility/data/numerical bug, say so clearly.
If the implementation missed an opportunity for a dramatic simplification, say that clearly too.

Good phrases:

- `this could grow into a mixed-concern file. worth splitting the scoring out from the io?`
- `this split by index risks leaking the same reciter across train/val — should the split be by reciter?`
- `no seed is set here; two runs will produce different vocab ordering. can we pin it?`
- `this tensor stays on cpu inside the batch loop — intended, or should it move to the gpu once?`
- `this port isn't checked against the swift fixtures — can we assert parity before trusting the score?`

## Output Expectations

Prioritize findings in this order:

1. ML-correctness defects — data leakage, label/reference corruption, non-reproducibility, device/dtype/numerical bugs, train-vs-inference skew
2. Behavioral correctness and regressions
3. Structural code-quality regressions and spaghetti / branching complexity increases
4. Missed opportunities for dramatic simplification / code-judo restructuring
5. Boundary / abstraction / type-contract problems that make the code harder to reason about
6. Modularity, decomposition, and file-organization concerns (advisory)
7. Legibility and maintainability concerns

Do not flood the review with low-value nits if there are larger structural issues.
Prefer a smaller number of high-conviction comments over a long list of cosmetic notes.

## Approval Bar

Approve when the change is correct and the structure is reasonable for ML/data code — do not withhold approval over domain-appropriate idioms or purely cosmetic decomposition preferences. The bar for approval is:

- no ML-correctness defect (data leakage, corrupted labels/references, non-reproducibility, device/dtype/numerical bug, train-vs-inference skew)
- no clear behavioral regression
- no clear structural regression or spaghetti-growth from special-case branching
- no obviously hacky or magical abstraction that makes the code harder to reason about
- no clear architecture-boundary leak or avoidable canonical-helper duplication
- no obvious missed opportunity to make the implementation dramatically simpler when such a path is clearly visible

Treat these as presumptive blockers unless the author can justify them clearly:

- an ML-correctness defect: split leakage, wrong/unvalidated label or reference pipeline, missing seeds where determinism is required, NaN/Inf or `-100`/padding mishandling, dtype/device errors, or a reintroduced non-`cu128` torch path
- a silent deviation from a documented decision (ADR-0001, VRAM budget, feature-extractor parity)
- ad-hoc branching that makes an existing flow materially more tangled
- solving a local problem by scattering feature checks across shared code
- an unnecessary abstraction/wrapper that makes the design more indirect without buying clarity
- duplicating an existing canonical helper or putting logic in clearly the wrong layer

Do NOT treat these as blockers on their own (raise as advisory suggestions at most):

- a file simply being long while remaining a cohesive, linear pipeline/training flow
- hyperparameter/threshold literals, config dicts, or seed constants
- limited duplication that keeps two stages/experiments independently legible
- a straightforward imperative script that could *in principle* be an elaborate abstraction

If a presumptive-blocker condition is met, leave explicit, actionable feedback and push for the fix. Otherwise, approve.

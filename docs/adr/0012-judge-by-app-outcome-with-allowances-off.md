---
status: accepted
amends: ADR-0011 §1, §2, §5
---

# Judge a model by the app outcome with the allowance switched off

## Context

The owner's problem statement (2026-10-09, verbatim):

> The model today is too strict and incorrectly flags correctly recited words. That's why the in
> app soft pair and tashkeel override is required. The goal for this is to increase accuracy and
> prevent the false incorrects while correctly flagging if something is actually wrong.

It has three goals:
1. interchanged soft pairs (with `ذ↔ظ`);
2. incorrect shaddah;
3. sukun and missing tashkeel are treated the same.

For goal 3, in the owner's words: "there is no way to differentiate between sukun and unknown
vowels. Model emits nothing on unclear vowels."

The owner was asked which allowances the model should make unnecessary. They chose:
- soft-pair forgiveness;
- shaddah suppression;
- the Tashkeel toggle, which users turn off.

ADR-0011 modelled Muraja as grading one decode per word. A reading of Muraja `99c326f` (v1.0.27)
showed otherwise:
- every full window and every ~200 ms preview regrades each word;
- `GradeStore` keeps only the best grade (`GradeStore.swift:182-207`).

So a correctly recited word is flagged only if *every* decode is wrong, and a real mistake is
hidden by a single decode that reads as the mushaf. Today balanced forgives almost every soft-pair
swap (word score ≥ 0.65) and every collapsed in-word geminate, so today's catch rate for both is
close to zero.

## Decision

- **What a candidate is judged on.** It is judged on the **app outcome**: Muraja's grading at a
  pinned commit, simulated over each clip's window and preview decodes, **with the allowance under
  test switched off**.
- **A required co-gate.** Per-decode fidelity guards (`docs/acceptance-rules.md` §2: no new
  per-letter errors) are required **as well**, not only diagnostics. The best-grade rule can reward
  a model that flip-flops between decodes.
- **Allowances in scope, and the bars.** The owner sets every number before any candidate is read,
  after seeing h448's baseline.

  | Allowance | Goal | False flags on correctly recited words | Real mistakes flagged |
  |---|---|---|---|
  | Soft-pair forgiveness (pooled; per-direction harm checks) | 1 | Allowance off: **lower** than h448's today with it on, by at least a set minimum reduction | At least a set **absolute floor** |
  | Shaddah suppression | 2 | Same as soft pairs. Tashkeel flags newly exposed by un-collapsed geminates count. | Same as soft pairs |
  | The Tashkeel toggle (tashkeel shown in balanced) | 3 | At most an absolute ceiling N per 1,000 correctly recited words | At least h448's |

- **Out of scope.** These allowances stay on in both baseline and candidate:
  - the missing-haraka exemptions (و ا ء ي, the final letter of a waqf word, geminate gaps);
  - `.lenient`'s `suppressHarakaDrop`.

  The owner: "#3 is usually caused by elongation and not really accurate on which letter appears.
  The goal for this training is not to make this more accurate." That answer is about **these
  exemptions**, not about goal 3. This amends ADR-0011 §2: the sukun class no longer *replaces*
  the per-letter exemption.
- **Meaning of an empty slot.** "Empty slot = not graded" (Ship B) means the decode **abstains**:
  it changes no grade. It never counts as correct, because one unsure decode would then lock in a
  real dropped haraka.
- **Two ships:**
  - **Ship A** is goals 1–2. It keeps the same vocabulary. The owner verifies h448-flagged sites by
    listening; fine-tune on the verified labels, keeping both outcomes, with a train/held-out split
    by reciter. A random word sample with whole-word verdicts is the model-independent denominator
    for false flags. Muraja then switches allowances off per the bars.
  - **Ship B** is goal 3. It adds the sukun class together with the matching Muraja change, and
    warm-starts from Ship A's model.
- **Muraja's grading is held fixed** at the pinned commit, apart from those allowance flips and
  Ship B's sukun change. Changes to the ratchet are a separate decision.

## Consequences

- **Superseded.** The "Ship 1" plan is superseded: commit block b=1 and "stop grading empty
  haraka". So are the acceptance rules that depend on the single-decode model
  (`docs/acceptance-rules.md` §3's system definition and headline, §4's allowance populations, §5,
  and §8's power target).
- **Pooled soft pairs.** Muraja has one switch for all soft pairs, so pooling them replaces
  §4's "per pair, never pooled". Per-direction "insufficient evidence" rows (§7) and harm checks
  stay.
- **Simulator fidelity.** The Python simulator is only as good as its parity with Muraja. It is
  checked by replaying recorded window and preview sequences through Muraja's own engine (query
  assembly and ratchet), not only its per-word grader.
- **Evaluation populations.** Training labels may be mined from sites the model flagged.
  Evaluation populations stay model-independent, or are stratified by outcome with known
  inclusion probabilities (§1).

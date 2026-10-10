---
status: accepted
amends: ADR-0011 §1, §5
---

# Judge a model by the app outcome with the allowance switched off

**Context.** ADR-0011 framed the problem as dropped tashkeel and modelled Muraja as grading one
decode per word. A reading of Muraja `99c326f` (v1.0.27) showed otherwise:
- every full window and every ~200 ms preview regrades each word;
- `GradeStore` keeps only the best grade (`GradeStore.swift:182-207`).

So a correctly recited word is flagged only if *every* decode is wrong, and a real mistake is
hidden by a single decode that reads as the mushaf. The owner restated the problem on 2026-10-09:
the model flags correctly recited words, which is why Muraja carries soft-pair and tashkeel
allowances. The goal is a model accurate enough to drop them while real mistakes are still flagged.
There are three goals:
1. interchanged soft pairs (with `ذ↔ظ`);
2. incorrect shaddah;
3. sukun and missing tashkeel treated the same.

## Decision

- **What a candidate is judged on.** It is judged on the **app outcome**: Muraja's grading at a
  pinned commit, simulated over the window and preview decodes of each clip, **with the allowance
  under test switched off**. Per-decode fidelity (ADR-0011 §5) stays as the diagnostic that
  explains why an app outcome moved.
- **Allowances in scope, and the bar each must clear:**

  | Allowance | Goal | Bar |
  |---|---|---|
  | Soft-pair forgiveness | 1 | Allowance off: false flags ≤ h448's today with it on (within a pre-registered margin); real-mistake catch rate higher than today's. Pooled across the pairs, plus a per-pair harm check. |
  | Shaddah suppression | 2 | Same bar as soft pairs. |
  | The Tashkeel toggle | 3 | An absolute ceiling on false tashkeel flags per 1,000 correctly recited words, set by the owner before any candidate is read. Catch of real dropped or wrong haraka ≥ h448's. |

  The owner sets the margins and the ceiling after seeing h448's baseline, before any candidate.
- **Out of scope.** These allowances stay on in both baseline and candidate:
  - the missing-haraka exemptions (و ا ء ي, the final letter of a waqf word, geminate gaps);
  - `.lenient`'s `suppressHarakaDrop`.

  They mostly cover elongation, where the decoded letter is unreliable, and this training does not
  aim to fix that.
- **Two ships:**
  - **Ship A** is goals 1–2. It keeps the same vocabulary. The owner verifies h448-flagged sites
    by listening; fine-tune on the verified labels, keeping both outcomes, with a train/held-out
    split by reciter. Muraja then switches allowances off per the bar.
  - **Ship B** is goal 3. It adds the sukun class together with the matching Muraja change, and
    warm-starts from Ship A's model.
- **Muraja's grading is held fixed** at the pinned commit, apart from those allowance flips and
  Ship B's sukun change. Changes to the ratchet are a separate decision.

## Consequences

- **Superseded.** The "Ship 1" plan is superseded: commit block b=1 and "stop grading empty
  haraka", which were built on the single-decode model. So are the acceptance rules that depend
  on that model (`docs/acceptance-rules.md` §3's system definition and headline, §4's allowance
  populations, §5, and §8's power target).
- **Simulator fidelity.** A Python simulator of Muraja's grading is only as good as its parity
  with Muraja. It is pinned to a commit and checked against Muraja's own `checkRecitation` harness.
- **Evaluation populations.** Training labels may be mined from sites the model flagged.
  Evaluation populations stay model-independent, or are stratified by outcome with known
  inclusion probabilities (`docs/acceptance-rules.md` §1).

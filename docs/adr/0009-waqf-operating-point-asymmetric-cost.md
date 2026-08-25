# The waqf operating point is chosen against an asymmetric cost, not F1

**Status:** Accepted. Makes operative the "calibrated bound" language in
[ADR-0004](0004-waqf-head-and-joint-whole-clip-fine-tune.md), which was never built, and governs
the calibration leg of #59. Touches only how the silence-posterior threshold is *chosen*; the
metrics, the partitions and the 300/700 ms duration gate are unchanged.

## The two errors are not equally costly, and the current objective prices them the same

`tadabur.waqf_event_eval.CALIBRATION_OBJECTIVE` reads:

> Tune the silence posterior threshold … to **maximise waqf F1**, tie-broken by the lower
> false-waqf@wasl rate (the more damaging error) then the threshold nearest the VAD's 0.5 argmax.

F1 is symmetric in precision and recall: it prices a spurious waqf and a missed waqf the same.
The tie-break names the asymmetry but only fires on an *exact* F1 tie, so in practice the
symmetric term decides and the asymmetry is decorative.

The asymmetry is not a matter of taste. It is mechanical, and it is in the shipped scorer:

| error | what the scorer does | grade | recoverable? |
|---|---|---|---|
| **false-waqf@wasl** — fires at a boundary the reciter ran through | selects the **pausal** reference, so the boundary word's ending haraka and cross-word idgham leave the comparison | the word grades `.correct` | **No.** Correct grades "pass through immediately so the ratchet locks them in" (`QuranFollowAlong+WordScoring.swift`), and the ratchet rank is "higher = better, **never degrade**" (`FollowAlongTypes.swift`) |
| **false-wasl@stop** — misses a genuine pause | selects the **continuation** reference, so a legitimate pausal ending reads as a dropped ending | non-correct | **Yes.** The end word is held back as `.pending`; "premature grades upgrade naturally as more audio becomes available" |

So one error is **absorbing** and the other is **transient**. A false-waqf permanently forgives a
real i'raab or idgham mistake for that session; a false-wasl produces a penalty that the hold-back
and ratchet are designed to walk back.

ADR-0004 already identified the direction — *"a spurious fire lets the scorer forgive a dropped
haraka / missed idgham — the discrimination ADR-0001 is trying to regain"* — and then never turned
it into a selection rule. This ADR is that rule.

### ADR-0004 and the implementation also simply disagree

ADR-0004 says the event metrics are reported "each against its **calibrated bound**". The
implementation does an F1 argmax with tie-breaks. Those are different selection rules, and the
one the ADR specified was never built. Fixing that is the same edit.

## A per-boundary rate understates the exposure

`data/quran.db` holds 83,668 words over 604 pages: **median 140 words per page**, mean 138.5.
So a page carries roughly 139 word boundaries, the large majority of them wasl. A per-boundary
false-waqf rate multiplies accordingly:

| false-waqf@wasl | permanently forgiven positions per page |
|---:|---:|
| 0.001 | ~0.1 |
| 0.005 | ~0.7 |
| 0.01 | ~1.4 |
| 0.05 | ~7 |

These are **upper bounds on exposure**: not every wasl boundary carries a gradeable i'raab or
idgham distinction, so the count of positions where the forgiveness actually hides something is
lower. The direction is the point — a rate that reads as negligible per boundary is not negligible
per page, and the ratchet makes each instance permanent for the session. Quoting the operating
point per boundary hides that multiplication from whoever signs off.

## Decision

- **Constrain, do not weight.** Choose the threshold that **minimises false-wasl@stop subject to
  false-waqf@wasl ≤ B** on the calibration partition. A weighted objective (F-beta, a cost matrix)
  needs a relative price for a permanent error against a transient one, which nobody can defend;
  a bound on the absorbing error needs one number, argued below. This is also literally
  ADR-0004's "against its calibrated bound", made operative.

- **B is argued from what the head is replacing, and stated per page.** Today's
  ignore-end-word-tashkeel hack forgives the boundary word at *every* boundary — an effective
  false-waqf@wasl of **1.0**. The waqf head exists to buy that back, so it earns its complexity
  only if it recovers nearly all of it. **Default B = 0.005**, i.e. at most ~0.7 permanently
  forgiven positions per page — a ~99% recovery of the discrimination the hack gives away. The
  report states B in both units so the trade is legible without arithmetic.

- **B is recorded, never inherited.** The chosen value and its per-page conversion go on the
  report. Changing B is a product decision and must be argued on its own merits, not moved to
  make a checkpoint pass. This is the failure ADR-0006 documents for `MAX_REGRESSION`
  ("the margin is doing all the work and has never been argued from the product"), and B is the
  same kind of constant.

- **No admissible operating point is a reportable outcome.** If no threshold on the calibration
  grid satisfies B, report **`no_admissible_threshold`** with the sweep — not the closest miss
  dressed as a choice. Mirrors ADR-0006's `above_margin` discipline: a rule that always returns
  an answer cannot fail, and a rule that cannot fail is not a gate.

- **The full sweep is published**, so a reader can see what the constraint cost in false-wasl and
  judge B against the curve rather than against a single point.

- **Unchanged:** the calibration/test partition split and the once-only test scoring; the
  300/700 ms duration gate at F1's fixed VAD definition; the binary scoring convention and
  `mid_word_closure` as a diagnostic tag (ADR-0004); the deterministic tie-break, which now
  applies within the admissible set.

## Consequences

- `CALIBRATION_OBJECTIVE` and `calibrate()` change; the report gains `B`, its per-page
  conversion, the achieved rates, and `no_admissible_threshold` as a possible outcome.

- **The recorded 0.0 / 0.210 numbers are not this rule's output.** They come from an F1 argmax
  over the *reconstructed* lattice — #34's harness is torch-free and never loaded a checkpoint —
  so they describe neither this rule nor a trained head. They must not be quoted as if they were.

- **This rule may well improve the current point rather than tighten it.** false-waqf@wasl is
  already 0.0, which satisfies any B; under the constrained objective the threshold is then free
  to move wherever it minimises false-wasl@stop, and 0.210 is a lot of missed genuine stops to be
  paying for headroom the constraint does not ask for. The F1 argmax had no reason to spend that
  slack; this rule does.

- **Teacher circularity still applies** (ADR-0004): the VAD labels the head *and* supplies the
  frame-F1 target, so a systematic VAD error in amateur audio can survive this calibration
  untouched. B bounds a measured rate against adjudicated events, not the truth.

- **Out of scope:** the mid-word-closure rejection bound and the boundary-snap accuracy target,
  which are separate gates with their own thresholds; and any change to the head itself.

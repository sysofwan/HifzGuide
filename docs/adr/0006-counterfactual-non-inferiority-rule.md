# The counterfactual non-inferiority rule

The tashkeel counterfactual eval (`training.counterfactual_eval`) asks whether a fine-tune has
lost the base model's ability to *flag* a wrong short vowel. Each item was recited twice by the
same reciter — once as written (`control`), once with the target word's single short vowel
replaced (`counterfactual`) — and the two models are compared **paired on item id**:

- **b** (*regressed*) — base flagged the deliberate error, the fine-tune silently corrected it.
- **c** (*recovered*) — the reverse.

Until now the gate certified non-inferiority only when `b <= c` **and** the Wilson upper bound
on `b / n` cleared `MAX_REGRESSION = 0.05`. Observed on the 41 paired items recorded to date:
`rung1_v3` `b=2, c=0`; `rung3_v2` `b=4, c=0`. Neither certified, and an earlier handoff read
that as "collect ~35 more items".

## Why the old rule could never be satisfied

**The base model silently corrects nothing on this set — 0 of 41.** `c` counts items base got
*wrong* and the fine-tune got right, so a clean baseline pins `c` at zero. `b <= c` therefore
degenerates to `b == 0`: not a conservative non-inferiority rule but a **zero-tolerance rule in
disguise**.

Additional recording cannot escape it. Concordant items change neither `b` nor `c` (McNemar
significance likewise depends only on discordant pairs), and a recovery requires base to *fail*
on a new item — which it never has. So the advice to record ~35 more items was unreachable and
is **withdrawn**. The ~73 figure it rested on was the sample size at which Wilson's bound on
*zero* regressions, `z² / (n + z²)`, first clears a 5% margin — a flawless-run figure, never a
power calculation for `b = 2` or `b = 4`.

## Decision

**Certify on a paired non-inferiority interval for the net difference `(b - c) / n`, and drop
the `b <= c` clause.** This is a deliberate **loosening**, recorded as one: a set with `b > c`
can now certify if `n` is large enough (20 regressions against 10 recoveries over 1,000 items
is a 1% net, comfortably inside a 5% margin).

Three constraints on the implementation, each of which a naive version gets wrong:

- **Score interval, never Wald.** A Wald paired interval's width is proportional to `b + c`,
  so at `b = c = 0` it collapses to zero width and would certify on ten items — *looser* than
  the rule being replaced. `paired_score_interval` uses Tango's score interval, which degrades
  to the Wilson bound `z² / (n + z²)` when there are no discordant pairs, so certifying a
  flawless run still costs 73 items.
- **The bound, never the point estimate.** `equality_finding` reports what was *observed*;
  `non_inferiority_certified` reports whether the set was large enough for that observation to
  mean anything. Zero regressions over 42 items still leaves an 8% bound.
- **"Inconclusive" must be earned.** A set whose *net* regression already sits at or above the
  margin is reported as `above_margin`, not `inconclusive`: recording more audio *at that rate*
  cannot certify it, so calling it "more items needed" is precisely the error this ADR exists
  to correct. `above_margin` is **not** a finding of inferiority — see the caveat below.
- **The level is one-sided α = 0.025.** `NON_INFERIORITY_Z = 1.96` gives a 95% two-sided
  interval, whose upper bound is a 97.5% one-sided bound. That is the conventional and
  stricter non-inferiority choice; a 95% one-sided bound (z = 1.645) would certify a flawless
  run at 52 items rather than 73, so the level is pre-specified here rather than inherited
  from whichever `z` happened to be lying around.

### The loosening changes no verdict on the recorded sets

When `c = 0`, the upper bound on `(b - c) / n` **coincides** with the Wilson bound on `b / n`.
The recorded sets all have `c = 0`, so the numbers are unchanged; what changes is that the
unfalsifiable clause is gone.

| set | b / c / n | net difference | 95% interval | finding |
|---|---|---:|---|---|
| `rung1_v3` | 2 / 0 / 41 | 0.0488 | [−0.0411, **0.1614**] | inconclusive |
| `rung3_v2` | 4 / 0 / 41 | 0.0976 | [+0.0035, **0.2255**] | **above_margin** |

### Power, replacing the withdrawn ~73

Paired items needed for an **80% probability** of certifying, by assumed *true* regression rate
with no recoveries (`training.counterfactual_eval.required_items`):

| assumed regression rate | 0% | 0.5% | 1% | 2% | 3% | 4.88% | 9.76% |
|---|---:|---:|---:|---:|---:|---:|---:|
| paired items needed | 73 | 110 | 173 | 337 | 826 | ~249,000 | unreachable |

The power target is not decoration. Asking instead whether the single most likely table
certifies — round the rate to whole items, test that one table — gives 202 items at a 2% rate,
which actually certifies **62%** of the time; at 3% the equivalent figure certifies 53% of the
time. Half of a recollection built on those numbers would come back unsettled, having spent
the audio.

Power is also **sawtoothed** in `n`, because the largest certifying regression count is an
integer: it leaps when that threshold increments and decays until the next leap. At a 1% rate,
142 items reach 83% power while 160 items — *more* audio — fall back to 78%. The quoted size is
therefore the first `n` from which the target holds across `[n, 2n]`, not the first `n` to touch
it, so a recollection cannot miss by overshooting.

So **no recollection of the same kind is warranted for either checkpoint**: at their observed
rates, `rung3_v2` cannot reach the target power at any `n` and `rung1_v3` would need ~6,000
times the audio that exists. #10 needs a human sign-off recorded as such.

### What `above_margin` does *not* mean

It is **not** a finding that the checkpoint is inferior, and the power table is conditional on
its assumption. 4 regressions in 41 items is a very loose estimate of the underlying rate; the
same four regressions diluted by concordant items — `b = 4, c = 0, n = 202` — would certify.
What is ruled out is the specific plan "record more of the same and expect the number to move",
because that plan assumes the observed rate is the true one, and under that assumption the
target power is never reached. A checkpoint that fails to be *shown* non-inferior has not been
*shown* inferior, and #10 must not read it as such.

### The margin, not the interval, is the operative lever

`MAX_REGRESSION = 0.05` is doing all the work and has never been argued from the product. At
these sample sizes one discordant item is worth ~2.4%, so the margin is barely coarser than the
measurement's own resolution — and resolution is not precision: even *zero* regressions at
n = 41 leaves an 8.6% upper bound. Certifying `rung1_v3` on the set that exists would take a
margin of **16.2%** (its bound is 0.1614), and `rung3_v2` 22.6%. That is the loosening that
would actually be *felt*. It must be derived from Muraja's tolerance — how often
a student's wrong vowel may go unflagged relative to base — and changed on its own merits, not
smuggled in through the choice of interval.

### Take order is a live confound in the existing set

All 47 items have the control take recorded **first**. The counterfactual take is longer in
**30 of 43** non-tied pairs (sign test, two-sided *p* = 0.0137): the takes differ systematically
in a way that is perfectly confounded with the condition.

The **direction of the resulting bias is unknown**. A longer second take could be slower and
more deliberate, which would favour the model transcribing what was said; it could equally
reflect hesitation, silence or difficulty producing an unnatural vowel, which would favour
canonical reconstruction. Nor need either effect touch base and fine-tune equally, which is
what the paired counts depend on. The confound is a reason to distrust the sign of the measured
effect, not a reason to adjust it.

Any future recollection **must randomize take order**, and the analysis must report the order
effect. Note that this makes new items non-comparable with the existing 47 (order becomes a
covariate rather than a constant), so the two sets cannot simply be pooled.

## Consequences

- `compare_to_baseline` gains `net_difference`, `net_difference_ci95` and
  `items_needed_at_observed_rate`; `regression_rate` / `regression_upper95` remain as raw
  context. `equality_finding` gains two states: `within_margin` (more regressions than
  recoveries, but the net difference is bounded inside the margin — the outcome the old rule
  made unreachable) and `above_margin` (observed net regression at or above the margin, so
  recording more at that rate cannot certify — not a finding of inferiority).
- `python -m training.counterfactual_eval --rescore <report> [--baseline <report>]` re-judges a
  stored report's per-item outcomes under the current rule with no model or GPU, so a rule
  change never requires re-decoding audio. The three recorded reports were re-scored this way.
- **Superseded advice:** "collect ~35 more counterfactual items to get past inconclusive"
  (handoff, pre-#60) is withdrawn. The `~73` figure survives only as the zero-regression case
  in the power table above.
- **Eligibility is model-dependent, and that is a latent selection risk.** An item is dropped
  when its *control* take fails or the two alignments disagree, judged per model, and the
  paired set is the intersection of the two models' scorable sets. A fine-tune that degrades on
  a control take can therefore delete a pair that also carried a regression. It does not bite
  today — there are zero control failures, all five madd exclusions are concordant
  `followed_audio`, and the single unstable item (`cf026`) is `followed_audio` for every
  checkpoint — but a future comparison should report a best/worst-case sensitivity over the
  excluded items rather than assume it stays benign.
- **Out of scope:** re-deriving `MAX_REGRESSION` from product tolerance, and the multi-voice
  validity question — the corpus is one reciter producing deliberate, exaggerated errors, so a
  tight bound on it is a bound on that voice.

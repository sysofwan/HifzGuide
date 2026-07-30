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
  margin is reported as `disqualified`, not `inconclusive`, because the interval only ever
  shrinks onto the point estimate. Calling that "more items needed" is precisely the error
  this ADR exists to correct.

### The loosening changes no verdict on the recorded sets

When `c = 0`, the upper bound on `(b - c) / n` **coincides** with the Wilson bound on `b / n`.
The recorded sets all have `c = 0`, so the numbers are unchanged; what changes is that the
unfalsifiable clause is gone.

| set | b / c / n | net difference | 95% interval | finding |
|---|---|---:|---|---|
| `rung1_v3` | 2 / 0 / 41 | 0.0488 | [−0.0411, **0.1614**] | inconclusive |
| `rung3_v2` | 4 / 0 / 41 | 0.0976 | [+0.0035, **0.2255**] | **disqualified** |

### Power, replacing the withdrawn ~73

Paired items needed for the bound to clear a 5% margin, by assumed regression rate with no
recoveries (`training.counterfactual_eval.required_items`):

| assumed regression rate | 0% | 1% | 2% | 3% | 4.88% (`rung1_v3`) | 9.76% (`rung3_v2`) |
|---|---:|---:|---:|---:|---:|---:|
| paired items needed | 73 | 110 | 202 | 414 | ~122,800 | impossible |

So **no recollection is warranted for either checkpoint.** `rung3_v2` is disqualified at any
`n`; `rung1_v3` would need a corpus five orders of magnitude larger than the one that exists.
#10 needs a human sign-off recorded as such — not a statistical pass, and not more audio.

### The margin, not the interval, is the operative lever

`MAX_REGRESSION = 0.05` is doing all the work and has never been argued from the product. At
these sample sizes one discordant item is worth ~2.4%, so the margin is barely coarser than the
measurement's own resolution, and a margin of ~10% would certify `rung1_v3` today. That is the
loosening that would actually be *felt*. It must be derived from Muraja's tolerance — how often
a student's wrong vowel may go unflagged relative to base — and changed on its own merits, not
smuggled in through the choice of interval.

### Take order is a live confound in the existing set

All 47 items have the control take recorded **first**. The counterfactual take is longer in
**30 of 43** non-tied pairs (sign test, two-sided *p* = 0.0137): the second take is slower and
more deliberate. The bias runs *toward* the model hearing the substituted vowel, so the observed
regressions are, if anything, understated — the confound does not rescue the checkpoints.

Any future recollection **must randomize take order**, and the analysis must report the order
effect. Note that this makes new items non-comparable with the existing 47 (order becomes a
covariate rather than a constant), so the two sets cannot simply be pooled.

## Consequences

- `compare_to_baseline` gains `net_difference`, `net_difference_ci95` and
  `items_needed_at_observed_rate`; `regression_rate` / `regression_upper95` remain as raw
  context. `equality_finding` gains the `disqualified` state.
- `python -m training.counterfactual_eval --rescore <report> [--baseline <report>]` re-judges a
  stored report's per-item outcomes under the current rule with no model or GPU, so a rule
  change never requires re-decoding audio. The three recorded reports were re-scored this way.
- **Superseded advice:** "collect ~35 more counterfactual items to get past inconclusive"
  (handoff, pre-#60) is withdrawn. The `~73` figure survives only as the zero-regression case
  in the power table above.
- **Out of scope:** re-deriving `MAX_REGRESSION` from product tolerance, and the multi-voice
  validity question — the corpus is one reciter producing deliberate, exaggerated errors, so a
  tight bound on it is a bound on that voice.

# Size distillation of the Muaalem phoneme head to a single ANE chunk

The deployed model is the 578M-parameter teacher (`obadx/muaalem-model-v3_2`) split into six
6-bit CoreML chunks totalling 504 MB. The split exists only because a single model that size
exceeds the on-device ANE compiler budget; it is what keeps every chunk on the ANE, and
without the ANE real-time inference is not viable at all (`cpuAndGPU` is 3x slower —
Muraja ADR-0016).

Per-window cost is now the binding constraint. Muraja issues **~6 inferences per
audio-second** — one full-window pass plus ~5 throttled previews (`previewMinNewSamples` =
200 ms) — and every one pays the full static `(1, 250, 160)` cost. On M4 the six-chunk model
measures 42.0 ms per inference (ADR-0016 §5); on A15 `ml-model-transformation.md` §7 records
150–200 ms, which at 6 inferences/second saturates the ANE and is the reported thermal
problem. The ANE is a shared serialized block (4 workers buy 26% over 1), so there is no
concurrency headroom to reclaim.

The 5 s window / 1 s hop contract is **frozen** — it is well tested for accuracy and
latency, and the redundant re-encoding of 4 s per hop is by design. Cadence is therefore not
available as a lever. Per-window cost is the only one.

This ADR covers **size only**. The accuracy work (Tadabur tolerance fine-tune, ADR-0001) and
the waqf head (ADR-0004) are a separate track against the same teacher.

## Decision

- **Distil, do not prune, and cut width rather than depth.** `ml-model-transformation.md` §6
  records that training-free depth reduction destroyed this backbone — 24→12 layers gave
  99.4% CER, and even −4 layers broke it. That is evidence against *pruning*, not against
  compression, but it does identify depth as the dangerous axis; the speech-distillation
  literature (DistilHuBERT, FitHuBERT) points the same way. Every student preset in
  `training.distill_student` keeps all **24 layers** and shrinks `hidden_size`. Depth
  reduction is deliberately not offered as a preset.

- **Students use rotary position embeddings, not the teacher's `relative_key`.** This is the
  single decision that determines whether `h384` is one chunk or two, and it was found by
  exporting rather than by estimating. A traced `relative_key` attention bakes one
  `(250, 64, 250)` fp16 constant into the CoreML graph **per layer** — 4M values each, 96M
  over 24 layers. That cost is set by sequence length and head dim, both frozen, and is
  **independent of `hidden_size`**: it is 16% of the 586M teacher and would be 53% of an
  `h384` student. Measured, exporting `h384` with `relative_key` produced a **130.3 MB**
  6-bit package against a 71.1 MB parameter-based prediction, putting it over budget; the
  same student with rotary produced **61.7 MB**. Students are trained from random init, so
  they are under no obligation to inherit the teacher's positional scheme.

  The corollary is that **package size must be estimated over graph constants, not
  parameters**. Sizing on parameters alone under-predicts a thin model badly, because the
  fixed positional cost it omits is precisely the term that stops shrinking.

- **Target `h384`; ship whatever the agreement curve justifies.** Parameter counts from
  instantiating each preset; `h384` size confirmed by a real CoreML export:

  All three presets were exported untrained and benchmarked on an **M4 ANE**, against the
  real six-chunk teacher through the identical harness. That harness reproduces ADR-0016's
  independently measured 42.0 ms teacher figure at 42.4 ms, so the comparison is sound:

  | preset | params | 6-bit | chunks | M4 ANE | speedup | duty @ 6/s |
  | --- | --- | --- | --- | --- | --- | --- |
  | teacher (`relative_key`) | 586.4M | 504 MB | 6 | 41.9 ms | 1.0x | 25.2% |
  | h512 (rotary) | 151.8M | 109.0 MB | 2 | 15.2 ms | 2.8x | 9.1% |
  | **h384 (rotary)** | **85.5M** | **61.7 MB** | **1** | **12.7 ms** | **3.3x** | **7.6%** |
  | h256 (rotary) | 38.2M | 27.9 MB | 1 | 8.7 ms | 4.8x | 5.2% |

  Run-to-run variance on these timings is roughly ±10% (repeat `h384` runs gave 11.4 and
  12.7 ms, i.e. 3.3–3.7x), so treat the ratios as one significant figure.

  **The measured speedups are far below the FLOP ratios** (4.0x / 7.1x / 16.0x predicted).
  At these sizes the model is overhead-bound on the ANE, not compute-bound, so latency
  scales much more slowly than parameters: h256 is 2.2x smaller than h384 but only 1.5x
  faster. Any argument for a smaller rung has to be made on size, not on speed.

  `h384` is the knee. 61.7 MB clears the 99 MB largest-chunk-we-have-actually-compiled with
  real margin, and 3.3x takes the A15 duty cycle from a saturated ~90–120% to roughly
  27–36%. `h256` buys another 1.5x for a 2.2x parameter cut, which is the worst
  agreement-per-millisecond trade on the ladder; `h512` is the fallback if agreement does
  not hold at `h384`. Which one ships is decided by the agreement curve, not by this table.

  Note that `h512` loaded and ran as a **single** model on the M4 ANE at 109 MB, above the
  99 MB iPhone-13 ceiling this ADR sizes against. The M-series ANE budget is evidently
  larger than the A15's, which is exactly why the chunk-count column is keyed to the iPhone
  figure and why the device test below is not optional.

- **The objective is behavioural cloning, not accuracy.** The student is correct exactly
  insofar as it reproduces the teacher, because the teacher's behaviour is what Muraja's
  scorer, thresholds and fixtures are tuned against. Every metric is student-vs-teacher.
  This mirrors how the quantization variants were already scored in
  `ml-model-transformation.md` §1.3 (exact match / char accuracy against the FP32
  reference), so a distilled student reads as another row of that same table.

  Two consequences follow immediately: **no labels are needed** — the teacher generates every
  target, so any recitation audio is training data, with no filtering, reference phonemes or
  poison audit — and **only the phoneme head is built**, since Muraja consumes only that one
  of the teacher's 11 heads.

- **The teacher runs online rather than from a cache.** Caching its 43-way logits is cheap
  (~11 KB/window) but forfeits the feature-matching term that carries a 7x width cut; caching
  hidden states instead costs ~3 MB/window, i.e. terabytes over the corpus. A live bf16
  forward with no gradient, optimiser state or stored activations costs ~1.2 GB resident,
  which the 16 GB card absorbs: measured peak is **10.97 GiB at batch 32** (batch 48 OOMs).

- **Two frame weightings shape the KL**, because an unweighted mean optimises the wrong
  thing:

  - *Non-blank frames are up-weighted.* CTC output is blank-dominated — per Muraja's own
    `CTCStats`, blank holds 30–60% of timesteps during speech and 85–98% during silence — so
    a flat KL is mostly a lesson in predicting blank. This is the mirror image of the problem
    `training.waqf_head` already solved with `pause_frame_weights`, and it takes the same two
    answers: a weighting, and a collapse diagnostic.
  - *The confirmed region is up-weighted.* Only the first **25** of each window's 125
    timesteps become the user-visible transcript (`predictSplit` splits on
    `seg.midpoint < 25`), because a phoneme is committed when its window position is
    *oldest*, having accumulated the full 4 s of right context. The other 100 keep weight 1.0
    rather than being masked — they still drive the provisional display and the hallucination
    gate — but the region that becomes the transcript is worth more.

  One caution: **the non-blank boost does not transfer to other objectives.** Under a soft
  KL its effect is moderated by the target distribution; under hard labels it became a
  direct class-prior bias and flipped the student from under- to over-emitting.

- **The objective is pure weighted KL. The CTC anchor was REMOVED, and the feature term with
  it.** This bullet previously argued the opposite, and the reversal is the single most
  important finding in this ADR, so the original reasoning is kept here rather than deleted.

  The argument *for* a CTC anchor was that the KL and feature losses are per-frame, and a
  per-frame objective cannot break alignment symmetry: until the student knows which frames
  carry which phoneme, blank is locally optimal everywhere, so all-blank is a stable fixed
  point. That reasoning is sound in general and was supported by a real measurement — at
  step 2000 the teacher's class sat at rank 8.2 with P=0.027 against blank's 0.822.

  It was still wrong here, because the blank collapse it was diagnosing had a different
  cause: the SpecAugment/dropout determinism bug (below). Once that was fixed the anchor was
  never re-examined, and measurement later showed it holding **71% of the gradient** and
  actively destabilising training as the data diversified — the ablation is in "The 84.58%
  ceiling was a self-inflicted objective bug" above. Removing it moved decoded agreement
  84.58% → 89.37% and gate agreement 81.0% → 91.5%.

  The feature-matching term is removed for a duller reason: it earned 0.5% of the gradient
  and scored 0.886x the predict-the-mean baseline. It was never doing work.

- **The release gate is confirmed-stream agreement, not frame agreement.** Frame agreement is
  cheap and smooth and so is used during training, but it averages over 100 timesteps the
  user never sees. `training.distill_eval` replays the deployed sliding-window protocol —
  5 s window, 1 s hop, `scanCTC` collapse, `midpoint < 25` confirmation — and compares the
  resulting transcripts. That is the number that predicts what Muraja does.

## Result of the first full run

`h384`, rotary, 40,000 steps at 1e-4 over the 61.8-hour `audit_run/clips_v2` corpus
(78,578 windows), 11.67 h on the RTX 5060 Ti.

| | value |
| --- | --- |
| on-device size | **62.3 MB**, 6-bit, **single ANE chunk** |
| M4 ANE latency | **11.1 ms**/window vs the teacher's 42.3 ms (**3.8x**) |
| frame confirmed-agreement | 0.882 |
| top-5 agreement | 0.992 |
| target rank | 1.25 |
| blank rate | 0.650 vs the teacher's 0.664 |
| **confirmed-stream char accuracy** | **84.58%** (200 held-out clips) |
| confirmed-stream exact match | 10.5% |

**The size goal is met and the quality goal is not.** 84.58% against the 97-99% that
section 1.3 measured for the teacher's own quantization variants is not a drop-in
replacement; it is a model that gets roughly six characters in seven right.

**The corpus is the binding constraint, and the plateau proves it.** Char accuracy went
45.4% -> 75.1% -> 83.8% -> 84.6% at steps 4k/10k/20k/40k: **doubling the steps from 20k to
40k bought 0.76 points.** More training is worthless at this corpus size. Nor is there more
audio staged -- `seg_v21`, `segment_audio_v2` and `segment_audio_v4` are all waqf cuts of
the same 18,075 source clips, so the 61.8 hours is the whole of it. Upstream Tadabur has
385 shards of which ~20 are processed, i.e. roughly 1,200 hours available, against the ~960
hours DistilHuBERT-class recipes use. Expanding the corpus is the next move, and only after
that does it make sense to ask whether `h384` has the capacity -- running `h512` now would
confound a capacity question with a data limit.

## The 84.58% ceiling was a self-inflicted objective bug

**The CTC anchor caused it.** It was added to escape blank collapse at step 2000, when the
real cause was the SpecAugment/dropout determinism bug found later. It was never
re-examined after that fix, and it held **71% of the gradient** while the feature term held
0.5%.

Three measurements found it, none of which existed while the four hypotheses below were
being tested — which is why none of them explained anything:

* **Gradient share per term.** Loss magnitude says nothing about influence: the CTC term
  read 3.3 against the KL's 6.5 and owned 71% of the update.
* **A scaling probe** asking *where* fitting breaks, at the working learning rate:
  32 windows decoded **1.000**, 256 windows **1.000**, 1024 windows **0.000** (collapsed,
  gradient norm spiking to 36.9). Not data -- 78,578 windows are available and it could not
  use 1,024. Not capacity -- it reproduced 256 windows exactly.
* **Term ablation at that scale**, decoded agreement at step 3000. The two terms were
  removed together in the full run, so each was re-ablated on its own afterwards to
  confirm neither removal was being carried by the other:

  | terms | step 1000 | step 2000 | step 3000 |
  | --- | --- | --- | --- |
  | **KL only** | 0.171 | 0.742 | **0.847** |
  | KL + feature | 0.000 | 0.283 | **0.676** |
  | KL + CTC | 0.015 | 0.000 | **0.000** |
  | CTC only | 0.150 | 0.111 | **0.000** |
  | feature only | -0.010 | -0.091 | **-0.252** |

  Adding CTC to a working KL destroys it. Feature matching is worth only 1.7% of the
  gradient and still costs 0.17 of decoded agreement at a matched step budget. The
  feature-only arm explains why: its own loss falls 0.610 -> 0.076, a clean 8x, while
  decoded agreement goes *negative and keeps falling*. Matching the teacher's hidden
  states is not a weak proxy for matching its output distribution -- on this backbone it
  is very nearly an unrelated objective, and optimising it well is not evidence of
  anything. That is the argument for scoring objective work on the decoded stream from
  the first step rather than on whichever loss happens to be falling.

Retraining with **pure weighted KL** -- no CTC, no feature matching -- on the identical
corpus, model and step budget:

Every row below is measured on the **same 200 held-out clips** for both checkpoints. An
earlier version of this table compared the old recipe on 100 clips against the new one on
200, which is not a comparison; the old checkpoint was re-run at 200 to replace it.

| | old recipe | KL-only |
| --- | --- | --- |
| confirmed-stream char accuracy | 84.58% | **89.37%** |
| exact match | 10.5% | **14.5%** |
| **gate-decision agreement** | **81.0%** | **91.5%** |
| teacher passed / student failed | 27 in 200 | **9 in 200** |
| student passed / teacher failed | 11 in 200 | 8 in 200 |
| mean match_ratio | 0.751 | **0.809** (teacher 0.847) |
| ratio correlation | 0.653 | **0.874** |
| frame confirmed_agreement | 0.8819 | **0.9177** |
| target_rank | 1.249 | **1.167** |

The directional failure that made the student unshippable -- rejecting recitation the
teacher accepts -- is gone: 27-vs-11 became 9-vs-8, and the false-rejection rate fell from
13.5% to 4.5%. On identical clips the agreement gain (81.0% -> 91.5%) clears an unpaired
two-proportion test at p ~ 0.002, and the paired test on the same clips can only be
stronger, so the recipe change is real rather than sampling noise.

**But 91.5% is not yet distinguishable from doing nothing.** A gate that passes every clip
scores 88.0% on this set, and 91.5% vs 88.0% is *not* significant (McNemar p ~ 0.23,
n=200). The KL-only recipe is measurably better than the old recipe; it is not yet
measurably better than a rubber stamp. Closing that gap needs both a higher score and a
larger evaluation set -- see the follow-up issue.

**The lesson worth keeping is procedural.** Every term in a distillation loss should be
justified by a measurement, and re-justified whenever the diagnosis that motivated it
changes. A term added to fix a misdiagnosed problem will not remove itself.

## What had been ruled out for the 84.58% ceiling

Four candidate explanations, three eliminated by measurement. Recorded because each cost
real GPU time and the negative results are what stop them being re-tried.

| hypothesis | verdict | evidence |
| --- | --- | --- |
| **more data** | ruled out | train **84.98%** vs val **84.16%** at step 40000. A 0.82-point gap: the student cannot reproduce the teacher on windows it has seen ~16 times, so more audio cannot be the fix. |
| **more steps** | ruled out | char accuracy 45.4 → 75.1 → 83.8 → 84.6 at 4k/10k/20k/40k. Doubling 20k→40k bought 0.76 points. |
| **objective mismatch** | ruled out | Two hard-label runs warm-started from the 40k checkpoint: with the 3x non-blank weighting **82.93%**, with neutral weighting **82.76%**. Both lose to the 84.16% baseline. |
| **capacity** | ruled out | `h448` (116.3M) trailed `h384` at every matched step and finished worse -- but note this was measured under the broken objective, so it is evidence about that objective, not a clean capacity result. |

Every one of these was measured **through the broken objective**, which is why none of them
explained the ceiling and why the two that looked most convincing (no train/val gap; a
larger student performing worse) were the most misleading.

The objective experiment was worth running — `target_rank` 1.25 with top-5 agreement 0.992
says the teacher's class is nearly always present and merely loses the argmax, which looks
exactly like a margin problem a hard-label loss should fix. It does not, and the reason is
the more useful finding:

**Frame agreement and decoded agreement are not the same objective.** Under hard labels,
frame `confirmed_agreement` *rose* (0.8819 → 0.8842) while decoded char accuracy *fell*
(84.16% → 82.76%). After the `scanCTC` collapse, **where** an error lands matters more than
how many there are: a flip in the middle of a run is absorbed, a flip at a segment boundary
splits or merges a token and costs an edit. Optimising per-frame agreement -- softly or
hard -- therefore does not straightforwardly optimise the gate. Any future objective work
should be evaluated on the decoded stream from the start, not on frame metrics.

A second trap surfaced in the same experiment. The 3x non-blank frame weighting exists to
escape the blank basin, and under a soft KL its effect is moderated by the target
distribution. Under **hard** labels it becomes a direct bias on the class prior, and the
student flipped from under-emitting (46.0 vs 47.2 tokens/clip) to over-emitting (55.5 vs
55.2). A weighting introduced for one objective does not transfer to another unexamined.

## The 91.5% gate number was a property of the panel, not of the model

Everything above measures gate agreement on 200 random `clips_v2` clips. That panel cannot
answer the question, for three separate reasons, and fixing them changed the conclusion
without changing the model.

**A rubber stamp scored 88.0% on it.** 91.5% against that floor is McNemar p ~ 0.23 at
n=200. Recorded already; it is what the follow-up issue was opened for.

**`clips_v2` is the teacher's own gate passers.** It *is* `passing_subset_full.jsonl` — every
clip in it cleared the whole-clip gate during staging. So it contains almost no teacher
rejections, and a panel drawn from it **cannot measure false acceptance at all**, which is
half of what a gate does. The 8 "student passed, teacher failed" clips in the original 200
were artefacts of the windowed protocol, not recitation the teacher judged bad.

**The protocol was transcribing only part of each clip.** `confirmed_stream` commits the
segments in a window's oldest second, because the next window re-decodes the rest with more
right context. It did that for the **last** window too, so the final four seconds of every
clip were decoded and discarded — and a clip shorter than one window was gated on its first
second alone. Muraja flushes what is pending when speech stops. That omission was harmless
while the module reported only *character* agreement, where both models lose the same tail;
it is not harmless for a `match_ratio` computed against the whole ayah. The flush is now
replayed, `PROTOCOL_VERSION` records it, and cached teacher decodes carry the version so a
set built under one protocol cannot be scored under another. **Numbers from before and after
this change are not comparable.**

### The replacement, and why it has two views

`training.gate_evalset` scans 20 **strided** reserved Tadabur shards — strided rather than a
block at the top of the range, because nothing guarantees shard order does not group reciters
and a tail block would then be a distribution shift rather than a sample. `distill_stream`
refuses both reservations in its constructor; the old "shards 0-19 are reserved" comment was
a reservation that held until someone typed a range by hand.

One scan draws two samples, because neither alone supports a decision. Only **4% of raw
Tadabur clips fall below the 0.65 bar**, so a population sample is saturated by construction
(pass-everything floor 85.2%) and nothing done to the model moves it. A sample enriched
around the bar has a 40.6% floor and is sensitive — and is deliberately unrepresentative, so
shipping on it alone would be dishonest. Both are always reported, with the scan's
per-stratum counts, which are the only route back from the enriched number to a population
one. Clips are split dev/test **by reciter**.

The teacher is decoded once and frozen into the manifest. Besides halving every later
evaluation, it makes two checkpoints comparable by construction rather than by hoping the
teacher ran identically twice.

**Store the clips as 32-bit float, not PCM_16.** The manifest caches the teacher's decode of
the in-memory waveform and the student reads the file back, so anything lossy between them is
a disagreement charged to a model that did nothing. Measured on 60 clips, a PCM_16 round trip
changes the *teacher's own* decoded string on 12 of them and moves `match_ratio` by up to
0.062; Tadabur audio also peaks at 1.037, so PCM_16 clips real signal. This was caught forty
minutes into a build and cost a rebuild.

### Score the distillation on Muraja's decision, not on the filter's

`tadabur.scorer` layers **two** rejects on top of the Muraja-faithful `match_ratio`, and its
own comments label both "NOT a Muraja parameter", "Tadabur-only" and "filter-side": a long
interior insertion run, and an *added* shadda (the ADR-0001 P3.5 asymmetric mitigation).
They exist to keep poison clips out of the fine-tune corpus, not to decide what a reciter is
shown. Holding a **size distillation** to them scores it against a corpus-filtering policy —
and against a condition whose verdict turns on the placement of one geminate — the most
edit-sensitive part of the gate by roughly nine to one against the ratio.

This ADR already carries the rule: the distillation "must not be confounded with the
ADR-0001 track". `gate_evalset.DISTILLATION_CRITERION` is that rule applied, and
`distill_gate --criterion` makes the choice visible in every report rather than assumed. The
insertion-run reject is **kept** — it is stable under the same perturbation (0.2%) and
excluding it changes nothing (96.85% against 96.50%), so dropping it would only make the
criterion look tuned.

### What the h384 checkpoint actually scores, and on which condition

Unchanged, no retraining, 2,000 population and 1,000 boundary clips:

| gate definition | population (n=2000) | boundary (n=1000) |
| --- | --- | --- |
| Tadabur filter gate, both poison rejects | 91.40% | 92.40% |
| **without the added-shadda reject** — the distillation criterion | **96.85%** [95.99, 97.53] | **95.80%** [94.37, 96.88] |
| Muraja-faithful: `match_ratio` alone | 96.50% [95.60, 97.22] | 95.70% [94.26, 96.79] |

**The >95% target is met.** On the population view the Wilson lower bound clears 95% outright;
on the boundary view the point estimate clears it and the lower bound (94.4%) does not, so
that view is met without margin. Directionally the student is slightly strict: 3.14% false
rejections against 3.31% false acceptances on the population view.

One caveat on reading the `ratio_only` row: on the population view its pass-everything floor
is 96.10% against 96.50% agreement (McNemar p = 0.56), i.e. **that row is saturated and
carries no evidence on that view**. Only ~4% of raw clips fail on ratio alone. The boundary
view is where it is informative (floor 50.0%, p = 4e-108), which is what the two-view design
is for.

`Scorer.gate` ands three conditions together and they are not comparable objects.
`match_ratio` is a smooth function of the decode that frame-weighted KL pushes on directly;
the insertion-run and added-shadda rejects are discrete alignment properties. A single number
over the filter gate answers two questions and reports neither.

Of 239 disagreements under the filter gate, **68% involve a poison reject rather than the
ratio**, and that is almost entirely added shadda: 52 of 54 missed rejections and 65 of 68
self-inflicted ones. Removing that one out-of-scope condition is the whole difference between
91.40% and 96.85%.

### Why more training does not close the rest

Perturb **the teacher's own decode** by N random single-phoneme edits — a third deletions, a
third substitutions, a third insertions of a random phoneme — and re-gate it, 600 clips:

| edits | gate flips | added-shadda | insertion-run | ratio |
| --- | --- | --- | --- | --- |
| 1 | 1.5% | 1.3% | 0.0% | 0.2% |
| 2 | 3.0% | 2.0% | 0.2% | 0.8% |
| 4 | 10.2% | 5.0% | 0.0% | 5.2% |
| 8 | 30.8% | 8.2% | 0.3% | 22.3% |

The edit model matters and a first version of this got it wrong. Inserting a *duplicate* of
the neighbouring phoneme — which is literally a geminate — put added shadda at 6.3% on one
edit and made it look like the whole story at every edit count. That was circular. With
neutral insertions, **one edit flips the gate on 1.5% of clips and added shadda is 1.3 of
those 1.5 points against the ratio's 0.2** — still by far the most edit-sensitive condition,
at roughly nine times the ratio's rate, but a fifth of the headline the biased model gave.

Note also that by eight edits the *ratio* dominates this simulation (22.3% against 8.2%),
which is the opposite of what the real decodes show. That gap is informative rather than
contradictory: a student's errors are not random, and it keeps `match_ratio` far closer to
the teacher's than random edits do (RMSE 0.079, offset −0.043) while still flipping discrete
geminate decisions. Random-edit simulations over-move the ratio and under-state added
shadda's real share.

The condition has an 8.5% base rate and turns on where a *single* geminate lands; geminate
counts already match (student 13.16 per clip against the teacher's 13.56, 0.97x), so this is
placement, not rate.

That perturbation is a **pessimistic** model of a real student, though, and the difference
matters enough that acting on the synthetic curve alone would have misdirected the work. Its
edits are uniformly distributed; a student's are not. Bucketing the 2,000 population clips by
the actual edit distance between the two decodes:

| edits | clips | gate agreement | share of all flips |
| --- | --- | --- | --- |
| 0 | 155 | **100.0%** | 0% |
| 1 | 186 | **100.0%** | 0% |
| 2–3 | 352 | 97.2% | 6% |
| 4–7 | 527 | 93.5% | 20% |
| 8–15 | 489 | 86.1% | 40% |
| 16+ | 291 | 79.4% | 35% |

Within one edit the gate never flips — 341 clips, zero disagreements — and **75% of all flips
come from the 39% of clips with eight or more edits**. Gate agreement is a steep, monotone
function of decode fidelity, and the errors are concentrated in a tail rather than spread
thin.

So the target is reachable after all, and the budget is legible: moving the edit distribution
one bucket left takes flips from 8.6% to roughly 4.4%, i.e. past the bar. That is character
accuracy around 95% against today's 90.4% on this set — a large jump, but a *training* jump,
not the ~99% the uniform-perturbation model implied. The efficient version targets the tail:
291 clips carry a third of the disagreement.

Two things are true at once and both belong in the decision. The gap is reducible by better
decodes, and `REJECT_ADDED_SHADDA` is what converts a decode error into a flipped decision at
an 8.5% base rate on the placement of a single geminate. The first is this issue's work; the
second is an ADR-0001 P3.5 question: a filter condition that accounts for 1.3 of the 1.5
points a single phoneme edit moves, against the ratio's 0.2, is admitting and rejecting
corpus clips partly on decode noise.

## Teacher-weight initialisation: what transfers, and what does not

`training.teacher_init` starts a student from a **selected sub-network** of the teacher
rather than a PCA rotation of it. The rotation is what the literature reaches for and it is
wrong on this backbone three times over: LayerNorm does not commute with a rotation, the
residual stream is added to in every block so a rotation must be globally consistent, and —
decisively — transformers' `Wav2Vec2BertSelfAttention` applies the rotary embedding to the
**hidden states, before** `linear_q`/`linear_k`, in `num_heads` contiguous blocks of the
*input* space. After an arbitrary rotation those blocks are groups of unrelated directions.
Selection keeps every student channel equal to one teacher channel, so LayerNorm gains
index-select exactly and the transplant is verifiable by reading it.

Three findings, all measured before any training:

**Copying query and key across `relative_key` → rotary is worth nothing.** Initial weighted
KL is 8.5765 with them left random, 8.5699 copied, 8.5751 copied-and-damped, against 10.4714
for a random student. Layer-by-layer cosine against the teacher moves by 0.003. The positional
mismatch is real — the teacher learned a separate `q·E[clamp(j−i, −64, 8)]` bias term that
rotary has no slot for — but it is not what costs the transfer.

**The folklore rescaling is wrong here.** Each branch is a sum over units and the student
keeps 37.5% of them, so the received fix is to scale the survivors by 16/6. The least-squares
optimum measures **median 1.10, range [0.76, 1.70]**: the dropped units contribute
*orthogonally*, so what is missing is a direction the student cannot express, not a magnitude,
and 16/6 would have amplified noise while looking principled. The fitted gain still earns its
place at the end of the stack, where the adapter output the CTC head reads goes from cosine
0.29 to 0.53.

### And it works: 5x fewer steps, and the blank basin disappears

`distill_overfit`, 1,024 fixed windows, batch 32, lr 1e-4 — matched to the ablation above, so
the random arm is directly comparable and does reproduce its 0.171 at step 1000. Decoded
agreement:

| step | random init | teacher init `--qk random` | `--qk copy` | `--qk damp` | `--qk random`, lr 3e-5 |
| --- | --- | --- | --- | --- | --- |
| 300 | 0.000 | **0.827** | 0.776 | 0.810 | 0.659 |
| 600 | 0.000 | 0.856 | 0.847 | 0.847 | 0.703 |
| 900 | 0.007 | 0.899 | 0.905 | 0.932 | 0.838 |
| 1200 | **0.232** | **0.919** | **0.929** | **0.939** | 0.899 |

Teacher init reaches 0.847 at step 600; random init needs 3,000. By 1,200 steps it is above
anything random init reached in the whole 3,000-step ablation.

**The all-blank basin is simply absent.** Random init sits at decoded 0.000 through step 900
— the failure mode that motivated the CTC anchor, cost this project multiple runs, and is
the reason `breakout_stats` exists. A transplanted student never enters it.

Two secondary results. The query/key variant is within single-seed noise, consistent with the
init-time KL, so the positional mismatch genuinely is not the obstacle. And **lr 3e-5 is
worse than 1e-4** (0.899 against 0.919 at 1,200): the usual advice to lower the rate for a
warm-started model does not hold here, it is just slower.

This is evidence about **optimisation**, which is what fitting a fixed set measures. It does
not displace the trained checkpoint — `h384_klonly@40k` has already paid the 40,000 steps
this saves, and the transplant is a better start than noise, not than a trained model. Where
it pays is every student not yet trained: the `h448` capacity retest, or any re-architecture,
no longer has to buy its way out of the blank basin first.

**The loss is intrinsic to the width cut, and it happens in one block.** The transplant is
mechanically exact at the feature projection (cosine 1.0000 against the teacher on the kept
channels) and a single conformer block takes it to 0.81, settling around 0.4–0.5. Keeping
37.5% of each block's additive contributions is what costs it. Two corrections were found by
running that check rather than by reading the code: the conv module's internal channels are
*not* the residual stream — they are a separate learned space of the same width — and were
being selected by residual importance with every shape still lining up; and `depthwise_layer_norm`
normalises over that space, so its scale correction was being measured across two different
ones.

## Consequences

- **The corpus problem disappears, and a small corpus suffices to start.** 61.8 hours of
  16 kHz recitation already on the GPU box (`audit_run/clips_v2`, 18,075 clips) yields 80,250
  training windows at a 2.5 s stride. Unlabelled, unfiltered audio is usable, so this scales
  to the remaining Tadabur shards whenever the student needs more.

- **The student must be deterministic in `train()` mode, and this is not the default.**
  Distillation asks the student to reproduce a teacher running in `eval()` on clean input.
  Any train-time stochasticity therefore makes the target unlearnable at the perturbed
  positions *and* changes the input on every step. Three sources were inherited from the
  teacher's config and all three were missed on the first pass:

  - `apply_spec_augment` left `True`. **Zeroing `mask_time_prob` does not disable it** —
    transformers computes `max(num_masked_span, min_masks)`, and the teacher config carries
    `mask_time_min_masks=2`, so a zero probability still masked two 10-frame spans per
    sequence: 20 of 250 frames randomised every step.
  - `conformer_conv_dropout` at 0.1, firing in all 24 layers.
  - `final_dropout` at 0.1, directly on the CTC head's input.

  The last two survived because they do not share the naming of the four obvious dropout
  fields, so zeroing those four left them on. Measured: two identical train-mode forwards
  differed by **1.70**, and the model could not overfit 32 fixed windows in 1500 steps.
  With them off, the forward is bit-identical and the same overfit reaches ctc **0.052**,
  non-blank agreement **0.979** and rank **1.02** by step 600.

  This cost two full training runs. From the outside it is indistinguishable from a
  converged model: the loss curve flattens, the metrics plateau, and every plausible
  explanation points at the objective or the learning rate. The tests enumerate the config
  rather than naming fields, so the next such default is caught.

- **Train at 1e-4, not 3e-4, and watch the pre-clip gradient norm.** With the determinism
  bug fixed, the run still failed — but differently, and the difference is the diagnosis. It
  improved all the way through warmup and then **regressed** the moment the learning rate
  reached its 3e-4 peak: between steps 2000 and 4000, non-blank agreement went 0.002 → 0.000,
  blank probability 0.749 → 0.801, rank 8.19 → 8.35. Improving at the ~1.5e-4 warmup average
  and regressing at 3e-4 is a learning rate that is too high, and the gradient norm confirms
  it: even at **1e-4** the pre-clip norm runs 3–7 and hits the 5.0 clip about half the time,
  so at 3e-4 it would have been clipped on essentially every step — training far below its
  nominal rate while the loss curve looks converged.

  At 1e-4 the same measurements move for the first time. By step 2000: rank **8.13 → 6.97**
  (the first real drop in any run), non-blank agreement **0.0002 → 0.0469**, top-5 **0.52 →
  0.61**, margin **0.736 → 0.534**, and `ctc` finally descending (3.19 → 2.86).

  Note that the overfit test converged fine at 3e-4. A 32-example landscape is far more
  forgiving than 78k diverse windows, so **the overfit test settles "broken or slow", not
  the hyperparameters** — a distinction worth keeping, since treating it as evidence about
  the real run would have pointed the wrong way here.

- **Overfit a fixed batch at the first plateau, not the third.** A model that cannot drive
  the loss toward zero on 32 examples it sees repeatedly has a structural problem that no
  amount of data, patience or loss reweighting will fix; one that can is telling you the
  architecture and objective are sound and the problem is elsewhere. That single test found
  the above in minutes, after three runs had been spent on the loss design, the position
  embeddings and the learning rate. `training.distill_overfit` exists so the next stall
  starts there, and it also reports the **pre-clip gradient norm** that `distill_train`
  hides — a run clipping 100 down to 5 trains at a twentieth of its nominal rate and looks
  exactly like convergence.

- **Blank collapse is the main training risk, and argmax metrics cannot diagnose it.** The
  trivial solution — predict blank everywhere — scores ~67% frame agreement for free,
  because that is the teacher's blank rate, and it is the observed starting basin in
  practice. `agreement_stats` makes a parked run visible immediately via
  `blank_collapse_margin` and `nonblank_agreement`.

  But those cannot tell you whether to keep waiting: `nonblank_agreement` reads a flat 0.0
  for thousands of steps whether the teacher's class holds 40% of the student's mass or
  0.1%, because argmax is a step function. `breakout_stats` reports the continuous
  quantities instead — `target_prob`, `target_rank`, `prob_margin` — so the decision to
  intervene is measured. It is what turned "this looks slow" into the concrete finding
  above, and it corrected an earlier read of the same run as nearly escaped.

- **Palettization costs nothing in size predictability and something real in quality.** Both
  measured exports land at the nominal 6/8 bytes per graph value (0.753 and 0.756), so there
  is no compression surcharge to budget for, and the palettized size does not drift with
  training (62.3 MB at step 2000 and again at step 10000).

  §1.3's *quality* result does not transfer, exactly as feared. On the teacher, 6-bit was
  argmax-identical to INT8, because a 586M model is heavily overparameterized. On the
  85.5M student it is not. Measured against the uncompressed FP16 export, over real windows
  from a step-10000 checkpoint:

  | | size | frame argmax agreement | windows fully identical | chunks |
  | --- | --- | --- | --- | --- |
  | FP16 (uncompressed) | 164.0 MB | reference | — | 2 |
  | 6-bit | **62.3 MB** | 98.91% | 2/11 | 1 |
  | 8-bit | **82.8 MB** | **99.85%** | **9/11** | 1 |

  8-bit removes ~7/8 of the disagreement for 20.5 MB, and **both are a single chunk**, so
  this is a quality decision with no architectural consequence. Two caveats keep it from
  being final: the checkpoint is not converged (a sharper model is likely *more* robust, so
  this reads pessimistic), and frame-level disagreement is not the metric that matters —
  low-confidence flips often survive CTC collapse unchanged. The decision belongs to
  confirmed-stream agreement of the palettized export against the PyTorch student at the end
  of training, not to this table.

- **Trace the student only after a warmup forward.** `Wav2Vec2BertRotaryPositionalEmbedding`
  caches its cos/sin table on first use, so the first and second forward passes produce
  structurally different graphs and `torch.jit.trace`'s `check_trace` fails with "Graphs
  differed across invocations". One warmup call settles it; the deployed shape is static, so
  the cache never invalidates afterwards. `export_student_coreml.py` does this, and with it
  the traced graph matches eager exactly (max abs diff 0.00e+00).

- **A single chunk removes more than parameters.** Five `copyToFresh` FP16→FP32 conversions
  and five CoreML dispatches per window disappear, so the realised speedup should exceed the
  pure-FLOP estimate. It also retires the chunk-boundary dtype hazard that
  `ml-model-transformation.md` §5 documents as a shipped bug, and the `errno 28` ANE
  temp-cache pressure from compiling six models on low-storage devices.

- **The size is now measured; the ANE acceptance is still inferred.** The throwaway export
  has been done: a randomly-initialised `h384` converts cleanly, palettizes to **61.7 MB**
  and compiles to a 61.8 MB `.mlmodelc`. What that does *not* establish is that the iPhone 13
  ANE compiler accepts it as one chunk — that budget is enforced at `MLModel` load on the
  device, and §2.5 warns it fails **silently** to CPU rather than raising. 61.7 MB against a
  99 MB demonstrated ceiling is strong evidence, not proof; a device load test is the proof,
  and it can be run on the untrained export without waiting for training.

- **The waqf head is not a blocker, and is now opt-in at export.** `convert_to_coreml.py`
  used to emit `waqf_logits` on every export, falling back to a **randomly-initialised**
  head when no `--waqf-head` weights were passed. Nothing consumes it: the Swift side reads
  only `phoneme_logits` (`MuaalemInference.predictSplit`), and no shipped asset carries
  trained waqf weights. So every export embedded a random signal under an output name that
  looks load-bearing — a trap for whoever wires it up later without checking whether the
  weights were real. The head is now exported only when `--waqf-head` is supplied, which
  also makes the default export match what the app actually reads.

  A phoneme-only student is therefore a complete replacement for what ships today. If the
  ADR-0004 fine-tune later produces weights worth shipping, the head is a per-frame linear
  on the same 40 ms lattice — negligible for sizing, but it would have to be distilled onto
  the student, which is out of scope here.

- **Measure the gate one condition at a time.** `Scorer.gate` is three conditions anded
  together and only one of them is a smooth function of the decode. Reported as a single
  number, a student that reproduces the ratio almost perfectly and the added-shadda heuristic
  poorly is indistinguishable from one that is uniformly mediocre — and the two call for
  completely different work. `training.gate_evalset.gate_verdicts` and `distill_gate`'s
  per-condition row exist so that split is in every report rather than being rediscovered.

- **Before optimising an agreement target, measure what the target does under a trivial
  perturbation of the *teacher*.** Perturbing the teacher's own decode by one phoneme shows
  which of the gate's conditions is carrying the sensitivity — here, added shadda at nine
  times the ratio's rate. It cost minutes of CPU and would have redirected this issue before
  a single GPU hour was spent on it.

  **And design the perturbation carefully**, because the first version of this was circular:
  inserting a duplicate of the neighbouring phoneme *is* an added shadda, so it manufactured
  the finding it was used to support and inflated it five-fold. A synthetic perturbation is
  only evidence about the metric if its edit distribution is neutral with respect to the
  thing being measured. The conclusion survived because it rests on the direct measurement
  (91.40% against 96.85% on real decodes), not on the simulation — but it was stated with
  the simulation's number.

- **This must not be confounded with the ADR-0001 track.** That fine-tune deliberately
  *increases* tolerance on the soft pairs; width distillation will involuntarily *reduce*
  discrimination on those same pairs. Distil against the current teacher and gate on
  agreement first; apply tolerance changes separately, or distil from an already-fine-tuned
  teacher. Doing both at once makes ADR-0001's success criterion unmeasurable.

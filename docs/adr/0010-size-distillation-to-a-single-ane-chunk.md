# Size distillation of the Muaalem phoneme head to a single ANE chunk

> **Amended by [ADR-0011](0011-transcription-fidelity-and-tashkeel-abstention.md)
> (2026-10-08).** Behavioural cloning remains the right metric for the **size** stage this
> ADR covers. The shipped student must now also be **more accurate than the teacher** on
> truth-labelled sites, so accuracy enters through a supervised term on trusted signal, or
> through the teacher it copies. The CTC-anchor collapse recorded below was measured **from
> init**; it does not settle whether a small supervised term on a converged student is safe,
> which must be measured. The waqf-head export option is to be removed, and the output
> vocabulary may grow (an explicit sukun class) within the same byte budget.

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
| **capacity** | ~~ruled out~~ **OVERTURNED** | `h448` (116.3M) trailed `h384` at every matched step and finished worse -- but this was measured under the broken objective. Re-run under the corrected objective, **`h448` is the lever**: 94.97% against `h384`'s 93.09%. See [Outcome](#outcome-h448-met-the-target-and-capacity-was-the-lever-after-all). Do not read this row as evidence against width. |

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
how many there are.

An earlier revision explained that as "a flip in the middle of a run is absorbed", which is
**wrong** and was checked only later: a mid-run substitution turns `AAA` into `ABA`, which
decodes to three tokens instead of one — two extra edits, the *most* expensive case, not the
cheapest. A mid-run blank turns `AAA` into `A_A`, decoding to `AA`, one extra edit. What
collapse actually absorbs is **duration** variation that leaves the run structure intact:
`AAA` and `AAAA` both decode to `A`.

The conclusion is unchanged and if anything stronger — per-frame agreement does not
straightforwardly optimise the decode, and frame flips inside a run are expensive rather than
free. Any objective work should be evaluated on the decoded stream from the start.

A second trap surfaced in the same experiment. The 3x non-blank frame weighting exists to
escape the blank basin, and under a soft KL its effect is moderated by the target
distribution. Under **hard** labels it becomes a direct bias on the class prior, and the
student flipped from under-emitting (46.0 vs 47.2 tokens/clip) to over-emitting (55.5 vs
55.2). A weighting introduced for one objective does not transfer to another unexamined.

## The metric was wrong before the model was: measure the decode, not a gate

Everything above scores this work on "gate agreement" — 91.5% on 200 random `clips_v2` clips.
Three separate things are wrong with that, and the third is the one that matters.

**A rubber stamp scored 88.0% on that panel**, so 91.5% carried a Wilson interval of roughly
[86.8%, 94.6%] and could not have established >95% whatever the model did.

**`clips_v2` is the teacher's own gate passers** — it *is* `passing_subset_full.jsonl` — so
it contains almost no teacher rejections.

**And a gate is the wrong kind of number for a distillation at all.** Size distillation is
behavioural cloning of the teacher's **phoneme decode**. `tadabur.scorer` takes a decode,
aligns it against an ayah reference with Smith-Waterman, and thresholds the result. None of
that apparatus appears in "does the student emit what the teacher emits". Worse, the
apparatus belongs to two *other* tracks:

- the two poison rejects layered on the score are, in the scorer's own comments, "NOT a
  Muraja parameter", "Tadabur-only", "filter-side" — they decide which clips enter the
  ADR-0001 **fine-tune corpus**;
- the threshold is not the product's either. ADR-0005 records Muraja's **advancement**
  decision as `matchRatio` against a hard-coded **0.70** that `scoringMode` does not touch;
  `.balanced`'s 0.65 is the filter's bar;
- and **ADR-0008 (Accepted) already ruled on exactly this**: that gate *is* ADR-0001's
  training-data filter and "should not be the headline metric at all".

All of that was already written down in this repo. Two passes of this work were spent
improving agreement with a corpus filter, then with an advancement decision, before reading
it. The cost was a rebuilt measurement aimed at the wrong question and a "target met" claim
that had to be withdrawn twice.

### What is measured instead

`training.decode_evalset` freezes a held-out clip set with the **teacher's decoded phoneme
string** cached per clip, and `distill_eval --eval-set` scores a student's decode against it.
No reference, no aligner, no threshold.

- 20 **strided** reserved shards, never trained on — strided because shard order may group
  reciters and a tail block would be a distribution shift rather than a sample.
  `distill_stream` refuses both reservations in its constructor.
- A plain **uniform** reservoir draw. The metric is a pooled character accuracy over ~168k
  phonemes; there is no threshold to saturate and nothing to enrich around, which is what an
  earlier ratio-stratified second sample existed for. Legacy manifests carry it and
  `load_manifest` drops it, since it is not a uniform draw.
- Split dev/test **by reciter**.
- Clips stored as **32-bit float**: the manifest caches the teacher's decode of the in-memory
  waveform and the student reads the file back, so anything lossy between them is charged to
  the student. A PCM_16 round trip changed the teacher's *own* decode on 12 of 60 clips, and
  Tadabur audio peaks at 1.037 so it clips real signal too.
- The teacher decodes once. Besides halving every later evaluation it makes two checkpoints
  comparable by construction rather than by hoping the teacher ran identically twice.

**The decode protocol also had a defect.** `confirmed_stream` committed only the oldest second
of each window *including the last*, so the final four seconds of every clip were decoded and
discarded, and a clip under 5 s was transcribed from its first second alone. Muraja flushes
what is pending when speech stops. The flush is replayed now, behind `PROTOCOL_VERSION`, which
cached decodes carry so a set built under one protocol cannot be scored under another. Numbers
either side of that change are not comparable.

### What the h384 checkpoint scores

Unchanged, 2,000 held-out clips over 286 reciters, 167,473 teacher phonemes, scored at the
manifest's batch size:

| | `h384_klonly` @40k | warm-start, +10k steps on unseen audio |
| --- | --- | --- |
| **pooled character accuracy** | **90.35%** [89.74, 90.89] | **91.12%** [90.59, 91.58] |
| macro (per-clip mean) | 91.02% | 91.69% |
| median clip | 92.11% | 92.67% |
| exact-match clips | 7.8% | 8.2% |
| per-clip error p90 | 16.22% | 15.38% |
| short clips / long clips | 91.82% / 89.82% | 92.49% / 90.62% |

Intervals are bootstrapped over **reciters**: 2,000 clips come from 286 voices and agreement
correlates within one, so an independent-sample interval is about 23% too narrow.

**Pooled is the target and macro is a diagnostic.** For cloning, every teacher phoneme is a
behaviour to reproduce and should count once, which is what pooling does; a per-clip mean
answers "how good is a uniformly chosen clip", a different question with a different error
budget. The two differ by ~0.7 points here, so a threshold stated without the aggregation is
not a threshold.

**Measurement precision.** The teacher is **bit-identical** on a re-run at the same batch size
(120 clips, zero edits), so differences are real rather than jitter — but the decode is bf16
and moves 0.17% of characters between batch 4 and batch 32, which the manifest now pins.

### The protocol is far less stable than the model

The perturbations above change the *audio*. This one changes only **where the 1 s window grid
falls**, by prepending silence — not one phoneme of content moves. Teacher against teacher,
250 clips:

| grid shift | char agreement | exact-match clips |
| --- | --- | --- |
| none (same grid) | 100.00% | 100% |
| 1/4 hop (0.25 s) | 82.44% | 7.2% |
| **1/2 hop (0.50 s)** | **78.86%** | 5.6% |
| 3/4 hop (0.75 s) | 82.29% | 5.6% |

The result validates itself: agreement is symmetric about the half-hop and worst exactly
there, which is the signature of grid phase (distance to the nearest original boundary is
1/4, 1/2, 1/4) and not of the added silence.

**The teacher reproduces itself far worse than the student reproduces the teacher.** The
student is at 92.85% on a fixed grid; the teacher is at 78.9% against itself when the grid
moves by half a hop. Three consequences:

1. **The 95% target is not near a noise floor.** On a fixed grid the teacher is bit-exact, so
   the ceiling for the metric as measured is 100%. The target stands.
2. **But the absolute number is grid-specific.** It is a valid basis for comparing
   checkpoints — they are all scored on the same grid — and it is *not* a prediction of what
   transcript the device produces, because on a device the grid phase relative to speech
   onset is arbitrary.
3. **It does not, however, contaminate the measurement.** Scoring *both* models on the same
   shifted grid holds agreement flat — 93.27% as measured, 93.19% at +0.25 s, 92.87% at
   +0.50 s — so the reported number is a property of the model and not of the grid it was
   taken on. Phase sensitivity affects both sides identically and cancels.

   It remains a reasonable hypothesis that the student's residual gap is weighted toward
   *spike timing* rather than phoneme identity, since the protocol demonstrably amplifies
   timing differences into character differences; the CTC-distillation literature calls this
   frame-level alignment disagreement and prescribes weighting the frames adjacent to a
   teacher spike rather than all non-blank frames uniformly. But the cross-phase result
   neither confirms nor refutes it, and it should be tested before it is acted on as fact.

### The streaming warm-start: 90.15% -> 93.09%, and where it stopped

40,000 steps warm-started from the staged-corpus checkpoint onto ~1,170 h of Tadabur audio
the student had never seen, lr 5e-5, warmup 500, cosine, EMA 0.999. Dev split, 970 clips:

| step | 8k | 18k | 24k | 30k | 40k |
| --- | --- | --- | --- | --- | --- |
| char accuracy | 90.94% | 92.17% | 92.59% | 92.85% | **93.09%** |
| per 1k steps | — | +0.124 | +0.068 | +0.044 | +0.024 |

Paired against the baseline over the same clips: **+2.94%** [+2.52, +3.44], 628 clips closer
against 123. The per-step rate halves roughly every window, which is the shape of a cosine
tail as much as of a model running out of room — the two are not separable from this run
alone, and a warm-restart probe is what would tell them apart.

**EMA is worth nothing at convergence.** Live and averaged weights at step 40,000 score
93.09% and 93.09%, 5,572 against 5,571 edits. That is the expected result once the schedule
has annealed to ~1% of peak — there is no oscillation left to average away — and it means the
value of keeping it is confined to the middle of a run.

**An inference-time blank bias is not a free win either.** The residual decomposition shows
the student over-emitting (1,534 insertions against 1,102 deletions on 600 clips), which
invites a scalar on the blank logit before the argmax. Swept over the dev split, the optimum
is **zero**: 92.77% at −0.25, **93.09% at 0.00**, 93.02% at +0.25, 92.49% at +0.50. The
student's blank threshold is already calibrated; the insertion excess is distributed, not a
global offset. Twenty minutes of GPU to close a plausible-sounding lever.

**What the residual is made of**, 600 dev clips, 3,570 edits: identity substitutions
**26.2%**, split/merge-shaped adjacent duplicates **18.0%**, missing or extra whole runs the
rest. Top confusions are acoustically sensible (ن→ل, ن→م, ا→َ). Neither a clean identity
problem nor a clean timing one.

**And a training/evaluation mismatch worth fixing before the next objective experiment:**
because the final window is flushed, **39.7% of all scored timesteps come from frames 25-124
of one window** — 50% at the median clip length, 83% at 6 s — and training weights those at
1x while giving the committed region 2x. Raising ``confirm_weight`` would push weight further
away from two fifths of the scored output. The indicated experiment is the opposite one.

### How stable the decode is, and what that does *not* tell us

The teacher against itself, 250 clips, under perturbations that carry no information:

| perturbation | char agreement |
| --- | --- |
| byte-identical, same batch size | 100.0000% |
| shift by one sample (62 µs) | 99.50% |
| gain +0.1 dB | 99.78% |
| additive noise at −60 dBFS | 98.54% |

These are **robustness probes, not a ceiling on achievable agreement**, and an earlier
revision of this section used them as one. The teacher is deterministic on a fixed input and
the student is given that same input, so a perfect clone would score 100% — nothing here
bounds the target from above. What they do establish is that the teacher is exactly
reproducible at fixed batch size, so the evaluation has no intrinsic floor to subtract, and
that the decode is sensitive enough to sub-perceptual input changes that the *robustness* of
any deployed variant is worth measuring separately.

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

  §1.3's *quality* result does not transfer at the frame level, exactly as feared. On the
  teacher, 6-bit was argmax-identical to INT8, because a 586M model is heavily
  overparameterized. On the 85.5M student it is not. Measured against the uncompressed FP16
  export, over real windows from a step-10000 checkpoint:

  | | size | frame argmax agreement | windows fully identical | chunks |
  | --- | --- | --- | --- | --- |
  | FP16 (uncompressed) | 164.0 MB | reference | — | 2 |
  | 6-bit | **62.3 MB** | 98.91% | 2/11 | 1 |
  | 8-bit | **82.8 MB** | **99.85%** | **9/11** | 1 |

  **That table is the wrong scale, and reading it as a quality decision would have cost
  20.5 MB for nothing.** A flipped frame may leave the decoded string untouched — it is
  inside a run that collapses to the same token — or split a run and cost several edits, so a
  frame-agreement figure cannot be subtracted from a character accuracy. `verify_student_export.py`
  measures the thing itself, replaying the deployed protocol through the CoreML model and
  comparing the resulting phoneme string to the teacher's cached decode. On 300 held-out
  clips (24,390 teacher phonemes) of a mid-run h384:

  | | agreement with the teacher | cost | characters moved vs PyTorch |
  | --- | --- | --- | --- |
  | PyTorch fp32 | 90.94% | — | — |
  | CoreML fp16 | 90.87% | −0.07 | 0.32% |
  | CoreML **8-bit** | 90.91% | **−0.03** | 0.31% |
  | CoreML **6-bit** | 90.84% | **−0.10** | 0.75% |

  **Palettization is not a constraint on the training target**: 6-bit costs a tenth of a
  point, so a student that reaches X in PyTorch ships at about X, and there is no need to
  inflate the target to pay for quantization. 8-bit buys back 0.07 points for 20.5 MB, which
  is not worth it — and note that the conversion to fp16 accounts for essentially all of the
  drift 8-bit shows, so 8-bit adds almost no error of its own.

  The gap between *drift* and *cost* is the interesting part: 6-bit moves 0.75% of characters
  while costing 0.10 points, because the moves are roughly orthogonal to the teacher rather
  than away from it. Frame-level tables cannot see that and will always read pessimistic.

  **The consequence for sizing.** h384 at 8-bit is 82.8 MB and h448 at 6-bit is 83.2 MB — the
  same byte budget, one chunk either way. Since precision is nearly free and capacity is not,
  the same bytes should be spent on **width**, not on bits.

  Two caveats: the checkpoint is mid-training (a sharper model is expected to be *more*
  robust, so read the costs as an upper bound), and this runs on a Mac's ANE rather than an
  iPhone 13's. Single-chunk acceptance is still enforced at `MLModel` load on device and
  still unproven.

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

- **Check what the metric is a metric *of* before optimising it.** This work spent two passes
  improving agreement with `Scorer.gate`, first at the corpus filter's threshold and then at
  Muraja's, before noticing that a distillation should not be scored through an aligner at
  all. ADR-0005 and ADR-0008 already said so and were in this repo the whole time. When an
  issue states a target in a metric's terms, resolve what that metric decides — and whose
  decision it is — before treating the number as the goal.

- **The distillation metric is decode fidelity.** Cloning the teacher's phoneme stream is the
  objective, so the measurement is that stream against the student's. Filtering is
  fine-tune-side and follow-along grading is product-side; both take a decode as input and
  neither is a property of the student. Reporting either as the headline makes a size
  distillation answer for someone else's policy.

- **Interval clustered data as clustered.** 2,000 evaluation clips come from 286 reciters and
  agreement correlates within a voice, so the independent-sample interval is ~23% too narrow
  on exactly the question a checkpoint comparison asks. Resampling reciters costs nothing.

- **Report the pooled number and the per-clip distribution together.** Pooled accuracy is
  dominated by long clips; the median clip is 1.7 points better and moves independently. One
  of them alone will eventually say a change helped when it did not.

- **A synthetic perturbation is only evidence if its edit distribution is neutral with respect
  to what is being measured.** A gate-sensitivity probe here inserted a *duplicate* of the
  neighbouring phoneme — which is literally a geminate — and so manufactured the
  added-shadda finding it was used to support, inflating it five-fold.

- **This must not be confounded with the ADR-0001 track.** That fine-tune deliberately
  *increases* tolerance on the soft pairs; width distillation will involuntarily *reduce*
  discrimination on those same pairs. Distil against the current teacher and gate on
  agreement first; apply tolerance changes separately, or distil from an already-fine-tuned
  teacher. Doing both at once makes ADR-0001's success criterion unmeasurable.

## Outcome: h448 met the target, and capacity was the lever after all

Issue #75 set out to raise decode agreement from 93.09% to 94%. Resolved at **94.97%**, on
the held-out reciter-disjoint test half, against a pre-registered acceptance rule.

| | `h384_stream_warm` | **`h448_stream`** |
| --- | --- | --- |
| dev char accuracy (970 clips / 146 reciters) | 93.09% [92.47, 93.60] | **94.87%** [94.47, 95.20] |
| **test char accuracy (1,030 / 140)** | — | **94.97%** [94.63, 95.27] |
| paired delta vs h384, test | — | **+1.67** [+1.49, +1.87] |
| edits / characters, test | — | 4,368 / 86,862 |
| exact-match clips | 12.7% | 17.4% (dev) |
| median / p90 clip error | 5.56% / 12.33% | 4.00% / 9.15% (dev) |

Recipe: `teacher_init --qk damp`, 40k streamed steps, WSD (1000 warmup, hold, 4000 cooldown),
lr 1e-4, batch 32, EMA 0.999, 11.5 h on one RTX 5060 Ti. lr 1e-4 rather than the 5e-5 of
`h384_stream_warm` because that was a *warm-start* rate; the comparable from-init stage is
`h384_klonly`, which used 1e-4 / warmup 1000.

**Width is not isolated.** `h448` trained on the streamed corpus from step one, while
`h384`'s first 40k ran on the 61.8 h staged corpus. Width, recipe and data exposure all
differ, so the established claim is that *this h448 recipe* beats the benchmark. A clean
width attribution would need a matched `h384` teacher-init run on the same stream, which the
target no longer requires.

### The export is verified and the size holds

| | |
| --- | --- |
| parameters | 116,318,635 |
| trace verification | max abs diff 0.00e+00 |
| FP16 `.mlpackage` | 222.6 MB |
| **6-bit `.mlpackage`** | **84.4 MB** |
| compiled `.mlmodelc` | 84.5 MB |
| largest chunk previously shipped | 99.0 MB |

**6-bit costs nothing on a converged model.** On 300 dev clips (24,390 phonemes): pytorch
fp32 94.71%, coreml 6-bit 94.74%, coreml fp16 94.70% — a spread of ~10 characters, i.e.
zero. The −0.10 recorded earlier in this ADR came from a *mid-run* h384 and should be read
as an upper bound, exactly as `verify_student_export.py` predicted. Precision is not a
constraint on this pipeline: a student that reaches X in PyTorch ships at X, and h448 at
6-bit (84.4 MB) is cheaper than h384 at 8-bit would have been.

Note the sizes are **MB, not MiB**. The 80.5 figure quoted for a 112.5M student earlier in
this work was MiB; idealised 6-bit weight arithmetic for 116.3M gives 87.2 MB and the real
artifact is 84.4 MB, because not every tensor palettizes to 6 bits. Only the compiled
artifact answers this question — state the unit and measure the bytes.

**Single-chunk acceptance on device is still unproven.** That budget is enforced at
`MLModel` load on the iPhone ANE and, per `ml-model-transformation.md` §2.5, fails
*silently to CPU* rather than raising. 84.4 MB against a 99 MB proven ceiling is strong
evidence, not proof, and `ane_ms` for h448 is unmeasured.

### What did not move it

Every cheap lever was tried first as a 10k-step warm restart from the 93.09% checkpoint,
each paired against a matched control. All were within noise of zero:

| arm | changed | paired delta |
| --- | --- | --- |
| WSD restart | LR schedule | +0.012 [−0.091, +0.120] |
| `confirm_weight` 2→4 | objective weighting | −0.024 [−0.102, +0.059] |
| unseen shards (live) | training data | +0.036 [−0.078, +0.145] |
| unseen shards (EMA) | training data | +0.113 [+0.013, +0.221] |

Only the last excludes zero, and it is one result among six paired tests with 77% of its
net gain concentrated in 10 of 146 reciters — worth ~+0.05 to +0.08 after shrinkage. Set
against h448's +1.78, none of these was the lever. Fresh data is also nearly exhausted:
~20 of 345 trainable shards remain untouched (~68 h), against the ~1,170 h that once bought
+2.94.

### The commit position is Pareto-dominated, and that is a product finding

`training/window_position.py` generalises the commit rule to all five within-window blocks.
Block *b* of window *w* covers absolute `[w+b, w+b+1)`, so the same audio second can be
decoded five ways. Student-teacher agreement, h384 / h448:

| block | left context | lookahead | agreement |
| --- | --- | --- | --- |
| **b=0 — deployed** | 0.0 s | **4 s** | **91.63% / 93.57%** |
| b=1 | 1.0 s | 3 s | 97.40% / 98.39% |
| b=2 | 2.0 s | 2 s | 97.52% / 98.44% |
| b=3 | 3.0 s | 1 s | 97.22% / 98.18% |
| b=4 | 4.0 s | 0 s | 81.49% / 85.35% |

Lookahead is `4 − b` seconds, because window `[w, w+5)` completes at `w+5` while block *b*
ends at `w+b+1`. **b=1 therefore beats the deployed b=0 on both axes at once** — roughly
five points of agreement *and* one second less latency. The two truncated-context positions
are the two weak ones, and h448's capacity narrowed the b=0 penalty (b=1 − b=0: 5.77 → 4.82)
rather than leaving it untouched.

**The qualifier is load-bearing.** The teacher disagrees with *itself* ~17% between b=0 and
b=1, so committing b=1 moves the reference as well as the student. The table shows agreement
is better at b=1; it does **not** show b=1 is more *correct*. Deciding that needs an
independent transcript reference, not teacher agreement. Startup also needs a rule (output
second 0 under b=1 would want window −1), and the change interacts with ADR-0009's
asymmetric-cost operating point. This belongs in its own issue and its own ADR.

### Lessons this round added

- **A null with a wide interval is not a falsification.** The `confirm_weight` arm moved the
  committed-region error rate from 8.1839% to 8.1821% and that was reported as "no response
  from the targeted region". It is ~1 edit in 55,800 characters — noise at that precision,
  and the arm's CI was wide enough to contain a useful effect.
- **Compute directional claims; do not eyeball them.** Four claims about the block table —
  which block was worst, whether capacity closed the gap, whether the blocks improved
  uniformly, and which direction latency moved — were each wrong and each one line of
  arithmetic from being right. The latency error pointed the wrong way on a change that is
  free on both axes.
- **Pre-register before opening a held-out panel.** The candidate, endpoint and acceptance
  rule for the test half were fixed in writing before it was scored. The test half is now
  spent; any future candidate needs a freshly built one.
- **Normalising a weighted loss removes the scale, not the mixture.** `weighted_kl` is a
  weighted mean, so re-weighting cannot change loss scale — but it still changes gradient
  norm, direction and conditioning, and "loss scale cannot change" was too strong a claim.


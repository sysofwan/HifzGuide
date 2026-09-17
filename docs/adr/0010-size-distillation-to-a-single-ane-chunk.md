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

- **The loss is a frame-weighted logit KL plus tapped feature matching**, and both weightings
  exist because an unweighted mean optimises the wrong thing:

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

- **A CTC anchor on the teacher's decoded sequence is part of the objective, not an
  optional extra.** The frame-weighted KL and the feature loss are both *per-frame*
  objectives, and a per-frame objective cannot break alignment symmetry: until the student
  knows which frames carry which phoneme, blank is the locally optimal answer at every
  individual frame, so all-blank is a stable fixed point. Reweighting frames does not help,
  because the problem is not which frames are weighted.

  Measured, not assumed. Running the KL-only recipe to step 2000 and then measuring over
  teacher-non-blank frames gave P(teacher's class) = **0.027** at rank **8.2**, against
  P(blank) = **0.822** — while top-5 agreement of **0.51** (chance: 0.12) showed the encoder
  had genuinely learned. The representation was forming; the alignment was not.

  `ctc_anchor_loss` is the standard fix: a *sequence* objective whose forward-backward sums
  over every valid alignment, under which an all-blank output has probability zero for any
  non-empty target. The basin stops being a fixed point.

- **The release gate is confirmed-stream agreement, not frame agreement.** Frame agreement is
  cheap and smooth and so is used during training, but it averages over 100 timesteps the
  user never sees. `training.distill_eval` replays the deployed sliding-window protocol —
  5 s window, 1 s hop, `scanCTC` collapse, `midpoint < 25` confirmation — and compares the
  resulting transcripts. That is the number that predicts what Muraja does.

## Consequences

- **The corpus problem disappears, and a small corpus suffices to start.** 61.8 hours of
  16 kHz recitation already on the GPU box (`audit_run/clips_v2`, 18,075 clips) yields 80,250
  training windows at a 2.5 s stride. Unlabelled, unfiltered audio is usable, so this scales
  to the remaining Tadabur shards whenever the student needs more.

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

- **Palettization itself behaves exactly as advertised; the risk is accuracy, not size.**
  Both measured exports land at the nominal 6/8 bytes per graph value (0.753 and 0.756), so
  there is no compression surcharge to budget for. What may not transfer is §1.3's *quality*
  result: 6-bit held argmax-identical to INT8 on the teacher because a 586M model is heavily
  overparameterized, and an 85.5M student trained to the edge of its capacity has far less
  redundancy. 8-bit is the fallback at ~82 MB — still a single chunk, so the architecture
  does not change either way — but it must be measured as its own row rather than assumed.

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

- **This must not be confounded with the ADR-0001 track.** That fine-tune deliberately
  *increases* tolerance on the soft pairs; width distillation will involuntarily *reduce*
  discrimination on those same pairs. Distil against the current teacher and gate on
  agreement first; apply tolerance changes separately, or distil from an already-fine-tuned
  teacher. Doing both at once makes ADR-0001's success criterion unmeasurable.

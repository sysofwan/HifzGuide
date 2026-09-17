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

- **Target `h384`; ship whatever the agreement curve justifies.** Measured by instantiating
  each preset, with the 6-bit size model calibrated against our own 504 MB / 672 MB INT8
  chunk table:

  | preset | params | 6-bit | chunks | compute vs teacher |
  | --- | --- | --- | --- | --- |
  | teacher | 586.4M | 486.5 MB* | 6 (actual) | 1.0x |
  | h512 | 151.9M | 126.0 MB | 2 | 4.0x less |
  | **h384** | **85.7M** | **71.1 MB** | **1** | **7.1x less** |
  | h256 | 38.3M | 31.8 MB | 1 | 16.0x less |

  (*the analytic count runs ~3.5% under the real teacher, which measures 504 MB.)

  `h384` is the knee: it clears the 99 MB largest-chunk-we-have-actually-compiled with
  margin, and 7.1x less compute takes the A15 duty cycle from ~90–120% to roughly 20%. Two
  chunks (`h512`) is an acceptable fallback if agreement does not hold; the goal is the
  smallest model without significant tradeoff, decided by measurement rather than by this
  table.

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

- **Blank collapse is the main training risk.** The trivial solution — predict blank
  everywhere — scores ~67% frame agreement for free, because that is the teacher's blank
  rate. It is the observed starting basin in practice. `agreement_stats` reports
  `blank_collapse_margin` and `nonblank_agreement` every logging step precisely so a run
  parked there is visible immediately rather than at the end, and `--nonblank-weight` is the
  knob for it.

- **The §1.3 palettization results may not transfer.** 6-bit held argmax-identical to INT8 on
  the teacher because a 578M model is heavily overparameterized. An 85.7M student trained to
  the edge of its capacity has far less redundancy. 8-bit is the fallback at ~99 MB, which is
  still a single chunk, so the architecture does not change either way — but it must be
  measured as its own row rather than assumed.

- **A single chunk removes more than parameters.** Five `copyToFresh` FP16→FP32 conversions
  and five CoreML dispatches per window disappear, so the realised speedup should exceed the
  pure-FLOP estimate. It also retires the chunk-boundary dtype hazard that
  `ml-model-transformation.md` §5 documents as a shipped bug, and the `errno 28` ANE
  temp-cache pressure from compiling six models on low-storage devices.

- **The single-chunk claim is inferred, not measured.** 99 MB is the largest chunk we have
  demonstrably compiled (chunk F); Apple publishes no budget, and §2.5 warns that exceeding
  it fails **silently** to CPU rather than raising. A throwaway export of a
  randomly-initialised `h384` through `convert_to_coreml.py` + `compile_models.sh` settles it
  in a day and should precede the full training run.

- **This must not be confounded with the ADR-0001 track.** That fine-tune deliberately
  *increases* tolerance on the soft pairs; width distillation will involuntarily *reduce*
  discrimination on those same pairs. Distil against the current teacher and gate on
  agreement first; apply tolerance changes separately, or distil from an already-fine-tuned
  teacher. Doing both at once makes ADR-0001's success criterion unmeasurable.

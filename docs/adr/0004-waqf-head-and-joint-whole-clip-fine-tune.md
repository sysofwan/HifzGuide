# Waqf head and joint whole-clip fine-tune

The fine-tune goal (ADR-0001/0003: soft-pair + tashkeel discrimination) is **extended** with a new
**waqf head** distilled from the **Recitation VAD** (`obadx/recitation-segmenter-v2`). Muaalem was
trained on waqf-pre-segmented clips, so it has no notion of waqf; and Muraja today handles waqf at
inference with a blunt hack — it **ignores end-word tashkeel** so a legitimate pause isn't punished,
which also throws away its ability to grade the final haraka (i'raab) and cross-word idgham when the
reciter does **not** pause (wasl). A frame-level waqf signal, running in the *same* forward pass as
the phoneme head, lets the scorer tell waqf from wasl and pick the realized reference form — moving
that tolerance out of a blunt hack and into the model, the same philosophy as ADR-0001.

## Decision

- **Add a waqf head that rides the adapter + CTC output.** A per-frame binary speech/silence head on
  the Muaalem backbone, attached at the **same place as the phoneme head** — after the adapter, on the
  post-downsample CTC lattice — so it exports inside the existing ChunkF (`adapter + CTC head`) with no
  new chunk. That lattice is **40 ms** (the adapter downsamples the 20 ms encoder frames 2×: 250→125
  over a 5 s window, per `ml-model-transformation.md`), whereas the **Recitation VAD teacher is 20 ms**.
  Distillation is therefore **2:1, not 1:1**: the teacher's 20 ms silence posteriors are **pooled to
  the 40 ms grid** (a window is silent iff its two teacher frames are, i.e. max-pool speech / min-pool
  silence) before the KL. Trained by **soft-label distillation**; frame classification, **not CTC** —
  unlike every existing head. No hand-derived waqf labels are invented: the F1≈0.996 VAD *is* the label
  source. Waqf→word-boundary snapping and the 300/700 ms post-processing stay **scorer-side**, now
  reasoning over 40 ms frames (300 ms ≈ 7–8 frames, 700 ms ≈ 17–18), cheap because the phoneme
  alignment is in the same output.

- **Un-waqf-segmented, fixed-window training; segmentation kept only for labels and audit.** The waqf
  head *must* see interior waqf in context, and waqf-segmented clips have their waqf events removed
  (they became the cut points). But the deployed model runs **fixed 5 s windows**, not whole clips, so
  the training unit is a **fixed 5 s window over the un-waqf-segmented recitation** (matching
  inference), inside which interior waqf pauses survive. Each window's phoneme CTC label is sliced from
  the **concatenation of the per-segment realized references** (waqf form at each interior stop, wasl
  inside each run), so ADR-0002's realized-form labelling is preserved without cutting audio *at the
  waqf*. The window/overlap/stitch contract — how a waqf or a word that straddles a window edge is
  owned — is pinned by the inference-contract slice and used **identically** in train, eval, and export.
  Waqf segmentation survives as (1) that label-construction step and (2) the poison-audit unit (#6).

- **Joint schedule, waqf gradient detached from the backbone.** One forward pass, one data pipeline;
  loss = phoneme CTC + waqf distillation KL. The waqf head reads **detached** (stop-gradient)
  backbone features, so the waqf loss contributes **zero** backbone gradient. This makes the joint
  run comparable to a **same-seed phoneme-only whole-clip run** — *not* bit-for-bit the original
  segmented ADR-0001/0003 fine-tune: moving to whole clips already changes batch shapes, padding,
  loss normalization, dropout/RNG order and bucketing. The isolation claim is therefore verified,
  not assumed, with an **ablation ladder**: (1) segmented phoneme-only (the ADR-0001/0003 baseline),
  (2) whole-clip phoneme-only, (3) whole-clip phoneme + detached waqf. The go/no-go is (2)→(3): assert
  backbone gradients (or phoneme logits) are identical between them. (1)→(2) measures the *whole-clip
  move* itself and must independently clear the should-accept / should-reject bars. The waqf head
  still exploits the phoneme-tuned sukun / madd-ʿaariḍ cues; it just cannot reshape them. Unfreeze
  into the backbone only if detached distillation misses the VAD's F1 — and only knowing that
  unfreezing **re-confounds** the eval (treat as a deliberate second backbone objective).

- **Sifat heads: dropped; forgetting is bounded by LoRA, not a KD anchor.** The fine-tune is **LoRA
  on the phoneme head** (ADR-0001), so the backbone's base weights are **frozen** and drift is bounded
  by construction — the catastrophic-forgetting risk a sifat anchor would guard against is already
  what LoRA prevents. Default: **drop the sifat heads** (backbone stays phoneme-only, isolation
  holds). A KD-to-base sifat anchor is *not* neutral — to do anything its gradient must reach the
  backbone, making it a second backbone objective that **confounds** the isolation; detached, it does
  nothing. So it is a **last resort**, and only if ablation (2) shows should-reject regression the
  first levers are **LoRA-native** (lower rank/alpha, or L2-SP on the adapters), not reattaching
  sifat. The exported CoreML model ships **phoneme + waqf heads only** regardless.

- **Inference reuses the deployed fixed-window pipeline.** ChunkF now emits the **waqf frame output
  alongside** the 40 ms phoneme lattice; Muraja runs the same 5 s windows it runs today, stitches the
  per-window waqf frames across the recitation, segments on them, and scores each run against its
  realized (waqf vs wasl) reference. Training matches inference because **both use the same window
  contract**, not because either sees a whole clip.

## Consequences

- **A silence VAD is not a "waqf-correctness" signal — eval must be event-level, not frame-F1.**
  The head detects *silence*, not whether waqf-vs-wasl was correctly realized, so frame-F1 against the
  teacher is only a distillation sanity check, never the product gate. The scorer-facing eval must
  measure, **after** the 300/700 ms post-processing and boundary snap: event-level waqf
  precision/recall, word-boundary snap accuracy, **false-waqf rate** at true wasl boundaries (a
  spurious fire lets the scorer forgive a dropped haraka / missed idgham — the discrimination
  ADR-0001 is trying to regain), **false-wasl rate** at genuine stops (breath-noise / filled-pause /
  sub-300 ms / madd-into-sukun stops the VAD may miss), and a **mid-word-closure rejection set**
  (qalqala on ق/ط, hamza in شَيء — silence the snap must *not* treat as waqf). Beware teacher
  circularity: the VAD labels the head **and** is the frame-F1 target, so a systematic VAD error in
  amateur audio can pass frame-F1 while failing the recitation task.

- **The blank-run baseline is already known to fail — the head is the response, not a gamble.** ADR-0002
  rejected CTC blank-runs as a *segmentation* pause source (they over-split), and empirically they also
  fail on **madd** (elongation confuses the blank pattern). So the waqf head is not competing against a
  live baseline: it exists precisely because the phoneme head's own blanks are an inadequate waqf
  signal. Keep the blank-run + post-processing number only as a **documented reference point** in the
  event-level eval, not a ship/no-ship gate.

- **The end product is conditional-reference scoring, not a VAD clone — it needs its own end-to-end
  gate.** Event-level waqf metrics (above) can look good while the *product goal* — regaining
  wasl-sensitive i'raab / idgham discrimination under `.strict` — still fails. So a dedicated
  integration eval consumes **phoneme logits + predicted waqf events together**: snap → per-run
  realized-reference selection → strict scoring, on adjudicated wasl/waqf cases that turn on a final
  haraka or a cross-word idgham, compared against today's "ignore end-word tashkeel" behaviour. This
  is what the sign-off (#10) must actually clear — not frame-F1, not event-F1 alone.

- **The windowing/overlap/stitch contract is mandatory, not conditional.** The deployed pipeline is
  *already* fixed 5 s windows at a 40 ms lattice with a hardcoded full-window mask — so "single-pass
  whole-recitation" is not available. The contract (window length, overlap, edge-frame ownership, how a
  waqf or word straddling a window boundary is handled, and how per-window waqf frames are stitched)
  must be pinned up front and used identically in training-data construction, eval, and export.

- **Whole-clip labels need clip-level eligibility, not just the two named exclusions.** `segment_score`
  already drops individual segments (short span, repeats, boundary mismatch, …). A window cannot
  concatenate only the *surviving* segments' labels — its audio still contains the dropped spans, so
  CTC would see spoken words with no target. Eligibility is therefore **clip-level**: exclude the whole
  clip if *any* constituent waqf segment is invalid/dropped (in addition to `repeated_recitation` /
  `low_alignment` / over-long / phonetizer-unsupported), and assert contiguous full word coverage over
  the window's audio. `target_len < frames` is checked against the **post-adapter 40 ms logit length**
  per window, not the 20 ms input-feature count.

- **Teacher↔student alignment is 2:1 and must be verified through export.** The 20 ms VAD teacher is
  pooled to the 40 ms student lattice; on top of that the two use different feature extractors, so
  frame counts/padding drift by a few and a 1–2 frame shift moves a boundary snap across a word edge.
  Pin the pooling rule, unit-test the correspondence, and add **golden fixtures** verifying PyTorch
  *and* the palettized ChunkF waqf output map to the same timestamps.

- **Whole-clip/windowed training risks OOM.** Backprop over 5 s windows is heavier than ADR-0002's
  one-clip-at-a-time decode, on a 16 GB RTX 5060 Ti. Mitigate with length bucketing, small batch +
  grad-accum, activation checkpointing, bf16; verify one real batch fits before committing.

- **Distillation loss must be pause-weighted — owned by the training slices.** Frame KL over the 40 ms
  lattice is dominated by the speech-frame majority, so a head can score high frame accuracy while
  missing rare silence/boundary frames. The teacher-posterior pooling + silence/boundary weighting and
  a no-pause-collapse diagnostic are acceptance criteria of the model/training slices; the eval slice
  only *tunes the inference threshold*, it does not define the training objective.

- **Detached-head capacity fallback is a governed loop, not a hope.** A linear binary head on
  *detached* Muaalem features may not reproduce the VAD if the silence cues are not linearly present.
  A held-out distillation/event-metric floor gates the outcome; if missed, remedies fire in order —
  small MLP head → pause-weighted retune → partial backbone unfreeze → blank-run fallback — and any
  **unfreeze re-runs the isolation ladder and the integration eval** because it breaks the isolation
  claim.

- **Export supersedes the phoneme-only #11.** #11 assumes conversion is unchanged, phoneme-only, and
  needs no output-shape change. Adding the waqf output to ChunkF and re-verifying alignment through
  palettization contradicts that, so #11 is **superseded**: early export plumbing + a published
  model-I/O schema (output names, 40 ms lattice shape, speech-vs-silence polarity, logit/probability
  semantics) + golden fixtures on an untrained model; the final palettized artifact and refreshed
  release manifest/checksums come after sign-off.

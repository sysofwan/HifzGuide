# Tashkeel in the Tadabur fine-tune labels

The `.balanced` scorer normalizes both sides before Smith-Waterman by collapsing each phoneme
group to its **bare core** consonant, which **strips every short vowel** (fatha/damma/kasra) —
correct for the gate (it mirrors Muraja's tolerance), but it means the filter is *blind* to
tashkeel. The Muaalem phoneme head, however, **emits** the three short vowels as real output
classes (ids 32–34), and the fine-tune (issues #7–11) wants to **raise the model's tashkeel
reliability**. The open question was whether wrong-tashkeel recitations poison that goal, and
whether the pipeline should reject them with a per-vowel "color-swap" gate on top of the
segment-level filter.

## Measured error mode

On the preview manifest (`tadabur.segment_score`, 387 segments), comparing the model decode
(`predicted_phonemes`) against the realized reference (`raw_reference_phonemes`) **at the word
level** — bucketing vowels per reference word so the ±1 carrier offset below cannot masquerade as
error:

| vowel | reference occurrences | matched (right color) | omitted (model blank) | swapped (wrong color) |
|---|---:|---:|---:|---:|
| fatha | 4,103 | 96.9% | 2.1% | 1.0% |
| kasra | 1,473 | 97.4% | 1.1% | 1.5% |
| damma | 1,257 | 96.4% | 1.8% | 1.8% |
| **all** | 6,833 | **96.9%** | **1.8%** | **1.3%** |

The model **omits when unsure, it does not confidently swap**: the dangerous case — a committed
*wrong* color — is only **1.3%**, and even that is a mix of model error and genuine reciter
i'raab error, so the truly-poisonous slice is a fraction of a percent. An emitted vowel is right
~97% of the time.

**Per-position comparison is misleading and must not be used.** A naive column-wise harakah
compare shows ~7% color disagreement and ~20% presence/absence disagreement, but that is almost
entirely a **±1 attribution artifact**: the phonetizer and the model attach a vowel to different
cores around the small madd carriers (`ۦ` U+06E6, `ۥ` U+06E5) and hamza, so the *same* vowel reads
as "omitted here, added next door." Word-level multiset comparison cancels it (agreement jumps
96.9%). This offset is a **scoring artifact only** — CTC marginalizes over monotonic alignments,
so it is not a training-label defect.

## Decision

- **Tashkeel goes into the training target.** The CTC label for the fine-tune is the tashkeel-bearing
  `raw_reference_phonemes` (the phonetizer's realized-form reference), **not** the vowel-stripped
  `reference_phonemes` the `.balanced` gate compares against. Both are already carried per segment in
  the manifest, so no new decode or phonetization is needed.

- **No per-vowel color-swap reject gate.** Poison control stays where ADR-0001 put it — at the
  **segment** level (`match_ratio` + the interior-insertion-run reject) plus reciter-reputation
  weighting. A color-swap drop gate is rejected: it would preferentially delete the *model-wrong /
  recitation-correct* segments — exactly the omission-correction examples (the 97%→ gain) that
  improve tashkeel — to chase a <1.3% poison slice the same-model filter cannot even separate from
  its own noise (the B≫C limit of ADR-0001). CTC absorbs minority label noise at this rate.

- **The per-vowel confusion matrix is the two-sided *eval* metric, not a filter.** Reuse the
  fatha/kasra/damma confusion matrix (word-level, ±1-canceled) as the ADR-0001 should-accept /
  should-reject harness for tashkeel: after fine-tuning, verify recall rose (kasra is the weakest,
  ~91–97%) **without** collapsing the model's ability to still flag a genuinely wrong vowel. Aggregate
  vowel accuracy improving while that discrimination collapses is the failure this eval must catch.

## Consequences

- **Feeds P4 data-prep (#8) and the fine-tune (#7–11):** the collator emits `raw_reference_phonemes`
  as the label; the eval harness (#9–11) adds the per-vowel confusion matrix computed word-level.
- **Evaluate tashkeel word-level, never per-position** — the ±1 madd-carrier offset makes
  column-wise harakah metrics unusable; bucket vowels per reference word (multiset compare).
- **Out of scope:** shadda/sukun as trainable targets — they are not distinct phoneme-head classes,
  so this decision covers only the three short vowels the model actually emits.
- **Preview-scale caveat:** the 387-segment preview subset is enough to establish the error mode
  (~97% match, ~1.3% swap); the rates are re-checked before scaling to the full corpus.
  **Fulfilled by [ADR-0005](0005-tashkeel-filtering-and-gating.md)**, which re-measures on all
  499,863 reference vowels: the error mode holds and the swap rate drops to **0.0005** once
  carrier-anchored. ADR-0005 also amends the "no per-vowel color-swap reject gate" decision above
  — it stands at the *segment* level, but a **reciter**-level swap filter is added, since
  thousands of aggregated vowels can separate a systematically non-Hafs reciter where one segment
  cannot.

# Tashkeel is unfiltered by the data gate and unmodelled by our port of it

**Status:** Accepted. Amends the "No per-vowel color-swap reject gate" decision in
[ADR-0003](0003-tashkeel-fine-tune-labels.md) and fulfils its "preview-scale caveat"
(re-check the rates before scaling to the full corpus).
**Amended by [ADR-0008](0008-the-eval-measures-the-decode-not-the-gate.md):** the "unmodelled by our
port" half of this ADR's title is withdrawn as a defect. Our `.strict` model *is* neither of Muraja's two layers — the measurement
below stands — but that is no longer something to fix, because the gate is ADR-0001's
training-data filter and not the fine-tune's metric. The first consequence below is superseded
accordingly; everything about the **filter** and the corpus rates is unaffected.

ADR-0003 put tashkeel into the training target. Two things were assumed rather than checked:
that *something* upstream would keep grossly wrong-vowel recitation out of the labels, and
that the eval harness could see the capability it was training. Neither is true.
## The measurement

`tadabur.normalization.normalize_phonemes` collapses every phoneme group to its **bare core**,
which deletes the combining marks. `match_ratio` is computed on that, so it cannot see a short
vowel. Over 2,000 segments of `seg_v21/manifest_raw.jsonl`, replacing **every** short vowel in
the reference with a different one:

| | result |
|---|---|
| segments whose `match_ratio` changed at all | **0 of 2000** |
| lowest `match_ratio` any fully vowel-scrambled reference scored | **1.0000** |
| `.strict` threshold it must clear | 0.75 |

A recitation with every vowel wrong is a **perfect** match. Two consequences follow:

- **The ADR-0001 filter cannot have rejected a single clip for wrong tashkeel.** Whatever
  tashkeel error exists in the corpus — qira'at variation, reciter i'raab slips, plain
  mistakes — reached the training labels unfiltered. There is no protection anywhere.
- **Our `.strict` model is neither layer, and understates both.** What this repo ported from
  Muraja is the *alignment* score, and `eval_report.strict_accept` reconstructs `.strict` from
  it plus a soft-pair rule. The app's **advancement** decision genuinely is vowel-blind, but its
  **word grading** is not — it expands the alignment back to original space and emits a
  `tashkeelError` grade. Our harness applies the word-level threshold to the clip-level blind
  ratio, so it models neither. The blindness is real in *this* repo; it is **not** a product gap.

### The two layers

Muraja decides two different things with two different sensitivities, and conflating them is
what produced both of my wrong claims above.

| decision | where | vowel-sensitive? |
|---|---|---|
| **advancement** — does the reciter move forward | `QuranFollowAlong.swift`, `matchRatio` on the *normalized* strings vs a hard-coded `correctThreshold = 0.70` | **No.** Vowel-blind, and `scoringMode` does not touch it |
| **word grading** — what the user is shown, what feeds the page score | `computeWordStatuses`, original-space expansion | **Yes.** Emits a `tashkeelError` grade |

So an all-vowels-wrong recitation still **advances** — its normalized ratio is 1.0 — while every
word is **graded** `tashkeelError`. The app catches the mistake by showing it, not by refusing
to move.

`.strict` (`correctThreshold 0.75`, `shaddahSuppression: false`, `suppressHarakaDrop: false`)
is a **word-grading** parameter set. Three of its rules bear on the decisions below:

| rule | behaviour |
|---|---|
| **wrong** haraka (damma heard as kasra) | flagged in **every** mode, never exempt |
| **dropped** haraka (no vowel emitted) | exempt only on waw/alif/hamzah/ya — the letters that double as madd carriers — and at waqf-final; lenient exempts all |
| shaddah-expansion gaps | **not** suppressed in `.strict`; they demote a word through the phoneme gate |

The first two are the same swap/omission split this ADR arrives at from the corpus side: a
confident wrong vowel is an error, a declined vowel on an unstable carrier is not. Two
independent derivations landing on the same rule is the strongest evidence either has.

`eval_report.strict_accept` sits between the two layers: it applies the `.strict` **word**
threshold of 0.75 to a **clip-level, vowel-blind** ratio. It is therefore not a faithful model
of either decision, and the `per_contrast` shadda row scoring **0/12** for every model
including base follows from that, not from any property of the app. Resolving which decision
the harness should predict is a design question, not a port — see issue #55.

## Full-corpus vowel rates

ADR-0003's rates, re-measured on the full corpus (499,863 reference vowels, 24,676 segments,
carrier-anchored via `training.tashkeel_eval.score_vowels`) rather than the 387-segment preview:

| outcome | rate | ADR-0003 preview |
|---|---|---|
| matched | 0.9617 | 0.969 |
| omitted (model declined to commit) | 0.0234 | 0.018 |
| unanchored (right vowel, carrier missed) | 0.0144 | — |
| **swapped (wrong colour on a carrier the decode got right)** | **0.0005** | 0.013 |
| unanchored_wrong (wrong colour on a misheard carrier) | 0.00004 | — |
| spurious (decode-side, no reference vowel) | 0.0029 | — |

**Both unanchored buckets are excluded from `swap_rate`.** A vowel hung on a consonant the
model misheard says nothing about i'raab — it is a decode failure. Counting it as a swap would
let consonant errors inflate the very rate the reciter filter reads as evidence of a different
qira'ah. This was wrong when the ADR was first written: 21 of the 262 reported swaps sat on
mismatched carriers, and the `swapped` branch checked no anchor even though `matched` did.

ADR-0003's core claim is **confirmed and strengthened**: the model omits when unsure, it does
not confidently swap. The preview's 1.3% "swap" is, at full scale and with carrier anchoring,
almost entirely the 1.44% *unanchored* bucket — the right vowel on a carrier the model missed,
not a wrong vowel. That said, the preview subset has not been re-scored under the current
classifier, so this is an inference about the mechanism rather than a direct re-measurement.

## Decision

- **Add a reciter-layer tashkeel filter** (`tadabur.reciter_tashkeel`). ADR-0003 rejected a
  per-vowel color-swap gate at the **segment** level, and that rejection stands — at segment
  scale the signal cannot be separated from the same model's own noise (the B≫C limit of
  ADR-0001), and it would preferentially delete the omission-correction examples the fine-tune
  needs. A **reciter** aggregates thousands of vowels, so a systematically non-Hafs reciter is
  separable where a single segment is not. Exclusion is by the **Wilson lower bound** on the
  reciter's swap rate, so a reciter is never dropped on a handful of noisy vowels.

- **This filter currently excludes nobody, and that is the finding — with a stated limit.**
  Across the 163 reciters clearing the 500-vowel evidence floor: swap rate median **0.0000**,
  p90 0.0017, max 0.0056 — every one an order of magnitude below the 1% default threshold, with
  no separated high-rate group. Those 163 cover ~91% of the corpus's reference vowels, so this
  is strong evidence about *corpus mass*; it says little about the 339 low-volume reciter IDs,
  and median/p90/max is not a modality test. The 1% threshold also has **no validated positive
  control** — no known non-Hafs reciter has been run through it to confirm it would fire. So the
  honest claim is "no high-swap population detected among the reciters carrying almost all the
  data", not "no qira'ah contamination". The gate ships as a **standing guard**, not as a corpus
  edit, and not as proof.

- **Do not filter on the `omitted` rate.** It is 3.6× spread across reciters (median 0.0225,
  max 0.0821) and tracks audio quality, pace and accent — model uncertainty, not reciter error.
  Gating on it would silently delete hard-but-correct audio, which is the opposite of ADR-0003's
  goal. Muraja reaches the same conclusion from the product side: a dropped haraka is exempt on
  exactly the letters where the model is unstable, while a *wrong* haraka never is.
- **`text_ar_uthmani` cannot supply a qira'ah label.** Over 979 comparable ayat the upstream
  text is identical to our Hafs reference modulo Quranic annotation signs (U+06D6–U+06ED) and
  our own per-segment truncation — **0 genuine text differences**, and **0 of 910 ayat carry
  more than one distinct text**. The dataset publishes no qira'ah field at all. Deducing it from
  the model decode is the only route, and it is the one this ADR takes.

## Consequences

- **The tashkeel objective has no gate *in this repo*.** ADR-0003 asked the eval harness to
  verify tashkeel recall rose "without collapsing the model's ability to still flag a genuinely
  wrong vowel". `training.tashkeel_eval` and `training.counterfactual_eval` measure it, but
  `eval_report.strict_accept` — what the ablation ladder and the #10 gates are read off — models
  only the alignment half of `.strict`. Until it models the original-space expansion, our
  should-reject numbers are a **lower bound** on what the app would catch, and the fine-tune is
  being judged by a weaker gate than the one it ships behind. Porting that layer is out of scope
  here and needs its own issue.
  **Superseded by [ADR-0008](0008-the-eval-measures-the-decode-not-the-gate.md).** That issue was #55,
  and the answer is that the layer is not ported at all. The premise above — that `strict_accept` should model what the app ships — is the
  category error ADR-0008 corrects: the fine-tune is judged on the **decode** (does the model emit
  the vowel and the consonant the reciter actually said), and `strict_accept` reverts to being the
  ADR-0001 training-data filter. Its vowel-blindness stays a real finding about the **training
  corpus** — the first bullet of the Decision above, wired report-only by #58 — and stops being a
  defect in the eval.
- **Any tashkeel conclusion drawn from `match_ratio`, `strict_accept`, or the `per_contrast`
  shadda row is void by construction.** Use the vowel-aware harnesses.
- **The reciter filter is measurement-mediated.** Swap is observed through the model's own
  decode, so a model that reconstructs vowels from the fixed Quran text would mask the
  deviations the filter hunts. `training.counterfactual_eval` is what bounds that failure mode,
  and it must stay green for this filter to mean anything.

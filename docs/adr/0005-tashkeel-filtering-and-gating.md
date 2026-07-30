# Tashkeel is unfiltered by the data gate and ungated in the product

**Status:** Accepted. Amends the "No per-vowel color-swap reject gate" decision in
[ADR-0003](0003-tashkeel-fine-tune-labels.md) and fulfils its "preview-scale caveat"
(re-check the rates before scaling to the full corpus).

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
- **Muraja's `.strict` gate cannot flag wrong tashkeel or wrong shadda.**
  `eval_report.strict_accept` rejects only on soft-pair substitution plus `match_ratio`. This
  is why shadda scores **0/12** should-reject for the base model and every fine-tune alike —
  a scorer property being misread as a model regression.

## Full-corpus vowel rates

ADR-0003's rates, re-measured on the full corpus (499,863 reference vowels, 24,676 segments,
carrier-anchored via `training.tashkeel_eval.score_vowels`) rather than the 387-segment preview:

| outcome | rate | ADR-0003 preview |
|---|---|---|
| matched | 0.9617 | 0.969 |
| omitted (model declined to commit) | 0.0234 | 0.018 |
| unanchored (right vowel, carrier missed) | 0.0144 | — |
| **swapped (confidently wrong colour)** | **0.0005** | 0.013 |
| spurious (decode-side, no reference vowel) | 0.0029 | — |

ADR-0003's core claim is **confirmed and strengthened**: the model omits when unsure, it does
not confidently swap. The preview's 1.3% "swap" is, at full scale and with carrier anchoring,
almost entirely the 1.44% *unanchored* bucket — the right vowel on a carrier the model missed,
not a wrong vowel.

## Decision

- **Add a reciter-layer tashkeel filter** (`tadabur.reciter_tashkeel`). ADR-0003 rejected a
  per-vowel color-swap gate at the **segment** level, and that rejection stands — at segment
  scale the signal cannot be separated from the same model's own noise (the B≫C limit of
  ADR-0001), and it would preferentially delete the omission-correction examples the fine-tune
  needs. A **reciter** aggregates thousands of vowels, so a systematically non-Hafs reciter is
  separable where a single segment is not. Exclusion is by the **Wilson lower bound** on the
  reciter's swap rate, so a reciter is never dropped on a handful of noisy vowels.

- **This filter currently excludes nobody, and that is the finding.** Across 163 reciters with
  ≥500 reference vowels: swap rate median **0.0000**, p90 0.0017, max 0.0056 — every reciter is
  an order of magnitude below the 1% default threshold. The distribution is **unimodal with no
  outlier population**. A qira'ah differing from Hafs on even 1% of vowels would stand out
  clearly here; nothing does. The gate ships as a **standing guard**, not as a corpus edit.

- **Do not filter on the `omitted` rate.** It is 3.6× spread across reciters (median 0.0225,
  max 0.0821) and tracks audio quality, pace and accent — model uncertainty, not reciter error.
  Gating on it would silently delete hard-but-correct audio, which is the opposite of ADR-0003's
  goal.
- **`text_ar_uthmani` cannot supply a qira'ah label.** Over 979 comparable ayat the upstream
  text is identical to our Hafs reference modulo Quranic annotation signs (U+06D6–U+06ED) and
  our own per-segment truncation — **0 genuine text differences**, and **0 of 910 ayat carry
  more than one distinct text**. The dataset publishes no qira'ah field at all. Deducing it from
  the model decode is the only route, and it is the one this ADR takes.

## Consequences

- **The tashkeel objective still has no product gate.** ADR-0003 asked the eval harness to
  verify tashkeel recall rose "without collapsing the model's ability to still flag a genuinely
  wrong vowel". `training.tashkeel_eval` and `training.counterfactual_eval` measure it, but
  `.strict` — what actually ships — remains blind. Closing that is out of scope here and needs
  its own decision.
- **Any tashkeel conclusion drawn from `match_ratio`, `.strict`, or the `per_contrast` shadda
  row is void by construction.** Use the vowel-aware harnesses.
- **The reciter filter is measurement-mediated.** Swap is observed through the model's own
  decode, so a model that reconstructs vowels from the fixed Quran text would mask the
  deviations the filter hunts. `training.counterfactual_eval` is what bounds that failure mode,
  and it must stay green for this filter to mean anything.

# The two-sided eval measures the decode, not the gate

**Status:** Accepted. Answers the design question in
[#55](https://github.com/sysofwan/HifzGuide/issues/55) — "what should the `.strict` eval gate
predict, advancement or word grading?" — with **neither**: the gate should not be the headline
metric at all. #55 stays open as the implementation of this decision. Amends the reporting layer
of [ADR-0001](0001-tadabur-filter-and-finetune-methodology.md); leaves its filter untouched.

## The confusion this corrects

ADR-0001 states the success criterion in Muraja's vocabulary — the fine-tune should let Muraja
"default to `.strict`". That is a true statement of the *product* consequence, and it was then
implemented as if it were the *measurement*: `tadabur.eval_report.strict_accept` turns each
fixture clip into an accept/reject bit and the headline numbers are read off it.

That conflates two different things which happen to share a scorer:

| | what it is | what it is for |
|---|---|---|
| the `.balanced` Smith-Waterman gate | ADR-0001's **training-data filter** | keep mislabelled clips out of the fine-tune corpus |
| `strict_accept` | the same alignment score at a tighter threshold | *nothing the product does* — see ADR-0005 |

The filter's job is already discharged and it already works: it runs on the **base** model over
the corpus and decides which clips are clean enough to train on. Whether it is vowel-blind is a
question about *training-data hygiene* (ADR-0005's finding, wired as report-only by #58), not a
question about what the fine-tune achieved.

**The fine-tune's objective is transcription fidelity.** The model should emit the vowel the
reciter said and the consonant the reciter said. Every ADR in this series says so directly:
ADR-0001 wants the confusable pairs genuinely discriminated; ADR-0003 wants tashkeel recall up
*without* losing the ability to flag a wrong vowel; ADR-0006 measures whether a deliberate wrong
vowel still gets flagged; ADR-0007 measures whether correct vowels stop being rejected. None of
those is a statement about a clip-level accept bit.

So #55's framing — port Muraja's word grading so the gate becomes vowel-sensitive — would build
a second scorer whose only input is the decode, in order to re-derive from the decode a
lower-resolution version of what the decode already says. The right move is to read the decode.

### This was decided once already, and the drift has a precise origin

[ADR-0003](0003-tashkeel-fine-tune-labels.md) states this ADR's position outright: *"the per-vowel
confusion matrix is the two-sided **eval** metric, not a filter."* It was never carried out. The
fixture sets grew buckets for the six soft pairs, shadda and `marginal`, and **no vowel bucket** —
which [ADR-0007](0007-tashkeel-acceptance-audit.md) later recorded as a structural blindness
without connecting it back to the decision it violated. The vowel measurement eventually
reappeared in a different module (`training.tashkeel_eval`). So this ADR is a **restoration**, not
a new direction.

The drift entered at implementation, and #7 shows exactly how. That issue asked for two things:

| #7 says | level |
|---|---|
| "confirm the model still emits the *distinct* wrong phoneme — discrimination retained, not collapsed" | **decode** |
| "Encodes the success criterion: model lets Muraja default to `.strict` without raising false-negatives on acceptable recitation" | product outcome |

The second is ADR-0001's statement of what success would *mean for the product*. It was
implemented as the *metric*: `_side` counts clips through `strict_accept`, and `discrimination`
is `rejected / total`. The first — the decode-level requirement in the same issue — was also
built, as `soft_pair_confusion`, and then never reported as a headline. Both halves have been in
`tadabur/eval_report.py` since #7; only the wrong one is read.

Naming this matters beyond the history: a product outcome stated in the product's vocabulary is
not automatically a measurement, and the next agent handed one will make the same substitution
unless the distinction is written down. It now is, here and in `CONTEXT.md`.

## The report already contains both, fused

`EvalReport` mixes two kinds of measurement in one object:

| field | level | touches `strict_accept`? |
|---|---|---|
| `soft_pair_confusion` | **decode** — per-pair confusion over aligned columns | no |
| `shadda_confusion` | **decode** — added/dropped gemination occurrences | no |
| `should_accept.recall`, `should_reject.discrimination` | gate | yes |
| `per_contrast`, `clip_outcomes` | gate | yes |

The decode-level half is already the right measurement and was never the headline. This ADR
promotes it and retires the other half from the headline.

### Pooling the two fixture sides cancels the signal

`evaluate` passes the **whole** clip list to `_soft_pair_confusion` and `_shadda_confusion`, so
each matrix sums should-accept and should-reject clips together. The same cell means opposite
things on the two sides:

For the pair `ذ↔ز`, cell `matrix[ذ][ز]` — reference `ذ`, decode `ز`:

| clip side | what the human label asserts | what that cell means |
|---|---|---|
| **should-accept** | the recitation was acceptable, so `ذ` really was said | the model **misheard** — an error |
| **should-reject** (bucket `ذ↔ز`) | the reciter genuinely said `ز` | the model **correctly heard the mistake** — the behaviour we want |

Summed, a fine-tune that gets better at both directions and one that gets worse at both can post
the same number. This is precisely the ADR-0001 failure mode — a discrimination collapsing behind
a stable aggregate — reproduced inside the metric meant to detect it.

## Decision

- **The headline is decode-level contrast fidelity, reported per fixture side.** Split every
  confusion matrix by `verdict`, and report the two sides with their own sign convention:
  should-accept measures **mishearing**, should-reject measures **collapse onto the reference**.
  Never pool them, and never sum them into a single "discrimination" scalar.

- **`strict_accept` is retired from the headline and stays what it is.** It remains available as
  the ADR-0001 filter and as observational context on a report, clearly labelled as the
  data-hygiene gate. Its vowel-blindness is no longer a defect to be fixed here, because it is no
  longer answering a question about the fine-tune. ADR-0005's consequence — that no tashkeel
  filtering ever happened to the training corpus — stands unchanged and is #58's business.

- **Do not port `computeWordStatuses`.** Beyond being the wrong question, it is not portable as
  written: it is a streaming function carrying `currentEndPosition` hold-back to `.pending`, a
  grade ratchet, local re-alignment against a nearby transcription window, and a minimal-word
  boost — none of which are defined for an offline whole-clip eval. Its output is also
  settings-dependent (`GradeFilter.DetectionSettings.isTashkeelOn` is a user toggle, orthogonal
  to `scoringMode`), so "does a tashkeel error count" is not even a property of `.strict`. A port
  would have to invent a clip-level aggregation Muraja never performs — `computeWordStatuses`
  returns `[WordStatus]`, and no shipped code reduces that to an accept bit.

- **The vowel side is not rebuilt; it is joined.** `training.tashkeel_eval` already measures the
  decode at vowel resolution with carrier anchoring — `vowel_sites`, `count_sites`,
  `VowelCounts` with `matched` / `swapped` / `omitted` / `unanchored` / `unanchored_wrong` /
  `spurious`. That is the tashkeel analogue of `soft_pair_confusion` and it is correct. The
  reporting layer joins the consonant matrices (`tadabur.eval_report`) and the vowel counts
  (`training.tashkeel_eval`) into one artifact rather than growing a second vowel scorer.

- **Advancement is not modelled.** It is a product behaviour, not a training objective, and
  nothing about the fine-tune is judged by it. If a reader wants it, it is
  `match_ratio >= 0.70` on strings the harness already normalizes — but it is not a gate here and
  must not be reported as one.

### What the selection criterion becomes (#62)

Checkpoint selection, early stopping and the LoRA lever select on decode-level fidelity:
per-colour vowel outcomes from `tashkeel_eval` and per-pair confusion from `eval_report`, each
split by fixture side. Not `strict_accept`, in any form.

## Consequences

- **Every headline number in the #10 table is superseded**, including the ones already retracted.
  The ablation ladder (`training.ablation_ladder`) reads `should_accept.recall` and
  `should_reject.discrimination` out of the report JSON and must be re-pointed at the decode-level
  quantities; `tadabur.signoff_results` reads the ladder. Both are mechanical changes, but the
  stored `ablation_ladder*.json` artifacts in `audit_run/seg_v21/` describe the old quantity and
  should not be compared across the change.

- **#57 is unblocked and its subject changes.** The base-vs-fine-tuned diff is now a diff of
  per-side confusion matrices, which is what its acceptance criteria (denominators, per-clip
  fingerprints, McNemar over `clip_outcomes`) were already reaching for. The paired test now runs
  over per-clip contrast outcomes rather than accept bits.

- **#59's integration half is unblocked in form but not in content.** `waqf_integration_eval`
  calls its own `strict_accepts`; under this ADR that gate stops being the product criterion, and
  what #35 should show is that conditional-reference selection improves the **decode's** agreement
  with the realized reference. That needs restating in #59 before it runs.

- **The fixture labels are clip-level; the confusion cells are column-level.** A clip labelled
  should-accept can still contain unrelated positions, so a per-side matrix must be restricted to
  the clip's labelled contrast bucket or it will mix graded and ungraded positions. State the
  restriction in the report rather than leaving a reader to assume it.

- **Small n, unchanged.** 87 should-accept and 35 should-reject clips. Moving to column-level
  counting raises the denominator per clip but does **not** make the clips independent — columns
  within a clip are correlated — so intervals must be clip-clustered, not computed as if each
  column were a sample. Reporting a tight binomial interval over columns would be the same
  overclaim ADR-0006 exists to prevent.

- **Ground truth for what was *said* is still absent from the corpus**, which is why the
  should-reject side is read as "did the model collapse onto the reference" rather than "was the
  model right". ADR-0007's audit is the only path to the stronger claim, and only for vowels.

- **Dead code becomes live.** `smith_waterman.RefMatchInfo` already carries a `"tashkeel"` kind
  and a `tashkeel(expected, heard)` constructor that **nothing calls**, and
  `normalization.map_to_original` / `map_to_original_end` are exercised only by their own tests.
  They were ported in anticipation of the original-space expansion this ADR declines to build.
  Either wire them to the vowel reporting or delete them — leaving a half-built hook implies a
  layer that, per this decision, is not coming.

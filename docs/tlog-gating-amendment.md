# tlog S7 gating: `both`, base teacher only, and an unfiltered training pool (2026-10-28)

**Status:** owner-approved on 2026-10-28, after the S6 audit closed — **post-look**, dated, and
frozen. This is an amendment, not an edit. The frozen
[`docs/tlog-holdout-preregistration.md`](tlog-holdout-preregistration.md) (SHA-256
`204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086`) and the dated documents
below are never changed. This document supersedes their **gating** rule and the 5% rule's
training consequence; every other rule stands.

Approval: [#173 comment 6466464463](https://github.com/sysofwan/HifzGuide/issues/173#issuecomment-6466464463).
Parents:

- the S6 audit at commit `da76351` (PR #207): [`docs/tlog-audit.md`](tlog-audit.md) and
  `tools/tadabur/tlog/audit/verdicts.jsonl` (150 of 150 owner verdicts);
- [`docs/tlog-ident-floor-preregistration.md`](tlog-ident-floor-preregistration.md) (SHA-256
  `7bd2e1958ea8330df472030aefb6c97de7d0e9ccc2aa81ac133db682e1d58e3a`);
- [`docs/tlog-s6-amendment.md`](tlog-s6-amendment.md) (SHA-256
  `5967fe3008ed822d0b61828da23b0c4d86d079667f203c8bb54d23ddb6402aa4`);
- [`docs/tlog-s6-amendment-note-overlap.md`](tlog-s6-amendment-note-overlap.md) (SHA-256
  `e47856b8f34b1bd10704498a896fdf7fcc88d903705eb665d12763ed675e285a`);
- the S5 pre-outcome amendment [`docs/tlog-ident-preregistration.md`](tlog-ident-preregistration.md)
  (SHA-256 `a383fcfdbd375c9e6c052d8032788a8eda69bd2d9fbd12fcd3b921b1fbb6a3ab`).

The change is defensible after the look for two reasons. It is threshold-free: it names no new
δ, b or floor. And **no candidate has been scored on tlog look**: all S6 decodes and
identifications are base-teacher only (`obadx/muaalem-model-v3_2@01a1ef9f`;
[`docs/tlog-ayah-identification.md`](tlog-ayah-identification.md)), and this finding is the
pre-registered audit's own output.

## 1. Gating items (S7, #192)

A gating item is a clean look or `confirm.k` row with a filename label, not excluded,
non-straddling, and `label_source == both`: the base teacher's best class is the filename ayah's
text class. The recitation floor and the Δ relabel bar no longer decide gating.

`both` is computed from base-teacher decodes only, at the fingerprint above. It is computed and
frozen in each tranche manifest **before any candidate is scored on it**, and never recomputed
with a candidate; otherwise the eval population would depend on the model being evaluated.

On clean look this is **2,853** rows of 6,327, against **3,059** under the frozen rule: 208
floor-failing `both` rows enter and 414 below-bar rows leave (about 385 of those 414 carried the
wrong reference). The 1,293 straddling `both` rows (31%) stay reported-only, as frozen.
`unclean` and `identified` rows remain reported strata.

## 2. Why the Δ relabel rule is out: a window defect

`score_window` scores a decode against the filename ayah **± 1** (`tlog_ident.py:590-595`), and
that window contains the neighbouring ayah. `_with_label` relabels only when Δ ≥ b
(`tlog_ident.py:1117-1121`), so a decode of the neighbour has Δ ≈ 0 and can never be relabelled.
413 of the 414 below-bar clean-look rows have Δ == 0.0 exactly (the 414th is −0.13), and none
overlaps the filename.

The audit measured the consequence. In the below-bar stratum the filename was right **1 of 15**,
the best class **14 of 15**, and **11 of the 14** were the adjacent ayah (e.g. 17:3 → 17:2,
23:8 → 23:7). The stratum is 414 of the 3,059 frozen gating rows (13.5%); 14/15 of them, about
385, carry the wrong reference text (Wilson 70–99%).

## 3. Evidence (S6 audit at `da76351`)

- The identifier is right: its best class is the heard ayah in **129 of 131** decoded recitation
  clips (98.5%, Wilson 94.6–99.6%).
- Recitation floor: **21 of 25** flagged clips were recitation. **7 of 7** flagged `both` clean
  clips were heard as the filename ayah.
- Precision of `both`: **40 of 40** (Wilson 91.2–100%). The 208 admitted rows rest on **10 of
  10** audited floor-failing `both` rows (Wilson 72.2–100%); those rows are harder clips, so §5
  pre-declares a sensitivity line for them.
- Real mistakes survive: all **52** owner-verified real-mistake decodes identify as `both` and
  pass the floor; 51 are non-straddling, and the straddling clip was already outside the frozen
  gating set.
- Floor-passing audited clips: **0 of 110** were non-recitation.
- Silence floor: **7 of 15** silence-excluded clips were recitation (clean 5/7, unclean 2/8).
- Training-pool non-recitation: 397 of the 2,000-row sample are flagged (19.85%) and the flag's
  audited precision is 4/25 = 16%, so the composed estimate is about 3%. By split the flagged
  rate is 15.7% clean and 27.6% unclean; audited, 3 of 12 clean and 1 of 13 unclean flagged
  clips were non-recitation, giving about 4% clean and 2% unclean, weighted about 3%. Principal
  r7's 3/9 and 1/16 split does not reproduce from the committed `worklist.jsonl` and
  `verdicts.jsonl`; the owner-approved 3–5% range covers the recomputed value. Not
  pre-registered; small samples.

## 4. Training (S8, #193)

`tlog.training` keeps the frozen pool predicate — `role == train`, not excluded, codec ok and
not silence — with **no `non_recitation` term**. The 5% rule's consequence (whole-pool decode
plus a `not non_recitation` filter) is withdrawn; the estimate in §3 is disclosed instead.
Training is KL to the teacher and uses no labels.

## 5. Pre-declared readout, selection and statement

- S7 reports both `both` (primary) and `both ∧ floor-pass` (sensitivity) on every gating
  readout. The full `both` set is primary because it is the audio the app hears and the floor
  rejects real recitation (21/25).
- Selection effect: gating means "clips the base teacher names correctly", so it leans toward
  clips the teacher hears reasonably. The frozen floor had the same property and was worse: it
  rejected the 208 hard, floor-failing `both` rows.
- The silence floor is unchanged. It only removes data, never admits it; on clean it removes
  recitation 5/7, so gating under-represents quiet and pause-heavy recordings.
- No candidate has been scored on tlog look. No threshold was tuned; the rule has none.

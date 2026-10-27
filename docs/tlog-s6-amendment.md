# tlog S6: the decode limit, the audit strata and the verdict referent (#191)

**Status:** frozen 2026-10-27, before any tlog audio is decoded. This document amends
`docs/tlog-holdout-preregistration.md` (sha256 `204fe7f9…`) for S6 only, and also governs S7
(#192). It was written after the S4 and S5 results and before the first tlog decode; no S6
outcome existed when it was pinned. The owner's approval was requested on 2026-10-27 and is
non-blocking for the decode.

It records architect r20's decisions Q1, Q3 and Q4. The frozen text's thresholds, exclusions,
roles and audit verdict vocabulary are unchanged; this document fills the procedures the frozen
text leaves open, as `docs/tlog-ident-floor-preregistration.md` (sha256 `7bd2e195…`) does for
the recitation rule.

## 1. Decode mode: whole-clip only, up to a measured limit D_max (r20 Q1)

- Every row is decoded whole-clip, exactly as the frozen pre-registration's Ayah identification
  says, when its pass-1 `duration_s` is at most **D_max**.
- D_max is measured on the GPU box **before any tlog row is decoded**, on seeded synthetic audio
  (no tlog content), whole-clip at the production settings (base teacher `01a1ef9`, bf16, batch
  1, the pinned loader). The ladder is 60, 90, 120, 150, 180, 240 and 300 s. D_max is the
  largest rung that completes with at least **1.5 GiB** of peak memory headroom. The GPU model
  and each rung's peak are committed in `tools/tadabur/tlog/ident/decode_limit.json` before the
  decode pass.
- A row above D_max is **not decoded and not identified**. It is written in the identification
  tree with `decode_mode = "not_decoded:over_d_max"` and `label_source = "none"`; it can never
  be a gating item. It still counts in `tlog.training`, which uses no teacher output.
- **Stop:** if more than 1% of `iter_pool_rows` usable rows are above D_max, S6 stops and asks
  the owner before S7. The alternatives are a stream fallback or a bigger GPU.
- `decode_mode` is required on every decode row and every identification row: `"whole_clip"` or
  `"not_decoded:over_d_max"`. The committed calibration inputs keep their own recorded modes.
- **Disclosure:** the floor and bars were calibrated on recitation clips of at most 20 s. Every
  S6 result is reported by duration band: `0-20` (≤ 20 s), `20-60` (20–60 s) and `60-d_max`
  (60 s to D_max). No new strata are added.

## 2. Artifact paths (r20 Q2)

One file per population and split, sorted by `(split, shard_file, row)`:

```
tools/tadabur/tlog/ident/decodes/look/clean.jsonl.gz
tools/tadabur/tlog/ident/decodes/look/unclean.jsonl.gz
tools/tadabur/tlog/ident/decodes/train_sample.jsonl.gz
tools/tadabur/tlog/ident/look/clean.jsonl.gz
tools/tadabur/tlog/ident/look/unclean.jsonl.gz
tools/tadabur/tlog/ident/train_sample.jsonl.gz
tools/tadabur/tlog/audit/worklist.jsonl
tools/tadabur/tlog/audit/verdicts.jsonl
```

Each tree's `meta.json` carries the model, revision, code SHA, calibration SHA,
`decode_limit.json` SHA, amendment SHA and a per-shard census that lists zero-row shards. No
tlog audio is committed.

## 3. The training sample and the 5% rule (r20 Q5)

- The sample is a seeded simple random sample of 2,000 rows drawn from the usable `train` rows
  **at or below D_max**, by salted rank `sha256("<salt>:<split>:<shard_file>:<row>")`; the first
  2,000 in rank order are the sample. The salt is `tlog-s6-2026-10-27-v1`.
- Rows above D_max are excluded from the 2,000-row denominator and reported (count and share).
  The 5% rule is therefore a statement about rows at or below D_max.
- The rule is the frozen one: if the upper 95% Wilson bound of the sample's `non_recitation`
  rate is above 5%, the whole pool is decoded and `not non_recitation` is added under a new
  overlay commit and a new exposure use. S6 records the outcome and opens a follow-up issue; it
  does nothing else.

## 4. The audit strata (r20 Q3)

- **Population:** usable `look` rows plus the 2,000-row training sample, pooled. Each item
  records whether it came from `look` or the sample.
- **Definitions:**
  - `both`: `label_source == "both"`.
  - relabelled: clean, filename-labelled, `label_source == "identified"`.
  - unclean identified: unclean, `label_source == "identified"`.
  - `non_recitation`: decoded, `id_floor_ok` false.
  - silence: a `look` row whose **only** exclusion is `silence`. It is not decoded, and it is
    drawn from look only. At least 8 items are unclean; the rest are clean, then unclean again
    if clean runs short.
  - below-bar disagreement: filename-labelled, `id_floor_ok` true, best class different from
    the filename class and not relabelled. The blocking reason is `delta` or `overlap`.
- Rows with `decode_mode = "not_decoded:over_d_max"` are in no stratum.
- **Sizes:** `both` 40, relabelled 30, `non_recitation` 25, silence 15, unclean identified 25,
  below-bar disagreement 15.
- **Draw:** the first n of each stratum by salted rank, with the salt above.
- **Shortfall:** a stratum with fewer eligible rows takes all of them and records target,
  eligible and shortfall. The freed items go to `both`, so the audit stays at about 150. No
  other substitution, and no stratum is topped up from another population.
- Wilson intervals are reported at any n; an n below 10 is printed as "n too small to read".

## 5. The verdict referent (r20 Q4)

- Items are shuffled across strata by salted order; the stratum is never shown.
- The page offers the candidate ayat as Arabic text, in salted order and with no attribution
  (the filename ayah, and the best class if different), plus "another ayah" with a `surah:ayah`
  entry, silence, not recitation and unclear.
- The frozen five verdicts are derived in code, scored against the label **under test**:

  | stratum | label under test |
  |---|---|
  | `both` | the shared label |
  | relabelled, unclean identified | the assigned label |
  | below-bar disagreement | the filename label; the best class is also scored, reported only |
  | `non_recitation`, silence | correct when the verdict is not recitation, or silence respectively; recitation verdicts are counted against them |

- The heard `surah:ayah` is stored, so filename precision and best-class precision can both be
  scored per stratum.

## 6. Disclosure only (r20 Q6)

The audit draws no item from the S4 duplicate exclusions, which are not audited:
15.985% of accepted near-duplicate matches are dominated by stop-words (12.7% combined, above
the 1% warning), and 3 look / 54 train usable rows were lost as `survivor_unusable`. The
generated `docs/tlog-audit.md` carries this paragraph; nothing is retuned.

## 7. Order

The synthetic measurement and this document are committed first; `decode_limit.json` is
committed before the first tlog decode. Then the decode pass, then identification, the 5% rule,
the audit draw and session, the verdicts, and finally the generated page. Identification and the
audit wait for `tools/tadabur/tlog/ident/margin_approval.json`; decoding applies no rule and
does not.

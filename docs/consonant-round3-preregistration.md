# Consonant round 3 (tlog): pre-registered bars (#77, #244)

**Status: FINAL TEXT for the freeze commit (architect round-3 r2, 2026-11-01).** The owner approved the
plan and chose G4 = +3.5 on 2026-11-01, after the principal's cross-check (principal round-3 r1). The
freeze issue (#253) commits this text as `docs/consonant-round3-preregistration.md`. It is frozen at that commit,
before any round-3 training, and never edited after that. A change needs a new dated amendment, never a
re-run.

Marks used below:
- **(owner 2026-10-31)**: set by the owner on #244.
- **(amendment A1, owner 2026-11-01)**: the pre-training amendment below.
- **(architect, owner-approved 2026-11-01)**: this text's choices, approved with it.
- Unmarked numbers are round 2's, unchanged (#244 decision 3: "every round-2 bar unchanged except G4").

The only placeholders filled in the freeze commit are the round-3 issue numbers and the SHA-256 of the
round-3 prefix plan.

Parents (cite by SHA-256 when committing):
- `docs/consonant-round2-preregistration.md` (`1860c66d4dc1c8ece0fc81fc84e4e5021783a138ba1445ae790df4172e0b6c3a`)
- `docs/consonant-round2-readout.md` (PR #243, `7d8320c`)
- `docs/consonant-round2-freeze.md`
- `docs/consonant-round2-candidate.md` (#228)
- `docs/tlog-holdout-preregistration.md` (`204fe7f9…`)
- `docs/tlog-gating-amendment.md` (#210)
- the S7 tranche manifests (PR #211, `e11fbf5`)
- `tools/tadabur/tlog/training_schedule.json` (`8669ceca5ec67215a5c0147e97c0f74c97e5b20d29c8b1446cd8212feb8e134d`)

**Statement carried by every verdict of this round:** *clip-level split, same users possible on both
sides* (holdout pre-registration item 8). qdat is an external-corpus check, not proof of
speaker-disjointness from anonymous tlog users.

## Why this round exists

Round 2 failed on G4 alone. Tashkeel-caused false flags per 1,000, candidate − C95, came out at **+0.19
[−1.46, +1.53]**. The bar was ≤ +1.0. That was a width failure, not evidence of harm:
- The +1.0 margin was carried over from Tarteel v1 (~5 tashkeel flags per 1,000) to tlog (12.46 per
  1,000) without rescaling.
- Round 2's freeze measured a one-tranche SE of 0.76 against the 0.31 it had projected. That should have
  triggered a pre-training amendment (architect r31, principal r12).

The owner accepted the FAIL on 2026-10-31. C95 remains the release candidate. Round 3 continues the
round-2 model on more tlog audio. It re-certifies that model against C95 with G4 sized to tlog **before
training**.

Round 3 is not a re-test of the round-2 model. The round-2 model is never decoded on confirm.3 or
confirm.4, and its round-2 margin is not re-derived.

## What changed from round 2, and why

| item | round 2 | round 3 | why |
|---|---|---|---|
| warm start | #178 EMA `05767895…14a2` | **round-2 candidate EMA `893523710186cb16854dd62a258331644e2284abbee8b2cb54e8ab9d32dbea58` (owner 2026-10-31)** | Continue the model whose gains round 2 measured |
| tlog rows | schedule rows 1…73,990 | **the same frozen schedule from row 73,991 on (owner 2026-10-31)**; round 2's 73,990 consumed keys are never read again | "More tlog audio" |
| Tadabur frame | the control's 211 shards (#223) | **those 211 minus the 17 round 2 entered: 194 shards, use `consonant3.training` (architect, owner-approved 2026-11-01)** | The lineage's continuation rule (`training.consumed_prefix`), applied as for `consonant1.training` and `consonant2.control.training` |
| seed | 2026 | **2030** (window shuffle 2031, augmentation 2032) **(architect, owner-approved 2026-11-01)** | "New seed" (owner 2026-10-31); no collision with 2026–2028 |
| control arm | yes (#223), reported | **none (owner 2026-10-31)** | Round 2's control already showed tlog data was the lever (−43.2/1,000) |
| main set | confirm.1 + confirm.2 gating, pooled | **confirm.3 + confirm.4 gating, pooled (owner 2026-10-31)** | confirm.1/2 are spent; holdout item 7 opens tranches in order 1 → 4 |
| G4 | paired 95% UB ≤ +1.0 | **paired 95% UB ≤ +3.5 per 1,000 (amendment A1, owner 2026-11-01)**; registered at +2.5 (owner 2026-10-31), which the power rule rejected | See "Amendment A1" |
| power check | "informative, decides nothing" | **mechanical rule at 0.85, before training (owner 2026-10-31)**, over every guard with a margin | Principal r12; principal round-3 r1 |
| catch-loss audit population | C95 flags, candidate does not | **C95 flags, round-2 model flags, round-3 model does not (owner 2026-10-31)** | Round 2's removals are already audited (5 of 98 real) |
| reproduction check | computed after the confirm read (round 2's own note) | **run in code before any confirm decode is opened, with every input's SHA-256 (#249)** | Round 2's text required "before"; the code did not enforce it |
| ungradable rows | left out, disclosed at readout | **the rule pre-registered; any other exclusion refuses (#248)** | Round 2 never pre-registered it |
| look-side context column | #178 | **round-2 EMA** (look decodes committed by #230) | The model this round continues |

Everything else is round 2's text, carried below.

## What is compared

- **Candidate:** the round-2 candidate's averaged weights (`--init-from … --init-weights ema`).
  - Checkpoint SHA-256 `893523710186cb16854dd62a258331644e2284abbee8b2cb54e8ab9d32dbea58`. The NAS copy at
    `/Volumes/NetBackup/HifzGuide/checkpoints/consonant2-candidate-issue228` is the source of truth;
    cuda-dev scratch is not.
  - Trained with a fresh optimiser and schedule, by frame-weighted KL to the clean base teacher
    `obadx/muaalem-model-v3_2` @ `01a1ef9fbe40d144ef845101e89ff924aed3fef5` (pinned).
  - #94's loss unchanged: temperature 2.0, nonblank weight 3.0, confirm weight 2.0, logit weight 1.0;
    feature, CTC and hard-label weights 0. **No supervised term, and no label of any kind.**
  - Augmentation `aug94-mixture-v2` verbatim. Preset `h448`, 43 classes, within the `h448` byte and ANE
    budget. No waqf head.
- **Baseline: C95** (`8490b63d…cd6`, step 10,000, EMA).
- **Context columns (reported only):**
  - the teacher;
  - the round-2 EMA, on look gating and the non-tlog sets only, whose decodes are committed. It is **not
    decoded on confirm.3 or confirm.4**.
- **No control arm.** Disclosed: with no control, candidate − C95 cannot separate "more tlog data" from
  "10,000 more steps". Round 2's control measured that separation once.
- **Weights:** live or EMA of the round-3 run, chosen on **tlog look gating** by #95's rule:
  - fewer all-cause false flags per 1,000 graded words at the shipped default;
  - a tie goes to higher word agreement with the teacher, then to live.

  The choice is committed before anything below it.

## Recipe

| item | value |
|---|---|
| data | `tlog.training` (S8 row-set use, 426,825 rows) **from row 73,991 of the frozen schedule's interleave walk**, and Tadabur from `consonant3.training` |
| tlog continuation | `tlog/training_schedule.json` (`8669ceca…134d`), K = 8 interleave, unchanged. The reader skips exactly round 2's consumed keys, committed as `tools/tadabur/tlog/consumed/consonant2.jsonl.gz` (#250): SHA-256 **`5e95c3c06f498768323781fc04b4394093233c2a7717ae193f80feda87929238`**, 73,990 unique keys (49,784 clean, 24,206 unclean, 130 shard files). Source of truth: the NAS copy `/Volumes/NetBackup/HifzGuide/round2-consumed/tlog_consumed.jsonl.gz`, copied from `root@cuda-dev:/root/scratch/issue-228/runs/consonant2_candidate/` on 2026-11-01. The freeze asserts that this key set **equals walk rows 1…73,990**, so "skip the keys" and "start at row 73,991" are the same rule. The reader refuses any of those keys |
| Tadabur frame | `consonant3.training`, a shard use with membership `exact`: the 211-shard frame of `consonant2.training` minus the first 17 shards of its seed-2026 permutation. Round 2 completed 16 and stopped inside shard 378, which is the 17th and is excluded. 194 shards, derived with `training.consumed_prefix` from committed facts |
| mix | exactly 24 tlog + 8 Tadabur windows per batch (3 : 1) |
| schedule | batch 32, lr 5e-5, warmup 500, WSD cooldown 3,000, EMA 0.999, bf16, **10,000 steps**, **seed 2030**; the final step is the candidate |
| attempts | one. An operational stop before step 10,000 restarts from scratch with the same seed, schedule and offset; the run directory is removed first. Only a completed run is decoded |
| monitoring | the last shard of the seed-2030 permutation of the 194 (`consumed_prefix.monitoring_shard`; `assert_monitor`). Round 2's monitor shard 72 is inside the 194 and becomes training audio. Round 2's monitor values (blank-collapse margin −0.020; teacher/student blank ratio 0.690 / 0.669) are printed in the run note as **an indicative line on a different shard**. Not a bar |
| launch check | `check_row_set_disjoint` for `tlog.training`; `training-stream --check` for `consonant3.training`; the warm start's SHA-256; the freeze record committed in HEAD with the power rule satisfied (#252) |
| run log | consumed tlog row keys, committed after the run as `tools/tadabur/tlog/consumed/consonant3.jsonl.gz`; windows and clips per split; real-audio seconds |
| budget | ≤ 6 h wall (BUDGET stop); estimate ~3–4 GPU-h (round 2: 2.81 h at 1.00 steps/s) |

**Coverage.** The planned prefix is a committed fact: `tools/tadabur/tlog/training_prefix.round3.json`
(SHA-256 `913a573aa0e7a4743cc508fb3e22900e5d18ba75344bc6d9e47e2787a09e542c`), written by
`tlog_schedule plan --after tools/tadabur/tlog/consumed/consonant2.jsonl.gz --windows 240000` in the
freeze commit. It records:
- rows, clips, windows and real seconds per split;
- shards;
- clean surahs covered;
- the cumulative pool share with round 2.

The freeze record quotes these numbers. The exact count the run consumes comes from the run log. As in
round 2, this is a seeded shard-cluster sample, not row-uniform sampling.

No row is selected, weighted or filtered by ayah, word, decode, or agreement with any model (#173). The
pool's ~3% non-recitation share is disclosed (gating amendment §3/§4).

**Smoke.** A 700-step smoke on the mixed stream may run before the freeze commit. Its gate is round 2's:
- throughput ≥ 0.6 steps/s;
- finite KL;
- a non-empty validation record;
- the averaged warm start proven;
- both tlog splits observed;
- every batch 24 + 8.

It adds two checks: the first tlog key read is walk row 73,991, and no round-2 consumed key appears in the
smoke's consumed log. Its checkpoint is discarded, nothing is decoded from it, and it does not count as
the one attempt.

## Grading and statistics (unchanged)

- Muraja engine at `99c326f` through the #95 harness (binary `cfcb99ba…`, protocol 3), `quran.db`
  `02fe3dd8…`, shipped default. Newer Muraja is reported only.
- Clustered percentile bootstrap, B = 10,000, seed 20261008, one shared draw per set (paired).
  - tlog clusters = surah.
  - confirm.3 and confirm.4 are pooled into one set before resampling.
- Causes from `training.word_flags`, as round 2.
  - Consonant-caused = `other_consonant`, `target_pair`, `soft_pair`, excluding `ا→∅`.
  - `shaddah` is its own cause and counts in all-cause.
- A tlog gating item's reference is the filename ayah's full text, words 1…N. The span-trimmed reading is
  a sensitivity line only.
- Every model is decoded in streaming `checks` mode with shared features.

**Ungradable rows (as round 2; rule pre-registered here, #248).** quran-transcript 0.5.2's phonetizer
cannot realise 8 ayat: 55:17, 90:8, 90:9, 90:10, 106:1, 106:2, 106:3, 106:4 (`FALLBACK_PHONEMES` in
`tools/generate_phonemes.py`). A gating row is left out of every model alike when:
- it names no ayah, or
- it names one of these 8 ayat.

Each left-out row is recorded by key, ayah and reason. **Any row left out for another reason refuses the
sitting** (`require_known_exclusions`). On look gating, the left-out set must equal round 2's 12 rows
(`consonant2_readout/freeze.json`). Surah 106 has no graded row, so look gating has 112 graded surah
clusters of its 113. The paired comparisons are unbiased for the graded population. They say nothing
about those 8 ayat, which the app does grade.

## Frozen sets

| set | definition | used for |
|---|---|---|
| tlog look gating | `tranches/look.jsonl`, `stratum == gating` (2,853 rows, 113 surahs) | live/EMA; G5 list; cause split; **catch-loss audit**; power rule |
| **tlog confirm.3 + confirm.4 gating** | `tranches/confirm.{3,4}.jsonl`, `stratum == gating` (2,804 + 2,866 = 5,670 rows; 114 surahs each) | bars 1–4, G4, G5; read **once**, pooled |
| tlog reported strata | the claimed tranches' `reported:*` rows, and `gating ∧ id_floor_ok` | reported only |
| qdat, QUD other half, Tadabur `decode_evalset` dev, qdat_bench, Ship A held-out verified sites (8 sites, `g6_sites.json`) | as round 2 | bars 4/5, G1, G2, G3, G6 |
| tlog confirm.1 + confirm.2 | spent by round 2 | **never read by this round** |
| Tarteel v1 confirm, RetaSy `in_correct` | as round 2 | reported / descriptive only |

## Execution order

1. **Freeze** (#253; this document, then the round-3 freeze record). It commits:
   - the G5 list under round 2's unchanged rule: top 10 C95 substitutions on look gating ∪ {`د→ت`}. C95
     and look gating are unchanged, so **the list must reproduce round 2's exactly**; a difference stops
     the freeze;
   - the look cause split for C95, the round-2 EMA and the teacher;
   - the look exclusions, equal to round 2's 12;
   - the **power-rule table** (below), with every guard at ≥ 0.85 at its effective margin;
   - the round-2 consumed-keys file, its walk-prefix check, and the round-3 prefix plan.

   No tranche is claimed.
2. Train the round-3 candidate (one attempt; #254).
3. **P2:** decode the round-3 live and EMA weights on look gating (#256). Commit the live/EMA choice and the
   identity.
4. **Pre-claim gates, on the chosen weights.** Any failure ends the round: no tranche is claimed, and there
   is no retraining, checkpoint substitution or fallback to the round-2 model.
   - the **catch-loss audit** on new removals (below);
   - the non-tlog gates, read once: G3 (consonant, haraka), G6, bars 4a.qdat, 4b.qdat and 5, G1, G2.
5. **Claim** confirm.3 (commit `confirm.3.claim.json`), then confirm.4 (#255). `require_claim_order` needs
   confirm.3's claim in HEAD. Both may share one squash commit, as #247 allowed for round 2.
6. **P3:** decode the teacher, C95 and the chosen candidate on confirm.3+4 gating (3 models).
7. **One pooled verdict:** bars 1, 2, 3, 4a.tlog, 4b.tlog, G4, G5. It is final.
   - Before any confirm decode is opened, the verdict command runs the **reproduction check** in code:
     re-grade C95 on Tarteel v1 confirm (1,971 / 41,572) and qdat (670 / 11,502), and record every input
     file's SHA-256 (#249).
   - A mismatch stops the sitting before the read. The claims stay committed.

After this round **no sealed tlog confirm tranche remains**. Any later tlog verdict (Ship B) needs the
tranche-replenishment amendment to `docs/tlog-holdout-preregistration.md` (#244 decision 8, drafted in
parallel; #258).

## The bars (all must pass)

**Main: tlog confirm.3+4 gating, shipped default** — unchanged

| # | condition | bar |
|---|---|---|
| 1 | all-cause false flags per 1,000 graded words, candidate − C95 | paired 95% UB **< 0** |
| 2 | the same difference as a point reduction | **≥ 4.7 per 1,000** |
| 3 | consonant-caused false flags per 1,000, candidate − C95 | paired 95% UB **< 0** |

Printed beside bar 2, reported only:
- 10% of C95's rate;
- the gap closed;
- flagged words the teacher also flags.

"False flags" on tlog means flags. A lower total means fewer false flags only together with the catch-loss
audit.

**Catch-loss audit (pre-claim gate; #257)**

| item | definition |
|---|---|
| population | look gating words graded by C95, the round-2 EMA and the chosen round-3 weights at the shipped default. **New removals:** C95 flags the word, the round-2 EMA flags it, and the round-3 model does not. **New additions (reported):** the round-3 model flags it and neither C95 nor the round-2 EMA does. Round 2's audited removals (C95 flags, round 2 does not) are outside both strata by definition **(owner 2026-10-31)** |
| sample | seeded simple random sample, seed `round3-catch-loss-2026-11` **(architect, owner-approved 2026-11-01)**, drawn and committed before any listening: **100 new removals** (gating) and **50 new additions** (reported). A smaller stratum is taken whole. Sampling probabilities are recorded |
| listening | the owner, blind, on the #61 listening UI on cuda-dev; the same single question; strata shuffled together; no model output, stratum or tally shown |
| bar | among new removals with a decisive verdict, the share heard as a real mistake **≤ 20%** (point estimate; Wilson 95% reported). **Cannot certify** if fewer than 80 are decisive (unchanged) |
| reported | the additions' real-mistake share; both strata by cause; the owner's noisier pairs (س/ص/ث) flagged |
| data | worklist, verdicts and exact provenance committed; no tlog audio leaves cuda-dev beyond playback |

Disclosed: on look gating, C95 flags 3,505 words and the round-2 EMA flags 2,623. **1,624 words are
flagged by both** (1,881 by C95 only, 999 by the round-2 EMA only). The new-removal population is drawn
from those 1,624, so a stratum under 100 is unlikely. If it happens, all of it is taken and the
80-decisive floor still applies.

**Mechanism** — unchanged

| # | condition | bar |
|---|---|---|
| 4a | word agreement with the teacher, tlog confirm.3+4 and qdat (each), candidate − C95 | paired 95% LB **> 0** |
| 4b | the same on teacher-departure words (each) | paired 95% LB **≥ −0.5 pt** |
| 5 | qdat all-cause false flags per 1,000 | candidate point **<** C95's |

**Guards**

| # | condition | bar |
|---|---|---|
| G1 | QUD other half, all-cause FF/1k, candidate − C95 | paired 95% UB ≤ +1.0 |
| G2 | Tadabur dev, decode agreement with the teacher at `stream_b0` | paired 95% LB ≥ −1.0 pt |
| G3 | qdat_bench stream arm: consonant and haraka agreement with the experts (each) | paired 95% LB ≥ −2.0 pt |
| G3s | qdat_bench shaddah agreement | reported only, with its interval |
| **G4** | tlog confirm.3+4, tashkeel-caused FF/1k, candidate − C95 | **paired 95% UB ≤ +3.5 (amendment A1, owner 2026-11-01)** |
| G5 | each substitution on the frozen list, flagged words per 1,000 on confirm.3+4 | no substitution's paired 99.5% LB > 0 |
| G6 | Ship A held-out verified real mistakes at ط↔ق and س↔ش, words flagged | candidate ≥ C95 − 2 sites |

**What G4 at +3.5 does and does not protect (required disclosure).** +3.5 is 28% of C95's tlog tashkeel
rate (12.46 per 1,000 on look gating). At the projected paired SE of 1.07:
- a candidate with **no true change passes about 91%** of the time;
- a candidate with a **true +1.0 regression passes about two thirds of the time** (0.65);
- a candidate with a true +2.0 regression passes about 29% of the time.

**This guard vetoes gross tashkeel regressions only.** The real tashkeel protection is Ship B's powered
guard against C95, which Ship B must carry.

Shaddah stays reported, not gating, as round 2 decided. Ship B must carry a powered shaddah guard.

## The mechanical power rule (owner 2026-10-31; SE rule: principal round-3 r1; #251)

Before training, the freeze computes, for each guard with a margin, the probability that a candidate with
**no true change** against C95 passes:

- G1, G2, G3 (consonant, haraka) and G4, which use 95% bounds: `P(pass | true 0) = Φ(margin / SE − 1.96)`.
- G6: the exact sign-test probability that the candidate − C95 site difference is ≥ −2, given the
  discordant-site count `d`: `P(Bin(d, ½) ≥ ⌈(d − 2)/2⌉)`. This is not monotone in `d` (d = 5 → 0.81,
  d = 7 → 0.77, d = 8 → 0.86), so it is computed, never thresholded on `d`.

**SE sources.** The round-2 EMA − C95 stands in for the round-3 model; these are widths only.

- **G4 (tlog):** `SE_G4 = SE_look(round-2 EMA − C95) × ρ`.
  - `SE_look` is the one-tranche look-gating paired bootstrap SE, **unscaled**.
  - `ρ` = round 2's realised pooled G4 SE ÷ round 2's freeze stand-in SE. That is (1.527 + 1.461)/3.92
    from `consonant2_readout/report.json`, divided by `power_check.tashkeel.se` in
    `consonant2_readout/freeze.json`: 0.762 / 0.764 = 1.00.
  - Why unscaled: in round 2, pooling two tranches shrank the SE by 0.72 (≈ √½). The further divergence
    of a newly trained model from C95 widened it by 1.39. The product was 1.00. So the stand-in's
    one-tranche look SE predicted the realised pooled SE to three figures.
  - Reported beside it, each labelled "assumes no further divergence from C95":
    - the √(2,853 / 5,670)-scaled figure (≈ 0.755);
    - round 2's realised pooled SE (0.762).
  - Expected value ≈ 1.07; the freeze recomputes it with `carries_cause` at B = 10,000.
- **G1, G2, G3:** the paired bootstrap SE on the same frozen sets the verdict reads, from round 2's
  committed decodes. No projection factor applies.
- **G6:** `d` from the committed truth-baseline check caches
  (`tools/tadabur/truth_baseline/checks/{r2_candidate_ema,c95}.json`), through
  `training.consonant_readout.catch_arm` with the round-2 EMA identity. Ship A has 8 sites; on round 2's
  caches d = 1. The same stand-in caveat applies as for G4.
- **G5** is reported, not tested. At a margin of 0, its no-change pass rate is fixed by the 99.5% level
  (~0.25% false veto per substitution), whatever the SE. A frozen substitution with no event in either
  model on look gating is listed as a cannot-certify risk.

**Rule.** If any guard's `P(pass | true 0)` at its effective margin is under 0.85, launch is refused. The
freeze refuses to write its record, and the launcher refuses without the record. Changing a margin needs
a dated pre-training amendment with the owner's approval and a principal cross-check.

**Amendment A1 (2026-11-01, pre-training; owner-approved; principal round-3 r1).**
- The owner registered G4 at +2.5 on 2026-10-31.
- Under the SE rule above, `SE_G4` ≈ 1.07 gives P(pass | true 0) ≈ **0.65** at +2.5. The rule fires.
- The principal's margin table at SE 1.07:
  - +3.0 → 0.80;
  - +3.2 → 0.85 (the minimum that satisfies the rule);
  - **+3.5 → 0.91**.
- The owner chose **+3.5** on 2026-11-01.
- This amendment was made before any round-3 training, from round-2 widths only. No round-3 output
  existed or informed it.
- The freeze table prints G4 at both margins: +2.5, marked "registered, rejected by the rule (A1)", and
  +3.5, effective.

Expected table at the freeze (to be recomputed):

| guard | margin | SE | P(pass, true 0) |
|---|---|---|---|
| G1 | +1.0 | ≈ 0.19 | ≈ 1.00 |
| G2 | −1.0 pt | ≈ 0.05 | ≈ 1.00 |
| G3 consonant | −2.0 pt | ≈ 0.45 | 0.99 |
| G3 haraka | −2.0 pt | ≈ 0.49 | 0.98 |
| G4 (registered) | +2.5 | ≈ 1.07 | ≈ 0.65 (rejected, A1) |
| **G4 (effective)** | **+3.5** | ≈ 1.07 | ≈ **0.91** |
| G6 | −2 sites | d = 1 | 1.00 |

The projection is disclosed as a projection. Round 3's realised pooled SE becomes the second observation
of `ρ`, for Ship B.

## Reported, not gating

- the reported strata and `gating ∧ id_floor_ok`;
- each tranche alone;
- partial-span coverage and empty-slot rates;
- every substitution's change, and flags by cause;
- the teacher column;
- the round-2 EMA on look gating and the non-tlog sets (round 3 − round 2 on look gating: the increment
  this round added);
- bar 2's companions;
- G3s and the shaddah lines;
- the catch-loss additions;
- the excluded rows by reason;
- Tarteel v1 confirm (round 1's table);
- per-reciter dose-response on Tadabur dev;
- newer Muraja;
- RetaSy.

## Decision rule

- **Pass:** the pre-claim gates (catch-loss audit, G3, G6, 4a.qdat, 4b.qdat, 5, G1, G2) and the pooled
  verdict (1, 2, 3, 4a.tlog, 4b.tlog, G4, G5) all hold. The round-3 candidate becomes the release
  candidate and Ship B's start. Publishing is the owner's call, with every reported line in view.
- **Fail:** any gate fails.
  - C95 stays, and the round-2 verdict stays FAIL.
  - A pre-claim fail leaves confirm.3 and confirm.4 sealed and unclaimed.
  - A release exception for any model remains the owner's separate call. It does not change any verdict
    in the record.
- **Cannot certify:** a missing input, an empty cell, an undefined interval, or too few decisive audit
  verdicts. Resolved only by a new dated pre-registration, never by a re-run.
- **Reproduction check (not a bar, #249):** see execution step 7. A mismatch stops the sitting before the
  confirm read. It is not a verdict.

## Data discipline (round 2's, plus three lines)

1. No grading of a round-3 decode before the freeze.
2. **Sealed tranches.** No student decodes a confirm.3 or confirm.4 row before step 5's claims. Nobody
   listens to a confirm tranche. This round never reads confirm.1/2.
3. Gating is frozen at the S7 manifests (`e11fbf5`).
4. **Shared-text rule.** Look readings fix what is measured: the G5 list, the cause split, the live/EMA
   choice, the audit sample and the power-rule SEs. They never select, weight or filter training rows.
5. One read of the claimed tranches, and of the guard sets for the final candidate. The guard sets'
   earlier reads (rounds 1 and 2) are disclosed.
6. **Never train on:**
   - any tlog row with `role ≠ train`, or excluded;
   - round 2's 73,990 consumed rows (by this round's recipe);
   - Tarteel v1, RetaSy, qdat, qdat_bench, QUD, `decode_evalset`, the sealed panel, Ship A held-out sites,
     or any truth site.
7. No tlog audio is copied off cuda-dev.
8. **New:** the round-2 model is not decoded on confirm.3 or confirm.4, under any outcome.
9. **New:** no gating row is left out except by the "Ungradable rows" rule (#248).
10. **New:** the reproduction check runs before the confirm read, in code (#249).

## GPU and wall-time budget

| step | estimate | basis |
|---|---|---|
| smoke (700 steps) | ~0.25 GPU-h | round 2 |
| candidate | ~3–4 GPU-h | round 2: 2.81 h |
| P2: look gating, live + EMA | 0.5–1 h | #216 |
| non-tlog gates, one model | ~2.2 h | rounds 1 and 2 |
| P3: confirm.3+4 gating, 3 models | ~2.5–3.5 h at P = 4 | #230 (P = 4 after the host-RAM OOM) |
| catch-loss audit | ~1 h of the owner's listening | 150 words |

## Changelog

- Frozen 2026-11-02, owner-approved (2026-11-01: plan "Approve", G4 "+3.5"). Drafted as architect round-3
  r1–r2 after principal round-3 r1.

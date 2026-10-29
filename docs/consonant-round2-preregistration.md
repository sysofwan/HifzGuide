# Consonant round 2 (tlog): pre-registered bars — DRAFT (#77, #173)

**Status: FINAL TEXT, owner-approved on 2026-10-28** ([#173 owner decisions](https://github.com/sysofwan/HifzGuide/issues/173#issuecomment-6488120029);
audit floor: [#225](https://github.com/sysofwan/HifzGuide/issues/225#issuecomment-6488136087)), after the
Fable/Astra stress-test (principal r9/r10, Astra r1/r2; architect r29). The freeze issue (#229) commits this text as
`docs/consonant-round2-preregistration.md`. It is frozen at that commit, before any round-2 **candidate**
training, and never edited after that. A change needs a new dated pre-registration, never a re-run. Numbers
marked **(owner 2026-10-28)** were set by the owner for this round. Unmarked numbers are round 1's,
unchanged, as the owner directed ("the bars are unchanged", #173, 2026-10-24). Two placeholders are filled
in the freeze commit: the SHA-256 of `tools/tadabur/tlog/training_schedule.json` (#224), and the round-2
issue numbers.

Parents (cite by SHA-256 when committing): `docs/consonant-round-preregistration.md` (round 1, #174),
`docs/tlog-holdout-preregistration.md` (`204fe7f9…`), `docs/tlog-gating-amendment.md` (#210), the S7
tranche manifests (PR #211, `e11fbf5`; `look.jsonl` `7d7df748…`), #212's look decodes (PR #215;
`consonant2_readout/decodes/meta.json`), `docs/tlog-shared-check-decode.md` (#216), and the VRAM check
(#217).

**Statement carried by every verdict of this round:** *clip-level split, same users possible on both
sides* (holdout pre-registration item 8). qdat is an external-corpus check: the frozen exclusions test
recordings, not voice identity, so it is not proof of speaker-disjointness from anonymous tlog users
(Astra r1 §5).

## What changed from round 1, and why

| item | round 1 (#174) | round 2 | why |
|---|---|---|---|
| main set | Tarteel v1 confirm half | **tlog confirm.1 + confirm.2 gating rows, one pooled verdict (owner 2026-10-28)** | Tarteel v1 confirm is spent (two verdicts; the warm start was chosen after #179) |
| when tranches are claimed | before training | **only after the non-tlog gates and the catch-loss audit pass (owner 2026-10-28)** | Saves both tranches when a known risk recurs (Astra r1 §4, principal r10) |
| bar 2 | ≥ 4.7 per 1,000 | **≥ 4.7 per 1,000 on the claimed set (owner 2026-10-28)** plus the catch-loss audit | Same number; the audit checks that the drop is in false flags, not lost catches |
| catch-loss audit | — | **new gate: blind owner listening on look gating (owner 2026-10-28)** | Fewer total flags ≠ fewer false flags (Astra r1 §2, principal r10) |
| G3 shaddah | gating | **reported only (owner 2026-10-28)** | The owner ranks consonants above shaddah; decided before the round-2 candidate exists |
| G5 list | top 10 C95 substitutions, Tarteel look | top 10 C95 substitutions on tlog look gating ∪ {`د→ت`} | The owner kept `د→ت` (2026-10-24) |
| live/EMA, G5 list, cause split | Tarteel v1 look | tlog look gating | The same population as the verdict |
| warm start | C95 EMA | #178 candidate EMA `057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2` | Owner, 2026-10-24 |
| training data | Tadabur, 278 unconsumed shards | **`tlog.training` : Tadabur = 3 : 1, 10,000 steps (owner 2026-10-28)** | tlog is the new lever |

## What is compared

- **Candidate:** the #178 candidate's averaged weights (`--init-from … --init-weights ema`), trained with a
  fresh optimiser and schedule by frame-weighted KL to the clean base teacher
  `obadx/muaalem-model-v3_2` @ `01a1ef9fbe40d144ef845101e89ff924aed3fef5` (pinned).
  - #94's loss is unchanged: temperature 2.0, nonblank weight 3.0, confirm weight 2.0, logit weight 1.0;
    feature, CTC and hard-label weights 0. **No supervised term, and no label of any kind.**
  - Augmentation `aug94-mixture-v2` verbatim. "Clean teacher input" means unaugmented, not denoised: tlog
    audio is app audio as recorded.
  - Preset `h448`, 43 classes, within the `h448` byte and ANE budget. No waqf head.
- **Baseline: C95** (`8490b63d…cd6`, step 10,000, EMA). The teacher and the #178 candidate are context
  columns. `h448` is not decoded on tlog.
- **Control (#223, owner-approved, reported only, never in the verdict):** the same warm start and schedule, Tadabur only,
  from the frozen 211-shard frame. It finished on 2026-10-29 08:13:25Z: 10,000 steps, 3.11 h, 0.90 steps/s,
  exit 0; checkpoint SHA-256 `da5fb5bf02c7b078efbfb781855c44f171bc24a2d709cd71eefc7d2443c20473` (NAS copy
  verified). Its exposure use is `consonant2.control.training`, 211 shards, `index_sha256
  a20ad072beca3b339824db321b87fda0d8b3e32a338f3aeca9cd05643aea493a`; its frame manifest records
  `shards_sha256 4bd00c2ba5d0979bcc017546e3beeea70807dad443b125fa293adfe7f5f5d7d8`,
  `stream_order_sha256 7f0f1533618d0fc4e32e1fa34e20c4d1d22f4995779ad816a58e9d54193674fe`,
  `shard_order_seed` 2026, `window_shuffle_seed` 2027 and the recipe record. Monitor shard 72 never
  entered. That manifest is the contract the candidate's Tadabur substream replays: the shard list, its
  seed-2026 stream order, both seeds and the recipe record. Disclosed:
  - candidate − control is n = 1 against n = 1, and no seed replicate exists;
  - "same seed" means only that no seed was chosen by outcome. The window order diverges once tlog is
    interleaved; augmentation draws are keyed by (seed, shard, row), so shared Tadabur rows keep theirs;
  - the control was launched before this freeze, under the 10k schedule the owner then approved, as the
    owner-approved freeze exception recorded on #223.
  The same look-only live/EMA rule applies to both arms. Both identities are committed before any
  confirm decode.
- **Weights:** live or EMA, chosen on **tlog look gating** by #95's rule: fewer all-cause false flags per
  1,000 graded words at the shipped default; a tie goes to higher word agreement with the teacher, then
  to live. The choice is committed before anything below it.

## Recipe (owner 2026-10-28)

| item | value |
|---|---|
| data | `tlog.training` (S8 row-set use, **426,825 rows**, clean and unclean) **and** Tadabur from the frozen 211-shard frame (#223's manifest) |
| mix | **exactly 24 tlog + 8 Tadabur windows per batch** (3 : 1) |
| tlog order | a **frozen, seeded, metadata-only shard permutation across both splits** (seed `round2-tlog-shards-2026-10`; `tlog/training_schedule.json`, SHA-256 `8669ceca5ec67215a5c0147e97c0f74c97e5b20d29c8b1446cd8212feb8e134d`, committed in the freeze, #224), read **K = 8 shards at a time, interleaved**, then the existing shuffle buffer. Clean shards are surah-sorted (median 1 surah per shard on look), so one shard at a time would feed ~125 steps of one surah. Required because the S8 reader sorts by `(split, shard_file, row)` (`tlog_pool.py:580,594`); unfixed, a 10k run would read 79,323 clean rows from shards 0–121, 34 of 114 surahs and no unclean row (principal r10). **Candidate training is blocked until this lands** (#224) |
| schedule | as #178: batch 32, lr 5e-5, warmup 500, WSD cooldown 3,000, EMA 0.999, bf16, **10,000 steps**, seed **2026**; the final step is the candidate |
| attempts | one. An operational stop before step 10,000 (Hub 429s, BUDGET) restarts from scratch with the same seed and schedule. Only a completed run is ever decoded (principal r9 Q3) |
| monitoring | the frame's monitor shard, as #178 (`assert_monitor`); blank-collapse margin and teacher/student blank ratio, with #178's values (−0.021; 0.636 / 0.615) as the reference line in the run note. Not a bar |
| launch check | `check_row_set_disjoint` for `tlog.training` (`check_disjoint` refuses it by design, `exposure.py:683-689`) and `training-stream --check` for the Tadabur use |
| run log | consumed tlog row keys, windows and clips per split, real-audio seconds |
| budget | ≤ 6 h wall (BUDGET stop); estimate ~3.5–4.5 GPU-h |

**Coverage.** The planned prefix is the frozen fact: `tools/tadabur/tlog/training_prefix.json` (committed in
`4625b75`, #224) plans **73,829 rows (17.3% of the pool's 426,825), 240,002 windows, 130 shards** —
clean 49,664 rows / 76 shards / 155,750 windows, unclean 24,165 rows (32.7%) / 54 shards / 84,252
windows. Clean rows cover 57 of 114 surahs. This measures the owner's "about 72k, ~17%", which was the
pre-#224 estimate. The exact count the run consumes comes from the run log.

This is a seeded shard-cluster sample, not row-uniform sampling and not all-surah coverage at every
prefix. The 57 absent clean surahs hold 28.4% of clean pool rows; unclean rows carry no label, so 57 is
not a measurement of total surah coverage. K = 16 / 24 / 32 reach only 57 / 58 / 60 clean surahs. A
sampling study is a follow-up for a future round, not a change to this frozen recipe.

No row is selected, weighted or filtered by
ayah, word, decode, or agreement with any model (owner rule, #173). The pool's non-recitation share, about
3%, is disclosed (gating amendment §3/§4).

**Smoke.** A smoke of 700 steps on the mixed stream may run before the freeze commit (#228). Its gate:
throughput ≥ 0.6 steps/s, finite KL, at least one non-empty validation record from the frame's monitor
clips, the averaged warm start proven, both tlog splits observed, every batch asserted 24 + 8
(`check_batch_composition`). Its checkpoint is discarded, nothing is decoded from it, and it does not
count as the one attempt.

## Grading and statistics (unchanged from round 1)

- Muraja engine at `99c326f` through the #95 harness (binary `cfcb99ba…`, protocol 3), `quran.db`
  `02fe3dd8…`, at the shipped default. Newer Muraja is reported only.
- Clustered percentile bootstrap, B = 10,000, seed 20261008, one shared draw per set (paired). **tlog
  clusters = surah**; confirm.1 and confirm.2 are pooled into one set before resampling.
- Causes from `training.word_flags`. Consonant-caused = `other_consonant`, `target_pair`, `soft_pair`,
  excluding `ا→∅`. `shaddah` (a collapsed geminate) is its own cause and counts in all-cause.
- A tlog gating item's reference is the **filename ayah's full text** (holdout pre-registration, "Evaluation population"). #212's committed look records carry S6's identified word span as `word_from`/`word_to` on 515 labelled rows (`tlog_eval_decode.merge_decodes`). **The readout (#227) ignores those fields and grades words 1…N of the filename ayah**; the span-trimmed reading is reported as a sensitivity line only. Every model is decoded in streaming
  `checks` mode, P=6 × 1 thread with shared features (#216), and bar 4's teacher text is the teacher's
  `checks`.

## Frozen sets

| set | definition | used for |
|---|---|---|
| tlog look gating | `tranches/look.jsonl`, `stratum == gating` (2,853 rows); C95, teacher and #178 decoded by #212 | live/EMA; G5 list; cause split; **catch-loss audit**; diagnostics |
| **tlog confirm.1 + confirm.2 gating** | `tranches/confirm.{1,2}.jsonl`, `stratum == gating` (5,762 rows) | bars 1–4, G4, G5; read **once**, pooled |
| tlog reported strata | the claimed tranches' `reported:*` rows, and `gating ∧ id_floor_ok` | reported only |
| qdat | as round 1 (1,505 items) | bars 4, 5 |
| QUD other half | as round 1 | G1 |
| Tadabur `decode_evalset` dev | as round 1 (671 clips) | G2 |
| qdat_bench | as round 1 (159 clips) | G3 |
| Ship A held-out verified sites | as round 1 (`g6_sites.json`) | G6 |
| Tarteel v1 look + confirm | as round 1 | reported only (spent) |
| RetaSy `in_correct` | as round 1 | descriptive only |

## Execution order (owner 2026-10-28)

1. **Freeze** (#229; this document; the G5 list, C95's cause split, the `د→ت` look reading and the
   one-tranche SE check, all graded from #212's look decodes). No tranche is claimed.
2. Train the candidate (and the control, #223, already launched).
3. **P2:** decode both arms' live and EMA weights on look gating. Commit the live/EMA choice and both
   identities.
4. **Pre-claim gates, on the chosen candidate.** This is execution order only: the acceptance rule is unchanged. Any failure ends the round: **no tranche is claimed**,
   and there is no retraining, checkpoint substitution or switch to the control.
   - the **catch-loss audit** (below, #225). It reads look only, so it is run before the claims, like the non-tlog gates the owner listed;
   - the non-tlog gates, read once: G3 (consonant, haraka), G6, bars 4a.qdat, 4b.qdat and 5, G1, G2.
5. **Claim** confirm.1 (commit `confirm.1.claim.json`), then confirm.2 (a second commit:
   `require_claim_order` needs claim 1 in HEAD, `tlog_tranches.py:233-269`).
6. **P3:** decode the teacher, C95, the chosen candidate and the chosen control on confirm.1+2 gating
   (4 models × 6 processes; #217 measured a 12,419 MiB peak of 16,311).
7. **One pooled verdict:** bars 1, 2, 3, 4a.tlog, 4b.tlog, G4, G5. It is final.

**confirm.3 and confirm.4 cannot be claimed by this round under any outcome.** A one-tranche read with a
conditional second (principal r9) and a futility-only design (Astra r2) were considered and not adopted:
the owner chose fixed two after the pre-claim gates.

## The bars (all must pass)

**Main: tlog confirm.1+2 gating, shipped default**

| # | condition | bar |
|---|---|---|
| 1 | all-cause false flags per 1,000 graded words, candidate − C95 | paired 95% upper bound **< 0** |
| 2 | the same difference as a point reduction | **≥ 4.7 per 1,000 (owner 2026-10-28)** |
| 3 | consonant-caused false flags per 1,000, candidate − C95 | paired 95% upper bound **< 0** |

Printed beside bar 2, reported only: 10% of C95's rate; the gap closed, (C95 − candidate) / (C95 −
teacher); and flagged words the teacher also flags, candidate vs C95. "False flags" on tlog means
**flags**: tlog gating keeps real mistakes, so a lower total means fewer false flags only together with
the catch-loss audit.

**Catch-loss audit (pre-claim gate, owner 2026-10-28)**

| item | definition |
|---|---|
| population | look gating words graded by both C95 and the chosen candidate at the shipped default. **Removed:** C95 flags the word and the candidate does not. **Added:** the candidate flags it and C95 does not |
| sample | seeded simple random sample, seed `round2-catch-loss-2026-10`, drawn and committed **before** any listening: **100 removed** (gating) and **50 added** (reported). If a stratum is smaller, all of it is taken. Sampling probabilities are recorded |
| listening | the owner, blind, on the #61 listening UI on cuda-dev. One "was the highlighted word recited correctly: yes / no / unclear" question; strata shuffled together; no model output, stratum or tally shown |
| bar | among removed words with a decisive verdict, the share heard as a **real mistake ("no") ≤ 20%** (point estimate; Wilson 95% reported). **Cannot certify** if fewer than 80 of the 100 are decisive (owner 2026-10-28, #225) |
| reported | the added stratum's real-mistake share; both strata by cause; the owner's noisier pairs (س/ص/ث) flagged |
| data | the worklist, the verdicts and exact provenance (split, shard, row, `audio_sha256`, word index) are committed. No tlog audio leaves cuda-dev beyond playback in the owner's browser |

**Mechanism**

| # | condition | bar |
|---|---|---|
| 4a | word agreement with the teacher, tlog confirm and qdat (each), candidate − C95 | paired 95% lower bound **> 0** |
| 4b | the same on teacher-departure words, tlog confirm and qdat (each) | paired 95% lower bound **≥ −0.5 pt** |

**Non-tlog-voice consistency (the fallback's qdat consistency bar)**

| # | condition | bar |
|---|---|---|
| 5 | qdat all-cause false flags per 1,000 | candidate point estimate **<** C95's |

Bars 4a.qdat, 4b.qdat and 5 are the qdat consistency bar holdout item 8 requires. G1 and G2 are the
no-harm half.

**Guards**

| # | condition | bar |
|---|---|---|
| G1 | QUD other half, all-cause FF/1k, candidate − C95 | paired 95% UB **≤ +1.0** |
| G2 | Tadabur dev, decode agreement with the teacher at `stream_b0` | paired 95% LB **≥ −1.0 pt** |
| G3 | qdat_bench stream arm: **consonant and haraka** agreement with the experts (each) | paired 95% LB **≥ −2.0 pt** |
| G3s | qdat_bench **shaddah** agreement | **reported only (owner 2026-10-28)**, with its interval |
| G4 | tlog confirm.1+2, tashkeel-caused FF/1k | paired 95% UB **≤ +1.0** |
| G5 | each substitution on the frozen list (top 10 by C95 false-flagged words on look gating, plus `د→ت` if not already in), flagged words per 1,000 on confirm.1+2 | no substitution's paired **99.5%** LB **> 0** |
| G6 | Ship A held-out verified real mistakes at ط↔ق and س↔ش, words flagged | candidate **≥ C95 − 2** sites |

**Shaddah (owner 2026-10-28).** G3's shaddah part is **reported, not gating**,
by the owner's stated priority: consonants above shaddah. It is decided before the round-2 candidate
exists, so no round-2 outcome informs it. It is informed by #178 (338 vs C95 341 of 365; LB −2.24). At a
paired SE of 0.71 pt, a candidate equal to C95 fails it ~20% of the time, and one at #178's level ~62%
(principal r9 Q5). Consequences:
- **Still protected:** shaddah flags the app *shows* (cause `shaddah`, a collapsed geminate) are inside
  the all-cause bars 1–2.
- **No longer gated:** the catch side, hearing a geminate the reciter actually held or dropped on
  expert-labelled audio.
- **Reported:** G3s with its interval; shaddah-caused FF/1k on confirm.1+2, candidate − C95; shaddah
  agreement with the teacher on confirm.1+2. These lines inform the owner's publish decision but cannot
  pass or fail the round.
- **Carried forward:** Ship B (the tashkeel round, starting from this round's winner) must carry a powered
  shaddah guard against C95.
- **Not revisitable:** no loosening or tightening of shaddah after any round-2 result.

- **`د→ت`, disclosed:** #178 carried 39 vs C95's 29 on Tarteel confirm (99.5% LB +0.02). At the same
  effect size, two tranches give an LB of about +0.05, a G5 fail (principal r9 Q6). It is the likeliest
  non-tlog reason to fail. It is read on look gating at the freeze and reported; that reading changes
  neither the list nor the bar.
- G5 with 11 substitutions at 99.5% keeps the family-wise false-veto rate near 5.4%.

## Reported, not gating

- the reported strata and `gating ∧ id_floor_ok`; each tranche on its own; partial-span coverage and
  empty-slot rates per tranche (515 of 2,853 look gating clips carry a partial span);
- every substitution's change; flags by cause; teacher and #178 columns;
- candidate − control: interpretable only if its paired interval excludes zero by a clear margin; no seed
  replicate exists;
- the bar 2 companions above; G3s and the shaddah lines; the catch-loss audit's added stratum;
- Tarteel v1 confirm (full round-1 table); per-reciter dose-response on Tadabur dev; the catch side as
  round 1; newer Muraja; RetaSy.

## Decision rule

- **Pass:** the pre-claim gates (catch-loss audit, G3, G6, 4a.qdat, 4b.qdat, 5, G1, G2) and the pooled
  verdict (1, 2, 3, 4a.tlog, 4b.tlog, G4, G5) all hold. The candidate becomes the release candidate
  (publishing is the owner's call, with every reported line in view) and Ship B's start.
- **Fail:** any gate fails. C95 stays. The architect interprets the result and proposes next steps. A
  pre-claim fail leaves confirm.1 and confirm.2 sealed and unclaimed.
- **Cannot certify:** a missing input, an empty cell, an undefined interval, or too few decisive audit
  verdicts. It is resolved only by a new dated pre-registration, **never by a re-run**.
- **Reproduction check (not a bar):** before any candidate is read, the readout (#227) re-grades C95 on Tarteel v1
  confirm (1,971 / 41,572) and qdat (670 / 11,502), and #178 on Tarteel v1 confirm (1,837 / 41,652).

**Dissent recorded (Astra r1/r2):** a release claim of improved correctness should rest on blinded truth
adjudication of changed app outcomes on the confirm set. The owner's choice is the look-side catch-loss
audit plus bar 2. Confirm-set listening is not pre-registered: holdout item 7 forbids listening to a
confirm tranche before its round, and a post-verdict co-gate was offered (principal r10) and not taken.

## Data discipline

1. **No grading before the freeze.** The first grade of any tlog decode is the freeze step.
2. **Sealed tranches.** No student decodes a confirm row before step 5's claims. confirm.3/4 are never
   touched. Nobody listens to a confirm tranche.
3. **Gating is frozen** at the S7 manifests (`e11fbf5`), never recomputed with a student.
4. **Shared-text rule.** Look readings may fix what is measured (the G5 list, the cause split, the
   live/EMA choice, the audit sample). They never select, weight or filter training rows.
5. **One read** of the claimed tranches and of the guard sets for the final candidate. The guard sets'
   round-1 read is disclosed.
6. **Never train on:** any tlog row with `role ≠ train` or excluded, Tarteel v1, RetaSy, qdat, qdat_bench,
   QUD, `decode_evalset`, the sealed panel, Ship A held-out sites, or any truth site.
7. **No tlog audio is copied off cuda-dev.** On-box caches and in-browser playback for the audit are
   allowed. Verdicts and provenance are committed; audio is not.

## Power (informative)

Look gating: 2,853 clips, 29,424 ayah words, surah 2 = 16.2%, 24.0 effective surah clusters (principal
r8). Projected all-cause SE: 0.83–0.94 on one tranche, 0.59–0.94 on two. Tashkeel SE ~0.31 on one, ~0.22
on two, so P(G4 fails a no-change candidate) is ~10% on one and ~0.4% on two. These are projections, not
measurements: sparse events can cluster by speaker under the clip-level split, and tlog's graded-word
coverage is unmeasured until the freeze grades look. The freeze reports the measured one-tranche SE of
#178 − C95, which decides nothing.

## GPU and wall-time budget

| step | estimate | basis |
|---|---|---|
| P1 (#212) look gating, 3 models | 1.18 h, done | measured |
| P3 VRAM check, 4 models × 6 processes | done (#217): peak 12,419 of 16,311 MiB, P=6 kept | measured |
| control (#223) | ~3.6 GPU-h | #178 measured 3.37 h |
| candidate | ~3.5–4.5 GPU-h | #178 rate; the mixed stream is not yet measured |
| P2: look gating, both arms' live + EMA | 0.8–1.6 h | #216 |
| non-tlog gates: candidate (+ control) | ~2–4.5 h | round 1, ~2.2 h per model at P=6 |
| P3: confirm.1+2 gating, 4 models | 3.2–4.2 h | #216 |
| catch-loss audit | ~1 h of the owner's listening | 150 words |

## Changelog

- Frozen 2026-10-28, owner-approved
  ([#173](https://github.com/sysofwan/HifzGuide/issues/173#issuecomment-6488120029)). Drafted as
  architect r25–r30 after principal r8–r10 and Astra r1–r2.

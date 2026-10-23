# #178's consonant round: continue C95 toward the teacher

**Status (2026-10-23, 17:05Z):** the recipe and data below are fixed and committed **before
launch**; the run has not started. #176 has finished (`logs/baseline.exit` = 0) and
`gpu.lock` is free, so `scripts/issue-178-train.sh launch` is the next step.
Everything marked *pending* is recorded from the run and never chosen: the candidate is the
final step. Checkpoints stay on cuda-dev.

This note is #174's pre-registration executed, under the amendment recorded on #178 on
2026-10-23 (data: the shards the student has *not* trained on are the evaluation frame, so
the run continues on the shards #94's C95 run did not consume). No bar, set or number in
`docs/consonant-round-preregistration.md` changes.

## What is compared

| arm | student input | exposure use |
|---|---|---|
| **candidate** | augmented under `aug94-mixture-v2` | `consonant1.training` |
| **baseline** | C95, `8490b63d…cd6`, step 10,000, weights `ema` | — |
| **control** | none | — |

**No control arm.** The round's only ingredient beyond C95 is more of C95's own recipe, so
"the same run without the new ingredient" *is* C95, which the verdict already compares
against. #174 makes the control optional and never part of the verdict; a second ~3.5 GPU-h
arm would learn nothing. Candidate budget: ~3.6 GPU-h (smoke + 10,000 steps), well inside
the owner's ~8.

**No extra corpus.** `muaalem-annotated-v3` has no reciter mapping to qdat, qdat_bench,
Tarteel v1 or QUD, and a zero-overlap fingerprint against every evaluation set plus staging
does not fit the round's envelope (cuda-dev had 35.6 GB free). #174 makes unseen audio
optional. If bar 2 fails, an unseen corpus is the natural next lever, as its own issue.

## The warm start

- **C95's averaged weights:** `/root/scratch/issue-94/repo-fix/tools/runs/aug94_candidate_v2/checkpoint.pt`,
  SHA-256 `8490b63d0664827d5a2e2e3121eb4af7b22069273db39268c7638cc7d8839cd6`, step 10,000,
  identity `candidate_ema` (`tools/tadabur/consonant_readout/models.json`).
- **Read with `--init-from … --init-weights ema`.** The flag is new here: `--init-from` used to
  read the live weights only, and #174 fixes the candidate's start as the averaged ones. It
  stays out of `TrainConfig` and the resume-critical fields, because it initialises a fresh
  run; the run's log line records `weights=ema`.
- **Preset `h448`, 43 classes** (`training.distill_student.py`): hidden 448, FFN 1792, 7 heads.
  Export size is unchanged from `h448`'s 80.5 MB, inside the ANE budget the preset was sized
  against (`distill_student.py:174-181`; `ml-model-transformation.md`). No export happens in
  this issue.
- **Fresh optimiser and schedule.** `--init-from` is not `--resume`: the LR schedule restarts
  with a 500-step warmup, deliberately, as #94's runs did.

## The objective

#94's frame-weighted KL to the clean base teacher, unchanged:

- **Teacher:** `obadx/muaalem-model-v3_2` @ `01a1ef9`, frozen, bf16, reading **clean** audio.
- **Loss:** temperature 2.0, nonblank weight 3.0, confirm weight 2.0, logit weight 1.0; feature,
  CTC and hard-label weights 0. **No supervised term on human labels** (#174).
- **Health metrics:** KL, pre-clip gradient norm, confirmed agreement, blank-collapse margin.

## Augmentation

`aug94-mixture-v2` verbatim (`training/augmentation.py:RECIPE_VERSION`, recorded in the run's
`run_config.json`): clean 0.30, noise 0.35, reverb 0.15, 8 kHz band limit 0.10, Opus 0.10.
Draws are keyed per clip by `(shard, row)`, so the teacher reads the same windows either way.
The noise pool is `/root/scratch/issue-94/noise_pool.npz` (458 recordings, SHA-256
`d2978d0f…fd43`), checked against `tools/tadabur/augmentation/noise_pool.json` at load.

## Data: `h448`'s 345 shards minus #94's consumed prefix

- **The 345** are `training.decode_evalset.training_shard_spec()` —
  `21-38,40-57,…,382-384`, the only shard set a training run may stream (`held_out_shards()`
  refuses the framed block 0-20 and the strided reserve 39, 58, …, 381).
- **What #94 consumed.** C95's arm streamed the head of the seed-9494 shard permutation. Its
  log (`/root/scratch/issue-94/logs/aug94_candidate_v2.log`) prints one line per shard left;
  the 66 lines are, in order, 169, 354, 207, …, 338, 304. The 67th shard of the same
  permutation, **114**, was entered before the run stopped at step 10,000 (its line would
  come only when the next shard began). So **67 shards are consumed** and **278 remain**.
- **Derivation, reproducible:** `python -m training.consumed_prefix --log <that log>` rebuilds
  the permutation (`random.Random(9494).shuffle` of the 345, exactly what
  `StreamingWindowDataset` does) and prints the remaining `--stream-shards` spec; it refuses a
  log whose order is not the permutation. `training/test_consumed_prefix.py` pins the whole
  chain against the log's order.
- **Rows:** the 278 shards hold **277,450** rows; **2,959 (1.07%)** are excluded before any
  audio is read, leaving **274,491 rows over 588 reciters**:

  | reason | rows |
  |---|---|
  | panel reciter (§6) | 1,601 |
  | `decode_evalset.dev` | 530 |
  | `decode_evalset.test` | 435 |
  | `decode_evalset.legacy_stratified` | 297 |
  | `human_label.p35_fixture` | 64 |
  | `truth_site.p35_fixture` | 42 |
  | `human_label.waqf_events` | 22 |
  | `truth_site.waqf_boundary` | 22 |
  | `human_label.tashkeel_counterfactual` | 9 |
  | `human_label.reject_reread` | 3 |
  | `human_label.reject_bleed` | 1 |
  | **total excluded** | **2,959** |

  A row tied to two uses counts under both, so the reasons sum above the total. The exclusion
  is `tadabur.exposure.RowExclusion.for_use("consonant1.training")` — the sealed panel's
  reciters and every row tied (as itself, an alias or a probable copy) to an evaluation
  recording of `TRAINING_BARRED_USES`.
- **Exposure registration:** `tools/tadabur/exposure/consonant1.training.shards.json`, a shard
  use with **membership `exact`** on the 278 shards, `shared_baseline` false, counted from the
  full shard index (SHA-256 in the file). Written by
  `python -m tadabur.exposure training-stream`.
- **Launch assertion:** `scripts/issue-178-train.sh` recomputes the derivation from the log and
  runs `training-stream --check`, which recomputes the shard use and exits non-zero on any
  drift in shards, admitted rows, index or membership. The trainer additionally refuses a
  held-out shard before it downloads anything, so the framed reserve cannot reach training.
- **Never read:** Tarteel v1, RetaSy, qdat, qdat_bench, QUD, `decode_evalset`, the sealed panel,
  Ship A held-out, any truth site. None is decoded, graded or streamed by this run.

## Optimisation

As #94, with a new seed: batch 32, lr 5e-5, warmup 500, WSD with `--cooldown-steps 3000`, EMA
0.999, bf16 autocast, 10,000 steps, **seed 178** (fresh shard order and fresh augmentation
draws; it is resume-critical). `--num-workers 1` (one shard on disk at a time),
`--save-every 2000`, `--log-every 50`, `--eval-every 2000`, `--shuffle-buffer 512`.

**The candidate is the final step.** Validation is logged and never selects a checkpoint; only
live vs EMA is chosen later, on the look half (#179).

## Validation (monitoring only, and not an evaluation set)

`--audio-root` is `$ROOT/val_clips`: up to 200 clips staged at launch from the **last shard of
the run's own seed-178 permutation**, which 10,000 steps (~150 steps per shard, ~41k steps of
capacity) never reaches. Its 2% hash split gives the logged val rows. This departs from #94,
which monitored on `decode_evalset` dev clips: #178 requires that no evaluation set is decoded
during training, and dev is one (G2 reads it in #179). The val numbers are a health signal on
training-set audio, not a generalisation estimate, and are not comparable to #94's.

## Launch

- **Script:** `scripts/issue-178-train.sh launch|check|report`, mirroring #176's launcher.
  `launch` refuses while `/root/scratch/gpu.lock` is held, archives HEAD to
  `/root/scratch/issue-178/repo`, stages the validation clips, and starts `run` detached
  under `flock /root/scratch/gpu.lock`.
- **Smoke first:** 300 steps, then a gate (throughput ≥ 0.6 steps/s, KL finite, averaged
  weights saved) before the 10,000-step run. Anything else stops the chain.
- **Done file:** `run` writes `logs/train.exit` and then `logs/train.done` on every path
  (early failure included); a budget stop writes exit 2 and a `budget-exceeded` done-file.
  `check` reads those, never a log tail. `BUDGET` is 6 h from the lock's acquisition.
- **Launch commit:** this note's commit descends from `8961fdc` (2026-10-23 09:52 -0700, #177's
  freeze) and carries #174's doc (`85da3b6`, 2026-10-22 22:54 -0700) in its history; the
  launcher records the exact archived SHA in `logs/launch.txt`.
- **Software:** conda `hifzguide` on cuda-dev, unchanged from #94 — torch 2.11.0+cu128,
  transformers 5.13.0, ffmpeg 8.1.2 with libopus; `OMP_NUM_THREADS=1`.

## Smoke run (300 steps) — pending

| | candidate |
|---|---|
| steps/s, steps 25-300 | pending |
| KL at steps 25 / 100 / 200 / 300 | pending |
| pre-clip gradient norm, range | pending |
| blank-collapse margin, range | pending |
| averaged weights saved | pending |
| peak VRAM, minimum free disk | pending |

## Full run (10,000 steps) — pending

| | candidate |
|---|---|
| launched (UTC) / wall time | pending |
| steps/s | pending |
| KL, first / last / range | pending |
| final val (monitor only) | pending |
| **checkpoint SHA-256, live and EMA** | pending (one file holds both; the identity is `weights`) |
| minimum free disk / peak VRAM | pending |

`report` prints the final step, the checkpoint SHA-256, throughput, KL range and the final val
row from `metrics.jsonl`; those fill the table above and nothing is re-run to get them.

## The look-half readings (shared-text rule)

No look-half reading shaped the recipe, the data, or any weight: the run is #94's recipe on a
shard-level partition, and no shard, ayah or word was selected or weighted using anything read
on the look half. Recorded for completeness, from `docs/consonant-round-freeze.md` (#177):

- C95's false flags by cause on the look half: 2,105 in total at the shipped default, of which
  `other_consonant` 1,532 (35.03 per 1,000) and `tashkeel` 212 (4.85).
- G5's ten substituted consonants (`ء→∅`, `ل→ن`, `ن→∅`, `ج→ش`, `م→∅`, `ء→ي`, `و→∅`, `د→ت`,
  `ر→ل`, `ل→∅`) — the veto list #179 reads on the confirm half. **No audio was selected or
  weighted by ayah or word.**

## Acceptance checklist

- [x] Recipe and data fixed and committed before launch.
- [x] Exposure registered (`consonant1.training`, membership `exact`, 278 shards).
- [x] Launch commit later than #174's doc and #177's freeze (see "Launch").
- [x] No extra corpus, so no fingerprint report is owed.
- [ ] Smoke table and full-run metrics recorded here, with the final checkpoint SHA-256.
- [ ] No evaluation set decoded or graded during training (asserted by the staged validation
      clips and by the run streaming only the registered 278 shards).

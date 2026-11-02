# #254's round-3 candidate: round 2's weights, the leftover frame, the next tlog prefix

**Status (2026-11-02, 21:29:26Z):** the run finished at step 10,000 with chain exit 0, **3.13 h**
after the full arm started (`18:21:41Z` → `21:29:26Z`; the smoke ran `18:06:50Z`-`18:21:41Z`). It
launched from `9257c9b` at `18:05:29Z` (`logs/launch.txt`), which the launcher archived to
`/root/scratch/issue-252/repo` before setup. The tables below are the run's readings, taken from
`logs/chain.out`, `logs/events`, `logs/train.log`, `logs/df.log` and
`runs/consonant3_candidate/metrics.jsonl`; nothing in them was chosen after seeing a result — the
candidate is the final step. The checkpoint stays in cuda-dev scratch, backed up to the NAS
(`/Volumes/NetBackup/HifzGuide/checkpoints/consonant3-candidate-issue254`).

This is #254's candidate for round 3: round 2's averaged weights, round 2's recipe unchanged, the
**leftover 194-shard Tadabur frame** and the tlog walk from row 73,991 at the pre-registered 3:1
mix. There is no round-3 control arm: the round reads C95 as its baseline and round 2's EMA as its
context column (pre-registration, "Recipe"). This note records the run; the verdict comes from the
round's readout, not here, and its decoding belongs to that readout.

## What is compared

| role | what it is |
|---|---|
| **candidate (this run)** | round 2's candidate `ema_state` continued for 10,000 steps under `aug94-mixture-v2`; `consonant3.training` (194 shards) plus `tlog.training` from walk row 73,991 |
| baseline `c95` | #94's shipped model, unchanged |
| context column `r2` | round 2's candidate at its averaged weights (#228), read from round 2's committed decodes |
| control | none — round 3 has no control arm |

## The warm start

- **Round 2's checkpoint:**
  `/root/scratch/issue-228/runs/consonant2_candidate/checkpoint.pt`, SHA-256
  `893523710186cb16854dd62a258331644e2284abbee8b2cb54e8ab9d32dbea58`, step 10,000. One file holds
  both the live and averaged weights; `--init-weights ema` reads `ema_state`.
- **Read with `--init-from … --init-weights ema`.** The launcher's assertion recomputes the SHA-256
  and refuses any other file. The trainer logged
  `[init] warm-started from … (its step 10000, weights=ema); fresh optimiser and schedule`, and
  `run_config.json` records both under `initialisation` (`init_from`, `init_weights`).
- **Fresh optimiser and schedule.** `--init-from` is not `--resume`: the LR schedule restarts with a
  500-step warmup, as every earlier arm's did.
- **Preset `h448`, 43 classes** (`training.distill_student`), unchanged. No export happens in this
  issue.

## The objective

#94's frame-weighted KL to the clean base teacher, unchanged from rounds 1 and 2:

- **Teacher:** `obadx/muaalem-model-v3_2` @ `01a1ef9`, frozen, bf16, reading **clean** audio.
  Pinned to `01a1ef9fbe40d144ef845101e89ff924aed3fef5` through preflight and training; the run's
  `run_config.json` records it.
- **Loss:** temperature 2.0, nonblank weight 3.0, confirm weight 2.0, logit weight 1.0; feature,
  CTC and hard-label weights 0. **No supervised term on human labels.**
- **Health metrics:** KL, pre-clip gradient norm, confirmed agreement, blank-collapse margin, and the
  same split per source (`tlog-kl`, `tadabur-kl`, `tlog-blank`, `tadabur-blank`).

## Augmentation

`aug94-mixture-v2` verbatim (recorded in `run_config.json` and the frame manifest): clean 0.30,
noise 0.35, reverb 0.15, 8 kHz band limit 0.10, Opus 0.10. Draws are keyed per clip by
`(seed + 2, shard, row)`, so a Tadabur row replayed from an earlier round keeps its augmentation.
The noise pool is `/root/scratch/issue-94/noise_pool.npz` (458 recordings, SHA-256
`d2978d0f…fd43`); the recipe record's `noise_pool_sha256` (`16c099d1…574b`) is the committed pool
manifest, `tools/tadabur/augmentation/noise_pool.json`.

## Data: the 24 + 8 mix

- **Mix:** `tlog=24,tadabur=8` in `run_config.json`. The smoke gate re-read it from the batch
  accounting: `every batch {'batch_tlog': 24, 'batch_tadabur': 8}`.
- **tlog:** `tlog.training`, the S8 row-set use (**426,825 rows**, 1,410,930 windows), read through
  `tlog/training_schedule.json` — seed `round2-tlog-shards-2026-10`, K = 8 shards interleaved, 739
  shard files (407 clean, 332 unclean), SHA-256
  `8669ceca5ec67215a5c0147e97c0f74c97e5b20d29c8b1446cd8212feb8e134d`. The walk starts **after**
  round 2's committed 73,990 keys (`--tlog-after`); the gate verified the first key read is walk row
  73,991 and that no round-2 key appears.
- **Tadabur:** `consonant3.training`, membership `exact`: `consonant2.training`'s 211 shards minus
  the first 17 of its seed-2026 permutation — **194 shards, 191,378 admitted rows**, re-derived at
  launch (`consonant3.training: 194 shards, 191,378 admitted rows -- matches the committed use`).
- **Launch assertions:** `check_row_set_disjoint` printed
  `tlog.training: 426825 rows disjoint from every evaluation use`.
- **Never read:** Tarteel v1, RetaSy, qdat, qdat_bench, QUD, `decode_evalset`, the sealed panel,
  Ship A held-out, any truth site, the tlog hold-out, and the tlog look/confirm tranches.

What the run actually read, from `train.log`'s own accounting lines:

| stream | rows | clips | windows | real audio |
|---|---|---|---|---|
| tlog, clean | 49,707 | 49,707 | 151,609 | 436,485.211 s (121.2 h) |
| tlog, unclean | 25,846 | 25,846 | 88,975 | 254,545.206 s (70.7 h) |
| **tlog, both splits** | **75,553** | **75,553** | **240,584** | **691,030.417 s (192.0 h)** |
| **Tadabur** | 16,581 | 16,581 | **80,535** | 221,521.737 s (61.5 h) |

- **tlog consumed 75,553 rows**, 139 shard files, with no duplicate `(split, shard_file, row)` key
  in `tlog_consumed.jsonl.gz`. The planned prefix (`tlog/training_prefix.round3.json`) was 75,358
  rows / 240,002 windows; the run read 195 more rows and 582 more windows (clean 49,707 vs 49,609
  planned, unclean 25,846 vs 25,749). The log is committed at
  `tools/tadabur/tlog/consumed/consonant3.jsonl.gz` (217 KB, SHA-256 `abc3e168…7d222`) with
  `consonant3.meta.json` beside it, and copied to
  `/Volumes/NetBackup/HifzGuide/round3-consumed/` (SHA-verified). Its keys are the walk's rows
  **73,991-149,543**, disjoint from round 2's 73,990 — a test rebuilds the frozen walk and asserts
  both facts.
- **Tadabur consumed 16,581 of the 191,378 admitted rows (8.7%)** — 16 shards completed (the
  log's own count), or 80,535 windows at 8 windows a step.
- The counters are the loaders' emitted totals: 10,000 steps trained 240,000 tlog and 80,000 Tadabur
  windows, so the small excess above those is windows emitted ahead of the final step.

## Optimisation

Round 2's schedule unchanged: batch 32, lr 5e-5, warmup 500, WSD with `--cooldown-steps 3000`, EMA
0.999, bf16 autocast, 10,000 steps, **seed 2030** (it orders the shards; the window shuffle derives
seed 2031), `--num-workers 1`, `--save-every 2000`, `--log-every 50`, `--eval-every 2000`,
`--shuffle-buffer 512`, weight decay 0.01, max grad norm 5.0, grad accumulation 1.

**The candidate is the final step.** Validation is logged and never selects a checkpoint.

## Validation (monitoring only, and not an evaluation set)

`--audio-root` is `$ROOT/val_clips`: **200 clips staged at launch from shard 150, the last of the
194 the run streams** (recorded position 193); its 2 % hash split gives the logged val windows —
`3 monitoring-only val windows`, from **1 clip in the val split** — so the logged val numbers are
thin and are a health signal only. The trainer's in-training evaluation is monitoring only: it
selects nothing.

Shard 150 is inside the 194, so the step budget is what keeps the clips out of training: 10,000
steps at 8 Tadabur windows each complete about 16 shards, far short of the 193 before shard 150.
Because that is arithmetic and not a property of the stream, it is **asserted after the run**:
`assert_monitor` reads `train.log`'s shard-completion lines and, if the run entered shard 150, fails
the chain and refuses to print a report. The launch recorded the shard and position in
`logs/val_shard.txt` (`150`, `193`). **The finished run passed it:**

```
[monitor] 16 shards completed, monitor shard 150 entered at 193: never streamed
```

| | round 3 (shard 150) | round 2, **indicative only, measured on a different shard (72)** |
|---|---|---|
| overall agreement | 0.9493 | 0.9556 |
| confirmed agreement | 0.9333 | 0.9467 |
| nonblank agreement | 0.9706 | 0.9599 |
| blank-collapse margin | −0.0293 | −0.020 |
| teacher / student blank ratio | 0.6373 / 0.608 | 0.690 / 0.669 |

The round-2 column is the pre-registered indicative line (Recipe → monitoring), not a bar and not
comparable: different shard, different clips, one val clip here.

**No evaluation set was decoded.** The chain has no decode stage; the only audio it opened is the
registered training streams (`tlog.training`, `consonant3.training`) and the 200 staged monitoring
clips cut from shard 150. The launch assertions above and `assert_monitor` are the evidence.

## Launch

- **Script:** `scripts/issue-252-train.sh launch|check|report`. `launch` refuses while
  `/root/scratch/gpu.lock` is held, below `MIN_FREE_GB` on `/root/scratch`, when the run directory
  already exists, and when the freeze record is not committed in HEAD with `power_rule.satisfied`
  (`freeze: 5 artefacts committed in HEAD, power_rule satisfied`); it archives HEAD to
  `/root/scratch/issue-252/repo`, stages the validation clips, and starts `run` detached under
  `flock /root/scratch/gpu.lock`.
- **Smoke first:** 700 steps with an evaluation every 150. The gate requires throughput ≥ 0.6
  steps/s, finite KL for both sources on every logged step, every batch exactly the pre-registered
  24/8 mix, both splits consumed, at most 8 tlog shards on disk, the log's
  `[init] warm-started from … weights=ema` line, a saved `ema_state`, and that the tlog walk began
  exactly at row 73,991 with no round-2 key. Anything else stops the chain.
- **Done file:** `run` writes `logs/train.exit` and then `logs/train.done` on every path; a budget
  stop writes exit 2 and a `budget-exceeded` done-file. `check` reads those, never a log tail.
  `BUDGET` is 6 h; the run used 3.13 h.
- **Launch commit:** `9257c9b8b60d11008265e56c57724064f718186d` (`Round 3 freeze (4/4): the look-side
  readings and the power rule`), archived at `2026-11-02T18:05:29Z`; `logs/launch.txt` records both.
- **Software:** conda `hifzguide` on cuda-dev, unchanged — Python 3.11.15, torch 2.11.0+cu128,
  transformers 5.13.0, ffmpeg 8.1.2; `OMP_NUM_THREADS=1`. Preflight peak 11.05 GiB of the 15.0 GiB
  budget at batch 32.

## Smoke run (700 steps)

| | candidate |
|---|---|
| steps/s, steps 25-700 | **0.98** (gate threshold 0.6) |
| KL at the gate | range 0.322-0.594, last 0.385 |
| mix accounting | 24 tlog + 8 Tadabur per batch; splits `['clean', 'unclean']`; max 8 tlog shards on disk |
| averaged weights saved | yes — `ema_state` beside the live `student` at step 700 |
| val records | 4 evaluations (steps 150, 300, 450, 600); the 200 clips put 1 in the val split (3 val windows) |
| first tlog key | walk row 73,991 (`clean-00085-of-00411` row 375); no round-2 key |
| smoke rows read | tlog clean 2,268 rows / 8,051 windows / 22,833.864 s; unclean 2,825 / 9,341 / 26,855.095 s; Tadabur 1,274 clips / 6,135 windows / 16,906.986 s |

The gate's lines, verbatim from `logs/events`:

```
[gate] first consumed key is walk row 73,991 (clean-00085-of-00411 row 375), after the 73,990 keys in /root/scratch/issue-252/repo/tools/tadabur/tlog/consumed/consonant2.jsonl.gz
[gate] 0.98 steps/s over steps 25-700, KL 0.322-0.594, last 0.385
[gate] every batch {'batch_tlog': 24, 'batch_tadabur': 8}; splits consumed: ['clean', 'unclean']; max 8 tlog shards on disk
[gate] [init] warm-started from /root/scratch/issue-228/runs/consonant2_candidate/checkpoint.pt (its step 10000, weights=ema); fresh optimiser and schedule
[gate] averaged weights saved beside the live ones at step 700
```

## Full run (10,000 steps)

| | candidate |
|---|---|
| launched (UTC) / wall time | `2026-11-02T18:05:29Z`; smoke `18:06:50Z`-`18:21:41Z`; full arm `18:21:41Z` → `21:29:26Z`, **3.13 h** (`[done]` logs 3.12 h) |
| steps/s | **0.89** over steps 50-10000 (11,224.4 s); the live sample at steps 350-650 measured **1.00** |
| KL, first / last / range | 0.3329 / 0.3438 / 0.2839-0.6632; last `tlog-kl` 0.3966, `tadabur-kl` 0.2006 |
| pre-clip gradient norm, max | 1.799 (none clipped) |
| final val (monitor only) | see the monitoring table above; 1 clip in the val split |
| **checkpoint SHA-256, live and EMA** | `1193e2117b56381ccc078a5b5e21d61d0ad0903565dcca0cb110b39ce24ed4a9` — one file holds both (`student` live, `ema_state` averaged; its `step` is 10,000) |
| shards consumed | **16 of 194 completed**; monitor shard 150 at position 193 not entered |
| minimum free disk / peak run directory | 27.60 GB (25.7 GiB) free at its minimum; the run's `$ROOT` (archived repo, `runs/`, val clips, tlog cache) peaked at 7.44 GB (6.93 GiB) / 11.05 GiB preflight VRAM |

The last steps and the final evaluation, verbatim from `logs/train.log`:

```
[eval 10000] {'overall_agreement': 0.9493, 'confirmed_agreement': 0.9333, 'nonblank_agreement': 0.9706, 'teacher_blank_ratio': 0.6373, 'student_blank_ratio': 0.608, 'blank_collapse_margin': -0.0293, 'target_prob': 0.9228, 'target_rank': 1.03, 'blank_prob': 0.0509, 'top5_agreement': 1.0, 'prob_margin': -0.872}
[done] 10000 steps in 3.12h -- final {'overall_agreement': 0.9493, 'confirmed_agreement': 0.9333, 'nonblank_agreement': 0.9706, 'teacher_blank_ratio': 0.6373, 'student_blank_ratio': 0.608, 'blank_collapse_margin': -0.0293, 'target_prob': 0.9228, 'target_rank': 1.03, 'blank_prob': 0.0509, 'top5_agreement': 1.0, 'prob_margin': -0.872}
```

## The frame manifest

Written into the run directory before the first step by `manifest`, from the spec `assert_stream`
verified (`runs/consonant3_candidate/frame_manifest.json`, SHA-256
`69dd9e86e7b98728356d7384798ab71003cac34f5cc4c6e13ab59d79c3e22024`):

```
frame manifest: 194 shards, shards_sha256 f2680472825a04c3443b77cad8f5e82e341f362066e42e6cd5071fb3b5813cfc, stream_order_sha256 2ad7f3692804a14fc1871f06cf7610ad8fcf2dd896740aaf1499151d24a203e6, stream-order seed 2030, window-shuffle seed 2031, recipe aug94-mixture-v2 -> /root/scratch/issue-252/runs/consonant3_candidate/frame_manifest.json
```

The manifest's load-bearing fields:

- `shards` / `shards_sha256`: the 194-shard frame, membership digest over the sorted list.
- `stream_order` / `stream_order_sha256`: the seed-2030 order the run streamed; its first 16 entries
  (`233, 243, 64, 128, 175, 344, 281, 95, 88, 140, 255, 300, 60, 313, 70, 327`) were completed at
  step 10,000, and the next in the order is shard 90.
- `shard_order_seed` 2030 and `window_shuffle_seed` 2031: the two RNG seeds.
- `augmentation`: the `aug94-mixture-v2` recipe record, keyed per source row.
- `exposure_use`: `consonant3.training`, the registration the launch checked.

## Disclosures

**There is no seed replicate.** One seed fixes one shard order and one augmentation draw, not a
distribution. Round 3 trains a fresh window on a frame earlier rounds never replayed, so a
difference against C95 **below about 1-2 per 1,000 graded words is not interpretable**; nothing in
this run estimates the arm's own variance.

**The frame is unseen by round 2, not new data.** `consonant3.training` is round 2's 211-shard
frame minus the first 17 shards of its seed-2026 permutation (round 2 completed 16 and stopped
inside shard 378, the 17th), with the same `TRAINING_BARRED_USES` and reciter exclusions. Membership
is `exact`, so no round-3 step replays audio round 2 streamed.

**The tlog prefix is a continuation, not a resample.** The walk starts at row 73,991 and read
75,553 rows; the rows are fixed by the frozen schedule and the seed, not selected.

**Coverage, from the pre-registration.** The planned prefix covers the clean surahs in
`tools/tadabur/tlog/training_prefix.round3.json` (`surahs_covered`), a seeded shard-cluster sample,
not row-uniform sampling. The run read the whole planned prefix plus 195 rows past it, so its
coverage is the frozen prefix's, not the pool's.

## Acceptance checklist

- [x] Run exits 0 at step 10,000 within BUDGET; `assert_monitor` passed — 16 shards completed,
      monitor shard 150 never entered.
- [x] Launch commit `9257c9b`, archived `2026-11-02T18:05:29Z`; config, the 24 + 8 mix and the warm
      start recorded from `run_config.json`, the gate and `train.log`.
- [x] Live and EMA checkpoint SHA-256 `1193e211…24ed4a9` recorded; the file is copied to
      `/Volumes/NetBackup/HifzGuide/checkpoints/consonant3-candidate-issue254` with a matching
      SHA-256 and listed in that folder's README.
- [x] tlog schedule SHA-256 `8669ceca…134d`; per-split rows, clips, windows and seconds quoted
      (clean 49,707 / 151,609 / 436,485.211 s; unclean 25,846 / 88,975 / 254,545.206 s; Tadabur
      16,581 clips / 80,535 windows / 221,521.737 s).
- [x] Consumed keys committed at `tools/tadabur/tlog/consumed/consonant3.jsonl.gz` +
      `consonant3.meta.json`; a test asserts they are walk rows 73,991-149,543, disjoint from round
      2's, and the NAS copy at `/Volumes/NetBackup/HifzGuide/round3-consumed/` is SHA-verified.
- [x] The frame manifest's `shards_sha256` and `stream_order_sha256` recorded; the seeds and the
      exposure use named.
- [x] No evaluation set decoded or graded: the chain decodes nothing, `check_row_set_disjoint`
      passed, and `assert_monitor` passed.
- [x] `CONSONANT3.models` pins `r3_candidate_{live,ema}` (one checkpoint, step 10,000, recipe
      `aug94-mixture-v2`, SHA above); `docs/consonant-round3-candidate.md` records this run.

# #223's round-2 control: #178's recipe on Tadabur alone

**Status (2026-10-29, 08:13:25Z):** the run finished at step 10,000 with chain exit 0, **3.11 h**
after the full arm started (`05:06:28Z` → `08:13:25Z`; the smoke ran `04:59:48Z`-`05:06:28Z`).
It launched from `ec69695` at `04:58:51Z` (`logs/launch.txt`), which the launcher archived to
`/root/scratch/issue-223/repo` before setup. The tables below are the run's readings, taken from
`logs/chain.out`, `logs/events`, `logs/train.log` and `runs/consonant2_control/metrics.jsonl` by
`scripts/issue-223-control.sh report`; nothing in them was chosen after seeing a result — the
control is the final step. The checkpoint stays in cuda-dev scratch, backed up to the NAS
(`/Volumes/NetBackup/HifzGuide/checkpoints/consonant2-control-issue223`).

This is #223's control arm for round 2's tlog candidate: #178's recipe one round further, on
Tadabur alone. It trains the same warm start under the same schedule and seed as the candidate, so
candidate − control shows what the tlog data added rather than what 10,000 more steps did. It never
enters the round-2 verdict, and nothing decoded it before the round's readout.

## Round 2's freeze exception

Round 2 freezes its sets, statement and frame **before any arm trains**. This arm trained before
that freeze, as an explicit, owner-approved exception recorded on #223: the control is #178's
recipe with #178's warm start, so running it early cannot change the candidate's bars, and it is
never part of the verdict. **No evaluation set was decoded:** the only audio read is the registered
211-shard Tadabur frame and nine monitoring windows staged from that same frame (see "Validation").

## What is compared

| arm | student input |
|---|---|
| candidate (round 2) | this frame replayed with the tlog app-user substream added |
| **control (this run)** | `consonant2.control.training`: 211 Tadabur shards, seed 2026 |
| warm start (both) | #178's averaged weights, step 10,000 |

The control is optional and non-gating in #174's pre-registration; here it is the arm that
separates the round's one new ingredient from another 10,000 steps of #178's recipe.

## The warm start

- **#178's checkpoint:**
  `/root/scratch/issue-178/repo/tools/runs/consonant1_candidate/checkpoint.pt`, SHA-256
  `057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2`, step 10,000. One file
  holds both the live and averaged weights; `--init-weights ema` reads `ema_state`.
- **Read with `--init-from … --init-weights ema`.** `assert_ready` recomputes the SHA-256 and
  refuses any other file (the same assertion covers the noise pool). The trainer logs
  `[init] warm-started from … (its step 10000, weights=ema); fresh optimiser and schedule`, and
  `run_config.json` records both under `initialisation` (`init_from`, `init_weights`).
- **Fresh optimiser and schedule.** `--init-from` is not `--resume`: the LR schedule restarts
  with a 500-step warmup, deliberately, as #94's and #178's runs did.
- **Preset `h448`, 43 classes** (`training.distill_student.py`), unchanged. No export happens in
  this issue.

## The objective

#94's frame-weighted KL to the clean base teacher, unchanged from #178:

- **Teacher:** `obadx/muaalem-model-v3_2` @ `01a1ef9`, frozen, bf16, reading **clean** audio.
  Pinned to `01a1ef9fbe40d144ef845101e89ff924aed3fef5` through preflight and training; the run's
  `run_config.json` records it and `report` reads it back.
- **Loss:** temperature 2.0, nonblank weight 3.0, confirm weight 2.0, logit weight 1.0; feature,
  CTC and hard-label weights 0. **No supervised term on human labels.**
- **Health metrics:** KL, pre-clip gradient norm, confirmed agreement, blank-collapse margin.

## Augmentation

`aug94-mixture-v2` verbatim (recorded in the run's `run_config.json` and frame manifest): clean
0.30, noise 0.35, reverb 0.15, 8 kHz band limit 0.10, Opus 0.10. Draws are keyed per clip by
`(seed + 2, shard, row)`, so the candidate reading the same rows gets the same augmented windows.
The noise pool is `/root/scratch/issue-94/noise_pool.npz` (458 recordings, SHA-256
`d2978d0f…fd43`), checked against the launcher's pin at launch; the recipe record's
`noise_pool_sha256` (`16c099d1…574b`) is the committed pool manifest,
`tools/tadabur/augmentation/noise_pool.json`.

## Data: #178's 278 shards minus its own consumed prefix

- **The 211** are `h448`'s 345 training shards minus #94's 67 consumed (seed 9494) minus #178's
  67 consumed (seed 178). `training.consumed_prefix.remaining_after_logs` rebuilds each
  permutation from its own log and refuses a log whose order is not that permutation;
  `scripts/issue-223-control.sh setup` prints the result and records it as `--stream-shards`.
- **Rows:** the 211 shards hold **210,450** rows; **2,252 (1.07%)** are excluded before any audio
  is read, leaving **208,198 rows over 579 reciters**. The exclusion is
  `tadabur.exposure.RowExclusion.for_use("consonant2.control.training")` — the sealed panel's
  reciters and every row tied (as itself, an alias or a probable copy) to an evaluation recording
  of `TRAINING_BARRED_USES`.
- **Exposure registration:** `tools/tadabur/exposure/consonant2.control.training.shards.json`, a
  shard use with **membership `exact`** on the 211 shards, `shared_baseline` false, counted from
  the full shard index (SHA-256 in the file). Written by
  `python -m tadabur.exposure training-stream`.
- **Launch assertion:** `assert_stream` reruns the derivation, names the monitor shard and its
  position, and runs `training-stream --check`, which recomputes the shard use and exits non-zero
  on any drift. The run printed `consonant2.control.training: 211 shards, 208,198 admitted rows --
  matches the committed use`. The trainer additionally refuses a held-out shard before it
  downloads anything.
- **Never read:** Tarteel v1, RetaSy, qdat, qdat_bench, QUD, `decode_evalset`, the sealed panel,
  Ship A held-out, any truth site, the tlog hold-out. None is decoded, graded or streamed by this
  run.

## Optimisation

As #178, with the candidate's seed: batch 32, lr 5e-5, warmup 500, WSD with
`--cooldown-steps 3000`, EMA 0.999, bf16 autocast, 10,000 steps, **seed 2026** (it orders the
shards; the window shuffle derives seed 2027), `--num-workers 1`, `--save-every 2000`,
`--log-every 50`, `--eval-every 2000`, `--shuffle-buffer 512`.

**The control is the final step.** Validation is logged and never selects a checkpoint.

## Validation (monitoring only, and not an evaluation set)

`--audio-root` is `$ROOT/val_clips`: clips staged at launch from **shard 72, the last of the run's
own seed-2026 permutation** (recorded position 210, the last of 211); its 2% hash split gives the
logged val rows (9 windows). The trainer's in-training evaluation is monitoring only: it selects
nothing.

Shard 72 is inside the 211, so the step budget is what keeps the clips out of training: 10,000
steps complete 66 shards, far short of the 210 before shard 72. Because that is arithmetic and not
a property of the stream, it is **asserted after the run**: `assert_monitor` reads `train.log`'s
shard-completion lines and, if the run entered shard 72, fails the chain (so `logs/train.exit` is
non-zero) and refuses to print a report. The launch assertion recorded the shard and position in
`logs/val_shard.txt` (`72`, `210`). **The finished run passed it:** `[monitor] 66 shards
completed, monitor shard 72 entered at 210: never streamed`.

The val numbers are a health signal on training-set audio, not a generalisation estimate.

## Launch

- **Script:** `scripts/issue-223-control.sh launch|check|report`, mirroring #178's launcher.
  `launch` refuses while `/root/scratch/gpu.lock` is held, below `MIN_FREE_GB` on `/root/scratch`,
  when the run directory already exists, and when `--resume` names no checkpoint; it archives HEAD
  to `/root/scratch/issue-223/repo`, stages the validation clips, and starts `run` detached under
  `flock /root/scratch/gpu.lock`.
- **Smoke first:** 300 steps with `--eval-every 150`, so the staged validation clips are exercised
  before the gate. The gate requires throughput ≥ 0.6 steps/s, finite KL, **at least one non-empty
  val record**, averaged weights saved, and the log's `[init] warm-started from … weights=ema`
  line. Anything else stops the chain.
- **Done file:** `run` writes `logs/train.exit` and then `logs/train.done` on every path (early
  failure included); a budget stop writes exit 2 and a `budget-exceeded` done-file. `check` reads
  those, never a log tail. `BUDGET` is 6 h from the lock's acquisition.
- **Launch commit:** `ec69695809cd9ca5af970ac2f01cf56c6c3d8df9` (`issue-223: the monitor guard
  counts a replayed shard once`), archived at `2026-10-29T04:58:51Z`; `logs/launch.txt` records
  both.
- **Software:** conda `hifzguide` on cuda-dev, unchanged from #178 — torch 2.11.0+cu128,
  transformers 5.13.0, ffmpeg 8.1.2; `OMP_NUM_THREADS=1`.

## Smoke run (300 steps)

| | control |
|---|---|
| steps/s, steps 25-300 | **0.88** (gate threshold 0.6) |
| KL at steps 25 / 100 / 200 / 300 | 0.249 / 0.170 / 0.181 / 0.167 (range 0.140-0.249) |
| pre-clip gradient norm, range | 0.4-0.8 (none clipped) |
| blank-collapse margin, range | −0.026 … −0.014 |
| averaged weights saved | yes — `ema_state` beside the live `student` at step 300 |
| val records (non-empty) / steps | 3 records at steps 150 and 300 (the step-300 record is logged again as the final eval); overall agreement 0.953 then 0.952 |
| warm start line (`weights=ema`) | present (see the gate lines below) |
| peak VRAM / minimum free disk | 11.05 GiB of the 15.0 GiB budget (the preflight probe; no in-loop sampler) / 19.57 GB (18.2 GiB) free after staging; the run's minimum is in the full-run table |

The gate's lines, verbatim from `logs/events`:

```
[gate] 0.88 steps/s over steps 25-300, KL 0.140-0.249, last 0.167
[gate] 3 val evaluations, last {"blank_collapse_margin": -0.0258, "blank_prob": 0.0577, "confirmed_agreement": 0.9289, "final": true, "nonblank_agreement": 0.9628, "overall_agreement": 0.952, "prob_margin": -0.8599, "split": "val", "step": 300, "student_blank_ratio": 0.664, "target_prob": 0.9176, "target_rank": 1.05, "teacher_blank_ratio": 0.6898, "top5_agreement": 1.0}
[gate] [init] warm-started from /root/scratch/issue-178/repo/tools/runs/consonant1_candidate/checkpoint.pt (its step 10000, weights=ema); fresh optimiser and schedule
[gate] averaged weights saved beside the live ones at step 300
```

## Full run (10,000 steps)

| | control |
|---|---|
| launched (UTC) / wall time | `2026-10-29T04:58:51Z`; smoke `04:59:48Z`-`05:06:28Z`; full arm `05:06:28Z` → `08:13:25Z`, **3.11 h** |
| steps/s | **0.90** over steps 50-10000 |
| KL, first / last / range | 0.188 / 0.187 / 0.134-0.337 |
| pre-clip gradient norm, max | 2.17 (none clipped) |
| final val (monitor only) | overall 0.956, confirmed 0.938, nonblank 0.963, blank-collapse margin −0.022, teacher/student blank ratio 0.690 / 0.668 |
| **checkpoint SHA-256, live and EMA** | `da5fb5bf02c7b078efbfb781855c44f171bc24a2d709cd71eefc7d2443c20473` — one file holds both (`student` live, `ema_state` averaged; the loader's `weights` picks) |
| shards consumed | **67 of 211 entered** (66 completed, stopped inside the 67th — shard 64); 144 never entered; monitor shard 72 at position 210 not entered |
| minimum free disk / peak VRAM | 16.60 GB (15.5 GiB) free, from 21.99 GB; the run directory peaked at 2.32 GB (2.16 GiB) / 11.05 GiB (preflight probe; no in-loop VRAM sampler) |

The last steps and the final evaluation, verbatim from `logs/train.log`:

```
[ 10000/10000] loss 0.1870 kl 0.1870 feat    off ctc    off hard    off cos   off conf-agree 0.935 blank-margin -0.021 |g| 0.5
[eval 10000] {'overall_agreement': 0.9556, 'confirmed_agreement': 0.9378, 'nonblank_agreement': 0.9628, 'teacher_blank_ratio': 0.6898, 'student_blank_ratio': 0.6676, 'blank_collapse_margin': -0.0222, 'target_prob': 0.9135, 'target_rank': 1.05, 'blank_prob': 0.0606, 'top5_agreement': 1.0, 'prob_margin': -0.853}
[done] 10000 steps in 3.11h -- final {'overall_agreement': 0.9556, 'confirmed_agreement': 0.9378, 'nonblank_agreement': 0.9628, 'teacher_blank_ratio': 0.6898, 'student_blank_ratio': 0.6676, 'blank_collapse_margin': -0.0222, 'target_prob': 0.9135, 'target_rank': 1.05, 'blank_prob': 0.0606, 'top5_agreement': 1.0, 'prob_margin': -0.853}
```

## The frame manifest

Written into the run directory before the first step by `manifest`, from the spec `assert_stream`
verified (`runs/consonant2_control/frame_manifest.json`). The round-2 candidate's all-Tadabur
substream replays this frame: the shard list, the order it streams, both seeds and the recipe
record. Copied verbatim:

```json
{"augmentation": {"band_limit_rate": 8000, "draw_seed": "random.Random('<seed + 2>:<shard>:<row>')", "drr_db": 3.0, "mixture": {"band_limit": 0.1, "clean": 0.3, "codec": 0.1, "noise": 0.35, "reverb": 0.15}, "noise_pool_sha256": "16c099d1099b43108dc1ab0ba7f4e0d3292cf6f17a268eee25f130621f5f574b", "opus": ["voip_8k_cutoff4000_20ms", "voip_12k_cutoff4000_20ms", "voip_16k_cutoff4000_20ms"], "rt60_s": [0.2, 1.0], "silent_noise": {"draws": 8, "exhausted": "clean", "floor_rms": 0.0001, "redraw_seed": "random.Random('<seed + 2>:<shard>:<row>:<attempt>')"}, "snr_db": [3.0, 20.0], "version": "aug94-mixture-v2"},
 "exposure_use": "consonant2.control.training",
 "shard_order_seed": 2026,
 "shards": [21, 22, 23, 25, 28, 30, 31, 32, 33, 35, 36, 37, 40, 41, 42, 44, 45, 48, 53, 54, 55, 56, 60, 61, 64, 65, 66, 67, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 86, 87, 88, 89, 90, 91, 93, 95, 97, 98, 99, 100, 102, 104, 105, 106, 108, 111, 113, 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 131, 132, 137, 139, 140, 141, 142, 143, 146, 147, 148, 149, 150, 151, 155, 156, 159, 162, 164, 167, 168, 170, 171, 173, 174, 175, 176, 177, 179, 180, 181, 184, 185, 186, 187, 188, 190, 192, 194, 196, 197, 199, 200, 204, 206, 208, 214, 215, 220, 221, 222, 223, 224, 226, 228, 230, 232, 233, 235, 236, 238, 239, 240, 241, 243, 245, 247, 252, 253, 254, 255, 259, 260, 261, 262, 263, 269, 270, 271, 273, 274, 276, 277, 278, 279, 280, 281, 284, 285, 287, 288, 289, 291, 294, 295, 296, 299, 300, 301, 302, 303, 308, 309, 310, 311, 313, 315, 316, 317, 320, 321, 327, 329, 332, 334, 336, 339, 340, 341, 344, 347, 351, 353, 355, 357, 358, 359, 360, 364, 366, 368, 369, 375, 376, 377, 378, 379, 380, 383, 384],
 "shards_sha256": "4bd00c2ba5d0979bcc017546e3beeea70807dad443b125fa293adfe7f5f5d7d8",
 "stream_order": [176, 37, 67, 97, 91, 98, 276, 36, 262, 89, 252, 25, 81, 174, 142, 188, 378, 261, 170, 196, 177, 294, 56, 41, 126, 87, 45, 90, 253, 179, 360, 78, 384, 42, 341, 151, 353, 236, 220, 340, 167, 35, 300, 383, 88, 48, 143, 329, 332, 32, 44, 125, 99, 204, 274, 316, 200, 197, 359, 132, 30, 344, 230, 140, 53, 102, 64, 287, 95, 288, 355, 33, 270, 105, 278, 296, 222, 301, 108, 243, 320, 86, 299, 180, 303, 187, 121, 123, 116, 284, 162, 245, 186, 93, 120, 260, 226, 141, 150, 302, 28, 22, 76, 80, 334, 106, 208, 185, 347, 364, 313, 149, 375, 129, 82, 155, 100, 131, 74, 128, 280, 310, 118, 156, 380, 327, 317, 73, 113, 224, 289, 221, 194, 377, 240, 40, 75, 376, 369, 239, 308, 223, 366, 124, 241, 315, 321, 235, 279, 71, 232, 255, 271, 184, 61, 192, 215, 171, 173, 31, 146, 269, 79, 214, 336, 139, 247, 368, 60, 159, 54, 291, 379, 175, 168, 164, 122, 181, 104, 147, 309, 357, 311, 23, 206, 65, 137, 70, 55, 281, 21, 119, 199, 273, 351, 228, 358, 339, 254, 263, 190, 259, 285, 277, 111, 66, 295, 238, 233, 148, 72],
 "stream_order_sha256": "7f0f1533618d0fc4e32e1fa34e20c4d1d22f4995779ad816a58e9d54193674fe",
 "window_shuffle_seed": 2027}
```

What it pins:

- `shards` / `shards_sha256`: the 211-shard frame, membership digest over the sorted list.
- `stream_order` / `stream_order_sha256`: the seed-2026 order the run actually streamed (the
  first 67 entries are the consumed prefix; 66 completed and the 67th, shard 64, was in
  progress at step 10,000).
- `shard_order_seed` 2026 and `window_shuffle_seed` 2027: the two RNG seeds. Same seed alone is
  not the frame (see "Disclosures").
- `augmentation`: the `aug94-mixture-v2` recipe record, keyed per source row.
- `exposure_use`: `consonant2.control.training`, the registration the launch checked.

## Disclosures

1. **There is no seed replicate.** One seed fixes one shard order and one augmentation draw, not a
   distribution. The control and the candidate are one realisation each, so a candidate − control
   difference **below about 1-2 per 1,000 graded words is not interpretable**; nothing in this run
   estimates the arm's own variance.
2. **"Same seed" gives no pairing once the data streams differ.** Seed 2026 pairs the schedule
   shape — step count, LR, EMA and the driver of the shard permutation — but the control streams
   211 Tadabur shards while the candidate adds a tlog substream. The samples per step differ, so a
   per-step or per-batch difference is not a paired comparison; only arm-level readings on a
   shared evaluation set can be compared, and those come from the round's readout, not from this
   note.
3. **The control anchors this schedule.** Its meaning is "the candidate's schedule without the
   new data": same warm start, 10,000 steps, lr 5e-5, warmup 500, WSD cooldown 3,000, EMA 0.999,
   `aug94-mixture-v2`. If the candidate's schedule changes — steps, learning rate, warmup,
   cooldown, EMA or recipe — this arm no longer isolates the data ingredient and becomes
   historical context, not attribution.

## Acceptance checklist

- [x] Recipe and data fixed and committed before launch: launch commit `ec69695`, the launcher
      that produced every path below (`scripts/issue-223-control.sh`).
- [x] Exposure registered (`consonant2.control.training`, membership `exact`, 211 shards) and
      re-checked at launch (`--check`: `211 shards, 208,198 admitted rows -- matches the
      committed use`).
- [x] Smoke gate passed on the staged validation clips; the averaged warm start is proven by the
      `[init] … weights=ema` line and the saved `ema_state`.
- [x] No evaluation set decoded or graded during training: the run streams only the registered
      211 shards and `assert_monitor` passed — 66 shards completed, monitor shard 72 never
      entered.
- [x] Final checkpoint SHA-256 recorded here and the file copied to
      `/Volumes/NetBackup/HifzGuide/checkpoints/consonant2-control-issue223` with a matching
      SHA-256, listed in that folder's README.
- [x] Frame manifest recorded verbatim, for the candidate's substream to replay.
- [x] Round 2's freeze exception recorded (owner-approved on #223; see "Round 2's freeze
      exception").

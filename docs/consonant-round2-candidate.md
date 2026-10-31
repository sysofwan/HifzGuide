# #228's round-2 candidate: #178's recipe plus the tlog substream

**Status (2026-10-29, 13:35:22Z):** the run finished at step 10,000 with chain exit 0, **2.81 h**
after the full arm started (`10:46:53Z` → `13:35:22Z`; the smoke ran `10:33:38Z`-`10:46:53Z`).
It launched from `a9ba80c` at `10:32:19Z` (`logs/launch.txt`), which the launcher archived to
`/root/scratch/issue-228/repo` before setup. The tables below are the run's readings, taken from
`logs/chain.out`, `logs/events`, `logs/train.log`, `logs/df.log` and
`runs/consonant2_candidate/metrics.jsonl`; nothing in them was chosen after seeing a result — the
candidate is the final step. The checkpoint stays in cuda-dev scratch, backed up to the NAS
(`/Volumes/NetBackup/HifzGuide/checkpoints/consonant2-candidate-issue228`).

This is #228's candidate for round 2: #178's recipe plus the tlog app-user substream, at the
pre-registered 3:1 mix. It trains the same warm start under the same schedule and seed as the
control (#223, [`consonant-round2-control.md`](consonant-round2-control.md)), so candidate −
control shows what the tlog data added rather than what 10,000 more steps did. This note records the
run; the verdict comes from the round's readout, not here, and its decoding belongs to that readout.

## What is compared

| arm | student input |
|---|---|
| **candidate (this run)** | the control's 211-shard Tadabur frame plus `tlog.training`: every 32-window batch is 24 tlog + 8 Tadabur |
| control (#223) | `consonant2.control.training`: 211 Tadabur shards, seed 2026 |
| warm start (both) | #178's averaged weights, step 10,000 |

The two arms therefore differ in the data ingredient alone. The control is reported only and never
enters the verdict.

## The warm start

- **#178's checkpoint:**
  `/root/scratch/issue-178/repo/tools/runs/consonant1_candidate/checkpoint.pt`, SHA-256
  `057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2`, step 10,000. Recomputed on
  cuda-dev when this note was written: the same digest. One file holds both the live and averaged
  weights; `--init-weights ema` reads `ema_state`.
- **Read with `--init-from … --init-weights ema`.** The launcher's assertion recomputes the SHA-256
  and refuses any other file. The trainer logs
  `[init] warm-started from … (its step 10000, weights=ema); fresh optimiser and schedule`, and
  `run_config.json` records both under `initialisation` (`init_from`, `init_weights`).
- **Fresh optimiser and schedule.** `--init-from` is not `--resume`: the LR schedule restarts with a
  500-step warmup, as #94's, #178's and the control's runs did.
- **Preset `h448`, 43 classes** (`training.distill_student`), unchanged. No export happens in this
  issue.

## The objective

#94's frame-weighted KL to the clean base teacher, unchanged from #178 and the control:

- **Teacher:** `obadx/muaalem-model-v3_2` @ `01a1ef9`, frozen, bf16, reading **clean** audio.
  Pinned to `01a1ef9fbe40d144ef845101e89ff924aed3fef5` through preflight and training; the run's
  `run_config.json` records it.
- **Loss:** temperature 2.0, nonblank weight 3.0, confirm weight 2.0, logit weight 1.0; feature,
  CTC and hard-label weights 0. **No supervised term on human labels.**
- **Health metrics:** KL, pre-clip gradient norm, confirmed agreement, blank-collapse margin, and the
  same split per source (`tlog-kl`, `tadabur-kl`, `tlog-blank`, `tadabur-blank`).

## Augmentation

`aug94-mixture-v2` verbatim (recorded in the run's `run_config.json` and frame manifest): clean
0.30, noise 0.35, reverb 0.15, 8 kHz band limit 0.10, Opus 0.10. Draws are keyed per clip by
`(seed + 2, shard, row)`, so shared Tadabur rows get the same augmented windows as the control's.
The noise pool is `/root/scratch/issue-94/noise_pool.npz` (458 recordings, SHA-256
`d2978d0f…fd43`); the recipe record's `noise_pool_sha256` (`16c099d1…574b`) is the committed pool
manifest, `tools/tadabur/augmentation/noise_pool.json`.

## Data: the 24 + 8 mix

- **Mix:** `tlog=24,tadabur=8` in `run_config.json`. `training.distill_mixed` asserts every batch is
  exactly 24 tlog + 8 Tadabur windows; the smoke gate re-read it from the batch accounting
  (`every batch {'batch_tlog': 24, 'batch_tadabur': 8}`).
- **tlog:** `tlog.training`, the S8 row-set use (**426,825 rows**, clean and unclean), read through
  `tlog/training_schedule.json` — seed `round2-tlog-shards-2026-10`, K = 8 shards interleaved, 739
  shard files in the frozen permutation (407 clean, 332 unclean). The schedule's SHA-256 is
  `8669ceca5ec67215a5c0147e97c0f74c97e5b20d29c8b1446cd8212feb8e134d`, the freeze's pin (#224); the
  file on cuda-dev hashes to the same digest.
- **Launch assertions:** `check_row_set_disjoint` (never `check_disjoint`, which refuses tlog by
  design) printed `tlog.training: 426825 rows disjoint from every evaluation use`; the Tadabur use
  was re-derived and checked: `consonant2.training: 211 shards, 208,198 admitted rows -- matches the
  committed use`. The trainer refuses a held-out shard before it downloads anything.
- **Tadabur:** `consonant2.training`, a shard use with **membership `exact`** on the control's 211
  shards, `shared_baseline` false — the same frame, order, seeds and recipe record
  (`tools/tadabur/exposure/consonant2.training.shards.json`).
- **Never read:** Tarteel v1, RetaSy, qdat, qdat_bench, QUD, `decode_evalset`, the sealed panel,
  Ship A held-out, any truth site, the tlog hold-out, and the tlog look/confirm tranches. None is
  decoded, graded or streamed by this run; the only tlog rows read are `tlog.training`'s.

What the run actually read, from `train.log`'s own accounting lines:

| stream | rows | clips | windows | real audio |
|---|---|---|---|---|
| tlog, clean | 49,784 | 49,784 | 156,205 | 448,766.7 s (124.7 h) |
| tlog, unclean | 24,206 | 24,206 | 84,383 | 240,859.476 s (66.9 h) |
| **tlog, both splits** | **73,990** | **73,990** | **240,588** | **689,626.176 s (191.6 h)** |
| **Tadabur** | 16,456 | 16,456 | **80,535** | 221,395.550 s (61.5 h) |

- **tlog consumed 73,990 rows over the whole planned prefix**: 130 of 130 planned shard files (76
  clean, 54 unclean), with no duplicate `(split, shard_file, row)` key in
  `tlog_consumed.jsonl.gz`. The pre-registration planned 73,829 rows / 240,002 windows; the run read
  161 more rows and 586 more windows (clean 49,784 vs 49,664 planned, unclean 24,206 vs 24,165).
- **Tadabur consumed 16,456 of the 208,198 admitted rows (7.9%)** — 16 shards completed and the 17th
  in progress, or 80,535 windows at 8 windows a step.
- The counters are the loaders' emitted totals: 10,000 steps trained 240,000 tlog and 80,000 Tadabur
  windows, so the small excess above those is windows emitted ahead of the final step.

## Optimisation

As #178 and the control: batch 32, lr 5e-5, warmup 500, WSD with `--cooldown-steps 3000`, EMA 0.999,
bf16 autocast, 10,000 steps, **seed 2026** (it orders the shards; the window shuffle derives seed
2027), `--num-workers 1`, `--save-every 2000`, `--log-every 50`, `--eval-every 2000`,
`--shuffle-buffer 512`, weight decay 0.01, max grad norm 5.0, grad accumulation 1.

**The candidate is the final step.** Validation is logged and never selects a checkpoint.

## Validation (monitoring only, and not an evaluation set)

`--audio-root` is `$ROOT/val_clips`: 9 clips staged at launch from **shard 72, the last of the
frame's seed-2026 permutation** (recorded position 210, the last of 211); its 2% hash split gives the
logged val windows. The trainer's in-training evaluation (`--eval-batches 40`) is monitoring only: it
selects nothing.

Shard 72 is inside the 211, so the step budget is what keeps the clips out of training: 10,000 steps
at 8 Tadabur windows each complete about 16 shards, far short of the 210 before shard 72. Because
that is arithmetic and not a property of the stream, it is **asserted after the run**:
`assert_monitor` reads `train.log`'s shard-completion lines and, if the run entered shard 72, fails
the chain and refuses to print a report. The launch assertion recorded the shard and position in
`logs/val_shard.txt` (`72`, `210`). **The finished run passed it:**

```
[monitor] 16 shards completed, monitor shard 72 entered at 210: never streamed
```

The val numbers are a health signal on training-set audio, not a generalisation estimate.

**No evaluation set was decoded.** The chain has no decode stage; the only audio it opened is the
registered training streams (`tlog.training`, `consonant2.training`) and the 9 staged monitoring
clips cut from shard 72. The launch assertions above and `assert_monitor` are the evidence.

## Launch

- **Script:** `scripts/issue-228-train.sh launch|check|report`. `launch` refuses while
  `/root/scratch/gpu.lock` is held, below `MIN_FREE_GB` on `/root/scratch`, when the run directory
  already exists, and when the freeze record is not committed in HEAD (`freeze: 4 artefacts committed
  in HEAD`); it archives HEAD to `/root/scratch/issue-228/repo`, stages the validation clips, and
  starts `run` detached under `flock /root/scratch/gpu.lock`.
- **Smoke first:** 700 steps with an evaluation every 150, so the staged validation clips and both
  splits are exercised before the gate. The gate requires throughput ≥ 0.6 steps/s, finite KL for
  both sources on every logged step, every batch exactly the pre-registered 24/8 mix, both splits
  consumed, at most 8 tlog shards on disk, the log's `[init] warm-started from … weights=ema` line,
  and a saved `ema_state`. The launch also refuses to stage when no staged clip falls in the val
  split. Anything else stops the chain.
- **Done file:** `run` writes `logs/train.exit` and then `logs/train.done` on every path (early
  failure included); a budget stop writes exit 2 and a `budget-exceeded` done-file. `check` reads
  those, never a log tail. `BUDGET` is 6 h; the run used 2.8 h.
- **Launch commit:** `a9ba80c34a2e8aee546c39c274cb9f829050dabb` (`Freeze round 2: pre-registration,
  readout pin, look-side freeze record`), archived at `2026-10-29T10:32:19Z`; `logs/launch.txt`
  records both.
- **Software:** conda `hifzguide` on cuda-dev, unchanged from #178 and the control — Python 3.11.15,
  torch 2.11.0+cu128, transformers 5.13.0, ffmpeg 8.1.2; `OMP_NUM_THREADS=1`.

## Smoke run (700 steps)

| | candidate |
|---|---|
| steps/s, steps 25-700 | **1.03** (gate threshold 0.6) |
| KL at the gate | range 0.370-0.727, last 0.410 |
| mix accounting | 24 tlog + 8 Tadabur per batch; splits `['clean', 'unclean']`; max 8 tlog shards on disk |
| averaged weights saved | yes — `ema_state` beside the live `student` at step 700 |
| val records (non-empty) | 5 evaluations (steps 150, 300, 450, 600, 700); overall agreement 0.952 → 0.9493 |
| warm start line (`weights=ema`) | present (see the gate lines below) |
| smoke rows read | tlog clean 5,977 rows / 16,519 windows; unclean 241 / 864; Tadabur 1,262 clips / 6,135 windows |

The gate's lines, verbatim from `logs/events`:

```
[gate] 1.03 steps/s over steps 25-700, KL 0.370-0.727, last 0.410
[gate] every batch {'batch_tlog': 24, 'batch_tadabur': 8}; splits consumed: ['clean', 'unclean']; max 8 tlog shards on disk
[gate] [init] warm-started from /root/scratch/issue-178/repo/tools/runs/consonant1_candidate/checkpoint.pt (its step 10000, weights=ema); fresh optimiser and schedule
[gate] averaged weights saved beside the live ones at step 700
```

## Full run (10,000 steps)

| | candidate |
|---|---|
| launched (UTC) / wall time | `2026-10-29T10:32:19Z`; smoke `10:33:38Z`-`10:46:53Z`; full arm `10:46:53Z` → `13:35:22Z`, **2.81 h** (`[done]` logs 2.80 h) |
| steps/s | **1.00** over steps 50-10000 |
| KL, first / last / range | 0.7075 / 0.3888 / 0.2966-0.7150 |
| pre-clip gradient norm, max | 2.06 (none clipped) |
| final val (monitor only) | overall 0.956, confirmed 0.947, nonblank 0.960, blank-collapse margin −0.020, teacher/student blank ratio 0.690 / 0.669 |
| **checkpoint SHA-256, live and EMA** | `893523710186cb16854dd62a258331644e2284abbee8b2cb54e8ab9d32dbea58` — one file holds both (`student` live, `ema_state` averaged; its `step` is 10,000) |
| shards consumed | **17 of 211 entered** (16 completed, stopped inside the 17th — shard 378); 194 never entered; monitor shard 72 at position 210 not entered |
| minimum free disk / peak VRAM | 8.94 GB (8.32 GiB) free, from 18.16 GB at the smoke start; the run's `$ROOT` (archived repo, `runs/`, val clips, tlog cache) peaked at 7.39 GB (6.89 GiB) / 11.05 GiB (preflight probe; no in-loop VRAM sampler) |

`logs/df.log` also holds one pre-run sample (`08:36:41Z`) from an earlier launch that stopped before
the smoke (`run.out` empty, nothing in `logs/events`); the budget readings above start at the smoke.

The last steps and the final evaluation, verbatim from `logs/train.log`:

```
[ 10000/10000] loss 0.3888 kl 0.3888 feat    off ctc    off hard    off cos   off conf-agree 0.921 blank-margin -0.022 |g| 1.1 | tlog-kl 0.4649 tadabur-kl 0.1588 tlog-blank 0.616 tadabur-blank 0.629
[eval 10000] {'overall_agreement': 0.9556, 'confirmed_agreement': 0.9467, 'nonblank_agreement': 0.9599, 'teacher_blank_ratio': 0.6898, 'student_blank_ratio': 0.6693, 'blank_collapse_margin': -0.0204, 'target_prob': 0.9164, 'target_rank': 1.05, 'blank_prob': 0.0585, 'top5_agreement': 1.0, 'prob_margin': -0.858}
[done] 10000 steps in 2.80h -- final {'overall_agreement': 0.9556, 'confirmed_agreement': 0.9467, 'nonblank_agreement': 0.9599, 'teacher_blank_ratio': 0.6898, 'student_blank_ratio': 0.6693, 'blank_collapse_margin': -0.0204, 'target_prob': 0.9164, 'target_rank': 1.05, 'blank_prob': 0.0585, 'top5_agreement': 1.0, 'prob_margin': -0.858}
```

## The frame manifest

Written into the run directory before the first step by `manifest`, from the spec `assert_stream`
verified (`runs/consonant2_candidate/frame_manifest.json`, SHA-256
`f4e524c0363c1aaf5f2c3d80caf585dc90091f0f31634872f5df7d63c9919c62`). **The Tadabur substream
replayed the control's frame exactly**, and the launch printed the two digests side by side with the
control's:

```
frame manifest: 211 shards, shards_sha256 4bd00c2ba5d0979bcc017546e3beeea70807dad443b125fa293adfe7f5f5d7d8, stream_order_sha256 7f0f1533618d0fc4e32e1fa34e20c4d1d22f4995779ad816a58e9d54193674fe, stream-order seed 2026, window-shuffle seed 2027, recipe aug94-mixture-v2 -> /root/scratch/issue-228/runs/consonant2_candidate/frame_manifest.json
```

Those are the control's `shards_sha256` and `stream_order_sha256`
([`consonant-round2-control.md`](consonant-round2-control.md)), so the candidate's 211 shards, their
seed-2026 order, both seeds and the recipe record are byte-identical to the frozen frame. The
manifest's load-bearing fields:

- `shards` / `shards_sha256`: the 211-shard frame, membership digest over the sorted list.
- `stream_order` / `stream_order_sha256`: the seed-2026 order the run streamed; its first 16 entries
  were completed and the 17th (shard 378) was in progress at step 10,000.
- `shard_order_seed` 2026 and `window_shuffle_seed` 2027: the two RNG seeds.
- `augmentation`: the `aug94-mixture-v2` recipe record, keyed per source row.
- `exposure_use`: `consonant2.training`, the registration the launch checked.

## Disclosures

The control arm's disclosures apply to this arm too, quoted from
[`consonant-round2-control.md`](consonant-round2-control.md):

> **There is no seed replicate.** One seed fixes one shard order and one augmentation draw, not a
> distribution. The control and the candidate are one realisation each, so a candidate − control
> difference **below about 1-2 per 1,000 graded words is not interpretable**; nothing in this run
> estimates the arm's own variance.
>
> **"Same seed" gives no pairing once the data streams differ.** Seed 2026 pairs the schedule
> shape — step count, LR, EMA and the driver of the shard permutation — but the control streams
> 211 Tadabur shards while the candidate adds a tlog substream. The samples per step differ, so a
> per-step or per-batch difference is not a paired comparison; only arm-level readings on a
> shared evaluation set can be compared, and those come from the round's readout, not from this
> note.

Augmentation draws are keyed by `(seed, shard, row)`, so shared Tadabur rows keep theirs. The
control's third disclosure stands unchanged: its meaning is "the candidate's schedule without the
new data", and the control trained before the freeze as the owner-approved #223 exception while the
candidate trained from the freeze commit. If the schedule ever changes, the control becomes
historical context, not attribution.

**Coverage, from the pre-registration.** "Clean rows cover 57 of 114 surahs. This is a seeded
shard-cluster sample, not row-uniform sampling and not all-surah coverage at every prefix. The 57
absent clean surahs hold 28.4% of clean pool rows; unclean rows carry no label, so 57 is not a
measurement of total surah coverage. K = 16 / 24 / 32 reach only 57 / 58 / 60 clean surahs." The
run read the whole planned 130-shard prefix, so its coverage is the frozen prefix's, not the
pool's.

## Acceptance checklist

- [x] Launch commit `a9ba80c`, archived `2026-10-29T10:32:19Z`; config and the 24 + 8 mix recorded
      from `run_config.json` and the smoke gate.
- [x] Warm start's init SHA-256 (`0576789…14a2`) recorded and recomputed; the `[init] … weights=ema`
      line and the saved `ema_state` prove it.
- [x] Smoke gate lines and the final eval line quoted verbatim.
- [x] Live and EMA checkpoint SHA-256 `893523710186cb16854dd62a258331644e2284abbee8b2cb54e8ab9d32dbea58`
      recorded; the file is copied to
      `/Volumes/NetBackup/HifzGuide/checkpoints/consonant2-candidate-issue228` with a matching
      SHA-256 and listed in that folder's README.
- [x] tlog schedule SHA-256 `8669ceca…134d`; per-split rows, clips, windows and seconds read quoted
      (clean 49,784 / 156,205 / 448,766.7 s; unclean 24,206 / 84,383 / 240,859.476 s; Tadabur
      16,456 clips / 80,535 windows / 221,395.550 s).
- [x] The Tadabur substream matched the control's `shards_sha256` and `stream_order_sha256`.
- [x] Peak disk (7.39 GB run directory; 8.94 GB free at its minimum) and the preflight VRAM probe
      (11.05 GiB) recorded.
- [x] No evaluation set decoded or graded: the chain decodes nothing, `check_row_set_disjoint`
      passed, and `assert_monitor` passed — 16 shards completed, monitor shard 72 never entered.
- [x] Exposure registered (`consonant2.training`, membership `exact`, 211 shards) and re-checked at
      launch; the exposure README row links this note.

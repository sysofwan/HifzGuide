# #94's augmentation experiment: run note

**Status (2026-10-16): both full arms are running on cuda-dev, the candidate first.** Their
exposure uses are registered. The recipe is the 2026-10-15 rewrite on #94, which supersedes the issue body. The
readout belongs to #95, and the bars were pre-registered on #128 before any candidate existed.
This note records what each arm trains on, under which configuration, and where its outputs
are. It contains no results. Checkpoints stay on cuda-dev.

## The two arms

| arm | student input | teacher input | exposure use |
|---|---|---|---|
| **candidate** | augmented under `aug94-mixture-v1` | clean | `aug.training` |
| **control** | clean (one tensor with the teacher's) | clean | `aug.control` |

Both arms continue `h448` with the existing frame-weighted KL. Everything else is the same in
both: the data stream, the shuffle, the schedule and the seed.

### Code

- `tools/training/augmentation.py`: the mixture, the draws, the training noise pool and the
  recipe record.
- `tools/training/distill_stream.py`: `StreamingWindowDataset(augment=...)` yields
  `(teacher, student)` feature pairs.
- `tools/training/distill_train.py`: `forward_step(teacher_features, student_features, ...)`,
  plus the flags `--augment`, `--noise-pool` and `--exposure-use`.

**Augmentation happens per clip, before windowing.** One condition is drawn per clip, and the
whole clip is perturbed. The clean clip and the augmented clip are then cut at the same
`window_starts`. Every condition keeps the clip's length and timing within 2 samples, under 1%
of a 20 ms frame. The test `training/test_augmentation.py` checks this per condition, at each
end of its range, on every verification probe of `training.perturbation`.

**The mixture**, with no stacking and no gain jitter:

| condition | share | parameters |
|---|---|---|
| clean | 0.30 | — |
| noise | 0.35 | SNR ~ U(3, 20) dB; one pool recording, uniform offset, looped |
| reverb | 0.15 | RT60 ~ U(0.2, 1.0) s, DRR 3 dB (`perturbation.synthetic_rir`) |
| 8 kHz band limit | 0.10 | down to 8 kHz and back |
| Opus | 0.10 | voip, 4 kHz cutoff, 20 ms frames, at 8, 12 or 16 kbit/s |

Each Opus configuration has its own calibrated lag. With ffmpeg 8.1.2 on cuda-dev:

| configuration | lag (samples) |
|---|---|
| 8k | 1.586 |
| 12k | 1.591 |
| 16k | 1.586 |

These match the table committed by #159.

**Draws.** Each clip's draws come from `random.Random(f"{seed + 2}:{shard}:{row}")`. `row` is
the clip's parquet row, numbered before the exposure exclusion. This stream is separate from
the shard permutation (`seed`) and from the shuffle buffer (`seed + 1`). The teacher therefore
reads the same windows in the same order in both arms. In the smoke runs, the teacher's blank
ratio matched to four places at every logged step.

**Resume guard.** The run's config carries the recipe as `TrainConfig.augment`
(`python -m training.augmentation record`). This record includes the committed pool's SHA-256.
`augment`, `exposure_use` and `shuffle_buffer` are resume-critical, so a resume under another
recipe, pool, arm or buffer is refused. A checkpoint written before these fields existed is
read as the clean run it was (`augment={}`, `exposure_use=""`, `shuffle_buffer=1024`), never
skipped. `--init-from` is a fresh run, not a resume, so it reads only the weights.

## Data

- **Shards:** exactly `h448`'s, from `training.decode_evalset.training_shard_spec()`. That is
  345 shards: `21-38,40-57,59-76,78-95,97-114,116-133,135-152,154-171,173-190,192-209,211-228,230-247,249-266,268-285,287-304,306-323,325-342,344-361,363-380,382-384`.
  `distill_stream.held_out_shards` refuses the rest.
- **Exposure:** the shard uses `aug.training` and `aug.control` are declared in
  `tadabur.exposure` and registered when the arms launched (see Exposure registration). Both are PRD training uses, so
  `tadabur.exposure.RowExclusion` drops two kinds of row before any audio is decoded. The same
  rows are dropped in both arms, and every kept row keeps its parquet row number.
  - **Excluded reciters** (acceptance rules §6): the sealed panel's 79 reciters, and the bias
    score half's (none yet).
  - **Rows tied to an evaluation recording** (`exposure.TRAINING_BARRED_USES`): a row that is,
    or that the registry ties to as an alias or a probable copy, a recording of:
    - `decode_evalset` dev, test or the legacy sample;
    - the sealed panel;
    - a truth site or a human-label fixture;
    - either Ship A half.
- **Not in these shards, but linked by copies:** none of these uses has its own recording in
  `h448`'s shards. The copy screen does tie shard rows to them, though. One example is
  `(322, 602)`, a probable copy of the `truth_site.waqf_boundary` recording `(16, 714)`.
  `h448` trained on all of these rows, and its 299 dev copies are why #95's Tadabur guard
  reads only the 671 dev clips that have none. The arms train on none of them.
- **What is dropped**, counted from the full shard index
  (`/root/scratch/issue-89/stage/full_index.jsonl`, SHA-256 `a20ad072…`). In all, 3,646 of
  the 344,450 rows are dropped (1.06%). No row falls in both categories.

  | reason | rows |
  |---|---|
  | panel reciter | 1,988 |
  | `decode_evalset.dev` | 642 |
  | `decode_evalset.test` | 534 |
  | `decode_evalset.legacy_stratified` | 363 |
  | `human_label.p35_fixture` | 74 |
  | `truth_site.p35_fixture` | 50 |
  | `truth_site.waqf_boundary` | 31 |
  | `human_label.waqf_events` | 31 |
  | `human_label.tashkeel_counterfactual` | 12 |
  | `human_label.reject_reread` | 5 |
  | `human_label.reject_bleed` | 1 |
  | sealed panel, Ship A halves | 0 |
  | **total** | **3,646** |

  A row tied to two uses counts under both, so the recording rows sum to more than the
  1,658 distinct rows. The running count is logged as each shard finishes
  (`[data] exposure exclusion through shard …`).
- **Never read:** muaalem-annotated-v3 and `nonpro.evaluation_only`. Neither is Tadabur
  audio, so the registry ties no shard row to them. `nonpro.evaluation_only` holds app users'
  and amateurs' recordings. #156's overlap sample compared 59 of its clips against
  EveryAyah's professional recordings and found no match. No audio check against Tadabur
  exists.
- **Seed:** 9494, the same in both arms (`h448` used 1234). The shuffle buffer is 512, half
  `h448`'s 1024, because a buffered entry is now a window pair.
- **Validation (monitoring only):** `--audio-root` is a directory of symlinks to the 970
  in-population dev clips of `decode_evalset` (`/root/scratch/issue-94/dev_clips`). Its 2%
  hash split gives 100 clean windows. The checkpoint rule is fixed (the final step), so
  nothing is selected on them.

## The training noise pool

- **Source:** `obadx/freesound-commercial-50k-noise-only` at `a5f48ab8`, shards
  `data/train-00026-of-00104.parquet` and `data/train-00078-of-00104.parquet`. D2 read 00000
  and 00052.
- **Filtering:** the same eligibility rule as D2's (at least 1 s long, finite, above
  -80 dBFS). Any `freesound_id` that is in D2's pool is then dropped.
- **Result:** 458 recordings (4.51 h). Left out: 71 ineligible, 0 undecodable, 0 in D2's pool.
- **Licences:**
  - CC0-1.0: 279
  - CC-BY-3.0: 87
  - CC-BY-4.0: 86
  - CC-Sampling+: 6
- **Provenance:** committed in `tools/tadabur/augmentation/noise_pool.json` (SHA-256
  `16c099d1…`). It lists each recording's id, licence, uploader, length and sample checksum.
- **Audio:** only on cuda-dev, at `/root/scratch/issue-94/noise_pool.npz` (1.04 GB, SHA-256
  `d2978d0f…`). `training.augmentation.load_training_pool` checks it against the committed
  record and refuses any overlap with D2's pool. The two shards were deleted once read.
- **Rebuild:** `python -m training.augmentation noise-pool --cache-dir <scratch> --out <npz>`.

## Configuration (both arms)

- **Initialisation:** `--init-from /root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt`.
  This loads the live weights, read only (SHA-256 `eadc3f75…`).
- **Model and optimisation:** preset `h448`, batch 32, lr 5e-5 with warmup 500, then WSD with
  `--cooldown-steps 3000`.
  - weight decay 0.01, clip 5.0
  - T 2, nonblank 3.0, confirm 2.0, KL only
  - EMA 0.999, bf16 autocast
- **Checkpointing and logging:** `--save-every 2000`, `--num-workers 1`, `--log-every 50`,
  `--eval-every 2000`.
- **Software:** torch 2.11.0+cu128, transformers 5.13.0, ffmpeg 8.1.2 with libopus.

Each arm runs from `/root/scratch/issue-94/run_arm.sh <aug|control> <out-dir> <steps> [args]`,
under `flock /root/scratch/gpu.lock`, with `OMP_NUM_THREADS=1` and the conda env `hifzguide`.
The code is a copy of this branch in `/root/scratch/issue-94/repo`. The script runs:

```bash
python -m training.distill_train --preset h448 --stream-shards "$(python -m training.decode_evalset --print-training-shards)" \
  --audio-root /root/scratch/issue-94/dev_clips --out-dir <out-dir> --steps <steps> --batch-size 32 \
  --learning-rate 5e-5 --warmup-steps 500 --cooldown-steps 3000 --temperature 2.0 --nonblank-weight 3.0 \
  --confirm-weight 2.0 --ema-decay 0.999 --seed 9494 --save-every 2000 --log-every 50 --eval-every 2000 \
  --num-workers 1 --init-from /root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt \
  # candidate: --exposure-use aug.training --augment --noise-pool /root/scratch/issue-94/noise_pool.npz
  # control:   --exposure-use aug.control
```

## Smoke runs (300 steps per arm, 2026-10-16)

Both smoke runs used the full configuration above with `--steps 300`. Over the first 300 steps
the learning rate is the same as in a 10k run, because warmup lasts 500 steps.

| | candidate | control |
|---|---|---|
| steps/s, steps 50-300 | **0.87** | **0.81** |
| wall time for 300 steps (with startup, first shard, final save) | 0.11 h | 0.11 h |
| KL loss at steps 50 / 100 / 150 / 200 / 250 / 300 | 0.212 / 0.359 / 0.252 / 0.202 / 0.243 / 0.187 | 0.164 / 0.236 / 0.148 / 0.160 / 0.171 / 0.146 |
| pre-clip gradient norm, range (clip 5.0) | 0.49-0.85, never clipped | 0.41-0.61, never clipped |
| train confirmed agreement, range | 0.922-0.950 | 0.931-0.959 |
| blank-collapse margin, range | -0.032 to -0.019 | -0.027 to -0.014 |
| val at step 300, overall / confirmed agreement (clean, 100 windows) | 0.9486 / 0.9466 | 0.9514 / 0.9490 |
| VRAM (`nvidia-smi`, peak) | 13.8 of 16.3 GB | 13.8 of 16.3 GB |
| preflight peak allocated | 11.05 GiB | 11.05 GiB |
| free disk, minimum (before, 33 GB) | 29 GB | 28 GB |
| available RAM, minimum (of 24 GB) | 14.0 GB | 15.2 GB |

**Health:** no NaN, no collapse warning and no clipping in either arm. The candidate's KL is
higher than the control's, as expected: its student hears perturbed audio and is scored
against the clean teacher.

**Throughput:** the rate dips about every 150 steps, at each shard boundary. One worker
downloads the next 2.5 GB shard while the GPU waits, as in `h448`'s run (0.96 steps/s without
augmentation).

**Disk:** a save writes a 1.87 GB checkpoint plus its temporary file, beside one streamed shard
per worker. 28 GB stayed free.

The smoke checkpoints were deleted. The logs and metrics are in
`/root/scratch/issue-94/{logs,repo/tools/runs/smoke_*}`.

**Step count: 10,000 per arm.** Both arms run at least 0.6 steps/s, the recipe's threshold for
cutting to 8k. At about 0.8 steps/s each arm takes about 3.5 h, so both arms take about 7 GPU-h
run back to back.

## Re-smoke after review (200 steps per arm, 2026-10-16)

GPT-6.1 Sol's review found two problems, and both are fixed:

- **P1:** a training row that is a probable copy of an evaluation recording was admitted.
  The recording exclusion above now drops such rows.
- **P2:** a pre-#94 checkpoint bypassed the new resume guards. A missing field is now read
  as its legacy value.

Both arms were then run again on the same configuration, with `--log-every 25`.

| | candidate | control |
|---|---|---|
| steps/s, steps 25-200 | 0.98 | 0.92 |
| KL loss, steps 25…200 | 0.444 / 0.220 / 0.343 / 0.218 / 0.250 / 0.262 / 0.252 / 0.192 | 0.165 / 0.171 / 0.155 / 0.185 / 0.175 / 0.199 / 0.189 / 0.147 |
| gradient norm, range | 0.48-1.27, never clipped | 0.37-0.86, never clipped |
| val at step 200, overall / confirmed | 0.9500 / 0.9481 | 0.9512 / 0.9500 |

- **The filter, on the first shard (169):** 8 of its 1,000 rows were dropped in each arm:
  3 for a panel reciter, 3 tied to `decode_evalset.dev` and 2 tied to
  `decode_evalset.legacy_stratified`. Recomputing from the full shard index gives the same
  counts.
- **Teacher batches:** the teacher's blank ratio is identical between the arms at all 8
  logged steps.
- **Resources:** peak VRAM 13.8 GB, at least 27 GB of disk free, at least 14.0 GB of RAM
  available.
- **Health:** no NaN and no warning. The checkpoints were deleted.

## Full runs (launched 2026-10-16T05:36:29Z)

Both arms run 10,000 steps, back to back, from one detached chain on cuda-dev. The candidate
runs first. The control starts when the candidate exits, so the two never hold the GPU
together. Each arm also takes `flock /root/scratch/gpu.lock` in `run_arm.sh`. The code is
`/root/scratch/issue-94/repo`, byte-identical to the reviewed commit `67749b9` in all 343
`.py`/`.json` files under `tools/training` and `tools/tadabur`; the rebase onto `main`
changed none of them. Registration ran from a separate copy (`/root/scratch/issue-94/register`),
so the code the control imports is unchanged.

**Scripts** in `/root/scratch/issue-94/`:

- **`full_runs.sh`** (the chain) runs, in order:

  ```bash
  ./run_arm.sh aug runs/aug94_candidate 10000 --num-workers 1 \
    --init-from /root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt > logs/aug94_candidate.log 2>&1
  ./run_arm.sh control runs/aug94_control 10000 --num-workers 1 \
    --init-from /root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt > logs/aug94_control.log 2>&1
  ```

  It appends each arm's start and exit (with exit code) to `logs/full_runs.events`.
- **`monitor_full.sh`** samples VRAM, disk and RAM every 60 s into `logs/monitor_full.log`,
  until the control exits.

**Processes:**

| process | PID |
|---|---|
| chain (`full_runs.sh`) | 1187415 |
| candidate `flock` | 1187418 |
| candidate trainer | 1187425 |
| candidate data worker | 1187685 |
| monitor | 1187427 |

The control's PIDs are assigned when it starts; `ps -eo pid,args | grep "[d]istill_train"`
lists them. Kill only by explicit PID.

**Where outputs land** (under `/root/scratch/issue-94/repo/tools/runs/`):

| arm | directory | log |
|---|---|---|
| candidate | `aug94_candidate/` | `/root/scratch/issue-94/logs/aug94_candidate.log` |
| control | `aug94_control/` | `/root/scratch/issue-94/logs/aug94_control.log` |

Each directory holds `checkpoint.pt` (live weights plus the EMA shadow), `metrics.jsonl` and
`run_config.json`. The checkpoint is saved every 2,000 steps and at the end; it is about
1.87 GB and stays on the box. **The candidate is fixed now: the final step, step 10,000.**
#95's look half chooses only live vs EMA weights.

**To resume** an interrupted arm, rerun its line from `full_runs.sh` with `--resume` added.
The resume guard refuses any change to the recipe, exposure use, buffer or schedule.

### Candidate: first 300 steps (health check)

| step | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| KL loss | 0.220 | 0.218 | 0.262 | 0.192 | 0.191 | 0.233 |
| pre-clip gradient norm | 0.64 | 0.67 | 0.95 | 0.48 | 0.68 | 0.97 |
| lr | 5.1e-6 | 1.0e-5 | 1.5e-5 | 2.0e-5 | 2.5e-5 | 3.0e-5 |

- **Health:** every loss is finite (no NaN), none was clipped, and there was no
  blank-collapse warning.
- **Throughput:** **0.89 steps/s** over steps 50-300.
- **Resources:** peak VRAM 13.8 of 16.3 GB, at least 29 GB of disk free, at least 13.9 GB
  of RAM available.
- **Exclusion, first two shards:** after shard 169, 8 of 1,000 rows were dropped; after
  shard 354, 19 of 2,000. Shard 169's breakdown is the same as in the re-smoke.

**Expected finish.** At 0.89 steps/s the candidate finishes about 3.2 h after launch, near
**08:55Z**. The control then starts. It ran at 0.81-0.92 steps/s in the smokes, so it should
finish about 3.1-3.5 h later, near **12:00-12:30Z** on 2026-10-16.

### Control: first 300 steps (check when it starts)

`logs/full_runs.events` shows `control start <time>` when the control starts. Its first
300 steps are then in `logs/aug94_control.log` and in
`repo/tools/runs/aug94_control/metrics.jsonl`. Check four things:

1. Every `total` is finite and there is no `[warn]`.
2. `grad_norm_preclip` stays well under 5.0.
3. Steps/s over steps 50-300, from `elapsed_s`, is at least 0.6.
4. The control's `teacher_blank_ratio` equals the candidate's at every logged step. They
   share one teacher stream, as in both smokes.

## Exposure registration

`aug.training` and `aug.control` are written as shard uses
(`tools/tadabur/exposure/aug.{training,control}.shards.json`). Each was written by:

```bash
python -m tadabur.exposure training-stream --use <aug.training|aug.control> \
  --index /root/scratch/issue-89/stage/full_index.jsonl --shards "$(python -m training.decode_evalset --print-training-shards)" \
  --membership exact --note "…"
```

- **Shards:** `h448`'s 345, with membership `exact` as for `h448.training`. At 10k steps an
  arm streams only a prefix of the seed-9494 shard permutation, so this overstates its
  exposure rather than understating it.
- **Not shared-baseline.**
- **`rows_per_reciter`:** counts only the admitted rows, 340,804 of 344,450, over 591
  reciters. `h448.training`'s 667 reciters minus the 76 panel reciters present in these
  shards gives 591. No other reciter loses all of its rows.
- **The two files** have identical shards and counts. Both pass
  `check_disjoint(<use>, sealed_panel)` by reciter.

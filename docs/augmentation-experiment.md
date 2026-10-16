# #94's augmentation experiment: run note

**Status (2026-10-16): plumbing, noise pool and smoke runs done; both full arms not yet
launched.** The recipe is the 2026-10-15 rewrite on #94, which supersedes the issue body. The
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
recipe, pool, arm or buffer is refused.

## Data

- **Shards:** exactly `h448`'s, from `training.decode_evalset.training_shard_spec()`. That is
  345 shards: `21-38,40-57,59-76,78-95,97-114,116-133,135-152,154-171,173-190,192-209,211-228,230-247,249-266,268-285,287-304,306-323,325-342,344-361,363-380,382-384`.
  `distill_stream.held_out_shards` refuses the rest.
- **Exposure:** the shard uses `aug.training` and `aug.control` are declared in
  `tadabur.exposure` and are empty until the arms launch.
  - Both are PRD training uses. The stream therefore drops the sealed panel's reciters (79,
    with 1,988 rows in these shards, 0.58%) and the bias score half's reciters (none yet)
    before decoding (acceptance rules §6). The same rows are dropped in both arms.
- **Never read:** `decode_evalset` dev or test, the sealed panel, the Ship A halves, the truth
  sites, `nonpro.evaluation_only` and muaalem-annotated-v3.
  - None of these has a recording in these shards. As for `h448`, 299 of the 970 dev clips
    have probable copies in them. #95's Tadabur guard reads the 671 dev clips that have none.
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

## Full runs

Not launched. The commands, PIDs, start times and output directories are added here when the
arms launch, with the first-300-step health check of each and the exposure registration
(`aug.training`, `aug.control`).

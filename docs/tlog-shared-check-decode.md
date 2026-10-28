# tlog round-2 decode: sharing the check features, and the re-projected budget (#213)

**Status:** measured on cuda-dev on 2026-10-28 at code `63d2938`, on the look gating selection
only. Decode text only: nothing here is graded. Raw records:
[`tools/tadabur/consonant2_readout/decodes/shared_parity.json`](../tools/tadabur/consonant2_readout/decodes/shared_parity.json)
and [`shared_bench.json`](../tools/tadabur/consonant2_readout/decodes/shared_bench.json).

## What was run

The 20 first gating shards (the parity selection, `tadabur.tlog_eval_decode.parity_shards`):
140 clips, 902.6 s of audio, decoded by `base`, `c95` and `candidate_ema` under the `checks`
store only, from the box clip cache (no shard read, no audio off the box). Every cell ran the
`decode` command directly (the launcher's worker line), with the selection named explicitly:

```bash
# From the repository root (the CLI is run with PYTHONPATH=tools, as the launcher's staged
# checkout does from tools/). The parity selection's shard names, then each process's share:
shards=$(PYTHONPATH=tools python -c "from tadabur.tlog_eval_decode import parity_shards; \
    print(' '.join(parity_shards()))")
PYTHONPATH=tools OMP_NUM_THREADS=1 python -m tadabur.tlog_eval_decode decode --parts-dir DIR \
    --cache-dir DIR --clip-cache DIR --models base c95 candidate_ema \
    --shard-files $shards --shard-index i --shard-count P
# The self-checking record of the shared parts against the committed decodes:
PYTHONPATH=tools python -m tadabur.tlog_eval_decode parity --parts-dir DIR \
    --out tools/tadabur/consonant2_readout/decodes/shared_parity.json --pr 216
```

The launcher itself decodes the whole gating selection and has no `--shard-files` path, so the
block above is the reproduction command for the parity selection.

## Parity: sharing changes no decode

The shared re-decode of those 20 shards is **byte-identical** to #212's committed look decodes
for all three models: 140 rows per model, 0 mismatches, and equal key-sorted digests
(`base 5e1f69cd…`, `c95 ad7b7c51…`, `candidate_ema 56606eb1…`). The per-shard part files were
also byte-identical between the shared and unshared runs, and a repeat of every cell reproduced
them again. The record proves the parts were decoded rather than copied: it carries the timing
lines' 140 items and 29.8 s of shared feature seconds, and a SHA-256 of each of the 60 part files.

## Where a check pass spends its time

Warm profile, five look shards after one warmup shard, one model per pass, `OMP_NUM_THREADS=1`
(279 checks per model, 87.6 s of audio):

| model | feature extraction (`device_pairs`) | forward pass (`_class_ids`) | check cuts (`app_checks`) |
|---|---|---|---|
| base (teacher) | 2.05 s — **15.8 %** | 10.88 s — 84.1 % | 0.01 s — 0.1 % |
| c95 | 2.03 s — **17.0 %** | 9.92 s — 82.9 % | 0.01 s — 0.1 % |
| candidate_ema | 2.05 s — **17.1 %** | 9.92 s — 82.8 % | 0.01 s — 0.1 % |

The forward pass dominates, at batch 1 on a single thread; the app's check logic is free. Feature
extraction is well under the issue's ~40 % stop threshold, so sharing features cannot be the
large lever the draft hoped for: for three models it can save at most `2/3` of one model's
feature time, i.e. about 10 % of the summed work.

## Measured throughput (same 20 shards)

| cell | wall | rows/h | audio s / wall s | audio s / wall s / model | summed checks | shared features |
|---|---|---|---|---|---|---|
| one unpinned process | 850 s | 593 | 1.06 | 0.35 | 832.6 s | — |
| `OMP_NUM_THREADS=1` × 3 | 189 s | 2,667 | 4.78 | 1.59 | 502.9 s | — |
| `OMP_NUM_THREADS=1` × 6 | 161 s | 3,130 | 5.61 | 1.87 | 599.3 s | — |
| × 6, shared features | **149 s** | **3,383** | **6.06** | **2.02** | 541.0 s | 29.9 s |

- Process parallelism: **5.7×** the one-unpinned-process wall with sharing, 5.3× without it.
- Sharing, at P = 6, over three script-timed repeats: 149, 150, 150 s shared against 161, 157, 157 s
  unshared — **4.5–7.5 % less wall (mean 5.4 %)**, and 4.7 % less summed work. The work saving is
  smaller than the 10 % upper bound because the teacher's forward is the bulk of the time; the
  wall saving is the number that matters and it costs nothing (same records, no extra passes).
- A first pass of the same four cells measured 841.1 / 184.3 / 148.0 / 139.2 s; its P = 6 pair
  (139.2 vs 148.0 s) sits just outside the later repeat band, so the later three repeats are the
  figure quoted above.

## Re-projected P2 + P3 budget

**Method.** Architect r26's one-unpinned-process cost `0.20·s + 0.0142·s²` per clip of `s`
seconds reproduces this sample (mean clip 6.45 s: predicted 789 s for three models, measured
833 s, +5.5 %). For the full look gating selection (2,853 clips, 25,185 s, mean 8.83 s) it gives
**2.28 h per model serial**. Confirm.1+2's ~14 h of audio keeps the pre-registered serial
**~6 h per model**. Each configuration's speed-up is the measured per-model wall ratio
(P = 3: 0.222, P = 6: 0.189, P = 6 shared: 0.175 of serial).

| step | models | serial (one unpinned) | P=3 × `OMP=1` | P=6 × `OMP=1` | P=6 × `OMP=1`, shared |
|---|---|---|---|---|---|
| P2: look gating | 2–4 | 4.6–9.1 h | 1.0–2.0 h | 0.9–1.7 h | 0.8–1.6 h |
| P3: confirm.1+2 gating | 3–4 | 18.0–24.0 h | 4.0–5.3 h | 3.4–4.5 h | 3.2–4.2 h |
| **P2 + P3** | | **22.6–33.1 h** | 5.0–7.4 h | 4.3–6.3 h | **4.0–5.8 h** |

The pre-registration's 25–35 h serial sits inside the 22.6–33.1 h the formula gives. The
projection is checked against the one run that exists: #212 decoded the whole look selection with
three models at P = 6 unshared in **1.18 h** (11:31:34 → 12:42:37 on cuda-dev); this table
projects 1.29 h for it, and 1.20 h through the shared path.

## Decision

- **Keep the launcher's `PROCESSES=6`, `THREADS=1`.** It is the best measured parallelism, ~5×
  the one-unpinned-process wall; the remaining budget is dominated by it, not by sharing.
- **Keep shared features on by default** (`SHARE_FEATURES=1`). It is bit-identical by
  construction and by the parity record, saves 4.5–7.5 % of the wall at P = 6 across the repeats
  (mean 5.4 %), and costs no additional forward passes. `SHARE_FEATURES=0` /
  `--no-share-features` remains the control.
- **The issue's step-1 gate is knowingly overridden, and this records it.** The issue said to stop
  after the benchmark if feature extraction is under ~40 % of a check pass; it measures
  15.8–17.1 %. The orchestrator's phase-2 instruction made the shared path the default anyway,
  because it is exact and free; the remaining budget is set by the forward pass and by process
  parallelism, not by sharing. The saving is small but consistent in direction across every
  repeat, so the default stands unless the owner rules otherwise.
- The round-2 decode budget moves from **~25–35 h serial to ~4–6 h** at the measured settings.
  The teacher's forward pass is the floor: only a larger batch or a cheaper teacher would move
  it further.

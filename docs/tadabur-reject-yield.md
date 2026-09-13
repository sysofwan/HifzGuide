# One-shard clean re-read yield

**19 clean re-reads per 1,000 clips. 27 shards reach 500. One shard costs 3m37s and
2.4 G of transfer, so the whole budget is ~98 minutes and fits on disk untouched.**

Deliverable 2 of HifzGuide #63, and the measurement
[`docs/tadabur-existing-inventory.md`](tadabur-existing-inventory.md) said was needed:
the surviving artifacts yield 0 clean re-reads structurally, so the reject pile has to be
streamed. This reports what one shard of it actually contains.

## How to reproduce

Shard 20 — the first shard the prior 20-shard run
(`audit_run/passing_subset_full.jsonl`, 20,202 clips) did not reach, so nothing here
overlaps material already on disk.

```bash
cd /root/repos/HifzGuide/tools

# The run (detached; ~3.5 min for one shard).
python -m tadabur.filter \
  --manifest          tadabur/reject_run/passing.jsonl \
  --rejects           tadabur/reject_run/rejects.jsonl \
  --reject-audio-out  tadabur/reject_run/reject_audio \
  --shards 20 --batch-size 4

# The numbers below.
python -m tadabur.reject_yield \
  --passing tadabur/reject_run/passing.jsonl \
  --rejects tadabur/reject_run/rejects.jsonl \
  --shards-run 1 --target 500 \
  --json tadabur/reject_run/yield.json
```

## Pass and reject rates

| | clips | of 1,000 |
| --- | --- | --- |
| Clips processed | 1,000 | |
| Passers | 925 | 92.5% |
| **Rejects** | **71** | **7.1%** |
| Skipped before the gate (over 50 s) | 4 | 0.4% |

Consistent with the prior run's 18,075 passers from 20,202 clips (89.5%).

## What the rejects failed on

Causes **overlap** — a clip can fail the ratio bar and carry a long insertion run — so
the shares sum past 100%. No clip failed on `min_query` or `no_alignment`; on real
recitation the gate always finds something to align.

| cause | clips | of 71 rejects |
| --- | --- | --- |
| `insertion_run` | 46 | 64.8% |
| `low_ratio` | 30 | 42.3% |
| `added_shadda` | 3 | 4.2% |

**The reject pile is mostly repeats.** Two thirds of everything the gate turns away
carries a repeated span of 5+ phonemes — which is the premise of ADR-0016 decision 2,
confirmed at a higher rate than it assumed.

## Clean re-reads

`max_insertion_run >= 5 and match_ratio >= 0.75 and not added_shadda`:

| | |
| --- | --- |
| **Clean re-reads** | **19** |
| …as a share of rejects | **26.8%** |
| …as a share of all clips | **1.9%** |
| Distinct reciters | 18 |
| Distinct ayat | 18 |
| Audio | 500.4 s (8.3 min) |
| Duration: min / median / max | 15.4 s / 22.5 s / 49.2 s |

Spread is as flat as it can be: 19 clips across 18 reciters, one reciter contributing 2
and every other contributing 1. No reciter or ayah dominates the bucket, so a corpus
built at this rate will not be tuning against one person's habits.

Their repeat lengths: 5 (×4), 6 (×2), 7 (×2), 8 (×2), 9 (×3), 11 (×4), 13, 18 — real
repeated words and phrases, not stumbles. None has a leading or trailing trim of 5+, so
this bucket carries no detectable neighbour-ayah bleed (ADR-0016 constraint 3); the
bleed shows up instead in the `low_ratio` rejects, where trims of 22–28 phonemes are
common.

## Second-guessing the thresholds

### `max_insertion_run` across all 71 rejects

| run | 0 | 1 | 5 | 6 | 7 | 8 | 9 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 20 | 21 | 22 | 26 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clips | 23 | 2 | 4 | 3 | 2 | 5 | 4 | 7 | 3 | 2 | 1 | 3 | 3 | 1 | 2 | 1 | 2 | 2 | 1 |

The distribution is **bimodal with an empty gap at 2–4**. Rejects either contain no
repeat at all (25 clips, run ≤ 1) or contain a substantial one (46 clips, run ≥ 5).
`MAX_INSERTION_RUN = 5` sits in dead space, not on a slope — moving it a little in
either direction would change nothing, which is the strongest thing that can be said
for a threshold.

### `match_ratio` across all 71 rejects

| bucket | 0.05–0.50 | 0.50–0.55 | 0.55–0.60 | 0.60–0.65 | 0.65–0.70 | **0.70–0.75** | **0.75–0.80** | 0.80–0.85 | 0.85–0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clips | 13 | 3 | 7 | 7 | 3 | **17** | **14** | 4 | 3 |

The 0.75 floor is **not** in dead space. It sits on the steepest part of the
distribution, with 17 clips immediately below it. Holding the run and shadda terms fixed
and moving only the ratio floor:

| floor | clean re-reads | reciters | audio |
| --- | --- | --- | --- |
| 0.80 | 6 | 6 | 3.1 min |
| **0.75** | **19** | **18** | **8.3 min** |
| 0.70 | 36 | 29 | 15.2 min |
| 0.65 | 39 | 30 | 16.1 min |

**0.70 would nearly double the yield and halve the shard budget** (14 shards instead of
27); 0.65 buys almost nothing more. Of the 27 rejects with a long run that the predicate
excludes, 26 are excluded by ratio alone and 1 by `added_shadda`.

This is a real choice and it is not made here. 0.75 is the conservative reading of
ADR-0016 decision 2 — it keeps the repeat as the *only* substantial divergence, so a word
that ends up graded `wrong` is a finding rather than a mispronunciation the oracle was
never told about. Whether the 0.70–0.75 band is still clean enough for a timing-free
oracle is answerable, but it is answerable by *listening to those 17 clips*, not by
reading this table. The predicate lives in one place
(`rejects.CLEAN_RE_READ_MIN_RATIO`), the sink stores every field it reads, and
`tadabur.reject_yield` recomputes from the stored rows — so re-deciding costs a re-run
of the report, not a re-run of the GPU.

## Wall clock and bandwidth

Measured on cuda-dev (RTX 5060 Ti, 16 G), one shard, `--batch-size 4`:

| stage | time | |
| --- | --- | --- |
| One-time setup (imports, references, model to GPU) | 8 s | paid once per run, not per shard |
| **Download** (2,375,752,988 B) | **37.2 s** | **63.8 MB/s (510 Mbit/s)** |
| **GPU** (1,000 clips decoded + gated) | **180 s** | 5.6 clips/s |
| **Per shard, marginal** | **217 s (3m37s)** | 10.9 MB/s end-to-end |

Download is 17% of the per-shard cost; the GPU is the bottleneck. Overlapping the two
would save at most that 17%, which is not worth the complexity — the run is detached
anyway.

## Recommended shard budget

At 19 clean re-reads per shard:

```
500 target / 19 per shard = 26.3  →  27 shards
27 shards × 217 s + 8 s setup = 5,872 s ≈ 98 minutes
27 shards × 2.376 GB          = 64 GB transferred
```

**Run shards 20–46** (27 shards, none of them touched by the prior run):

```bash
cd /root/repos/HifzGuide/tools
tmux new-session -d -s rejectrun 'python -m tadabur.filter \
  --manifest         tadabur/reject_run/passing.jsonl \
  --rejects          tadabur/reject_run/rejects.jsonl \
  --reject-audio-out tadabur/reject_run/reject_audio \
  --shards 20-46 --batch-size 4 --delete-shards \
  > tadabur/reject_run/filter.log 2>&1'
```

It resumes shard 20 as already done (`clips_processed // ROWS_PER_SHARD`), so pointing it
at the same manifest continues rather than repeats. If the ratio floor moves to 0.70
first, the same 500 needs **14 shards / ~51 minutes** instead.

Two caveats on the extrapolation, both pointing the same way: 19 clips is a small sample,
so the per-shard rate has real variance; and Tadabur shards are reciter-ordered, so a
27-shard run reaches fewer distinct reciters per clip than shard 20's 18-in-19 suggests.
Neither threatens the budget — both argue for checking the yield again at the halfway
mark, which is one `reject_yield` invocation against the partial sink.

## Disk

| | |
| --- | --- |
| Free before | 25 G |
| Free after | 23 G |
| `reject_run/` (19 WAVs + both manifests) | 31 M |
| Shard 20 blob, left in the HF cache | 2.4 G |

**Nothing was deleted and nothing was reclaimed.** This run deliberately omitted
`--delete-shards` so the measurement can be repeated against a warm cache. The 2 G
shortfall is that cached blob.

For the 27-shard run, `--delete-shards` is in the command above: peak usage is then one
shard in flight (~2.4 G transient) plus ~840 M of staged audio and manifests across the
whole run, against 23 G free. **No reclamation is needed** — the superseded
`segment_audio_v2/` (6.3 G) and `segment_audio_v4/` (6.1 G) generations noted in the
inventory can stay where they are.

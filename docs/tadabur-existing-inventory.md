# What the surviving Tadabur artifacts hold, for re-read mining

**Verdict: the existing artifacts yield 0 clean re-read clips. A re-stream is required,
and it is the critical path, not a "go wider" follow-on.**

Muraja [ADR-0016](https://github.com/sysofwan/Muraja/blob/main/docs/adr/0016-tadabur-corpus-for-reread-optimization.md)
needs natural re-reads to tune follow-along resync against. Before commissioning a
stream of the 938 GB source, this measures what a prior filtering run already left on
the fine-tune box. The answer is unambiguous and structural rather than marginal, so it
is worth stating plainly up front: **no passing clip can be a clean re-read, because
clearing the repeat bar is what "passing" means.**

## How to reproduce

Torch-free; ~1m37s on cuda-dev over 18,075 clips.

```bash
cd /root/repos/HifzGuide/tools
python -m tadabur.reread_inventory \
  --passing     tadabur/audit_run/passing_subset_full.jsonl \
  --clip-status tadabur/audit_run/seg_v21/manifest.jsonl.clip_status.jsonl \
  --clips-dir   tadabur/audit_run/clips_v2 \
  --json        tadabur/audit_run/reread_inventory.json
```

Every number below comes from that one command. `--clip-status` is the newest
segmentation generation (`seg_v21`, 27 Jul); it is byte-identical to `seg_v20`'s and to
`seg_v21/manifest_raw.jsonl.clip_status.jsonl` (same md5), so the choice of generation
does not move any figure here.

## The subset on disk

| | |
| --- | --- |
| Passing clips (`passing_subset_full.jsonl`) | 18,075 |
| …with a staged WAV in `clips_v2/` | **18,075 (100%)** |
| Distinct reciters | 503 |
| `clips_v2/` on disk | 6.7 G |

No audio is missing. Every manifest row is a usable clip; the question is only whether
it is the *right* clip.

## Two re-read signals, and why they disagree

`ClipStatus.re_reads` counts the seams `waqf_detect.segment_clip` cut a clip at — a
**word-space** signal from the segmenter. `max_insertion_run` is the length of the
longest repeated span in the whole-clip decode — a **phoneme-space** signal, and the one
the mining predicate keys on. The inventory reports both, because they disagree and the
disagreement is the finding.

### Segmenter-flagged re-reads

| `re_reads` | clips |
| --- | --- |
| 0 | 17,818 |
| 1 | 241 |
| 2 | 16 |
| **≥ 1** | **257** (1.42%) |

All 257 have audio. They span **129 reciters** and **226 distinct ayat**, totalling
**5,854 s (1 h 38 m)** of recitation. On breadth alone this looks like a usable corpus.

### The same 257 clips, re-gated in phoneme space

Re-running the `.balanced` gate offline on each clip's stored `predicted_phonemes`
recovers the `max_insertion_run` the original run computed and discarded:

| `max_insertion_run` | all passing clips | of the 257 flagged |
| --- | --- | --- |
| 0 | 17,630 | **172** |
| 1 | 331 | 6 |
| 2 | 21 | 7 |
| 3 | 31 | 26 |
| 4 | 62 | 46 |
| ≥ 5 | **0** | **0** |

**172 of the 257 (67%) contain no repeated span at all.** Those flags are word-edge
snapping wobble in the segmenter, not recitation the reciter repeated. The best material
present is the 46 clips with a 4-phoneme run — under one short word.

The `≥ 5` row is 0 by construction, not by luck: `scorer.MAX_INSERTION_RUN = 5` is a
gate *reject*, so any clip with a run that long is absent from this file by definition.
That is ADR-0016's constraint 1 measured rather than asserted.

### Clean re-reads: 0

The mining predicate — `max_insertion_run >= 5 and match_ratio >= 0.75 and not
added_shadda` — matches **0 of 18,075** clips. It cannot match any of them.

### The segmenter's own repeat verdicts

| `skip_reason` | clips |
| --- | --- |
| *(none)* | 17,767 |
| `phonetizer_unsupported` | 280 |
| `repeated_recitation` | 26 |
| `low_alignment` | 2 |

`repeated_recitation` is the segmenter giving up *because of* a repeat, so it is the
strongest re-read-adjacent signal in the file. All 26 have audio, across 22 reciters
(716 s). Re-gated, 25 of the 26 have `max_insertion_run == 0` and one has 1 — the
overlap is between adjacent *segments*, not a repeat the whole-clip decode can see.

The union of "flagged as a re-read" and "skipped for repeated recitation" is **283
clips**, below the issue's 300-clip bar even before quality is considered — and the
quality is the decisive part.

## Recommendation

**Stream.** The follow-on ordering in the issue is inverted by this measurement: the
reject sink is the critical path, not a widening pass.

- **0 clips** meet the predicate Muraja #127/#128/#129 consume, and no amount of
  re-segmenting the passing subset will produce one. The passing subset is the wrong
  population, not a small sample of the right one.
- The 257 flagged clips are **not worthless**. Their 1–4 phoneme runs are real sub-word
  stumbles, and all have audio, wide reciter spread and clean references. They are good
  enough to **build** the Muraja replay harness and the word-space oracle against — a
  scenario file can be cut from them today, in parallel with the stream — but they cannot
  **calibrate** `CommitAndTrimPass2`, because a 4-phoneme repeat does not provoke the
  backward tracking-position move the cycles-to-resync metric is defined over.
- Nothing here needs to be re-derived after the stream. `tadabur.reread_inventory` runs
  against any passing manifest + clip-status pair, and `tadabur.rejects` records the
  whole `GateResult`, so the yield report is the same arithmetic over the new sink.

## Disk before the stream

98 G volume, **25 G free**, `audit_run/` at 40 G. The largest occupants:

| path | size | |
| --- | --- | --- |
| `seg_v21/` | 16 G | current fine-tune generation — keep |
| `clips_v2/` | 6.7 G | the 18,075 staged clips this report measures — keep |
| `segment_audio_v2/` | 6.3 G | superseded worklist generation |
| `segment_audio_v4/` | 6.1 G | superseded worklist generation |
| `clips_full/`, `segment_audio_full/` | 3.5 G | superseded |
| `~/.cache/huggingface` | 4.5 G | |

A shard is ~2.4 G, so `--delete-shards` bounds a run at roughly one shard plus its
outputs and fits in 25 G without reclaiming anything. The yield report records what was
actually done.

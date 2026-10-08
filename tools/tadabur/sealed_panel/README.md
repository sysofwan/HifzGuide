# The sealed held-out panel (#89)

**Sealed.** Nothing scores this panel until the ship criterion (#97) does, once, with the
owner's authorization (acceptance rules §3). `tadabur.sealed_panel.open_for_scoring` is the
only way to it for scoring, and it raises `SealedPanelError` without the flag only #97
passes. A test fails if any other module imports the panel module, names its directory or
names that flag. Select, tune and diagnose on the mining pool or `decode_evalset` dev.

Code: [`tools/tadabur/sealed_panel.py`](../sealed_panel.py). Exposure rows:
[`../exposure/sealed_panel.jsonl`](../exposure/README.md).

| file | what it is |
|---|---|
| `staged_clips.jsonl` | provenance of every panel clip: shard, row, canonical reciter, length, checksum (`staged_audio` schema, use `sealed_panel`) |
| `frame.json` | rows per unseen shard by outcome, unseen-shard reciters each use barred, and per panel reciter its clips and its rows in `h448`'s training shards |
| `clips.jsonl` | one row per clip: word times, segmentation status, every segment's sample span, realized reference and word offsets (the mining pool's shape) |
| `teacher_decodes.json` | the base teacher's decode of every kept segment, with the decode fingerprint |
| `summary.json` | counts per shard and reciter, segmentation outcomes, and the reference-only capacity |

## The frame

Every row of the shards `h448` never trained on (0-20 and the strided reserve 39, 58, …,
381) whose **canonical reciter has no other use** in the exposure registry: no truth site,
no human label, no mining-pool clip, no `decode_evalset` record (dev, test or the legacy
stratified sample), no Muraja corpus clip. Used reciters leave whole; "a new salt" would
not make them fresh (§6). Then the pool's row bounds: 1.5-50 s and a phonetizable ayah.
**The panel is the whole frame** (no cap, no draw), so a site worklist mined from it later
records its own inclusion probabilities against `frame.json`.

Of the 581 reciters in those 40 shards, 492 are barred. The binding exclusion is
`decode_evalset`: 151 reciters were untouched by the labels and the pool (#83), but 59 of
them, and nearly all of their 2,466 clips, are `decode_evalset` reciters.

**The one overlap allowed is `h448.training`.** Its 345 shards hold 667 of Tadabur's 671
reciters, so no panel could avoid it: 84 of the 87 panel reciters have rows there (2,109
rows, median 19 per reciter). The panel takes **no recording** from those shards
(`check_disjoint(..., by="source")` holds). `h448_init`'s calibration windows and `h448`'s
validation windows came from the lost `clips_v2` corpus (shards 0-19); no gradient step
used them, but which clips they were cannot be recovered.

## Capacity

| | |
|---|---|
| clips / reciters | **216 / 87** (212 / 86 with a kept segment) |
| audio | 0.74 h, of which 0.66 h in 289 kept segments |
| clips per reciter | 43 reciters have 1 clip, 21 have 2; the largest has 13 |
| haraka carriers (fatha / damma / kasra) | 3,125 / 1,010 / 1,098 |
| reference geminates | 906 |
| mid-word sukun carriers | 598 |
| pair carriers | `ق↔ك` 502, `ح↔ه` 487, `ت↔ط` 362, `س↔ص` 247, `ذ↔ز` 201, `ذ↔ظ` 169, `ض↔ظ` 67 |

These count candidate sites in the realized references alone; no decode enters them, so
they size mining and score nothing. Real-mistake sites are a small fraction of carriers
(the base teacher showed 127 pair and 522 gemination events in the pool's 12,413 census clips), so at this size
the panel holds few: its human labelling is sized by #105 against this table.

## Rebuilding

From `tools/` on the GPU box (`--index` from `python -m tadabur.staged_audio index --shards
0-384`; the registry in `../exposure/` must be current first):

```bash
python -m tadabur.sealed_panel select --index stage/full_index.jsonl --out stage/panel.jsonl
python -m tadabur.sealed_panel stage --index stage/full_index.jsonl --selection stage/panel.jsonl \
    --audio-dir stage/panel_clips --shard-cache stage/hf_cache
python -m tadabur.resegment --registry tadabur/sealed_panel/staged_clips.jsonl \
    --use sealed_panel --audio-dir stage/panel_clips --out-dir stage/seg_panel
python -m tadabur.sealed_panel build --selection stage/panel.jsonl --seg-dir stage/seg_panel
```

The 2026-10-08 run staged 216 clips (83 MB of PCM_16 WAV) under
`/root/scratch/issue-89/stage/panel_clips/` on the GPU box; `stage` read 39 shards one at a
time and deleted each. `resegment` decodes only with the base teacher, so the panel's one
decode is the cache above.

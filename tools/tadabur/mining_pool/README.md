# The mining pool (#83)

A fixed set of whole Tadabur clips from shards the shipped student `h448` never trained
on, segmented and decoded once by the frozen base teacher: 2,000 drawn uniformly
(reciter-balanced) plus every clip of the same reciters where the base teacher showed one of
the rare events the listening session mines. It is the input to the haraka-gap diagnosis
(#85), the shaddah probe (#86) and the listening session's site mining (#87). The audio stays
on the GPU box. It is re-stageable, and verifiable, from
[`../staged_audio/clips.jsonl`](../staged_audio/README.md), whose rows carry each pool clip's
shard, row, canonical reciter id, length and checksum (`uses` contains `mining_pool`).

Code: [`tools/tadabur/mining_pool.py`](../mining_pool.py) (draw, census scan, manifest,
loader, capacity) and [`tools/tadabur/resegment.py`](../resegment.py) (segmentation and
decode).

| file | what it is |
|---|---|
| `selection.json` | the draw's rule, inputs, frame and census: salt, size, cap, shards, exclusions, events, stratum sizes |
| `clips.jsonl` | one row per clip: its strata, word times, segmentation status, and every segment's sample span, realized reference and word offsets |
| `base_decodes.json` | the base teacher's decode of every kept segment, with the decode fingerprint |
| `summary.json` | counts per stratum, shard and reciter, segmentation outcomes, and per-stratum capacity |

## The draw

- **Frame.** The strided reserve `decode_evalset.gate_eval_shards()` past the held-out block
  0-20: shards 39, 58, …, 381 (19 shards, 19,000 rows). Excluded: rows in the frozen
  `decode_evalset` (2,759; its dev half is the teacher-agreement guard, so no pool clip can
  leak into it through later training), clips outside 1.5-50 s (197), and the eight ayat
  the phonetizer cannot realize (21 clips). 16,023 clips from 500 reciters remain.
- **`uniform` (2,000 clips, 394 reciters).** Reciters are ranked by
  `sha256("issue-83-mining-pool-v1:reciter:<id>")` and clips within a reciter by
  `sha256("issue-83-mining-pool-v1:clip:<filename>")`. Up to 8 clips are taken per reciter,
  reciters in rank order, until there are 2,000. The cap keeps one prolific reciter (one holds
  ~800 reserve clips) from dominating a reciter-clustered interval. Taking reciters whole
  leaves the other 106 frame reciters, and every reciter outside the frame, unused by the
  pool, so the sealed panel (#89) still has reciters nothing here has touched.
- **`consonant_pair` and `geminate` (censuses).** On the 2,000 uniform clips the base teacher
  heard the other letter of a target pair at only 21 sites (none for `ذ↔ظ`) and left a
  geminate single at only 56, short of what #87 mines (~40 consonant and ~60 geminate sites).
  So every frame clip of the 394 drawn reciters (12,413 clips) was decoded whole by the base
  teacher, from exactly the samples its staged file would hold, against its whole-ayah
  realized reference. Every clip with a substitution of a target pair (the six soft pairs and
  `ذ↔ظ`) is in `consonant_pair`; every clip with a gemination mismatch is in `geminate`. The
  censuses are complete within that sub-frame: a stratum's population is its clip count and
  every weight is 1. A clip can be in several strata; `clips.jsonl` lists them.

| stratum | clips | in the scan |
|---|---|---|
| `uniform` | 2,000 | |
| `consonant_pair` | 122 | 127 pair sites: `ت↔ط` 4, `ح↔ه` 5, `ذ↔ز` 31, `ذ↔ظ` 4, `س↔ص` 16, `ض↔ظ` 32, `ق↔ك` 35 |
| `geminate` | 734 | 712 single-at-geminate, 67 double-at-single |
| **pool** | **2,698** | 394 reciters |

The scan's whole-clip counts select clips; the sites #87 mines are re-found on each kept
segment (`summary.json` → `capacity`), so the two counts differ slightly.

```bash
python -m tadabur.staged_audio index --shards <0-20 and the reserve> --out stage/shard_index.jsonl
python -m tadabur.mining_pool select --index stage/shard_index.jsonl \
    --evalset-manifest tadabur/gate_eval/manifest.json --out stage/pool_uniform.jsonl
python -m tadabur.mining_pool scan --index stage/shard_index.jsonl \
    --evalset-manifest tadabur/gate_eval/manifest.json --uniform stage/pool_uniform.jsonl \
    --out stage/scan.jsonl --shard-cache stage/hf_cache
python -m tadabur.mining_pool select --index stage/shard_index.jsonl \
    --evalset-manifest tadabur/gate_eval/manifest.json --scan stage/scan.jsonl \
    --out stage/pool_selection.jsonl
python -m tadabur.staged_audio stage ... --pool-selection stage/pool_selection.jsonl
python -m tadabur.resegment --registry tadabur/staged_audio/clips.jsonl --use mining_pool \
    --audio-dir stage/clips --out-dir stage/seg_pool
python -m tadabur.mining_pool build --selection stage/pool_selection.jsonl --seg-dir stage/seg_pool
```

## Segmentation and decode

`tadabur.resegment` runs today's segmentation (`tadabur.segment_score`: recitation VAD,
pause-to-word placement, drop rules) on the staged PCM_16 WAVs, with every decode through
`training.decoding.Decoder`: base teacher `obadx/muaalem-model-v3_2`, bf16 weights, whole
spans, batch size 1 (the fingerprint is in `base_decodes.json` and `summary.json`). References
come from `hafs_phonetizer.phonetize` (#100), so they end in the correct pausal form. Its native outputs
(`segment_manifest.jsonl`, `clip_status.jsonl`, `pause_attrib.jsonl`) stay on the GPU box in
`segment_score`'s formats; `clips.jsonl` here carries the same segmentation in one committed
file, and `mining_pool.load_manifest` checks it against the staged-clip registry.

A segment's `start_sample` / `end_sample` is exactly the slice that was decoded
(`segment_score.segment_sample_bounds`). A dropped segment (`kept: false`) keeps its span and
reference but has no decode.

## Capacity

`summary.json` → `capacity` counts, on kept segments, the candidate sites each stratum the
listening session mines (#87) holds. They come from the base decode against the realized
reference, so they size mining and say nothing about truth.

Whole pool (2,698 clips, 11.2 h, 4,387 segments of which 4,197 kept) against the uniform
stratum alone (2,000 clips):

| stratum (#87) | candidate sites in the whole pool | in `uniform` alone |
|---|---|---|
| fatha left empty / matched | 1,162 / 49,393 | 793 / 33,072 |
| damma left empty / matched | 513 / 15,441 | 302 / 10,179 |
| kasra left empty / matched | 436 / 18,567 | 239 / 12,335 |
| geminate decoded single / reference geminates | 308 / 17,170 | 56 / 10,980 |
| single consonant decoded double | 35 | 7 |
| mid-word consonant with no haraka (prescribed sukun) | 10,486 | 6,999 |
| `ذ↔ز` other letter heard / carriers | 28 / 2,965 | 3 / 1,983 |
| `ض↔ظ` | 28 / 1,181 | 7 / 733 |
| `ق↔ك` | 29 / 7,859 | 4 / 5,247 |
| `س↔ص` | 11 / 3,932 | 4 / 2,591 |
| `ت↔ط` | 3 / 6,192 | 2 / 4,144 |
| `ح↔ه` | 3 / 8,282 | 1 / 5,405 |
| `ذ↔ظ` | 2 / 2,590 | 0 / 1,772 |

#87's targets (50 per haraka left empty, 50 matched controls, ~60 geminates decoded single,
~40 consonant sites, a mid-word sukun stratum) are all covered. Per pair, `ت↔ط`, `ح↔ه` and
`ذ↔ظ` stay rare (2-3 sites each): 12,413 clips of the drawn reciters hold only 4-5 scan sites
of each, so no pool of this frame can support a per-pair claim for them. The truth-site
schema does not accept `ذ↔ظ` as a `mark` yet; #87 adds it when it writes those sites.

74 clips (2.7%) have no segments: quran-transcript cannot phonetize one of their references
(`phonetizer_unsupported`, as in `segment_score`). 28 are kept whole as a repeated
recitation; both are listed by `skip_reason` in `clips.jsonl`.

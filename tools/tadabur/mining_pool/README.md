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
| `selection.json` | the draw's rule, inputs and census: salt, size, cap, shards, exclusions, the scan's identity, events, stratum sizes |
| `frame.json` | the frame the draw was made from: eligible and excluded rows per shard (by reason), and per reciter its eligible clips, how many the uniform stratum took, and whether it was drawn |
| `clips.jsonl` | one row per clip: its strata and inclusion probability, word times, segmentation status, and every segment's sample span, realized reference and word offsets |
| `base_decodes.json` | the base teacher's decode of every kept segment, with the decode fingerprint |
| `h448_stream_decodes.json` | the shipped `h448`'s b=0 streaming decode of every clip (whole clip, fp32 weights, no bias), cut per kept segment by commit time, with the decode fingerprint and the checkpoint's SHA-256 (`tadabur.pool_stream`, #87) |
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
  geminate single at only 47, short of what #87 mines (~40 consonant and ~60 geminate sites).
  So every frame clip of the 394 drawn reciters (12,413 clips) was decoded whole by the base
  teacher, from exactly the samples its staged file would hold, against its whole-ayah
  realized reference. Every clip with a substitution of a target pair (the six soft pairs and
  `ذ↔ظ`) is in `consonant_pair`; every clip with a gemination mismatch, confirmed on the raw
  strings (a decode that keeps both consonants of a geminate but drops its haraka is not
  one), is in `geminate`. The censuses are complete **only within that sub-frame of the
  drawn reciters**: there a stratum's population is its clip count and inclusion is
  certain. A clip can be in several strata; `clips.jsonl` lists them.

| stratum | clips | in the scan |
|---|---|---|
| `uniform` | 2,000 | |
| `consonant_pair` | 122 | 127 pair sites: `ت↔ط` 4, `ح↔ه` 5, `ذ↔ز` 31, `ذ↔ظ` 4, `س↔ص` 16, `ض↔ظ` 32, `ق↔ك` 35 |
| `geminate` | 502 | 468 single-at-geminate, 54 double-at-single |
| **pool** | **2,508** | 394 reciters |

The scan's whole-clip counts select clips; the sites #87 mines are re-found on each kept
segment (`summary.json` → `capacity`), so the two counts differ slightly.

### Inclusion probabilities

Each row of `clips.jsonl` carries `inclusion_probability`, the clip's chance of being in the
pool **given the 394 drawn reciters**, counting the strata as a union:

- a clip in `consonant_pair` or `geminate` has probability 1, whatever its uniform chance;
- any other clip was drawn by the uniform stratum as `k` of its reciter's `n` eligible clips
  (a salted-hash order is a simple random sample of them), so its probability is `k / n`.
  For reciter 66 that is 8 / 689; a reciter with one eligible clip gives it probability 1;
  the last reciter drawn (183) gave 3 of its 5.

1,239 pool clips have probability 1 and 1,269 less (down to 0.0116). `frame.json` freezes
what the probabilities are computed from: eligible and excluded rows per shard (with the
reason), and per reciter its eligible clips, the clips the uniform stratum took, and whether
it was drawn. Inference from the pool is **restricted to the realized reciter allocation**:
the 394 drawn reciters with the within-reciter probabilities recorded here. The reciter
draw is not a simple random sample of the 500 eligible reciters with probability 394 / 500.
It stops at 2,000 *clips*, so how many reciters it takes depends on their unequal capped
clip counts (with eligible counts 1, 1 and 2 and a cap and target of 2, the three reciters'
inclusion probabilities over all hash orders are 1/2, 1/2 and 2/3). No probability is
recorded for reaching reciters outside the allocation.

### The scan's provenance

`scan` stores only each census clip's decode. Its identity (the census frame's clip names,
shards and rows, hashed, and the decode fingerprint) is written beside it when it starts; a
resumed scan refuses to append under another identity, and `select --scan` refuses a scan
made over another census frame. `select` turns the decodes into events itself and records
the identity and the phonetizer revision it used (`selection.json` → `census`).

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

Whole pool (2,508 clips, 10.2 h, 3,955 segments of which 3,785 kept) against the uniform
stratum alone (2,000 clips):

| stratum (#87) | candidate sites in the whole pool | in `uniform` alone |
|---|---|---|
| fatha left empty / matched | 1,055 / 44,834 | 793 / 33,072 |
| damma left empty / matched | 459 / 14,010 | 302 / 10,179 |
| kasra left empty / matched | 395 / 16,852 | 239 / 12,335 |
| geminate decoded single / reference geminates | 235 / 15,426 | 47 / 10,980 |
| single consonant decoded double | 35 | 7 |
| mid-word consonant with no haraka (prescribed sukun) | 9,400 | 6,999 |
| `ذ↔ز` other letter heard / carriers | 28 / 2,732 | 3 / 1,983 |
| `ض↔ظ` | 28 / 1,054 | 7 / 733 |
| `ق↔ك` | 28 / 6,970 | 4 / 5,247 |
| `س↔ص` | 11 / 3,565 | 4 / 2,591 |
| `ت↔ط` | 3 / 5,634 | 2 / 4,144 |
| `ح↔ه` | 3 / 7,433 | 1 / 5,405 |
| `ذ↔ظ` | 2 / 2,386 | 0 / 1,772 |

#87's targets (50 per haraka left empty, 50 matched controls, ~60 geminates decoded single,
~40 consonant sites, a mid-word sukun stratum) are all covered. Per pair, `ت↔ط`, `ح↔ه` and
`ذ↔ظ` stay rare (2-3 sites each): 12,413 clips of the drawn reciters hold only 4-5 scan sites
of each, so no pool of this frame can support a per-pair claim for them. The truth-site
schema accepts `ذ↔ظ` as a `mark` since #87.

69 clips (2.8%) have no segments: quran-transcript cannot phonetize one of their references
(`phonetizer_unsupported`, as in `segment_score`). 25 are kept whole as a repeated
recitation; both are listed by `skip_reason` in `clips.jsonl`.

# The mining pool (#83)

A fixed set of 2,000 whole Tadabur clips from shards the shipped student `h448` never
trained on, segmented and decoded once by the frozen base teacher. It is the input to the
haraka-gap diagnosis (#85), the shaddah probe (#86) and the listening session's site mining
(#87). The audio stays on the GPU box. It is re-stageable, and verifiable, from
[`../staged_audio/clips.jsonl`](../staged_audio/README.md), whose rows carry each pool clip's
shard, row, canonical reciter id, length and checksum (`uses` contains `mining_pool`).

Code: [`tools/tadabur/mining_pool.py`](../mining_pool.py) (draw, manifest, loader, capacity)
and [`tools/tadabur/resegment.py`](../resegment.py) (segmentation and decode).

| file | what it is |
|---|---|
| `selection.json` | the draw's rule, inputs and frame: salt, size, cap, shards, exclusions |
| `clips.jsonl` | one row per clip: word times, segmentation status, and every segment's sample span, realized reference and word offsets |
| `base_decodes.json` | the base teacher's decode of every kept segment, with the decode fingerprint |
| `summary.json` | counts per shard and reciter, segmentation outcomes, and per-stratum capacity |

## The draw

- **Frame.** The strided reserve `decode_evalset.gate_eval_shards()` past the held-out block
  0-20: shards 39, 58, …, 381 (19 shards, 19,000 rows). Excluded: rows in the frozen
  `decode_evalset` (its dev half is the teacher-agreement guard, so no pool clip can leak
  into it through later training), clips outside 1.5-50 s, and the eight ayat the
  phonetizer cannot realize.
- **Reciter-balanced.** Reciters are ranked by `sha256("issue-83-mining-pool-v1:reciter:<id>")`
  and clips within a reciter by `sha256("issue-83-mining-pool-v1:clip:<filename>")`. The pool
  takes up to 8 clips per reciter, reciters in rank order, until it holds 2,000. The cap keeps
  one prolific reciter (one holds ~800 reserve clips) from dominating a reciter-clustered
  interval. Taking reciters whole leaves every lower-ranked reciter unused by the pool, so the
  sealed panel (#89) still has reciters nothing here has touched.

```bash
python -m tadabur.staged_audio index --shards <0-20 and the reserve> --out stage/shard_index.jsonl
python -m tadabur.mining_pool select --index stage/shard_index.jsonl \
    --evalset-manifest tadabur/gate_eval/manifest.json --out stage/pool_selection.jsonl
python -m tadabur.staged_audio stage ... --pool-selection stage/pool_selection.jsonl
python -m tadabur.resegment --registry tadabur/staged_audio/clips.jsonl --use mining_pool \
    --audio-dir stage/clips --out-dir stage/seg_pool
python -m tadabur.mining_pool build --seg-dir stage/seg_pool
```

## Segmentation and decode

`tadabur.resegment` runs today's segmentation (`tadabur.segment_score`: recitation VAD,
pause-to-word placement, drop rules) on the staged PCM_16 WAVs, with every decode through
`training.decoding.Decoder`: base teacher `obadx/muaalem-model-v3_2`, bf16 weights, whole
spans, batch size 1 (the fingerprint is in `base_decodes.json` and `summary.json`). References
end in the correct pausal form (`pausal_taa_marbuta`, #79 / #100). Its native outputs
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

CAPACITY_PLACEHOLDER

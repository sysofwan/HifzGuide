# The sealed held-out panel (#89)

**Sealed.** Nothing scores this panel until the ship criterion (#97) does, once, with the
owner's authorization (acceptance rules §3). The seal is
[`tools/tadabur/panel_seal.py`](../panel_seal.py), enforced at the choke points every
reader goes through, outside an authorized block:

- **files:** `tadabur.audio.read_audio` and `read_audio_bytes`, the only ways the tools
  read an audio file (a lint test pins it), refuse a name carrying a panel clip id (the
  clip, a `__seg<n>` segment, a hash-prefixed copy) or a file with a panel checksum. The
  glob loader `discover_clips` checks every path it lists.
- **bytes:** `tadabur.audio.decode_to_mono_16k` refuses a panel clip's bytes, whatever path
  or buffer they came through.
- **streamed rows:** `shard_reader.iter_shard_rows` and `dataset_source.stream_rows` (the
  only row sources) keep a panel row's place and name but seal its audio, so reading its
  samples raises. every `except Exception` that skips corrupt inputs re-raises
  `SealedPanelError` first (a lint test pins it), so none can swallow it: a tool that decodes every row of shards 0-20 or the
  reserve (the filter, a `decode_evalset` rebuild) stops loudly at the first panel row.
  A legitimate full-shard consumer instead names its use and drops the excluded
  reciters' rows before decoding (`exposure.RowExclusion`, e.g.
  `StreamingWindowDataset(..., exposure_use="probe.training")`), with the count reported.
- **registries:** the shared staging-registry loader refuses a panel clip by name or
  checksum.

Only `tadabur.sealed_panel.open_for_scoring` with #97's flag opens it for scoring. Select,
tune and diagnose on the mining pool or `decode_evalset` dev.

Code: [`tools/tadabur/sealed_panel.py`](../sealed_panel.py). Exposure rows:
[`../exposure/sealed_panel.jsonl`](../exposure/README.md).

## What it certifies, and what it cannot

Acceptance rules §6, owner amendment 2026-10-08 (the **paired-claim panel**):

- **Reciter-disjoint from everything this PRD tunes, trains on or selects with**: truth
  sites, human labels, the mining pool, `decode_evalset` (dev, test, legacy), Muraja's
  corpus, and the uses still empty (listening sites, bias halves, probe and control
  training, synthetic-edit sources and donors, shaddah-probe tuning). Tested against the
  registry, which must be complete.
- **Recording-held-out only, toward `h448`'s own training and initialisation.** 84 of the 87
  panel reciters have rows in `h448`'s training shards (Tadabur has 671 reciters; those
  shards hold 667). The panel takes no recording from them. `h448_init`'s calibration
  windows and `h448`'s validation windows came from the lost `clips_v2` corpus (shards
  0-19); which rows is unknown, and 136 panel clips are in those shards, so recording
  disjointness from that use cannot be shown either way.
- So the panel certifies the **paired** #97 ship criterion: both `h448` and every
  candidate warm-started from it carry those shared exposures. It **does not** certify
  absolute accuracy on unseen reciters, or anything about a model not derived from `h448`.
- A teacher-decode cache certifies nothing about §3: before #97 the panel must carry blind,
  human-adjudicated correct-recitation and real-mistake sites with weights and support.

| file | what it is |
|---|---|
| `staged_clips.jsonl` | provenance of every panel clip: shard, row, canonical reciter, length, checksum (`staged_audio` schema, use `sealed_panel`) |
| `frame.json` | rows per unseen shard by outcome, unseen-shard reciters each use barred, and per panel reciter its clips and its rows in each shared-baseline use |
| `clips.jsonl` | one row per clip: word times, segmentation status, every segment's sample span, realized reference and word offsets |
| `teacher_decodes.json` | the base teacher's decode of every decodable segment, with its fingerprint |
| `summary.json` | counts, the preparation record, and the reference-only capacity |

## The frame

Every row of the shards `h448` never trained on (0-20 and the strided reserve 39, 58, …,
381) whose canonical reciter has no use in the registry other than a shared-baseline one.
Used reciters leave whole; "a new salt" would not make them fresh (§6). Then the pool's
row bounds: 1.5-50 s and a phonetizable ayah. **No model's decode decides eligibility.**
The panel is the whole frame (no cap, no draw), so a site worklist mined from it later
records its own inclusion probabilities against `frame.json`.

Of the 581 reciters in those 40 shards, 492 are barred. The binding exclusion is
`decode_evalset`: 151 reciters were untouched by the labels and the pool (#83), but 59 of
them, with nearly all of those 2,466 clips, are `decode_evalset` reciters.

## Preparation: segmentation and a decode cache, no scoring

`python -m tadabur.sealed_panel segment` runs the recitation VAD and today's pause-to-word
placement, whose whole-clip decode by the base teacher places pauses on words, and decodes
every segment once with the base teacher (bf16, whole spans, batch 1) for the cache. No
gate, no `match_ratio`, no contrast attribution, no decode-dependent drop: every VAD
segment with a reference is kept. Two segments shorter than the feature extractor's
400-sample minimum are kept without a decode.

The first preparation run (2026-10-08, via `tadabur.resegment`) also passed each teacher
segment decode through the filter gate and its drop rules, writing `match_ratio` and
contrast fields for 289 segments in a scratch directory on the GPU box. Nothing read
those scores; they were never committed, and the directory was deleted. The committed panel
comes from the second, gate-free run. No model other than the base teacher has decoded the
panel.

## Capacity

| | |
|---|---|
| clips / reciters | **216 / 87** (214 / 86 with segments; 2 ayat quran-transcript cannot phonetize) |
| audio | 0.74 h, of which 0.68 h in 298 segments |
| clips per reciter | 43 reciters have 1 clip, 21 have 2; the largest has 13 |
| haraka carriers (fatha / damma / kasra) | 3,231 / 1,037 / 1,130 |
| reference geminates | 954 |
| mid-word sukun carriers | 615 |
| pair carriers | `ق↔ك` 517, `ح↔ه` 498, `ت↔ط` 372, `س↔ص` 263, `ذ↔ز` 206, `ذ↔ظ` 171, `ض↔ظ` 70 |

These count candidate sites in the realized references alone; no decode enters them.
Real-mistake sites are a small fraction of carriers (the base teacher showed 127 pair and
522 gemination events in the pool's 12,413 census clips), so at this size the panel holds
few: its human labelling is sized by #105 against this table.

## Rebuilding

From `tools/` on the GPU box (`--index` from `python -m tadabur.staged_audio index --shards
0-384`; the registry in `../exposure/` must be complete and current first):

```bash
python -m tadabur.sealed_panel select --index stage/full_index.jsonl --out stage/panel.jsonl
python -m tadabur.sealed_panel stage --index stage/full_index.jsonl --selection stage/panel.jsonl \
    --audio-dir stage/panel_clips --shard-cache stage/hf_cache
python -m tadabur.sealed_panel segment --audio-dir stage/panel_clips --out-dir stage/seg_panel
python -m tadabur.sealed_panel build --selection stage/panel.jsonl --seg-dir stage/seg_panel
```

The 2026-10-08 run staged 216 clips (83 MB of PCM_16 WAV) under
`/root/scratch/issue-89/stage/panel_clips/` on the GPU box; `stage` read 39 shards one at a
time and deleted each.

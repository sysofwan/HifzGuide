# Synthetic edits and their decoys (#88)

[ADR-0011](../../../docs/adr/0011-transcription-fidelity-and-tashkeel-abstention.md) §3 admits
**synthetic edits** of real audio as training signal on three conditions: each edit is paired
1:1 with an **unedited decoy** labelled unchanged, so a model cannot learn "edited = mistake";
the edits come from reciters no evaluation item uses (acceptance rules §6); and a **blind
listen** confirms the edits sound real before any is used. This directory holds the record of
every edit and decoy, and the blind-check worklist. The audio stays on the GPU box.

Code: [`tools/tadabur/synthetic_edits.py`](../synthetic_edits.py) (frame, staging, decode,
rendering, manifest, worklist), [`synthetic_edit_plan.py`](../synthetic_edit_plan.py)
(where each edit and decoy goes, from the teacher's frame times) and
[`waveform_edits.py`](../waveform_edits.py) (the sample-exact crop, splice and stretch).

| file | what it is |
|---|---|
| `frame.json` | the edit frame: its rule, the indexes it was computed from (by checksum), every clip in it, the reciters it excludes |
| `base_frames.json` | the base teacher's whole-clip CTC segments of every frame clip, with its realized reference and the decode fingerprint |
| `edits.jsonl` | one row per edit or decoy: provenance, the change, the label, the output's checksum |
| `summary.json` | candidates and pairs per operation and mark, the parameters, the blind check's make-up |
| `blind_check.jsonl` | the 30-item blind-check worklist, as truth-site skeletons |
| `teacher_check.json` | whether the base teacher hears each item's label (a pre-screen, not truth) |

## The frame: reciters no evaluation item can reach

Every evaluation item this work has or plans is drawn from the shards `h448` never trained on
(the held-out block 0-20 and the strided reserve 39, 58, … 381): the truth sites (shards 0-19),
the frozen `decode_evalset` (the strided shards), the mining pool #87 listens to (the reserve),
and the sealed panel #89 builds (both). The mining pool's 394 reciters are the listening frame,
so edits cannot come from them, and the reciters the pool left untouched are the panel's.

So the frame is the rows of the **other 345 shards** (the ones `h448` trained on) whose
reciter has **no row in any of the 40 unseen shards**: 600 clips of 90 reciters (of 671 in
Tadabur; 581 appear in the unseen shards), 1.5-50 s long (9 more were not), all
phonetizable. The 344,450 training-shard rows were indexed for this
(`stage/train_shard_index.jsonl` on the GPU box, its checksum in `frame.json`). Such a reciter
cannot reach an evaluation item unless a new shard range is opened for evaluation. Training
on audio `h448` has already seen is no leak: these edits are training signal, and only
evaluation needs unseen audio.

`frame.json` records the unseen shards' 581 reciters and the `decode_evalset`'s reciters, so
`test_synthetic_edits.py` asserts disjointness from the committed artifacts alone: no edit
source or donor is one of them, or the reciter of any clip the staging registry holds for
another use, or of any truth site. The clips are staged into the registry
([`../staged_audio/clips.jsonl`](../staged_audio/README.md)) with use `synthetic_edit`.

## Operations

Each source clip is decoded whole by the base teacher (`obadx/muaalem-model-v3_2`, bf16,
batch 1, through `training.decoding.Decoder.span_class_ids`). One CTC segment is one decoded
character, at the teacher's 40 ms step. Edits touch only **anchored** positions: inside a
block where the decode equals the whole-ayah realized reference, with two equal characters
on each side, so the segment there is the teacher's emission of exactly that letter.

| operation | edit | decoy (same source clip) | label of the edit / decoy |
|---|---|---|---|
| `shaddah_removed` | crop a doubled consonant (exactly two letters; its emissions ≥ 80 ms apart) from the centre of its first emission to the centre of its second: the hold, or for a stop its closure | crop the same length from the middle of a long madd (a run of ≥ 4 madd letters) at least 0.5 s away | `not_held` / `held` |
| `shaddah_added` | stretch a single voiceless fricative (ث ح خ س ش ص ف) between two harakat by the frame's median held span (2,880 samples, 180 ms) | stretch a long madd by the same length | `held` / `not_held` |
| `consonant_swap` | splice the carrier's cell from a same-reciter donor of the other letter of `س↔ص`, `ذ↔ز`, `ض↔ظ` or `ذ↔ظ`, followed by the same haraka | splice the same cell from a same-reciter donor of the **same** letter | the other letter / the carrier's |

- A **cell** runs from the end of the previous emission to the end of the following
  haraka's, so the splice carries the consonant-to-haraka transition, where emphasis
  (`ص` against `س`) is heard. The donor's window is aligned on emission centres and must
  stay within its own neighbouring emissions. A donor with the same preceding character is
  preferred. The donor is scaled to the RMS of the span it replaces (within 0.5-2×).
- Both spans were chosen on a dry run over the first 327 staged clips, by what the base
  teacher heard (decoys stayed unchanged throughout): crop centre to centre, 18 of 25 edits
  heard single, against 9 of 25 for the gap between the emissions alone; splice through the
  haraka, 23 of 53 heard as the other letter, against 6 of 68 for a cell ending at the
  haraka's start.
- A **stretch** is pitch-synchronous where the region is periodic (whole periods, in phase
  with the join) and built from seeded random offsets where it is not (frication).
- Every join is a 10 ms Hann crossfade. Outside the crossfades an item is its source,
  sample for sample, shifted by the length change after it.

An edit and its decoy have the same kind of change and the same length change, on the same
source clip; only where it falls differs. Selection takes, per operation and mark, pairs in
salted-hash order, one per source clip and at most three per reciter, up to 60.

### The 2026-10-08 run

| operation, mark | candidates | pairs | teacher hears the edit | teacher hears the decoy unchanged |
|---|---|---|---|---|
| `shaddah_removed` | 332 | 60 | 35 / 60 | 60 / 60 |
| `shaddah_added` | 138 | 60 | 48 / 60 | 60 / 60 |
| `consonant_swap` `س↔ص` | 153 | 60 | 35 / 60 | 53 / 60 |
| `consonant_swap` `ذ↔ز` | 108 | 41 | 9 / 41 | 39 / 41 |
| `consonant_swap` `ذ↔ظ` | 33 | 21 | 8 / 21 | 21 / 21 |
| `consonant_swap` `ض↔ظ` | 6 | 5 | 0 / 5 | 5 / 5 |

247 pairs (494 items, 2.7 h of audio, 303 MB) from 42 of the 90 frame reciters; the others
have no anchored target with a neutral madd or a same-reciter donor. Re-running `generate`
reproduces every byte. The teacher column is `teacher_check.json`: the base teacher's decode
of the item spells the label at the carrier. It is a pre-screen, not truth. The teacher
leans on frequent words: it still hears the geminate in 9 of the 11 crops in `ٱللَّه`
(16 of 49 elsewhere) and `ذ` in all 16 `ذ→ز` swaps in `ٱلَّذِينَ`. The blind check is drawn
without looking at it.

## `edits.jsonl`

One JSON object per item, keys sorted. `item_id` is `<pair_id>:<role>`, and `pair_id` is
`<operation>:<mark>:<source clip>:<reference_index>`.

| field | meaning |
|---|---|
| `role` | `edit` or `decoy` |
| `operation`, `mark`, `prescribed` | what was changed; `mark` and `prescribed` as in a truth site |
| `label` | the state the item's audio holds at the carrier: the edit's changed state, the decoy's `prescribed` |
| `reference`, `reference_index` | the source's whole-ayah realized reference and the carrier in it |
| `labelled_reference` | the reference the item carries: the decoy's is the source's; the edit's drops or doubles the carrier, or swaps its letter |
| `source` | the source clip's `audio_filename`, `shard`, `row_index`, `reciter_id`, `surah_ayah`, `num_samples`, `audio_sha256` (as registered) |
| `change` | `place` (`carrier` or `madd`), `kind` (`crop`, `stretch`, `splice`), the replaced span `[start_sample, end_sample)` in the source, `inserted_samples`, a stretch's `fill_region`, a splice's `donor` (`audio_filename`, `reciter_id`, `audio_sha256`, `reference_index`, `letter`, span, `gain`) |
| `length_change`, `seed` | samples added (negative: removed); the stretch's random seed |
| `output` | the rendered file's opaque `audio_filename` (`se_<hash>.wav`), `num_samples`, `audio_sha256` |

`synthetic_edits.check_manifest` enforces the pairing (one edit and one decoy per pair,
identical but for the change; equal length change; labels and labelled references following
the edit; output length; opaque names; same-reciter donors).

Synthetic edits never enter a real-mistake rate (acceptance rules §1).

## The blind check

`blind_check.jsonl` holds 30 items, edits and decoys mixed (per operation 5 edits and 5
decoys; the swaps are 4 `ذ↔ز`, 6 `س↔ص`): per operation, pairs in
salted-hash order, alternating edit and decoy, never two items from one source clip (a
listener who heard both versions of a recitation could tell which was changed). Rows are
truth-site skeletons (`tadabur.truth_sites`, `source: synthetic_edit`, `heard: pending`):

- `audio_filename` is the rendered file's opaque name, `start_sample` 0 and `end_sample` its
  length, `audio_sha256` its checksum; `shard` is the source clip's (the edit manifest maps
  the file back to its source);
- `reference`, `reference_index`, `mark` and `prescribed` are the source's, so an edit and its
  decoy would read the same;
- `site_id` is `synthetic_edit:<hash>` and `stratum` is `synthetic_edit:blind_check` for all.

The listener answers **what was said** at the carrier (held / not held, or which letter of the
pair) and **whether it sounds natural**. The truth-site schema has no field for the second
question, so the blind UI (#87) records it beside the verdict. `ذ↔ظ` is not a truth-site mark
yet, so `ذ↔ظ` swaps stay out of the worklist until #87 adds it. The page must never show the
manifest's `role`, `operation` or `label`.

The audio is served from `/root/scratch/issue-88/stage/edits/audio/` on the GPU box: pass
that directory as the UI's audio directory, and every row's file is verified against its
`audio_sha256` by `load_truth_sites(path, audio_dir=...)`.

## Exposure

`synthetic_edits.exposure_rows` gives this work's rows for the exposure registry (#89), in
its `Exposure` shape: `synthetic_edit.source` (every source clip, whole) and
`synthetic_edit.donor` (every donor span, with its clip's checksum). These reciters are,
by construction, in the registry's `h448.training` use (shard-level) and in no other.

## Reproducing

From `tools/` on the GPU box:

```bash
python -m tadabur.staged_audio index --shards <every shard not in 0-20 or the reserve> \
    --out stage/train_shard_index.jsonl
python -m tadabur.synthetic_edits frame --unseen-index stage/shard_index.jsonl \
    --train-index stage/train_shard_index.jsonl \
    --evalset-manifest tadabur/gate_eval/manifest.json
python -m tadabur.synthetic_edits stage --train-index stage/train_shard_index.jsonl \
    --audio-dir stage/clips --shard-cache stage/hf_cache --workers 2
python -m tadabur.synthetic_edits decode --audio-dir stage/clips
python -m tadabur.synthetic_edits generate --audio-dir stage/clips --out-dir stage/edits
python -m tadabur.synthetic_edits audit --out-dir stage/edits
```

On the GPU box the run lives in `/root/scratch/issue-88/`: `stage/clips/` (the 600 staged
clips, 273 MB) and `stage/edits/audio/` (the 494 items, 303 MB).

`stage` reads one or two rows from each of 283 shards (2.4 GB each, deleted once read;
each worker holds a shard in memory, ~5 GB);
every staged clip is verified against the registry before it is decoded or edited.
`generate` is CPU-only and deterministic: re-running it reproduces every output checksum.

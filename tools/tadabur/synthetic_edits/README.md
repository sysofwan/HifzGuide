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
| `summary.json` | candidates and pairs per operation and mark, the parameters, the blind check's make-up (items per operation, mark and role) |
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

- A **cell** runs from halfway between the previous emission and the carrier's to the end
  of the following haraka's, so the splice carries the consonant-to-haraka transition,
  where emphasis (`ص` against `س`) is heard, and both crossfades fall between emissions.
  The donor's window is aligned on emission centres and must fit the donor's own
  boundaries, crossfade context included: it starts after the donor's previous emission and
  no later than its carrier's, and ends past the centre of its haraka's emission and
  before the next emission. A donor whose timing cannot fit is incompatible, so no other
  phoneme is imported and the haraka is never left out. A donor with the same preceding
  character is preferred.
- Both spans were chosen on a dry run over the first 327 staged clips, by what the base
  teacher heard (decoys stayed unchanged throughout): crop centre to centre, 18 of 25 edits
  heard single, against 9 of 25 for the gap between the emissions alone; splice through the
  haraka, 23 of 53 heard as the other letter, against 6 of 68 for a cell ending at the
  haraka's start.
- **Periodic regions keep their phase** (`waveform_edits`). A crop or a stretch of a vowel
  or a voiced hold is made of whole pitch periods, which join in phase, and the remainder
  (at most half a period) is absorbed by resampling the 12 periods that follow (a local
  pitch change of at most 4%). Cutting an arbitrary length instead joins mismatched points
  of the cycle and cancels part of the signal. A stretch repeats a unit of whole periods
  aligned to the clip's phase; where a short region cannot hold the aligned unit, the unit
  loses whole periods until it fits (an earlier revision took the truncated slice NumPy
  returned, and 13 `shaddah_added` edits advanced by half a period). Every slice whose
  length matters is checked, and a change that cannot fit is rejected (`does_not_fit`). An aperiodic region (frication, a closure)
  is cut directly or extended from seeded random offsets.
- Every cut is joined with a 10 ms Hann crossfade. Outside the span it replaced (recorded
  as `render.changed_start` / `changed_end`) an item is its source, sample for sample,
  shifted by the length change after it.

**An edit and its decoy go through one processing chain.** Same source clip, same kind of
change, same length change, same crossfades, and the same method: the periodic or aperiodic
treatment is chosen from the signal alone, and a pair whose two items would be treated
differently is rejected (`render_path`), so the processing never tells them apart; only
where the change falls does. Both items of a swap are spliced, from same-reciter donors, and
both donors are level-matched the same way. The renderer also rejects:

- a splice whose donor is more than 2× louder or quieter (RMS) than the span it replaces
  (`donor_level`); it is never clamped, which would leave a level jump;
- a pair either of whose items would exceed full scale (`peak`). Writing fails on any
  sample beyond ±1, so PCM_16 never clips silently.

**A recording is its audio, not its file name** (#117). Tadabur holds byte-identical clips
under more than one file name: one recording filed under two filename speaker ids
(`spk0215_S17_A58` and `spk0234_S17_A58`, both canonical reciter 215), or one recording
filed under two or three ayahs of identical text (`37:81`, `37:111` and `37:132`). The
frame's 600 clips are 576 recordings: 23 checksums are held by two or three file names, all
within one canonical reciter. So the planner keeps one clip per checksum (the first file
name), a donor is never the target's own carrier in any copy of its recording, and selection
lets each recording back **one pair in all**, across operations and marks.

Selection takes, per operation and mark, pairs in salted-hash order, at most three per
reciter, up to 60, and never a recording an earlier group took; the groups are filled
scarcest first (fewest candidates), so a recording goes where it is hardest to replace. A
rejected pair is skipped for the next one.

### The 2026-10-08 run (regenerated by #117)

| operation, mark | candidates | rejected | pairs | teacher hears the edit | teacher hears the decoy unchanged |
|---|---|---|---|---|---|
| `shaddah_removed` | 330 | 11 `render_path` | 56 | 30 / 56 | 56 / 56 |
| `shaddah_added` | 135 | 44 `render_path` | 33 | 25 / 33 | 33 / 33 |
| `consonant_swap` `س↔ص` | 108 | 11 `donor_level`, 5 `peak` | 45 | 14 / 45 | 45 / 45 |
| `consonant_swap` `ذ↔ز` | 24 | 2 `donor_level` | 14 | 1 / 14 | 13 / 14 |
| `consonant_swap` `ذ↔ظ` | 2 | | 1 | 0 / 1 | 1 / 1 |
| `consonant_swap` `ض↔ظ` | 1 | | 1 | 0 / 1 | 1 / 1 |

150 pairs (300 items, 1.6 h of audio, 179 MB) from 44 of the 90 frame reciters; the others
have no anchored target with a neutral madd or a compatible same-reciter donor. Both items of
every pair took the same path: `shaddah_removed` 56 periodic; `shaddah_added` 23 periodic and
10 aperiodic; every swap a splice. Donor gains run 0.58-1.96, and no item exceeds full scale
(`summary.json` → `max_peak`). Re-running `generate` reproduces every byte.

**Against #88's run** (167 pairs from 39 reciters): that run let one recording back a pair
per operation and mark, and saw two file names as two recordings. 31 recordings backed two or
three pairs, the two `ذ↔ظ` pairs were one recording under two speaker ids (byte-identical
edits), and each of their decoys took its donor from its own carrier in the other copy, so
it replaced samples with themselves and came out byte-identical to its source. Both `ذ↔ظ`
candidates left are in that one recording (each the other's decoy donor), so one pair
remains, and its decoy now splices another `ذ` of the recording. 16 items take a donor from
elsewhere in their own recording, which is a real splice.

The teacher column is `teacher_check.json`: whether the base teacher's decode of the item
spells the label at the carrier. It is a pre-screen, not truth. The teacher leans on
frequent words: in the first run (before the fixes below) it still heard the geminate in 9
of the 11 crops in `ٱللَّه` (16 of 49 elsewhere) and `ذ` in all 16 `ذ→ز` swaps in
`ٱلَّذِينَ`. The blind check is drawn without looking at it.

**Against the first run.** The first run (247 pairs) let a donor window start at the
previous emission's start and end anywhere before the haraka's end: 26 edit and 16 decoy
windows overlapped another phoneme's emission, and 2 ended at the haraka's onset. Its
periodic stretches lost phase at the second join (about a third of the energy left there),
and its donor gain was clamped rather than refused, leaving level jumps up to 4.3× and four
outputs that clipped. Fixing these cut the swap candidates (`ذ↔ز` 108 → 24) and moved the
teacher's reading of the `س↔ص` swaps from 35 of 60 to 14 of 45; their decoys went from 53 of
60 unchanged to 45 of 45. Every committed donor window now passes the boundary checks.

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
| `render` | `path` (`periodic`, `aperiodic` or `splice`), `period_samples`, and the source span `[changed_start, changed_end)` the output differs on |
| `output` | the rendered file's opaque `audio_filename` (`se_<hash>.wav`), `num_samples`, `audio_sha256`, `peak` |

`synthetic_edits.check_manifest` enforces the pairing (one edit and one decoy per pair,
identical but for the change; equal length change; the same render path; labels and
labelled references following the edit; output length; no clipping; opaque names;
same-reciter donors) and the recordings, by audio checksum: no two pairs share source
audio, no item is byte-identical to its source, and no splice takes its donor span from the
samples it replaces.

Synthetic edits never enter a real-mistake rate (acceptance rules §1).

## The blind check

`blind_check.jsonl` holds 30 items, edits and decoys mixed, 10 per operation, drawn by
`synthetic_edits.blind_check` from the committed manifest:

- per operation, in rounds over its marks (codepoint order), so every mark is represented:
  each round takes every mark's next pair in salted-hash order;
- a pair contributes one item; within a mark the roles alternate from a first role drawn
  by a salted hash of the first pair's id (`first_role`), which never reaches the page. A
  mark with one item is an edit or a decoy with equal chance and one with several is
  balanced to within one, so no count or grouping the page shows (the pair a consonant
  item offers, its ayah) fixes an item's role. (A fixed "edit first" would make every
  one-item mark an edit, readable from the page alone.);
- never two items of one **recitation**: the same source audio (Tadabur holds
  byte-identical clips under more than one speaker id) or the same reciter reciting the
  same ayah. A listener who heard both versions of a recitation could tell which was
  changed.

| operation | mark | edits | decoys |
|---|---|---|---|
| `shaddah_removed` | `shaddah` | 5 | 5 |
| `shaddah_added` | `shaddah` | 5 | 5 |
| `consonant_swap` | `ذ↔ز` | 2 | 2 |
| `consonant_swap` | `ذ↔ظ` | | 1 |
| `consonant_swap` | `س↔ص` | 2 | 2 |
| `consonant_swap` | `ض↔ظ` | | 1 |

**Redrawn for #107.** The first draw ranked an operation's pairs in one hash order, so the
swaps came out 9 `س↔ص` and 1 `ذ↔ز`, and the `ذ↔ظ` swap (a truth-site mark since #84) was
never drawn. Drawing in rounds over the marks, with hashed first roles, keeps all 20
shaddah items; the 10 swaps are 4 `ذ↔ز` and 4 `س↔ص` (2 edits and 2 decoys each) and one
item each of `ذ↔ظ` and `ض↔ظ`.

**Redrawn for #117** with the same code, from the deduplicated manifest: the make-up above
is unchanged and 26 of the 30 items are new. `ذ↔ظ` and `ض↔ظ` each have **one pair** (one
recording each in the whole frame), so neither can have an edit and a decoy in the check,
and a lone item's role must stay unreadable: both lone items are decoys by the hash, so
**no `ذ↔ظ` or `ض↔ظ` edit gets a blind listen.** That needs at least two distinct recordings
per pair, which the frame does not hold. Redraw without the audio, from `tools/`:

```bash
python -m tadabur.synthetic_edits blind-check
```

Rows are truth-site skeletons (`tadabur.truth_sites`, `source: synthetic_edit`,
`heard: pending`):

- `audio_filename` is the rendered file's opaque name, `start_sample` 0 and `end_sample` its
  length, `audio_sha256` its checksum; `shard` is the source clip's (the edit manifest maps
  the file back to its source);
- `reference`, `reference_index`, `mark` and `prescribed` are the source's, so an edit and its
  decoy would read the same;
- `site_id` is `synthetic_edit:<hash>` and `stratum` is `synthetic_edit:blind_check` for all.

The listener answers **what was said** at the carrier (held / not held, or which letter of the
pair) and **whether it sounds natural**, in the owner's listening session
([`../listening_session/README.md`](../listening_session/README.md), #107): the items share
its shuffled queue and blinding, and both answers go into one verdict in its
`verdicts.jsonl`. The page never receives the manifest's `role`, `operation` or `label`;
the UI does not even load the manifest. `python -m tadabur.edit_check_summary` tallies the
verdicts per operation for #90, after the session.

The audio is on the GPU box in `/root/scratch/issue-117/stage/edits/audio/`.
`python -m tadabur.listening_session fetch` copies the drawn items next to the session's
clips, and every row's file is verified against its `audio_sha256` before it is served.

## Exposure

`synthetic_edits.exposure_rows` gives this work's rows for the exposure registry (#89), in
its `Exposure` shape: `synthetic_edit.source` (every source clip, whole) and
`synthetic_edit.donor` (every donor span, with its clip's checksum). These reciters are,
by construction, in the registry's `h448.training` use (shard-level) and in no other, and
no source or donor is another use's recording under any row or file name
(`check_disjoint(..., by="source")` compares checksums too; asserted by
`test_synthetic_edits.py`).

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

On the GPU box the staged clips are in `/root/scratch/issue-88/stage/clips/` (the 600
clips, 273 MB) and the items in `/root/scratch/issue-117/stage/edits/audio/` (the 300
items, 179 MB; #88's 334 superseded items remain in `/root/scratch/issue-88/stage/edits/`).

`stage` reads one or two rows from each of 283 shards (2.4 GB each, deleted once read;
each worker holds a shard in memory, ~5 GB);
every staged clip is verified against the registry before it is decoded or edited.
`generate` is CPU-only and deterministic: re-running it reproduces every output checksum.

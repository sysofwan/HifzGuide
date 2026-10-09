# The exposure registry (#89)

Acceptance rules §6: **one committed registry** lists every canonical reciter id, source
recording, span and checksum, with each use it has had, so a held-out claim is checked
rather than asserted. Code: [`tools/tadabur/exposure.py`](../exposure.py) (schema, loader,
writers, `check_disjoint`, and the builder for the uses that existed on 2026-10-08).

One file per declared use (`exposure.USES`); the file name is the use. Each issue writes
only its own use's file. **Every declared use has a file**: an unused one is an empty
`<use>.jsonl`, so "unused" is recorded rather than inferred. A declared use with no file is
missing evidence: `check_disjoint` naming it raises `ExposureIncomplete`, and
`require_complete(registry)` (run before any certification) fails.

| shape | file | one entry is |
|---|---|---|
| recording-level | `<use>.jsonl` | one recording, or one span of it, that the use touched |
| shard-level | `<use>.shards.json` | the shards a use consumed, with rows per reciter |
| frozen inputs | `sources/` | the `decode_evalset` manifest and Muraja's clip list the indexed uses derive from |
| recording records | `recording_aliases.jsonl`, `probable_copies.json` | rows that are one recording by checksum, and rows that probably are by the screen (see "Duplicate recordings") |

## Row schema (`<use>.jsonl`)

| field | meaning |
|---|---|
| `audio_filename` | Tadabur's `audio_filename` |
| `shard`, `row_index` | the recording's parquet shard and row: the **source** `check_disjoint(by="source")` compares |
| `reciter_id` | the **canonical** reciter id: Tadabur's `reciter_id` column, never the filename's `spkNNNN` |
| `start_sample`, `end_sample` | the span the use took, or both `null` for the whole recording |
| `audio_sha256` | the staged 16 kHz PCM_16 WAV's SHA-256 when the recording was re-staged, else `null` |

The loader refuses a file not named for a known use, unsorted or duplicate rows, and any
recording that two uses describe differently (another row, reciter or checksum).

A shard use (`<use>.shards.json`) also records its `membership`: `exact` when the shards are
the use's exact input, `uncertain` when it took an unknown subset of their rows (every row
then counts as possibly exposed). `shared_baseline` marks an exposure of `h448` itself,
which the baseline and every candidate warm-started from it share.

## The uses

| use | recordings | reciters | from |
|---|---|---|---|
| `truth_site.waqf_boundary` | 123 | 81 | `truth_sites/waqf_boundaries.jsonl` (item spans) |
| `truth_site.p35_fixture` | 151 | 96 | `truth_sites/p35_fixtures.jsonl` (item spans) |
| `human_label.p35_fixture` | 206 | 113 | `eval_fixtures/should_{accept,reject}.jsonl` |
| `human_label.waqf_events` | 126 | 82 | `waqf_event_fixtures/waqf_events.{calibration,test}.jsonl` |
| `human_label.reject_reread` | 31 | 24 | `eval_fixtures/reject_reread_verdicts.jsonl` |
| `human_label.reject_bleed` | 9 | 7 | `eval_fixtures/reject_bleed_labels.jsonl` |
| `human_label.tashkeel_counterfactual` | 47 | 21 | `tashkeel_counterfactual_fixtures/counterfactual_items.jsonl` |
| `mining_pool` | 2,508 | 394 | `staged_audio/clips.jsonl` (#83) |
| `decode_evalset.dev` | 970 | 146 | `sources/decode_evalset.manifest.json` (sha256 `3d5d237a…`) |
| `decode_evalset.test` | 1,030 | 140 | the same; the half ADR-0010 spent |
| `decode_evalset.legacy_stratified` | 902 | 235 | the same manifest's ratio-stratified records, scored by earlier gates |
| `muraja.reread_corpus` | 447 | 153 | `sources/muraja_clips.json`: the #70 re-read corpus and scenario bundles shipped to Muraja |
| `h448.training` | 345 shards | 667 | `h448`'s `--stream-shards`; membership `exact`, shared baseline |
| `h448.init_validation` | 20 shards | 512 | `h448_init`'s calibration and `h448`'s validation windows, from the lost `clips_v2` corpus (shards 0-19); membership `uncertain`, shared baseline |
| `sealed_panel` | 197 | 79 | [`../sealed_panel/staged_clips.jsonl`](../sealed_panel/README.md) (#89) |
| `synthetic_edit.source` | 150 | 44 | [`../synthetic_edits/edits.jsonl`](../synthetic_edits/README.md) (#88, deduplicated by #117): source clips, whole |
| `synthetic_edit.donor` | 77 | 22 | the same: donor spans |

Empty until their issues write them: `truth_site.new_audit` (#87),
`truth_site.synthetic_edit`, `probe.training`, `probe.kl_control`, `bias.tune`,
`bias.score`, `shaddah_probe.tuning`.

Tadabur has 671 reciters and `h448`'s training shards hold 667 of them, so every held-out
set overlaps `h448.training` by reciter. Under the owner's §6 amendment (2026-10-08) the
sealed panel is reciter-disjoint from every **PRD** use and only recording-held-out from the
shared-baseline ones; [its README](../sealed_panel/README.md) says what that certifies.

## Using it

```python
from tadabur.exposure import SEALED_PANEL, SYNTHETIC_EDIT_SOURCE, TRUTH_SITE_USES, check_disjoint
check_disjoint(SYNTHETIC_EDIT_SOURCE, SEALED_PANEL, *TRUTH_SITE_USES.values())  # by reciter
check_disjoint(SYNTHETIC_EDIT_SOURCE, SEALED_PANEL, by="source")
```

`check_disjoint(use, *others)` compares `use` with each of `others` (never the others
among themselves) and raises `ExposureOverlap` naming what they share. `by="source"`
matches a shard row, an `audio_sha256` both rows carry, a checksum-confirmed alias or a
probable copy from the screen, against a shard use's rows too (and between two shard uses):
a copy of a recording under another row is the same recording (see below). A pass means
no overlap **detected** under the screen, not proven recording disjointness.

A run that streams whole shards for a PRD use excludes, before decoding, the reciters §6
keeps out of it: `excluded_reciters(use)` is the sealed panel's reciters for every PRD use,
plus the bias score half's for a training or tuning use, and `RowExclusion.for_use(use)`
filters a row stream by them and counts what it dropped. `training.distill_stream` takes it
as `exposure_use` (`--exposure-use` on the probe). A new use is
written with `write_use(use, rows)` (or `write_shard_use`), which validates it against the
whole registry. Name uses through the constants.

```bash
python -m tadabur.exposure build --index stage/full_index.jsonl
python -m tadabur.exposure describe
python -m tadabur.exposure copies --index stage/full_index.jsonl   # the recording records
python -m tadabur.exposure duplicates
```

`build` rewrites every use above except `sealed_panel` from the committed sources and a
full shard index (`python -m tadabur.staged_audio index --shards 0-384`, ~3 minutes over
HTTP), and writes an empty file for any declared use that has none.

## Duplicate recordings (#117)

Tadabur holds byte-identical clips under more than one file name and shard row: one
recording filed under two filename speaker ids (`spk0215_S17_A58` and `spk0234_S17_A58`),
or under two or three ayahs of identical text (`37:81`, `37:111`, `37:132`). A row's
`audio_filename` and `(shard, row_index)` do not show it. Two committed records do, and
`check_disjoint(by="source")` and the panel's frame use both. Only the relation is stored;
which uses a group touches is derived when the registry is read, so no write can leave it
stale (`python -m tadabur.exposure duplicates` reports it).

**`recording_aliases.jsonl`: confirmed by checksum.** One line per row (`audio_sha256`,
`audio_filename`, `shard`, `row_index`, `reciter_id`) of every checksum two or more rows
carry, gathered from the staging registry and every use's checksums. It only grows: the
writer merges into what is committed, so evidence outlives the use that brought it (the
panel's pruned duplicates still decide its next `select`). 32 checksums over 65 rows: 23
among the synthetic-edit frame's clips (which the edit generator deduplicates), 6 within
the mining pool, 1 a P3.5-fixture label's clip that is also a pool clip (reciter 728,
`37:111` and `37:132`), and 2 that the sealed panel held twice. Only re-staged rows have a
checksum: 3,636 in the committed records, none of `decode_evalset`'s, Muraja's, or any
`h448` training row's.

**`probable_copies.json`: the screen, `copy-screen-v2`.** Rows with **one canonical reciter
and one duration to the millisecond**, whatever their file name or ayah, over a full shard
index (its SHA-256 recorded): 45,360 groups holding 110,367 of the 384,450 rows. A copy
keeps its reciter and its length; the name is no evidence either way. Against every row
that has a checksum (`screen_agreement`):

| screen | confirmed groups found (recall) | grouped checksummed pairs: one checksum / two (false) |
|---|---|---|
| `copy-screen-v1` (same name but the `spkNNNN` prefix, same duration) | 19 / 32 | 19 / 0 |
| **`copy-screen-v2`** (same reciter, same duration) | **32 / 32** | **34 / 4** |

v1 missed every copy filed under another ayah's name, the panel's two included, so a pass
under it certified nothing about them. v2 finds every confirmed group at the price of
false positives (4 of the 38 checksummed pairs it groups are different audio), and a
probable copy counts as a conflict on suspicion. Its recall on copies no checksum has seen
is unmeasured, so **a pass is "no overlap detected under `copy-screen-v2` (recall 32 of 32
on checksum-confirmed groups)", not recording disjointness.**

Every confirmed duplicate is within one canonical reciter (and the v2 screen only groups
within one), so no **reciter-level** guarantee of §6 is touched. The **recording-level**
one was: under the screen, 18 sealed-panel recordings had a probable copy in an
`h448.training` shard, against the owner's §6 amendment, and a 19th was in the panel twice.
They were removed ([the panel's README](../sealed_panel/README.md)); the panel is now 197
recordings of 79 reciters, and `check_disjoint(SEALED_PANEL, …, by="source")` detects no
overlap. `h448.training` and `h448.init_validation` (shards 0-19) share 9,643 rows' probable
copies; both are shared-baseline uses, so nothing requires them apart, but the check now
says so.

**Known limitation, outside §6.** Recordings of other uses with an alias or probable copy
under a row of another use (`python -m tadabur.exposure duplicates`, at v2), the largest:

| use | copy in `h448.training` / `h448.init_validation` |
|---|---|
| `decode_evalset.dev` / `.test` / `.legacy_stratified` | 289 / 298 / 180, and 30 / 34 / 22 |
| `mining_pool` | 363 / 45 |
| `human_label.p35_fixture`, `truth_site.p35_fixture` | 43 / 1, 25 / 1 |
| `truth_site.waqf_boundary`, `human_label.waqf_events` | 22 / 2 each |
| `muraja.reread_corpus` | 35 / 2 |

v2's false positives inflate these (at most 34 of 38 grouped pairs were real where
checksums could tell). The frozen `decode_evalset` and the mining pool, both meant to be
audio `h448` never saw, may not be for some of these recordings; neither is changed here
(`decode_evalset` is frozen and its test half spent, ADR-0010; the pool's listening draw
is fixed), and a claim on either should carry this caveat. Under v1, the conservative
floor, the counts were 50 of 2,902 and 27 of 2,508.

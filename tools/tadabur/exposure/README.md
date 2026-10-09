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
| `sealed_panel` | 216 | 87 | [`../sealed_panel/staged_clips.jsonl`](../sealed_panel/README.md) (#89) |
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
matches a shard row, or an `audio_sha256` both rows carry: a copy of a recording under
another row is the same recording (see below).

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
python -m tadabur.exposure duplicates --index stage/full_index.jsonl
```

`build` rewrites every use above except `sealed_panel` from the committed sources and a
full shard index (`python -m tadabur.staged_audio index --shards 0-384`, ~3 minutes over
HTTP), and writes an empty file for any declared use that has none.

## Duplicate recordings (#117 audit, 2026-10-08)

Tadabur holds byte-identical clips under more than one file name and shard row: one
recording filed under two filename speaker ids (`spk0215_S17_A58` and `spk0234_S17_A58`),
or under two or three ayahs of identical text (`37:81`, `37:111`, `37:132`). A row's
`audio_filename` and `(shard, row_index)` do not show it; only a checksum does, and only
re-staged recordings have one (3,228 of the 6,618 recordings here; none of
`decode_evalset`'s, Muraja's or three of the human-label files'). `python -m tadabur.exposure
duplicates` reports two things:

- **by checksum** (`duplicate_recordings`): 9 checksums are held under two rows here: 6
  pairs within `mining_pool`, 2 within `sealed_panel` (two of its recordings are each in it
  twice), and 1 a `human_label.p35_fixture` clip that is also a `mining_pool` clip (reciter
  728, `37:111` and `37:132`). The staging registry
  ([`../staged_audio/clips.jsonl`](../staged_audio/README.md)) holds 30 such checksums over
  61 of its 3,436 clips: these 7, plus 23 among the synthetic-edit frame's clips, which the
  edit generator now deduplicates. Every one is within one canonical reciter.
- **by screen** (`probable_copies`, needs a full shard index): rows whose file name is the
  same but for its `spkNNNN` prefix and whose duration is the same to the millisecond. Of
  the 384,450 rows, 6,392 fall in 3,196 such groups, none across two canonical reciters.
  Wherever both rows of a screened pair had a checksum (19 pairs) they were identical, but
  the screen finds only 19 of the 32 checksum groups (it misses copies filed under another
  ayah), so its counts are a floor. Recordings of one use with a probable copy under a row
  of another:

| use | copy in | recordings |
|---|---|---|
| `sealed_panel` | `h448.training` | 12 |
| `decode_evalset.dev` / `.test` / `.legacy_stratified` | `h448.training` | 14 / 25 / 10 |
| `decode_evalset.dev` | `h448.init_validation` | 1 |
| `mining_pool` | `h448.training` / `h448.init_validation` | 25 / 2 |
| `mining_pool` | `muraja.reread_corpus` | 1 |
| `truth_site.waqf_boundary`, `human_label.waqf_events` | `h448.training` | 2 (the same 2) |
| `human_label.p35_fixture` / `.reject_reread` / `.tashkeel_counterfactual` | `h448.training` | 2 / 1 / 1 |
| `muraja.reread_corpus` | `h448.training` | 2 |

**What it means for §6.** Every duplicate found, by checksum or by screen, is within one
canonical reciter, so no **reciter-level** guarantee is touched: the panel's reciter
disjointness from every PRD use, and the score half's from training, stand. The
**recording-level** guarantees are where a duplicate bites, since a copy under another row
passes a row comparison. By checksum, no use's recording is another use's that §6 keeps
apart, and `check_disjoint(by="source")` now matches checksums, so a re-staged copy cannot
pass. But a shard use has no checksums, and by the screen **12 of the panel's 216
recordings probably have a copy in `h448.training`'s shards**: the panel's
recording-level hold-out from `h448`'s training (the owner's §6 amendment) probably fails
for them. Confirming it by checksum means decoding those rows, which are panel audio by
content, so it is left to the owner (#89). Outside §6, the frozen `decode_evalset` and the
mining pool, both meant to be audio `h448` never saw, probably are not for 50 of 2,902 and
27 of 2,508 recordings (a copy in `h448.training` or `h448.init_validation`).

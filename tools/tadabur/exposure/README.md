# The exposure registry (#89)

Acceptance rules §6: **one committed registry** lists every canonical reciter id, source
recording, span and checksum, with each use it has had, so a held-out claim is checked
rather than asserted. Code: [`tools/tadabur/exposure.py`](../exposure.py) (schema, loader,
writers, `check_disjoint`, and the builder for the uses that existed on 2026-10-08).

One file per use; the file name is the use. Each issue writes only its own use's file.

| shape | file | one entry is |
|---|---|---|
| recording-level | `<use>.jsonl` | one recording, or one span of it, that the use touched |
| shard-level | `<use>.shards.json` | the shards a use consumed whole, with rows per reciter |

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
| `decode_evalset.dev` | 970 | 146 | the frozen manifest (sha256 `3d5d237a…`) on the GPU box |
| `decode_evalset.test` | 1,030 | 140 | the same; the half ADR-0010 spent |
| `decode_evalset.legacy_stratified` | 902 | 235 | the same manifest's ratio-stratified records, scored by earlier gates |
| `muraja.reread_corpus` | 447 | 153 | the #70 re-read corpus and scenario bundles shipped to Muraja (shards 20-30) |
| `h448.training` | 345 shards | 667 | `h448`'s `--stream-shards` (`H448_STREAM_SHARDS`), counted from a full index |
| `sealed_panel` | 216 | 87 | [`../sealed_panel/staged_clips.jsonl`](../sealed_panel/README.md) (#89) |

Reserved for the issues that will make them: `synthetic_edit.source`,
`synthetic_edit.donor` (#88), `probe.training`, `probe.kl_control`, `bias.tune`,
`bias.score`, `shaddah_probe.tuning`. Tadabur has 671 reciters; `h448`'s training shards
hold 667 of them, so every held-out set is reciter-overlapping with `h448.training` and
can only be source-disjoint from it.

**Known, unrecorded:** `h448_init` (teacher-init calibration) and `h448`'s validation
windows came from the `clips_v2` corpus, the filter's passes over shards 0-19, which was
lost with `audit_run/`. Neither took a gradient step on those clips, but which clips they
were cannot be recovered, so they are not rows here.

## Using it

```python
from tadabur.exposure import SEALED_PANEL, SYNTHETIC_EDIT_SOURCE, TRUTH_SITE_USES, check_disjoint
check_disjoint(SYNTHETIC_EDIT_SOURCE, SEALED_PANEL, *TRUTH_SITE_USES.values())  # by reciter
check_disjoint(SYNTHETIC_EDIT_SOURCE, SEALED_PANEL, by="source")
```

`check_disjoint(use, *others)` compares `use` with each of `others` (never the others
among themselves) and raises `ExposureOverlap` naming what they share. A new use is
written with `write_use(use, rows)` (or `write_shard_use`), which validates it against the
whole registry. Name uses through the constants: a module that spells the panel's name
outright is refused by the panel's seal test.

```bash
python -m tadabur.exposure build --index stage/full_index.jsonl \
    --evalset-manifest tadabur/gate_eval/manifest.json --muraja-clips stage/muraja_clips.json
python -m tadabur.exposure describe
```

`build` rewrites every use above except `sealed_panel`; `--index` is
`python -m tadabur.staged_audio index --shards 0-384` (~3 minutes over HTTP).

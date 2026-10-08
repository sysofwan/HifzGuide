# Staged audio (#83)

`audit_run/` was cleared on 2026-10-08 and took the audio behind every label with it. This
directory holds what makes that audio re-stageable: the **provenance of every Tadabur clip
this repo has re-staged**, committed per AGENTS.md (*Data & assets*). The audio itself is
never committed. It is re-downloadable from `FaisaI/tadabur`, and the checksum here proves a
re-download is the same audio.

The schema, loader and staging code are in [`tools/tadabur/staged_audio.py`](../staged_audio.py).

| file | what it is |
|---|---|
| `clips.jsonl` | one row per staged whole clip, sorted by `audio_filename` |

## Schema

| field | meaning |
|---|---|
| `audio_filename` | Tadabur's `audio_filename` (the `audio.path` of the row) |
| `shard` | full-config parquet shard `data/train-<shard:05d>.parquet` |
| `row_index` | the row's position in that shard |
| `reciter_id` | the **canonical reciter id**: Tadabur's own `reciter_id` column. The filename's `spkNNNN` disagrees with it on 347 of the 40,000 rows indexed, so it is never parsed for one. |
| `surah_ayah` | `"surah:ayah"`, from Tadabur's 0-indexed `surah_id` + 1 and `ayah_id` |
| `num_samples` | length of the staged 16 kHz mono clip |
| `audio_sha256` | SHA-256 of the staged WAV's bytes (`truth_sites.audio_sha256`) |
| `uses` | why it was staged, sorted: `waqf_boundary` and `p35_fixture` (its truth sites), `mining_pool` |

The staged file is the row's audio decoded with `tadabur.audio.decode_to_mono_16k` and
written by `soundfile` as **16 kHz mono PCM_16 WAV**, the format the labels were made on.
Every decode recorded alongside these clips reads that file back, so the decode and the
checksum describe the same samples. A re-stage with different `librosa` / `soundfile`
versions could change the bytes; `stage_clips` then refuses the clip rather than silently
replacing its checksum.

This is the clip half of the exposure registry the acceptance rules (§6) ask for. Spans
inside a clip live with the sites or pool segments that use them.

## Re-staging

From `tools/` on the GPU box (each shard is ~2.4 GB, downloaded into a cache directory of
the run's own and deleted once its rows are staged):

```bash
python -m tadabur.staged_audio index --shards 0-20,39,58,77,96,115,134,153,172,191,210,229,248,267,286,305,324,343,362,381 \
    --out stage/shard_index.jsonl
python -m tadabur.staged_audio stage --index stage/shard_index.jsonl \
    --pool-selection stage/pool_selection.jsonl --audio-dir stage/clips \
    --shard-cache stage/hf_cache --registry tadabur/staged_audio/clips.jsonl
```

`index` reads only the light columns of each shard over HTTP (about a minute for 40
shards). `stage` stages the clips behind the committed labels (the waqf-boundary truth
sites and the P3.5 fixtures) plus the mining pool's selection, checks every row's filename
and reciter against the index, and checkpoints this registry after each shard. With an
existing registry it verifies every re-staged clip against its recorded checksum.

## The 2026-10-08 staging run

STAGING_RUN_PLACEHOLDER

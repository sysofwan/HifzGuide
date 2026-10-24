# tlog metadata index (#185)

`index/` is the committed metadata of `tarteel-ai/tlog` @
`9fe4f7e6d1b372c2c11157d6d2f0f91ca344eb51`: one `jsonl.gz` per source shard
(`index/<split>/<shard>.jsonl.gz`, 743 files) plus `index/meta.json`. It is written by
`tadabur.tlog_index`; the row schema and the rules are in that module's docstring, and the frozen
hold-out rules are in `docs/tlog-holdout-preregistration.md`.

**Metadata only.** No tlog audio is copied, committed or published, and the dataset's own `label`
column is never read: a row's label comes from its `audio.path` filename, checked against
`quran.db`'s ayah counts. The committed copy is the source of truth — the pre-registration cites
index commit `683c3e8`, so do not rewrite it to make a build "current".

`meta.json` records the pin, the counts posted on #173, and `code_sha`: the commit that built the
index (not necessarily HEAD). Every shard file is the byte witness of a rerun; `meta.json` differs
by `code_sha` alone.

## Rebuilding

The dataset is gated, so the build needs a Hub token whose account accepted its access terms. That
token lives on cuda-dev; a run anywhere else fails with HTTP 401 (and bursts fail with 429).

```bash
# On cuda-dev, from tools/. --quran-db is gitignored, so point it at a checkout that has it.
python -m tadabur.tlog_index build --workers 4 \
  --quran-db /root/repos/HifzGuide/data/quran.db
```

- Use 4–8 workers. Two back-to-back full sweeps trip the Hub's rate limit, whose retry waits run
  to minutes; one sweep of 743 shards took 170 s at 16 workers, so give 4-8 a few minutes.
- `--code-sha` defaults to `git rev-parse HEAD`; pass the commit that holds the code being built.
- The shard files are byte-identical on the build interpreter. gzip's header OS byte is
  platform-dependent (Python 3.12 on cuda-dev writes `0x03`, macOS Python 3.13 writes `0xff`), so
  a rerun elsewhere changes that one byte in every file and nothing else.

## Checking

```bash
python -m pytest -q tools/tadabur/test_tlog_index.py   # from the repo root
```

The tests assert every shard's coverage and row schema, the counts in `meta.json`, the pre-
registration's SHA-256, and the build itself against fake pins (offline).

## The S2 validation (#186)

`s2/` holds what the speaker-cluster procedure was validated on: `sample.jsonl` (the fixed-seed
Tadabur draw — `audio_filename`, `shard`, `row`, `reciter_id`, `surah_ayah` and the 20 s embedding
span), `selection.json` (its census: salt, shards, frame checksum, deciles, reciter and
same-ayah counts) and `report.json` (every model's τ-grid metrics and the chosen model, revision
and τ). `docs/tlog-speaker-clustering.md` is generated from the report by
`python -m tadabur.tlog_s2 report`; neither is edited by hand. Embeddings are scratch, never
committed.

```bash
python -m tadabur.tlog_s2 sample                 # redraw the sample (same bytes)
python -m tadabur.tlog_s2 embed --out-dir DIR    # on cuda-dev: pin and embed, GPU
python -m tadabur.tlog_s2 report --embeddings-dir DIR
```

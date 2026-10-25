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

## The pass-1 facts and the silence floors (#187)

`pass1/` is what `tadabur.tlog_pass1` commits about the tracer shards:
`pass1/<split>/<shard>.jsonl.gz` carries, per row, `split`, `shard_file`, `row`, `audio_path`,
`audio_sha256`, `codec_ok`, `sample_rate`, `duration_s`, `speech_s`, `speech_ratio` and
`decoder` (`soundfile` or `pyav`, the two decoders pass 1 tries in that order), and
`pass1/meta.json` records the code SHA, the VAD config, the installed decoder versions, the
floors' SHA-256, the counts and the measured throughput. Nothing else is computed: no label,
transcript, teacher/student output or speaker embedding (the owner accepted the clip-level
fallback on 2026-10-24, so `"embeddings"` records it). The Haitsma–Kalker fingerprint goes to
cuda-dev scratch (`/root/scratch/tlog/fp/`) and is never committed; each tlog shard is
downloaded, read and deleted.

`silence_floors.json` is the pre-registration's Exclusions floor: `f_s` and `d_s` are the
type-7 0.5th percentiles of `speech_ratio` and `speech_s` over the Tarteel v1 look clips that
decode. `build_floors` refuses a run that does not yield every id in the frozen look split, and
the record hashes both `split.json` (the id set) and the decodes (the population's provenance).
The speech is measured by Silero VAD `6.2.3` (ONNX, CPU), with the config recorded in the file;
`run` refuses to start when its config differs. The floors commit must land before any pass-1
output commit.

The tracer rule, fixed before the run: clean shard indices 0 and 205 (of 411) and unclean index
0 (of 332). On cuda-dev, with the Hub token in the environment — **no GPU lock**: decode, VAD
and fingerprinting are CPU-only, so taking `/root/scratch/gpu.lock` would block every GPU job
for the length of the run:

```bash
python -m tadabur.tlog_pass1 floors --pr 198 --cache-dir /root/scratch/tlog/cache
python -m tadabur.tlog_pass1 run --tracer --pr 198 --cache-dir /root/scratch/tlog/cache
```

Both commands refuse a dirty checkout and a missing or unpinned PyAV, and `run` refuses a floors
file whose VAD config differs. Rows are **schema 2** (`schema_version` in each row; `decoder`
arrived in schema 2); `meta.json` and `silence_floors.json` are schema 1 and are extended
additively.

The committed rows were measured at `d2af0e3`; `f139f16` later narrowed the decoder's error path
so resource failures abort instead of becoming a codec exclusion. That changed no recorded fact
(the 268 `codec_ok: false` rows are all empty-byte audio, the 48 MP4/AAC rows still decode), and
S3b will cite its own later `code_sha`. The floors were measured at `5e6d121`, **before the
container fallback existed**, which is why all 4,537 look clips decoded with soundfile. The
pre-registration SHA recorded in `s2/report.json` and `docs/tlog-speaker-clustering.md` was
re-stamped for the dated `codec` amendment, not recomputed: the S2 decision and results are
unchanged.

The branch is squash-merged, so a `code_sha` it cites survives only under
`refs/pull/<N>/head`; record the PR number beside any cited SHA. The offline tests, which need
no network, GPU or Silero install (PyAV is needed for the MP4/AAC fallback tests), are
`python -m pytest -q tools/tadabur/test_tlog_pass1.py`.

## The S3b full pass and the split freeze (#188)

The gate is `run --gate --workers 3`: clean 0-5 and unclean 0-5. Clean 0 and unclean 0 are the
tracer's committed outputs, so the resume guard verifies them and skips them; they do not enter
the measured rate. Each split is timed as its own phase, and `throughput.gate_projection`
weights each split's measured rows/s by the index's row counts for the shards the run did not
process (no second division by the worker count; the five-shard tail on three workers is
conservative). Start the full run (`--all --workers 3`) only when `within_owner_envelope` is
true: <= 8 wall hours and 0 GPU hours. Outputs go to scratch
(`--out-dir /root/scratch/tlog/s3b/pass1`), never the measured checkout, so the clean-tree
guard still holds, a resumed run re-verifies and skips existing shards, and the fetched files
are metadata only. `scripts/issue-188-pass1.sh` stages the committed branch on cuda-dev, runs
the gate and the full pass detached without the GPU lock, and fetches the rows; it reports the
PID, log and completion files.

`tlog/split/` is the freeze: `<split>/<shard>.jsonl.gz` carries `(split, shard_file, row,
audio_path, speaker_cluster, group_key, role)`. Under the owner's fallback, `speaker_cluster`
is always `null`, `group_key` is the S1 id group's smallest `audio_path`, and the role is
`tadabur.tlog_index.role(group_key)`, reproducible from the committed group keys alone.
`split/meta.json` records the code SHA with its PR, the pre-registration and S2 report
SHA-256s, `"embeddings": "none (owner fallback 2026-10-24)"`, the counts per role and split,
the 1% group cap and largest share, each role's surah/ayah distribution against the training
pool, the yield after the floors, and each shard's producing SHA + PR. Rebuild it offline, from
the committed index and pass 1, with `python -m tadabur.tlog_split build --pr <N>`.


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

`pass1/` is what `tadabur.tlog_pass1` commits: now all 743 shards of both splits (the three
tracer shards were its first). `pass1/<split>/<shard>.jsonl.gz` carries, per row, `split`,
`shard_file`, `row`, `audio_path`,
`audio_sha256`, `codec_ok`, `sample_rate`, `duration_s`, `speech_s`, `speech_ratio` and
`decoder` (`soundfile` or `pyav`, the two decoders pass 1 tries in that order), and
`pass1/meta.json` records the code SHA, the VAD config, the installed decoder versions, the
floors' SHA-256, the counts and the measured throughput. Its `throughput.extrapolated` block is
the **single-worker stage model** (`"model": "single_worker_model"`), not a finished run's cost:
`throughput.wall_hours_actual`, `wall_hours_max` and `scope` are, and `pass1/run_outcome.json`
restates them as the run's own record. Nothing else is computed: no label,
transcript, teacher/student output or speaker embedding (the owner accepted the clip-level
fallback on 2026-10-24, so `"embeddings"` records it). The Haitsma–Kalker fingerprint goes to
cuda-dev scratch (`/root/scratch/tlog/fp/`) and is never committed; each tlog shard is
downloaded, read and deleted.

The store is one `<shard>.npz` per shard (`numpy.savez_compressed`), each member keyed
`<audio_path>|<audio_sha256>` — the path alone is not unique, and the digest distinguishes two
rows that share it. Member count equals the shard's `codec_ok` rows on every committed shard.
`tlog/fingerprints.json` is its committed census (`python -m tadabur.tlog_pass1 fingerprints`,
`tadabur.tlog_pass1.fingerprint_store_record`): per shard, the npz's SHA-256, its member count
and the `codec_ok` rows it must cover, so a lost or short store is detected without
re-streaming the corpus.

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
tracer's committed outputs: `--seed-dir` (the checkout's `pass1/`) verifies them against the S1
index and copies them into the scratch output directory, so the resume guard skips them. The
throughput sample is therefore clean 1-5 and unclean 1-5, one timed phase per split; each phase
is settled in `$OUT/gate_phases.json` as it finishes, bound to the tranche, worker count and code
SHA, so an interrupted gate keeps the split it measured. A retry reuses a stored phase only for
the same sample at the same binding, and refuses to gate — rather than pricing one split at the
other's rate — when a split's phase is neither measured completely nor stored.
`$OUT/gate_projection.json` is the gate's own record, so the full run (which reuses the scratch
directory) cannot overwrite it; only a gate run writes it or `throughput.gate_projection`. Its
`within_owner_envelope` is wall hours **and** the measured peak disk (`within_wall_envelope`,
`within_disk_envelope`), and it names the shards each phase measured. Start the full run
(`--all --workers 3`) only when disk and GPU fit and the projected wall hours fit the limit in
force: the pre-registered <= 8 wall hours, 0 GPU hours and <= 5 GB peak disk — or an owner's
raised wall limit for this run, passed as `--owner-wall-hours` with `--owner-decision` (the
comment that grants it). The override is refused without its citation, cannot relax the disk or
GPU bound, and never rewrites `gate_projection.json`: the raised limit, the citation and the
measurement it covered land in `meta.json` and `gate_override.json`. Outputs go to scratch
(`--out-dir /root/scratch/tlog/s3b/pass1`), never the measured checkout, so the clean-tree
guard still holds, a resumed run re-verifies and skips existing shards, and the fetched files
are metadata only. `gate_projection.json`, `gate_phases.json` and `gate_override.json` are the
gate's envelope evidence: commit all three with the pass-1 rows and `meta.json`.
`$OUT/producers.json` records each published shard's producing commit, so an interrupted run
keeps its attribution and a resumed run at a later commit does not claim rows it only verified;
an output with no producing record is refused. `scripts/issue-188-pass1.sh` stages the committed
branch on cuda-dev, runs the gate and the full pass detached without the GPU lock, refuses
`full` unless the gate's verdict fits the limit in force (`OWNER_WALL_HOURS` and
`OWNER_DECISION` are forwarded to the guard), and fetches the rows; it reports the PID, log and
completion files.

`tlog/split/` is the freeze: `<split>/<shard>.jsonl.gz` carries `(split, shard_file, row,
audio_path, speaker_cluster, group_key, role)`. Under the owner's fallback, `speaker_cluster`
is always `null`, `group_key` is the S1 id group's smallest `audio_path`, and the role is
`tadabur.tlog_index.role(group_key)`, reproducible from the committed group keys alone.
`split/meta.json` records the code SHA with its PR, the pre-registration and S2 report
SHA-256s, `"embeddings": "none (owner fallback 2026-10-24)"`, the counts per role and split,
the 1% group cap and largest share, each role's surah/ayah distribution against the training
pool, the yield after the `codec` and `silence` floors, and each shard's producing SHA + PR.
The yield is not the whole exclusion set: `exclusions` names `eval_overlap` and `duplicate` as
deferred to the S4 tree, so a training pool is `codec ∪ silence ∪ S1's
eval_overlap:tarteel_v1 ∪ overlap/excluded`. `split/pool.json` is the gating population S4
consumes, written by the same build: per split and per role, `rows`, `usable` after the codec
and silence floors, `eval_overlap_tarteel_v1` and `usable_after_eval_overlap` (495,030 usable
rows contain 10,386 Tarteel v1 eval-overlap rows; `meta.json` keeps the floors-only yield).
Those counts stop before S4; `overlap/meta.json` carries `usable_after_s4` per role, train
included, and `tadabur.tlog_overlap.iter_pool_rows` is the one reader that joins the frozen row
to its S4 record as a single ordered `exclusions` tuple. Rebuild it offline, from the committed
index and pass 1, with `python -m tadabur.tlog_split build --pr <N>`; a shard the pass-1 meta
omits is refused rather than attributed to the build. A later HEAD changes `meta.json`'s
`code_sha` and the producers it cites, so reproduce the committed freeze with
`build --code-sha 7380e1c --pr 199` (every file byte-identical, `pool.json` included).


## The S4 overlap tree (#189)

`overlap/` is what `tadabur.tlog_overlap` commits: `<split>/<shard>.jsonl.gz`, one record per
frozen row keyed `(split, shard_file, row)` — never by `audio_path`, which 4,052 paths share
across 8,104 rows — with
`{schema_version, split, shard_file, row, eval_overlap, duplicate_of, duplicate_ber, excluded}`.
`eval_overlap` lists **every** set hit, the best match per set, as `{set, clip, ber}` in the
fixed set order `tarteel_v1, qud, gate_eval, tadabur_frame, qdat, qdat_bench, retasy`;
`duplicate_of` is the surviving copy's `{split, shard_file, row}`; `duplicate_ber` is `0` only
for an equal `audio_sha256`; and `excluded` is S4's own primary reason, `eval_overlap:<set>`
beating `duplicate`, or null. Rows excluded upstream (codec, silence, S1's id overlap) still
get their S4 record: the final exclusion is the union, and S4 never rewrites another stage.

The sets are fingerprinted with `qud_fingerprint`'s parameters into a scratch store
(`/root/scratch/tlog/overlap/fp/<set>/<unit>.npz`, one npz per source shard, atomic and
resumable) and matched by a Haitsma–Kalker inverted index of 32-bit sub-fingerprints: exact-key
votes at one aligned time offset propose a clip, and `best_match` verifies it at the committed
line `0.238` (`qud_eval/overlap_rules.json`, reused with its SHA-256 in `meta.json`; the sets
offer no clean hard negatives to re-derive it from). `0` and `0xFFFFFFFF` are stop-words — not
indexed, not queried — and every verified match reports the share of its overlap window that is
a stop-word in either fingerprint, so digital-silence matches are counted rather than hidden.
The clip ids come from the committed checksum lists
(`exposure/nonpro.evaluation_only.clips.jsonl`: Tarteel v1's 8,950, qdat's 1,505, qdat_bench's
159, RetaSy's 440; `exposure/qud.correct_recitation.qud.jsonl`: the two pilot mushafs' 12,471
ayat), which also give the exact-checksum check. The frame is one pass over
`held_out_shards()`' 40 shards: a row is labelled `gate_eval` when it is a
`decode_evalset.{dev,test}` row, else `tadabur_frame`; the 197 sealed-panel clips are skipped
and recorded as a blind spot. That pass resolves Tadabur's commit once and stamps it into
`<store>/<set>/revision.txt` before any unit is written, so a resume under a re-uploaded
dataset is refused rather than mixed, and the stamped commit is the two frame sets' `revision`
in `meta.json` — every committed `eval_overlap:gate_eval` / `eval_overlap:tadabur_frame` fact
names the audio it was measured against. It also writes `<store>/frames/index.json`, the
held-out shard list the store covers, and one disposition record per shard
(`frames/<NNN>.json`: the rows read, the rows per set, and the sealed-panel, undecodable and
too-short rows by id). The build refuses a store whose declared shards and records do not match,
or whose two frame stores hold anything but the fingerprints those records claim, and it lists
every frame row it could not fingerprint in the same blind spot. A row that will not decode is
skipped, not fatal; a gate row is the exception — its 2,000-clip population is an invariant, so
it raises. Every unit publishes its controls first and the file resume keys on last (a frame
shard's disposition record, a set unit's fingerprint archive), so an interrupted write is
recomputed instead of trusted.

Intra-tlog duplicates are equal `audio_sha256` across **all** rows (empty bytes are never a
key) plus verified near matches against an index of the frozen eval-role rows only; a
train–train near pair is not searched and that gap is in `meta.json`. Resolution is by
component: one survivor per component, the earliest-opened eval role first (`look`, then
`confirm.1` … `confirm.4`, then `train`), then the smallest `(audio_path, split, shard_file,
row)`. No eval row can lose to a training copy, and train–train exact duplicates are recorded
too. Survivor choice ignores usability by design; `meta.json` records how many usable rows lose
to an unusable survivor (`duplicates.survivor_unusable`) rather than changing the rule.
`tadabur.tlog_overlap.iter_pool_rows` joins the frozen row to its record as one view: S4's
fields under their own names (its `eval_overlap` list no longer collides with S1's boolean,
renamed `s1_eval_overlap`) plus the ordered `exclusions` union and one `usable` flag.

`meta.json` records the set census and provenance (source, pinned revision, checksums or
fingerprints, each with its SHA-256), the threshold and vote threshold, both controls — the
required synthetic-positive recall (≥ 200 re-encoded, re-cut copies through the real index,
≥ 99% of them verified as candidates and every candidate BER within the line; the run refuses
to write otherwise) and the stop-word match report — the BER histogram of every verified
candidate, `counts` inside the line and `rejected` above it, the ambiguous zone, hits per set
and role, duplicates by role pair, the survivor-usability counts, the net-size tables (all rows
usable before S4, and the gating-eligible subset), `usable_after_s4` per role with `train`
included, the blind spots, the tlog fingerprint store's census (the committed
`fingerprints.json`, which the build verifies npz by npz), and the frozen `split/` tree and
`pass1/meta.json` digests. Three pieces need no audio: the S1 id-overlap recall control verifies
every S1-flagged row that S4 missed against the Tarteel v1 clips sharing its recording id and
records the miss BER distribution and stop-word share; the stop-word report keeps the
all-candidate figure and adds the accepted (≤ line) share split into external hits and
near-duplicate links; and the survivor-usability counts (rows otherwise usable after S4, lost to
an unusable survivor) are read from the written tree.

Two resumable, CPU-only stages on cuda-dev (no GPU lock), driven by
`scripts/issue-189-overlap.sh`:

```bash
PR=NNN scripts/issue-189-overlap.sh launch      # stage HEAD as a clean clone on the box
scripts/issue-189-overlap.sh external           # stream + fingerprint every set (detached)
scripts/issue-189-overlap.sh check external
PR=NNN scripts/issue-189-overlap.sh build       # index, controls, records, meta (detached)
scripts/issue-189-overlap.sh check build
scripts/issue-189-overlap.sh fetch DIR          # pull the overlap tree back (no audio)
```

Or directly, from `tools/` on the box:
`python -m tadabur.tlog_overlap external --fp-dir DIR` then
`python -m tadabur.tlog_overlap build --pr <N> --out DIR2`. The build fails loudly if a set's
store holds a clip its checksums do not name, if the two frame stores disagree about Tadabur's
revision, if a frame shard has no disposition record or one that does not cover its rows, if
the tlog npz files are not the committed census's, if a tlog shard's fingerprint store is
short, or if the recall control misses. The offline tests need no network, GPU or audio:
`python -m pytest -q tools/tadabur/test_tlog_overlap.py`,
`python -m pytest -q tools/tadabur/test_tlog_overlap_build.py` (which drives `build` end to
end over a fabricated store and one real index shard), and
`python -m pytest -q tools/tadabur/test_tlog_overlap_launch.py`, which sweeps a stub box.

## Ayah identification and its calibration (#190, S5)

`ident/` is what `tadabur.tlog_ident` commits: `references.json.gz` (all 6,236 ayat' normalised
references and word offsets), `real_mistake_clips.jsonl` (the 52 clips holding the 56
owner-verified real-mistake sites, committed before any look clip was scored),
`decodes/real_mistake.base.json` and `decodes/noise.base.json` (the base teacher's whole-clip
decodes) and `ident_calibration.json`; the generated page is
`docs/tlog-ayah-identification.md`. The rules are the frozen hold-out text plus three dated
documents: `docs/tlog-ident-preregistration.md` (S5, pre-outcome), `docs/tlog-ident-bu-amendment.md`
(the owner's `b_u` erratum) and `docs/tlog-ident-margin-amendment.md` (the margin's competitor
definition). The frozen text still says "42 clips" at its real-mistake line; the committed store is
52 clips over 46 ayat, and the S5 documents are the record of that erratum.

```bash
python -m tadabur.tlog_ident references   # rebuild references.json.gz (~5 min)
python -m tadabur.tlog_ident calibrate    # floors, bars, counts and the page (~40 min CPU)
python -m tadabur.tlog_ident report       # re-render the page from the committed report
python -m tadabur.tlog_ident oracle       # the character-start margin bound (look only)
python -m tadabur.tlog_ident oracle --limit 100   # its cost check: prints the projection
```

**The margin's bound (S5b).** `ident/ident_calibration.json` records `margin_definition {path,
sha256}`: the competitor is the pinned **word-span** search of
`docs/tlog-ident-margin-amendment.md`, which discloses that it can overstate the margin, never
understate it. `ident/oracle_margin.json` records the one measured bound: every character start of
every free word run of every shortlisted window, `SHW`, attributed by the engine's own rule (one
end and one path per start, so a stricter bound, not the exact margin). It holds `b_u` (pinned),
`b_u_oracle`, `clips_m_changed` and `assignments_changed_at_b_u`; `oracle_consequence` turns those
into `margin_definition.status = "final"` only when the bound changed neither, else `"owner"`. The
oracle never chooses `b_u` or a label.

**Confirmed only (S5b, #204).** S5's binned floor stopped on its own noise bar (159 of 445
Freesound recordings passed = 35.7% against a 1% ceiling). S5b replaces it for **verdicts** with
the per-decode rule in `docs/tlog-ident-floor-preregistration.md`: a decode is
`non_recitation` when its normalised length is under `L_min = 8` characters or its own decoy is
within `DELTA = 0.30` of its real best from the same shortlisted search. `identify()` computes
the verdict; `floor_ok` / `id_floor_ok` are always booleans and every identification record
carries `floor_rule {path, sha256}`. The binned floors and the three empty-bin conventions stay
in the report as history. `tadabur.tlog_ident.load_calibration()` returns `(bars, k)` and
refuses any report whose `floor_status` is not `"confirmed"`, whose
`margin_definition.status` is not `"final"`, or whose margin definition has no recorded owner
approval: S6 and S7 may label under nothing else.

**The owner's margin approval.** The frozen floor pre-registration's §7 requires the owner to
approve the pinned margin definition at the S5b check-in before `b_u` is final, which the oracle's
mechanical `final` cannot supply. The approval is its own small committed record,
`ident/margin_approval.json`, written by the owner with:

```bash
python -m tadabur.tlog_ident approve --issue 204   # the owner runs this
```

after the amendment, the floor document and the calibration are committed (it refuses a dirty
tree or an existing record). The record is:

```json
{"schema": "tadabur.tlog_ident/margin-approval-v1",
 "doc": {"path": "docs/tlog-ident-margin-amendment.md", "sha256": "<the pinned SHA-256>"},
 "b_u": 0.01, "approved_on": "YYYY-MM-DD", "approved_by": "owner", "issue": 204}
```

It is **absent until the owner approves** (this slice's PR adds the gate and the command, not the
record). The loader validates the schema, `approved_by == "owner"`, the date, the document's
path and SHA-256 **and the `b_u` it clears**, so a recalibration that moves `b_u` needs a new
approval. A matching record clears the gate with no recalibration; a stale SHA-256 or `b_u`
refuses again.

**The S6/S7 precondition.** #191 and #192 may label only once this PR is merged *and* the owner's
approval commit is on main: `load_identifier()` (the consumer entry point) refuses until then.
The approval commit is recorded as an explicit blocker on both issues.

**The fresh confirmation draws.** `ident/confirm/freesound.json` and
`ident/confirm/common_voice.jsonl` (with its `common_voice.json` counts sidecar) are drawn and
committed before any of them is decoded, under the salt `tlog-s5b-floor-confirm-v1`: 1,000
Freesound recordings from the 100 shared shards, 500 Common Voice Arabic clips and 500 non-Arabic
clips (100 each from `en`, `ur`, `fa`, `tr`, `id`, with `es`, `hi`, `ckb`, `sw`, `bn` as one-shot
substitutes). No audio is committed.

**The five bars.** `ident_calibration.json` → `confirm` records them: the three negative pass
rates with Wilson 95% intervals, the 52 real-mistake clips, look retention, the per-language
rates, the rule's catches split into `length` / `margin`, the items that passed with their decodes
and spans, and the doc → manifests → stores provenance chain (`scoring.commit` is the commit the
calibration ran from). `floor_status` is that status, `bars.provisional` is true unless it is
exactly `"confirmed"`, and the page shows the five bars above S5's binned stop as history.
`docs/tlog-ident-floor-erratum.md` records the three disclosures the confirmed run required: the
pre-registration's look figures are S5's `dc602df` measurement, one search change (`659bce0`) from
the re-measured 99.63%/99.50%, the Quran-text guard does not match Urdu/Persian letter variants
(the folded check excludes none of the committed draw), and `samples_sha256` is host-scoped.

**The retired bins (history).** The binned floors stay in the report as history; no verdict reads
them. They are keyed on the **decode's** character length, and the look half's longest decode is
138 characters, so `161-` never had a floor. Neither committed pool fact is the decode's length:
`iter_frozen_rows` joins the clip's `duration_s` and the pass-1 VAD `speech_s`, and characters
per second vary between clips, so a short-speech row can still decode past 160 characters. The
page's "The pool's VAD speech duration (indicative)" section therefore reports the speech
histogram with no claim about a decode's length; S7's decodes measure the real lengths (#192).

**Decode mode.** The frozen rule fixes a whole-clip greedy decode, and both the look decodes and
the confirmation sets are whole-clip (the Freesound arm repeats the 445 pool's `stream_b0` so it
confirms the same condition). S6's amendment decided it for the pool: whole-clip up to a measured
`D_max`, with a row above it recorded as `not_decoded:over_d_max` and no label (see the S6
section below). `identify` takes a bare string, so every committed row also carries which mode
produced it and no tree can mix modes silently.

Consumer contract: `identification_record()` fixes the committed row's field names and
`load_calibration()` / `load_identifier()` the consumer gates.

## The S6 decodes, sample, 5% rule and audit (#191)

`tadabur.tlog_ident_run` is the first content read of tlog. Its rules are the frozen
pre-registration plus `docs/tlog-s6-amendment.md` (Q1 decode limit, Q3 strata, Q4 verdict
referent; it also governs S7). The amendment and the measured
`ident/decode_limit.json` are committed **before** the first tlog decode.

The code is three modules, so S7 imports only what it needs:

- `tadabur.tlog_pool` — the pool pipeline: `select_rows(rows, roles=..., d_max=...)`,
  `decode_rows(rows, d_max=..., cache_dir=..., scratch=..., already=...)`,
  `identify_rows(rows, database=..., bars=..., k=...)`, `tree_provenance(pull_request=...)`,
  `read_ident_tree(population)`, `is_non_recitation(row)`, `decode_limit()` and the D_max
  measurement. Generic; it knows neither S6's population nor the audit.
- `tadabur.tlog_ident_run` — S6's population (`select`), the salted 2,000-row sample, the
  duration bands, the 5% rule and the `limit`/`decode`/`identify` CLI.
- `tadabur.tlog_audit` — S6's audit only: the strata draw, `stage`, verdict scoring and
  `render_audit`, with the `audit`/`stage`/`report` CLI.

**The S7 reuse rule.** Reuse S6's committed decodes verbatim: pass them as
`decode_rows(..., already=...)`, never re-decode a row S6 decoded. A second GPU pass need not
reproduce the first bit for bit, and the pool tree must agree with the rows the audit heard.
Write one tree per population and read it back by name with `read_ident_tree`;
`identification_record()` fixes the row's fields, so every row names its `decode_mode`,
`label_source` and `assigned_label`. Every cited commit is recorded with its PR number and
`refs/pull/<N>/head` (`tlog_index.provenance`), because the squash merge drops branch commits.

- **Decode mode:** whole-clip only. D_max is measured on seeded synthetic audio on the GPU box
  (ladder 60–300 s, production settings) and is the largest rung keeping 1.5 GiB of headroom.
  A row above D_max is not decoded or identified: it carries
  `decode_mode = "not_decoded:over_d_max"` and `label_source = "none"`. If more than 1% of the
  usable pool is above D_max, the run stops for the owner. Every row and every identification
  carries `decode_mode`.
- **Look:** every usable `look` row at or below D_max (9,710 before the cap).
- **Training sample:** the first 2,000 usable `train` rows at or below D_max by salted rank of
  `(split, shard_file, row)`; salt `tlog-s6-2026-10-27-v1`. The 5% rule's Wilson 95% upper
  bound over the sample's `non_recitation` rate is recorded in
  `ident/train_sample.meta.json`; above 5%, the follow-up is a new overlay commit, not S6 work.
- **Audit:** the frozen strata, drawn by salted rank from look and the sample only. A stratum
  short of its target takes all its rows, and the freed slots go to `both`. A row that satisfies
  two predicates is drawn in both and served once; the §9 sample is the first 25 floor-failing
  rows over the whole flag, marked `floor_sample` in the worklist. The page hides the stratum and
  shows the candidate ayat unattributed; the server derives right/other ayah from the ayah the
  owner picks. The page reads the Uthmani text from `quran.db` (a gitignored release asset copied
  into the scratch root) and refuses to start if a candidate label has no text. `report` renders
  `docs/tlog-audit.md` from the committed worklist and verdicts and deletes the staged audio.

`decode_limit.json`, the six trees and the two audit files are the slice's committed outputs.
No tlog audio is committed.

```bash
# GPU box, under /root/scratch/gpu.lock; `scripts/issue-191-ident.sh` drives these detached.
python -m tadabur.tlog_ident_run limit --out tools/tadabur/tlog/ident/decode_limit.json
python -m tadabur.tlog_ident_run decode --cache-dir /root/scratch/tlog/s6/cache --pr 207
python -m tadabur.tlog_ident_run identify --pr 207   # after margin_approval.json is on main
python -m tadabur.tlog_audit audit
python -m tadabur.tlog_audit stage --draw /root/scratch/tlog/s6/draw.jsonl \
    --audio-dir /root/scratch/tlog/s6/audio --cache-dir /root/scratch/tlog/s6/cache
python -m tadabur.tlog_audit_ui --worklist tools/tadabur/tlog/audit/worklist.jsonl \
    --audio-dir /root/scratch/tlog/s6/audio \
    --verdicts /root/scratch/tlog/s6/verdicts.jsonl \
    --quran-db /root/scratch/tlog/s6/data/quran.db --host 0.0.0.0 --port 8000
# after a sitting, `scripts/issue-191-ident.sh ui fetch DIR` pulls the verdicts; commit them at
# tools/tadabur/tlog/audit/verdicts.jsonl, then `report` renders the doc from the committed copy.
python -m tadabur.tlog_audit report --audio-dir /root/scratch/tlog/s6/audio
```

## The S7 confirm tranches, manifests and eval uses (#192)

`tadabur.tlog_tranches` applies S5's frozen identification **mechanically** to `confirm.1` …
`confirm.4` with the base teacher only, and writes the five manifests of the hold-out. Its
rules are `docs/tlog-gating-amendment.md` (owner-approved 2026-10-28; it REPLACES the frozen
pre-registration's gating rule) plus the S6 amendment's decode mode
(`docs/tlog-s6-amendment.md` §1) — both never edited in place.

- **Decode:** every usable row of each confirm tranche, whole-clip up to D_max; a row above it
  is `not_decoded:over_d_max` with no label and no floor verdict. The 1% D_max stop is checked
  **per tranche**. `look` is not decoded again: S7 reuses S6's trees verbatim.
- **The teacher:** only `tlog_pool.decoder()` (`obadx/muaalem-model-v3_2@01a1ef9f`). No
  student, C95 or candidate model may decode a confirm row; a locked AST test checks the
  module's imports, and the trees' metas record the base model and revision.
- **Manifest rows:** one per usable row of a role, never omitted, with the row key,
  `audio_sha256`, `group_key` (the S1 id group's smallest `audio_path`, from `split/`; the
  **bootstrap cluster is the surah** under the owner's S2 fallback — `assigned_label`'s,
  counted in the meta as `gating_surahs`), `filename_label`,
  `assigned_label`, `label_source`, `straddles`, `decode_mode`, `id_floor_ok` and one
  `stratum`. The strata are single-valued in a fixed precedence: `reported:unclean`, then
  `reported:undecided` (`label_source == none`), then `reported:straddles`, then `gating`
  (`label_source == both` — amendment §1), then `reported:identified`, then
  `reported:below_bar` (the old below-bar rows); `STRATA` is only the summary order.
  `tranches/meta.json` counts every stratum per
  role, plus `gating ∧ id_floor_ok` (amendment §5's sensitivity line), the gating surah count
  (the cluster count), the pre-registration, amendment, calibration and decode-limit SHAs, the
  split/overlap overlay commits and the timed slice.
- **Eval uses:** `exposure/tlog.<role>.clips.jsonl` lists **every** manifest row of the role —
  the whole role is reserved for evaluation, whatever its stratum. `surah_ayah` is the assigned
  label or `unknown`, the dataset is `tarteel-ai/tlog` at the recorded revision, and the reciter
  is `unknown`. One clip carrying two ayahs inside a use, or appearing in two uses, is an error.
- **Claim order:** a future round writes `tranches/<role>.claim.json`
  (`{role, issue, commit, date}`) when it opens a tranche; `require_claim_order` refuses
  `confirm.k` until `confirm.(k-1)` is claimed, so one round may claim two consecutive
  tranches. S7 writes no claim.
- **Sealed discipline:** logs, failure messages and the PR carry aggregate counts and row keys
  only — never a label, a decode or audio. Nobody listens to or inspects a tranche's items.

Trees follow S6's layout (`ident/decodes/confirm.k/` and `ident/confirm.k/`); this is a
deliberate deviation from the issue's `tlog/decodes/`, so one tree layout serves both slices.
`ident/confirm/` (S5b's Common Voice and Freesound) is a different population: a tree is read by
its exact name, never a `confirm*` glob (a test enforces it).

```bash
# GPU box, from tools/; `scripts/issue-192-ident.sh` drives these detached (decode holds
# /root/scratch/gpu.lock). Commit the decodes, then the ident trees, then run manifests.
python -m tadabur.tlog_tranches decode --cache-dir /root/scratch/tlog/s7/cache --pr 211
python -m tadabur.tlog_tranches identify --pr 211
python -m tadabur.tlog_tranches manifests --pr 211
```

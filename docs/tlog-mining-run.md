# tlog mining: run note

**Status (2026-11-20, 14:02:09Z):** phases A, B, C and D are complete. Each run wrote
`run.exit = 0` and the launcher fetched its merge; this note is committed with the output it
describes.

Issue: [#263](https://github.com/sysofwan/HifzGuide/issues/263) (parent #261). Phase A is the
teacher-only timing pass [`docs/tlog-mining-amendment.md`](tlog-mining-amendment.md) §5 allows
without a claim. Phase C ran after round 3's verdict was committed `pass` (PR #289) and its
committed choice named `r3_candidate_ema`; its section is below. Phase D is phase A's alignment
again over phase B's rows, run in idle GPU time so [#264](https://github.com/sysofwan/HifzGuide/issues/264)
can snip a single word from a mined clip; its section follows phase C's.

## Phase A: the teacher's word times

`tools/tadabur/tlog_mining/word_times/`, produced by
`scripts/issue-263-mining-decode.sh` (`tadabur.tlog_mining`):

| file | SHA-256 |
|---|---|
| `look.jsonl.gz` | `09ae1a578fc86dab360eeaa93ad10241ac7036cbc698fe0d20fc1e4b9b0e077d` |
| `confirm.1.jsonl.gz` | `52d5619727b65ed9b0d5db48cc8285fbb5c246823dca528ffc4968b818f1b769` |
| `confirm.2.jsonl.gz` | `9a5206e0ff98c1c7a5d52fab029842ff21f9eb8b526a8c5d99ff9d0250d94e32` |
| `meta.json` | `5e84888f89e591cb08a2063795a2cb080b187bd01ec1249985eafdafc940ca06` |

`meta.json` pins `code_sha` `469ab906955a651a726e3eeac4f3f09098eae0bd` and the teacher's
identity, and records **no pull request**: `pull_request` and `pull_request_ref` are null
because #263 is the issue, not a PR, and `refs/pull/263/head` would not resolve. The branch
`agent/mine-fetch` carries the SHA until a PR exists; the launcher passes `PR=<n>` only then.

**Rows are the gating stratum of each tranche**, cross-checked against
`tools/tadabur/tlog/tranches/meta.json` and the role manifests. Counts match exactly, and every
gating clip carries a record — aligned or listed:

| tranche | gating rows | aligned | fallback (`aligned: false`) | fallback rate |
|---|---|---|---|---|
| look | 2,853 | 2,767 | 86 | 3.01% |
| confirm.1 | 2,843 | 2,755 | 88 | 3.10% |
| confirm.2 | 2,919 | 2,820 | 99 | 3.39% |
| total | 8,615 | 8,342 | 273 | 3.17% |

The reference is the filename ayah's full text, phonetized spaceless with
`waqf_segments.hafs_word_reference`; the base teacher's frame posteriors are aligned with
`training.ctc_paths.ctc_best_path` and the labels mapped to Uthmani word offsets. Each record
carries `key`, the manifest provenance (`split`, `shard_file`, `row`, `audio_path`,
`audio_sha256`, `surah_ayah`), `word_times`, `seconds_per_frame`, `ll_per_frame`, and `aligned`
/ `reason`. The times are a function of the reference and the teacher only: no decoded mark is
read, so the set is reproducible and cannot leak a model's output through a snip boundary.

**Fallbacks have two causes, both committed as rows rather than raised:**

- **241 short posteriors** (look 74, confirm.1 80, confirm.2 87): `N frames cannot hold the
  M-label transcript`. The clip is shorter than the reference's phoneme count allows, so no
  alignment exists.
- **32 phonetizer refusals** (look 12, confirm.1 8, confirm.2 12): the phonetizer raises on
  these ayahs — 106:1–106:4, 55:17, 90:8, 90:9, 90:10 — which is #248's failure, recorded here
  per clip. A re-read of that issue, not a phase-A change, resolves them.

### Compute and give-way

The run's teacher is `obadx/muaalem-model-v3_2@01a1ef9fbe40d144ef845101e89ff924aed3fef5`
(fingerprint in `word_times/meta.json`: whole-spans, bf16 weights, fp32 features, batch 1), and
the committed merge records `code_sha` `469ab906955a651a726e3eeac4f3f09098eae0bd`.

Two attempts were needed. **The first aborted on its own memory, not round 3's:**

- Launched 2026-11-02 13:47:49Z at `P=2` on the pre-#272 code. It finished look and confirm.1
  and most of confirm.2, then at **15:03:25Z** confirm.2 worker 1 raised
  `torch.OutOfMemoryError` inside `wav2vec2_bert` self-attention: "this process has 5.97 GiB …
  Process 725252 has 8.85 GiB … total capacity 15.48 GiB", and the parent raised
  `confirm.2 worker 1 exited 1`. It had written **8,029 of the 8,615 rows**; the resume wrote the
  other 586, and the first attempt's `wrote` lines sum 396 short of 8,029 — 291 confirm.1 rows
  from a batch the parent reaped at its deadline without terminal lines, and 105 confirm.2 rows
  from the crashed worker — so the split cannot be read off the lines alone.
- **#272** (`469ab90`, merged 15:11:55Z) sizes the whole-clip workers from the card's free
  memory and caps the count (`WORKER_VRAM_MIB = 9500`); it changes how many clips are in
  flight, never how one is aligned. The resume restaged the box at `469ab90` (15:20:37Z),
  printed `parallel capped to 1: whole-clip workers need 9500 MiB each`, and continued by row
  key: the remaining **586 rows** in two worker batches (`wrote 410`, then `wrote 176,
  done=True`), 15:20:38Z → 15:29:06Z, exit 0, then the merge.

The split comes from the final totals, not from part-file mtimes (parts are appended across
attempts) or a line sum: 8,615 records minus the resume's two logged batches. Per tranche the
first attempt held 2,853 look rows, 2,843 confirm.1 and 2,333 confirm.2 (2,919 − 586).

**GPU time.** The resumed attempt's line is the measured one: `batches=3 gpu_seconds=431
yielded_seconds=60` — 0.12 GPU-h held and 60 s yielded, against 508 s wall, for 586 rows. The
aborted attempt's line was lost with the crash, so only a bound is recoverable. Its batches are
counted from the 28 teacher-load starts before the resume marker: **14 `P=2` worker
batches — look 1, confirm.1 8, confirm.2 5**. Twelve stopped at the 300 s budget (`BATCH_SECONDS`:
seven confirm.1, five confirm.2, including the OOM batch, whose surviving worker printed
`wrote 426, done=False`); look's single batch and confirm.1's last finished early (`done=True`).
The first attempt therefore held between 3,600 s (twelve full budgets) and 4,340 s (fourteen at
the 310 s deadline ceiling). With the measured 431 s, **the total is 4,030–4,770 lock-held
seconds ≈ 1.12–1.33 GPU-h** — a bound, not a measurement, and the pre-registration's ≲ 1 GPU-h
estimate was optimistic. Wall time was ≈ 84 min across both attempts (75.6 min before the
abort, 8.5 min after). These are lock-held seconds rather than pure compute: clip fetches
inside a batch count too.

**Time yielded to round 3.** 60 s measured — 12% of the resumed run's held-plus-yielded 491 s.
The 15:03:25Z–15:20:38Z window was a dead run, not a yield, and round 3's smoke gate used it
(15:05:24Z–15:20:15Z, `/root/scratch/issue-252/logs/events`). Phase A therefore cost round 3 no
GPU time; the box was idle for round 3's own run.

### Audio

8,615 clips (2.3 GB) were fetched to `/root/scratch/tlog-mining/clips/` by provenance, hardlinked
from round 2's look clip cache where present, and every clip's SHA-256 was checked before
decoding. **No audio is committed and no NAS copy was made** (#262's owner answers). The clips
were deleted from the box once this commit was pushed. The dataset revision is not in the
committed word-time files: the pin is `TLOG.revision` in `tools/tadabur/nonpro_sources.py`,
`9fe4f7e6d1b372c2c11157d6d2f0f91ca344eb51` at the recorded `code_sha`. Each row carries its
`split`, `shard_file`, `row`, `audio_path` and `audio_sha256`, so the audio re-downloads at that
revision and is checked against the row's checksum.

## Phase B: the confirm.1/2 pool extension

`tools/tadabur/tlog_mining/decodes/`, produced by the same launcher (`run --phase B`, then the
merge) from the staged clone at `94e7ba0b4fc4ae28e71a1045a4eba5fffa948de3`:

| file | SHA-256 |
|---|---|
| `tlog.mined.base.jsonl.gz` | `a31c4f0535c0bd3a4cd8ff6e7d1b6578fb03e2757e0530a7263a1b6784e49448` |
| `tlog.mined.c95.jsonl.gz` | `b9b1e3526a77201a91725943062d3bf4faf15a30bf1b1bc2622321cff0444607` |
| `tlog.mined.r2_candidate_ema.jsonl.gz` | `4630d0902c96c884f01357432036522d5d64ebc636f7ca85ffe157f6c9c12150` |
| `meta.json` | `f879be7520ff3c1930a28ab449455a5648037d6c3f5d6def8aa82062b57457d3` |

`meta.json` pins `code_sha`, the producer set (`base`, `c95`, `r2_candidate_ema`), the two
tranches, the per-model row count, the `label_source` tallies and the 26 listed rows. Its
`identities` section carries each producer's identity — the teacher at
`obadx/muaalem-model-v3_2@01a1ef9f…`, C95 and the EMA candidate at their checkpoints — with the
`checks` and `spans` fingerprints (bf16 weights, fp32 features, batch 1). `pull_request` and
`pull_request_ref` are null for the same reason as phase A.

**Rows.** `phase_b_rows` selects the labelled `reported:*` rows of confirm.1 and confirm.2:
10,668. The 26 whose `assigned_label` does not phonetize (#248) are listed in `meta.json` and
never fetched or decoded, so each file holds **10,642 rows** — the same 10,642 unique keys in
all three:

| tranche | mineable rows | decoded | listed (#248) |
|---|---|---|---|
| confirm.1 | 5,341 | 5,325 | 16 |
| confirm.2 | 5,327 | 5,317 | 10 |
| total | 10,668 | 10,642 | 26 |

The listed rows are the failure phase A recorded per clip, the same eight ayat (106:1–106:4,
55:17, 90:8, 90:9, 90:10): 15 straddles, 6 labelled unclean, 4 below-bar and 1 identified. The
phase B fallback rate is therefore **0.24%** of the mineable pool.

| stratum | rows | `label_source` |
|---|---|---|
| `reported:straddles` | 4,256 | `both` 2,634, `identified` 806, `filename` 816 |
| `reported:unclean` | 3,840 | `identified` 3,840 |
| `reported:below_bar` | 1,719 | `filename` 1,719 |
| `reported:identified` | 827 | `identified` 827 |

Per `label_source`: **identified 5,473, both 2,634, filename 2,535**. The source follows the
stratum — `both` only on straddles, `filename` on below-bar and straddles, `identified` on
identified and unclean — because the stratum is where the reference was read.

**Shape.** Each record carries round 2's confirm-decode keys — `key`, `tranche`, `stratum`,
`split`, `shard_file`, `row`, `audio_filename`, `audio_sha256`, `start_sample`, `end_sample`,
`surah_ayah`, `word_from`, `word_to`, `assigned_label` — plus the two phase B fields: the
reference's `label_source`, and `spans`, the whole-clip span decode. `checks` is the app's check
sequence. Both stores are per model, so the three files differ in `checks` and `spans` and agree
on everything else.

### Compute and give-way

`run.out`'s line: `stopped=done batches=264 gpu_seconds=80547 yielded_seconds=12900`, and the
merge's counts (`label sources: {'filename': 2535, 'both': 2634, 'identified': 5473}`,
`unphonetizable: 26 rows listed (#248)`, 10,642 rows for each model) match the committed files.

**The launcher asked for `P=2` and the code capped it to 1**, its first log line: `parallel
capped to 1: phase B holds three decoders and its peak is unmeasured`. Phase B therefore ran one
whole-clip worker at a time — the position the #272 VRAM cap leaves it in until three resident
decoders' peak is measured.

**GPU time: 80,547 lock-held seconds = 22.37 GPU-h**, over 263 worker batches plus the terminal
empty iteration `batches=` counts (mean 305 s held per increment — essentially every worker ran
its 300 s `BATCH_SECONDS` budget out). Each row costs 7.57 s of GPU across the six decodes it
needs (two stores × three models); the pre-registration's ~4–6 GPU-h estimate assumed far less
per row and did not survive contact with a `P=1` worker decoding `checks` and whole-clip `spans`
for three models.

**Time yielded to round 3: 12,900 s = 3.58 h**, 13.8% of the held-plus-yielded 93,447 s. Every
increment is one 30 s `yield_seconds` sleep: 263 after a worker batch, and 167 before a lock try
whose non-blocking acquire found the lock taken. `batches=264` is 263 worker invocations plus the
iteration that finds no row left — it acquires the lock, starts no worker, returns, and does not
sleep. A failed acquire observes the lock only at that instant: the holder can release it moments
later, leaving most of the following sleep idle, so the 167 failures are yield attempts, not a
measure of round 3's occupancy. The measured yield is the 12,900 s itself. Wall time was
26 h 09 m 10 s (2026-11-03T00:27:29Z → 2026-11-04T02:36:39Z); the 703 s outside held-plus-yielded
is the loop's gates checks, lock tries and start-up.

**One batch shape worth knowing.** The batch loop raises a fresh worker per batch, so the
decoders reload from disk each time: the log repeats the 816-tensor `Loading weights` group 263
times — one per worker batch. The loads are fast (≈9,000 tensors/s), so they are not the GPU
cost — the cost is the six per-row decodes at one worker.

### Audio

10,642 clips (5.0 GB) were fetched to `/root/scratch/tlog-mining/B/clips/`, hardlinked from
round 2's clip cache where present. Each is stored under its recorded `audio_sha256` and checked
before its rows decode: a mismatch raises, and the run exited 0 with a record for every row, so
all 10,642 matched. The 26 listed rows were never fetched. **No audio is committed.** The clips
are still on the box (disposable scratch, re-downloadable at the pinned revision); the 162 MB of
parts are what make a resume possible.

## Phase C: the round-3 chosen weights

`tools/tadabur/tlog_mining/decodes/`, produced by the same launcher (`run --phase C`, then the
merge) from the staged clone at `29592f904b30561aee9410fe5be8b4e97a6d5236`. That is the
implementation commit this section's code describes; main's #291 squash carries the same phase C
code, so the merge that brings this note in does not change what ran.

| file | SHA-256 |
|---|---|
| `tlog.mined.r3_candidate_ema.jsonl.gz` | `427e76e7f0c5637ac6c9954ebbbd34f41c5e89081e64b847a7a561c5abdf4066` |
| `tlog.gating.r3_candidate_ema.jsonl.gz` | `b6f383bb825502f1e99b006d4eade61bdc26bde90273692f1b2ea7c8d706be80` |
| `meta.r3_candidate_ema.json` | `abea387728cde8043beabdb5dbab3026f8162e50e34c0c0ddc149352a713fcb0` |

`meta.r3_candidate_ema.json` pins `code_sha` `29592f904b30561aee9410fe5be8b4e97a6d5236`, the two
tranches, the one model, the per-file row counts, the `label_source` tallies and the 26 listed
rows. Its identity section names the weights that actually decoded: `r3_candidate_ema` at `ema`,
step 10,000, recipe `aug94-mixture-v2`, checkpoint
`1193e2117b56381ccc078a5b5e21d61d0ad0903565dcca0cb110b39ce24ed4a9` (the round-3 spec's
`R3_CANDIDATE_SHA256`), ref `/root/scratch/issue-252/runs/consonant3_candidate/checkpoint.pt`,
with `checks`, `spans` and confirmed-stream fingerprints (bf16 weights, fp32 features, batch 1).
`pull_request` and `pull_request_ref` are null for the same reason as phases A and B.

The one `tlog.mined/r3_candidate_ema` identity covers both committed files: the gating decode ran
the same weights, so its producer is that record. Unlike round 2's per-artifact
`tlog.confirm/<model>` keys, `meta.r3_candidate_ema.json` has no separate `tlog.gating/` entry by
decision (the locked test pins the single-key set).

The launcher refuses phase C unless round 3's choice is committed at that candidate and its
verdict is committed `pass`, and both were; nothing read the output before the verdict, and this
note is the first readout of it.

**Rows.** Phase C decodes exactly what round 2's gating decodes and phase B's mined decode hold,
so the round-3 weights line up with base/C95/r2 row by row: `phase_c_rows` takes phase A's gating
rows and `phase_b_rows`' decodable rows unchanged. The gating keys are exactly round 2's
`tlog.confirm.*` keys (5,762) and the mined keys exactly phase B's `tlog.mined.base` keys
(10,642); the two sets are disjoint, and each appears once.

| tranche | gating | mined | decoded | listed (#248) |
|---|---|---|---|---|
| confirm.1 | 2,843 | 5,325 | 8,168 | 16 |
| confirm.2 | 2,919 | 5,317 | 8,236 | 10 |
| total | 5,762 | 10,642 | 16,404 | 26 |

The 26 listed rows are phase B's, the same eight ayat (106:1–106:4, 55:17, 90:8–90:10); they are
listed here and never fetched. Per `label_source`, over the mined rows only: **identified 5,473,
both 2,634, filename 2,535** — phase B's tallies unchanged. The gating rows carry no
`label_source`: they are the gating stratum, not a labelled report.

**Shape.** The mined file is phase B's record shape — round 2's confirm-decode keys plus
`tranche`, `label_source` and `spans`. The gating file is round 2's confirm record shape —
`key`, `tranche`, `stratum`, `split`, `shard_file`, `row`, `audio_filename`, `audio_sha256`,
`start_sample`, `end_sample`, `surah_ayah`, `word_from`, `word_to`, `assigned_label`, `checks` —
plus `spans`. The keys match round 2's gating decodes exactly; the shape is round 2's plus
`spans`, which is the accepted shape by decision (the gating stratum decodes both stores),
pinned by the locked test — consumers join the gating rows on `key` and ignore `spans`.

Two consequences for #261's readout, stated here so they are not found the hard way:

1. **Join the gating rows on `key`.** Round 2's gating decodes are `checks`-only, so base/C95/r2
   have no gating `spans` counterpart; a naive comparison of every field of every line trips on
   the extra `spans` field. `checks` is comparable as it stands.
2. **The gating `spans` cost about a third of the run and buy nothing comparative.** 5,762 of the
   16,404 row-decodes (35.1%) are gating rows taking the whole-clip `spans` pass, for which no
   other model has a gating counterpart. They are kept because the issue table says phase C is
   "checks + spans" and the locked test pins the shape.

### Compute and give-way

`run.out`'s line: `stopped=done batches=130 gpu_seconds=39017 yielded_seconds=3870`, and the
merge's counts (`label sources: {'filename': 2535, 'both': 2634, 'identified': 5473}`,
`unphonetizable: 26 rows listed (#248)`, 10,642 mined and 5,762 gating rows) match the committed
files.

**The launcher asked for `P=2` and the code capped it to 1**, its first log line: `parallel
capped to 1: phase C's worker peak is unmeasured`. `plan C` prints the requested count (`at 2
processes`), so the preview is an upper bound on the process count, never the count that runs.

**GPU time: 39,017 lock-held seconds = 10.84 GPU-h**, over `batches=130` — 129 worker
invocations plus the terminal empty iteration `batches=` counts. Mean held per worker batch is
302 s, essentially every batch running its 300 s `BATCH_SECONDS` budget out. Each row costs
2.38 s of GPU across the two decodes it needs (two stores × one model), i.e. 2.38 s per
model-row, vs phase B's 2.52 s per model-row (7.57 s per selected row across its three models;
1.26 s per store-decode). The issue's ~5–6 GPU-h estimate for phase C was optimistic in the same
way phase B's was (estimated ~4–6, measured 22.37).

128 worker batches printed their terminal `decoded N` line; the 129th was reaped at the parent's
batch deadline without printing one (the only path that skips the line), and its rows resumed by
key. The merge holds every row exactly once, so no row was decoded twice into the committed
files.

**Time yielded to round 3: 3,870 s = 1.08 h**, 9.0% of the held-plus-yielded 42,887 s. That is
exactly 129 × the 30 s `yield_seconds`, which decomposes as one sleep after each of the 129
worker batches and none on a failed acquire: every non-blocking acquire found
`/root/scratch/gpu.lock` free, i.e. round 3 was idle for the whole run. Wall time was 12 h 06 m 27 s (2026-11-19T07:51:52Z → 2026-11-19T19:58:19Z); the 700 s
outside held-plus-yielded is the loop's gates checks, lock tries and start-up.

### Audio

Phase C's mined rows are phase B's, so the 10,642 mined clips were **hardlinked** from
`/root/scratch/tlog-mining/B/clips/` into `C/clips/` before the launch and re-verified by
SHA-256 as each row decoded: no bytes re-downloaded. The 5,762 gating clips (not in that cache)
were streamed from the pinned shards, ≈1.6 GB; `C/clips/` held 6.6 GB with the hardlinks
counted. The clips were deleted from the box once this commit was pushed. **No audio is
committed.**

## Phase D: the teacher's word times for the mined rows

`tools/tadabur/tlog_mining/word_times/mined.*` and `meta.mined.json`, produced by the same
launcher (`run --phase D`, then the merge) from the staged clone at
`e056fbdda1fefe8b912db969b9cfc9d4ec89248d` — the #301 squash merge, which is the code this
section describes.

| file | SHA-256 |
|---|---|
| `mined.confirm.1.jsonl.gz` | `5d8b275dd7f052da0d50598473326cf6b0327d7409b76c690409c52893caf541` |
| `mined.confirm.2.jsonl.gz` | `e2c801671faa9000d78faed29284b352821adb97f03ed65e0975a8b47057111a` |
| `meta.mined.json` | `01594bc682098d5149404a22bef5799f32a6947af0d664fa1ba7eb52490b2223` |

`meta.mined.json` pins `code_sha` `e056fbdda1fefe8b912db969b9cfc9d4ec89248d`, the base teacher's
identity (`obadx/muaalem-model-v3_2@01a1ef9fbe40d144ef845101e89ff924aed3fef5`, mode
`whole-spans`, bf16 weights, fp32 features, batch 1), the two tranches, the per-tranche counts,
the `label_source` tallies and the 26 listed rows. `pull_request` and `pull_request_ref` are
null, for the same reason as phases A–C: #263 is the issue, not a PR.

**Rows.** `phase_d_rows` is phase B's row set exactly — the labelled `reported:*` rows of
confirm.1/2 — aligned against each row's **`assigned_label`**, the reference phase B decodes and
the dataset is built on. (The filename label is missing for a third of these rows and names
another ayah for more, so phase A's filename reference would leave most of them unalignable.)
Every aligned record is phase A's shape, unchanged — so its `surah_ayah` is the **assigned**
label, while phase B's record of the same `key` keeps that row's filename label (absent for 3,840
of these keys, a different ayah for 1,633). A joiner takes `assigned_label` from phase B's record
and never compares the two files' `surah_ayah`.

| tranche | rows | aligned | fallback | fallback rate |
|---|---|---|---|---|
| confirm.1 | 5,325 | 4,418 | 907 | 17.03% |
| confirm.2 | 5,317 | 4,480 | 837 | 15.74% |
| total | 10,642 | 8,898 | 1,744 | 16.39% |

That is **88,140 words** with committed start/end times (9.9 per aligned clip on average); the
median aligned record's frame is 40.06 ms and its per-frame forward alignment log-likelihood is
−1.41.

**Not every aligned row is usable by #264.** Phase A aligns a known-correct transcript, the
gating clip's filename ayah. A mined row aligns its *assigned* ayah instead, and for the 5,473
`identified` rows — 51% of the 10,642 — that assignment came from content identification, with
no filename counterpart (3,840 rows) or a filename naming another ayah (1,633). Where the
reference is wrong, a clip long enough to hold the transcript aligns inside it silently instead
of falling back: `aligned: true` with a poor likelihood, never `aligned: false`. The committed
likelihoods separate the two:

| set | aligned rows | p5 | p25 | median | p75 |
|---|---|---|---|---|---|
| phase A, aligned | 8,342 | −3.49 | −0.65 | −0.23 | −0.07 |
| D, aligned | 8,898 | −7.25 | −3.06 | −1.41 | −0.52 |
| D, `both` | 2,575 | −4.86 | −1.64 | −0.75 | −0.32 |
| D, `filename` | 2,355 | −5.97 | −3.38 | −2.12 | −1.20 |
| D, `identified` | 3,968 | −8.83 | −3.83 | −1.52 | −0.46 |

The `identified` source is the worst of the three at every percentile, and all three sit well
below phase A's distribution. **The rule #264 applies:** a row whose `ll_per_frame` is below −3.5
(phase A's aligned 5th percentile is −3.49) is treated exactly as `aligned: false` — play the
whole clip. That is 1,861 of the 8,898 aligned rows (20.9%), so 3,605 of the 10,642 mined rows
(33.9%) play whole at presentation.

**All 1,744 fallbacks have the same cause**: the posteriors are too short for the assigned
ayah's transcript — CTC cannot place more labels than the clip has frames. A mined clip often
holds less than its assigned ayah (the draw took a span of it), so the full-ayah reference does
not fit. The failure is concentrated where the assignment is a *guess* rather than part of the
audio:

| stratum | rows | fallback |
|---|---|---|
| reported:identified | 827 | 304 (36.76%) |
| reported:unclean | 3,840 | 1,159 (30.18%) |
| reported:below_bar | 1,719 | 151 (8.78%) |
| reported:straddles | 4,256 | 130 (3.05%) |

and by reference source: `identified` 27.50% (5,473 rows), `filename` 7.10% (2,535), `both`
2.24% (2,634). Phase A's full-ayah gating clips fell back at only 3.0–3.4%; these clips are not
full ayahs. A fallback row is a committed record with `aligned: false` and its `reason`, so #264
plays the whole clip for it — the presentation fallback the pre-registration prescribes.

**The 26 listed rows.** Phase B's #248 refusal is unchanged: 16 in confirm.1 and 10 in
confirm.2, the eight ayat 106:1–106:4, 55:17 and 90:8–90:10, by stratum 15 straddles, 6 unclean,
4 below-bar and 1 identified. They are listed in `meta.mined.json` (tranche, key, filename,
`assigned_label`, stratum) and are in neither `mined.*` file; they were never fetched.

### Compute and give-way

`run.out`'s line: `stopped=done batches=59 gpu_seconds=17303 yielded_seconds=1740`.

**The launcher asked for `P=2` and the code capped it to 1**, its first log line: `parallel
capped to 1: phase D's mined clips are longer than phase A's, which OOM'd at two workers`.
`plan D` prints the requested count (`at 2 processes`), so the preview is an upper bound on the
process count, never the count that runs.

**GPU time: 17,303 lock-held seconds = 4.81 GPU-h**, over `batches=59` — 58 worker invocations
plus the terminal empty iteration `batches=` counts. Mean held is 298 s of the 300 s budget:
essentially every batch ran its budget out. That is **3.2× the brief's ~1.5 GPU-h estimate**, and
the cause is I/O, not alignment: 1.63 s held per row, where the forward pass is a fraction of
it. The 10,642 clips (5.0 GB) were **re-streamed** — phase B's and C's clip caches were already
cleared, and the surviving `/root/scratch/tlog/r2/clips` cache holds round 2's look gating clips,
not these confirm rows — inside the lock-held batches, so the fetch is counted as GPU time. Phase
A's resumed rate, with its clips hardlinked, was 0.74 s/row.

43 worker batches printed their terminal `wrote N` line, summing 8,901 rows; the other 15 were
reaped at the parent's batch deadline without printing one (the only path that skips the line,
and the only one that loses an in-flight row). The merge holds every row exactly once — 10,642
records for 10,642 rows, the 26 listed and never fetched — so the 1,741 rows those batches wrote
before being stopped are all in the committed files.

**Time yielded to round 3: 1,740 s = 0.48 h**, 9.1% of the held-plus-yielded 19,043 s. That is
exactly 58 × the 30 s `yield_seconds`, one sleep after each worker batch and none on a failed
acquire: every non-blocking acquire found `/root/scratch/gpu.lock` free, i.e. round 3 was idle
for the whole run. Wall time was **5 h 28 m 48 s** (2026-11-20T08:33:21Z → 2026-11-20T14:02:09Z);
the 685 s outside held-plus-yielded is the launch's selection and the merge's #248 phonetizer —
each phonetizes both tranches, four tranche-passes, ≈160 s for both sites on this workstation —
plus the merge's I/O over the 58 parts; the run instrumented neither this split nor the
launch/merge boundary. The teacher loads are **not** here: an align worker folds its selection
and `load_decoder()` into the budget it reports, so they are part of the 17,303 s.

### Audio

5.0 GB was streamed into `/root/scratch/tlog-mining/D/clips/`, stored under each recorded
`audio_sha256` and checked before the teacher saw it. The round-2 cache is look-gating only and
phase B/C's caches were gone, so no clip could be hardlinked. Nothing was deleted from the box;
**no audio is committed**.

## Changelog

- 2026-11-20: self-check r3 corrections. The phase D section now carries the per-label-source
  `ll_per_frame` percentiles (D's median −1.41 against phase A's −0.23) and the bar #264 applies
  — below phase A's aligned 5th percentile (−3.5) a row plays its whole clip, 3,605 of 10,642
  rows (33.9%). The yielded share is 9.1% of held-plus-yielded, the 685 s outside it is the two
  #248 phonetizer sites plus the merge's I/O (the teacher loads are lock-held), the two files'
  `surah_ayah` are stated to differ or be absent for 5,473 keys, and phase D's three checksums
  are pinned by `test_tlog_mining_phase_d_artifacts.py`.
- 2026-11-20: phase D committed. The note records the `P=1` cap, 10,642 rows aligned against
  `assigned_label` (8,898 aligned, 1,744 fallback — all posteriors too short for the full ayah),
  the 26 #248 rows listed, 4.81 GPU-h held over 58 worker batches and 0.48 h yielded to an idle
  round 3: 3.2× the ~1.5 GPU-h estimate, because 5.0 GB of clips was re-streamed inside the
  lock-held batches.
- 2026-11-19: phase C committed. The note records the `P=1` cap, 16,404 decoded rows (5,762
  gating + 10,642 mined) against phase B's 26 listed, the `label_source` counts, the
  gating-`spans` caveat for #261's key join, 10.84 GPU-h held over 129 worker batches and the
  1.08 h yielded to an idle round 3.
- 2026-11-04: review correction. The yield decomposition is 263 post-worker and 167
  failed-acquire sleeps — the 264th `batches=` increment is the terminal empty iteration, which
  starts no worker and sleeps not at all — and the failed polls are no longer read as a lower
  bound on round 3's occupancy. The 22.37 lock-held GPU-h and 3.58 yielded hours stand.
- 2026-11-04: phase B committed. The note records the `P=1` cap, 10,642 decoded rows against
  10,668 mineable (26 listed, 0.24%), the `label_source` counts, 22.37 GPU-h held over 263
  worker batches and the 3.58 h yielded to round 3.
- 2026-11-02: phase A committed; the note records the 15:03:25Z OOM abort, the `P=1` resume
  under #272, the measured and estimated GPU time, and the 60 s yielded to round 3.
- 2026-11-02: the row split and the GPU reconstruction corrected after review. The earlier
  attempt values (6,819 / 1,796) came from part-file mtimes, which do not partition attempts
  because parts are appended to; the final totals give 8,029 / 586. `meta.json` no longer
  records issue #263 as a pull request, and the dataset revision's location is stated.
- 2026-11-02: second review correction. The batch count rises to 14 (one confirm.1 batch was
  reaped without terminal `wrote` lines, so its 291 rows are not in the log sums), and the
  first attempt is now reported as a 3,600–4,340 s bound rather than a point estimate; the OOM
  batch also ran to its budget.

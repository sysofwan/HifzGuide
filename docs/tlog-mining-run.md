# tlog mining: run note

**Status (2026-11-04, 02:36:39Z):** phases A and B are complete. Each run wrote `run.exit = 0`
and the launcher fetched its merge; this note is committed with the output it describes.

Issue: [#263](https://github.com/sysofwan/HifzGuide/issues/263) (parent #261). Phase A is the
teacher-only timing pass [`docs/tlog-mining-amendment.md`](tlog-mining-amendment.md) §5 allows
without a claim. Phase C did not run — it waits for round 3's verdict and its committed
live/EMA choice — and is not covered here; its section is added when its own run finishes.

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

## Changelog

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

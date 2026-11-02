# tlog mining: run note

**Status (2026-11-02, 15:29:06Z):** phase A is complete. The box wrote `run.exit = 0` and the
launcher fetched the merge; this note is committed with the output it describes.

Issue: [#263](https://github.com/sysofwan/HifzGuide/issues/263) (parent #261). Phase A is the
teacher-only timing pass [`docs/tlog-mining-amendment.md`](tlog-mining-amendment.md) §5 allows
without a claim. Phases B and C did not run and are not covered here; their sections are added
when their own runs finish.

## Phase A: the teacher's word times

`tools/tadabur/tlog_mining/word_times/`, produced by
`scripts/issue-263-mining-decode.sh` (`tadabur.tlog_mining`):

| file | SHA-256 |
|---|---|
| `look.jsonl.gz` | `09ae1a578fc86dab360eeaa93ad10241ac7036cbc698fe0d20fc1e4b9b0e077d` |
| `confirm.1.jsonl.gz` | `52d5619727b65ed9b0d5db48cc8285fbb5c246823dca528ffc4968b818f1b769` |
| `confirm.2.jsonl.gz` | `9a5206e0ff98c1c7a5d52fab029842ff21f9eb8b526a8c5d99ff9d0250d94e32` |
| `meta.json` | `5fd6743f800f30081cf3aff4443a00cdf6d82deb581f6a9c966548cd26b89e2c` |

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

## Compute and give-way

The run's teacher is `obadx/muaalem-model-v3_2@01a1ef9fbe40d144ef845101e89ff924aed3fef5`
(fingerprint in `word_times/meta.json`: whole-spans, bf16 weights, fp32 features, batch 1), and
the committed merge records `code_sha` `469ab906955a651a726e3eeac4f3f09098eae0bd`.

Two attempts were needed. **The first aborted on its own memory, not round 3's:**

- Launched 2026-11-02 13:47:49Z at `P=2` on the pre-#272 code. It finished look and confirm.1
  and most of confirm.2, then at **15:03:25Z** confirm.2 worker 1 raised
  `torch.OutOfMemoryError` inside `wav2vec2_bert` self-attention: "this process has 5.97 GiB …
  Process 725252 has 8.85 GiB … total capacity 15.48 GiB", and the parent raised
  `confirm.2 worker 1 exited 1`. 6,819 of the 8,615 rows were already in the parts.
- **#272** (`469ab90`, merged 15:11:55Z) sizes the whole-clip workers from the card's free
  memory and caps the count (`WORKER_VRAM_MIB = 9500`); it changes how many clips are in
  flight, never how one is aligned. The resume restaged the box at `469ab90` (15:20:37Z),
  printed `parallel capped to 1: whole-clip workers need 9500 MiB each`, and continued by row
  key: the remaining 1,796 rows in 89 parts, 15:20:38Z → 15:29:06Z, exit 0, then the merge.

**GPU time.** The resumed attempt's line is the measured one: `batches=3 gpu_seconds=431
yielded_seconds=60` — 0.12 GPU-h held and 60 s yielded, against 508 s wall. The aborted
attempt's line was lost with the crash; from the two attempts' throughput (1,796 rows in 431 s;
6,819 rows in 317 parts before the abort) it ran about 1,640 GPU-s ≈ 0.45 GPU-h. **Total
≈ 0.57 GPU-h** against the pre-registration's ≲ 1 GPU-h estimate, in ≈ 84 min of wall time
across both attempts (75.6 min before the abort, 8.5 min after).

**Time yielded to round 3.** 60 s measured — 12% of the resumed run's held-plus-yielded 491 s.
The 15:03:25Z–15:20:38Z window was a dead run, not a yield, and round 3's smoke gate used it
(15:05:24Z–15:20:15Z, `/root/scratch/issue-252/logs/events`). Phase A therefore cost round 3 no
GPU time; the box was idle for round 3's own run.

## Audio

8,615 clips (2.3 GB) were fetched to `/root/scratch/tlog-mining/clips/` by provenance, hardlinked
from round 2's look clip cache where present, and every clip's SHA-256 was checked before
decoding. **No audio is committed and no NAS copy was made** (#262's owner answers). The clips
were deleted from the box once this commit was pushed; the committed rows carry the pinned
dataset revision and per-clip provenance, so the audio re-downloads at that revision.

## Changelog

- 2026-11-02: phase A committed; the note records the 15:03:25Z OOM abort, the `P=1` resume
  under #272, the measured and estimated GPU time, and the 60 s yielded to round 3.

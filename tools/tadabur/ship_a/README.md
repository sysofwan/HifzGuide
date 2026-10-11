# Ship A's sites, split, word sample and worklist (#126)

Ship A (ADR-0012, goals 1-2) trains on sites the owner verifies by ear: where the shipped
`h448` (streamed at b = 0, no bias) swaps the letters of a target pair, or decodes an
in-word geminate as one consonant. The target pairs are nine (D9, `CONTEXT.md`): the six
Muraja soft pairs, `ذ↔ظ`, `س↔ش` and `ط↔ق`. This directory holds what #61's sittings ask and
how it was mined and sized. Everything here is a function of the committed mining pool
(`../mining_pool/`), Ship A's mining extension (`mining/`) and their frozen decodes; no
candidate model is read, and no app model: every count is decode level (the #84 scorer's
`training.site_outcomes`). The sizing and its
answers are in [`docs/ship-a-sizing.md`](../../../../docs/ship-a-sizing.md).

| file | what it is | written by |
|---|---|---|
| `frame.json` | the planning frame: per reference-defined stratum (pair direction, in-word / cross-word geminates, single consonants, each tashkeel mark) and per canonical reciter, the sums of `power_frame.FIELDS` (what `h448` and the base teacher commit against the mushaf, the base already right where `h448` flags, listening seconds) | `python -m tadabur.power_frame` |
| `sites.jsonl` | every Ship A site, one per line: a worklist row of `listening_session` (truth-site skeleton with provenance: `audio_filename`, `shard`, sample span, `audio_sha256`; stratum and population; inclusion probabilities; excerpt) plus `reciter_id`, `half` and what `h448` and the base committed at the carrier | `python -m tadabur.ship_a mine` |
| `sites.summary.json` | its population sidecar: per Ship A stratum, per family and for partner swaps against other letters, sites and reciters per half and how many the base already gets right; the frame's carrier populations; input checksums | the same |
| `split.json` | the reciter split, confirmed by the owner (D11): salt, rule, cutoff, and every reciter's half (pool and extension) | the same |
| `mining/` | the mining extension (D9): its frame, compact scan record, hit clips' manifest and both decode caches, and a summary with the yield per pair and the GPU time | `python -m tadabur.ship_a_mining scan / register / build / stream` |
| `word_sample.jsonl`, `word_sample.summary.json` | the random word sample: 160 words, one per held-out reciter drawn, with their inclusion probabilities; the frame per reciter | `python -m tadabur.word_sample draw` |
| `worklist.jsonl`, `worklist.summary.json` | #61's worklist (site rows) and its summary: parts, minutes, rows per half, the queue cut into three sittings, and the SHA-256 that freezes `worklist.jsonl` and `word_sample.jsonl`; the build also writes the split's two exposure uses | `python -m tadabur.ship_a_worklist build` |
| `waqf_guard.segmentations.jsonl`, `waqf_guard.segmentations.json` | the segmentations the guard's windows are read on (#134): per clip of the waqf fixtures' re-segmentation and of the P3.5 re-location that a guarded P3.5 row sits on, its canonical reciter, skip reason, re-reads, word times and the segmenter's cuts; each run's record and files' checksums | `python -m tadabur.waqf_guard record` (on the GPU box, after `tadabur.resegment`) |
| `waqf_guard.boundaries.jsonl`, `waqf_guard.vad.json` | the waqf guard's measurements (#134): every boundary the local pause check read, Ship A's and the reciter-disjoint waqf fixtures', with its window and VAD scores (the fixtures' with the canonical reciter, the owner's verdict and the segmenter's decisions), and the VAD pass's provenance | `python -m tadabur.waqf_guard windows`, then `python -m tadabur.local_pause score` (GPU) |
| `waqf_guard.jsonl`, `waqf_guard.summary.json` | the waqf guard: per site, worklist row and sampled word, whether its mark depends on wasl and, where it does, the local check's decision; counts per part and stratum at the verdicts, and the calibration | `python -m tadabur.waqf_guard build` |
| `onset_guard.synthetic.jsonl`, `onset_guard.synthetic.json` | the onset guard's synthetic trims (#138): one clean segment start per canonical reciter of the mining pool, with its provenance (clip, checksum, shard, row, reciter, split), segment and the first letter's onset sample; the selection's counts and rules | `python -m tadabur.onset_guard synthetic` (reads the staged audio) |
| `onset_guard.scores.jsonl`, `onset_guard.run.json` | every span the onset detector scored (each exposed segment and each synthetic cut) with the five candidates' scores and the head of the base teacher's free decode; the decode and VAD pass's provenance | `python -m tadabur.onset_guard spans`, then `score` (GPU) |
| `onset_guard.jsonl`, `onset_guard.summary.json`, `onset_guard.curves.json` | the onset guard: per site, worklist row and sampled word, whether its mark is read in its segment's first word (exposed), the cut it starts at, its scores and the detector's decision; counts per part and stratum at the verdicts, the operating points, the owner check and the re-listen list; every candidate's calibration curve | `python -m tadabur.onset_guard build` |
| `sizing.json` | the four sizing questions' results, after the shared exclusion (both guards) | `python -m training.ship_a_sizing run` |

Rebuild in that order, from `tools/` (torch-free but the guard's VAD pass; `quran-transcript`
for the Uthmani words). Each step checks the inputs it reads against the checksums the previous
one recorded.

## The waqf guard (#134)

A geminate made by joining two words, a ta marbuta read `ت` or a word-final haraka exists only
in wasl, and the segmenter misses about one real pause in five. `tadabur.waqf_guard` decides,
through the phonetizer, which sites, worklist rows and sampled words change their prescribed
mark under a waqf at a boundary of their word; where one does, `tadabur.local_pause` reads the
VAD there, at the operating point its calibration on the waqf fixtures picks. A site judged
paused is `waqf_realized`: out of every evaluation denominator (the sizing, and
`truth_scorer.score(excluded_items=tadabur.exclusions.current())`), its verdict kept as a
training target. **Owner notes decide first**: type `waqf` in the note box when the reciter
stopped (or `wasl` when the check is wrong about a pause); notes are read from
`listening_session/verdicts.jsonl` at every use. After a sitting, refresh the counts with
`python -m tadabur.waqf_guard build`, then re-run the sizing and its page. The frozen worklist
and word sample are untouched.

## The onset guard (#138)

Some Tadabur files start inside their first letter (the `ء` of `ءَلَّذِينَ`, the `ث` of `ثُمَّ`):
a mark read there tests the clip, not the model, and the app never hears audio cut that way.
`tadabur.onset_guard` marks every item whose mark is read in the first word of its segment as
exposed and scores the start of that segment with five candidate detectors, calibrated on
synthetic trims of clean segment starts. A detector decides only where it is reliable against
the owner's flags too; at the committed verdicts none is (`detector_decides` in the summary),
so **owner notes decide**: type `trim`, `cut` or `split`
in the note box when the opening sound is cut off (`no trim` when it is not). The item becomes
`onset_trimmed`: out of every evaluation denominator through the shared exclusion
(`tadabur.exclusions`, the one the waqf guard uses) and no training target. After a sitting,
refresh with `python -m tadabur.onset_guard build`, then re-run the sizing and its page.

## The sites

Every population here is `h448`-unseen: the pool's clips, the mined clips and the sampled
words hold no recording tied, even as an alias or a probable copy, to what `h448` trained on,
`decode_evalset` or the sealed panel (`exposure.unseen_violations`; a test asserts it for
sites, worklist, word sample and mined clips). The P3.5 re-listens are §8's targeted
safeguards and exempt.

- **pair**: every target-pair carrier where `h448` commits a consonant other than the mushaf
  letter, once per carrier. A carrier in two pairs (`ذ`, `ظ`, `س`, `ط`, `ق`) is asked as the
  pair whose partner `h448` committed, else as the first pair in label order. Strata
  `ship_a:<prescribed>→<partner>:h448_partner:base_right|base_wrong` for a **partner swap**
  (by the base teacher's outcome), `ship_a:kasb:س→ش:h448_partner` for a `س`→`ش` swap on the
  `س` of `كسب`, and `ship_a:<prescribed>:h448_other` for a letter outside every target pair
  (never asked, D9; `ت`→`ه` at a paused ta marbuta is a reference problem, #130).
- **mining extension**: the partner swaps on the clips `tadabur.ship_a_mining` scanned
  (`ship_a_mining:…` strata). The scan stopped on GPU time, so inference from it is
  restricted to the scanned population (its hits are a census, inclusion probability 1).
- **geminate**: every in-word reference geminate the pool's `h448` decodes as one consonant
  (`ship_a:shaddah:held:h448_single:base_held` / `…:base_not_held`). Cross-word geminates are
  out: the app never scores them.

## What is listened to (owner decisions D10 and 2026-10-10)

`tadabur.ship_a.select_listening` sets each site's `listen_probability` (0 when it is never
asked); a worklist row carries it as its draw probability.

- a partner swap the base teacher gets **wrong**: always;
- a partner swap the teacher gets **right**: one per **held-out** reciter and family (first by
  salted rank, chance `1 / k`), for evaluation. The others are not listened to: their target
  is the teacher's decode, as D10 does for the geminates the teacher holds, and they are no
  verified site in any evaluation;
- the `كسب` swaps: a sample of 20 spread over reciters (round robin in salted order). A
  site's chance is its reciter's expected picks over every salted order, divided by the
  reciter's sites (`round_robin_shares`; here `(20 / 80) / n`). The rest take the sample's
  verdict: the letter a majority of the sampled sites with a verdict are heard as, or no
  label without a majority;
- a geminate the teacher does not hold: always; one it holds: never (D10).

**What a score describes (owner decision, 2026-10-10).** The estimand is the frozen verified
site set (the pool's and extension's sites as mined here, and the frozen word sample), not
all of Tadabur: no weight expands a site to a wider frame. A listened site weighs
`1 / listen_probability` (`ship_a.listening_weights`; 1 where a stratum is listened to whole),
a sampled word `1 / draw_probability` (its selection within the sample's frame). The #84
scorer takes them as `truth_scorer.score(design_weights=…)`; acceptance rules §1 is amended.

## The reciter split (confirmed, D11)

`train` when the first 8 bytes (big-endian) of SHA-256 over `"ship-a-verified-split-2026-10:<reciter id>"`
are below 2**63, else `held_out` (acceptance rules §9; `training.acceptance_stats.reciter_hash`).
The pool's 394 reciters fall 190 / 204. The exposure registry records both halves:
`ship_a.verified_train` (the recordings of every train-half Ship A site, listened or the
teacher's target) and `ship_a.verified_held_out` (every held-out site's and sampled word's
recording), plus `ship_a.mining` (every recording the extension's scan decoded, hit or not). Keeping the held-out reciters
out of the probe's full-shard training stream is #94's decision.

## The session for #61

The queue holds, in this order: the sites and the 30 synthetic-edit check items (#107) in
one seeded order of their ids, then the 160 sampled words. A sampled word is shown whole
(judging it needs its text) and may be a word a site asks about in another recitation, so
the words come after every site. `worklist.summary.json` gives the minutes per part and the
sittings as runs of the queue of at most 75 minutes (the UI resumes at the first unanswered
row). The clips come from two directories on cuda-dev: the pool's
(`/root/scratch/issue-83/stage/clips`) and the mining extension's
(`/root/scratch/issue-126/stage/clips`); `fetch` takes each clip from the right one.

From the Mac checkout whose `listening_session/verdicts.jsonl` should receive the answers:

```bash
cd ~/repos/HifzGuide/tools
# Once: copy the clips the worklist and the word sample play (669 clips, ~390 MB) and the
# 30 edit-check items from cuda-dev, verifying every checksum. Files already there are kept.
/Users/sysofwan/repos/HifzGuide/.venv-test/bin/python -m tadabur.listening_session fetch \
  --audio-dir ~/hifzguide-listening/clips \
  --worklist tadabur/ship_a/worklist.jsonl --words tadabur/ship_a/word_sample.jsonl
# Launch, reachable from a phone on the LAN.
/Users/sysofwan/repos/HifzGuide/.venv-test/bin/python -m tadabur.tashkeel_audit_ui \
  --audio-dir ~/hifzguide-listening/clips --host 0.0.0.0 --port 8000 \
  --worklist tadabur/ship_a/worklist.jsonl --words tadabur/ship_a/word_sample.jsonl
```

Everything else (checking it answers on the LAN, the keys, committing `verdicts.jsonl`) is as
in [`../listening_session/README.md`](../listening_session/README.md). A word asks **"Was the
highlighted word recited correctly?"**: Yes (`correct`), No (`mistake`) or Unclear
(`unclear`, which leaves every denominator).

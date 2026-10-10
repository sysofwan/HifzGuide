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
| `sizing.json` | the four sizing questions' results | `python -m training.ship_a_sizing run` |

Rebuild in that order, from `tools/` (torch-free; `quran-transcript` for the Uthmani words).
Each step checks the inputs it reads against the checksums the previous one recorded.

## The sites

- **pair**: every target-pair carrier where `h448` commits a consonant other than the mushaf
  letter, once per carrier. A carrier in two pairs (`ذ`, `ظ`, `س`, `ط`, `ق`) is asked as the
  pair whose partner `h448` committed, else as the first pair in label order. Stratum
  `ship_a:<prescribed>→<partner>:h448_partner` for a **partner swap** or
  `ship_a:<prescribed>:h448_other` for a letter outside every target pair. **Only partner
  swaps are listened to** (D9). The other letters stay mined in `sites.jsonl` but off the
  worklist; `ت`→`ه` at a paused ta marbuta (70 sites) is a reference problem, filed as #130.
- **mining extension**: the partner swaps found on the clips `tadabur.ship_a_mining` scanned
  (`ship_a_mining:<prescribed>→<partner>:h448_partner`, inclusion probability the scanned
  share of the frame's shards).
- **geminate**: every in-word reference geminate the pool's `h448` decodes as one consonant
  (`ship_a:shaddah:held:h448_single:base_held` / `…:base_not_held`). Cross-word geminates are
  out: the app never scores them.
- **Base already right** (ADR-0011 §3): where the base teacher commits the mushaf value, the
  target is its decode. The 718 geminates it holds are not listened to (D10); the 115 it does
  not hold are. Partner swaps are listened to whatever the teacher did.

Each stratum is a census of its manifest (inclusion probability = the clip's), so training
labels may come from it; evaluation reads it on held-out reciters only.

## The reciter split (confirmed, D11)

`train` when the first 8 bytes (big-endian) of SHA-256 over `"ship-a-verified-split-2026-10:<reciter id>"`
are below 2**63, else `held_out` (acceptance rules §9; `training.acceptance_stats.reciter_hash`).
The pool's 394 reciters fall 190 / 204. The exposure registry records both halves:
`ship_a.verified_train` (the recordings of every train-half Ship A site, listened or the
teacher's target) and `ship_a.verified_held_out` (every held-out site's and sampled word's
recording), plus `ship_a.mining` (the extension's staged clips). Keeping the held-out reciters
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
# Once: copy the clips the worklist and the word sample play (1,376 clips, ~850 MB) and the
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

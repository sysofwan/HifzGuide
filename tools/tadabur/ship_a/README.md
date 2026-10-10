# Ship A's sites, split, word sample and worklist (#126)

Ship A (ADR-0012, goals 1-2) trains on sites the owner verifies by ear: where the shipped
`h448` (streamed at b = 0, no bias) commits a wrong letter at a target-pair carrier, or
decodes an in-word geminate as one consonant. This directory holds what #61's sittings ask
and how it was mined and sized. Everything here is a function of the committed mining pool
(`../mining_pool/`) and its two frozen decodes; no candidate model is read, and no app model:
every count is decode level (the #84 scorer's `training.site_outcomes`). The sizing and its
answers are in [`docs/ship-a-sizing.md`](../../../../docs/ship-a-sizing.md).

| file | what it is | written by |
|---|---|---|
| `frame.json` | the planning frame: per reference-defined stratum (pair direction, in-word / cross-word geminates, single consonants, each tashkeel mark) and per canonical reciter, the sums of `power_frame.FIELDS` (what `h448` and the base teacher commit against the mushaf, the base already right where `h448` flags, listening seconds) | `python -m tadabur.power_frame` |
| `sites.jsonl` | every Ship A site, one per line: a worklist row of `listening_session` (truth-site skeleton with provenance: `audio_filename`, `shard`, sample span, `audio_sha256`; stratum and population; inclusion probabilities; excerpt) plus `reciter_id`, `half` and what `h448` and the base committed at the carrier | `python -m tadabur.ship_a mine` |
| `sites.summary.json` | its population sidecar: per Ship A stratum, per family and for partner swaps against other letters, sites and reciters per half and how many the base already gets right; the frame's carrier populations; input checksums | the same |
| `split.json` | the reciter split **proposal**: salt, rule, cutoff, and every pool reciter's half at 50/50 and at the alternative (two thirds held out) | the same |
| `word_sample.jsonl`, `word_sample.summary.json` | the random word sample: 160 words, one per held-out reciter drawn, with their inclusion probabilities; the frame per reciter | `python -m tadabur.word_sample draw` |
| `worklist.jsonl`, `worklist.summary.json` | #61's worklist (site rows) and its summary: parts, minutes, rows per half, the queue cut into three sittings, and the SHA-256 that freezes `worklist.jsonl` and `word_sample.jsonl` | `python -m tadabur.ship_a_worklist build` |
| `sizing.json` | the four sizing questions' results | `python -m training.ship_a_sizing run` |

Rebuild in that order, from `tools/` (torch-free; `quran-transcript` for the Uthmani words).
Each step checks the inputs it reads against the checksums the previous one recorded.

## The sites

- **pair**: every target-pair carrier where `h448` commits a consonant other than the mushaf
  letter, once per carrier. A `ذ` or `ظ` carrier sits in two pairs; it is asked as the pair
  whose partner `h448` committed, else as the first pair in label order. Stratum
  `ship_a:<prescribed>→<partner>:h448_partner` (the swap the soft-pair allowance forgives) or
  `ship_a:<prescribed>:h448_other` (any other letter).
- **geminate**: every in-word reference geminate `h448` decodes as one consonant
  (`ship_a:shaddah:held:h448_single:base_held` / `…:base_not_held`). Cross-word geminates are
  out: the app never scores them.
- **Base already right** (ADR-0011 §3): where the base teacher commits the mushaf value, the
  target is its decode. Those geminates leave the worklist; pair sites stay on it.

Each stratum is a census of the pool (inclusion probability = the clip's in the pool), so
training labels may come from it; evaluation reads it on held-out reciters only.

## The reciter split (a proposal)

`train` when the first 8 bytes (big-endian) of SHA-256 over `"ship-a-verified-split-2026-10:<reciter id>"`
are below `round((1 − held-out share) · 2**64)`, else `held_out` (acceptance rules §9;
`training.acceptance_stats.reciter_hash`). At 50/50 the pool's 394 reciters fall 190 / 204. The
owner confirms the split before the first sitting. No exposure-registry use is written for
it until then (`probe.training` and its held-out counterpart belong to #94).

## The session for #61

The queue holds, in this order: the sites and the 30 synthetic-edit check items (#107) in
one seeded order of their ids, then the 160 sampled words. A sampled word is shown whole
(judging it needs its text) and may be a word a site asks about in another recitation, so
the words come after every site. `worklist.summary.json` gives the minutes per part and the
three sittings as runs of the queue (the UI resumes at the first unanswered row).

From the Mac checkout whose `listening_session/verdicts.jsonl` should receive the answers:

```bash
cd ~/repos/HifzGuide/tools
# Once: copy the clips the worklist and the word sample play (657 clips, ~390 MB) and the
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

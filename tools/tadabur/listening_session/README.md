# The listening session (#87 for #61)

The sites the owner adjudicates by ear in one sitting, and what was heard. Every row is a
**truth-site skeleton** ([`../truth_sites/README.md`](../truth_sites/README.md)): provenance
from the staged-clip registry, the realized reference and carrier, the mark under test, what
the mushaf prescribes, `heard: pending`, and the stratum with its population. The verdicts
fill `heard`.

Code: [`../listening_session.py`](../listening_session.py) (mining, worklist, verdicts) and
[`../tashkeel_audit_ui.py`](../tashkeel_audit_ui.py) (the blind UI).

| file | what it is |
|---|---|
| `worklist.jsonl` | one site per line, in the shuffled order the UI asks them: the truth-site fields plus the sampling design and the excerpt |
| `worklist.summary.json` | per stratum: population and its reciters, sites drawn and their reciters, draw and inclusion probabilities, estimated listening minutes; the sizes used, input checksums, the base decode's fingerprint |
| `verdicts.jsonl` | the answers, `{site_id, heard, note}` per line, sorted by site id, written by the UI as they are given |

## Launching the session

The UI runs on the Mac, from the repo checkout, so every answer is written straight into
this directory's tracked `verdicts.jsonl`. It plays a local copy of the session's clips,
copied once from the GPU box and checked against the staged-clip registry (checksum and
length) every time the server starts.

```bash
cd ~/repos/HifzGuide/tools
# Once: copy the clips the worklist plays (320 WAVs, ~180 MB) from cuda-dev.
/Users/sysofwan/repos/HifzGuide/.venv-test/bin/python -m tadabur.listening_session clips \
  | rsync -a --files-from=- root@cuda-dev:/root/scratch/issue-83/stage/clips/ ~/hifzguide-listening/clips/
# Launch, reachable from a phone on the LAN (the launch-audit-ui pattern: bind 0.0.0.0).
/Users/sysofwan/repos/HifzGuide/.venv-test/bin/python -m tadabur.tashkeel_audit_ui \
  --audio-dir ~/hifzguide-listening/clips --host 0.0.0.0 --port 8000
```

| setting | value |
|---|---|
| working directory | `tools/` of the checkout whose `verdicts.jsonl` should receive the answers |
| audio | `~/hifzguide-listening/clips/` (any directory holding the copied clips; `--audio-dir`) |
| worklist | `tools/tadabur/listening_session/worklist.jsonl` (the default; `--worklist`) |
| verdicts | `tools/tadabur/listening_session/verdicts.jsonl` (the default; `--verdicts`), rewritten atomically after every answer |
| host / port | `0.0.0.0` / `8000`; open `http://<the Mac's LAN IP>:8000/` (`ipconfig getifaddr en0`) on the phone |

Check it answers on the LAN before handing the phone over:
`curl -s -o /dev/null -w "%{http_code}\n" http://$(ipconfig getifaddr en0):8000/` must print
`200`. If the phone is refused while that prints 200, allow incoming connections for python
in the macOS firewall. Ctrl-C stops the server; answers are already on disk. Afterwards:
`git add tools/tadabur/listening_session/verdicts.jsonl` and commit.

On the page: **Play** (space), **Slower** at 0.6× (s), **Whole** segment (w), then tap the
answer (1-5). An answer counts once it is saved; the next site then appears and plays.
Answers are refused for 0.4 s after a site appears and while a save is in flight, so a double
tap or a held key cannot answer a site nobody heard. A failed save shows an error and leaves
the site unanswered.

## How the sites were mined

From the mining pool (#83, [`../mining_pool/README.md`](../mining_pool/README.md)).
Eligibility depends only on the realized reference (acceptance rules §1). Two frozen decodes
assign each eligible site to a stratum, and no candidate is involved:

- the **base teacher's** decode of each kept segment (`mining_pool/base_decodes.json`);
- for tashkeel also the **shipped `h448`** streamed at **b=0 with no bias**, the comparator of
  the ship criterion (§3): each pool clip decoded whole through the deployed streaming
  protocol (`training.decoding.Decoder.decode_stream(block=0)`, last window flushed, fp32
  weights), and each segment's text cut from it by commit time (`tadabur.pool_stream`,
  `mining_pool/h448_stream_decodes.json`, with the decode fingerprint and the checkpoint's
  SHA-256).

| question (UI mode) | eligible site | strata |
|---|---|---|
| tashkeel | a **mid-word haraka** on a consonant | per haraka, `base_<o>:h448_<o>` for each decode's outcome `o`: `empty` (the carrier aligned, no haraka after it), `matched` (that haraka), `other` (another haraka, or the carrier misheard or unaligned) |
| tashkeel | a **mid-word prescribed sukun**: a single consonant followed in its word by another consonant or the qalqala mark | `base_<o>:h448_<o>`, `o` in `empty`, `haraka`, `other` |
| shaddah | every reference geminate (first of the doubled consonant) inside one word, `prescribed: held` | `held:base_single`, `held:base_rest` |
| shaddah | every geminate whose halves fall in two words (idgham across a word boundary), `prescribed: held` | `held:cross_word` (default size 0) |
| shaddah | every single consonant, `prescribed: not_held` (an added shaddah's site) | `not_held:base_double` (the base doubled it), `not_held:base_rest` |
| consonant | every carrier of a target-pair letter (six soft pairs and `ذ↔ظ`), per direction `prescribed→partner` | `base_partner` (the base heard the partner: the reject-pile sites), `base_rest` |

*Mid-word*: a letter of the same word follows the mark, by the segment's `raw_word_offsets`.
A word whose Uthmani text has tanween ends at its last haraka, because the `ن` / `ں` /
assimilated letter after it realizes the tanween, so case endings are never mid-word.
Word-final marks depend on waqf and wasl and stay out (acceptance rules §1, *Weak labels*).
A haraka and a sukun share one question, so the page cannot tell them apart.

*Cross-word geminates.* The #86 shaddah probe found that 182 of the base's 419 collapsed
geminate runs cross a word boundary (166 are و after a tanween), and that at 18 of 20 it
sampled the reciter had paused there: segmentation missed the pause, and the realized
reference wrongly assumes continuation. A "not held" verdict at such a site is a correct
pause, not a mistake, so cross-word geminates are kept out of the shaddah strata
(`held:cross_word`, counted, drawn 0). No tashkeel site depends on such an assimilation: a
haraka on the second half of a cross-word geminate is the next word's own initial haraka,
said whether or not the reciter paused (948 such sites stay in the tashkeel strata), and a
sukun site is never part of a geminate.

Plus the **23 nominal P3.5 rejects** (`../truth_sites/p35_fixtures.jsonl`, `heard: pending`:
10 soft-pair, 13 shaddah), re-adjudicated at site level (acceptance rules §8). They keep
their own site ids, strata and populations; inclusion probability 1.

**The draw.** Within a stratum, sites are ranked by
`sha256("issue-87-listening-session-v1:<stratum>:<site_id>")` and the top `n` taken, so a
re-mine over a changed population re-draws mostly the same sites. `n` per stratum is
`--sizes` (JSON `{stratum: n}`); the post-#84 power simulation (#105) sets it. The default
(`DEFAULT_SIZES`) fits the ~1.5 h plan of #61 and is weighted toward the slots `h448`
leaves empty, above all where the base heard the haraka (`HARAKA_SIZES`, `SUKUN_SIZES`),
with a small positive draw in every other tashkeel cell; 60 within-word geminates decoded
single; 5 per
consonant direction the base heard as its partner. Shaddah and consonant `base_rest` strata
and `held:cross_word` are 0 and only counted.

**Inclusion probability** of a site = its clip's inclusion probability in the pool (given the
394 drawn reciters, `mining_pool/clips.jsonl`) × the within-stratum draw probability
`n / N`. The design weight is its inverse. Rows carry both factors and the product.

**Site ids** are model-free and do not name the prescribed mark:
`new_audit:<clip>#<segment_index>@<reference_index>:<question>`, where the question is
`tashkeel` (haraka and sukun alike), `shaddah` or the pair.

**Excerpt.** The UI plays the carrier's word and one word either side, padded 0.3 s on each
side and clamped only to the clip (never to the segment), so it never cuts inside those
words. The **Whole** button plays the segment. The word times are the clip's whole-clip
alignment (`ClipStatus.word_times`, from the base teacher's decode of the clip), because the
pool has no independent timings. That alignment places *words*: the excerpt is a function of
the reference and the word times only, never of the mark a decode emitted at the carrier,
so it is the same whatever the reciter turns out to have said there (pinned by a test). The
whole segment plays instead when the times cannot place the words: a clip with re-reads, no
word times, a word span outside the segment, or an excerpt under 1 s.

**Blinding.** The page receives the whole queue, so every site's answer is hidden wherever
its word appears (the same ayah, word index and realized word), in every site's text, not
only on its own screen; with each answer go the cues that would give it away (madd or
qalqala after a tashkeel carrier, the doubling and the haraka after a shaddah carrier, the
qalqala after a hidden letter). See `tashkeel_audit_ui.py`.

**Departure from ADR-0007's wording.** ADR-0007 stratified the static audit by the *base
outcome alone*. Here the tashkeel strata are also crossed with `h448`'s b=0 outcome, because
the ship criterion's comparator is `h448` at b=0 (acceptance rules §3) and base-only strata
put the shipped model's own empty slots in a stratum drawn at a weight of ~2,000. §1 allows
it: it forbids model-dependent *eligibility*, not baseline-dependent stratification, and
both decodes are frozen and recorded.

**Choices #84 may define differently** (it fixes the streaming carrier semantics, §9). This
work uses `training.tashkeel_eval.vowel_sites`' anchoring for haraka outcomes and
`carrier_marks` (an aligned carrier and the character right after it) for sukun, and:
a haraka on the second half of a geminate the decode collapsed is `other` (its carrier did
not align), not `empty`; the stream's text for a segment is every token committed within
0.5 s of the segment's span, matched by local alignment; and a token is timed by its run's
centre. A different rule in #84 changes stratum *membership*, never eligibility, so a
re-mine under it re-draws the same sites where the strata agree.

Regenerate, from `tools/` (torch-free; `quran-transcript` for the Uthmani words; the P3.5
re-location's `clip_status.jsonl` and `segmentation.jsonl` give those clips' word times):

```bash
python -m tadabur.listening_session mine --p35-seg-dir stage/seg_p35 [--sizes sizes.json]
```

## The default worklist

385 sites on 344 clips from 171 reciters (~198 MB of clips); **~79 min** of listening by the
summary's model (each excerpt played twice plus 4 s to answer); 41 rows play their whole
segment. Populations are in the whole pool (2,508 clips, 3,785 kept segments).

Tashkeel, population and draw per cell, the three `h448` outcomes in order (`empty` /
`matched` / `other`; for sukun `empty` / `haraka` / `other`):

| haraka, base outcome | population | drawn | minutes |
|---|---|---|---|
| fatha `base_empty` | 80 / 183 / 47 | 15 / 5 / 3 | 4.1 |
| fatha `base_matched` | 1,361 / 29,756 / 1,781 | 25 / 10 / 4 | 8.9 |
| fatha `base_other` | 32 / 386 / 850 | 5 / 3 / 3 | 1.7 |
| damma `base_empty` | 20 / 51 / 7 | 15 / 5 / 3 | 4.8 |
| damma `base_matched` | 490 / 7,411 / 695 | 25 / 10 / 4 | 8.9 |
| damma `base_other` | 13 / 102 / 190 | 5 / 3 / 3 | 2.2 |
| kasra `base_empty` | 4 / 9 / 4 | 4 / 5 / 3 | 2.4 |
| kasra `base_matched` | 263 / 10,287 / 853 | 25 / 10 / 4 | 8.9 |
| kasra `base_other` | 5 / 93 / 219 | 5 / 3 / 3 | 2.8 |
| sukun `base_empty` | 9,269 / 319 / 653 | 15 / 15 / 5 | 7.4 |
| sukun `base_haraka` | 16 / 50 / 4 | 4 / 8 / 2 | 2.7 |
| sukun `base_other` | 86 / 3 / 113 | 3 / 2 / 3 | 1.3 |

| other strata | population | drawn | minutes |
|---|---|---|---|
| shaddah `held:base_single` / `held:base_rest` (within a word) | 89 / 8,391 | 60 / 0 | 11.3 |
| shaddah `held:cross_word` | 950 | 0 | |
| shaddah `not_held:base_double` / `not_held:base_rest` | 35 / 86,363 | 0 / 0 | |
| consonant `base_partner`, per direction | `ز→ذ` 26, `ظ→ض` 27, `ك→ق` 20, `ق→ك` 8, `س→ص` 7, `ص→س` 4, `ط→ت` 3, `ذ→ظ` 2, `ه→ح` 2, `ض→ظ` 1; `ذ→ز`, `ت→ط`, `ح→ه`, `ظ→ذ` 0 | 37 | 7.2 |
| P3.5 pending (6 strata) | 151 sites in those strata | 23 | 4.3 |

By question: tashkeel 265 sites (55.9 min), shaddah 73 (13.7), consonant 47 (9.2).

`h448` leaves a mid-word slot empty far more often than the base teacher: 1,473 fatha, 523
damma and 272 kasra against the base's 310, 78 and 17, and 1,361 / 490 / 263 of them where
the base heard the haraka. The base leaves a mid-word **kasra** empty only 17 times in the
whole pool; most of its omissions are case endings (word-final), out of scope. Four
consonant directions have no site where the base heard the partner: `ذ→ز`, `ت→ط`, `ح→ه`,
`ظ→ذ`.

## The verdicts

`heard` takes the truth-site vocabulary of the mark: `fatha` `damma` `kasra` `sukun` for
tashkeel, `held` `not_held` for shaddah, either letter of the pair, or `unclear` (which
leaves every denominator). The file is rewritten atomically on every answer, so it is
always complete and resumable; a verdict whose site a re-mine dropped is kept.

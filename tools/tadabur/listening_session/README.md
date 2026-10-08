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

## How the sites were mined

From the mining pool (#83, [`../mining_pool/README.md`](../mining_pool/README.md)) and the
**frozen base teacher's** decode of its kept segments (`base_decodes.json`). No candidate
and no `h448` decode. Eligibility depends only on the realized reference; the base decode
assigns each eligible site to a stratum.

| question (UI mode) | eligible site | strata (by the base decode) |
|---|---|---|
| tashkeel | a **mid-word haraka** on a consonant | per haraka: `base_empty` (no haraka after the matched carrier), `base_matched` (that haraka), `base_other` (another haraka, or the carrier misheard or unaligned) |
| tashkeel | a **mid-word prescribed sukun**: a single consonant followed in its word by another consonant or the qalqala mark | `base_empty`, `base_haraka`, `base_other` |
| shaddah | every reference geminate (first of the doubled consonant), `prescribed: held` | `held:base_single`, `held:base_rest` |
| shaddah | every single consonant, `prescribed: not_held` (an added shaddah's site) | `not_held:base_double` (the base doubled it), `not_held:base_rest` |
| consonant | every carrier of a target-pair letter (six soft pairs and `ذ↔ظ`), per direction `prescribed→partner` | `base_partner` (the base heard the partner: the reject-pile sites), `base_rest` |

*Mid-word*: a letter of the same word follows the mark, by the segment's `raw_word_offsets`.
A word whose Uthmani text has tanween ends at its last haraka, because the `ن` / `ں` /
assimilated letter after it realizes the tanween, so case endings are never mid-word.
Word-final marks depend on waqf and wasl and stay out (acceptance rules §1, *Weak labels*).
A haraka and a sukun share one question, so the page cannot tell them apart.

Plus the **23 nominal P3.5 rejects** (`../truth_sites/p35_fixtures.jsonl`, `heard: pending`:
10 soft-pair, 13 shaddah), re-adjudicated at site level (acceptance rules §8). They keep
their own site ids, strata and populations; inclusion probability 1.

**The draw.** Within a stratum, sites are ranked by
`sha256("issue-87-listening-session-v1:<stratum>:<site_id>")` and the top `n` taken, so a
re-mine over a changed population re-draws mostly the same sites. `n` per stratum is
`--sizes` (JSON `{stratum: n}`); the post-#84 power simulation (#105) sets it. The default
(`DEFAULT_SIZES`) realizes the ~1.5 h plan of #61: 50 per haraka left empty, 17 matched
controls per haraka, 40 + 10 prescribed sukun, 60 geminates decoded single, 5 per consonant
direction the base heard as its partner; every other stratum is 0 and only counted.

**Inclusion probability** of a site = its clip's inclusion probability in the pool (given the
394 drawn reciters, `mining_pool/clips.jsonl`) × the within-stratum draw probability
`n / N`. The design weight is its inverse. Rows carry both factors and the product.

**Site ids** are model-free and do not name the prescribed mark:
`new_audit:<clip>#<segment_index>@<reference_index>:<question>`, where the question is
`tashkeel` (haraka and sukun alike), `shaddah` or the pair.

**Excerpt.** The UI plays the carrier's word and one word either side (the clip's word times,
padded 0.25 s, clamped to the segment), or the whole segment when the clip has re-reads, no
word times, or the excerpt would be under 1 s (33 of the default rows).

Regenerate, from `tools/` (torch-free; `quran-transcript` for the Uthmani words; the P3.5
re-location's `clip_status.jsonl` and `segmentation.jsonl` give those clips' word times):

```bash
python -m tadabur.listening_session mine --p35-seg-dir stage/seg_p35 [--sizes sizes.json]
```

## The default worklist

339 sites on 320 clips from 180 reciters; ~65 min of listening by the summary's model
(each excerpt played twice plus 4 s to answer). Populations are in the whole pool (2,508
clips, 3,785 kept segments).

| stratum | population | drawn | minutes |
|---|---|---|---|
| fatha `base_empty` / `base_matched` / `base_other` | 328 / 32,899 / 1,249 | 50 / 17 / 0 | 8.0 / 3.2 |
| damma | 81 / 8,600 / 298 | 50 / 17 / 0 | 9.6 / 3.4 |
| kasra | **18** / 11,409 / 310 | **18** / 17 / 0 | 3.5 / 3.7 |
| sukun `base_empty` / `base_haraka` / `base_other` | 10,241 / 70 / 202 | 40 / 10 / 0 | 7.9 / 1.8 |
| shaddah `held:base_single` / `held:base_rest` | 212 / 9,218 | 60 / 0 | 12.5 |
| shaddah `not_held:base_double` / `not_held:base_rest` | 35 / 86,363 | 0 / 0 | |
| consonant `base_partner`, per direction | `ز→ذ` 26, `ظ→ض` 27, `ك→ق` 20, `ق→ك` 8, `س→ص` 7, `ص→س` 4, `ط→ت` 3, `ذ→ظ` 2, `ه→ح` 2, `ض→ظ` 1; `ذ→ز`, `ت→ط`, `ح→ه`, `ظ→ذ` 0 | 37 | 7.0 |
| P3.5 pending (6 strata) | 151 sites in those strata | 23 | 4.1 |

The base teacher leaves a mid-word **kasra** empty only 18 times in the whole pool, so the
plan's 50 cannot be met; most of its omissions are case endings (word-final), which are out
of scope. Four consonant directions have no site where the base heard the partner, so they
have no reject-pile sites: `ذ→ز`, `ت→ط`, `ح→ه`, `ظ→ذ`.

## The verdicts

`heard` takes the truth-site vocabulary of the mark: `fatha` `damma` `kasra` `sukun` for
tashkeel, `held` `not_held` for shaddah, either letter of the pair, or `unclear` (which
leaves every denominator). The file is rewritten atomically on every answer, so it is
always complete and resumable; a verdict whose site a re-mine dropped is kept.

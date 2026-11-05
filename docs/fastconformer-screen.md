# FastConformer screen: a fourth mining detector on Ship A's verified sites

## Summary

**Verdict: PRIOR** (the frozen rule on the bars below; the owner's
acknowledgement of the erratum is required before the verdict is read).

NVIDIA's Arabic FastConformer was run on CPU as a detector only: its committed text was
read at each Ship A site and at each synthetic edit, with one letter-level rule for every
column. It is never a label, reference or training signal.

- Ship A mistakes flagged: FastConformer 49.3%, teacher 83.6% (the teacher's bar moved 5 points).
- Ship A correct sites flagged: FastConformer 13.5%, teacher 63.6% (the stop bar is the teacher + 5).
- Edits heard: FastConformer 26.0%, teacher 22.7%; decoys unchanged: FastConformer 78.0%, teacher 58.7%.
- Unique catches (a decisive mistake it carries and none of the three models does): 0.

A GO means only **worth trying** in #261's scale-up draw. It does not rank this model
above the three detectors, and it does not adopt it.

## What was run

- Model `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` at revision
  `7f32349d952f42a28dce979ba73270aa2bbdfa89`; the `.nemo` SHA-256 was verified locally
  against the full LFS object id (`d29d19d7...c3e31`).
- Decoder: greedy CTC head, whole item, 16 kHz mono, on CPU. NeMo ran in its own
  environment behind a 10 GB disk reserve; the environment and the weights were deleted
  after the decodes were committed. No GPU time and no round-3 input.
- Card sanity line: CTC WER 1.50 on 200 benchmark EveryAyah clips, the card's
  normalisation (`card_norm`); the card's CTC WER is 7.61 and the run stops only above twice it.
  Measured on the benchmark's studio reciters, not NVIDIA's own test split.
- Sets: Ship A's verified sites minus the 8 G6 held-out sites (located, 67 mistakes and 407 correct sites), the 300 synthetic edit/decoy items and qdat's 1,505 clips.

## Bars

| column | mistakes flagged | correct flagged | edits heard | decoys unchanged |
|---|---:|---:|---:|---:|
| teacher | 83.6% | 63.6% | 22.7% | 58.7% |
| FastConformer | 49.3% | 13.5% | 26.0% | 78.0% |

FastConformer's intervals (reciter-clustered bootstrap):

- mistakes flagged: 49.3% [35.5, 61.4]
- correct flagged: 13.5% [10.1, 17.2]
- edits heard: 26.0% [19.3, 33.3]
- decoys unchanged: 78.0% [71.3, 84.7]

Outcome breakdown (FastConformer):

| side | heard | prescribed | other | none |
|---|---:|---:|---:|---:|
| mistakes | 29 | 34 | 2 | 2 |
| correct | 352 | 0 | 51 | 4 |

Measured text prior: at 34 of 67 decisive mistake sites it carries the prescribed letter (the teacher does so at 11).

## Unique catches

None: no decisive mistake site is carried by FastConformer alone.

## Sensitivity (committed output only; never gates)

A site is flagged here only when an explicit different output is committed, so a site
with nothing aligned is not flagged:

| column | mistakes (sensitivity) | correct (sensitivity) |
|---|---:|---:|
| fastconformer | 31/67 | 51/407 |
| base | 55/67 | 198/407 |
| c95 | 55/67 | 199/407 |
| r2_candidate_ema | 54/67 | 183/407 |

## Site mapping

516 of 522 Ship A sites and 300 of 300 edits were located from text alone, before any decode. The gate is 95% of every site and 99.4% of the sites that have a side; the mapping never
reads a decode; an unlocated site is excluded, never counted as a miss.
The 6 sites that did not locate are listed in
`report.json`. 45 sites have no human verdict (42 of them located) and are neither scored nor
in either denominator, so the mapping count and the scored count differ by exactly
those two groups.

## qdat

2371 of 12040 reference words disagree with the Imlaei text, 196.93 per 1,000 words (consonant and shaddah; harakat are not read). Its own column only: qdat's
orthography is not compared with the harness numbers.

## What the intervals cannot carry

The catch sets are small: 67 located mistake sites and 150 edits. Their intervals cannot
carry a fine difference between detectors, and the correct side is a verified sample
enriched for hard sites (the base teacher's own committed text is flagged at 198 of 407), not a population
rate. The verdict is a screen: STOP refuses this model, GO says only worth trying, and
PRIOR says the correct side passed but the catch side or the unique catches did not.

## Provenance and contamination

- The model is CC-BY-4.0 (attribution only); the `NOTICE` beside the decodes carries the
  identity, and no audio or weights are committed.
- Ship A and the edits come from Tadabur. EveryAyah's professional reciters may also be
  in Tadabur; the overlap by reciter is unknown from the committed metadata, so it is
  disclosed as unknown rather than resolved.
- Ship A holds consonant-pair and shaddah sites only. This screen cannot certify haraka.

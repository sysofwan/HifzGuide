# The match_ratio floor is a repeat-length filter

Settles the `match_ratio >= 0.75` floor in the clean-re-read predicate (#66, ADR-0016 decision 9).
All 36 shard-20 rejects carrying a 5+ phoneme repeat were adjudicated by ear, plus the 9 below
0.70 and the 1 excluded by `added_shadda` — 46 clips in the population, 26 judged.

## The floor does not measure what it was thought to measure

`match_ratio = alignment.score / query_phoneme_count`. A repeat of length R adds R to the
denominator while the numerator only pays the affine gap cost (`GAP_OPEN -0.5`,
`GAP_EXTEND -0.1`), so the ratio falls as the repeat grows — mechanically, not incidentally.
Measured over the 36 staged clips:

| | |
| --- | --- |
| `corr(match_ratio, max_insertion_run)` | **−0.589** |
| `corr(match_ratio, run / query_length)` | **−0.825** |
| median repeat length, ratio >= 0.75 | **8 phonemes** |
| median repeat length, 0.70–0.75 | **15 phonemes** |
| longest repeat, ratio >= 0.75 | 18 phonemes |
| longest repeat, below 0.75 | 26 phonemes |

So the floor is a repeat-length filter wearing a ratio's clothes. Raising it does not merely cost
yield — it removes **the longest re-reads first**, which are the hard cases for Muraja's
commit-and-trim (ADR-0009's Pass 2 problem is precisely a long live tail that Smith-Waterman drags
through). Tuning thresholds on 8-phoneme stumbles and shipping them against 22-phoneme re-reads is
a validity problem, not a sample-size one.

## Adjudication

`eval_fixtures/reject_reread_verdicts.jsonl` — 26 clips, by ear, against the audio, the Uthmani
text, the Hafs reference and the vowel-retaining decode.

| band | judged / staged | clean | non-Hafs |
| --- | --- | --- | --- |
| 0.70 – 0.75 | 17 / 17 | **17** | 0 |
| below 0.70 | 9 / 9 | 8 | **1** |
| >= 0.75 | 4 / 19 (spot check) | 4 | 0 |
| shadda | 1 / 1 | 1 | 0 |
| **total** | **31 / 46** | **30** | **1 (3.2%)** |

The `>= 0.75` row is a spot check, not a pass: 4 of 19, all at the low end of the bucket
(0.753–0.779). It is reassurance that the bucket behaves like the band, not clearance of it.
Dropping the floor makes hearing every corpus clip mandatory regardless.

**Every clip in the 0.70–0.75 band is a correct Hafs recitation** whose score was dragged down by
the re-read itself. Below 0.70 the picture holds, with two qualifications recorded in the notes:
one non-Hafs reading (68:41, ratio 0.604), and several clips carrying **neighbour-ayah bleed**
that the ratio cannot distinguish from a repeat, since bleed inflates the same denominator.

## Recommendation

**Drop the ratio floor.** The predicate becomes

```
max_insertion_run >= 5  and  not added_shadda
```

with quality assured downstream instead: the per-word divergence mask (ADR-0016 decision 9) keeps
a genuine mispronunciation out of the assertion range, and adjudication (decision 10) removes
non-Hafs readings, which the floor never could — `normalize_phonemes` deletes short vowels, so a
recitation with every vowel wrong scores a perfect `match_ratio`.

Yield and budget, for a 500-clip corpus:

| predicate | clips / shard | shards for 500 |
| --- | --- | --- |
| ratio >= 0.80 | 6 | 84 |
| ratio >= 0.75 (as shipped) | 19 | 27 |
| ratio >= 0.70 | 36 | 14 |
| **no floor** | **45** | **11** |

### What gets worse

- **Every clip needs adjudication**, not just a sampled band. Non-Hafs is orthogonal to ratio, so
  there is no cheap screen; at the observed 3.8% a 500-clip corpus carries ~19 contaminated clips
  unless each is heard. That is the real cost of removing the floor, and it is a human cost.
- **Bleed enters the corpus unfiltered.** It is not currently detectable (see below), so clips
  carrying it will be staged. Assertions scoped to the covered word range (decision 1) contain the
  damage, but session-start conditions stay noisier than they would be with a recut.
- **`added_shadda` stays a reject** on one clip's evidence alone, which is not evidence. It is
  kept out of conservatism, not measurement.

## Bleed is not detectable with what we have

Three signals were tried against the 9 human-labelled clips, whose notes name the bleed directly:

| signal | result |
| --- | --- |
| `leading_trim` / `trailing_trim` >= 5 | finds **3 of 9** |
| `waqf_detect` edge recut (`recut_start`/`recut_end`) | inherits the same blind spot — it is computed from `alignment.query_start`/`query_end`, the very span the trims come from, so a dragged alignment recuts nothing |
| per-ayah reference comparison (this / prev / next, quarter-slices) | finds **0 of 9** — a re-read's tail is a repeat of *this* ayah's words, so it legitimately scores best against this ayah |

A bleed with `trim = 0` means Smith-Waterman dragged the alignment through it rather than stopping
— the same drag Muraja's ADR-0009 documents, where cheap gap extension consumes unrelated text.
Detection therefore needs new work, and the 9 clips in the fixture with their notes are the
validation set for it.

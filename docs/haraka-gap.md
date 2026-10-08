# The haraka gap: 96% on whole segments, 84% on 5 s windows (#85)

> **This measures agreement with the mushaf, not truth.** Every rate below compares a decode
> with the realized reference (what the mushaf prescribes, in waqf form at a segment end). A
> haraka the model leaves empty may be one the reciter did not say, and nothing here can tell
> the two apart. A Muraja protocol change needs the truth-site comparison, which the baseline
> report adds once the listening exists. **This report recommends no product change.**

Generated numbers: [`haraka-gap-tables.md`](haraka-gap-tables.md) (every arm, per haraka, per
model, with intervals) and [`haraka-gap.json`](haraka-gap.json). Code:
`tools/training/haraka_gap.py` (CLI), `haraka_gap_arms.py` (arms, sites, classes),
`haraka_gap_report.py` (statistics). Terms follow `CONTEXT.md`: haraka = fatha, damma, kasra;
an **empty slot** is a reference haraka the decode left without a mark (`omitted`).

## The answer

The gap between the base teacher's two numbers is the **decoding unit**, not the population
of reciters or clips. On the mining pool the whole segments score **96.45%** [96.06, 96.78] and
ADR-0007's windows **84.71%** [83.95, 85.37], intervals that contain both ADR figures (96.17
and 84.1). On identical window occurrences, decoding the window instead of the whole segment
costs **11.43 pts** (96.14% → 84.71%), 94.7% of the 12.07-point gap. That cost is **97.8% edge
words** (11.18 of 11.43 pts) and sits mostly in two cells: the **word-final haraka of a
window's last word when the window cuts through a segment** (56.62% empty, against 1.06% for
the same sites on the whole segment) and the **word-initial haraka of a window's first word
when it cuts through a segment** (40.79% empty, against 0.14%). Those two cells alone are
6.86 pts, 56.9% of the gap.

The deployed stream (b=0) costs the base teacher 9.74 pts against whole segments, more than
half of it (5.70 pts) at the start of each commit block, where the window that commits it
has no audio before it. At b=1 the steady state costs nothing measurable; the 2.02-point total
is the tail (flush 0.91 + undecoded 1.20 = 2.11 pts, against +0.10 at startup, whose interval
contains zero).

## What was measured

| | |
|---|---|
| audio | the #83 mining pool: 2,084 of its 2,508 clips that ADR-0007's window build accepts (381 reciters, 7.17 h streamed), staged WAVs verified by checksum |
| sites | 58,202 harakat of kept waqf segments, each keyed `(clip, segment, reference index)`; every arm maps its reference back onto these keys, so **every arm and both models score the same sites over the same denominator** |
| arms | `segment` (each waqf segment decoded whole, ADR-0005's unit); `window` (ADR-0007: 5 s grid, 4 s hop, snapped inward to whole words, decoded whole); `stream_b0`, `stream_b1` (the recitation span through `training.decoding`'s stream: 5 s windows, 1 s hop, commit block b, startup rule, tail flush) |
| held fixed | audio, sites, reference strings, feature extractor, bf16 weights, batch size 1, scorer (`tashkeel_eval.vowel_sites`), weights, denominators. One `DecodeFingerprint` per model differing only in `model`; the base whole-segment decodes are identical to the #83 cache for 3,785 / 3,785 segments |
| counting | a site in two overlapping windows counts ½ in each (`window`); `window_occ` counts every occurrence once, as ADR-0007 did |
| weights | 1 / clip inclusion probability (the pool over-samples gemination and consonant-pair clips); unweighted counts are in the tables beside every rate |
| intervals | reciter-clustered percentile bootstrap, B = 10,000, seed 20261008; every difference is recomputed inside each resample on the same draws |
| units | percentage points (pts) of reference harakat, weighted, unless marked |

Excluded, and reported: 424 clips the window build refuses (163 re-read, 137 dropped segment,
69 unphonetizable, 29 over 40 s, 25 repeated recitation, 1 target too long), and 4,329
harakat of eligible clips in words no window holds (a word whose span, which includes any
pause after it, is longer than the 1 s window overlap). Both restrictions are terms of the gap
below.

**Classes.** A window occurrence is an **edge** word (first, last or only word of the window)
or an **interior** word; an edge is split by whether it is also a segment edge (a real pause
beside it) or cuts through a segment (speech continues past it). A stream site is classed by
the protocol region that commits it: **startup** (committed by the first window, `[0, b+1)` s),
**seam** (the 0.2 s either side of a commit boundary in steady state), **block middle**,
**flush** (the last window's blocks after b) and **undecoded** (past the last full window).
Stream times come from the base teacher's whole-segment decode (56,728 sites from the haraka's
own token, 1,285 from a neighbour in its word, 189 interpolated from word times), so a site's
region is the same for both models and every arm.

## The gap, term by term (base teacher, all harakat, weighted)

`0.9617 − 0.841 = 0.1207`. With `S_all` every kept pool segment decoded whole, `S_elig` those
in window-eligible clips, `S` the 58,202 windowed sites, `S_occ` the same counted once per
window occurrence and `W_occ` the window decodes of those occurrences:

```
0.9617 − 0.841 = (0.9617 − S_all) + (S_all − S_elig) + (S_elig − S) + (S − S_occ) + (S_occ − W_occ) + (W_occ − 0.841)
       12.07   =    −0.28         +    0.24         +    0.15      +  (−0.07)    +    11.43        +    0.61        (pts)
```

| term | what it is | pts [95% CI] | share of 12.07 pts |
|---|---|---|---|
| 0.9617 − S_all (96.45) | ADR-0005 corpus vs this pool (reciters, segmentation revision, numerics); not measured here | −0.28 [−0.61, +0.11] | −2.3% |
| S_all − S_elig (96.22) | dropping clips the window build refuses | +0.24 [+0.09, +0.39] | 2.0% |
| S_elig − S (96.07) | dropping words no window holds | +0.15 [+0.09, +0.21] | 1.2% |
| S − S_occ (96.14) | counting overlap words twice, as ADR-0007 did | −0.07 [−0.09, −0.06] | −0.6% |
| **S_occ − W_occ (84.71)** | **decoding a window instead of a segment, same occurrences** | **+11.43 [+10.82, +12.11]** | **94.7%** |
| W_occ − 0.841 | this pool's windows vs ADR-0007's val windows; not measured here | +0.61 [−0.14, +1.27] | 5.1% |

The terms add to 12.069 pts against 12.07 (rounding of the five-decimal JSON). The two
unmeasured outer terms both have intervals containing zero, and their unweighted values are
+0.15 and −0.36 pts: the pool reproduces both ADR numbers, so the gap is not a population
difference. The windowing term splits exactly over the window classes:

| window class | share of occurrences | empty %, window | empty %, same sites whole | pts of the gap [95% CI] | share of 12.07 |
|---|---|---|---|---|---|
| last word, cuts a segment | 19.06% | 25.37 | 0.76 | 4.84 [4.51, 5.18] | 40.1% |
| first word, cuts a segment | 13.81% | 17.20 | 0.91 | 4.22 [4.02, 4.42] | 35.0% |
| only word | 3.73% | 20.89 | 1.74 | 1.23 [0.95, 1.58] | 10.1% |
| first word, at a segment start | 12.53% | 13.89 | 12.78 | 0.83 [0.66, 1.01] | 6.9% |
| last word, at a segment end | 7.46% | 2.28 | 1.70 | 0.06 [0.01, 0.10] | 0.5% |
| interior words | 43.41% | 1.11 | 0.65 | 0.25 [0.15, 0.35] | 2.1% |
| **edge words (all but interior)** | 56.59% | 17.50 | 3.65 | **11.18 [10.57, 11.85]** | **92.6%** |

So **edge words carry 11.18 of the 11.43 windowing points (97.8%) and interior words 0.25**;
the interior cost is above zero (interval [0.15, 0.35], 0.20 pts more empty slots than on the
whole segment) but 44 times smaller. Edges that coincide with a pause cost little: 0.06 pts at
a segment end, and 0.83 at a segment start, where the whole-segment decode already leaves
12.78% empty (the segment's own onset). The two edges that **cut through a segment** carry
9.07 pts, 75.1% of the gap.

**Where in the edge word.** Split by the haraka's place among its word's harakat
(`window_occ`, base):

| cell | occurrences | empty %, window | empty %, whole segment | pts of the gap | empty slots only in window / only on segment |
|---|---|---|---|---|---|
| last word cutting a segment, its **last** haraka | 3,910 | 56.62 | 1.06 | 3.45 [3.22, 3.68] | 2,089 / 2 |
| first word cutting a segment, its **first** haraka | 2,931 | 40.79 | 0.14 | 3.42 [3.22, 3.61] | 1,231 / 0 |
| last word cutting a segment, first haraka | 3,910 | 8.15 | 0.65 | 0.58 | |
| first word cutting a segment, last haraka | 2,931 | 3.63 | 2.32 | 0.14 | |

The cost is at the cut itself: the word-final haraka just before the window ends, and the
word-initial haraka just after it begins. At the cut first word, only 24.8% of those
harakat match (99.6% on the whole segment); 40.8% are empty and the remaining 34.4% are
almost all `unanchored` (the haraka on a misheard carrier; the window arm's swapped rate is
0.04% overall), so the onset consonant is lost too: 1.86 of that cell's 3.42 pts are empty
slots. This is consistent with the model treating a cut window end as a
pause (dropping the final haraka as in waqf) and a cut window start as an onset, but it does
not separate that from audio truncation: a window ends at the next word's forced-alignment
onset and is rounded in by up to 40 ms, so a word-final haraka that runs into the next word
may be partly outside the window.

## The stream: startup, seam and tail

Same 58,202 sites, against the whole-segment decode (base teacher):

| region | b=0: share | b=0: empty % (vs whole) | b=0: Δ matched, pts | b=1: share | b=1: empty % (vs whole) | b=1: Δ matched, pts |
|---|---|---|---|---|---|---|
| startup (first window) | 12.98% | 9.62 (12.16) | +0.07 [−0.05, +0.19] | 23.94% | 5.35 (6.83) | +0.10 [−0.03, +0.23] |
| seam, block start | 10.82% | **39.35** (0.91) | **−5.70 [−6.11, −5.31]** | 10.47% | 1.09 (1.00) | −0.06 [−0.16, +0.04] |
| block middle | 32.98% | 4.20 (0.99) | −1.73 [−1.95, −1.52] | 32.34% | 0.83 (1.02) | +0.04 [−0.02, +0.10] |
| seam, block end | 10.39% | 3.04 (0.72) | −0.27 [−0.34, −0.19] | 10.15% | 0.64 (0.77) | +0.02 [−0.01, +0.04] |
| flush (last window) | 31.54% | 3.56 (0.85) | −0.92 [−1.13, −0.73] | 21.82% | 4.84 (0.92) | −0.91 [−1.12, −0.73] |
| undecoded tail | 1.28% | 99.50 (3.10) | −1.20 [−1.42, −0.99] | 1.28% | 99.50 (3.10) | −1.20 [−1.42, −0.99] |
| **total** | | 9.61 (2.39) | **−9.74 [−10.37, −9.16]** | | 4.06 (2.39) | **−2.02 [−2.40, −1.65]** |

The contributions add to the totals (−9.74 and −2.02). Read separately:

- **Startup** costs nothing measurable at either block (both intervals contain zero): the
  recitation's first second does no worse streamed than on the whole segment, where the onset
  is already hard.
- **Seam.** At b=0 the block start is the start of the committing window's audio, with no
  left context at all; it is 39.35% empty and carries 5.70 of the 9.74 pts (58.5%). The block
  end costs 0.27. At b=1 the same boundaries have a second of left context and the seam,
  block middle and their sum (0.00 pts) are indistinguishable from the whole segment.
- **Tail.** The flush costs 0.92 (b=0) and 0.91 (b=1). The **undecoded** tail (under a second
  past the last full window, which the replay never decodes; `training.decoding` lists it as a
  gap to the deployed protocol, whose handling on device is unverified) costs 1.20 at both
  blocks. It is a replay artefact as much as a protocol one, so it is reported apart.
- b=1 minus b=0: **+7.72 pts matched [+7.25, +8.20]**, −5.55 pts empty [−5.94, −5.18].

The stream at b=0 and the ADR-0007 windows land close (86.33% and 84.71%) for different
reasons: windows cut at word boundaries on both sides, the b=0 stream at arbitrary times on
one side only.

## Per haraka (base teacher, matched %, weighted)

| arm | fatha | damma | kasra | all |
|---|---|---|---|---|
| segment | 95.72 | 96.06 | 97.01 | 96.07 |
| window | 85.41 | 85.89 | 85.10 | 85.43 |
| window_occ | 84.77 | 85.46 | 83.93 | 84.71 |
| stream b=0 | 87.12 | 82.80 | 87.19 | 86.33 |
| stream b=1 | 93.99 | 94.13 | 94.13 | 94.05 |

Damma is the haraka the b=0 stream hurts most: −13.26 pts against the whole segment
[−14.66, −11.93], against −8.60 for fatha [−9.25, −7.98] and −9.83 for kasra [−11.15, −8.52];
damma's interval lies entirely below fatha's (two separate intervals, not a paired test of
the difference). Its block-start seam is 37.36% empty (fatha 19.00%,
kasra 15.26%). In the windows the three harakat lose similar amounts (−10.32, −10.17, −11.91
pts on per-site counting, overlapping intervals). Intervals and counts per haraka, per arm and
per class are in the tables.

## h448

| arm | matched % | empty % | vs whole segment, Δ matched pts |
|---|---|---|---|
| segment | 92.85 [92.28, 93.40] | 3.43 | |
| window | 90.84 [90.41, 91.20] | 4.63 | −2.01 [−2.52, −1.51] |
| window_occ | 90.48 [90.05, 90.84] | 4.83 | −2.45 [−2.93, −1.97] |
| stream b=0 | 87.50 [86.84, 88.13] | 8.25 | −5.35 [−5.82, −4.88] |
| stream b=1 | 94.67 [94.19, 95.13] | 3.54 | +1.82 [+1.37, +2.28] |

`h448` was distilled on the 5 s stream, so whole segments are not its home ground: it scores
lower on them than on its own b=1 stream (+1.82 pts), and its empty-slot rate at b=1 is
indistinguishable from whole segments (+0.10 pts [−0.31, +0.49], net −42 sites). Windows cost
it 2.45 pts, not 11.43: it keeps the word-final haraka at a cut window end (5.66% empty
against the base teacher's 56.62%) but still loses the word-initial one after a cut start
(37.75% empty, 2.88 pts). On the stream at b=0 its block-start seam is 35.94% empty and costs
5.17 pts; b=1 minus b=0 is +7.18 pts matched [+6.71, +7.64].

## What this does not show

- **Truth.** An empty slot at a cut edge may be what a listener would also refuse to grade,
  and a matched haraka may be the text prior (ADR-0011). Only the truth-site comparison can
  say which protocol is more *correct*.
- **The device.** The replay ignores the VAD gate and preview inferences and never decodes the
  sub-second tail past the last full window (`training.decoding`).
- **Why the two outer terms are small.** Neither the ADR-0005 corpus nor the ADR-0007 val
  windows survive (`audit_run/` was cleared), so the outer terms are residuals, not
  measurements; that they are within a point of zero says the pool reproduces both numbers,
  not why.

## Reproduce

From `tools/` on the GPU box (decode needs the staged pool audio; `report` is CPU-only and
deterministic, and ran in about a minute):

```bash
flock /root/scratch/gpu.lock python -m training.haraka_gap decode \
    --audio-dir /root/scratch/issue-83/stage/clips --cache-dir /root/scratch/issue-85/cache \
    --model base=obadx/muaalem-model-v3_2 \
    --model h448=/root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt \
&& python -m training.haraka_gap report --cache-dir /root/scratch/issue-85/cache \
    --out-json ../docs/haraka-gap.json --out-md ../docs/haraka-gap-tables.md
```

The decode took roughly half an hour per model (one RTX 5060 Ti, batch size 1). The caches
(`base.json`, `h448.json` and each model's per-window stream argmax rows, 8.8 MB together)
stay in `/root/scratch/issue-85/cache`.

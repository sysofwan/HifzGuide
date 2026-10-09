# Shaddah probe (#86): pre-registration

**Status:** fixed 2026-10-08, **before** any posterior, duration or stretch result on the mining pool
was computed. The commit that adds this file precedes the commit that adds the results
(`docs/shaddah-probe.json`, `docs/shaddah-probe.md`); `git log --follow` on both shows the order.
The constants named here are the ones in `tools/training/shaddah_probe.py` in the same commit.
Nothing below is an acceptance threshold: those come from the decision issue (#92) under
`docs/acceptance-rules.md`, where every shaddah rule is provisional until #92 freezes the
held / not held / unsure state table.

Terms follow `CONTEXT.md` (shaddah, haraka, tashkeel, decode).

## What was seen before this was written

- The committed #83 artifacts: the census counts in `tools/tadabur/mining_pool/` (235 geminates the
  base decode left single across the pool) and, from the committed base decodes, the consonants of
  those 235 census sites (و 121, ن 30, ل 14, ت 13, ب 10, others fewer). No posterior of a pool
  segment had been computed.
- The base teacher's posteriors on 25 P3.5 segments (`seg_p35`, not in the pool), to see their shape.
  The output is **peaky**: a token's argmax run is 1-2 frames (40 ms each), 45-85% of frames are
  blank, and a geminate decoded double is two spikes of the consonant ~7-9 frames apart with blanks
  between (`مَسسَ`: س at frames 79 and 86). A "mid-run dip" inside a long consonant run therefore
  cannot be the main shape; the second consonant, if present, is expected as a **sub-argmax second
  spike** in the blank frames around the first. Both are measured (below).

## Population

Every **kept** segment of the mining pool (`tools/tadabur/mining_pool/clips.jsonl`, 3,785 segments,
2,508 clips, 394 reciters), whose audio is the staged clip's `[start_sample, end_sample)` verified
against the staging registry's checksum. Both strata the issue names are included: the census
`geminate` stratum and the uniform draw (and `consonant_pair`, which is part of the same pool).

Each model decodes each segment **whole**, through `training.decoding.Decoder` with bf16 weights,
batch size 1, CUDA autocast: the base teacher's committed decodes were made this way and the base pass
must reproduce them exactly (mismatches are reported). `h448` is decoded under the same fingerprint so
the two models are comparable (its historical agreement numbers use fp32 weights and the stream).

## Sites and populations

- A **site** is a maximal run of one consonant (ids 1-28 of the vocabulary) in the segment's realized
  reference. Length ≥ 2 is a **geminate**; length 1 a **single**.
- The greedy decode's tokens are aligned to the reference with word spaces removed (Levenshtein,
  `training.edit_decomposition.align`, diagonal first). A site's `decoded` count is the number of its
  consonant's tokens matched onto the run or inserted against it.
- Populations, per model: **collapsed geminate** = geminate with `decoded == 1` (the decode emitted
  one consonant); **double geminate** = run of exactly 2 decoded 2; **single** = single decoded 1.
  The census definition of #83 (`contrast_sites`, raw-string check) is reported as a cross-tab
  against `decoded`, not used to select.
- **Weights**: every site carries 1 / its clip's `inclusion_probability`. Shares are reported weighted
  (headline) and unweighted. Intervals are reciter-clustered percentile bootstraps of the weighted
  share, B = 10,000, seed 20261008.

## Q1: posterior mass of the second consonant

- **`log_ratio`** (primary): `log P(doubled transcript) − log P(single transcript)` under CTC
  (forward algorithm over the whole segment), the rest of the decode held fixed. At a collapsed site
  the doubled transcript inserts a second consonant next to the decoded one; at a double geminate the
  single transcript removes one.
- **Present / weak / absent** (pre-registered): present if `log_ratio ≥ ln 0.1` (the doubled reading
  has at least a tenth of the single reading's posterior); absent if `log_ratio < ln 0.001`; weak in
  between. Reported with the full curve: the weighted share with `log_ratio ≥ θ` at
  θ ∈ {−30, −20, −15, −10, ln 0.001, −6, −5, −4, −3, ln 0.1, −2, −1, 0, 1, 2, 5}, and quantiles.
- **`second_peak`** (secondary): the highest posterior of the consonant on any frame of the site's
  interval (below) outside its decoded run. Present if ≥ 0.1; curve at
  {0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}.
- **Where** the mass sits: the Viterbi alignment of the doubled transcript puts the extra consonant
  `before` or `after` the decoded run, `split` (it carves the run: a mid-run dip), or `elsewhere`.
- Reported for collapsed geminates, double geminates and singles (singles are the baseline: the same
  statistic where the reference has no geminate), overall and for every consonant with ≥ 10 collapsed
  sites.

## Q2: stretch the held segment

- The **held segment** of a site is its **interval**: the frames strictly between the decoded token
  before the site and the one after it, which contain every spike of the consonant; in samples,
  `[start × 640, end × 640)` of the segment.
- Edit: WSOLA in place (`tadabur.time_stretch.stretch_span`, 20 ms Hann frames, ±5 ms search; all
  audio outside two frames of the span untouched) by factors **1.25 and 1.5** (the issue's) and
  **2.0** (added to see the curve's end).
- **Decoy**: the same unedited segment with the same number of zero samples appended, so it has the
  edited audio's length (feature normalization is per utterance).
- **Matched single control**: for each collapsed geminate, the nearest single site of the same
  consonant in the same segment (else another segment of the same clip; each control used once),
  stretched and decoyed the same way. It measures how often stretching *any* such interval makes the
  model emit a double.
- Outcome: the site decodes double (`decoded ≥ 2`) after the edit; **net** = edited − decoy share.
  Reported per factor and population, with the `log_ratio` after the edit and the number of decode
  characters the edit changed elsewhere.

## Q3: held-segment durations

The interval (ms) at sites whose flanking tokens are both harakat (one phonetic context, V_V), for
double geminates, collapsed geminates and singles; and the same **rate-normalized** by the median
interval of the segment's V_V singles (segments with ≥ 3 of them). Quantiles p10-p90 and the curve of
the share with normalized interval ≥ r for r ∈ {0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3}.
**Geminate-like**: a collapsed site's normalized interval is at least the midpoint between the
double-geminate and single medians.

## Verdict (per model)

Two readings of a collapsed geminate:

- **Representation**: the model perceives a geminate-length hold, but its output (a doubled consonant
  that needs a blank between two spikes) does not surface it. A decode rule or a gemination class is
  the lever.
- **Data**: the hold the model perceives is single-length and its output carries no second consonant:
  the decode is consistent with the reciter not holding it, and what is missing is truth (human
  labels), not an output representation.

Evidence: **E1** = geminate-like share (weighted) of collapsed V_V sites; **E2** = present share of
collapsed geminates − present share of singles; **E3** = net stretch flip of collapsed geminates −
net stretch flip of their single controls, at ×1.5.

| condition | call |
|---|---|
| E1 ≥ 0.5 | **representation** |
| E1 < 0.5 and E2 < 0.10 and E3 < 0.10 | **data** |
| otherwise | **mixed** (the report says which evidence disagrees) |
| an input undefined | insufficient_evidence |

**Candidate decode rule** (for #92 to accept, change or reject): at a geminate, *held* if the decode
is double or `log_ratio ≥ ln 0.1`; *unsure* if `ln 0.001 ≤ log_ratio < ln 0.1`; *not held* below.
It is reported as **looking viable** when, at these thresholds, it calls ≤ 2% of singles held,
≤ 10% of singles held-or-unsure, and its held share at collapsed geminates exceeds its held share at
singles by ≥ 10 points. Whatever the call, its full state table per population and the `log_ratio`
curve are reported, so #92 can place its own operating point. A duration rule is reported as a curve
only; no operating point is proposed here.

## Not decided here

Any operating point, threshold or margin of a shipped rule (#92); whether a gemination class is
added (owner decision on #92's evidence). No truth labels are read: without human verdicts at
collapsed sites, "held" in this probe always means "held as the model's output reads it".

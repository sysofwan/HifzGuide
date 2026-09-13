# Re-read seams: what the VAD sees there, and what excising them yields

Settles HifzGuide #64 / Muraja ADR-0016 decisions 3 and 4. Two questions, in the order the
issue insisted on: **is there a pause where a re-read seam is?** — measured before any
excision machinery was written — and then **how many usable differential pairs does
cutting at those seams actually produce?**

Everything below is measured on the shard-20 reject pile: 71 rejects of 1,000 clips, of
which **45** meet the clean-re-read predicate (`max_insertion_run >= 5 and not
added_shadda`, the floor dropped in `docs/tadabur-ratio-floor.md`).

## Deliverable 0 — pause coverage at a real seam

ADR-0016 constraint 4 quotes a 0.7% spurious / 21% missed pause-detection rate from the
2,050 adjudicated boundaries in `waqf_event_fixtures/`. Those are **waqf boundaries,
which are pauses by definition**. #67 measured the same thing at an *ayah edge* and found
80% coverage — again a boundary class where stopping is what the reciter is doing. A
re-read seam is neither, and ADR-0016 decision 4 says so plainly: the VAD surfaces only
silences >= 300 ms bounded by speech spans >= 700 ms, "and a reciter doubling back on a
single word may leave none".

Reproduce with (no GPU — it runs off the staged decodes #68 already produced):

```bash
python -m tadabur.seam --decodes bleed_run/decodes.jsonl \
  --rejects reject_run/rejects.jsonl --recuts bleed_run/recuts.jsonl \
  --json bleed_run/seam_pauses.json
```

### The seam, located

A re-read is an **insertion run** in the alignment: a stretch of decoded phonemes the
reference does not contain. `tadabur.seam.insertion_runs` recovers each run's query *and*
reference coordinates, which the gate's `longest_insertion_run` does not — it only ever
needed the length. The predicate and the geometry agree exactly: **all 45 clips carry at
least one run, 50 runs in total**, with 3 clips carrying more than one (a reciter can
double back twice). No run falls outside its clip's post-re-cut recitation span.

| | |
| --- | --- |
| Repeat length | 5 – 26 phonemes, median **11** |
| Repeat duration | 0.76 – 5.80 s, median **2.44 s** |

These are words and phrases, not stumbles — which is what dropping the ratio floor was
for.

### The pause coverage

A cut can only be snapped to a pause it can *reach*: one lying wholly inside the window
between the previous decoded phoneme's onset and this one's. That window is exactly as
wide as the cut may travel without eating a phoneme on either side, and a silence of the
VAD's own minimum length fits inside it precisely when the reciter paused between those
two phonemes. It is the same predicate `bleed_recut` uses at an ayah edge.

| edge of the excised span | seams with a usable silence |
| --- | --- |
| opening | 21 / 50 — **42%** |
| closing | 4 / 50 — **8%** |
| either | 25 / 50 — 50% |
| **both** | **0 / 50 — 0%** |

**No seam in the pile has a pause at both of its edges.** Five clips have no interior VAD
silence at all. Across the 65 unanchored edges the nearest silence anywhere in the clip
is a median **1.65 s** away (0.0 – 7.55 s), so this is mostly not a near miss: there is
no pause to snap to.

The asymmetry is not noise, and it says what the seam is. Where Smith-Waterman keeps the
*first* utterance of a repeated phrase and calls the second one the insertion — the usual
outcome on real clips — the excised span **opens** where the reciter re-started, which is
immediately after the hesitation, and **closes** mid-phrase where he carried on. So the
opening edge inherits the pause and the closing edge cannot have one. The tell is
arithmetic: on nearly every anchored-opening seam, the closing edge's nearest-silence
distance equals the repeat's own duration — the single nearby silence is the one at the
other end. (Four seams run the other way, the aligner having kept the later copy; on
those the pause is at the closing edge and the opening edge has none.)

Two opening edges sit at distance 0.0 and are still unanchored: their onset falls inside
a silence that *starts before* the previous phoneme's onset, so no cut could use it
without eating that phoneme. Four closing edges miss by <= 0.19 s. Relaxing the window
would move a handful of edges and would not move the 0.

### What this settles

**A pause-bounded excision is not available.** That was the version of decision 4 this
issue was opened against — "excised at its bounding VAD pauses" — and it does not survive
contact with the seams.

It does not follow that the differential is unavailable, because the amended decision 4
had already stopped depending on the pause:

> They are deliberately *not* required to be pause-anchored ... Safety does not rest on
> the cut being pause-anchored, because every excised clip is re-run through the
> `.balanced` gate; if it does not come back clean the pair is discarded. A bad cut
> therefore becomes **yield loss, never a false finding**.

So the cut is placed at the CTC onset bounding the repeat, moved onto a silence at
whichever single edge happens to have one, and disbelieved until the re-gate says
otherwise. Whether that is worth doing is a **yield** question, and the rest of this
report answers it by measuring it rather than by arguing from the 0%.

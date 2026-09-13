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

## The stager

`tadabur.scenario` turns the reject pile into the bundle Muraja consumes. One command, one
model load, one pass:

```bash
python -m tadabur.scenario --rejects reject_run/rejects.jsonl \
  --clips bleed_run/clips/ --recuts bleed_run/recuts.jsonl \
  --out scenario_run/shard20
```

1. select the clean re-reads, in `clip_id` order, so a `--limit` run is a **prefix** of the
   full one rather than a sample of it;
2. cut each clip to the recitation span #68 found — the bleed re-cut is applied **during**
   staging, because retrofitting it afterwards means re-decoding the whole corpus;
3. VAD the result, free the VAD, decode the result (the two models never co-reside);
4. gate it, locate its seams, and map the alignment's reference span to a **word** range;
5. cut every repeat out, re-decode, re-gate, and keep the pair only if it comes back clean.

The manifest schema is documented in `tools/README.md`. Two runs of the same command produce
byte-identical output, checked by running `--limit 4` twice and diffing the bundles.

### What one shard stages

| | |
| --- | --- |
| Clean re-reads staged | **45** (45 selected, none skipped) |
| Distinct reciters / ayat | 33 / 43 |
| Staged audio | 18.7 min, 35 MB |
| Re-read seams | **51** in 45 clips (41 clips with one, 2 with two, 2 with three) |
| Bleed re-cut applied | 4 (12:48, 18:57, 18:18, 2:278 — the four #68 accepted on this population) |
| Truncated (`uncovered_tail > 0`) | 6 — kept with a shortened range, per #68 |
| `early_start` flagged | **1** (40:16, `leading_trim` 11) |
| Words the oracle may assert over | **873**, median 18 per clip (7 – 45) |

`early_start` firing once in 45 is the confirmation that #68's re-cut did its job: the flag
exists for the clips where a lead-in survives, and after bleed-clipping there is almost never
one. Muraja #128 should read it per clip, as ADR-0016 decision 3 says, and not start every
session an ayah early.

**The covered range costs a word at the head on 18 of 45 clips** — 13 lose one reference
phoneme and 5 lose two, and a word is admitted only when *every* phoneme of it is covered. That
strictness is deliberate: a word the decode only reached halfway into is a word the oracle
cannot grade, and admitting it would manufacture exactly the false-skip class ADR-0016 exists
to measure. It costs roughly 5% of assertable words, and it is the setting to revisit first if
#129 finds the corpus short of coverage rather than short of clips.

## Excision yield

**44 of 45 pairs survived re-gating — 97.8%.**

| | |
| --- | --- |
| Seams found | 51 |
| Pairs attempted | 45 |
| Pairs kept | **44** |
| Discarded | 1 (`ratio_too_low`) |
| Control audio | 16.0 min, 30 MB |
| Audio excised | 141.8 s across 51 cuts, median 2.52 s (0.96 – 5.80 s) |

The gate's verdict moves the way the cut claims it should, on every kept pair:

| | before | after |
| --- | --- | --- |
| `match_ratio` (min / median / max) | 0.521 / 0.746 / 0.863 | **0.819 / 0.895 / 0.928** |
| `max_insertion_run` | 5 – 22 | 0 on 39 clips, 1 – 4 on 5 |
| `leading_trim` / `trailing_trim` after | — | 0 / 0 on 44 and 42 of 44 |

Every pair's ratio **rose** (median +0.141, minimum +0.031), which is the excision's whole
claim: the repeat was inflating `query_phoneme_count`. The trims staying at zero is the
independent check that no cut landed inside real recitation — a cut that ate a phoneme shows up
there first, as the aligner starts trimming the new edge.

The 5 clips with a residual run of 1–4 phonemes are the **splice artifact**: joining two
non-silent points leaves about a phoneme's worth of the removed span behind, and the model
transcribes it. All five sit well below the gate's bar of 5, so the artifact costs a mismatch,
not a pair.

### The pause did not matter

This is the number deliverable 0 was measured for:

| clips | pairs kept |
| --- | --- |
| with at least one pause-anchored cut edge (22) | **22 / 22** |
| with no anchored edge at all (23) | **22 / 23** |

Pause anchoring makes **no difference to yield**. Cutting at a bare CTC onset produces a
control clip the gate accepts just as readily as cutting through a silence. That is the
amended decision 4 vindicated: the re-gate, not the pause, is what makes the cut safe, and the
0% coverage deliverable 0 found turns out to cost nothing.

It is worth being precise about why. The excised span usually *opens* just after the reciter's
hesitation, so removing it removes the second utterance and leaves the pause in place: the
control clip reads as "…phrase, [pause], continues", a reciter making an ordinary mid-ayah
waqf. There is nothing acoustically strange about the result, which is why the gate does not
notice the join.

## The one discarded pair — a worked example

`tadabur_spk0088_S39_A16_33e18a4f_000007`, **40:16**, 21.7 s. The reference is

```
يوم همبارزۥن لا يخفا عل للاه منهم شيء للمن لملك ليوم لللاه لواحد لقههار
```

and the decode (normalized, so vowel-free) is

```
يومهمبارزۥن يومهمبارزۥن لايخفاعلللاه لايخفاعلللاه منهمشيءلمنلملكليوملللاه
```

Two repeats, and the clip stops before `لواحد لقههار` — it is also one of #68's four truncated
clips (`uncovered_tail` 13). The seam finder reports **one** seam: query 28–39, the second
`لا يخفا عل للاه`, sitting at reference position 20. It cannot report the first, and this is
structural — **a repeat at a clip edge is not an insertion run.** The local aligner trims it
instead, which is what `leading_trim = 11` is, and what made this the single clip in the pile
where `early_start` fires.

So the excision removes the interior repeat and does exactly what it was asked to:
`max_insertion_run` 11 → **0**, `match_ratio` 0.565 → 0.698. And the pair is discarded anyway,
because 0.698 is below the 0.75 bar — the *leading* repeat is still there, still inflating
`query_phoneme_count`, and the control clip is therefore not the clean recitation the
differential needs it to be. The control decode is kept on the record (`excised_phonemes`) so
this is readable from the artifact rather than only from a log.

This is the yield-loss-not-false-finding property working exactly as ADR-0016 decision 4
designed it. Nothing here produced a wrong answer; one clip produced no answer, and said why.

It also names the one cheap improvement available: an edge repeat is visible in `leading_trim`
/ `trailing_trim` and could be cut with the same machinery. It would have recovered this pair.
It is not done here — one clip in 45 does not justify a second cut path whose failure mode
(eating the start of the recitation) is the one the trims cannot distinguish from success.

## Handoff to Muraja

The bundle is a single directory. Ship it and check it:

```bash
# On the Mac — 64 MB for shard 20's 45 clips.
rsync -av root@cuda-dev:/root/repos/HifzGuide/tools/tadabur/scenario_run/shard20/ \
  ~/repos/Muraja/tools/tadabur_corpus/

# Verify the transfer. Needs no model, GPU or reference cache — json and the filesystem.
cd ~/repos/HifzGuide/tools
python -m tadabur.scenario --verify ~/repos/Muraja/tools/tadabur_corpus/
```

`--verify` exits non-zero and names every problem: a missing WAV, a path that escapes the
bundle, a duplicate `clip_id`, a `schema_version` from a different normalization, and — the
checks a truncated transfer would *not* trip — a record covering no words or carrying no seam.

```
tadabur_corpus/
  scenario.jsonl        68 KB   one record per clip, keys sorted, clip_id order
  staging.json          the run's tallies
  audio/<clip_id>.wav   35 MB   16 kHz mono PCM_16, bleed already cut
  excised/<clip_id>.wav 30 MB   the control clip, present only for a kept pair
```

**Budget.** 1.4 MB per clip staged. A 500-clip corpus is therefore **~710 MB**, of which ~380 MB
is the staged audio and ~330 MB the control clips; `scenario.jsonl` reaches ~750 KB. The Mac
side should plan for that, not for the 64 MB of one shard.

**Cost on cuda-dev.** 47 s for 45 clips (one VAD pass, two decode passes, two model loads), and
64 MB of output. Against #70's 217 s/shard for the filter itself this is noise; the staging
disk is what grows.

## Limits

- **One shard.** 45 clips, 51 seams, one 97.8% keep rate. The *rates* will move; #70 is what
  measures them at corpus scale.
- **An edge repeat is invisible to the seam finder** — it is a trim, not an insertion run — so
  a clip carrying one stages fine but its pair may fail the re-gate on the repeat that was
  never cut. Observed once in 45, and it is the clip `early_start` flags.
- **The covered range is strict at word granularity.** 18 of 45 clips lose their first word to
  a one- or two-phoneme miss at the head. Deliberate, and revisitable by #129.
- **The splice is not free.** Five of 44 control clips carry 1–4 phonemes of residue at the
  join. Below the gate's bar, but it is there, and a differential that asserted on phoneme
  counts rather than word grades would see it.
- **Decode nondeterminism is bounded, not absent.** Two `--limit 4` runs of the same command
  are byte-identical, which holds because clip order and batch composition are fixed. Change
  `--batch-size` and #68's observation stands: bf16 batching can move a madd length or a
  diacritic.
- **The corpus is still not usable until it is heard.** ADR-0016 decision 10 requires every
  clip to be adjudicated for non-Hafs readings, which no automated screen can do. 31 of the 46
  repeat-carrying shard-20 clips are judged (`eval_fixtures/reject_reread_verdicts.jsonl`); the
  rest, and every clip #70 adds, are not.

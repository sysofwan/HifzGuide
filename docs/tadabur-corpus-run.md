# The re-read corpus run: 11 shards, 445 staged clips

**11,000 Tadabur clips streamed, 447 clean re-reads mined, 445 staged and verified on the Mac.
46 minutes of wall clock and 1.3 GB of disk. The corpus carries 8,402 assertable words across
153 reciters, and its repeat lengths run 5 to 44 phonemes with a median of 10 — which is the
number the ratio floor was dropped to get.**

HifzGuide #70, the corpus run Muraja ADR-0016 exists to produce. Everything it needed had
landed: the reject sink (#63), the bleed detector and re-cut (#67/#68) and scenario staging with
the excision differential (#71). What had never happened is the four stages running **in
sequence** — shard 20's artifacts were produced by hand with a human between each stage — and
one stage was not in the repo at all.

## How to reproduce

```bash
cd /root/repos/HifzGuide/tools
RUN=tadabur/corpus_run SHARDS=20-30 tmux new-session -d \
  'bash tadabur/run_corpus.sh 2>&1 | tee /root/corpus.log'
```

`tadabur/run_corpus.sh` is new here, and so is stage 2. The pipeline is five stages and their
order follows from where the cost is: **only stage 1 is resumable and only stage 1 is per
shard**, so it runs across every shard first and the rest run once over the accumulated sink.
Chaining all five per shard would have paid the ~11-minute downstream tail eleven times.

| | stage | what it does | this run |
| --- | --- | --- | --- |
| 1 | `tadabur.filter` | decode + gate each clip, keep rejects, stage the clean re-reads' WAVs | **34m47s** (10 shards) |
| 2 | `tadabur.bleed_stage` | VAD intervals + per-frame CTC class ids, so a phoneme has a *time* | **3m49s** |
| 3 | `tadabur.bleed_recut` | find neighbour-ayah bleed, cut it, re-decode and re-gate, keep or drop | **33s** |
| 4 | `tadabur.scenario` | re-cut, seam, word range, excision pair, bundle + `--verify` | **6m23s** |
| 5 | `reject_yield` + `bleed_detect` | the numbers below | **29s** |
| | | **total** | **46m01s** |

Stage 2 is the one that did not exist. `bleed_recut` needs per-frame class ids and VAD
intervals, and neither the reject sink nor the staged WAV carries them; the shard-20 artifacts
came from a script in `/root` that re-read the parquet shard to get them. On an 11-shard run
that means re-downloading 26 GB `--delete-shards` had just discarded, so `tadabur.bleed_stage`
decodes the **staged WAV** instead — the same 16 kHz mono waveform the gate scored.

It reports how often the restaged decode reproduces the stored one, which is the check that it
is reading the right audio: **432 of 447 (96.6%)**. Every one of the 15 that differ differs by a
**single phoneme** (one by two) in decodes 80–454 characters long — 8 deletions, 7 insertions,
1 substitution in total. That is bf16 batch composition, not different audio: the filter batches
in shard order and stage 2 in filename order, so a clip sits beside different padding and an
occasional greedy argmax flips.

## Yield against the budget

The budget assumed 45 clean re-reads per shard, measured on shard 20.

| | this run | budget assumed |
| --- | --- | --- |
| Shards | 11 (shard 20 resumed, 10 streamed) | 11 |
| Clips processed | 11,000 | |
| Passers | 10,096 (91.8%) | |
| **Rejects** | **812 (7.4%)** | 7.1% on shard 20 |
| Skipped before the gate (over 50 s) | 92 (0.8%) | |
| Failed on `min_query` | 1 | 0 on shard 20 |
| **Clean re-reads** | **447** | ~500 |
| …per shard | **40.6** | **45** |
| …as a share of rejects | 55.1% | |
| Shards for 500 at this rate | 12.3 | 11 |

One clip in 11,000 failed on `min_query`, which shard 20's report said it had never seen — "on
real recitation the gate always finds something to align". It still essentially does; the claim
just needed an order of magnitude more clips to find its exception.

**The budget held to 90%.** 11 shards yields 447, not 500 — a shortfall, not a miss, and the
stopping rule was never a clip count anyway (ADR-0016 settles corpus size on the scoreboard's
metric distribution stabilising, which is Muraja #129 and does not exist yet).

The halfway re-check, run against the partial sink at 6 shards as the issue asked:

| | halfway (6 shards) | final (11 shards) |
| --- | --- | --- |
| Clips processed | 6,000 | 11,000 |
| Clean re-reads | 252 | 447 |
| Per shard | 42.0 | 40.6 |
| Shards for 500 | 11.9 | 12.3 |

It said the budget nearly held and would land near 460, and it did. The drift from 42.0 to 40.6
is the second half yielding 39.0/shard — a mild decline, not a cliff.

**A top-up costs one command.** The sink is checkpointed and idempotent, so reaching 500 is
`SHARDS=20-32` over the same `RUN` — stage 1 skips the 11 finished shards and streams 31 and 32.
That is the right thing to do *after* #129 can measure stability, not before.

## Reciter and ayah spread

Tadabur shards are reciter-ordered, so the worry was that a long run reaches fewer distinct
reciters per clip than one shard's spread suggests.

| | shard 20 alone | this run |
| --- | --- | --- |
| Clips | 19 (pre-floor-drop) | 447 |
| Distinct reciters | 18 | **153** |
| Largest single reciter | 2 clips (10.5%) | **31 clips (6.9%)** |
| Top five reciters | — | 87 clips (19.5%) |
| Distinct ayat | 18 | **368** |
| Largest single ayah | 1 | 4 |

**Concentration did not compound.** The top reciter held 6.7% of the corpus at the halfway mark
and 6.9% at the end, so streaming further did not tip the corpus toward one person's habits — it
is *less* concentrated by share than shard 20 was. 153 reciters is the number to carry forward;
"447 clips" would overstate the independence between them, but not by much.

## Repeat lengths — what dropping the ratio floor was for

The floor was dropped (`docs/tadabur-ratio-floor.md`) precisely to keep the long re-reads that
are ADR-0009's hard case. The test stated in #70 was whether the median lands near shard 20's 8
phonemes or spans past 20.

| | min | p25 | **median** | p75 | p90 | max |
| --- | --- | --- | --- | --- | --- | --- |
| `max_insertion_run`, 447 clean re-reads | 5 | 7 | **10** | 13 | 18 | **44** |

93 clips (20.8%) carry a repeat of 15 phonemes or more, and the tail reaches 44. Shard 20's
bucket topped out at 18. **The corpus is what the decision intended.** The mass is still at the
short end — 211 of 447 sit at 9 phonemes or fewer — but that is the shape of real recitation
rather than an artifact of the threshold. Across all 812 rejects the run histogram is still the
bimodal one shard 20 showed, though the "empty gap at 2–4" it reported turns out to be a deep
trough rather than a gap: 296 rejects at 0 and 34 at 1, then **13 clips in the whole 2–4 band**,
then 48 at 5 and 48 at 6. `MAX_INSERTION_RUN = 5` sits at the bottom of that trough, not on a
slope, which is still the strongest thing that can be said for a threshold.

## Bleed and the re-cut

Measured over the corpus population — the 447 clean re-reads, not the whole reject pile.

| | shard 20 | this run |
| --- | --- | --- |
| Clips | 46 repeat-carrying | 447 |
| **Carry detected bleed** | **4 (8.7%)** | **28 (6.3%)** |
| Re-cuts accepted | 4 of 4 | **28 of 28** |
| Trims collapsing to (0, 0) | — | 24 of 28 |

**Shard 20 was representative.** 6.3% against 8.7% on a bucket of 46 is well inside what that
sample can resolve, so nothing about the bleed estimate needs revisiting. Every re-cut passed
its own re-gate — `match_ratio` rose and neither edge trim grew — which is the property that
makes a bad detection cost yield rather than correctness. The trims collapsing to `(0, 0)` on 24
of 28 is the independent confirmation that the cuts landed on the recitation's real edges;
nothing told the aligner where they were.

Prevalence across the whole 812-clip reject pile, which needs no audio and so is reported for
every band:

| band | clips | bleed | share | truncated |
| --- | --- | --- | --- | --- |
| low (< 0.70, repeat) | 86 | 26 | 30% | 2 |
| marginal (0.70–0.75) | 99 | 0 | 0% | 0 |
| clean (>= 0.75) | 262 | 2 | 1% | 6 |
| added shadda | 22 | 1 | 4% | 1 |
| **no repeat (run < 5)** | **343** | **240** | **70%** | 5 |

The shape shard 20 showed holds at scale: **bleed concentrates almost entirely in the rejects
that carry no repeat at all**, 240 of 343. Those are correct recitations of short ayat whose
staged clip contained two or three ayat's worth of audio. They are not ADR-0016 corpus material,
but the same re-cut would recover them as ordinary ADR-0001 training data — a yield question for
the filter, noted here and not pursued, now with 240 clips behind it instead of 19.

The detector's regression scores against the labelled set are unchanged (clip P = R = 1.00
against `trim_baseline` 0.67/0.50 and `reference_baseline` 1.00/0.25).

## The bundle

`tadabur.scenario` selected 447 and staged **445**.

| | |
| --- | --- |
| Records | **445** |
| Seams | **466** (427 clips with one, 15 with two, 3 with three) |
| Assertable words | **8,402** (median 17 per clip, min 4, max 60) |
| Staged audio | 11,014 s (**3.06 h**) |
| Re-cuts applied | 27 |
| `early_start` flagged | 9 |
| Truncated (`uncovered_tail > 0`) | **20** (median 1 phoneme, max 26; 8 at the 5-phoneme bar) |
| Excision pairs kept | **438 of 445** (7 refused, all `ratio_too_low`) |

**Truncated clips are kept with a shortened covered range**, per #68 — there is nothing to cut,
the audio is all correct recitation, and ADR-0016 decision 1 already scopes every assertion to
the covered word range. Only 8 of the 20 are truncations in the sense `bleed_detect` means
(5+ uncovered reference phonemes); the other 12 miss the last phoneme or two of the ayah.

185 clips have a non-zero `uncovered_head`, median **1 phoneme** — the strict coverage rule
costing a whole word for a one-phoneme miss at the head, exactly the cost ADR-0016 records and
names as the first setting to revisit if the scoreboard turns out short of coverage.

The excision differential is healthy: median `match_ratio` **0.768 → 0.896**, and
`max_insertion_run` collapses to **0 on 386 of the 438** pairs. Cutting the repeat out does what
decision 4 assumes it does.

### 0 of 466 seams have a pause at both edges

Corroborating `fd7c37e` at ten times the scale. 208 seams have a VAD pause at the start edge, 18
at the end, **240 at neither, and none at both**. A re-read seam is not a waqf, so a cut placed
there cannot rely on silence — which is why the re-cut treats a pause as a preference and not a
requirement.

### The two clips that did not make it

Both were selected legitimately and then failed once re-cut and re-decoded. The stager now drops
such rows by name rather than shipping them (`unusable_reason`, `staging.json`'s `drops`), since
`verify_bundle` refuses them on the Mac anyway and a silent zero in the scoreboard is worse than
a drop.

| clip | ratio | run | why | |
| --- | --- | --- | --- | --- |
| 89:1 | 0.091 | 5 | `no_words` — no word falls wholly inside the alignment span | no re-cut |
| 56:38 | 0.306 | 6 | `no_seam` — the repeat was **inside the lead-in** | 4.88 s clipped from the head |

56:38 is the interesting one, and it is the re-cut working correctly rather than a loss: the
"repeat" that selected the clip lived in the neighbour-ayah bleed, so clipping the bleed removed
it. The clip was never a re-read *of this ayah*.

Both sat at ratios the dropped floor would have excluded (0.091 and 0.306). That is the floor
decision playing out as `docs/tadabur-ratio-floor.md` predicted — quality assurance moved
downstream to the coverage and seam checks, and here it visibly caught what the floor used to.

## Handoff to the Mac

The bundle is **630 MB**: 337 MB of staged audio (445 WAVs), 291 MB of excised control clips
(438), and an 806 KB manifest. It moves as one directory — every path inside `scenario.jsonl` is
relative to the manifest.

```bash
rsync -a "root@cuda-dev:/root/repos/HifzGuide/tools/tadabur/corpus_run/scenario/" \
         "$HOME/corpora/tadabur-reread/"
cd ~/repos/HifzGuide/tools
python3 -m tadabur.scenario --verify ~/corpora/tadabur-reread
```

**Done, and verified:** `~/corpora/tadabur-reread` on the dev Mac, `445 records, 0 problems`.
`--verify` needs no model, GPU or reference cache — it ran in 0.08 s under the system `python3`.
It is deliberately outside the repo: 630 MB is not a thing to risk committing, and Muraja #128
takes the bundle path as an argument.

The `scenario.jsonl` schema is in `tools/README.md`. No timestamp from Tadabur's
`word_alignments` appears in it.

## The audit this run still owes

ADR-0016 decision 10's full adjudication pass is gone (Muraja `a09d959`): a **vowel-only**
non-Hafs divergence cannot move the alignment cursor, because alignment runs on the normalized
string and normalization deletes short vowels, so it is inert to cycles-to-resync and
falsely-skipped words alike. What replaces it is a ~50-clip random audit — and that audit is not
a screen. It is the test of the inertness argument's one premise, that vowel-only divergence
dominates.

**The sample is staged and the listening is not done.** It is a human step and it is the one
outstanding deliverable of this issue:

```bash
python3 -m tadabur.reread_audit --bundle ~/corpora/tadabur-reread \
    --out ~/corpora/tadabur-reread-audit --size 50 --seed 0
```

50 clips, 41 MB, at `~/corpora/tadabur-reread-audit` with a `worklist.jsonl`. The draw is seeded
and sorted, so it reproduces from the seed and a re-run tops it up rather than redrawing. Two of
the 50 are already judged from the shard-20 pass. Verdicts append to
`tools/tadabur/eval_fixtures/reject_reread_verdicts.jsonl`; the schema now carries a
`divergence` field, and `--summary` reports it.

One thing to state plainly before the listening starts. The standing evidence for vowel-only
dominance is **one instance in 31 adjudicated clips, and its divergence mode was never
recorded** — so `--summary` reports it as `unclassified`, not as vowel-only. The evidence base
for the assumption this audit tests is currently zero classified instances. If the audit turns
up consonantal divergence at any material rate it reopens decision 10, since consonantal
non-Hafs *does* perturb alignment and therefore the cursor.

## Wall clock and disk

| | |
| --- | --- |
| Pipeline wall clock | **46m01s** |
| Stage 1, per shard | **208.7 s** (budget assumed 217 s) |
| Download, per shard | ~37 s, 2.4 GB |
| Peak disk in flight | one shard (~2.4 GB), reclaimed by `--delete-shards` |
| Left on disk in `tools/tadabur/corpus_run/` | **1.3 GB** — 651 MB staged clips, 630 MB bundle, 20 MB re-cut clips, 1.4 MB decodes, 400 KB sinks |
| Free on cuda-dev after the run | **22 GB of 98 GB** (`audit_run`'s ~40 GB untouched) |

The 46 minutes excludes one 6m23s re-run of stage 4: the first pass staged the two unusable rows
above, `--verify` refused the bundle, and `set -e` stopped the driver there. That is the
behaviour to keep — a bundle that fails its own verifier is not a handoff — and it is why the
drop moved into the stager.

Stage 1's 208.7 s/shard came in under the 217 s the budget carried. The downstream tail is
**11m14s once**, not per shard: the issue's original 40-minute estimate folded a per-shard cost
that does not exist, and the bleed re-cut it warned was unmeasured turned out to be the cheapest
stage in the pipeline at 33 seconds.

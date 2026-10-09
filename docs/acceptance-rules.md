# Acceptance rules for PRD #77, fixed before results

**Status:** Owner-confirmed 2026-10-08. Decided in #82 under [ADR-0011](adr/0011-transcription-fidelity-and-tashkeel-abstention.md) §5.
No result from #84 onward may be read against a rule that is not in this document. A rule changes only
by an owner-signed revision recorded below, and **never after the result it governs exists**.

Terms follow `CONTEXT.md`: tashkeel, haraka, sukun, shaddah, empty tashkeel slot, commit rate.
"Site" means one truth site (`tools/tadabur/truth_sites/`, #79): one labelled position in one clip.

## 1. Conventions shared by every rule

### Outcomes per site

| symbol | definition |
|---|---|
| Cᵢ (tashkeel commit) | The model emits **any explicit mark** (a haraka, or sukun once #93 adds the class) at the site's carrier. |
| Aᵢ (tashkeel correct) | The model emits **the human-heard mark** at the carrier. |
| Cᵢ (consonant commit) | The model emits an aligned consonant at the site's carrier position, whatever the letter. |
| Aᵢ (consonant correct) | That consonant is **the human-heard letter**. |
| Fᵢ (flagged, decode level) | The model commits an explicit output that differs from the prescribed (mushaf) mark or letter. An empty slot is not flagged. Muraja plays no part in this definition. |

Unaligned and wrong-carrier outcomes are Cᵢ = 0. Rates are weighted ratios: **commit rate** = ΣwᵢCᵢ / Σwᵢ;
**committed accuracy** = ΣwᵢAᵢ / ΣwᵢCᵢ, each arm over its own committed sites. A comparison of committed
accuracy is the difference of the two arms' ratios, recomputed inside every resample; it is **never**
tested on the intersection of sites both arms committed.

### Populations and sampling frames

| item | rule |
|---|---|
| Unit of observation | One frozen physical site. A site observed by several windows counts once, at the commit block the protocol under test assigns it. A site present in several sources counts once, under its directly adjudicated verdict. |
| Eligibility | Model-independent: decided by the truth record and the realized reference only, never by a model's decode. |
| Truth population | Sites whose human verdict is not `unclear`. The estimand is *audible, adjudicable* sites. Exclusions are reported per stratum with best/worst-case sensitivity (all excluded counted as success / as failure). |
| Sides | Heard = prescribed is **correct recitation**; heard ≠ prescribed is a **real mistake**. The sides are never pooled (ADR-0008). Cells are indexed by the **human-heard** mark or letter. |
| Weighting | Stratum weight = population count / sampled count, from the frozen mining frame. Sites lost to re-location (#83) are reported as exclusions; weights are not inflated to cover them. |
| Source manifests | Every gate names a frozen source/partition manifest. The **headline population is the sampled subpopulation**: the strata #87 mines, with their inclusion probabilities. It extends to all correct recitation only if every outcome stratum is sampled with positive probability. |
| Targeted safeguards | The re-located P3.5 fixtures are reported as separate targeted safeguards and are not pooled into headline rates. Synthetic edits never enter a real-mistake rate. |
| Weak labels | The 1,622 `waqf_boundary:wasl` sites and the 35 long-vowel haraka sites in `waqf_boundary:waqf` assume a competent reciter: **diagnostic only**, in no gate. The 135 sukun-at-pause sites (`waqf_boundary:waqf`, 170 sites = 135 sukun + 35 long-vowel haraka) form their own row and are not the in-scope sukun population. |
| Shaddah | Every shaddah rule is **provisional** until #92 freezes the held / not held / unsure state table. Until then shaddah cells report descriptively and return `cannot_certify`. |

### Intervals and verdicts

| item | rule |
|---|---|
| Interval | Reciter-clustered paired bootstrap, B = 10,000, seed 20261008. Percentiles use linear interpolation (NumPy's default, type 7); the lower bound is the 2.5th percentile and the upper bound the 97.5th of the B replicate statistics. A "bound" is that one-sided 97.5% end of the two-sided 95% interval. |
| Undefined replicates | A replicate whose statistic is undefined (e.g. a zero denominator) takes the **adverse endpoint**: −∞ for a lower-bound test, +∞ for an upper-bound test. Replicates are never dropped. |
| Sparse cells | A cell with fewer than **10 reciters or 20 sites**, or whose replicates are all-identical or all-zero, is sparse. A Wilson score bound (single proportion) or a Tango score bound (paired difference on a common denominator) is used **only** where the cell's observations are independent and equally weighted (one site per reciter, unit weights, fixed denominator). Every other sparse or degenerate case returns `cannot_certify`. |
| Zero denominators | Never pass. |
| Verdicts | Every rule returns **pass**, **fail** or **cannot_certify**. |
| Aggregation | Any required condition fails → **fail**. Otherwise any required condition is `cannot_certify` → **cannot_certify**. Otherwise → **pass**. |
| Required cells | The list of required cells for each gate is frozen with the #84 scorer, before any candidate result. Unsupported target pairs stay on the list, so they produce `cannot_certify` rather than vanishing. Continuing past a `cannot_certify` gate needs a separate owner authorisation, and that continuation is exploratory, never labelled a pass. |

## 2. Training-probe gate (#95): probe vs matched control, decode level

Both arms start from the same grown head (#93), see the same data and steps, and are decoded with the
same protocol: **b = 0, no blank bias, same precision and batch size** (one `DecodeFingerprint`, #80).
The arms differ only in the supervised term. Today's product and Muraja's rules play no part here.

Sukun is judged by **absolute floors**, not against the control. The KL-only control receives no sukun
targets, so its sukun output stays at its initialisation and any relative gain would pass by construction.
The probe-vs-control comparison is kept for the harm guards (haraka, consonants, shaddah, teacher
agreement) and for what the synthetic-edit term adds on real mistakes.

**Sukun floors** (probe arm; population: correct-recitation, directly adjudicated, in-scope sukun sites, i.e.
heard = prescribed = sukun, from the listening session and not pause sites):

| rule | formula | threshold |
|---|---|---|
| Sukun commit rate | ΣwC / Σw | ≥ **70%** (point) |
| Sukun committed accuracy | ΣwA / ΣwC | ≥ **95%** (point) **and** lower bound ≥ **90%** |

Sukun said by mistake (heard sukun where the mushaf prescribes a haraka) is reported separately and never
pooled with correct recitation.

**Harm guards** (probe − control, paired):

| rule | formula | population | threshold |
|---|---|---|---|
| Commit rate, correct side | ΣwC / Σw | correct-recitation sites, per haraka (fatha, damma, kasra), per consonant pair, and shaddah (provisional until #92 freezes its state table) | lower bound ≥ **−2 pts** |
| Committed accuracy, correct side | ΣwA / ΣwC | correct-recitation sites, per haraka, per pair, and shaddah (provisional) | lower bound ≥ **−2 pts** |
| Committed accuracy, mistake side | ΣwA / ΣwC | real-mistake sites, per mark and per pair | lower bound ≥ **−2 pts** |
| Missed mistakes | Σw(1 − F) / Σw | real-mistake sites, pooled and per mark/pair (abstentions count as not flagged) | upper bound ≤ **+5 pts**; no count floor |
| Silent corrections | Σw·[committed = prescribed] / Σw | real-mistake sites | upper bound ≤ **+5 pts** |
| Teacher agreement | 1 − Σ edit distance / Σ teacher tokens | frozen `decode_evalset` dev manifest, b = 0, no bias, sukun projected out of the student string, any gemination encoding expanded to the legacy doubled consonant. Clip records resampled by reciter (the one exception to the one-site unit). | lower bound ≥ **−0.5 pts** |

**Overall:** the aggregation rule in §1 over the sukun floors and every required guard cell. A reduction in
empty haraka slots is **not** claimed from the probe: no training source supplies haraka targets where the
teacher is silent.

## 3. Ship criterion (#97): the whole system vs today's system

**Today's system:** `h448`, b = 0, no bias, and the frozen Muraja configuration of §9 (including the
dropped-haraka exemption on و ا ء ي and `suppressHarakaDrop`). **Candidate system:** the candidate model,
its commit block, its selected bias, and Muraja's "empty slot = not graded" rule.

| rule | definition | threshold |
|---|---|---|
| **Headline: false flags** | false flags / correct-recitation sites, in the headline population of §1. A false flag is a correct-recitation site the system grades as wrong under the #84 scorer's site semantics. | relative change ≤ **−50%** (point) **and** upper bound ≤ **−25%** |
| **Coverage** | graded correct-recitation sites / correct-recitation sites | falls by at most **10 pts** (lower bound of the change ≥ −10 pts) |
| **Missed mistakes** | as §2, candidate system vs today's system | upper bound of the increase ≤ **+5 pts** |

- The relative change uses today's rate as denominator; if today's weighted false-flag count is zero the headline is `cannot_certify`.
- The report gives **policy-only** (h448 + new Muraja rule), **b = 1-only** and **bias-only** ablations next to the full system, each with the same metrics, so no lever's gain is credited to another.
- The headline counts **sites**. A claim about **word flags** as the reciter sees them needs a versioned Muraja replay; the offline scorer cannot stand in for it (ADR-0008).
- **Final truth panel.** #97 is not authorised until the sealed panel (#89) carries blind, human-adjudicated correct-recitation and real-mistake sites, with population weights and support counts. Its listening is part of the post-#84 sizing (§8), and its results are unavailable for any selection. A panel with only a teacher-decode cache certifies cloning agreement, not this criterion. It is scored once, with the owner's authorisation, under exactly these rules.

## 4. Retiring a Muraja allowance

Decided per allowance, and for soft pairs **per pair**, never pooled across pairs. The population of each
rate is the sites **that allowance affects**, not all correct sites.

| condition (allowance switched off, everything else the candidate system) | threshold |
|---|---|
| false flags / affected correct-recitation sites | upper bound ≤ **5%** |
| missed mistakes / affected real-mistake sites | upper bound ≤ **50%** |

- A flawless run needs about **73 correct-recitation sites** in a cell for the Wilson upper bound to reach 5% (3.8416 / (73 + 3.8416) = 0.050). No soft-pair cell in the existing fixtures has that many (largest: ذ↔ز, 33), so none can retire without new labels.
- `ح↔ه` has no real-mistake evidence and **cannot receive a retirement verdict**.
- Under "empty = not graded", the per-letter dropped-haraka exemption and `suppressHarakaDrop` may change nothing. Such a switch is removed by a **behavioural-equivalence check over its full applicable state table** (§9), not by these statistics and not by identical outcomes on a finite panel. Switching an allowance off never re-grades empty slots (ADR-0011 §2).

## 5. Blank-bias probe (#91)

| item | rule |
|---|---|
| Comparator | The same frozen `h448` checkpoint and decode protocol at δ versus δ = 0. A selected δ is bound to that checkpoint and protocol; it is not validated for any later model. |
| Transform | z′ₕ = zₕ + δ on the three haraka logits only (fatha, damma, kasra), before argmax and CTC collapse, temperature 1 |
| Grid | δ ∈ {0, 0.125, 0.25, 0.5, 1, 1.5, 2, 3} logits; one δ shared by the three harakat |
| Split | 50/50 by hash of the canonical reciter id, salt `issue-91-blank-bias-2026-10` (algorithm and serialization in §9) |
| Spurious haraka | Σw·[any haraka emitted] / Σw over directly adjudicated heard-sukun sites; the in-scope and pause rows are reported separately. |
| Guards | Each, δ vs δ = 0: committed accuracy on correct recitation per haraka, lower bound ≥ −2 pts; committed accuracy on real mistakes per mark and per pair, lower bound ≥ −2 pts (filling an empty slot with a third mark can cut missed mistakes while making the transcription worse); missed mistakes and silent corrections, upper bound ≤ +5 pts; spurious haraka per supported row, upper bound of the increase ≤ **+2 pts**. The same guards apply to tune-half selection and score-half certification. |
| Admissibility | An offset is admissible only if **every** applicable guard passes. `cannot_certify` is not a pass. |
| Selection | On the **tune half only**: the admissible δ maximising weighted correctly committed haraka on correct recitation (ΣwA / Σw). Ties go to the smaller δ. `no_admissible_offset` is a valid outcome. |
| Scoring | The selected δ is locked and recorded **before** the score half is decoded. The score half is then evaluated once, and every guard is reapplied there; if any fails, the offset is rejected. The full score-half sweep is published afterwards as exploratory only. |
| Safety report | Spurious haraka, consonant changes and shaddah changes, per δ |

## 6. Exposure registry

One committed registry lists every canonical reciter id, source recording (Tadabur `audio_filename`),
span and checksum, with each use it has had: training, KL-only control, synthetic-edit source or donor,
mining, bias tune, bias score, shaddah-probe tuning, final panel. The partition is made **before** mining
and training. Score-half reciters are excluded from every tuning and training use, edit donors
included. The sealed panel (#89) is reciter-disjoint from every other use; "a new salt" over previously
used reciters does not make them fresh. Halves that fail this are labelled development data.

**Owner amendment (2026-10-08): the paired-claim panel.** The sealed panel (#89) is reciter-disjoint from
everything this PRD tunes, trains on or selects with: listening sites, both bias halves, probe and control
training data, synthetic-edit sources and donors, shaddah-probe tuning, truth sites, the mining pool and
`decode_evalset`. Toward `h448`'s original training and initialisation it is held out by recording only.
Those exposures are shared by the baseline (`h448`) and by every candidate warm-started from it, so the panel
certifies the **paired** ship criterion of §3, not absolute accuracy on unseen reciters. The registry marks
them as shared-baseline uses. Where such a use's recordings are unknown (`h448_init`'s calibration and
`h448`'s validation windows, from the lost `clips_v2` corpus of shards 0-19), the panel may hold recordings
in its shards; recording-level disjointness is asserted against every other use.

## 7. Target pairs

The six soft pairs plus **ذ↔ظ** (owner decision). Every pair is reported with **directional** strata
(e.g. ذ→ظ and ظ→ذ separately) on both recitation sides. A pair without support on both sides is
reported as "in scope, insufficient evidence" and gets no discrimination or retirement verdict.

## 8. The listening session (#61) and its sizing

- The ~40 reject-pile consonant sites are **included**.
- The 23 nominal real-mistake fixture clips (11 soft-pair, 12 shaddah) are **re-adjudicated at site level** inside the session (letter heard; held / not held / unclear). Several carry notes such as "Not hafs", "Unclear reading" or "Segmentation issue"; a clip-level reject is never translated automatically into "the other letter was said".
- #87 adds a **prescribed-sukun (mid-word) stratum**, so the in-scope sukun floors of §2 have a population.
- **Session size is set after #84**, by a power simulation that uses only the truth-labelled baselines of `h448` and the base teacher, never a candidate. The owner's target: ≥ **80%** probability of passing at a true **60%** reduction in false flags.

The simulation is pre-registered as follows:

| item | rule |
|---|---|
| Candidate-outcome generator | Per correct-recitation site: today's outcome drawn from the #84 baseline rates for its stratum; a flagged site is recovered with probability r, an unflagged site gains a new flag with probability q. r is set so the marginal reduction is 60% at each q. |
| Sensitivity range | q ∈ {0, 1, 2, 4} pts of correct-recitation sites; the size chosen must meet the target at every q in the range, or the shortfall is reported per q. |
| Allocation | Strata and reciters allocated as in the frozen mining frame; sites per reciter drawn from its empirical distribution. |
| Clustering | Reciter random effect on the logit scale, intra-reciter correlation ∈ {0, 0.05, 0.15}. |
| Baseline uncertainty | Baseline stratum rates redrawn per repetition from their reciter-clustered bootstrap. |
| Repetitions | 2,500 per scenario, seed 20261008. Monte Carlo rule: the power estimate's standard error √(p(1 − p)/2,500) ≤ 1 pt. |
| Scope | Sizes **every** mandatory gate: the §3 headline and coverage, the §2 sukun floors and required guard cells, both bias halves (§5), and the §3 final truth panel. |
| Flawless floors | For reference, a fixed-denominator paired guard at 2 pts needs n ≥ **189** even with zero discordances (3.8416 / (189 + 3.8416) = 1.99%); the sukun accuracy floor needs ≥ **35** flawless commits for a Wilson lower bound ≥ 90% (35 / 38.84 = 90.1%), i.e. 50 sites at a 70% commit rate. |

The owner then chooses the budget. Every gate the chosen budget cannot support is published as
**exploratory** before any candidate result exists; the #97 headline becomes a pilot if it is one of them.

## 9. Operational definitions frozen in the #84 scorer

These choices change answers, so they are fixed **as code and tests in #84, before any candidate result
exists**. #84 scores only the base teacher and `h448`, so fixing them there is still pre-registration under
ADR-0011 §5. Each is recorded in the scorer's module docs and pinned by a test:

| item | what #84 fixes |
|---|---|
| Flagged at decode level | Fᵢ as in §1, including how a site with no aligned output is treated |
| Substitutions, insertions, unaligned | How a substituted mark, more than one mark at a carrier, an inserted mark and an unaligned carrier map to Cᵢ, Aᵢ and Fᵢ |
| Consonant commitment | The alignment rule that decides which emitted consonant sits at a pair site's carrier |
| Muraja configuration | The Muraja revision, mode, tashkeel toggle and every allowance setting that defines today's system (§3), with each allowance's ON/OFF state table |
| Allowance-affected populations | Which sites each allowance (§4) affects, per pair |
| Streaming | Startup, tail flush, window phase, per-window normalization and pairing across protocols, so a site available under one protocol never vanishes from a paired comparison |
| Teacher agreement | 1 − Σ edit distance / Σ teacher tokens over clip records, resampled by reciter (§2) |
| Reciter split | The hash algorithm, canonical serialization of the reciter id with the salt, and the cutoff that assigns a reciter to a half |
| Required cells | The frozen list of required cells for §2, §3 and §5 (§1) |

## Revision history

| date | change |
|---|---|
| 2026-10-08 06:30Z | Owner locked the #82 sheet: −50% false-flag headline, per-mark guards, sukun floors, allowance limits 5% / 50%, bias grid, ذ↔ظ in scope, consonant sites included. |
| 2026-10-08 | GPT-6 Astra review returned REVISE (gates conflated model and product; interval and power gaps; sparse-cell and weak-label issues). Owner chose to split the probe gate (#95) from the ship criterion (#97) and to size the listening session after the #84 baseline. This document records the revised rules. |
| 2026-10-08 | Owner confirmed: sukun judged by absolute floors on directly adjudicated in-scope sukun sites (commit rate ≥ 70%, committed accuracy ≥ 95%, its lower bound ≥ 90%) in place of a relative gain over the control; spurious-haraka guard ≤ +2 pts in bias selection; sparse-cell threshold of < 10 reciters or < 20 sites; power target of ≥ 80% at a true 60% reduction for sizing the listening session. |
| 2026-10-08 | GPT-6 Astra round 2 returned CHANGES_REQUESTED. Revised without changing any owner number: per-site outcome definitions (C, A, F); an aggregation truth table and frozen required cells; statistic-specific sparse-cell rules and adverse endpoints; per-mark commit-rate guards restored; headline restricted to the sampled subpopulation; a human-truth final panel required before #97; bias comparator, admissibility and held-out guards; a pre-registered power simulation covering every gate, with a prescribed-sukun stratum; operational definitions frozen in #84. |
| 2026-10-08 | GPT-6 Astra round 3: shaddah added to the correct-side commit-rate guard (provisional); the mistake-side committed-accuracy guard added to bias admissibility. Truth-site counts aligned with `tools/tadabur/truth_sites/README.md` on main. |
| 2026-10-08 | Owner amendment to §6 (#89 review): the sealed panel is reciter-disjoint from every use this PRD makes and recording-held-out toward `h448`'s original training and initialisation, which baseline and candidates share; it certifies the paired §3 criterion. No number changed. |

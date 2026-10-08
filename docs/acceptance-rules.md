# Acceptance rules for PRD #77, fixed before results

**Status:** Draft for owner sign-off. Decided in #82 under [ADR-0011](adr/0011-transcription-fidelity-and-tashkeel-abstention.md) §5.
No result from #84 onward may be read against a rule that is not in this document. A rule changes only
by an owner-signed revision recorded below, and **never after the result it governs exists**.

Terms follow `CONTEXT.md`: tashkeel, haraka, sukun, shaddah, empty tashkeel slot, commit rate.
"Site" means one truth site (`tools/tadabur/truth_sites/`, #79): one labelled position in one clip.

## 1. Conventions shared by every rule

| item | rule |
|---|---|
| Unit of observation | One frozen physical site. A site observed by several windows counts once, at the commit block the protocol under test assigns it. |
| Eligibility | Model-independent: decided by the truth record and the realized reference only, never by a model's decode. |
| Truth population | Sites whose human verdict is not `unclear`. The estimand is therefore *audible, adjudicable* sites. Exclusions are reported per stratum with best/worst-case sensitivity (all excluded counted as success / as failure). |
| Weighting | Stratum weight = population count / sampled count, frozen at mining. Rates are weighted ratios Σwᵢ·numᵢ / Σwᵢ·denᵢ, recomputed inside every resample. |
| Pairing | Every comparison is paired on the same sites (both arms decode the same audio). |
| Interval | Reciter-clustered paired bootstrap, B = 10,000, seed 20261008, percentile. Two-sided 95%; a "bound" in this document is the one-sided 97.5% end of that interval. |
| Sparse cells | A cell with fewer than 10 reciters or 20 sites, or whose resamples are all-identical / all-zero, uses a Wilson score bound (paired: the ADR-0006 score interval on discordant pairs) instead, clustered by design-effect where reciters repeat. |
| Zero denominators | Never pass. A resample with a zero denominator is counted as a failure of that resample, not dropped. |
| Verdicts | Every rule returns **pass**, **fail** or **cannot_certify**. Insufficient support is `cannot_certify`, never pass. |
| Commit / empty | *Commit* = the model emitted a mark at the site's carrier. *Empty* = no mark (from #93 on, empty means only "unsure"; before it, an empty slot is also how sukun is emitted). Unaligned or wrong-carrier outcomes count as not committed. |
| Correct recitation / real mistake | Side is set by the human verdict versus the mushaf: heard = prescribed is correct recitation; heard ≠ prescribed is a real mistake. The two sides are never pooled (ADR-0008). |
| Indexing | Per-mark and per-pair cells are indexed by the **human-heard** mark or letter. |
| Shaddah | Every shaddah rule is **provisional** until #92 freezes the held / not held / unsure state table. Until then shaddah cells report descriptively and return `cannot_certify`. |
| Weak labels | The 1,625 `waqf_boundary:wasl` sites assume a competent reciter. They are **diagnostic only** and enter no gate. Sukun-at-pause sites (`waqf_boundary:waqf`, 124 sukun) are reported as their own row and are not the in-scope sukun population below. |

## 2. Training-probe gate (#95): probe vs matched control, decode level

Both arms start from the same grown head (#93), see the same data and steps, and are decoded with the
same protocol: **b = 0, no blank bias, same precision and batch size** (one `DecodeFingerprint`, #80).
The arms differ only in the supervised term. Today's product and Muraja's rules play no part here.

| rule | numerator / denominator | population | direction and threshold |
|---|---|---|---|
| **Sukun gain** | correctly committed sukun / adjudicated sukun sites | directly adjudicated in-scope sukun sites (listening session; not pause sites) | probe − control ≥ **+5 pts** (point) **and** lower bound > 0 |
| **Discrimination guard, correct side** | correct committed marks / committed marks | correct-recitation sites, per mark (fatha, damma, kasra, sukun, shaddah) and per pair | lower bound of probe − control ≥ **−2 pts**, else fail; unsupported → `cannot_certify` |
| **Discrimination guard, mistake side** | committed marks equal to the heard mark / committed marks | real-mistake sites, per mark and per pair | as above |
| **Missed mistakes** | mistakes not flagged / real-mistake sites (abstentions count as not flagged) | real-mistake sites, pooled and per mark/pair | upper bound of probe − control ≤ **+5 pts**; no count floor |
| **Silent corrections** | committed mushaf mark / real-mistake sites | real-mistake sites | upper bound of probe − control ≤ **+5 pts** |
| **Teacher agreement** | pooled character accuracy vs the cached base decode | frozen `decode_evalset` dev manifest, b = 0, no bias, sukun projected out of the student string (any gemination encoding expanded to the legacy doubled consonant) | lower bound of probe − control ≥ **−0.5 pts** |

**Overall:** pass only if the sukun gain passes and no guard fails. Guards that return `cannot_certify` are
listed by name in the verdict; the report states which claims the probe therefore cannot support. A
reduction in empty haraka slots is **not** claimed from the probe: no training source supplies haraka
targets where the teacher is silent.

## 3. Ship criterion (#97): the whole system vs today's system

**Today's system:** `h448`, b = 0, no bias, Muraja's rules as shipped (including the dropped-haraka
exemption on و ا ء ي and `suppressHarakaDrop`). **Candidate system:** the candidate model, its commit
block, its selected bias, and Muraja's "empty slot = not graded" rule.

| rule | definition | threshold |
|---|---|---|
| **Headline: false flags** | false flags / correct-recitation sites. A false flag is a correct-recitation site the system grades as wrong under the site semantics of the #84 scorer. | relative change ≤ **−50%** (point) **and** upper bound ≤ **−25%** |
| **Coverage** | graded correct-recitation sites / correct-recitation sites | falls by at most **10 pts** (lower bound of the change ≥ −10 pts) |
| **Missed mistakes** | as §2, candidate system vs today's system | upper bound of the increase ≤ **+5 pts** |

- The relative change uses today's rate as denominator; if today's weighted false-flag count is zero the headline is `cannot_certify`.
- The report gives **policy-only** (h448 + new Muraja rule), **b = 1-only** and **bias-only** ablations next to the full system, each with the same metrics, so no lever's gain is credited to another.
- The headline counts **sites**. A claim about **word flags** as the reciter sees them needs a versioned Muraja replay; the offline scorer cannot stand in for it (ADR-0008).
- The sealed panel (#89) is scored once, with the owner's authorisation, under exactly these rules.

## 4. Retiring a Muraja allowance

Decided per allowance, and for soft pairs **per pair**, never pooled across pairs.

| condition (allowance switched off, everything else the candidate system) | threshold |
|---|---|
| false flags / correct-recitation sites | upper bound ≤ **5%** |
| missed mistakes / real-mistake sites | upper bound ≤ **50%** |

- A flawless run needs about **73 correct-recitation sites** in a cell for the Wilson upper bound to reach 5% (3.84 / (73 + 3.84) = 0.050). No soft-pair cell in the existing fixtures has that many (largest: ذ↔ز, 33), so none can retire without new labels.
- `ح↔ه` has no real-mistake evidence and **cannot receive a retirement verdict**.
- Under "empty = not graded", the per-letter dropped-haraka exemption and `suppressHarakaDrop` may change nothing. Such a switch is removed by a **behavioural-equivalence check** (same grading on every site with and without it), not by these statistics. Switching an allowance off never re-grades empty slots (ADR-0011 §2).

## 5. Blank-bias probe (#91)

| item | rule |
|---|---|
| Transform | z′ₕ = zₕ + δ on the three haraka logits only (fatha, damma, kasra), before argmax and CTC collapse, temperature 1 |
| Grid | δ ∈ {0, 0.125, 0.25, 0.5, 1, 1.5, 2, 3} logits; one δ shared by the three harakat |
| Split | 50/50 by hash of the canonical reciter id, salt `issue-91-blank-bias-2026-10` |
| Selection | on the **tune half only**: maximise weighted correctly committed haraka on correct recitation, subject to §2's discrimination guards and a spurious-haraka guard on adjudicated sukun sites (upper bound of the increase ≤ +2 pts). Ties go to the smaller δ. `no_admissible_offset` is a valid outcome. |
| Scoring | the selected δ is locked and recorded **before** the score half is decoded; the score half is then evaluated once. The full score-half sweep is published afterwards as exploratory only. |
| Safety report | spurious haraka on sukun sites, consonant changes and shaddah changes, per δ |

## 6. Exposure registry

One committed registry lists every canonical reciter id, source recording (Tadabur `audio_filename`),
span and checksum, with each use it has had: training, KL-only control, synthetic-edit source or donor,
mining, bias tune, bias score, shaddah-probe tuning, final panel. The partition is made **before** mining
and training. Score-half reciters are excluded from every tuning and training use, edit donors
included. The sealed panel (#89) is reciter-disjoint from every other use; "a new salt" over previously
used reciters does not make them fresh. Halves that fail this are labelled development data.

## 7. Target pairs

The six soft pairs plus **ذ↔ظ** (owner decision). Every pair is reported with **directional** strata
(e.g. ذ→ظ and ظ→ذ separately) on both recitation sides. A pair without support on both sides is
reported as "in scope, insufficient evidence" and gets no discrimination or retirement verdict.

## 8. The listening session (#61)

- The ~40 reject-pile consonant sites are **included**.
- The 23 nominal real-mistake fixture clips (11 soft-pair, 12 shaddah) are **re-adjudicated at site level** inside the session (letter heard; held / not held / unclear). Several carry notes such as "Not hafs", "Unclear reading" or "Segmentation issue"; a clip-level reject is never translated automatically into "the other letter was said".
- **Session size is set after #84.** Using only the truth-labelled baseline of `h448` and the base teacher (never a candidate), a paired, weighted, reciter-clustered power simulation estimates the sites needed for ≥ **80%** probability of satisfying both headline conditions in §3 at a true **60%** reduction. The owner then chooses the budget. If the chosen budget falls short, the #97 headline is labelled a pilot.

## Needs owner confirmation

These were added by the revision and have not been seen by the owner:

1. **Sukun gain ≥ +5 pts with lower bound > 0** (§2). Note: if only the probe arm receives sukun targets, the control's sukun output stays at its initialisation and the gain is near-certain. The informative evidence is then the guards. The alternative is to give both arms the sukun targets and let the arms differ only in the synthetic-edit term.
2. **Spurious-haraka guard ≤ +2 pts** in bias selection (§5).
3. **Sparse-cell threshold** of 10 reciters or 20 sites (§1).
4. **Power target** of 80% at a true 60% reduction (§8).

## Revision history

| date | change |
|---|---|
| 2026-10-08 06:30Z | Owner locked the #82 sheet: −50% false-flag headline, per-mark guards, sukun floors, allowance limits 5% / 50%, bias grid, ذ↔ظ in scope, consonant sites included. |
| 2026-10-08 | GPT-6 Astra review returned REVISE (gates conflated model and product; interval and power gaps; sparse-cell and weak-label issues). Owner chose to split the probe gate (#95) from the ship criterion (#97) and to size the listening session after the #84 baseline. This document records the revised rules. |

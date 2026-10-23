# Consonant round: pre-registered bars (#77)

**Status:** frozen 2026-10-22, owner-approved, before any training of this round. Nothing below is
changed after training starts. A changed bar needs a new, dated pre-registration, and never a re-run.

## What is compared

- **Candidate (the round):** C95 continued with KL to the clean base teacher (`obadx/muaalem-model-v3_2`
  @ `01a1ef9`).
  - It keeps augmented input (`aug94-mixture-v2` or a successor recorded before launch). It may add
    unseen distillation audio (see "Training discipline").
  - It has no supervised term on human labels.
  - Same preset `h448`, same 43-class output, within the `h448` byte and ANE budget.
- **Baseline: C95**, as identified above. `h448` and the base teacher are shown beside it, for
  context only.
- **Control (optional):** the same run without the round's new ingredient. It runs only if the GPU
  budget easily covers it, and it is never part of the verdict.
- **Weights:** live or EMA, chosen on the Tarteel v1 look half by #95's rule. That is fewer look-half
  false flags per 1,000 graded words at the shipped default; a tie goes to higher word agreement with
  the teacher, then to live. The choice is committed before any confirm-half decode.

## Grading and statistics

- **Grading:** Muraja's engine at `99c326f`, through the #95 harness
  (`tools/muraja_harness`, binary SHA-256
  `cfcb99ba583fd2324f7ede0e5ab896379c54fa52783460d583e44d103eb68d8b`, protocol 3), with `quran.db`
  SHA-256 `02fe3dd80db1878b7027dd02dba86b42177ead660036362c69791e85994689ad`. Results are read at the
  **shipped default** unless stated.
- **The pin holds even if Muraja#263 or #264 lands.** Results under a newer Muraja may be reported,
  never gated on.
- **Units:**
  - a false flag is a word every compared model places, graded and shown as not correct, counted per
    1,000 graded words;
  - agreement is in percentage points.
- **Intervals:**
  - clustered percentile bootstrap, B = 10,000, seed 20261008, one draw shared by every statistic on a
    set, so every difference is **paired**;
  - clusters as #128: Tarteel v1 by surah, qdat and qdat_bench by reciter, QUD by mushaf and surah,
    Tadabur by reciter.
  - "95%" means the two-sided 95% interval. "99.5%" is used only for the veto.
- **Causes:** from `training.word_flags`.
  - **Consonant-caused:** a flag carrying `other_consonant`, `target_pair` or `soft_pair` through a
    substitution other than `ا→∅`. The madd alif is Muraja#263.
  - **Tashkeel-caused:** a flag carrying `tashkeel`.

## Frozen sets

| set | definition | used for |
|---|---|---|
| Tarteel v1 look half | `tadabur.tarteel_split`, salt `tarteel-v1-split-2026-10` (`nonpro_eval/tarteel_v1/split.json`); 4,530 evaluable clips in #95 | live/EMA choice; the top-10 veto list; C95's cause split; every diagnostic |
| Tarteel v1 confirm half | same split; 4,408 evaluable clips in #95 | bars 1–4, guards G4–G5; read **once** |
| qdat | as #95; 1,505 items | bars 4–5 |
| QUD, other half | QUD pilot segments **not** kept by #95's salt `aug95-qud-half-2026-10` (`aug_readout/qud_selection.json`): 10,261 decoded segments, never read for any candidate | G1 |
| Tadabur `decode_evalset` dev | as #95 (no copy in `h448`'s training or this round's); 671 clips | G2 |
| qdat_bench | stream arm, as #95; 159 clips | G3 |
| Ship A held-out verified sites | `truth_sites/ship_a_verified.jsonl`, held-out reciters of #126's split. Real mistakes only (heard = the partner letter) at ط↔ق and س↔ش; Unclear excluded | G6 |
| RetaSy `in_correct` | as #95 | descriptive only (#171) |

Every set reports its coverage, as #95 §3 did: decoded, evaluated, excluded and why. The waqf/leen
phonetizer gap applies to QUD.

## The bars (all must pass)

**Main: Tarteel v1 confirm half, shipped default**

| # | condition | bar |
|---|---|---|
| 1 | all-cause false flags per 1,000 graded words, candidate − C95 | paired 95% upper bound **< 0** |
| 2 | the same difference as a point reduction | **≥ 4.7** per 1,000 (10% of C95's 47.41) |
| 3 | consonant-caused false flags per 1,000, candidate − C95 | paired 95% upper bound **< 0** |

**Mechanism**

| # | condition | bar |
|---|---|---|
| 4a | word agreement with the teacher, Tarteel confirm and qdat (each), candidate − C95 | paired 95% lower bound **> 0** |
| 4b | the same, only on reference words where the teacher's word **departs** from the reference word (projected to the teacher's vocabulary, as the readout does), Tarteel confirm and qdat (each) | paired 95% lower bound **≥ −0.5 pt** |

Bar 4b exists so a model cannot pass 4a by writing what the text says. On the look half, the
teacher's word equals the reference on 43% of words. The teacher-equals-reference subset is reported
beside it.

**Same direction**

| # | condition | bar |
|---|---|---|
| 5 | qdat all-cause false flags per 1,000 | candidate point estimate **<** C95's |

**Guards**

| # | condition | bar |
|---|---|---|
| G1 | QUD other half, all-cause false flags per 1,000, candidate − C95 | paired 95% upper bound **≤ +1.0** |
| G2 | Tadabur dev, decode agreement with the teacher at `stream_b0`, candidate − C95 | paired 95% lower bound **≥ −1.0 pt** |
| G3 | qdat_bench stream arm: consonant, haraka and shaddah agreement with the experts (each), candidate − C95 | paired 95% lower bound **≥ −2.0 pt** |
| G4 | Tarteel confirm, tashkeel-caused false flags per 1,000, candidate − C95 | paired 95% upper bound **≤ +1.0** |
| G5 | each of the 10 consonant substitutions behind the most C95 false-flagged words on the **look** half (list frozen before training), flagged words per 1,000 on the confirm half, candidate − C95 | no substitution's paired **99.5%** lower bound **> 0** |
| G6 | Ship A held-out verified real mistakes at ط↔ق and س↔ش: words the app flags (checks, shipped default) | candidate **≥ C95 − 2** sites (paired count) |

- **G4's companion:** the QUD-other-half tashkeel difference is reported beside it.
- **G5 is a veto, not a detector.** With tens to a few hundred flags per substitution per half, it
  trips only on a large regression. A pass does not mean "no consonant got worse".
- **G6 is small (about 10 sites)** and catches only a gross loss. Bar 4b is the catch-side bar with
  power. Real-mistake sites at س↔ص, ث and ض↔ظ (noisier verdicts) are reported only.

## Reported, not gating

- every substitution's change on the confirm half;
- false flags by cause;
- `h448` and teacher columns;
- candidate − control, if a control ran;
- the catch side as #95 (RetaSy `in_correct`; qdat_bench departure words);
- empty-slot (abstention) rates;
- coverage;
- the #95 QUD half;
- results under any newer Muraja.

## Decision rule

- **Pass:** bars 1, 2, 3, 4a, 4b and 5 and guards G1–G6 all hold.
  - The candidate becomes the release candidate; publishing is the owner's call.
  - It becomes Ship B's starting checkpoint.
- **Fail:** any bar or guard fails.
  - C95 stays the release candidate and Ship B's start.
  - The architect interprets the result and proposes next steps to the owner.
- **Cannot certify:** an input is missing, a cell is empty, or an interval is undefined. It is resolved
  only by a new dated pre-registration, **never by a re-run**.
- **Reproduction check (not a bar):** before reading any candidate, the readout re-grades C95 and must
  reproduce #95's committed confirm numbers exactly (1,971 flagged / 41,572 graded at the shipped
  default; qdat 670 / 11,502). A mismatch stops the readout as a tooling fault.

## Data discipline

1. **No reading C95 on the confirm half.** C95's confirm-half flags (per word, per cause, per
   substitution) are not read before the verdict. The scripts that compute the cause split and the
   top-10 list take `--half look` and refuse `confirm` outside the verdict phase. #95's published
   aggregates may be cited.
2. **The shared-text rule.** 55.9% of confirm-half clips recite an ayah that also appears in the
   look half.
   - Look-half readings may fix what is **measured** (the G5 list, the cause split) and which
     **substitution classes** a recipe targets.
   - They may **never select or weight training audio by ayah or word**.
   - The run note lists every look-half reading that shaped the recipe.
3. **One read.** The confirm half and the guard sets are read once, for the final candidate.
4. **Never train on:** Tarteel v1, RetaSy, qdat, qdat_bench, QUD, `decode_evalset`, the sealed panel
   (#89), Ship A held-out sites, any truth site. Any new training corpus is fingerprinted against
   every one of these before launch, and every use is registered in `tools/tadabur/exposure/`.

# Follow-along audit: the pre-registered design for S5's draw and decision (#310)

**Status:** dated 2026-11-24, frozen the day it lands, and **before any draw or listening** of
the follow-along audit. Once this document is committed, the population, the strata and their
sizes, the attribution rules, the decision rule, the estimates and the pilot's resize rule do not
change; a change is a new dated amendment, never a redraw.

Issue: [#310](https://github.com/sysofwan/HifzGuide/issues/310). Parent: #305 (the follow-along
audit). This document replaces #261's unfiled dataset-goal design amendment; it cites the
committed slices it is built from by full commit, and it is doc-only — no draw and no sampling
happens here.

Committed inputs this design is computed from:

| slice | commit |
|---|---|
| S0 — the re-pinned Muraja harness, strict and per-word traces (gate P1) | `c4bdea30bbaee4bafc81b2489a2a558529b4beb4` |
| S1 — the tracer: 20 tlog clips through the App decoder and the Oracle (gate P1b) | `c7c6cd072b7438794f2de582ac685cc606bf8923` |
| S3 — the App arm on 5,742 clips, pools, P2 and P3 | `9f3f6da7e40da458ef27a38876a31eed49d1cc7c` |
| S2 — the oracle and planted-mistake sweeps, the S1 oracle fill | `314dd8653735a22f0f2d11d32e469b116715ed72` |

Owner approval: pending

Principal cross-check: pending

## 1. Population

The population is the **placeable phone audio** of confirm.1 ∪ confirm.2: every claim-gated
gating clip whose ayah the Teacher could place. Concretely:

- **5,762** phase A gating clips carry confirm.1/confirm.2 `gating` rows; **minus** the 20 clips
  #248 excludes — 7 of the **8 unreadable mushaf ayat**, `55:17`, `90:8`, `90:10` and `106:1`–`106:4`,
  whose reference phonemes the phonetizer cannot produce (`tadabur/consonant2_readout/report.json`
  `coverage.graded.excluded_clips`; `90:9` is unreadable too but has no gating clips) — leaves
  **5,742 clips**, not 5,762. The count is
  `app_arm/summary.json`'s `population.clips`, and S3 graded all of them (no failed clip).
- A clip's **gating** row is one whose `stratum` is `gating`. Its words are the Teacher's placed
  Mushaf words: **gating = the Teacher placed the ayah**, so a word outside the placed ayah is not
  in this population and a clip the Teacher could not place is not in it either.
- Words are Muraja's phoneme words, keyed `(item, word)` on the App arm's committed table;
  `app_arm/word_grades.jsonl.gz` holds **57,935** of them.

**Disclosures (A5).**

- **Round 3 warm-started from round 2's EMA**, which was selected on these tranches. The App arm
  therefore grades weights whose predecessor was chosen on part of its own evaluation population.
- **Pre-registration item 7 was already relaxed** by round 2's 150-word catch-loss audit and by
  #261 phase C; confirm.1/confirm.2 were mined under [`docs/tlog-mining-amendment.md`](tlog-mining-amendment.md)
  § 1 before this audit grades them.
- The App arm's own rates (106.1 strict flags per 1,000 graded words) are **not** this audit's
  false-flag estimate: that needs the listening this document pre-registers.
- **5,742 clips, not 5,762**: the 20-clip #248 exclusion above.

**Holdout note.** [`docs/tlog-followalong-amendment.md`](tlog-followalong-amendment.md), dated
2026-11-24, records this reuse under the mining amendment § 1 and cites the frozen holdout
document's SHA-256 `204fe7f9…2086`. It adds no new claim and does not edit the frozen
[`docs/tlog-holdout-preregistration.md`](tlog-holdout-preregistration.md), which stays
byte-identical.

**S3/S2 caveats disclosed beside every number.** `app_arm/payload_probe.json` ran
`41_29_undefined_ItjrgCt.flac#0` twenty times: two distinct CoreML payloads at **diacritic-level**
variation with an identical word table, so the decoder is not bit-stable and S7 diagnoses that. A
clip the CoreML driver exited 0 on with no output was retried (the **E5RT** transient, three
attempts) rather than recorded as failed; S3's record is 0 failed clips. S2's clean mushaf sweep
tallies **8 unreadable** ayat (`55:17`, `90:8`–`90:10`, `106:1`–`106:4`, as
`oracle_sweep/run_note.md` records), never silently dropped.

## 2. Arms and identities

Four arms, each pinned by checksum or commit:

| arm | identity |
|---|---|
| **App** | the app's own CoreML decoder at assets release **v1.0.7**, `models.zip` sha256 `f8dc3311fe4506c505b22c4a205d30fdcf8c3b7a45b9cd1ae9008d1f7b12c439`; Muraja `028bc513eeb233caa7240a5c37c423da59f3d9d7` (`028bc51`); 40 ms per inference; harness binary sha256 `6913c94384950653c0fa7889eb320b63d16c22424244d38d12d57b6af0aa7bef`; driver `d58dbd066ddaea4640f79b561b307c35e1f1e8ab9e6f1447b23f085833192108` |
| **Student-PT** | the round-3 candidate EMA checkpoint, sha256 `1193e2117b56381ccc078a5b5e21d61d0ad0903565dcca0cb110b39ce24ed4a9`, recipe `aug94-mixture-v2`, step 10000; the PyTorch side of P2 and the weights v1.0.7 ships (`models.zip` above) |
| **Teacher** | `obadx/muaalem-model-v3_2` at hub revision `01a1ef9fbe40d144ef845101e89ff924aed3fef5`, arm `muraja-app-checks-v2`; its gating decode is pinned at sha256 `91b66205beb25e36044558e91cd96876b6e439c8d33d24ca4165cb896abaf172` |
| **Oracle** | the clip's reference phonemes placed on its phase A Teacher word times and replayed through the same harness — no checkpoint; `tracer/oracle` at `c7c6cd07` |

The App run's record (`app_arm/summary.json`, `run`): audio manifest
`1162aa8b4adba6c0a8b7a0d053bcc9e5dcd3c71ccd820e7f88af819676d57982`, word table
`6064be7082341690db9534f987b636c0e6bd482abf66aa1bd708dd9473dbb3b2`, seeds P2 `319` and P3
`309`.

**Gates and their results.**

- **P1** (S0, `c4bdea30`): all **74** Muraja `tools/test_logs` replay exact under the shipped
  default (final position and every word's quality); the owner's sessions reproduce every
  `Grade end word` decision, and the 9 differing words at the audit's commit are each explained
  by a 1.0.28 change (`docs/followalong-audit-parity.md`). **Pass.**
- **P1b** (S1, `c7c6cd07`): 20/20 drawn clips replayed both ways through the 30-Character
  `.log` overlap clip; **8** word decisions change against the full overlap
  (`tracer/p1b.json`, verdict `pass`). The audit's answer is the full-overlap run.
- **P2** (S3, `9f3f6da7`): CoreML vs PyTorch strict outcomes — **2,150 / 2,183 = 98.5%** on the
  200-clip subset and **56,768 / 57,935 = 98.0%** over all clips, against the **97%** bar. **Met**
  (seed 319).
- **P3** (S3, `9f3f6da7`): App cadence at 0 ms and 130 ms over 500 clips — strict **59 /
  5,006 = 1.2%** of word outcomes change, balanced **54 / 5,006 = 1.1%**, under the **2%** bar. No
  finding (seed 309). The device's preview:window ratios (4.8–5.8) are whole-session logs against
  the App arm's 7.8–8.3, so they are disclosed and not read as a latency comparison.
- **S2's oracle share** (`314dd865`): **1 / 5,157** App strict flags (0.019%) is explained by the
  oracle, and **0 oracle candidates** exist — no pattern covers ten words. The App's false flags
  are therefore not an oracle-reference artifact, and S2's planted misses are catch capacity, not
  patterns.

## 3. Strata and sizes

The pools come from S3 (`app_arm/pools.json`). They are **word pools**, not slices: the names S1,
S3, S4, S5 below are the pools `assign_stratum` returns, and the repeated names are unavoidable.

| pool | words | what it is |
|---|---:|---|
| S1 | 5,151 | the App flags, the oracle passes |
| S1-pending | 5 | the App flags, the oracle left the word ungraded |
| oracle-explained | 1 | the App flags and the oracle flags too |
| S3 | 1,459 | the App passes and the base detector, or ≥ 2 detectors, flag |
| S4 | 38,598 | the App passes and nothing flags |
| S5 | 9,342 | the App left the word ungraded |
| residual | 3,379 | the App passes and exactly one non-base detector flags |
| **total** | **57,935** | 5,742 clips |

S1 ∪ S1-pending is the S1 pool proper, **N = 5,156**: the 5 pending words are the oracle's
*ungraded* words, not oracle-explained ones. The `residual` and `oracle-explained` pools are
**not sampled** in this audit; their counts are reported only.

**The design.** The main sitting is **540** stratum snips — S1 200, S3 250, S4 60, S5 30 — and a
**150**-snip pilot precedes it. **S4 is kept as the control** (unlike #261's proposed drop): the
audit needs a miss bound from words nothing flags.

| stratum | pool | n | pilot | inclusion |
|---|---:|---:|---:|---:|
| S1: the App flags and the oracle passes | 5,156 | 200 | 56 | 0.0388 |
| S3: the App passes and the base or ≥ 2 detectors flag | 1,459 | 250 | 69 | 0.171 |
| S4: the App passes and nothing flags | 38,598 | 60 | 17 | 0.00155 |
| S5: the App left the word ungraded | 9,342 | 30 | 8 | 0.00321 |
| repeats: seeded re-asks of this audit's own items | 150 + 540 | 25 | 7 | - |

The table's inclusion column is each stratum's overall n ÷ pool: S3 and S4 are one cell each, so
that is every word's inclusion probability there. **S1 and S5 are drawn cell by cell**, so a
word's inclusion probability is its own cell's π = n_cell ÷ N_cell — tabulated with S1's cells
below and S5's buckets — and §7 weights by that π, never the stratum average. The n above are the
frozen design; the §8 resize extends the same permutations and recomputes each cell's π as final
n ÷ N. The repeats are re-asks, not a pool: the pilot's 7 come from its 150 served items and the
main sitting's 18 from its 540 (the row's pool is both), and they carry no inclusion probability.

**S1's cells.** A word can carry several causes; its cell is allocated by one **primary cause**,
the first match in
`target_pair, soft_pair, other_consonant, shaddah, tashkeel, skipped, unrecorded` — deliberately not
`training.word_flags.CAUSES`' order, because decision rule 3 turns on soft and target pairs.
Primary cause × base-detector flag gives 14 cells summing to 5,156. `pools.json`'s
`s1_by_cause_and_base` counts a word once per cause it carries (any-cause), not by primary cause;
the cells below and S4's ungraded split are the committed `app_arm/cells.json`, written by
`python -m tadabur.followalong_audit strata` at the pinned harness:

| cause | base detector flags | words | n | π |
|---|---|---:|---:|---:|
| other_consonant | no | 1,150 | 44 | 0.0383 |
| other_consonant | yes | 1,607 | 61 | 0.038 |
| shaddah | no | 61 | 3 | 0.0492 |
| shaddah | yes | 174 | 6 | 0.0345 |
| skipped | no | 147 | 5 | 0.034 |
| skipped | yes | 296 | 11 | 0.0372 |
| soft_pair | no | 129 | 5 | 0.0388 |
| soft_pair | yes | 344 | 13 | 0.0378 |
| target_pair | no | 24 | 3 | 0.125 |
| target_pair | yes | 41 | 3 | 0.0732 |
| tashkeel | no | 283 | 11 | 0.0389 |
| tashkeel | yes | 416 | 16 | 0.0385 |
| unrecorded | no | 98 | 4 | 0.0408 |
| unrecorded | yes | 386 | 15 | 0.0389 |

The 200 snips are allocated **proportionally with a floor of 3 per cell**, the remainder by
**largest remainder**, 200 in total. The floor binds only for the shaddah-no cell and the two
target_pair cells; without it they would get 2 and 1–2 snips. Readouts report **every cause** a word carries, not only the primary; rule 3's
soft/target test is any cause in {soft_pair, target_pair}.

**S5's composition.** S5's 9,342 words are mostly clip edges: **leading 3,933**, **trailing
5,103**, **interior 306**, and **250 clips** are entirely ungraded. The oracle arm leaves only 245
of 57,935 words ungraded, so these are real decodes at clip edges (single-ayah clips put a session
start and end every ~10 words) or words not in the recorded audio. S5's 30 snips split
**leading 10 / trailing 10 / interior 10**, and the buckets do not share π:

| S5 bucket | words | n | inclusion |
|---|---:|---:|---:|
| leading | 3,933 | 10 | 0.00254 |
| trailing | 5,103 | 10 | 0.00196 |
| interior | 306 | 10 | 0.0327 |
 An "unusable audio" verdict carries the sub-reason
`not_in_clip` when the word is not in the recording. Every headline rate is also reported without
each clip's first and last engine word. A possible end-word mechanism — the withheld grade is
stored only at score ≥ 0.75 (`GradeStore.swift:663,676-691`) — is checked in S7, never asserted
here.

**S4 means "nothing that graded it flagged it".** `assign_stratum` treats an ungraded detector as
not flagging, so S4 is "the App passes and no detector that graded the word flags it". Of the
39,936 word × detector pairs that are ungraded, **2,184** S4 words had **at least one detector** ungraded and **57** had **all four detectors** ungraded. S4's bound is labelled accordingly.

**Draw mechanics.** S5 draws the full design first, as one **seeded** random **permutation** per
cell. The pilot is each cell's **prefix**, allocated within a stratum by largest remainder over the
cells' main allocations with no floor; later and resized draws extend the same permutations, so
every cell stays a simple random sample. The **inclusion probabilit**y of a word is its cell's
final n ÷ N.

**Repeats.** Round 2's catch-loss worklist **matched 0** of its 150 words in this population (they
are look-tranche rows), and round 3's worklist is checked the same way; any match is excluded. So
the 25 **seeded** **re-ask**s are this audit's own items: 7 in the pilot and 18 in the main
sitting, each served at least **20 items** after its original. The **first verdict** counts in
every rate; repeats feed only the agreement figure.

## 4. Verdicts

Every served item gets exactly one verdict:

- **correct**;
- **mistake**, with one or more kinds: **letter**, **haraka**, **shaddah**, **skipped**, **added**;
- **not sure** — offered for every item, and expected on the confusable letters س/ص/ث;
- **different text** — the clip recites words the reference does not hold;
- **unusable audio**, with the sub-reason `not_in_clip` when the word is not in the recording.

"Not sure", "different text" and "unusable audio" are **excluded from the rates** and counted
beside them, so no denominator silently shrinks.

## 5. Attribution

The audit attributes every decisive verdict to a bucket with the functions in
`tadabur/followalong_audit/attribution.py`, pinned by fixtures. A check's **effective quality** is
its `withheld` value when it has one, else its `quality`; **below correct** means an effective
quality not in {correct, pending}. The **kept grade** is the displayed one
(`training.muraja_policy.displayed`): a kept `pending` that carries a `withheld` value is
**ungraded**. The buckets read **strict**; the **balanced** computation is the same and reported
second.

**False flags** — the App flagged the word and the owner heard it correct. First match wins:

**F1 → F1-align → F-ref → F2 → F3**

- **F1**: the oracle grades the word below correct (its kept outcome is `flagged`).
- **F1-align**: the oracle's kept outcome is `passed`, and one of the **setting check**s of the
  App trace holds the word's reference phonemes verbatim: its full `hop + overlap` contains, as an
  exact **code-point** substring, `ref(w−1) + R` or `R + ref(w+1)` — the word's reference R joined
  with a **neighbour** as the oracle text joins them. A clip with no neighbour at all matches on R
  alone only when `len(R) ≥ 4`. The **sensitivity** row is R alone, reported and never deciding;
  the primary rule leans toward the model side, which the sensitivity row bounds. An ungraded
  oracle is neither F1 nor F1-align.
- **F-ref**: the same deviation at the same word in ≥ 3 distinct tranche `group_key`s, a
  **reference suspect** computed by `reference_suspects`. The deviation is the App strict kept
  grade's error set as the **sorted** tuple of `(group_index, kind.type, kind.expected,
  kind.heard or None)`, over the whole population's traces. That needs the full-population App
  traces: S5 regenerates them at the frozen identity and must reproduce every committed
  `word_grades` row (flagged, graded, causes). A word that does not reproduce is listed and never
  drawn. `app_arm/payload_probe.json` shows the decoder is not bit-stable (two payloads with an
  identical word table), so this gate can fire; it removes a word from the draw rather than
  re-grading it. It is checked after F1/F1-align and before F2/F3, and it is never settled by hand: rule 2 is computed with F-ref on **both** sides
  and both are reported if they differ.
- **F2**: the Teacher arm, through the engine, passes the word. **F3**: it flags the word or left
  it **ungraded** — a Teacher-ungraded word is **not learnable** (D8), counted and shown to the
  owner and Fable.

**Missed mistakes** — the owner heard a mistake the App did not flag. Every label met is
recorded, in precedence order:

**M1r → M1 → M1u → M2 → M3**

- **M1r**: the App's kept grade is `passed` and some check's status is below correct.
- **M1**: the kept grade is `passed` and a setting check decoded the deviation away — its status
  has errors or a score below 1.
- **M1u**: the kept grade is **ungraded** and some check's status is below correct — the engine
  graded the word and the app dropped the grade.
- **M2**: the Teacher arm, through the engine, flags the word. **M3**: neither arm flags it (the
  Teacher's ungraded outcome is M3, D8).

`side(bucket)` puts F1, F1-align, M1r, M1 and M1u on the follow-along side and F2, F3, M2 and M3
on the model side; F-ref goes to whichever side rule 2 is reading. `model_share` is rule 2's
fraction. A false-flag call on a word the App did not flag, and a miss call on a word it did,
raise.

## 6. Decision rule

The PRD's rule, frozen with these numbers:

1. **Follow-along first** if a bucket has **≥ 3 instances in ≥ 3 clips**, or the oracle's pattern
   covers **≥ 10 words**. A defect of the follow-along path is fixed before any model work.
2. **A fix is accepted only if a corpus replay loses no verified catch**: the replay must not turn
   a known correct catch into a miss.
3. **Fine-tune at ≥ 50% model share** — compute it after the follow-along fixes, as the model
   side's fraction of the attributed words (F-ref on each side, both reported).
4. **The miss side decides only with ≥ 60 confirmed misses.** Below 60 the audit reports
   **direction** and decides nothing on misses.
5. **F2 + M2 vs F3 + M3** picks the kind of model work: F2/M2 say the Teacher sees what the App
   does not (learnable), F3/M3 say the word is beyond the Teacher. If **soft or target** pairs
   hold **half of F3**, propose a **strict-mode allowance** instead of a fine-tune.

## 7. Estimates

Two sides, never pooled (ADR-0008):

- **False flags per 1,000 correctly recited words**, weighted by the word's **cell** inclusion
  probability π (S1's 14 cells and S5's 3 buckets do not share it), with a clip-bootstrap **95%**
  interval.
- **Misses per 100 confirmed mistakes**, the same way, with the same interval.

Every headline rate is also reported without each clip's first and last engine word, and in
balanced beside strict. **S4 is a labelled bound, not a headline**: it bounds the miss rate on
words nothing flags, and its detectors that graded nothing weaken the bound.

The **catch rate on known verified tlog mistakes** runs the App arm on the **33** verified
look-tranche mistakes — **11** from round 2 and **22** from round 3 (`look`, not this population).
It is a separate table labelled **biased upward** (the detections chose them) and it is **not a decision input**.

## 8. Pilot and resize rule

The pilot is 150 stratum snips (35-min) plus 7 repeats; the main sitting is **2.5 h**, with the
owner's listening allowed to run to the full **3 h**.

After the pilot, each stratum's confirmed-mistake yield over its decisive verdicts and the
measured seconds per snip are read, and the remaining snips are re-split to maximise the expected
number of confirmed misses toward 60 within the 3 h total:

- floors for the main sitting's totals: **S1 ≥ 150**, **S4 ≥ 40**, **S5 ≥ 20**, **repeats ≥ 15**;
- fill the floors, then add to the stratum with the **highest** pilot **miss yield**, up to its
  pool; **ties** go to **S3**;
- stop once the expected confirmed misses reach **60**, or when the **3 h** budget runs out.

At the frozen sizes the expectation is **~42** misses: 250 × 0.17, S3 only, from round 3
catch-loss's 17.39% removed-real share (`docs/consonant-round3-catch-loss.md`). S1 words are
App-flagged and cannot be misses, and S4/S5 have no stated prior, so they are assumed ~0; the
~42 is therefore S3's, and the **direction**-only conclusion rests on that assumption. Applying
17% to all 540 snips would give ~92 and cross 60, which is why the prior is scoped to S3. **Any other change is a dated amendment, made before the main sitting** — never a redraw after listening begins.

## 9. Dataset rows

The audit's verdicts will seed `tools/tadabur/truth_sites/tlog_mistakes.jsonl` (and its card), one
row per verified mistake. Each row carries:

- **provenance**: `audio_path`, `audio_sha256`, `shard`, `row`, the **sample span**
  (`start_sample`–`end_sample`) and the dataset revision — enough to re-download the audio and
  verify its **sha256**;
- the **verdict** and its **kinds** (letter, haraka, shaddah, skipped, added);
- `whole_played`: whether the whole clip was served or a snip;
- the word's `(surah, ayah, word)` and its `group_key`;
- licence **CC-BY-4.0** attribution for the Tadabur audio.

**No audio** is committed, ever. **No training without a dated `group_key` split**: a row may be
trained on only under a split that keeps one `group_key` on one side. **Mined label tiers stay out
of this audit**: its rows are listening verdicts only. **Open for the owner at approval:** seed v0
from the existing verdicts (#225, #257, Ship A)?

## Changelog

- 2026-11-24: written and frozen before S5's draw; all nine sections, with S3's pools and S0, S1,
  S3 and S2's gate results cited by commit.

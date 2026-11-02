# tlog mining pilot: the pre-registered design for confirm.1 ∪ confirm.2 (#262)

**Status:** frozen 2026-11-02, **before the draw** and before any listening, and dated. It is the
pilot [`docs/tlog-mining-amendment.md`](tlog-mining-amendment.md) opens; that amendment is the
mining rule, this document is the design. Once this is committed, the strata, the seed, the
question, the readout and the scale-up rule do not change. A change needs a new dated
document, never a redraw.

Issue: [#262](https://github.com/sysofwan/HifzGuide/issues/262). Parent: #261 (epic #77; tlog
#173). Slice 3 builds the snips this document's presentation names; the draw waits for both
this document and slice 3's snip code to be committed.

## 1. Population

The **clean-split gating** clips of confirm.1 ∪ confirm.2: **2,843 + 2,919 = 5,762** clips.
Every gating row of both tranches is on the `clean` split, and all 5,762 were decoded whole-clip
by each of the three detectors (§2), so the words they carry are the same population in all
three decodes.

**Minus the known exclusions.** The 20 clips in
`tools/tadabur/consonant2_readout/report.json` under `coverage.graded.excluded_clips.keys`
(SHA-256 of the report `2149ea0f2f5dc4ba125f718f89f6a942fd2c30a39850f1d259e2823e338b1117`;
its reason is "no item: the row names no ayah, or the phonetizer cannot handle the ayah the row
names"). These are **#248**'s known exclusions: the 8 unphonetizable ayat **55:17**, **90:8–10**
and **106:1–4**. Seven of the 8 carry gating clips here; 90:9's four clips are `reported:*` rows,
outside the gating population either way. Eight of the 20 excluded clips are in confirm.1 and 12
in confirm.2. After them the pilot's population is **5,742
clips in 5,657 `group_key`s** — the same count round 2's report grades.

A **word** is (clip key, 0-based Uthmani word index), exactly as in
`tools/tadabur/catch_loss_audit.graded_words`: the clip's `key` (`audio_path`, plus the segment
index the export carries) and the word's position in the filename ayah's Uthmani text. The word
population is every word of those 5,742 clips that all three detectors graded; its size is
recorded with the draw.

**Phase B's rows are a later population, not this pilot's.** The `reported:*` rows this
amendment makes mineable, the other tranches and the round-3 tranches are out of scope here.

## 2. Detectors

All three are reused **by checksum** from
`tools/tadabur/consonant2_readout/decodes/tlog.confirm.*` (the round-2 probe decodes; each file
holds the 5,762 gating rows):

| detector | identity | arm |
|---|---|---|
| base teacher | `obadx/muaalem-model-v3_2` @ `01a1ef9fbe40d144ef845101e89ff924aed3fef5` | `checks` |
| C95 | checkpoint SHA-256 `8490b63d0664827d5a2e2e3121eb4af7b22069273db39268c7638cc7d8839cd6`, recipe `aug94-mixture-v2`, step 10000, EMA | `checks` |
| round-2 EMA | checkpoint SHA-256 `893523710186cb16854dd62a258331644e2284abbee8b2cb54e8ab9d32dbea58`, recipe `aug94-mixture-v2`, step 10000, EMA | `checks` |

The decode files and their pins:

- `decodes/tlog.confirm.base.jsonl.gz` — SHA-256
  `91b66205beb25e36044558e91cd96876b6e439c8d33d24ca4165cb896abaf172`
- `decodes/tlog.confirm.c95.jsonl.gz` — SHA-256
  `644a1bd1a3e25ceb320ff32e3d3125729e3d3dc14f5270a82627772d0418a5fc`
- `decodes/tlog.confirm.r2_candidate_ema.jsonl.gz` — SHA-256
  `4de59c3c8dbceb630a854fb00982b16874885414839c7bc63919c8bc53d1c46a`
- their record, `decodes/meta.confirm.json` (SHA-256
  `726bdd381f35a8e19091418b8f8539bbe897cefbe43788e8b519bc8d3449d329`), is the pin for the
  identities above and for decode code SHA `99374138b7e8ccde47c7d5281fe4e2fa0f4ba5d5`.

Every detector decodes at the `checks` arm: `muraja-app-checks-v2`, policy
`fp32-features-v1`, bf16 weights, batch 1. `r2_control_ema` (checkpoint
`da5fb5bf…`) also decoded the tranches; it is **not** a detector here and its grades are
unused.

**No zipformer, in any role.** No zipformer decode of a confirm tranche exists, and no
zipformer is a detector, a tie-break or a check here.

The **round-3 candidate** is a later detector and scored model (slice 2, phase C). No readout of
it on the mined set comes before round 3's verdict is committed.

## 3. Flag

A word's flag is the **Muraja harness word grade** at
`99c326f3fb7c6c53ff525568976c7e2442ccde3c` (`MURAJA_COMMIT`, protocol 3) under **every allowance
off** (ADR-0012): the `allowances_off` scoring of `tools/training/muraja_policy.py`
(`all_off(ALLOWANCES_ON)`) — mode `balanced`, `softPairsEnabled`: false,
`shaddahSuppression`: false, `tashkeelShown`: true. Its exact scoring key is pinned here by
those fields, not by prose.

A word is a **flag** when the grade's export `flagged` boolean is true — the grades `flagged`,
`flagged_elsewhere` and `flagged_unattributed` (`WORD_FLAGGED`). The `outcome` string is not the
flag: a flagged word's outcome is `flagged_unattributed` (the carrier's group is unknown) or
`flagged_elsewhere`, never `flagged` itself, and every one is still a flag — the same definition
`catch_loss_audit` uses.

The shipped-default grade (`shipped_default`) is recorded alongside every word, so the shipped
presentation stays measurable, and is **never used for selection**: no stratum, ranking, readout
line or scale-up decision reads it.

## 4. Strata and pilot sizes

A word's stratum is the set of detectors that flag it: all three, exactly two (any pair), or one
named detector; plus the "none" stratum.

| stratum | words |
|---|---|
| all 3 detectors flag | 60 |
| exactly 2 flag | 50 |
| only teacher | 25 |
| only C95 | 25 |
| only round-2 EMA | 25 |
| none flag | 55 |
| **total** | **240** |

These sizes are a **judgment call**, aimed at a yield of about 25 to 40 mistakes. They are not
derived from a power rule.

- The **"none flag"** stratum exists because C95 and the round-2 EMA are distilled from the
  teacher and share its blind spots: a disagreement census alone cannot find a mistake every
  model misses.
- **Take all of a smaller stratum.** A short stratum's shortfall goes to "all 3", then to
  "exactly 2", then to "none". If the whole population is under 240 words, take all of it. The
  **actual sizes** per stratum are reported after the draw.
- At most 2 words per clip and at most 4 per `group_key`, so one reciter's single clip cannot
  dominate a stratum.
- **Ranking within a stratum:** a **salted hash**, ascending —
  `sha256(f"{SEED}:{stratum}:{clip key}:{position}").hexdigest()`, the
  `catch_loss_audit._rank` form — with seed `tlog-mining-pilot-2026-11`. Nothing about a model
  enters the order.
- The **realised inclusion probability** is recorded for every word: `n / N` in a sampled
  stratum (the planned size over the stratum's population), and 1 in a take-all stratum.

## 5. Presentation

Each item plays the snip: the word, **one word either side**, and **0.3 s** of padding — the
rule of `tadabur.listening_session.excerpt_span` (`EXCERPT_PAD_S = 0.3`,
`MIN_EXCERPT_S = 1.0`). It is cut from the teacher's **forced alignment** of the **reference**
(the teacher-only timing pass, amendment §5), never from a decoded mark (Ship A's rule).

The **fallback** is the whole clip, whenever the word times cannot place the words: re-reads, no
word times, a span that does not overlap the segment, or an excerpt shorter than 1.0 s. A
**"play whole clip"** button is always available regardless. The **fallback rate** is reported
with the readout.

## 6. Question (blind, ADR-0007)

The item shows no model output, no detector, no flag state, no stratum and no tally. The
question is: **"Was this word recited correctly?"** — **yes** / **no** / **unclear**.

On "no", the owner marks one or more kinds: **wrong letter** / **haraka** / **shaddah** /
**missing or extra** letter or word / **tajweed only** (madd, ghunna, qalqalah) / **other**.

A **verified mistake** is a **decisive** "no" with at least one kind other than "tajweed only".
"Unclear" and tajweed-only "no"s are reported, never counted as verified mistakes.

## 7. Readout

**Reported only**: nothing here gates a ship, a release or a later round's verdict.

- **Yield per stratum**, with a **Wilson** interval.
- **Per detector**: the **Horvitz–Thompson**-weighted catch rate and precision, with a
  `group_key`-clustered **bootstrap** (each word weighted by the inverse of its realised
  inclusion probability, §4).
- The breakdown **by kind**, and the **unclear** rate.
- Words whose flag involves **س**, **ص** or **ث**, reported apart: the owner's ear is noisier
  there.
- **Owner seconds per word**, and the snip **fallback rate**.
- **Correct-side and mistake-side outcomes are never pooled** (ADR-0008): the readout keeps the
  two sides separate, per stratum and overall.

## 8. Scale-up rule

Fixed now, as a judgment call: recommend a scale-up only if **both** hold —

- the **"all 3" stratum yields ≥ 15% verified mistakes**;
- the pilot yields **≥ 20 verified mistakes**.

Otherwise, **re-plan** (new sizes, new strata or a different detector set), with a new dated
document.

## 9. Later training use

Any use of a mined word for training needs a further dated split by `group_key`, committed before
training starts. That split divides the mined rows into eval and training halves — never by clip,
word or take. Until it exists, mined rows are read, listened to and reported, never trained on.

## 10. Changelog

- 2026-11-02: written and frozen before the draw; owner approved the design on #262
  ("#1 yes, #2 agreed, #3 yes, #4 yes").

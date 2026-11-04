# Erratum: the FastConformer screen's teacher bars are letter-level, not harness word grades

Dated 2026-11-04, committed before any FastConformer decode (#267). The screening rule and its
thresholds are unchanged; this corrects how the teacher's bars are read, so the frozen rule's
comparison is like-for-like.

## What was wrong

The pre-registered rule quoted the teacher's #245 Ship A figures — 26/75 mistakes flagged and
180/410 correct sites flagged — as if they were the rule's measure. They are Muraja's
**word-level, any-cause** grade of the whole-item decode at the shipped default
(`_whole_item_outcomes` / `WORD_FLAGGED`). The rule's own words define a different measure:
a site is flagged when the decode **does not carry the prescribed letter or shaddah**. The
edits figures (34/150 heard, 88/150 decoys unchanged) were already letter-level
(`teacher_agrees`) and do not change.

Every column is now read the same way, on the same sites: what the whole-item decode carries at
the site's carrier — **heard** (the letter the human heard), **prescribed** (the mushaf's letter;
the text prior), **other**, or **none** (nothing aligned). `none` counts as flagged. All rates
below are on the **located** sites, minus the 8 G6 held-out sites, for every model.

## The bars (the run's comparison basis)

| column | Ship A mistakes flagged | Ship A correct flagged | edits heard | decoys unchanged |
|---|---:|---:|---:|---:|
| **teacher (base)** | **56/67 = 83.6%** | **259/407 = 63.6%** | **34/150 = 22.7%** | **88/150 = 58.7%** |
| C95 | 55/67 = 82.1% | 261/407 = 64.1% | 54/150 = 36.0% | 93/150 = 62.0% |
| r2 EMA | 57/67 = 85.1% | 245/407 = 60.2% | 47/150 = 31.3% | 90/150 = 60.0% |

Effective bars under the unchanged rule: **STOP** if FastConformer's correct-side flag rate is
**> 68.6%** or its decoys unchanged are **< 53.7%**; **GO** only if the correct side passes, its
mistakes flagged are **≥ 78.6%**, edits heard **≥ 17.7%**, and it has **≥ 3 unique catches**.

## Per-outcome breakdown (teacher, located set)

| side | n | heard | prescribed | other | none | flagged |
|---|---:|---:|---:|---:|---:|---:|
| mistakes | 67 | 54 | 11 | 1 | 1 | 56 (83.6%) |
| correct | 407 | 148 | 0 | 198 | 61 | 259 (63.6%) |

C95: mistakes 53/12/2/0; correct 146/0/199/62. r2 EMA: mistakes 53/10/1/3; correct 162/0/183/62.

- **83.6% is a flag rate, not a miss rate.** 54 of the 67 decisive mistakes are genuine catches
  (the heard letter). The 11 `prescribed` are the text prior writing the mushaf over a real
  mistake; the GO bar is dominated by catches.
- **The correct side is high because the rule reads the letter.** The 198 `other` are real
  partner-letter substitutions in the teacher's committed text (ط→ق 74, س→ش 30, ظ→ض 24,
  shaddah held→not_held 24, ز→ذ 9, ط→ت 6, س→ص 5, ك→ق 5, ق→ك 3, …; together 186 of 198 are a
  partner letter or the shaddah state). Muraja's word grade forgives soft pairs and reads
  words, which is why its 43.9% is not comparable.
- **The 61 `none` are model-side.** In every one the teacher's committed `stream_b0` text has no
  consonant aligned at the carrier; all 61 sites are located, so mapping contributes **zero**
  of them.

## Sampling disclosure

The correct side is not a population sample of correct recitation: it is the sites the screen's
sampling drew that a human verified as correct, and it is enriched for hard sites. By stratum:
`base_right 197, base_wrong 133, base_not_held 45, h448_partner 20, base_partner 12`
(133/407 located are sites the teacher got wrong; 136/410 before the 3 unlocated were removed).
The mistake side: `base_not_held 31, base_wrong 20, base_right 13, base_partner 3`. These bars
describe this screen's verified set, not the corpus.

## Committed-only sensitivity column (`none` not flagged)

Reported for context; the frozen rule never reads it. A site is flagged under the canonical
decode rule only when an explicit different output is committed, so an unaligned carrier is not
flagged:

| column | mistakes (sensitivity) | correct (sensitivity) |
|---|---:|---:|
| teacher | 55/67 = 82.1% | 198/407 = 48.6% |
| C95 | 55/67 = 82.1% | 199/407 = 48.9% |
| r2 EMA | 54/67 = 80.6% | 183/407 = 45.0% |

## Mistakes none of our three models hears

Five decisive mistake sites where the teacher, C95 and the r2 EMA all carry the **prescribed**
letter (the text prior wrote over the human-heard mistake). They are the outside detector's only
way to earn its place; the screen counts FastConformer's unique catches against them:

- `new_audit:tadabur_spk0033_S57_A1_f0377864_000000.wav#1@28:ق↔ك` (heard ق)
- `new_audit:tadabur_spk0054_S19_A123_b38ff3f4_000069.wav#0@141:ض↔ظ` (heard ظ)
- `new_audit:tadabur_spk0088_S6_A145_2180fb52_000003.wav#0@58:ض↔ظ` (heard ض)
- `new_audit:tadabur_spk0219_S1_A116_175bd41b_000066.wav#0@38:ح↔ه` (heard ه)
- `new_audit:tadabur_spk0518_S37_A12_cb5c02d3_000057.wav#0@54:ذ↔ز` (heard ز)

## Mapping provenance

Step 2's text-only mapping locates **516/522** Ship A sites (98.9%, above the 95% gate) and
**300/300** edits. The six unlocated sites are excluded and listed here; three are correct-side,
three have no human side:

- `…spk0002_S17_A18_ede49820_000011.wav#0@20:ض↔ظ` (no side)
- `…spk0016_S17_A18_e1d73e40_000012.wav#0@20:ض↔ظ` (correct)
- `…spk0051_S17_A18_5a3f865e_000018.wav#0@20:ض↔ظ` (correct)
- `…spk0066_S23_A3_33c43374_000002.wav#0@2:shaddah` (no side)
- `…spk0114_S17_A18_0356123b_000011.wav#0@20:ض↔ظ` (no side)
- `…spk0350_S37_A13_fc588788_000000.wav#0@23:ط↔ق` (correct)

## Model pin (SHA correction)

The issue quoted the `.nemo` SHA-256 as `d29d19d7…2e31`. The full LFS object id at revision
`7f32349d952f42a28dce979ba73270aa2bbdfa89` is
`d29d19d7c054a5fc010ac6815e9cbb0dd1b21a30e0a7f7f2982e1fecaf0c3e31` (459,233,280 bytes); the
`…2e31` suffix was a typo. The decoder hashes the downloaded file locally and refuses anything
but this full hash.

## Still context-only

#245's harness numbers for the teacher, C95 and EMA, recomputed minus G6, are due at readout as
context only; no bar above reads them. Ship A holds consonant-pair and shaddah sites only: this
screen cannot certify haraka.

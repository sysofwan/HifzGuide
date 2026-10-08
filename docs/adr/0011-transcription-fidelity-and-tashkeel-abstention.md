# The accuracy objective is transcription fidelity, and an empty tashkeel slot means "unsure"

**Status:** Accepted, 2026-10-08. Restates the accuracy goal in the owner's words.
**Supersedes** [ADR-0004](0004-waqf-head-and-joint-whole-clip-fine-tune.md),
[ADR-0009](0009-waqf-operating-point-asymmetric-cost.md) and the objective of
[ADR-0001](0001-tadabur-filter-and-finetune-methodology.md).
**Amends** [ADR-0002](0002-waqf-aware-tadabur-reference-labels.md),
[ADR-0003](0003-tashkeel-fine-tune-labels.md), [ADR-0008](0008-the-eval-measures-the-decode-not-the-gate.md)
and [ADR-0010](0010-size-distillation-to-a-single-ane-chunk.md).

## Why this is written down again

The accuracy track spent a cycle optimising the wrong thing. Two objectives were bundled
under "the model is over-strict on amateurs", and the training data served only one of them:

| | objective | what it needs |
|---|---|---|
| **tolerance** (ADR-0001) | stop rejecting acceptable-imperfect articulation | the model emits the mushaf's letter more readily |
| **discrimination** (ADR-0001, ADR-0003, ADR-0008) | emit the letter and the tashkeel the reciter actually said | the model follows the audio, including where it departs from the mushaf |

ADR-0001 trained toward canonical mushaf labels over clips admitted by a gate that forgives
the confusable pairs and cannot see tashkeel at all (ADR-0005). That objective is literally
*maximise P(decode = mushaf | audio)*, and with 91% of validation ayat also present in the
training split it is learnable as a text prior. The measurements say that is what happened:

| | base teacher | LoRA fine-tunes (after the vowel-stripping fix `08e74e8`) |
|---|---|---|
| ambiguous-skeleton words, against a text-only baseline of 0.9734 | 0.9342 (below it: listens) | ~0.962 (moving toward it) |
| deliberately wrong haraka decoded as the mushaf's (`followed_text`) | 0 / 41 | 2 / 42 and 4 / 42 |

The poison ADR-0001 accepted was not a minority either: the P3.5 audit found the gate's
admitted *added-shadda* cases 86% genuinely wrong (`tadabur.scorer`).

The vowel-stripping bug fixed in `08e74e8` is not the explanation. Checkpoints trained on
stripped labels emitted no haraka at all; the drift above was measured on checkpoints that
emit haraka at ~0.98 recall, i.e. trained on the corrected labels.

## Decision

### 1. The objective

The shipped model decodes **what the reciter actually said**, in both directions, for:

- the confusable consonants — the six soft pairs `ذ↔ز`, `ت↔ط`, `ض↔ظ`, `ك↔ق`, `س↔ص`, `ح↔ه`
  (whether `ذ↔ظ` is in scope is open);
- **tashkeel**: haraka (fatha, damma, kasra), **sukun** and **shaddah**.

A correct recitation must not decode as a mistake, and a mistake must not decode as the mushaf.
**Tolerance belongs in Muraja's scorer**, which is tunable per mode; the model is not.
"Let Muraja default to `.strict`" stays a statement of the *product consequence*, judged per
allowance (§5), never a metric.

The failure the owner sees in the app is **dropped tashkeel**: a correctly recited mark the
model leaves empty, which Muraja then flags.

### 2. An empty tashkeel slot means "unsure", and sukun gets its own output

The model today emits nothing both when the reciter said **sukun** and when it is **unsure**
of the haraka. Muraja cannot tell the two apart, so every allowance it has for tashkeel is a
guess about which one it is looking at:

| mushaf | model emits | could mean | Muraja today |
|---|---|---|---|
| haraka X | haraka Y | a wrong haraka | flagged in every mode |
| haraka X | **nothing** | **reciter said sukun (a mistake), or model unsure** | **flagged**, except on و ا ء ي (every mode) or any letter (`.lenient`) |
| sukun | nothing | correct, or model unsure | match |
| sukun | a haraka | reciter added a haraka | flagged |
| shaddah | one consonant | not held, or model unsure | ignored in `.balanced`; the group's tashkeel is discarded in every mode |

No setting can be right both ways while one symbol means two things. So:

- The student's output gains an explicit **sukun** class. An empty slot then means only
  "unsure", and Muraja's rule becomes uniform: a *different* mark (sukun included) is an error,
  an empty slot is **not graded**. That replaces the per-letter exemption and lets the
  `suppressHarakaDrop` / shaddah-suppression allowances be retired as evidence permits.
- **Shaddah** needs the same three states — held, not held, unsure. Whether a CTC decode rule
  over the existing doubled-consonant encoding can supply them, or a gemination class is
  needed, is decided by a probe of the teacher's posteriors at geminates it collapsed, not
  in advance.
- The output vocabulary may change. It is kept in lockstep with Muraja's
  `PhonemeVocabulary` snapshot (`tadabur.phoneme_vocab`, `fixtures/muraja_phoneme_vocabulary.json`),
  and a model with a new vocabulary ships only together with the matching Muraja change.

### 3. Where training signal may come from

- **Never from the mushaf where the teacher omits or disagrees.** That is the text prior
  entering at exactly the site the goal is about.
- The **base** teacher's posteriors (the KL ADR-0010 already uses). The base teacher listens;
  the fine-tunes did not.
- **Human-adjudicated sites** — evaluation first; training only once enough exist.
- **Synthetic edits** of real audio (shaddah by stretching or cropping the hold; consonant
  splices from same-reciter donors), each paired 1:1 with an **unedited decoy** labelled
  unchanged, and admitted only after a blind listen confirms the edits sound real.
- The mushaf only where it **agrees** with the teacher. The sukun target, for example, is set
  where the mushaf has sukun *and* the teacher emits no haraka; where the mushaf has a haraka
  and the teacher is silent the target is "unsure", because the truth is unknown there.

### 4. The shipped artifact

Within the byte and ANE budget of the current student `h448` (~85 MB at 6-bit, one ANE
chunk). The default route is to keep training `h448` with KL to the **base** teacher plus a
small supervised term on trusted signal (§3); a fine-tuned teacher and re-distillation are the
escalation if `h448` cannot absorb the signal without losing teacher agreement. Starting a new
teacher from `w2v-bert-2.0` is rejected: it does nothing about the label problem and costs
days of full fine-tuning to reproduce what Muaalem already does.

### 5. How a change is judged

Against **truth**, never against the teacher or the mushaf:

- Per mark (fatha, damma, kasra, sukun, shaddah) and per pair: the **commit rate** on correct
  recitation (how often the model emits a mark rather than leaving it empty), and the
  **accuracy of committed marks**, reported separately on correct recitation and on real
  mistakes — never pooled (ADR-0008).
- Baselines: the base teacher **and the shipped `h448`**, which has never been scored against
  truth.
- Intervals clustered by clip or reciter. Acceptance rules, margins and any tuned values are
  **fixed before** results are read (ADR-0006, ADR-0009 record what happens otherwise).
- Each Muraja allowance gets its own pair of numbers, so "switch off allowance X" is a
  decision a reader can make from the report.

### 6. The waqf head is dropped

ADR-0004's head and ADR-0009's operating point no longer have a purpose. Muraja keeps its
end-word allowance (the last letter's haraka is ignored at a pause, a waqf sign or an ayah end),
and haraka at wasl boundaries stays ungraded.

## Consequences

- **Truth labels already exist for part of this.** The P3.5 fixtures (174 should-accept /
  35 should-reject clips) become per-site consonant and shaddah truth once each clip's
  labelled site is re-located; the 2,050 adjudicated waqf boundaries give 179 *sukun at a
  pause* sites and 1,857 wasl boundaries. The mid-word haraka audit (ADR-0007, #61) has never
  been listened to and is the main new labelling.
- **The counterfactual recordings no longer exist** (deleted with `audit_run/`, 2026-10-08).
  ADR-0006's rule stands for any re-recording, which must alternate which take is recorded
  first.
- **Data the repo depends on is committed**: human verdicts and recordings, site
  definitions with their population counts, and exact audio provenance (Tadabur filename,
  shard, sample span, checksum) for every labelled item. Disposable run directories are not a
  home for any of it.
- **Training labels from the mushaf are retired**, and with them the reason to filter the
  corpus for label poison. The ADR-0001 gate stays as corpus hygiene and as the miner of the
  reject pile, which is valuable precisely because it holds real mistakes.
- **Muraja must change with the model**: sukun on reference letters, sukun among the
  recognised marks, a larger vocabulary, and "empty = not graded" in place of the per-letter
  exemption and shaddah suppression.

# Tashkeel counterfactual fixtures (#10)

The recording sheet for the one experiment that can tell whether the Muaalem phoneme head
**hears** short vowels or **reconstructs** them from the Quran's fixed text.

## Why this exists

`training.tashkeel_eval` reports ~0.98 short-vowel recall for every fine-tuned rung, up from
0.841 for the base model. That number cannot support the claim ADR-0003 cares about.
`training.minimal_pairs` measured why: on the 6,403 held-out words whose consonant skeleton
is genuinely ambiguous, a **text-only** baseline that never hears any audio scores **0.9734**
— above every checkpoint.

| model | ambiguous-word accuracy | vs. text-only baseline |
| --- | --- | --- |
| base Muaalem | 0.9342 | −0.0390 |
| rung1_v3 | 0.9619 | −0.0114 |
| rung3_v2 | 0.9616 | −0.0117 |

The baseline is built from the training split's reference text alone, keyed on the word's
skeleton plus both neighbouring skeletons. The model is bidirectional and decodes surrounding
consonants reliably, so it can reach that number without hearing a single harakah. 91.37% of
val ayahs also appear in train, so the split separates *reciters* but not Quranic *content*.

Every clip in the corpus is **correct** recitation, so no observation in it can distinguish
the two hypotheses. That matters for the product: Muraja is a recitation **checker**, and a
model that reconstructs the canonical vowel would silently *correct* a student's mistake
instead of flagging it — a failure that only appears on incorrect recitation.

## The experiment

Audio where the spoken vowel and the canonical vowel **disagree**. There the hypotheses
predict opposite things: a model that hears transcribes what was *said*; a model that
reconstructs transcribes what the *text* says should be there.

- `counterfactual_sheet.csv` — the human recording sheet: phrase, target word and its
  position, the vowel it normally carries, and the vowel to say instead.
- `counterfactual_items.jsonl` — one `CounterfactualItem` per line (schema and generator in
  `tools/training/counterfactual_script.py`), read by the scorer.

Both are regenerated deterministically by:

```bash
python -m training.counterfactual_script \
  --manifest tadabur/audit_run/seg_v21/manifest_raw.jsonl \
  --labels   tadabur/audit_run/seg_v21/windowed_labels_v3.jsonl \
  --items 50 \
  --out-sheet    tadabur/tashkeel_counterfactual_fixtures/counterfactual_sheet.csv \
  --out-manifest tadabur/tashkeel_counterfactual_fixtures/counterfactual_items.jsonl
```

This directory ships the sheet and the item manifest — **not** the audio, which is recorded
per the protocol below.

## Recording protocol

Record each item **twice, same reciter, same phrase**:

| take | file | what to say |
| --- | --- | --- |
| control | `<item_id>_control.wav` | the phrase exactly as written |
| counterfactual | `<item_id>_counterfactual.wav` | the same phrase with the target word's vowel replaced by `instead_say` |

The control take is not optional. It is what makes a negative result interpretable: if the
model cannot decode the target word correctly even when it is recited *correctly*, that item
says nothing about hearing, and the scorer drops it rather than counting it as "failed to
follow the audio".

### Recording UI

`tadabur.counterfactual_record_ui` serves the items one at a time and captures both takes:

```bash
cd tools && python -m tadabur.counterfactual_record_ui
# then open http://127.0.0.1:8000
```

Takes are written to `--out-dir` (default `tadabur/audit_run/counterfactual_audio/`, which is
gitignored — audio does not belong in this directory), under the exact `take_1_file` /
`take_2_file` names the sheet specifies. The output directory is the only state, so the
session is resumable: restarting lands on the first item still missing a take.

Two things the UI does deliberately:

- **It serves on loopback only.** Browsers expose the microphone only in a secure context
  (HTTPS, `localhost` or `127.0.0.1`), so a LAN address records nothing. To record from
  another device, forward the port: `ssh -L 8000:127.0.0.1:8000 <server>`.
- **It refuses audio the scorer could not read.** The page encodes 16 kHz mono WAV itself
  rather than using `MediaRecorder` (whose WebM/Opus output `soundfile` cannot open), and
  every upload is decoded through `tadabur.audio.decode_to_mono_16k` before it is written.

## Why these words

Each item is a held-out word chosen so the recording is scorable and the result decisive:

- **exactly one short vowel**, in both the phoneme form and the written form — so the
  substitution and the scoring are unambiguous, and the sheet never asks the reciter to
  change two vowels at once (the written and phoneme forms disagree on vowel count);
- **no madd** — a vowel opening an elongation is not a free choice (مَا cannot be said as
  مُا), and ADR-0003 documents madd carriers as an alignment-artifact source;
- **a deterministic context prior**, attested at least 3 times in training — so a
  reconstructing model has every reason to emit the canonical vowel, which is exactly what
  makes following the audio decisive;
- **from the val split**, so the word was not in the fine-tune's training material.

Items are balanced round-robin across all six directed vowel swaps, so a model that hears one
colour but is deaf to another cannot hide behind an average. Each ayah and each word appears
at most once — three takes of بَلْ probe one lexical item, not three.

## Sample size

47 items (94 clips). The hypotheses are far apart — a hearing model follows the spoken vowel
~90% of the time, a reconstructing one ~10% — so the 95% intervals separate at n≈25:

| n | 95% CI at p=0.9 | 95% CI at p=0.1 |
| --- | --- | --- |
| 25 | [0.73, 0.97] | [0.03, 0.28] |
| 47 | [0.79, 0.96] | [0.04, 0.21] |
| 100 | [0.83, 0.95] | [0.06, 0.17] |

47 is capped by damma-initial words, which are genuinely scarce after deduplication. More
items would narrow the estimate, not change the verdict.

**Scope limit:** one reciter is one voice cluster. This settles the *mechanism* — hearing
versus reconstruction — which is what blocks #10. It does not show the answer generalizes
across voices.

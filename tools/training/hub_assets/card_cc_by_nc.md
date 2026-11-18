---
license: cc-by-nc-4.0
base_model: obadx/muaalem-model-v3_2
base_model_relation: finetune
language: ar
library_name: transformers
pipeline_tag: automatic-speech-recognition
datasets:
  - FaisaI/tadabur
  - tarteel-ai/tlog
tags:
  - quran
  - phoneme-recognition
  - ctc
  - wav2vec2-bert
  - distillation
---

# $repo_name

A size-distilled student of the **Muaalem** Quran phoneme-recognition model, built for
real-time, on-device Quran recitation-checking applications, as part of
[HifzGuide](https://github.com/sysofwan/HifzGuide). It replaces
[`sysofwan/hifzguide-muaalem-mini`](https://huggingface.co/sysofwan/hifzguide-muaalem-mini)
as the published student and is the model in HifzGuide asset release `v1.0.7`.

Same architecture and the same loading contract as `mini`; the weights are this release's
round-3 candidate, its **$weights_label** set, at step $step
(checkpoint sha256 `$checkpoint_sha256`). The
layout matches the previous repository file for file — `config.json`,
`preprocessor_config.json`, `phoneme_vocab.json` and the two vendored modules are
byte-identical, so `trust_remote_code=True` finds the same classes at the same names.

## Comparison

Teacher agreement is the distillation metric: student and teacher greedy CTC decodes compared
character-by-character on the same held-out clips. The 300-clip column was produced in one
pass over the same clips, with one protocol, for both students here; `training.hub_agreement`
in HifzGuide's `tools/` prints this table and the decode drift.

| | **mini-v2** (this) | [`mini` @ `a2bebf7`](https://huggingface.co/sysofwan/hifzguide-muaalem-mini/tree/a2bebf7c2e9a74d9ff5c3269594132d7caea7696) | [`$teacher_model_id`]($teacher_model_id_url) |
|---|---|---|---|
| Parameters | $param_count | $param_count | ~$teacher_param_count |
| `hidden_size` / layers / heads | $hidden_size / 24 / $num_attention_heads | $hidden_size / 24 / $num_attention_heads | 1024 / 24 / 16 |
| Positional encoding | rotary | rotary | `relative_key` |
| CTC heads | `phonemes` only ($num_phoneme_classes) | `phonemes` only ($num_phoneme_classes) | phonemes + 10 *sifat* attributes |
| Training objective | KL to the teacher's phoneme posteriors | the same | supervised on `obadx/muaalem-annotated-v3` |
| Training steps / peak lr | $step / 5e-5, warm-started from round 2's averaged weights | 40,000 / 1e-4, teacher-initialised | — |
| Training audio | `tarteel-ai/tlog` + `FaisaI/tadabur` | `FaisaI/tadabur` | `obadx/muaalem-annotated-v3` |
| Teacher agreement, 300 `gate_eval` dev clips | **94.690%** (exact 18.67%) | **94.707%** (exact 16.33%) | reference |
| Teacher agreement, its own published dev/test | not measured at that size for this release | 94.87% [94.47, 95.20] / 94.97% [94.63, 95.27] | — |
| All-cause false flags per 1,000 graded words, tlog confirm.3+4 | 98.7 | not measured | 95.5 |
| Licence | CC BY-NC 4.0 | AGPL-3.0 | MIT |

The `mini` column is that publication at Hub revision
[`a2bebf7c2e9a74d9ff5c3269594132d7caea7696`](https://huggingface.co/sysofwan/hifzguide-muaalem-mini/tree/a2bebf7c2e9a74d9ff5c3269594132d7caea7696)
(`model.safetensors` LFS sha256 `2570fd3780a8435a11eb20e336bac29ac6f4dedd7c367d39abc20627e6784bb9`);
download that revision to reproduce the row. Its own publication records no training checkpoint
hash, so that Hub artifact — not a checkpoint — is the identity behind these numbers. It is
**not** C95: all 772 tensors differ (see below).

The two students are within 0.017 pt of each other on the 300 clips (95% clustered CI
width ≈ 1 pt each), and their decodes differ in 2.37% of characters — a tie on teacher
agreement is expected, because both imitate the same teacher. What the weights change is
end-to-end behaviour in the app, below.

## What the weights change in the app

Muraja's shipped model before this release was C95, which is a different checkpoint from the
`mini` weights above (all 772 tensors differ). On the round-3 pooled gating set — 5,651
graded clips over 113 surah clusters, 56,807 graded words — this model against C95:

| cause | this model | C95 | difference (95% CI) |
|---|---|---|---|
| all-cause false flags / 1,000 words | 98.7 | 157.9 | −59.2 [−67.8, −51.7] |
| consonant-caused | 62.0 | 123.3 | −61.3 [−69.8, −53.4] |
| tashkeel-caused | 12.1 | 12.2 | −0.03 [−1.67, +1.38] |
| shaddah-caused (reported only) | 2.48 | 3.12 | −0.64 [−1.15, −0.10] |

Agreement with the teacher rose +12.9 pt on the same population (95% lower bound +11.9).
Catch loss: of 92 decisive flags this model removes that the owner then heard, 16 (17.4%,
Wilson 95% [11.0%, 26.4%]) were real mistakes — the point estimate passes the round's ≤ 20%
bar and the upper bound crosses it. The round had **no control arm**, so the −59.2 cannot
separate "more tlog data" from "10,000 more steps". Full record:
`docs/consonant-round3-readout.md` in the HifzGuide repository.

## Loading

The custom `Wav2Vec2BertForMultilevelCTC` architecture is vendored verbatim from
MIT-licensed [`obadx/quran-muaalem`](https://github.com/obadx/quran-muaalem).
Load with `trust_remote_code=True`; `phoneme_vocab.json` maps ids to characters.
Greedy decoding collapses repeats and drops blank/pad:

```python
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoFeatureExtractor

repo_id = "$repo_id"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

waveform = ...  # 16 kHz mono float32 numpy array of real audio
inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits["phonemes"]  # (1, num_frames, $num_phoneme_classes)

vocab_path = hf_hub_download(repo_id, "phoneme_vocab.json")
id_to_char = json.load(open(vocab_path))["id_to_char"]

ids, prev, decoded = logits.argmax(dim=-1)[0].tolist(), None, []
for i in ids:
    if i != prev and i != 0:  # collapse repeats, drop the CTC blank
        decoded.append(id_to_char[i])
    prev = i
print("".join(decoded))
```

`$feature_extractor_type` produces slightly fewer than $deployed_feature_frames frames
from 5 s of audio; the fixed `($deployed_feature_frames, $feature_input_dim)`
training/evaluation window comes from HifzGuide's own mel front-end
(`mel_filters.bin`/`window.bin`), not raw feature-extractor output.
The CoreML build of these weights (`MuaalemStudent_h448_6BIT.mlmodelc`, 6-bit palettized,
84.4 MB) ships as a GitHub release asset in
[`sysofwan/HifzGuide`](https://github.com/sysofwan/HifzGuide/releases); the PyTorch → CoreML
export was verified to move agreement by +0.016 pt and 0.365% of characters.

## Training

- **Objective:** frame-weighted KL divergence between student/teacher phoneme posteriors,
  up-weighting non-blank and confirmed-region frames. No CTC loss, hard-label/decode targets
  or transcripts: only teacher soft outputs (temperature 2.0, non-blank weight 3.0, confirm
  weight 2.0).
- **Warm start:** round 2's averaged weights, fresh optimiser and schedule; $step steps,
  batch 32, lr 5e-5, 500-step warmup, WSD with a 3,000-step cooldown, EMA decay 0.999.
- **Data:** 24 rows of `tarteel-ai/tlog` + 8 rows of `FaisaI/tadabur` per batch, from frames
  earlier rounds never read (row-level and shard-level disjointness asserted at launch).
- **Augmentation:** clean 0.30, additive noise 0.35 (3-20 dB SNR, 458 Freesound recordings),
  synthetic reverb 0.15 (RT60 0.2-1.0 s), 8 kHz band limit 0.10, Opus round-trip 0.10.
- **Seeds:** 2030 (shard order), 2031 (window shuffle); the checkpoint is the final step —
  validation was monitoring only and selected nothing.

## Limitations

- Trained to imitate the teacher, not independent ground-truth transcription; evaluated
  only against teacher decodes. Output approximates the teacher, not a validated standalone
  phoneme-recognition result.
- Evaluation covers the corpora's recording conditions (Tadabur re-reads, tlog recitations)
  plus simulated augmentation; real deployment microphones, other reciting styles and
  dialects are untested.
- One seed, no replicate. A difference against C95 below about 1-2 false flags per 1,000
  graded words is not interpretable, and the round had no control arm.
- The tashkeel guard that passed is loose (+3.5 per 1,000 against a paired SE near 1.07): it
  vetoes gross tashkeel regressions, not a +1.0 regression.
- Phoneme identity only: no *sifat*/tajweed-attribute heads or suitability for
  tajweed-attribute judgments.
- The single-chunk ANE budget is a demonstrated 99 MB ceiling, not a specification; a model
  over it fails silently to CPU. The previous asset release's build was confirmed to load on
  an iPhone 13; these weights' on-device load is not yet tested.
- Inherits teacher errors/biases; the teacher's model card documents no bias evaluation.

## License

Weights, configuration and this README are **CC BY-NC 4.0** (`LICENSE`), because the
distillation used `FaisaI/tadabur` (CC BY-NC 4.0). The vendored architecture and model
lineage from `$teacher_model_id` and `facebook/w2v-bert-2.0` retain their original
**MIT** terms. `NOTICE.md` carries the full upstream notices, the training-data attributions
(Tadabur, `tarteel-ai/tlog`, the Freesound noise pool) and the citations.

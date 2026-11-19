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

A **$param_count_millions-parameter Quran phoneme-recognition model** for non-commercial recitation-checking
applications and research. Distilled from the
[base model]($teacher_model_id_url), it produces phoneme tokens,
not ordinary Arabic text or correctness grades. This repository contains PyTorch
weights; CoreML exports are distributed separately through
[HifzGuide](https://github.com/sysofwan/HifzGuide/releases).

## Quick start

Requires PyTorch for your platform, `transformers==5.13.0`, `huggingface_hub`, and
`soundfile`. While the repository is private, authenticate with a Hugging Face account
that has access. Loading uses vendored custom code; review it before enabling
`trust_remote_code=True`.

```python
import json
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, AutoModel

repo = "$repo_id"
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()
extractor = AutoFeatureExtractor.from_pretrained(repo)

audio, sample_rate = sf.read("recitation.wav", dtype="float32")
if sample_rate != 16000 or audio.ndim != 1:
    raise ValueError("Provide 16 kHz mono audio")
inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")

with torch.inference_mode():
    logits = model(**inputs).logits["phonemes"]

vocab_file = hf_hub_download(repo, "phoneme_vocab.json")
vocab = json.loads(Path(vocab_file).read_text(encoding="utf-8"))["id_to_char"]
ids = logits.argmax(-1)[0].unique_consecutive().tolist()
print("".join(vocab[i] for i in ids if i != 0))
```

Use a short clip for this CPU example. The extractor converts mono float32 audio into
`input_features` shaped `(batch, frames, 160)`; the model does not accept raw audio.
Output is `logits["phonemes"]`, shaped `(batch, output_frames, $num_phoneme_classes)`.
`phoneme_vocab.json` defines the classes; id 0 is CTC blank/pad. Collapse consecutive
repeats before removing blank, as above.

## Results

Agreement with the base model compares phoneme strings, **not independent transcription
accuracy**. This and the previous version were evaluated on the same 300 held-out Tadabur
development recordings using streaming inference, unlike the short-clip example above.

| | This model (v2) | [Previous version (hifzguide-muaalem-mini)](https://huggingface.co/sysofwan/hifzguide-muaalem-mini) | [Base model](https://huggingface.co/obadx/muaalem-model-v3_2) |
| --- | --- | --- | --- |
| Parameters | $param_count_millions | $param_count_millions | $teacher_param_count_millions |
| Agreement with base model | 94.69% | 94.71% | Reference |
| False flags per 1,000 correctly recited words | 78.1 | 167.7 | 73.6 |

A false flag is a word a checker marks as a mistake in a recitation that is in fact correct.
The false-flag row is measured on held-out tlog recordings judged correctly recited, over the
43,840 words all three models graded, so each is scored on the same words: the previous
version flags 89.6 more words per 1,000 than this model (95% confidence interval 80.2 to 101.8
more, clustered by surah). Agreement alone says nothing about error detection; the false-flag
row is the measured comparison.

The row counts false alarms on correct recitation only. It says nothing about how many real
mistakes either version catches, so it does not establish better recitation-error detection
than the previous version.

## Intended use and limits

Use as a phoneme recognizer within a recitation-checking pipeline; alignment and grading
are separate. It has no sifat/tajweed-attribute heads and is not a standalone tajweed
evaluator. It imitates the base model and inherits its errors. Validate your users,
microphones, and recording conditions; generalization is not established. Results come
from one training seed, without a replicate. Python inference does not establish
on-device latency or ANE residency.

## Licence and provenance

Weights, configuration, and this card are **[CC BY-NC 4.0](LICENSE)**:
attribution required, non-commercial use only. Vendored architecture code retains **MIT**
terms.

Training used [Tadabur by Faisal Alherran](https://huggingface.co/datasets/FaisaI/tadabur)
(CC BY-NC 4.0), [tlog by Tarteel AI](https://huggingface.co/datasets/tarteel-ai/tlog)
(CC BY 4.0), and Freesound noise augmentation. Full upstream notices and noise-recording
credits are in [NOTICE.md](NOTICE.md).

[Release details](https://github.com/sysofwan/HifzGuide/releases/tag/v1.0.7)
and the [technical report](https://github.com/sysofwan/HifzGuide/blob/main/docs/release-v1.0.7.md)
provide evaluation and export details.

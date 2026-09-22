"""Export a distillation checkpoint to a Hugging Face model repo.

Loads a ``training.distill_train`` checkpoint's ``student`` weights into the
matching preset from :mod:`training.distill_student`, verifies the frozen
``(1, 250, 160) -> (1, 125, 43)`` shape contract, writes a self-contained HF
model folder (config, safetensors weights, feature extractor, phoneme vocab,
model card, license notices, and the vendored ``modeling``/``configuration``
source so ``trust_remote_code=True`` works with no other file needed), and
pushes it to the Hub.

Usage::

    python -m training.publish_hf \\
        --preset h448 \\
        --checkpoint runs/h448_stream/checkpoint.pt \\
        --repo-id sysofwan/hifzguide-muaalem-mini \\
        --private

Runs on Linux + CUDA (or CPU) with ``tools/requirements-train.txt`` installed;
the Hub push additionally needs an authenticated ``hf auth login``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import SeamlessM4TFeatureExtractor

from tadabur.muaalem import Wav2Vec2BertForMultilevelCTC, Wav2Vec2BertForMultilevelCTCConfig
from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR
from training.distill_student import (
    DEPLOYED_FEATURE_FRAMES,
    DEPLOYED_LOGIT_FRAMES,
    FEATURE_INPUT_DIM,
    NUM_PHONEME_CLASSES,
    PRESETS,
    build_student_config,
)

TEACHER_MODEL_ID = "obadx/muaalem-model-v3_2"
TEACHER_MODEL_URL = f"https://huggingface.co/{TEACHER_MODEL_ID}"
REPO_LICENSE_PATH = Path(__file__).parent.parent.parent / "LICENSE"

TEACHER_PARAM_COUNT = 605_754_251  # measured: obadx/muaalem-model-v3_2, all 11 heads


def repo_name(repo_id: str) -> str:
    return repo_id.split("/")[-1]

MODEL_CARD_TEMPLATE = """\
---
license: agpl-3.0
base_model: {teacher_model_id}
base_model_relation: finetune
language: ar
library_name: transformers
pipeline_tag: automatic-speech-recognition
datasets:
  - FaisaI/tadabur
tags:
  - quran
  - phoneme-recognition
  - ctc
  - wav2vec2-bert
  - distillation
---

# {repo_name}

A size-distilled, teacher-initialised student of the **Muaalem** Quran phoneme-recognition
model, built for real-time, on-device Quran recitation-checking applications, as part of
[HifzGuide](https://github.com/sysofwan/HifzGuide).

| Architecture | Student | Teacher: [`{teacher_model_id}`]({teacher_model_id_url}) |
|---|---|---|
| Backbone | Wav2Vec2-BERT | Wav2Vec2-BERT |
| Parameters | **{param_count:,}** | ~{teacher_param_count:,} |
| Encoder layers | 24 | 24 |
| `hidden_size` | {hidden_size} | 1024 |
| CTC heads | `phonemes` only | Phoneme identity + 10 *sifat* attributes |
| Positional encoding | Rotary | `relative_key` |

The student reduces width only: `intermediate_size={intermediate_size}`,
`num_attention_heads={num_attention_heads}`. It decodes only `phonemes`:
**{num_phoneme_classes} CTC classes**, with id 0 as blank/pad. The teacher's one-layer,
stride-2 adapter topology preserves the frame-rate contract
`({deployed_feature_frames}, {feature_input_dim}) -> ({deployed_logit_frames}, {num_phoneme_classes})`;
adapter weights necessarily differ with hidden size.

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

repo_id = "{repo_id}"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

waveform = ...  # 16 kHz mono float32 numpy array of real audio
inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits["phonemes"]  # (1, num_frames, {num_phoneme_classes})

vocab_path = hf_hub_download(repo_id, "phoneme_vocab.json")
id_to_char = json.load(open(vocab_path))["id_to_char"]

ids, prev, decoded = logits.argmax(dim=-1)[0].tolist(), None, []
for i in ids:
    if i != prev and i != 0:  # collapse repeats, drop the CTC blank
        decoded.append(id_to_char[i])
    prev = i
print("".join(decoded))
```

`{feature_extractor_type}` produces slightly fewer than {deployed_feature_frames} frames
from 5 s of audio; the fixed `({deployed_feature_frames}, {feature_input_dim})`
training/evaluation window comes from HifzGuide's own mel front-end
(`mel_filters.bin`/`window.bin`), not raw feature-extractor output.

CoreML export was numerically verified (bit-exact trace); the single-chunk Apple Neural
Engine load path was confirmed working on-device at deployment.

## Accuracy: teacher decode agreement

Teacher/student greedy CTC decodes are compared character-by-character, reporting pooled
accuracy with reciter-clustered 95% CIs (not scored against an independent reference
transcript).

| Split | Accuracy | 95% CI | Clips | Reciters |
|---|---|---|---|---|
| dev | **94.87%** | [94.47, 95.20] | 970 | 146 |
| test | **94.97%** | [94.63, 95.27] | 1,030 | 140 |

## License

Weights, configuration and this README are **AGPL-3.0** (`LICENSE`), matching HifzGuide.
The vendored architecture and model lineage from `{teacher_model_id}` and
`facebook/w2v-bert-2.0` retain their original **MIT** terms, which permit relicensing
the combined work.

Training audio: [`FaisaI/tadabur`](https://huggingface.co/datasets/FaisaI/tadabur),
**CC BY-NC 4.0** (research/educational use; attribution required). AGPL-3.0 governs this
repository's own contents, not the corpus. Before commercial use of these weights,
independently assess how the corpus's non-commercial term applies.

## Training

- **Objective:** frame-weighted KL divergence between student/teacher phoneme posteriors,
  up-weighting non-blank and confirmed-region frames. No CTC loss, hard-label/decode
  targets or transcripts: only teacher soft outputs.
- **Recipe:** teacher-initialised (`teacher_init --qk damp`); 40,000 streamed optimizer
  steps; batch 32; WSD schedule (1,000-step warmup / hold / 4,000-step cooldown);
  peak lr 1e-4; EMA decay 0.999. ~11.5 h on one RTX 5060 Ti (16 GB).
- **Architecture choices:** training-free depth reduction destroyed the backbone, so
  all 24 layers remain. Rotary is deliberate: students are trained from random init
  and need not copy the teacher's positional scheme; `relative_key` embeds a large
  per-layer position constant in traced graphs, mainly costing on-device export,
  not accuracy.

## Limitations

- Trained to imitate the teacher, not independent ground-truth transcription; evaluated
  only against teacher decodes. Output approximates the teacher, not a validated
  standalone phoneme-recognition result.
- Evaluation covers only `FaisaI/tadabur`-distribution audio; other recording conditions,
  reciting styles and dialects are untested.
- Phoneme identity only: no *sifat*/tajweed-attribute heads or suitability for
  tajweed-attribute judgments.
- Evaluation uses one fixed window position. Teacher self-agreement varies by tens
  of points across positions: headline accuracy compares checkpoints, not a bound
  on real-world transcription accuracy.
- Inherits teacher errors/biases; the teacher's model card documents no bias evaluation.

## Notices and citations

See `NOTICE.md` for full upstream notices and the Muaalem paper and Tadabur dataset citations.
"""

NOTICE_TEMPLATE = """\
# Third-party notices

Copyright (C) 2026 Sayyid Sofwan Haddad. This repository's weights, configuration,
`README.md`, and this notice are licensed under AGPL-3.0 (see `LICENSE`).

## Vendored code (MIT)

`modeling_multi_level_ctc.py` and `configuration_multi_level_ctc.py` are vendored
**verbatim** from [`obadx/quran-muaalem`](https://github.com/obadx/quran-muaalem),
commit `e9e692c87667ea6353486b2429bfcbaf32670cbe` (`src/quran_muaalem/modeling/`),
under the following license:

> MIT License
>
> Copyright (c) 2025 Abdullah
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The configuration class's docstring is additionally derived from Hugging Face
`transformers`' `Wav2Vec2BertConfig` docstring (Apache License 2.0,
<https://github.com/huggingface/transformers>).

## Model lineage (MIT)

This model's architecture and initial weights derive from
[`{teacher_model_id}`]({teacher_model_id_url}) (MIT, Copyright (c) 2025 Abdullah) and
its backbone [`facebook/w2v-bert-2.0`](https://huggingface.co/facebook/w2v-bert-2.0)
(MIT, Copyright (c) Meta Platforms, Inc.).

The Muaalem architecture and data pipeline are described in:

```bibtex
@article{{abdelfttah2025quran,
  title   = {{Automatic Pronunciation Error Detection and Correction of the Holy
             Quran's Learners Using Deep Learning}},
  author  = {{Abdelfttah, Abdullah and Khalil, Mahmoud I. and Abbas, Hazem}},
  journal = {{arXiv preprint arXiv:2509.00094}},
  year    = {{2025}}
}}
```

## Training data (CC BY-NC 4.0)

Distillation used audio from
[`FaisaI/tadabur`](https://huggingface.co/datasets/FaisaI/tadabur), licensed
**CC BY-NC 4.0** (<https://creativecommons.org/licenses/by-nc/4.0/>) — research and
educational use only, attribution required. Cite as:

```bibtex
@misc{{alherran2026tadabur,
  author        = {{Alherran, Faisal}},
  title         = {{Tadabur: A Large-Scale Quran Audio Dataset}},
  year          = {{2026}},
  eprint        = {{2604.18932}},
  archivePrefix = {{arXiv}},
  primaryClass  = {{cs.SD}},
  doi           = {{10.48550/arXiv.2604.18932}},
  url           = {{https://arxiv.org/abs/2604.18932}}
}}
```
"""


def export_and_publish(
    preset_name: str,
    checkpoint_path: Path,
    out_dir: Path,
    repo_id: str,
    private: bool,
    push: bool,
) -> None:
    spec = PRESETS[preset_name]
    config = build_student_config(spec)
    model = Wav2Vec2BertForMultilevelCTC(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["student"], strict=True)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        probe = torch.zeros(1, DEPLOYED_FEATURE_FRAMES, FEATURE_INPUT_DIM)
        logits = model(input_features=probe).logits["phonemes"]
    expected_shape = (1, DEPLOYED_LOGIT_FRAMES, NUM_PHONEME_CLASSES)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"shape contract violated: got {tuple(logits.shape)}, want {expected_shape}")

    config.id2label = {i: ch for i, ch in enumerate(PHONEME_ID_TO_CHAR)}
    config.label2id = {ch: i for i, ch in enumerate(PHONEME_ID_TO_CHAR)}

    out_dir.mkdir(parents=True, exist_ok=True)

    Wav2Vec2BertForMultilevelCTCConfig.register_for_auto_class()
    Wav2Vec2BertForMultilevelCTC.register_for_auto_class("AutoModel")
    model.save_pretrained(out_dir, safe_serialization=True)

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(TEACHER_MODEL_ID)
    feature_extractor.save_pretrained(out_dir)

    with (out_dir / "phoneme_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"id_to_char": list(PHONEME_ID_TO_CHAR), "pad_id": 0, "pad_token": "[PAD]"},
            f,
            ensure_ascii=False,
            indent=2,
        )

    (out_dir / "README.md").write_text(
        MODEL_CARD_TEMPLATE.format(
            repo_name=repo_name(repo_id),
            repo_id=repo_id,
            teacher_model_id=TEACHER_MODEL_ID,
            teacher_model_id_url=TEACHER_MODEL_URL,
            teacher_param_count=TEACHER_PARAM_COUNT,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_attention_heads=spec.num_attention_heads,
            num_phoneme_classes=NUM_PHONEME_CLASSES,
            deployed_feature_frames=DEPLOYED_FEATURE_FRAMES,
            deployed_logit_frames=DEPLOYED_LOGIT_FRAMES,
            feature_input_dim=FEATURE_INPUT_DIM,
            feature_extractor_type=type(feature_extractor).__name__,
            param_count=param_count,
        ),
        encoding="utf-8",
    )
    (out_dir / "NOTICE.md").write_text(
        NOTICE_TEMPLATE.format(
            teacher_model_id=TEACHER_MODEL_ID,
            teacher_model_id_url=TEACHER_MODEL_URL,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(REPO_LICENSE_PATH, out_dir / "LICENSE")

    print(f"Exported {param_count:,} params to {out_dir}")
    print(f"Shape contract OK: {tuple(logits.shape)}")

    if not push:
        print("--push not set; skipping Hub upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model")
    print(f"Pushed to https://huggingface.co/{repo_id} (private={private})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true", help="Actually upload to the Hub.")
    args = parser.parse_args()

    out_dir = args.out_dir or Path(__file__).parent.parent / "export" / repo_name(args.repo_id)
    export_and_publish(
        preset_name=args.preset,
        checkpoint_path=args.checkpoint,
        out_dir=out_dir,
        repo_id=args.repo_id,
        private=args.private,
        push=args.push,
    )


if __name__ == "__main__":
    main()

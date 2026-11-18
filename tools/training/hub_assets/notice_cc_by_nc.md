# Third-party notices

Copyright (C) 2026 Sayyid Sofwan Haddad. This repository's weights, configuration,
`README.md`, and this notice are licensed under **CC BY-NC 4.0** (see `LICENSE`). The
non-commercial term comes from the training corpus; commercial licensing of models derived
from it is the dataset owner's to grant.

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

This model's architecture and training objective derive from
[`$teacher_model_id`]($teacher_model_id_url) (MIT, Copyright (c) 2025 Abdullah) and
its backbone [`facebook/w2v-bert-2.0`](https://huggingface.co/facebook/w2v-bert-2.0)
(MIT, Copyright (c) Meta Platforms, Inc.). These weights are a distillation of that
teacher's phoneme head: 116,318,635 parameters against its ~605,754,251, trained against its
soft outputs only.

The Muaalem architecture and data pipeline are described in:

```bibtex
@article{abdelfttah2025quran,
  title   = {Automatic Pronunciation Error Detection and Correction of the Holy
             Quran's Learners Using Deep Learning},
  author  = {Abdelfttah, Abdullah and Khalil, Mahmoud I. and Abbas, Hazem},
  journal = {arXiv preprint arXiv:2509.00094},
  year    = {2025}
}
```

## Training data

Distillation used Quran recitation audio from two corpora, plus a noise pool for augmentation.
No training audio is redistributed in this repository.

**[`FaisaI/tadabur`](https://huggingface.co/datasets/FaisaI/tadabur)** — **CC BY-NC 4.0**
(<https://creativecommons.org/licenses/by-nc/4.0/>), research and educational use only,
attribution required. Cite as:

```bibtex
@misc{alherran2026tadabur,
  author        = {Alherran, Faisal},
  title         = {Tadabur: A Large-Scale Quran Audio Dataset},
  year          = {2026},
  eprint        = {2604.18932},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  doi           = {10.48550/arXiv.2604.18932},
  url           = {https://arxiv.org/abs/2604.18932}
}
```

**[`tarteel-ai/tlog`](https://huggingface.co/datasets/tarteel-ai/tlog)** — **CC BY 4.0**
(<https://creativecommons.org/licenses/by/4.0/>), by Tarteel AI
(<https://tarteel.ai/>). The audio is not redistributed here; this project records only the
row keys it read, at revision `9fe4f7e6d1b372c2c11157d6d2f0f91ca344eb51`.

**Noise pool** — 458 recordings (4.51 h) from Freesound, distributed as
[`obadx/freesound-commercial-50k-noise-only`](https://huggingface.co/datasets/obadx/freesound-commercial-50k-noise-only)
at revision `a5f48ab83b64ba1d35ee4ec363dae75d4e2c69d3`, used for additive noise at 3-20 dB
SNR. Each recording carries its own licence — 279 CC0-1.0, 87 CC BY 3.0, 86 CC BY 4.0, 6
CC Sampling+ — and per-recording attribution (Freesound id, uploader, licence) is recorded in
the HifzGuide repository at `tools/tadabur/augmentation/noise_pool.json`.

"""Export a distillation checkpoint to a Hugging Face model repo.

The one writer of the hub layout: it produced the published
``sysofwan/hifzguide-muaalem-mini`` and every later publication of the same architecture
(the v1.0.7 model's ``sysofwan/hifzguide-muaalem-mini-v2`` included). It writes a
self-contained HF model folder -- config, safetensors weights, feature extractor, phoneme
vocab, model card, licence notices, and the vendored ``modeling``/``configuration`` source so
``trust_remote_code=True`` works with no other file needed -- and optionally pushes it.

The weight set is a scored decision, not this tool's: every round's ``choice.json`` records
``weights: ema``, so ``--ema`` reads the average through the loader every eval tool uses
(:func:`training.decoding.load_student_from_checkpoint`) rather than ``checkpoint["student"]``.
That loader raises when a checkpoint has no average, so an ``--ema`` publication cannot
silently ship the live weights.

The frozen ``(1, 250, 160) -> (1, 125, 43)`` shape contract is checked on the loaded model
before anything is written, and the stage is atomic: files are written to a sibling temporary
directory and renamed, so a failure leaves no partial directory the next run would refuse.
``--teacher-revision`` defaults to the revision the checkpoint's run config records, and any
other value is refused -- a feature extractor saved from a different teacher revision would
describe preprocessing the checkpoint never trained against.

Usage::

    python -m training.publish_hf \\
        --checkpoint runs/consonant3_candidate/checkpoint.pt --ema \\
        --repo-id sysofwan/hifzguide-muaalem-mini-v2 --profile cc-by-nc \\
        --out-dir ../hf-stage/hifzguide-muaalem-mini-v2

``--profile agpl`` (the default) writes the AGPL-3.0 card and notice of the ``mini``
publication. ``--profile cc-by-nc`` writes the CC BY-NC 4.0 card and notice of the v1.0.7
publication -- the licence a student distilled from the Tadabur corpus must carry, attached to
the one checkpoint whose comparison and readout that card reports; any other checkpoint (or
the live weights of that one) is refused, because the prose would then attribute another
artifact's measured results to these weights. Neither profile pushes unless ``--push`` is set;
without it the directory is a draft.

A card is prose about a stage, so correcting it takes the stage, not the checkpoint::

    python -m training.publish_hf \\
        --from-stage <the v1.0.7 stage> --repo-id sysofwan/hifzguide-muaalem-mini-v2 \\
        --out-dir ../hf-stage/hifzguide-muaalem-mini-v2-restaged

That path copies the stage file for file and re-renders only ``README.md``/``NOTICE.md`` from
the weight file's own header, and refuses a stage whose ``model.safetensors`` is not the one
the card's results were measured on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import string
import struct
import sys
from pathlib import Path

from transformers import SeamlessM4TFeatureExtractor

from tadabur.muaalem import Wav2Vec2BertForMultilevelCTC, Wav2Vec2BertForMultilevelCTCConfig
from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR
from training.distill_student import (
    DEPLOYED_FEATURE_FRAMES,
    DEPLOYED_LOGIT_FRAMES,
    FEATURE_INPUT_DIM,
    NUM_PHONEME_CLASSES,
    PRESETS,
    verify_shape_contract,
)

TEACHER_MODEL_ID = "obadx/muaalem-model-v3_2"
TEACHER_MODEL_URL = f"https://huggingface.co/{TEACHER_MODEL_ID}"
REPO_LICENSE_PATH = Path(__file__).parent.parent.parent / "LICENSE"
# The CC BY-NC 4.0 legal code, committed so a stage is reproducible offline and does not
# depend on what creativecommons.org serves today. `gitattributes` is the published
# repository's LFS rule set, kept beside it so the layout writer emits every file the Hub
# repository carries.
HUB_ASSETS = Path(__file__).parent / "hub_assets"
CC_BY_NC_LICENSE_PATH = HUB_ASSETS / "cc-by-nc-4.0.txt"
GITATTRIBUTES_PATH = HUB_ASSETS / "gitattributes"

TEACHER_PARAM_COUNT = 605_754_251  # measured: obadx/muaalem-model-v3_2, all 11 heads


def repo_name(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def sha256_file(path: Path) -> str:
    """The checkpoint's identity, recorded in the card for the weights it describes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def note(message: str) -> None:
    """Progress goes to stderr: ``--json`` writes the only thing that belongs on stdout."""
    print(message, file=sys.stderr, flush=True)

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


# --- The CC BY-NC 4.0 publication -------------------------------------------------------
#
# A student distilled from `FaisaI/tadabur` cannot be relicensed AGPL-3.0: the corpus's
# non-commercial term travels with the weights. Its card and notice are data files beside this
# module (`hub_assets/`), where the prose is reviewable as prose; they are rendered with
# `string.Template` so braces in the citations need no escaping. The identity fields are the
# only things this tool decides -- the measured tables in the card are the release's readings,
# and `training.hub_agreement` reproduces them.
CARD_CC_BY_NC = string.Template((HUB_ASSETS / "card_cc_by_nc.md").read_text(encoding="utf-8"))
NOTICE_CC_BY_NC = string.Template(
    (HUB_ASSETS / "notice_cc_by_nc.md").read_text(encoding="utf-8")
)


def _as_is(fields: dict) -> dict:
    return fields


def _formatted(fields: dict) -> dict:
    """The million-parameter figures the release card prints, from the exact counts."""
    return {
        **fields,
        "param_count_millions": f"{fields['param_count'] / 1_000_000:.0f}M",
        "teacher_param_count_millions": f"{fields['teacher_param_count'] / 1_000_000:.0f}M",
    }


def _render(template, fields: dict) -> str:
    return template.format(**fields) if isinstance(template, str) else template.substitute(fields)


# `cc-by-nc` is one release's card, not a generic licence template: it names the round-3
# candidate, its warm start, its corpora and its measured comparison and readout. Those
# claims are true of exactly one weight set, so the profile pins that identity. Choosing a
# licence is a separate statement from claiming these numbers -- a different CC BY-NC
# checkpoint needs its own card, not this one with a different hash in the identity line.
RELEASE_CANDIDATE = {
    "checkpoint_sha256": "1193e2117b56381ccc078a5b5e21d61d0ad0903565dcca0cb110b39ce24ed4a9",
    "weights": "ema",
}
#: The stage those weights were written to, as the v1.0.7 record lists its ``model.safetensors``.
#: A revised card needs no checkpoint, so the weight file itself is the identity its measured
#: claims may be re-rendered onto.
RELEASE_STAGE_SAFETENSORS_SHA256 = (
    "7b760511334e53ca115a2dd466e57a9241c4e3c7a30498916b721d0f6d650d89"
)


def require_release_candidate(checkpoint_sha256: str, weights: str) -> None:
    """Refuse to attach one release's measured claims to another weight set."""
    actual = {"checkpoint_sha256": checkpoint_sha256, "weights": weights}
    mismatches = [
        f"  {key}: got {actual[key]!r}, card describes {expected!r}"
        for key, expected in RELEASE_CANDIDATE.items()
        if actual[key] != expected
    ]
    if mismatches:
        raise SystemExit(
            "the cc-by-nc card publishes the round-3 candidate's comparison table, training "
            "history and readout; those results belong to one checkpoint and its ema weights "
            "only.\n"
            + "\n".join(mismatches)
            + "\nStage that artifact (--ema with the checkpoint the readout scored), or write "
            "the card for this weight set instead of reusing one that measured another."
        )


def require_release_stage(safetensors_sha256: str) -> None:
    """Refuse to re-render the release card onto another stage's weights."""
    if safetensors_sha256 != RELEASE_STAGE_SAFETENSORS_SHA256:
        raise SystemExit(
            "the cc-by-nc card reports one weight file's measured results; this stage's "
            f"model.safetensors is {safetensors_sha256[:12]}…, not the release's "
            f"({RELEASE_STAGE_SAFETENSORS_SHA256[:12]}…). Re-render the card of the stage "
            "those results were measured on."
        )


PROFILES = {
    "agpl": {
        "card": MODEL_CARD_TEMPLATE,
        "notice": NOTICE_TEMPLATE,
        "license": REPO_LICENSE_PATH,
        "values": _as_is,
        "require": None,
    },
    "cc-by-nc": {
        "card": CARD_CC_BY_NC,
        "notice": NOTICE_CC_BY_NC,
        "license": CC_BY_NC_LICENSE_PATH,
        "values": _formatted,
        "require": require_release_candidate,
    },
}


def resolve_teacher_revision(run_config: dict, given: str | None) -> str:
    """The teacher revision the feature extractor is saved from, pinned to the run's own.

    A ``preprocessor_config.json`` is a statement about the preprocessing the weights were
    trained on. Fetching it from the teacher's default branch, or from a revision other than
    the one the run recorded, publishes a preprocessing contract the checkpoint was never
    trained against, and nothing downstream would notice.
    """
    recorded = run_config.get("teacher_revision")
    if given and recorded and given != recorded:
        raise SystemExit(
            f"--teacher-revision {given!r} is not the revision the run trained against "
            f"({recorded!r}); the saved feature extractor would describe preprocessing the "
            f"checkpoint never saw."
        )
    revision = given or recorded
    if not revision:
        raise SystemExit(
            "the checkpoint records no teacher_revision, so the feature extractor cannot be "
            "pinned to the run; pass --teacher-revision <revision>."
        )
    return revision


def load_candidate(checkpoint_path: Path, use_ema: bool):
    """The student a publication carries, through the loader every eval tool uses.

    ``use_ema`` selects the averaged weights (:func:`training.decoding.averaged_state`) and
    raises rather than falling back when the checkpoint has none, so ``--ema`` can never
    publish the live weights under the averaged name.
    """
    from training.decoding import load_student_from_checkpoint

    model, run_config, step = load_student_from_checkpoint(
        checkpoint_path, "cpu", use_ema=use_ema
    )
    note(
        f"Loaded {run_config['preset']} from {checkpoint_path} @ step {step} "
        f"({'ema' if use_ema else 'live'})"
    )
    return model, run_config, step


def prepare_staging(out_dir: Path) -> Path:
    """Refuse a populated target and return the sibling directory to write into.

    Staging then renaming keeps the publication atomic: a Hub fetch or a disk error halfway
    through leaves a ``.tmp`` directory, not a half-written stage that the non-empty guard
    would refuse to overwrite on the retry.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"{out_dir} is not empty: a stage is written once, from one checkpoint, so a merge "
            f"into an older stage would mix two weight sets. Point --out-dir at a new directory."
        )
    staging = out_dir.with_name(out_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    return staging


def push_stage(out_dir: Path, repo_id: str, *, private: bool, push: bool) -> None:
    """Upload a finished stage, or say it stayed local. Only ``--push`` reaches the hub."""
    if not push:
        note("--push not set; the stage is a draft and nothing was uploaded.")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model")
    note(f"Pushed to https://huggingface.co/{repo_id} (private={private})")


def export_and_publish(
    checkpoint_path: Path,
    out_dir: Path,
    repo_id: str,
    profile: str = "agpl",
    use_ema: bool = False,
    teacher_revision: str | None = None,
    private: bool = False,
    push: bool = False,
) -> dict:
    """Write the hub layout for one checkpoint and return its record.

    The record is what a release note quotes: the file checksums, the shape contract the model
    was probed against, and the weight set and teacher revision the stage was written from.
    """
    model, run_config, step = load_candidate(checkpoint_path, use_ema)
    weights = "ema" if use_ema else "live"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    chosen = PROFILES[profile]
    # Before anything is written: a card's claims are about one weight identity, not about
    # whichever checkpoint happens to be staged with its licence.
    if chosen["require"] is not None:
        chosen["require"](checkpoint_sha256, weights)
    spec = PRESETS[run_config["preset"]]
    param_count = sum(p.numel() for p in model.parameters())

    frames, classes = verify_shape_contract(model)
    note(f"Shape contract OK: (1, {frames}, {classes})")
    revision = resolve_teacher_revision(run_config, teacher_revision)

    model.config.id2label = {i: ch for i, ch in enumerate(PHONEME_ID_TO_CHAR)}
    model.config.label2id = {ch: i for i, ch in enumerate(PHONEME_ID_TO_CHAR)}

    staging = prepare_staging(out_dir)
    staging.mkdir(parents=True)

    Wav2Vec2BertForMultilevelCTCConfig.register_for_auto_class()
    Wav2Vec2BertForMultilevelCTC.register_for_auto_class("AutoModel")
    model.save_pretrained(staging, safe_serialization=True)

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        TEACHER_MODEL_ID, revision=revision
    )
    feature_extractor.save_pretrained(staging)

    (staging / "phoneme_vocab.json").write_text(
        json.dumps(
            {"id_to_char": list(PHONEME_ID_TO_CHAR), "pad_id": 0, "pad_token": "[PAD]"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fields = {
        "repo_name": repo_name(repo_id),
        "repo_id": repo_id,
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_model_id_url": TEACHER_MODEL_URL,
        "teacher_param_count": TEACHER_PARAM_COUNT,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_phoneme_classes": NUM_PHONEME_CLASSES,
        "deployed_feature_frames": DEPLOYED_FEATURE_FRAMES,
        "deployed_logit_frames": DEPLOYED_LOGIT_FRAMES,
        "feature_input_dim": FEATURE_INPUT_DIM,
        "feature_extractor_type": type(feature_extractor).__name__,
        "param_count": param_count,
        "step": step,
        "weights": weights,
        "checkpoint_sha256": checkpoint_sha256,
        "augmentation": (run_config.get("augment") or {}).get("version"),
    }
    values = chosen["values"](fields)
    (staging / "README.md").write_text(_render(chosen["card"], values), encoding="utf-8")
    (staging / "NOTICE.md").write_text(_render(chosen["notice"], values), encoding="utf-8")
    shutil.copyfile(chosen["license"], staging / "LICENSE")
    shutil.copyfile(GITATTRIBUTES_PATH, staging / ".gitattributes")

    if out_dir.exists():
        out_dir.rmdir()  # empty by the guard in prepare_staging
    staging.rename(out_dir)

    checksums = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir())}
    note(f"Exported {param_count:,} params to {out_dir}")
    note(f"Teacher revision {revision}; profile {profile}")
    push_stage(out_dir, repo_id, private=private, push=push)

    return {
        "output_dir": str(out_dir),
        "repo_id": repo_id,
        "profile": profile,
        "weights": weights,
        "step": step,
        "checkpoint_sha256": fields["checkpoint_sha256"],
        "teacher_revision": revision,
        "shape": [frames, classes],
        "num_params": param_count,
        "sha256": checksums,
    }


def safetensors_param_count(path: Path) -> int:
    """A stage's parameter count from its ``model.safetensors`` header.

    Header only -- an 8-byte length and the tensor table -- so re-rendering a card does not
    reach for the checkpoint the stage was written from.
    """
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return sum(
        math.prod(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
    )


def render_release_card(
    stage_dir: Path, out_dir: Path, repo_id: str, *, private: bool = False, push: bool = False
) -> dict:
    """Copy a release stage and re-render its card and notice.

    A card is prose about a weight set, so correcting it must not reach for the checkpoint:
    the stage is the artifact the Hub holds, and the copy carries every file byte for byte,
    with ``README.md``/``NOTICE.md`` re-rendered from the stage's own parameter count and the
    profile's constants. The release identity is checked before anything is written, and the
    copy is staged under a sibling temporary directory like a first export.
    """
    require_release_stage(sha256_file(stage_dir / "model.safetensors"))
    fields = _formatted(
        {
            "repo_name": repo_name(repo_id),
            "repo_id": repo_id,
            "teacher_model_id": TEACHER_MODEL_ID,
            "teacher_model_id_url": TEACHER_MODEL_URL,
            "teacher_param_count": TEACHER_PARAM_COUNT,
            "num_phoneme_classes": NUM_PHONEME_CLASSES,
            "param_count": safetensors_param_count(stage_dir / "model.safetensors"),
        }
    )
    staging = prepare_staging(out_dir)
    shutil.copytree(stage_dir, staging)
    (staging / "README.md").write_text(_render(CARD_CC_BY_NC, fields), encoding="utf-8")
    (staging / "NOTICE.md").write_text(_render(NOTICE_CC_BY_NC, fields), encoding="utf-8")
    if out_dir.exists():
        out_dir.rmdir()
    staging.rename(out_dir)

    checksums = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir())}
    note(f"Re-rendered the cc-by-nc card of {stage_dir} into {out_dir}")
    push_stage(out_dir, repo_id, private=private, push=push)
    return {
        "output_dir": str(out_dir),
        "restaged_from": str(stage_dir),
        "repo_id": repo_id,
        "profile": "cc-by-nc",
        "num_params": fields["param_count"],
        "sha256": checksums,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--from-stage",
        type=Path,
        help="re-render this release stage's card and notice into --out-dir, copying its "
        "files unchanged; no checkpoint or hub access",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="publish the averaged weights a --ema-decay run stored beside the live ones",
    )
    parser.add_argument(
        "--teacher-revision",
        help="defaults to the revision the checkpoint's run config records; any other is refused",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=None,
        help="which card/notice/licence set: 'agpl' is the mini publication, 'cc-by-nc' is "
        "the v1.0.7 release, whose card reports one checkpoint's results and is refused for "
        "any other weights. Defaults to 'agpl', or 'cc-by-nc' with --from-stage",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true", help="Actually upload to the Hub.")
    parser.add_argument("--json", action="store_true", help="print the stage record as JSON")
    args = parser.parse_args()

    out_dir = args.out_dir or Path(__file__).parent.parent / "export" / repo_name(args.repo_id)
    if args.from_stage is not None:
        if args.ema or args.teacher_revision or args.profile not in (None, "cc-by-nc"):
            raise SystemExit(
                "--from-stage re-renders an existing stage's cc-by-nc card: --ema and "
                "--teacher-revision describe a checkpoint, and --profile cannot be other than "
                "cc-by-nc"
            )
        record = render_release_card(
            stage_dir=args.from_stage,
            out_dir=out_dir,
            repo_id=args.repo_id,
            private=args.private,
            push=args.push,
        )
    else:
        record = export_and_publish(
            checkpoint_path=args.checkpoint,
            out_dir=out_dir,
            repo_id=args.repo_id,
            profile=args.profile or "agpl",
            use_ema=args.ema,
            teacher_revision=args.teacher_revision,
            private=args.private,
            push=args.push,
        )
    if args.json:
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

"""Muaalem PyTorch phoneme inference — the shared engine for the Tadabur filter.

Loads the Muaalem model (``obadx/muaalem-model-v3_2``, the vendored
``Wav2Vec2BertForMultilevelCTC``) in bf16 on the CUDA GPU, feature-extracts a
16 kHz mono waveform with the model's own ``SeamlessM4TFeatureExtractor`` (so
train/inference preprocessing stays identical), runs one variable-length forward
pass, and greedy-CTC-decodes the **phoneme head** into a phoneme string.

Deliberately variable-length: the whole ayah is processed in a single pass with a
real attention mask — the 5 s / 250-frame windowing in ``convert_to_coreml.py`` is
a CoreML/ANE static-shape constraint that does not apply to PyTorch inference.

The sifat heads are computed by the model but ignored here; only the phoneme head
feeds the ``.balanced`` filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor

from .audio import TARGET_SAMPLE_RATE
from .muaalem import (
    Wav2Vec2BertForMultilevelCTC,
    Wav2Vec2BertForMultilevelCTCConfig,
)
from .phoneme_vocab import NUM_PHONEME_CLASSES, greedy_ctc_decode

MODEL_ID = "obadx/muaalem-model-v3_2"
PHONEME_LEVEL = "phonemes"


@dataclass(frozen=True)
class PhonemeDecode:
    """The greedy-decoded phoneme string plus the frame counts behind it."""

    phonemes: str
    num_feature_frames: int  # feature-extractor frames fed to the model
    num_logit_frames: int    # phoneme-head timesteps (feature frames // 2)
    class_ids: tuple[int, ...] = ()  # per-frame argmax ids (blank == PHONEME_PAD_ID)


class MuaalemPhonemeModel:
    """Loaded Muaalem model + feature extractor, pinned to one device and dtype.

    Instantiate with :meth:`load`. :meth:`decode` takes a single 16 kHz mono
    waveform and :meth:`decode_batch` a list of them, returning the greedy-CTC
    phoneme string(s) from the phoneme head.
    """

    def __init__(
        self,
        model: Wav2Vec2BertForMultilevelCTC,
        feature_extractor: SeamlessM4TFeatureExtractor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.dtype = dtype

    @classmethod
    def load(
        cls,
        model_id: str = MODEL_ID,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MuaalemPhonemeModel":
        """Load the model (bf16) and feature extractor onto ``device``.

        Fails loudly if the model's phoneme head is not the expected 43-class
        vocabulary or the feature extractor is not 16 kHz — either would silently
        corrupt decoded labels.
        """
        device = torch.device(device)
        config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(model_id)
        phoneme_classes = config.level_to_vocab_size[PHONEME_LEVEL]
        if phoneme_classes != NUM_PHONEME_CLASSES:
            raise ValueError(
                f"{model_id} phoneme head has {phoneme_classes} classes, expected "
                f"{NUM_PHONEME_CLASSES} (tadabur.phoneme_vocab). Vocabulary drift — "
                "the decode/label mapping would be corrupt."
            )
        model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
            model_id, config=config, dtype=dtype
        )
        model.to(device).eval()

        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_id)
        if feature_extractor.sampling_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"{model_id} feature extractor expects "
                f"{feature_extractor.sampling_rate} Hz, not {TARGET_SAMPLE_RATE} Hz."
            )
        return cls(model, feature_extractor, device, dtype)

    def decode(self, waveform: np.ndarray, sample_rate: int) -> PhonemeDecode:
        """Greedy-CTC-decode the phoneme head for one 16 kHz mono ``waveform``.

        Convenience wrapper over :meth:`decode_batch` for a single clip; see it for
        the preprocessing/rate contract.
        """
        return self.decode_batch([waveform], sample_rate)[0]

    @torch.inference_mode()
    def decode_batch(
        self, waveforms: list[np.ndarray], sample_rate: int
    ) -> list[PhonemeDecode]:
        """Greedy-CTC-decode the phoneme head for a batch of 16 kHz mono waveforms.

        The clips are feature-extracted together with padding and run through the
        model in one variable-length forward pass under a real attention mask, so a
        padded clip decodes identically to processing it alone. Each clip's logits
        are sliced to its own valid length — computed with the model's
        ``_get_feat_extract_output_lengths`` from the feature-extractor attention
        mask, the same length mapping the training CTC loss uses — before greedy
        decode, so padding frames never leak into a decoded string.

        ``waveforms`` must already be 16 kHz mono (decode upstream via
        ``tadabur.audio.decode_to_mono_16k``); the ``SeamlessM4TFeatureExtractor``
        does not resample and this asserts the rate to avoid a silent mismatch.
        """
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"Expected {TARGET_SAMPLE_RATE} Hz mono audio, got {sample_rate} Hz. "
                "Resample before calling (e.g. tadabur.audio.decode_to_mono_16k)."
            )
        if not waveforms:
            return []

        features = self.feature_extractor(
            [np.asarray(w, dtype=np.float32) for w in waveforms],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = features.input_features.to(self.device, self.dtype)
        attention_mask = features.attention_mask.to(self.device)

        logits = self.model(
            input_features=input_features, attention_mask=attention_mask
        ).logits[PHONEME_LEVEL]

        feature_frames = attention_mask.sum(dim=1)
        logit_frames = self.model._get_feat_extract_output_lengths(
            feature_frames
        ).to(torch.long)
        class_ids = logits.argmax(dim=-1)

        decodes: list[PhonemeDecode] = []
        for i in range(len(waveforms)):
            valid_logit_frames = int(logit_frames[i])
            ids = class_ids[i, :valid_logit_frames].tolist()
            decodes.append(
                PhonemeDecode(
                    phonemes=greedy_ctc_decode(ids),
                    num_feature_frames=int(feature_frames[i]),
                    num_logit_frames=valid_logit_frames,
                    class_ids=tuple(ids),
                )
            )
        return decodes

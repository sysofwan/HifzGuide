"""Waqf pause detection via the dedicated recitation VAD, not the phoneme head.

Deriving pauses from the Muaalem phoneme head's CTC blank runs (the earlier approach)
is **over-eager**: a single blank threshold splits at every inter-word silence the
greedy decode blanks on, so nearby micro-pauses each cut a tiny one-word segment. The
phoneme head was trained to transcribe, not to tell a genuine waqf from a breath / sakt
/ inter-word gap.

`obadx/recitation-segmenter-v2` is a Wav2Vec2-BERT **fine-tuned specifically for waqf
segmentation** — a binary speech/silence frame classifier at 20 ms resolution
(F1 0.996). Its training labels come from Silero VAD post-filtered with
`min_silence_duration_ms=300` / `min_speech_duration_ms=700`, so the model has learned
that definition of a waqf and does not fire on the sub-pauses that made us over-split.

This module wraps that VAD: it decodes each staged clip to **clean speech intervals**
(`recitations_segmenter.clean_speech_intervals`, same 300/700/30 ms cleaning) and
returns the *interior silence gaps* between them — the waqf pauses. Those pause *times*
are handed to :func:`tadabur.waqf_detect.segment_clip`, which maps each to a *word*
boundary (so a segment can be phonetized into its realized reference). The torch import
is lazy, so :func:`pauses_from_intervals` (pure list logic) is unit-testable without a
GPU or transformers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import TARGET_SAMPLE_RATE
from .manifest import ManifestRecord

# The pretrained recitation VAD (speech/silence frame classifier, 20 ms resolution).
VAD_MODEL_ID = "obadx/recitation-segmenter-v2"

# The VAD's two per-frame softmax classes. Silence is the waqf-head distillation
# target (ADR-0004), so :func:`frame_silence_posteriors` reads class ``VAD_SILENCE_LABEL``.
VAD_SILENCE_LABEL = 0
VAD_SPEECH_LABEL = 1

# Waqf definition the VAD was trained on: silence ≥ 300 ms between speech ≥ 700 ms,
# each interval padded 30 ms. Reusing the training thresholds keeps inference on-model.
DEFAULT_MIN_SILENCE_MS = 300
DEFAULT_MIN_SPEECH_MS = 700
DEFAULT_PAD_MS = 30


def pauses_from_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """The interior silence gaps (seconds) between consecutive speech intervals.

    The VAD returns *speech* spans; a **waqf** is the silence *between* two of them.
    Leading silence (before the first span) and trailing silence (after the last) are
    clip edges, not interior stops, so they are excluded. ``intervals`` is assumed
    sorted and disjoint; a non-positive gap (padding made two spans meet) yields no
    pause.
    """
    return [
        (prev_end, next_start)
        for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:])
        if next_start > prev_end
    ]


def _load_vad(device, dtype):
    """Load the VAD model + feature extractor onto ``device`` in ``dtype``."""
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioFrameClassification,
    )

    processor = AutoFeatureExtractor.from_pretrained(VAD_MODEL_ID)
    model = AutoModelForAudioFrameClassification.from_pretrained(VAD_MODEL_ID)
    model.to(device, dtype=dtype)
    model.eval()
    return model, processor


def frame_silence_posteriors(
    waveforms: list[np.ndarray],
    model,
    processor,
    *,
    device,
    dtype,
    batch_size: int,
) -> list[np.ndarray]:
    """Per-20 ms-frame silence posteriors ``P(silence)`` for each waveform.

    Runs the VAD forward in fixed-size batches and returns, per waveform, the
    softmax probability of the **silence** class (``VAD_SILENCE_LABEL``) at each
    20 ms frame — the soft teacher signal the waqf head is distilled from (ADR-0004),
    *before* any 300/700 ms interval cleaning. The softmax is taken in float32 (the
    model runs in ``dtype``, typically bf16) so the posteriors are deterministic and
    numerically stable. Each waveform is trimmed to its own valid frame count via the
    feature-extractor attention mask, so batch padding never leaks a frame.
    """
    import torch

    posteriors: list[np.ndarray] = []
    for start in range(0, len(waveforms), batch_size):
        batch = [np.asarray(w, dtype=np.float32) for w in waveforms[start : start + batch_size]]
        features = processor(
            batch,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_features = features.input_features.to(device, dtype=dtype)
        attention_mask = features.attention_mask.to(device)
        with torch.inference_mode():
            logits = model(
                input_features=input_features, attention_mask=attention_mask
            ).logits
        silence = torch.softmax(logits.float(), dim=-1)[..., VAD_SILENCE_LABEL]
        valid_frames = attention_mask.sum(dim=1).to(torch.long)
        for i in range(len(batch)):
            frames = int(valid_frames[i])
            posteriors.append(silence[i, :frames].cpu().numpy().astype(np.float32))
    return posteriors


def compute_clip_silence_posteriors(
    records: list[ManifestRecord],
    clips_dir: Path,
    *,
    device,
    dtype,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Per-20 ms silence posteriors per staged clip, keyed by ``audio_filename``.

    The teacher half of the waqf distillation (ADR-0004): loads the VAD, decodes
    every clip present under ``clips_dir`` to its per-frame ``P(silence)``
    (:func:`frame_silence_posteriors`), and frees the VAD before returning so the
    Muaalem model can be loaded without holding both on the GPU. Clips missing from
    ``clips_dir`` are omitted (the caller tallies them), matching
    :func:`compute_clip_pauses`. Pooling to the 40 ms Muaalem lattice is left to the
    torch-free :mod:`training.waqf_distill`.
    """
    import soundfile as sf
    import torch

    present = [r for r in records if (clips_dir / r.audio_filename).exists()]
    model, processor = _load_vad(device, dtype)
    posteriors: dict[str, np.ndarray] = {}
    try:
        for start in range(0, len(present), batch_size):
            batch = present[start : start + batch_size]
            waveforms = [
                sf.read(clips_dir / r.audio_filename, dtype="float32")[0] for r in batch
            ]
            for record, silence in zip(
                batch,
                frame_silence_posteriors(
                    waveforms, model, processor,
                    device=device, dtype=dtype, batch_size=batch_size,
                ),
            ):
                posteriors[record.audio_filename] = silence
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return posteriors


def _clip_intervals(
    waveforms: list[np.ndarray],
    model,
    processor,
    *,
    device,
    dtype,
    batch_size: int,
    min_silence_ms: float,
    min_speech_ms: float,
    pad_ms: float,
) -> list[list[tuple[float, float]]]:
    """Clean speech intervals (seconds) for each waveform, batched through the VAD."""
    import torch
    from recitations_segmenter import (
        NoSpeechIntervals,
        TooHighMinSpeechDuration,
        clean_speech_intervals,
        segment_recitations,
    )

    waves = [torch.as_tensor(w, dtype=torch.float32) for w in waveforms]
    outputs = segment_recitations(
        waves, model, processor, device=device, dtype=dtype, batch_size=batch_size
    )
    intervals: list[list[tuple[float, float]]] = []
    for out in outputs:
        try:
            clean = clean_speech_intervals(
                out.speech_intervals,
                out.is_complete,
                min_silence_duration_ms=min_silence_ms,
                min_speech_duration_ms=min_speech_ms,
                pad_duration_ms=pad_ms,
                return_seconds=True,
            )
            intervals.append(
                [(float(a), float(b)) for a, b in clean.clean_speech_intervals.tolist()]
            )
        except (NoSpeechIntervals, TooHighMinSpeechDuration):
            intervals.append([])  # no confident speech → no interior pauses (whole clip)
    return intervals


def compute_clip_pauses(
    records: list[ManifestRecord],
    clips_dir: Path,
    *,
    device,
    dtype,
    batch_size: int = 8,
    min_silence_ms: float = DEFAULT_MIN_SILENCE_MS,
    min_speech_ms: float = DEFAULT_MIN_SPEECH_MS,
    pad_ms: float = DEFAULT_PAD_MS,
) -> dict[str, list[tuple[float, float]]]:
    """Waqf pause gaps (seconds) per staged clip, keyed by ``audio_filename``.

    Loads the VAD, decodes every clip present under ``clips_dir`` to clean speech
    intervals in fixed-size batches, and returns the interior silence gaps
    (:func:`pauses_from_intervals`). Clips missing from ``clips_dir`` are omitted (the
    caller tallies them). The VAD is freed before returning so the Muaalem phoneme
    model can be loaded without holding both on the GPU.
    """
    import soundfile as sf
    import torch

    present = [r for r in records if (clips_dir / r.audio_filename).exists()]
    model, processor = _load_vad(device, dtype)
    pauses: dict[str, list[tuple[float, float]]] = {}
    try:
        for start in range(0, len(present), batch_size):
            batch = present[start : start + batch_size]
            waveforms = [
                sf.read(clips_dir / r.audio_filename, dtype="float32")[0] for r in batch
            ]
            for record, intervals in zip(
                batch,
                _clip_intervals(
                    waveforms, model, processor,
                    device=device, dtype=dtype, batch_size=batch_size,
                    min_silence_ms=min_silence_ms, min_speech_ms=min_speech_ms,
                    pad_ms=pad_ms,
                ),
            ):
                pauses[record.audio_filename] = pauses_from_intervals(intervals)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pauses

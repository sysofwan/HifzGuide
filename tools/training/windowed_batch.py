"""Phoneme-only windowed CTC batches for the whole-clip fine-tune (ADR-0004 P7.D2).

The whole-clip phoneme-only run (rung (2) of ADR-0004's ablation ladder,
:mod:`training.whole_clip_phoneme`) trains on **fixed 5 s windows over the
un-waqf-segmented recitation** — the A2 frozen contract. This module is the data path that
turns the per-window CTC labels :mod:`training.windowed_labels` wrote, plus the staged clip
audio, into padded feature batches the CTC loss consumes. It is the phoneme half of the
data collator (#8); the waqf soft-label half is added by the joint run (#31).

Three things are pinned here:

* **The CTC target strips word-separator spaces.** ``reference_phonemes`` carries spaces as
  *word* separators for the Smith-Waterman aligner, but the model's 43-class phoneme head
  has no space class and its greedy decode never emits one. :func:`encode_phoneme_label`
  drops the separators and maps the survivors through the canonical
  :data:`tadabur.phoneme_vocab.PHONEME_CHAR_TO_ID`, failing loudly on any out-of-vocabulary
  character rather than silently corrupting a label.

* **Window audio is sliced on the clip-relative sample span the label carries.** Each
  :class:`~training.windowed_labels.WindowLabel` names its exact ``[start_sample,
  start_sample + num_samples)`` span in the whole clip, so the window waveform is the same
  span the label (and, in the joint run, the waqf soft target) is built over — no
  re-derivation, no drift. Clip waveforms are decoded once and cached across their windows.

* **Batches are length-bucketed under a frame budget.** Backprop over 5 s windows is heavy
  on 16 GB (ADR-0004 OOM risk), so batches are formed by grouping windows of similar length
  and capped by a total-feature-frame budget (the knob the memory preflight sets) — keeping
  padding waste low and peak activation memory bounded without a fixed batch size.

Runs on Linux + CUDA (see ``tools/environment.yml``): the collator feature-extracts with the
model's own ``SeamlessM4TFeatureExtractor`` (train/inference parity, :mod:`tadabur.inference`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tadabur.audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from tadabur.audit_sampler import local_audio_path
from tadabur.phoneme_vocab import PHONEME_CHAR_TO_ID
from training.windowed_labels import WindowLabel, read_labels

# Word-separator in the realized reference phonemes — not a model class, dropped for CTC.
WORD_SEPARATOR = " "


def encode_phoneme_label(reference_phonemes: str) -> list[int]:
    """Realized reference phoneme string → CTC class-id target.

    Drops the :data:`WORD_SEPARATOR` word separators (the model has no space class) and
    maps every remaining character through the canonical
    :data:`tadabur.phoneme_vocab.PHONEME_CHAR_TO_ID`. An out-of-vocabulary character raises
    — a silently-skipped one would shorten the target and mis-teach the alignment.
    """
    ids: list[int] = []
    for char in reference_phonemes:
        if char == WORD_SEPARATOR:
            continue
        try:
            ids.append(PHONEME_CHAR_TO_ID[char])
        except KeyError as exc:
            raise ValueError(
                f"phoneme {char!r} (U+{ord(char):04X}) in reference "
                f"{reference_phonemes!r} is not a model phoneme class — the CTC label "
                "would be corrupt. Re-normalize the reference (tadabur.normalization)."
            ) from exc
    return ids


@dataclass(frozen=True)
class WindowedCtcExample:
    """One training window: its audio slice, CTC target, and its 20 ms length.

    ``key`` is ``(clip_audio_filename, window_index)`` — the stable identity the bucketing
    and any join with the waqf soft labels share. ``feature_frames`` is the window's 20 ms
    length, the sort key the length bucketing groups on; ``logit_frames`` its post-adapter
    40 ms length, asserted ``>= len(label_ids)`` so the window is a feasible CTC target.
    """

    key: tuple[str, int]
    audio: np.ndarray
    label_ids: tuple[int, ...]
    feature_frames: int
    logit_frames: int

    def __post_init__(self) -> None:
        if self.logit_frames < len(self.label_ids):
            raise ValueError(
                f"window {self.key} has {len(self.label_ids)} label ids but only "
                f"{self.logit_frames} logit frames — infeasible CTC target. The label "
                "build (training.windowed_labels) should have excluded it (target_too_long)."
            )


class ClipAudioCache:
    """Decode each staged clip once, reuse its waveform across all its windows.

    A clip has up to ~8 windows (the A2 cap); decoding it once and slicing keeps the data
    path from re-reading and re-decoding the same file per window. ``audio_dir`` holds the
    staged clips under their :func:`tadabur.audit_sampler.local_audio_path` names — the
    same layout :mod:`tadabur.eval_harness` reads.
    """

    def __init__(self, audio_dir: Path) -> None:
        self._audio_dir = audio_dir
        self._cache: dict[str, np.ndarray] = {}

    def waveform(self, clip_audio_filename: str) -> np.ndarray:
        cached = self._cache.get(clip_audio_filename)
        if cached is None:
            path = self._audio_dir / local_audio_path(clip_audio_filename)
            if not path.is_file():
                raise FileNotFoundError(
                    f"clip audio {path} for {clip_audio_filename!r} not found under "
                    f"{self._audio_dir} — stage it (tadabur.audit_sampler) before training."
                )
            cached = decode_to_mono_16k(path.read_bytes())
            self._cache[clip_audio_filename] = cached
        return cached


def build_example(label: WindowLabel, audio: ClipAudioCache) -> WindowedCtcExample:
    """Pair one window label with its clip-relative audio slice and encoded CTC target."""
    waveform = audio.waveform(label.clip_audio_filename)
    end = label.start_sample + label.num_samples
    if end > len(waveform):
        raise ValueError(
            f"window {label.clip_audio_filename}#{label.window_index} spans samples "
            f"[{label.start_sample}, {end}) but the clip has only {len(waveform)} — the "
            "label build and this audio are out of sync."
        )
    window_audio = np.asarray(
        waveform[label.start_sample : end], dtype=np.float32
    )
    return WindowedCtcExample(
        key=(label.clip_audio_filename, label.window_index),
        audio=window_audio,
        label_ids=tuple(encode_phoneme_label(label.phoneme_label)),
        feature_frames=label.feature_frames,
        logit_frames=label.logit_frames,
    )


def load_examples(
    labels_path: Path, audio_dir: Path, split: str
) -> list[WindowedCtcExample]:
    """Read one split's window labels and build its examples in deterministic key order.

    ``split`` is ``"train"`` or ``"val"`` as written by
    :func:`training.windowed_labels.write_labels`. An unknown split raises rather than
    silently returning no examples.
    """
    by_split = read_labels(labels_path)
    if split not in by_split:
        raise KeyError(
            f"split {split!r} not in {labels_path} (have {sorted(by_split)})."
        )
    cache = ClipAudioCache(audio_dir)
    ordered = sorted(by_split[split], key=lambda w: (w.clip_audio_filename, w.window_index))
    return [build_example(label, cache) for label in ordered]


def length_bucketed_batches(
    examples: list[WindowedCtcExample],
    max_frames_per_batch: int,
    max_windows_per_batch: int,
    seed: int,
) -> list[list[WindowedCtcExample]]:
    """Group windows of similar length into batches under a feature-frame budget.

    Windows are sorted by length so a batch pads to nearly-uniform length (low waste), then
    packed greedily until either ``max_windows_per_batch`` windows or ``max_frames_per_batch``
    *padded* feature frames (``batch_size * longest_window``) would be exceeded — the padded
    frame count is what drives peak activation memory, so the budget the preflight sets bounds
    VRAM directly. Batch *order* is shuffled with ``seed`` (so epochs differ) while each
    batch's contents stay length-homogeneous. Deterministic for a given ``seed``.
    """
    if max_frames_per_batch <= 0 or max_windows_per_batch <= 0:
        raise ValueError("max_frames_per_batch and max_windows_per_batch must be positive")

    ordered = sorted(examples, key=lambda e: (e.feature_frames, e.key))
    batches: list[list[WindowedCtcExample]] = []
    current: list[WindowedCtcExample] = []
    longest = 0
    for example in ordered:
        if example.feature_frames > max_frames_per_batch:
            raise ValueError(
                f"window {example.key} needs {example.feature_frames} feature frames, over "
                f"the {max_frames_per_batch}-frame batch budget — raise the budget or shorten "
                "the window contract."
            )
        prospective_longest = max(longest, example.feature_frames)
        prospective_frames = prospective_longest * (len(current) + 1)
        if current and (
            len(current) >= max_windows_per_batch
            or prospective_frames > max_frames_per_batch
        ):
            batches.append(current)
            current, longest = [], 0
        current.append(example)
        longest = max(longest, example.feature_frames)
    if current:
        batches.append(current)

    random.Random(seed).shuffle(batches)
    return batches


@dataclass
class WindowedCtcBatch:
    """A collated, padded batch on one device: features, mask, and ``-100``-padded labels.

    ``input_features`` ``(B, F, feat_dim)`` and ``attention_mask`` ``(B, F)`` come from the
    model's feature extractor; ``labels`` ``(B, L)`` pads short targets with ``-100`` (the
    CTC ignore index the loss masks on). ``keys`` records each row's identity for logging.
    """

    input_features: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    keys: list[tuple[str, int]]

    def to(self, device: torch.device, dtype: torch.dtype) -> "WindowedCtcBatch":
        """Move features to ``device``/``dtype``; mask and labels stay integer on ``device``."""
        return WindowedCtcBatch(
            input_features=self.input_features.to(device, dtype),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            keys=self.keys,
        )


def pad_labels(label_ids: list[tuple[int, ...]]) -> torch.Tensor:
    """``(B, max_len)`` label tensor, short rows right-padded with ``-100`` (CTC ignore)."""
    max_len = max((len(ids) for ids in label_ids), default=0)
    padded = torch.full((len(label_ids), max_len), -100, dtype=torch.long)
    for row, ids in enumerate(label_ids):
        if ids:
            padded[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded


class WindowedCtcCollator:
    """Feature-extract a window batch and pad it to a :class:`WindowedCtcBatch`.

    Wraps the model's own ``SeamlessM4TFeatureExtractor`` so training preprocessing is
    identical to inference (:mod:`tadabur.inference`): the raw window waveforms are
    feature-extracted together with padding, and the CTC targets are ``-100``-padded to the
    batch's longest target. Feature extraction is dtype-agnostic here (float32); the
    training step casts to bf16.
    """

    def __init__(self, feature_extractor) -> None:
        if feature_extractor.sampling_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"feature extractor expects {feature_extractor.sampling_rate} Hz, not "
                f"{TARGET_SAMPLE_RATE} Hz — window audio is 16 kHz mono."
            )
        self._feature_extractor = feature_extractor

    def __call__(self, examples: list[WindowedCtcExample]) -> WindowedCtcBatch:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        features = self._feature_extractor(
            [np.asarray(e.audio, dtype=np.float32) for e in examples],
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        return WindowedCtcBatch(
            input_features=features.input_features,
            attention_mask=features.attention_mask,
            labels=pad_labels([e.label_ids for e in examples]),
            keys=[e.key for e in examples],
        )


__all__ = [
    "WORD_SEPARATOR",
    "encode_phoneme_label",
    "WindowedCtcExample",
    "ClipAudioCache",
    "build_example",
    "load_examples",
    "length_bucketed_batches",
    "WindowedCtcBatch",
    "pad_labels",
    "WindowedCtcCollator",
]

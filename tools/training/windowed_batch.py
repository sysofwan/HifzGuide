"""Windowed CTC batches for the whole-clip fine-tune (ADR-0004 P7.D2/D3).

The whole-clip runs train on **fixed 5 s windows over the un-waqf-segmented recitation**
(the A2 frozen contract, :mod:`training.waqf_distill`), *not* on the individual waqf
segments. This module is the data path that turns the per-window CTC labels
:mod:`training.windowed_labels` wrote, plus the staged clip audio, into padded feature
batches. It carries both halves of the collator (#8): the **phoneme** batch the rung-(2)
phoneme-only run (:mod:`training.whole_clip_phoneme`) consumes, and the **joint** batch the
rung-(3) detached-waqf run (:mod:`training.joint_waqf`) consumes — the phoneme batch plus
each window's 2:1-pooled VAD silence teacher, joined on the shared window key.

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
    """One training window: its audio slice, CTC target, and its clip-relative span.

    ``key`` is ``(clip_audio_filename, window_index)`` — the stable identity the bucketing
    and any join with the waqf soft labels share. ``start_sample`` / ``num_samples`` are the
    **clip-relative** span the window audio was sliced at (carried verbatim from
    :class:`~training.windowed_labels.WindowLabel`), so a join with the waqf soft labels can
    assert *both* windows describe the same audio span — not merely the same length — and
    reject a store on a shifted hop/origin that reuses the same ``window_index``.
    ``feature_frames`` is the window's 20 ms length, the sort key the length bucketing groups
    on; ``logit_frames`` its post-adapter 40 ms length, asserted ``>= len(label_ids)`` so the
    window is a feasible CTC target.
    """

    key: tuple[str, int]
    audio: np.ndarray
    label_ids: tuple[int, ...]
    start_sample: int
    num_samples: int
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
    path from re-reading and re-decoding the same file per window. Only the **current**
    clip is retained (callers build examples in clip order), so the cache costs one clip
    of RAM rather than the whole corpus. ``audio_dir`` may use
    either staged layout: the :func:`tadabur.audit_sampler.local_audio_path` hash-prefixed
    name :mod:`tadabur.eval_harness` reads, or the plain ``audio_filename`` that
    :mod:`tadabur.segment_score` and :mod:`training.waqf_distill` read — so one staged clip
    directory drives the label build, the distillation and the training run alike.
    """

    def __init__(self, audio_dir: Path) -> None:
        self._audio_dir = audio_dir
        # Only the most recently decoded clip is retained. Callers build examples in clip
        # order, so a clip is decoded exactly once either way — but holding every clip
        # would cost ~10 GB of host RAM at corpus scale (16 k clips), which is enough to
        # push a 24 GB box into thrashing before the first training step.
        self._cache: dict[str, np.ndarray] = {}

    def waveform(self, clip_audio_filename: str) -> np.ndarray:
        cached = self._cache.get(clip_audio_filename)
        if cached is None:
            self._cache.clear()
            path = self._audio_dir / local_audio_path(clip_audio_filename)
            if not path.is_file():
                path = self._audio_dir / clip_audio_filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"clip audio for {clip_audio_filename!r} not found under "
                    f"{self._audio_dir} under either the hash-prefixed "
                    f"(tadabur.audit_sampler) or plain name — stage it before training."
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
        start_sample=label.start_sample,
        num_samples=label.num_samples,
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


@dataclass(frozen=True)
class JointWindowedExample:
    """One joint-run window: the phoneme CTC example plus its pooled silence teacher.

    ``ctc`` is the exact :class:`WindowedCtcExample` the phoneme-only run (#29) uses — the
    joint run adds *only* the waqf target, so the phoneme path stays bit-identical to rung
    (2) (ADR-0004 isolation). ``target_silence`` ``(logit_frames,)`` is the 2:1-pooled VAD
    ``P(silence)`` on the window's 40 ms lattice
    (:class:`training.waqf_distill.WindowSilenceTarget`); its length equals the window's
    ``logit_frames``, so it lines up frame-for-frame with the CTC lattice the waqf head
    rides. ``feature_frames`` / ``key`` delegate to ``ctc`` so the same length bucketing
    (:func:`length_bucketed_batches`) groups joint windows unchanged.
    """

    ctc: WindowedCtcExample
    target_silence: np.ndarray

    def __post_init__(self) -> None:
        if self.target_silence.shape != (self.ctc.logit_frames,):
            raise ValueError(
                f"window {self.ctc.key} has {self.ctc.logit_frames} logit frames but its "
                f"silence teacher is shape {self.target_silence.shape} — the phoneme labels "
                "and the soft labels are on different window grids; regenerate both."
            )

    @property
    def key(self) -> tuple[str, int]:
        return self.ctc.key

    @property
    def feature_frames(self) -> int:
        return self.ctc.feature_frames


@dataclass
class JointWindowedBatch:
    """A collated joint batch: the phoneme :class:`WindowedCtcBatch` plus the silence target.

    ``phoneme`` carries the features / mask / ``-100``-padded CTC labels unchanged; the joint
    run adds ``target_silence`` ``(B, T)``, the pooled VAD teacher padded to the batch's 40 ms
    lattice length with ``0`` (speech) on the trailing frames. Those padded frames are the same
    ones the model's ``student_lengths`` exclude, so the padding value is never scored — the
    frame mask (:func:`training.waqf_head.frame_mask_from_lengths`) drops it from the KL.
    """

    phoneme: WindowedCtcBatch
    target_silence: torch.Tensor

    @property
    def keys(self) -> list[tuple[str, int]]:
        return self.phoneme.keys

    def to(self, device: torch.device, dtype: torch.dtype) -> "JointWindowedBatch":
        """Move to ``device``/``dtype``; the silence teacher rides the feature dtype."""
        return JointWindowedBatch(
            phoneme=self.phoneme.to(device, dtype),
            target_silence=self.target_silence.to(device, dtype),
        )


def pad_target_silence(targets: list[np.ndarray]) -> torch.Tensor:
    """``(B, max_frames)`` silence-teacher tensor, short rows right-padded with ``0`` (speech).

    The pad value is speech (``P(silence) = 0``); the training step masks padded frames out
    of the KL by ``student_lengths``, so the value only has to be a valid, non-silence float.
    """
    max_frames = max((len(t) for t in targets), default=0)
    padded = torch.zeros(len(targets), max_frames, dtype=torch.float32)
    for row, target in enumerate(targets):
        if len(target):
            padded[row, : len(target)] = torch.from_numpy(np.asarray(target, dtype=np.float32))
    return padded


class JointWindowedCollator:
    """Collate joint windows: the phoneme collator plus the padded silence teacher.

    Delegates the feature extraction and CTC-label padding to :class:`WindowedCtcCollator`
    (so the phoneme half is byte-for-byte the rung-(2) path) and pads each window's
    ``target_silence`` to the batch's 40 ms lattice length. The padded silence length equals
    the longest window's ``logit_frames``, which is the model's output frame count for the
    padded feature batch — so the teacher lines up with ``silence_logits`` frame-for-frame.
    """

    def __init__(self, feature_extractor) -> None:
        self._phoneme = WindowedCtcCollator(feature_extractor)

    def __call__(self, examples: list[JointWindowedExample]) -> JointWindowedBatch:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        return JointWindowedBatch(
            phoneme=self._phoneme([e.ctc for e in examples]),
            target_silence=pad_target_silence([e.target_silence for e in examples]),
        )


def load_joint_examples(
    labels_path: Path, audio_dir: Path, soft_label_root: Path, split: str
) -> list[JointWindowedExample]:
    """Load one split's windows and join each to its pooled silence teacher.

    Builds the phoneme examples exactly as the phoneme-only run does
    (:func:`load_examples`), then attaches each window's silence teacher from the
    recitation-grid soft-label store (:class:`training.waqf_distill.SoftLabelReader`),
    joining on the shared ``(clip_audio_filename, window_index)`` key. The join **asserts the
    two artifacts describe the same clip-relative audio span** — the soft target's
    ``start_sample`` *and* ``num_samples`` must both equal the phoneme window's — so a store
    on a shifted hop/origin that reuses the same ``window_index`` (same length, different
    start) is rejected rather than pairing a misaligned silence teacher into the joint loss
    (ADR-0004 fail-fast on silent label corruption). Order is the deterministic key order
    :func:`load_examples` returns.
    """
    from training.waqf_distill import SoftLabelReader

    reader = SoftLabelReader.open(soft_label_root)
    joint: list[JointWindowedExample] = []
    for ctc in load_examples(labels_path, audio_dir, split):
        clip_audio_filename, window_index = ctc.key
        target = reader.target(clip_audio_filename, window_index)
        if (target.start_sample, target.num_samples) != (ctc.start_sample, ctc.num_samples):
            raise ValueError(
                f"window {ctc.key}: phoneme label spans clip-relative samples "
                f"[{ctc.start_sample}, {ctc.start_sample + ctc.num_samples}) but its silence "
                f"teacher spans [{target.start_sample}, "
                f"{target.start_sample + target.num_samples}) — the phoneme and soft labels "
                "are on different window grids (shifted origin/hop or length); regenerate both "
                "on the same recitation grid."
            )
        joint.append(JointWindowedExample(ctc=ctc, target_silence=target.silence_40ms))
    return joint


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
    "JointWindowedExample",
    "JointWindowedBatch",
    "JointWindowedCollator",
    "load_joint_examples",
]

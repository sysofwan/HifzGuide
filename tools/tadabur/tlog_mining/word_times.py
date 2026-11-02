"""One gating clip's word times (#263): CTC-force-align the filename ayah, map it to words.

The reference is the filename ayah's **full text** — the phonemes
:mod:`training.tlog_eval` grades, whose readout reference is
:func:`tadabur.waqf_segments.hafs_segment_reference` (space-bearing). CTC labels are phoneme
classes, and the model's alphabet has no space, so the alignment uses
:func:`tadabur.waqf_segments.hafs_word_reference`: the **spaceless remap of that same
phonetization**, with the per-word offsets into it. Do not swap in the spaced builder: every
character would be refused as outside the vocabulary. The teacher's posteriors are aligned to
those phonemes with :func:`training.ctc_paths.ctc_best_path`, and word ``j`` takes the frames of
the labels in ``offsets[j]:offsets[j + 1]``.

Word times are in seconds of the clip: ``start`` is the first frame of the word's first
phoneme, ``end`` the last frame of its last phoneme plus one (so adjacent words do not share a
frame). The frame length is ``duration_s / frames`` — the model's strided length is not the
clip's duration divided by a fixed 40 ms. The per-frame alignment log-likelihood is the forward
probability over every alignment (``ctc_log_likelihood / frames``), not the Viterbi score: it is
the existing helper's value and needs no change to :mod:`training.ctc_paths`.

A clip that cannot be aligned is a committed record with ``aligned: false`` and a ``reason``,
never an exception and never a missing row: an ayah with no filename label, an ayah the
phonetizer cannot handle, a reference character outside the model's 43 classes, or posteriors
too short for the transcript. The times depend only on the reference and the posteriors — no
decoded mark is read — so re-running reproduces them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from tadabur.phoneme_vocab import PHONEME_CHAR_TO_ID
from training.ctc_paths import ctc_best_path, ctc_log_likelihood
from training.nonpro_decode import item_key


@dataclass(frozen=True)
class Alignment:
    """A transcript's alignment: one ``(start_s, end_s)`` per word plus the two frame rates."""

    word_times: tuple[tuple[float, float], ...]
    seconds_per_frame: float
    ll_per_frame: float


def align_words(log_posteriors: np.ndarray, labels: Sequence[int], offsets: Sequence[int], *,
                duration_s: float) -> Alignment:
    """Word times from the best CTC path of ``labels`` in ``log_posteriors``.

    ``offsets`` has ``len(words) + 1`` entries partitioning ``labels``: word ``j`` spans
    ``labels[offsets[j]:offsets[j + 1]]``. Raises ``ValueError`` for offsets that do not
    partition the labels, or (from :func:`training.ctc_paths.ctc_best_path`) for posteriors too
    short to hold the transcript.
    """
    frames = len(log_posteriors)
    if frames == 0:
        raise ValueError(f"no frames for the {len(labels)}-label transcript")
    _require_offsets(offsets, len(labels))
    path = ctc_best_path(log_posteriors, labels)
    seconds_per_frame = duration_s / frames
    word_times = tuple(
        (path[offsets[word]][0] * seconds_per_frame,
         (path[offsets[word + 1] - 1][1] + 1) * seconds_per_frame)
        for word in range(len(offsets) - 1))
    return Alignment(word_times, seconds_per_frame,
                     ctc_log_likelihood(log_posteriors, labels) / frames)


def build_record(row: dict, *, reference_for: Callable[[str], tuple[str, Sequence[int]]],
                 log_posteriors: np.ndarray, duration_s: float) -> dict:
    """The committed record of one gating clip, aligned or listed unaligned with its reason.

    ``reference_for(label)`` returns the ayah's raw spaceless phoneme reference and its word
    offsets. The reference is the **filename** ayah (:func:`training.tlog_eval._canonical_words`
    over ``hafs_word_reference``), never the row's ``assigned_label``: the row's own label is
    what the clip claims, not what it says.
    """
    record = {"key": item_key(row["audio_path"]), "split": row["split"],
              "shard_file": row["shard_file"], "row": row["row"],
              "audio_path": row["audio_path"], "audio_sha256": row["audio_sha256"],
              "surah_ayah": row.get("filename_label")}
    label = row.get("filename_label")
    if label is None:
        return _unaligned(record, "the row names no ayah")
    try:
        reference, offsets = reference_for(label)
    except (KeyError, IndexError) as error:
        return _unaligned(record, f"the phonetizer cannot handle {label}: {error!r}")
    try:
        labels = _labels(reference)
        alignment = align_words(log_posteriors, labels, offsets, duration_s=duration_s)
    except ValueError as error:
        return _unaligned(record, str(error))
    return {**record, "aligned": True, "reason": None,
            "word_times": [[start, end] for start, end in alignment.word_times],
            "seconds_per_frame": alignment.seconds_per_frame,
            "ll_per_frame": alignment.ll_per_frame}


def _unaligned(record: dict, reason: str) -> dict:
    return {**record, "aligned": False, "reason": reason, "word_times": None,
            "seconds_per_frame": None, "ll_per_frame": None}


def _labels(reference: str) -> list[int]:
    """The reference's class ids; a character outside the model's vocabulary is a refusal."""
    unknown = sorted(set(reference) - set(PHONEME_CHAR_TO_ID))
    if unknown:
        raise ValueError(f"reference characters outside the phoneme vocabulary: {unknown}")
    return [PHONEME_CHAR_TO_ID[char] for char in reference]


def _require_offsets(offsets: Sequence[int], count: int) -> None:
    """Refuse offsets that are not ``0 = offsets[0] < ... < offsets[-1] = count``."""
    values = list(offsets)
    if (len(values) < 2 or values[0] != 0 or values[-1] != count
            or any(later <= earlier for earlier, later in zip(values, values[1:]))):
        raise ValueError(f"offsets {values} do not partition the {count} labels")

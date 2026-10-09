"""The CTC probability of a given transcript, and its best path, from a model's posteriors.

A greedy decode (:func:`training.decoding.scan_ctc`) keeps only the best class per frame, so
it cannot say how close a transcript it did *not* emit came. These two functions read that
off the per-frame log-posteriors (:meth:`training.decoding.Decoder.span_log_posteriors`):

* :func:`ctc_log_likelihood` -- ``log P(labels | audio)``, summed over every frame alignment
  CTC allows (the forward algorithm). The difference of two transcripts' values is their
  posterior log-ratio, which is what the shaddah probe (:mod:`training.shaddah_probe`) uses to
  ask how much mass a doubled consonant gets where the decode emitted one.
* :func:`ctc_best_path` -- the single most probable alignment of ``labels`` (Viterbi), as the
  frames each label occupies, which says *where* that mass sits.

A repeated label (``c c``) can only be produced with a blank between the two, which is the
whole reason a geminate is hard for CTC. Both functions are pure numpy over a
``(frames, classes)`` array of natural-log posteriors, with the blank at class
:data:`tadabur.phoneme_vocab.PHONEME_PAD_ID`, so they are tested on synthetic posteriors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tadabur.phoneme_vocab import PHONEME_PAD_ID as BLANK_ID


def _extended(labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """The blank-interleaved label sequence and, per state, whether it may skip back two.

    State ``2k + 1`` is label ``k``; even states are blanks. A label state may be entered from
    two states back (skipping the blank between) unless that state holds the same label.
    """
    if any(label == BLANK_ID for label in labels):
        raise ValueError("a transcript cannot contain the blank")
    states = np.full(2 * len(labels) + 1, BLANK_ID, dtype=np.int64)
    states[1::2] = labels
    skip = np.zeros(len(states), dtype=bool)
    skip[3::2] = states[3::2] != states[1:-2:2]
    return states, skip


def ctc_log_likelihood(log_posteriors: np.ndarray, labels: Sequence[int]) -> float:
    """``log P(labels | audio)`` under CTC: the log-sum over every alignment (``-inf`` if none fits)."""
    states, skip = _extended(labels)
    emit = np.asarray(log_posteriors, dtype=np.float64)[:, states]
    if not labels:
        return float(emit[:, 0].sum())
    alpha = np.full(len(states), -np.inf)
    alpha[:2] = emit[0, :2]
    for frame in emit[1:]:
        stay = alpha
        step = np.concatenate(([-np.inf], alpha[:-1]))
        jump = np.where(skip, np.concatenate(([-np.inf, -np.inf], alpha[:-2])), -np.inf)
        alpha = np.logaddexp(np.logaddexp(stay, step), jump) + frame
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def ctc_best_path(log_posteriors: np.ndarray, labels: Sequence[int]) -> list[tuple[int, int]]:
    """The Viterbi alignment of ``labels``: each label's ``(first_frame, last_frame)``.

    Raises if the posteriors have too few frames for the transcript (every label needs a
    frame, and every repeated pair a blank frame between).
    """
    if not labels:
        return []
    states, skip = _extended(labels)
    emit = np.asarray(log_posteriors, dtype=np.float64)[:, states]
    frames, width = emit.shape
    score = np.full(width, -np.inf)
    score[:2] = emit[0, :2]
    back = np.zeros((frames, width), dtype=np.int64)
    index = np.arange(width)
    for t in range(1, frames):
        candidates = np.stack(
            [
                score,
                np.concatenate(([-np.inf], score[:-1])),
                np.where(skip, np.concatenate(([-np.inf, -np.inf], score[:-2])), -np.inf),
            ]
        )
        best = candidates.argmax(axis=0)
        back[t] = index - best
        score = candidates[best, index] + emit[t]
    state = width - 1 if score[-1] >= score[-2] else width - 2
    if not np.isfinite(score[state]):
        raise ValueError(f"{frames} frames cannot hold the {len(labels)}-label transcript")
    path = [state]
    for t in range(frames - 1, 0, -1):
        state = back[t, state]
        path.append(state)
    path.reverse()
    spans: list[list[int]] = [[] for _ in labels]
    for t, state in enumerate(path):
        if state % 2:
            spans[state // 2].append(t)
    return [(occupied[0], occupied[-1]) for occupied in spans]

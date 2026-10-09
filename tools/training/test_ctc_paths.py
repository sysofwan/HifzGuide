"""The CTC forward and Viterbi against brute force over every frame path (torch-free)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from training.ctc_paths import ctc_best_path, ctc_log_likelihood

BLANK, A, B = 0, 1, 2


def _collapse(path) -> tuple[int, ...]:
    out = []
    previous = None
    for token in path:
        if token != previous and token != BLANK:
            out.append(token)
        previous = token
    return tuple(out)


def _random_log_posteriors(frames: int, classes: int, seed: int) -> np.ndarray:
    logits = np.random.default_rng(seed).normal(size=(frames, classes)) * 2
    return logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)


def _paths(frames: int, classes: int):
    return itertools.product(range(classes), repeat=frames)


@pytest.mark.parametrize("labels", [(), (A,), (A, A), (A, B), (A, B, A), (B, B, A)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_equals_the_sum_over_every_collapsing_path(labels, seed):
    log_posteriors = _random_log_posteriors(5, 3, seed)
    total = -np.inf
    for path in _paths(5, 3):
        if _collapse(path) == labels:
            total = np.logaddexp(total, sum(log_posteriors[t, c] for t, c in enumerate(path)))
    assert ctc_log_likelihood(log_posteriors, labels) == pytest.approx(total, abs=1e-9)


@pytest.mark.parametrize("labels", [(A,), (A, A), (A, B), (B, A, B)])
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_best_path_is_the_most_probable_collapsing_path(labels, seed):
    log_posteriors = _random_log_posteriors(6, 3, seed)
    best_path, best = None, -np.inf
    for path in _paths(6, 3):
        if _collapse(path) == labels:
            score = sum(log_posteriors[t, c] for t, c in enumerate(path))
            if score > best:
                best_path, best = path, score
    spans = ctc_best_path(log_posteriors, labels)
    expected, label_index, previous = [], -1, BLANK
    for t, token in enumerate(best_path):  # the brute-force path's per-label frame spans
        if token != BLANK and token != previous:
            label_index += 1
            expected.append([t, t])
        elif token != BLANK:
            expected[label_index][1] = t
        previous = token
    assert spans == [tuple(span) for span in expected]


def test_a_repeated_label_needs_a_blank_between_the_two():
    log_posteriors = np.log(np.full((2, 3), 1 / 3))
    assert ctc_log_likelihood(log_posteriors, (A, A)) == -np.inf
    assert np.isfinite(ctc_log_likelihood(log_posteriors, (A, B)))
    with pytest.raises(ValueError, match="cannot hold"):
        ctc_best_path(log_posteriors, (A, A))


def test_the_blank_is_not_a_label():
    with pytest.raises(ValueError, match="blank"):
        ctc_log_likelihood(np.zeros((3, 3)), (A, BLANK))


def test_a_dip_between_two_spikes_makes_the_doubled_transcript_likely():
    """Two A spikes with a blank between favour ``A A``; one spike favours ``A``."""
    def posteriors(rows):
        return np.log(np.array(rows, dtype=float))

    two_spikes = posteriors([[0.1, 0.89, 0.01], [0.89, 0.1, 0.01], [0.1, 0.89, 0.01]])
    one_spike = posteriors([[0.1, 0.89, 0.01], [0.97, 0.02, 0.01], [0.97, 0.02, 0.01]])
    log_ratio = lambda lp: ctc_log_likelihood(lp, (A, A)) - ctc_log_likelihood(lp, (A,))  # noqa: E731
    assert log_ratio(two_spikes) > 0 > log_ratio(one_spike)
    assert ctc_best_path(two_spikes, (A, A)) == [(0, 0), (2, 2)]

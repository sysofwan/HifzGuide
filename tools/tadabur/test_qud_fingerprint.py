"""Tests for tadabur.qud_fingerprint: one recording matches across a cut, two do not."""

from __future__ import annotations

import numpy as np

from tadabur.qud_fingerprint import BANDS, HOP, best_match, fingerprint


def _noise(seed: int, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(round(seconds * 16000)) * 0.1).astype(np.float32)


def test_a_fingerprint_is_one_32_bit_word_per_hop():
    prints = fingerprint(_noise(0, 3.0))
    assert prints.dtype == np.uint32 and BANDS == 32
    assert abs(len(prints) - (3.0 * 16000 - 5920) / HOP) < 3
    assert len(fingerprint(_noise(0, 0.2))) == 0


def test_the_same_audio_cut_later_matches_at_its_offset_and_other_audio_does_not():
    audio = _noise(1, 6.0)
    cut = audio[HOP * 40:]
    same = best_match(fingerprint(audio), fingerprint(cut))
    assert same.ber == 0.0 and same.offset == 40
    other = best_match(fingerprint(audio), fingerprint(_noise(2, 6.0)))
    assert other.ber > 0.4


def test_too_little_overlap_is_no_match():
    assert best_match(np.zeros(0, dtype=np.uint32), fingerprint(_noise(3, 3.0))) is None

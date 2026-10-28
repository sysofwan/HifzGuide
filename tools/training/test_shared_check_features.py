"""Tests for the shared-feature check path in ``training.decoding`` (#213); no GPU, no download.

A check sequence's pieces and their device features do not depend on the model, so a
multi-model decode can featurise an item once and run every decoder on the result. The contract:

* :func:`check_features` is the model-independent half: the item's check schedule and every
  piece's ``(pairs, real)``, exactly what :meth:`Decoder.device_pairs` computes;
* :meth:`Decoder.rows_from_pairs` and :meth:`Decoder.app_checks_from` are the model's half and
  return what :meth:`~Decoder.device_rows` and :meth:`~Decoder.app_checks_at` return;
* :func:`shared_extractor` refuses to share between decoders whose feature extractors are
  configured differently, since the shared features would not be what one of them computes.

The models here are tiny fakes whose argmax rows depend on the features and on a per-model
offset, so a decoder that read another's features, or none, would show in its rows.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from transformers import SeamlessM4TFeatureExtractor

from training import decoding as dc
from training.distill_data import SAMPLE_RATE
from training.distill_student import DEPLOYED_LOGIT_FRAMES, NUM_PHONEME_CLASSES
from training.test_decoding import _one_hot, _PairExtractor, _teacher_extractor

CPU = torch.device("cpu")
BLOCKS = (0, 1)


class _ConfiguredExtractor(_PairExtractor):
    """A pair extractor that reports its configuration as a Hugging Face extractor does."""

    def __init__(self, **config):
        super().__init__()
        self.config = config

    def to_dict(self) -> dict:
        return dict(self.config)


class _OffsetModel:
    """Emits, per output step, the class ``(feature pair at the step's first pair + offset)``
    modulo the class count, so its rows follow the features and differ between models."""

    def __init__(self, offset: int, steps: int = DEPLOYED_LOGIT_FRAMES):
        self.offset = offset
        self.steps = steps

    def __call__(self, features, attention_mask=None, return_dict=True):
        assert attention_mask is None and features.shape[1:] == (dc.DEVICE_FEATURE_PAIRS, 2)
        ids = (features[:, : 2 * self.steps : 2, 0].long() + self.offset) % NUM_PHONEME_CLASSES
        return {"logits": {"phonemes": torch.stack([_one_hot(r.numpy()) for r in ids])}}


def _decoder(offset: int, batch_size: int = 1, **config) -> dc.Decoder:
    return dc.Decoder(f"offset{offset}", _OffsetModel(offset), _ConfiguredExtractor(**config),
                      CPU, batch_size)


def _decoders() -> list[dc.Decoder]:
    """Three models over equal extractors; their batch sizes differ, as the features do not."""
    return [_decoder(offset, batch_size, padding_value=1.0)
            for offset, batch_size in ((0, 1), (7, 3), (19, 16))]


def _audio(seconds: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(seconds * SAMPLE_RATE)).astype(
        np.float32) * 0.1


def test_the_features_are_the_schedule_and_each_pieces_device_pairs():
    extractor = _teacher_extractor()
    samples = _audio(3.4)

    features = dc.check_features(extractor, samples)

    spans = dc.check_schedule(len(samples))
    assert features.spans == spans
    reference = dc.Decoder("x", torch.nn.Linear(1, 1), extractor, CPU)
    assert len(features.pairs) == len(spans)
    for (pairs, real), span in zip(features.pairs, spans):
        expected_pairs, expected_real = reference.device_pairs(samples[span.start : span.end])
        assert real == expected_real and torch.equal(pairs, expected_pairs)


def test_each_decoders_rows_from_the_features_equal_its_device_rows():
    samples = _audio(5.0)
    spans = dc.check_schedule(len(samples))
    pieces = [samples[s.start : s.end] for s in spans]
    features = dc.check_features(_ConfiguredExtractor(padding_value=1.0), samples)
    assert features.spans == spans

    shared_rows = []
    for decoder in _decoders():
        shared = decoder.rows_from_pairs(features.pairs)
        unshared = _decoder(decoder.model.offset, decoder.batch_size,
                            padding_value=1.0).device_rows(pieces)
        assert len(shared) == len(unshared) == len(spans)
        assert all(np.array_equal(a, b) for a, b in zip(shared, unshared))
        shared_rows.append(shared)
    first, second = shared_rows[:2]
    assert not all(np.array_equal(a, b) for a, b in zip(first, second))  # the models differ


def test_every_decoders_shared_checks_equal_its_unshared_checks():
    samples = _audio(7.0, seed=1)
    decoders = _decoders()

    features = dc.check_features(dc.shared_extractor(decoders), samples)
    shared = [decoder.app_checks_from(features, BLOCKS) for decoder in decoders]

    for decoder, checks in zip(decoders, shared):
        unshared = _decoder(decoder.model.offset, decoder.batch_size,
                            padding_value=1.0).app_checks_at(samples, BLOCKS)
        assert checks == unshared
        assert set(checks) == set(BLOCKS) and checks[1] != checks[0]
    assert len({json.dumps(checks, sort_keys=True) for checks in shared}) == 3  # all differ


def test_the_features_are_computed_once_per_piece_for_every_decoder():
    samples = _audio(6.0, seed=2)
    decoders = _decoders()
    extractors = [decoder.extractor for decoder in decoders]

    features = dc.check_features(dc.shared_extractor(decoders), samples)
    for decoder in decoders:
        decoder.app_checks_from(features, BLOCKS)

    spans = dc.check_schedule(len(samples))
    assert sum(len(e.lengths) for e in extractors) == len(spans)
    assert extractors[0].lengths == [s.end - s.start for s in spans]  # the first did it all


def test_app_checks_at_featurises_through_the_shared_half(monkeypatch):
    samples = _audio(4.0, seed=3)
    decoder = _decoder(5, batch_size=3, padding_value=1.0)
    calls = []
    real = dc.check_features
    monkeypatch.setattr(dc, "check_features",
                        lambda extractor, item: calls.append(extractor) or real(extractor, item))

    checks = decoder.app_checks_at(samples, BLOCKS)

    assert calls == [decoder.extractor]
    again = _decoder(5, batch_size=3, padding_value=1.0).app_checks_from(
        real(_ConfiguredExtractor(padding_value=1.0), samples), BLOCKS)
    assert checks == again
    assert decoder.app_checks(samples) == checks[dc.DEPLOYED_BLOCK]


def test_a_model_that_does_not_emit_125_steps_is_refused_from_the_features_too():
    features = dc.check_features(_ConfiguredExtractor(padding_value=1.0), _audio(2.0))
    decoder = dc.Decoder("short", _OffsetModel(0, steps=100), _ConfiguredExtractor(), CPU)

    with pytest.raises(ValueError, match="not 125"):
        decoder.rows_from_pairs(features.pairs)


def test_equal_extractors_share_and_the_first_is_the_one_used():
    first, second = _decoder(0, padding_value=1.0), _decoder(1, padding_value=1.0)

    assert dc.shared_extractor([first, second]) is first.extractor
    assert dc.shared_extractor([second]) is second.extractor


def test_two_teacher_extractors_built_alike_are_the_same_configuration():
    decoders = [dc.Decoder("a", torch.nn.Linear(1, 1), _teacher_extractor(), CPU),
                dc.Decoder("b", torch.nn.Linear(1, 1), _teacher_extractor(), CPU)]

    assert dc.shared_extractor(decoders) is decoders[0].extractor


def test_two_teacher_extractors_built_alike_compute_equal_features():
    """Production builds one extractor per decoder; the sharing premise is that their features
    are equal, not only their configurations."""
    samples = _audio(3.4, seed=5)
    first = dc.check_features(_teacher_extractor(), samples)
    second = dc.check_features(_teacher_extractor(), samples)

    assert first.spans == second.spans and len(first.pairs) == len(first.spans)
    for (pairs, real), (other_pairs, other_real) in zip(first.pairs, second.pairs):
        assert real == other_real and torch.equal(pairs, other_pairs)


def test_extractors_configured_differently_are_refused():
    alike = _decoder(0, padding_value=1.0)
    other = _decoder(1, padding_value=0.0)

    with pytest.raises(ValueError, match="feature extractor"):
        dc.shared_extractor([alike, other])


def test_a_teacher_extractor_with_another_padding_value_is_refused():
    teacher = dc.Decoder("a", torch.nn.Linear(1, 1), _teacher_extractor(), CPU)
    odd = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, padding_value=0.0,
                                      stride=2, sampling_rate=SAMPLE_RATE)

    with pytest.raises(ValueError, match="feature extractor"):
        dc.shared_extractor([teacher, dc.Decoder("b", torch.nn.Linear(1, 1), odd, CPU)])

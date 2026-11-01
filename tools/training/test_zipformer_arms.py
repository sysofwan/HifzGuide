"""Tests for the zipformer decoder's arm sharing and line reading (#245, self-check r1).

``spans`` and ``checks`` on one item must share one stream (two streams would double every
item's front end and ONNX cost across every screen set), and :func:`whole_item_text` must name
a missing arm rather than raise ``KeyError``. Torch-free, no ONNX file.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from training.zipformer_decode import BLANK_ID, ZipformerDecoder, whole_item_text


class _Session:
    """A cache-aware export that emits one known unit at every call's first frame."""

    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="x", shape=[1, 61, 80], type="tensor(float)"),
                SimpleNamespace(name="cached_len_0", shape=[1, 1], type="tensor(int64)")]

    def run(self, output_names, feed):
        self.calls += 1
        log_probs = np.full((1, 12, 251), -10.0, dtype=np.float32)
        log_probs[0, :, BLANK_ID] = 0.0
        log_probs[0, 0, 3] = 1.0
        return [log_probs, feed["cached_len_0"] + 1]


class _Fbank:
    def __call__(self, waveform, **options):
        return np.zeros((len(np.asarray(waveform)) // 160, 80), dtype=np.float32)


def _decoder() -> tuple[ZipformerDecoder, _Session]:
    session = _Session()
    decoder = ZipformerDecoder(session, {3: "ب", BLANK_ID: "<blank>"}, _Fbank(), scale=1.0)
    return decoder, session


def test_spans_and_checks_on_one_item_share_one_stream():
    decoder, session = _decoder()
    samples = np.zeros(16_000, dtype=np.float32)
    spans = decoder.spans(samples)
    calls = session.calls
    assert calls and spans == "ب" * calls
    checks = decoder.checks(samples)
    assert session.calls == calls  # the second arm sliced the stream the first one ran
    assert checks[-1]["hop"] == spans


def test_a_different_item_is_streamed_again():
    decoder, session = _decoder()
    samples = np.zeros(16_000, dtype=np.float32)
    decoder.spans(samples)
    calls = session.calls
    decoder.spans(samples.copy())
    assert session.calls > calls


def test_a_line_without_its_arm_is_refused():
    with pytest.raises(ValueError, match="spans"):
        whole_item_text({"checks": []}, "zipformer")
    with pytest.raises(ValueError, match="stream_b0"):
        whole_item_text({"spans": "x"}, "base")

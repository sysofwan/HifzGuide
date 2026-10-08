"""Tests for the torch-free parts of re-segmentation (#83): the decoder adapter, the pausal
reference wrapper and the per-clip segmentation record. The GPU driver itself is ``main``."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from tadabur.resegment import (
    DecoderSegmentationModel,
    segmentation_rows,
    with_pausal_taa_marbuta,
)
from tadabur.waqf_segments import SegmentRecord


class _FakeDecoder:
    def span_class_ids(self, spans):
        return [np.array([0, 3, 3, 0, 5], dtype=np.int64) for _ in spans]

    def decode_spans(self, spans):
        return [f"decode{len(span)}" for span in spans]


def test_the_adapter_serves_frame_ids_for_a_clip_and_strings_for_segments():
    from training.decoding import tokens_to_phonemes

    model = DecoderSegmentationModel(_FakeDecoder())
    clip = model.decode(np.zeros(10, dtype=np.float32), 16000)
    assert clip.class_ids == (0, 3, 3, 0, 5)
    assert clip.phonemes == tokens_to_phonemes([3, 5])
    assert [d.phonemes for d in model.decode_batch([np.zeros(4), np.zeros(7)], 16000)] == [
        "decode4", "decode7"]


def test_the_adapter_refuses_audio_at_another_rate():
    with pytest.raises(ValueError, match="16000 Hz"):
        DecoderSegmentationModel(_FakeDecoder()).decode(np.zeros(4), 8000)


def test_only_the_last_word_is_put_in_pausal_form():
    seen = []
    reference = with_pausal_taa_marbuta(lambda words: seen.append(words) or "x")
    rahmatan = "رَحْمَةًۭ"
    reference([rahmatan, "وَ", rahmatan])
    assert seen == [[rahmatan, "وَ", "رَحْمَةَ"]]


def _segment(index: int, clip: str = "a.wav") -> SegmentRecord:
    return SegmentRecord(clip, "2:2", 7, index, index, index + 3, float(index), index + 1.0,
                         f"ref{index}", (0, 2, 4))


def test_a_segmentation_row_lists_every_segment_and_which_were_kept():
    segments = [_segment(1), _segment(0), _segment(0, clip="b.wav")]
    kept = [{"clip_audio_filename": "a.wav", "segment_index": 1}]
    rows = segmentation_rows(segments, kept, {"a.wav": Counter(short_segment=1)})

    assert [r["audio_filename"] for r in rows] == ["a.wav", "b.wav"]
    first = rows[0]
    assert first["drops"] == {"short_segment": 1}
    assert [(s["segment_index"], s["kept"]) for s in first["segments"]] == [(0, False), (1, True)]
    assert first["segments"][1]["reference"] == "ref1"
    assert first["segments"][1]["raw_word_offsets"] == [0, 2, 4]
    assert rows[1]["drops"] == {}

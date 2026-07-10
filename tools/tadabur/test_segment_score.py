"""Unit tests for the per-segment decode+score pass (``tadabur.segment_score``).

Cover the model-free logic — segment-id derivation, waveform slicing/clamping,
row construction (normalized-once reference, contrasts, per-segment duration),
deterministic manifest writing, and audio staging — with a stub model so no GPU
or network is touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from .audio import TARGET_SAMPLE_RATE
from .audit_sampler import local_audio_path
from .normalization import normalize_phonemes
from . import segment_score
from .segment_score import (
    score_segments,
    segment_id,
    slice_segment,
    stage_segment_audio,
    write_segment_manifest,
)
from .waqf_segments import SegmentRecord


def _segment(index: int, start_s: float, end_s: float, ref: str) -> SegmentRecord:
    return SegmentRecord(
        audio_filename="clip.wav",
        surah_ayah="2:77",
        reciter_id=7,
        segment_index=index,
        word_start=index,
        word_end=index + 1,
        start_s=start_s,
        end_s=end_s,
        realized_reference_phonemes=ref,
    )


class _StubModel:
    """Returns a queued phoneme string per waveform, in call order."""

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)

    def decode_batch(self, waveforms, sample_rate):
        assert sample_rate == TARGET_SAMPLE_RATE
        return [
            SimpleNamespace(phonemes=self._outputs.pop(0), num_feature_frames=0,
                            num_logit_frames=0)
            for _ in waveforms
        ]


def _write_clip(clips_dir, name: str, seconds: float) -> None:
    clips_dir.mkdir(parents=True, exist_ok=True)
    n = int(seconds * TARGET_SAMPLE_RATE)
    sf.write(clips_dir / name, np.zeros(n, dtype=np.float32), TARGET_SAMPLE_RATE,
             subtype="PCM_16")


def test_segment_id_is_unique_per_index():
    a = segment_id(_segment(0, 0.0, 1.0, "ب"))
    b = segment_id(_segment(1, 1.0, 2.0, "ب"))
    assert a == "clip__seg0.wav"
    assert b == "clip__seg1.wav"
    assert a != b


def test_slice_segment_clamps_bounds():
    clip = np.arange(TARGET_SAMPLE_RATE, dtype=np.float32)  # 1 s ramp
    mid = slice_segment(clip, 0.25, 0.5)
    assert len(mid) == TARGET_SAMPLE_RATE // 4
    assert mid[0] == 4000.0
    # End overshooting the clip is clamped to its length (no error, no wraparound).
    tail = slice_segment(clip, 0.9, 2.0)
    assert len(tail) == TARGET_SAMPLE_RATE - int(0.9 * TARGET_SAMPLE_RATE)


def test_score_segments_builds_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_score, "_uthmani_words",
                        lambda sa: ["أَوَّلْ", "ثَانِي"])
    _write_clip(tmp_path, "clip.wav", 2.0)
    raw = "ءَننننَ"
    seg = _segment(0, 0.0, 1.5, raw)
    model = _StubModel(["ءنن"])

    rows = score_segments([seg], tmp_path, model)

    assert len(rows) == 1
    row = rows[0]
    assert row["audio_filename"] == "clip__seg0.wav"
    assert row["surah_ayah"] == "2:77"
    assert row["predicted_phonemes"] == "ءنن"
    # Reference is the realized phonemes normalized exactly once (raw kept too).
    assert row["reference_phonemes"] == normalize_phonemes(raw).normalized
    assert row["raw_reference_phonemes"] == raw
    assert row["uthmani"] == "أَوَّلْ"  # words[word_start:word_end]
    assert row["ayah_duration_s"] == 1.5
    assert row["reciter_id"] == 7
    assert isinstance(row["contrasts"], list)
    assert 0.0 <= row["match_ratio"] <= 1.0


def test_score_segments_orders_by_clip_then_index(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_score, "_uthmani_words", lambda sa: ["و"] * 5)
    _write_clip(tmp_path, "clip.wav", 3.0)
    # Pass out of order; rows must come back sorted by (audio_filename, index).
    segs = [_segment(2, 2.0, 3.0, "و"), _segment(0, 0.0, 1.0, "و"),
            _segment(1, 1.0, 2.0, "و")]
    model = _StubModel(["a", "b", "c"])  # decoded in sorted order
    rows = score_segments(segs, tmp_path, model)
    assert [r["segment_index"] for r in rows] == [0, 1, 2]


def test_write_segment_manifest_is_deterministic(tmp_path):
    rows = [
        {"audio_filename": "b.wav", "contrasts": ["shadda"], "match_ratio": 0.9},
        {"audio_filename": "a.wav", "contrasts": [], "match_ratio": 0.8},
    ]
    p1, p2 = tmp_path / "m1.jsonl", tmp_path / "m2.jsonl"
    write_segment_manifest(p1, rows)
    write_segment_manifest(p2, rows)
    assert p1.read_bytes() == p2.read_bytes()
    # Keys within each line are sorted for a stable diff.
    first = json.loads(p1.read_text(encoding="utf-8").splitlines()[0])
    assert list(first.keys()) == sorted(first.keys())


def test_stage_segment_audio_writes_local_audio_path(tmp_path):
    clips = tmp_path / "clips"
    _write_clip(clips, "clip.wav", 2.0)
    seg = _segment(1, 0.5, 1.0, "و")
    out = tmp_path / "seg_audio"
    stage_segment_audio([seg], clips, out)
    expected = out / local_audio_path(segment_id(seg))
    assert expected.is_file()
    waveform, sr = sf.read(expected, dtype="float32")
    assert sr == TARGET_SAMPLE_RATE
    assert len(waveform) == int(0.5 * TARGET_SAMPLE_RATE)

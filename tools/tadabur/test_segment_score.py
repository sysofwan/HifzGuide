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
from .manifest import ManifestRecord
from .normalization import normalize_phonemes
from . import segment_score
from .segment_score import (
    score_segments,
    segment_clips,
    segment_id,
    slice_segment,
    stage_segment_audio,
    write_segment_manifest,
)
from .waqf_segments import SegmentRecord
from .phoneme_vocab import PHONEME_ID_TO_CHAR


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


class _ClassIdModel:
    """Returns a queued per-frame class-id sequence per whole-clip ``decode`` call."""

    def __init__(self, class_ids_per_clip: list[list[int]]) -> None:
        self._queue = list(class_ids_per_clip)

    def decode(self, waveform, sample_rate):
        assert sample_rate == TARGET_SAMPLE_RATE
        return SimpleNamespace(class_ids=tuple(self._queue.pop(0)))



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

    rows, kept, drops = score_segments([seg], tmp_path, model)

    assert len(rows) == 1
    assert len(kept) == 1 and sum(drops.values()) == 0
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
    rows, _kept, _drops = score_segments(segs, tmp_path, model)
    assert [r["segment_index"] for r in rows] == [0, 1, 2]


def test_score_segments_drops_repeated_phrase_poison(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_score, "_uthmani_words", lambda sa: ["w"])
    _write_clip(tmp_path, "clip.wav", 2.0)
    ref = "بتثجحخدذرزسشصضطظعغفقكلمنهوي"
    good = _segment(0, 0.0, 1.0, ref)
    poison = _segment(1, 1.0, 2.0, ref)
    # Second decode repeats سشصضطظ (a 6-phoneme interior insertion run) → poison.
    model = _StubModel([ref, "بتثجحخدذرز" + "سشصضطظ" + "سشصضطظعغفقكلمنهوي"])

    rows, kept, drops = score_segments([good, poison], tmp_path, model)

    assert drops["repeated_phrase"] == 1
    assert [r["segment_index"] for r in rows] == [0]  # only the clean segment survives
    assert [s.segment_index for s in kept] == [0]  # audio staged for kept rows only


def test_score_segments_drops_boundary_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_score, "_uthmani_words", lambda sa: ["w"])
    _write_clip(tmp_path, "clip.wav", 3.0)
    # Two segments of one clip. seg0's audio overruns its reference at the *interior*
    # waqf split: its decode matches the reference then trails 9 extra phonemes the local
    # aligner trims (a repeat straddling the pause). seg0 is first (its leading edge is a
    # true clip start, ignored) but NOT last, so its trailing trim is an interior boundary
    # → boundary_mismatch. seg1 is the clip's last segment and decodes cleanly → kept.
    seg0 = _segment(0, 0.0, 1.5, "بتثجحخدذرز")
    seg1 = _segment(1, 1.5, 3.0, "قكلمنهوي")
    model = _StubModel(["بتثجحخدذرز" + "سشصضطظعغف", "قكلمنهوي"])

    rows, kept, drops = score_segments([seg0, seg1], tmp_path, model)

    assert drops["boundary_mismatch"] == 1
    assert [r["segment_index"] for r in rows] == [1]  # only the well-bounded segment
    assert [s.segment_index for s in kept] == [1]


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


# --- segment_clips (model-driven segmentation wiring) ------------------------

# Three "words" with distinct phoneme ids so the greedy decode never collapses
# adjacent duplicates (see test_waqf_detect for the algorithm itself).
_WORD0 = [2, 3, 4, 5, 6]
_WORD1 = [7, 8, 9, 10, 11, 12, 13, 14, 15]
_WORD2 = [16, 17, 18]
_SPF = 0.04


def _c(ids: list[int]) -> str:
    return "".join(PHONEME_ID_TO_CHAR[i] for i in ids)


def _phon(text: str) -> str:
    return {
        "w0 w1 w2": " ".join(_c(w) for w in (_WORD0, _WORD1, _WORD2)),
        "w0": _c(_WORD0),
        "w1 w2": _c(_WORD1 + _WORD2),
    }[text]


def _wordref(words: list[str]) -> tuple[str, list[int]]:
    """Stub of ``hafs_word_reference``: spaceless reference + per-word offsets."""
    assert words == ["w0", "w1", "w2"]
    return _c(_WORD0 + _WORD1 + _WORD2), [0, 5, 14, 17]


def _ids(spec: list[tuple[int, int]]) -> list[int]:
    out: list[int] = []
    for cid, count in spec:
        out += [cid] * count
    return out


def _manifest_record(name: str, surah_ayah: str) -> ManifestRecord:
    return ManifestRecord(
        audio_filename=name, surah_ayah=surah_ayah, match_ratio=0.9,
        ayah_duration_s=8.0, reciter_id=7, predicted_phonemes="",
    )


def test_segment_clips_splits_at_model_heard_pause(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_score, "_uthmani_words", lambda sa: ["w0", "w1", "w2"])
    class_ids = _ids(
        [(cid, 2) for cid in _WORD0] + [(0, 12)]
        + [(cid, 2) for cid in _WORD1 + _WORD2]
    )
    _write_clip(tmp_path, "a.wav", len(class_ids) * _SPF)
    model = _ClassIdModel([class_ids])
    pauses = {"a.wav": [(len(_WORD0) * 2 * _SPF, (len(_WORD0) * 2 + 12) * _SPF)]}

    records, skips = segment_clips(
        [_manifest_record("a.wav", "78:2")], tmp_path, model, _phon, _wordref, pauses
    )

    assert not skips
    assert [(r.word_start, r.word_end) for r in records] == [(0, 1), (1, 3)]
    assert records[0].realized_reference_phonemes == _phon("w0")
    assert records[1].realized_reference_phonemes == _phon("w1 w2")
    assert records[0].segment_index == 0 and records[1].segment_index == 1


def test_segment_clips_keeps_unsegmentable_clip_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_score, "_uthmani_words", lambda sa: ["w0", "w1", "w2"])
    unrelated = _ids([(cid, 2) for cid in [19, 20, 21, 22, 23, 24, 25, 26]])
    _write_clip(tmp_path, "a.wav", len(unrelated) * _SPF)
    model = _ClassIdModel([unrelated])
    pauses = {"a.wav": [(0.2, 0.5)]}  # a pause exists, but the clip won't align

    records, skips = segment_clips(
        [_manifest_record("a.wav", "78:2")], tmp_path, model, _phon, _wordref, pauses
    )

    assert skips["low_alignment"] == 1
    assert len(records) == 1
    assert (records[0].word_start, records[0].word_end) == (0, 3)
    assert records[0].realized_reference_phonemes == _phon("w0 w1 w2")


def test_segment_clips_keeps_clip_whole_when_no_pauses(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_score, "_uthmani_words", lambda sa: ["w0", "w1", "w2"])
    class_ids = _ids([(cid, 2) for cid in _WORD0 + _WORD1 + _WORD2])
    _write_clip(tmp_path, "a.wav", len(class_ids) * _SPF)
    model = _ClassIdModel([class_ids])

    records, skips = segment_clips(
        [_manifest_record("a.wav", "78:2")], tmp_path, model, _phon, _wordref, {}
    )

    assert not skips
    assert len(records) == 1
    assert (records[0].word_start, records[0].word_end) == (0, 3)


def test_segment_clips_tallies_missing_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_score, "_uthmani_words", lambda sa: ["w0", "w1", "w2"])
    model = _ClassIdModel([])  # decode never called — clip file absent
    records, skips = segment_clips(
        [_manifest_record("gone.wav", "78:2")], tmp_path, model, _phon, _wordref, {}
    )
    assert records == []
    assert skips["clip_missing"] == 1


def test_segment_clips_skips_unphonetizable_ayah(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_score, "_uthmani_words", lambda sa: ["w0", "w1", "w2"])
    class_ids = _ids([(cid, 2) for cid in _WORD0 + _WORD1 + _WORD2])
    _write_clip(tmp_path, "a.wav", len(class_ids) * _SPF)
    model = _ClassIdModel([class_ids])

    def _raises(_words):
        raise IndexError("leen madd on sukoon")

    records, skips = segment_clips(
        [_manifest_record("a.wav", "78:2")], tmp_path, model, _phon, _raises, {}
    )
    assert records == []
    assert skips["phonetizer_unsupported"] == 1

"""Tests for the Tadabur filtering pipeline (streaming/batching/scoring wiring).

These exercise clip parsing, batch scoring, and resumable orchestration with a fake
model and in-memory stream, so they need neither the GPU model nor network access.
The real ``.balanced`` gate (``tadabur.scorer``) is used unchanged.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from tadabur import filter as filter_mod
from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.filter import Clip, parse_clip, run_filter, score_batch
from tadabur.inference import PhonemeDecode
from tadabur.manifest import FilterManifest
from tadabur.scorer import BALANCED_SCORER

REFERENCES = {"3:82": "بتثج"}


def _wav_bytes(num_samples: int) -> bytes:
    tone = 0.1 * np.sin(np.linspace(0, 3.0, num_samples, endpoint=False)).astype(
        np.float32
    )
    buffer = io.BytesIO()
    sf.write(buffer, tone, TARGET_SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _clip(name: str, num_samples: int = TARGET_SAMPLE_RATE) -> Clip:
    return Clip(
        audio_filename=name,
        surah_ayah="3:82",
        reciter_id=88,
        audio_bytes=_wav_bytes(num_samples),
    )


class _FakeModel:
    """Returns a preset phoneme string per clip, ignoring the actual waveform."""

    def __init__(self, phonemes: list[str]) -> None:
        self._phonemes = phonemes

    def decode_batch(self, waveforms, sample_rate):
        assert sample_rate == TARGET_SAMPLE_RATE
        assert len(waveforms) == len(self._phonemes)
        return [PhonemeDecode(p, 0, 0) for p in self._phonemes]


def test_parse_clip_extracts_metadata():
    clip = parse_clip(
        {
            "audio": {"bytes": b"xyz", "path": "f.wav"},
            "audio_filename": "f.wav",
            "surah_id": 3,
            "ayah_id": 82,
            "reciter_id": 88,
        }
    )
    assert clip == Clip("f.wav", "3:82", 88, b"xyz")


@pytest.mark.parametrize("missing", ["audio_filename", "surah_id", "ayah_id", "reciter_id"])
def test_parse_clip_fails_loudly_on_missing_field(missing):
    row = {
        "audio": {"bytes": b"xyz"},
        "audio_filename": "f.wav",
        "surah_id": 3,
        "ayah_id": 82,
        "reciter_id": 88,
    }
    del row[missing]
    with pytest.raises(ValueError):
        parse_clip(row)


def test_parse_clip_fails_loudly_on_missing_audio():
    with pytest.raises(ValueError):
        parse_clip({"audio": {"bytes": None}, "audio_filename": "f.wav",
                    "surah_id": 3, "ayah_id": 82, "reciter_id": 88})


def test_score_batch_keeps_only_passers_with_computed_duration():
    clips = [_clip("pass.wav", TARGET_SAMPLE_RATE), _clip("fail.wav", TARGET_SAMPLE_RATE)]
    model = _FakeModel(["بتثج", "محك"])  # match, then unrelated

    records = score_batch(clips, model, REFERENCES, BALANCED_SCORER)

    assert [r.audio_filename for r in records] == ["pass.wav"]
    assert records[0].match_ratio == pytest.approx(1.0, abs=1e-3)
    assert records[0].ayah_duration_s == pytest.approx(1.0, abs=1e-6)
    assert records[0].surah_ayah == "3:82"
    assert records[0].reciter_id == 88


def test_score_batch_duration_reflects_actual_waveform():
    clips = [_clip("half.wav", TARGET_SAMPLE_RATE // 2)]
    model = _FakeModel(["بتثج"])

    (record,) = score_batch(clips, model, REFERENCES, BALANCED_SCORER)

    assert record.ayah_duration_s == pytest.approx(0.5, abs=1e-6)


def test_score_batch_attaches_contrasts_to_passers():
    # A soft-pair decode (ص for س) still passes and is tagged with that contrast;
    # a clean decode passes with no contrasts.
    references = {"3:82": "سلمن"}
    clips = [_clip("soft.wav"), _clip("clean.wav")]
    model = _FakeModel(["صلمن", "سلمن"])

    by_name = {r.audio_filename: r for r in score_batch(clips, model, references, BALANCED_SCORER)}

    assert by_name["soft.wav"].contrasts == ("\u0633\u2194\u0635",)  # س↔ص
    assert by_name["clean.wav"].contrasts == ()


def test_score_batch_fails_loudly_on_missing_reference():
    clips = [Clip("x.wav", "999:1", 88, _wav_bytes(TARGET_SAMPLE_RATE))]
    with pytest.raises(ValueError, match="No cached reference"):
        score_batch(clips, _FakeModel(["بتثج"]), REFERENCES, BALANCED_SCORER)


def test_run_filter_is_resumable(tmp_path, monkeypatch):
    all_clips = [_clip(f"c{i}.wav") for i in range(5)]

    def fake_stream(dataset_id, config_name, split, start, limit):
        rows = all_clips[start:]
        if limit is not None:
            rows = rows[:limit]
        return iter(rows)

    monkeypatch.setattr(filter_mod, "stream_clips", fake_stream)

    manifest_path = tmp_path / "subset.jsonl"

    # First run: only the first 2 clips (all pass under the identity model).
    with FilterManifest.open(manifest_path) as manifest:
        run_filter(
            manifest,
            _AllPassModel(),
            REFERENCES,
            BALANCED_SCORER,
            batch_size=2,
            limit=2,
        )
        assert manifest.clips_processed == 2

    # Resume: picks up at clip 2 and finishes the remaining 3.
    with FilterManifest.open(manifest_path) as manifest:
        assert manifest.clips_processed == 2
        run_filter(manifest, _AllPassModel(), REFERENCES, BALANCED_SCORER, batch_size=2)
        assert manifest.clips_processed == 5
        assert manifest.passers_written == 5

    names = [
        json.loads(line)["audio_filename"]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert names == ["c0.wav", "c1.wav", "c2.wav", "c3.wav", "c4.wav"]


class _AllPassModel:
    """Decodes every clip to the reference string, so all clips pass the gate."""

    def decode_batch(self, waveforms, sample_rate):
        return [PhonemeDecode("بتثج", 0, 0) for _ in waveforms]

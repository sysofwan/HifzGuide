"""Tests for training.nonpro_decode and the shared decode loop (#156); no GPU, no model.

A fake decoder stands in for both models; the clips are synthetic WAVs.
"""

from __future__ import annotations

import io
import json

import numpy as np
import soundfile as sf

from tadabur.nonpro_sources import QDAT, audio_sha256
from training import nonpro_decode as nd
from training.external_decode import decode_recordings


def _wav(seconds: float, rate: int = 16000, channels: int = 2) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros((int(seconds * rate), channels), dtype=np.float32), rate,
             format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _row(clip_id: str, audio: bytes) -> dict:
    return {"audio": {"path": f"{clip_id}.wav", "bytes": audio}, "id": clip_id,
            "original_id": "s4_2", "separate_tide": 1, "the_tight_noon": 0.0, "concealment": 1,
            "target": 0}


class FakeDecoder:
    def __init__(self):
        self.spans = 0

    def decode_spans(self, spans):
        self.spans += 1
        return [f"{len(spans[0])}"]

    def decode_stream(self, span, block, flush_tail):
        return "stream"

    def app_checks(self, span):
        return [{"hop": "x"}]


def test_a_clip_is_one_item_with_its_provenance_and_whole_span(monkeypatch):
    audio = _wav(0.5, rate=44100)
    monkeypatch.setattr(nd, "iter_rows", lambda dataset, shard, cache_dir: iter([_row("c1", audio)]))
    (recording,) = nd.recordings(QDAT, "s", None)
    assert recording.keys == ("c1#0",)
    samples, (record,) = recording.load()
    assert record == {"key": "c1#0", "audio_filename": "c1", "audio_sha256": audio_sha256(audio),
                      "surah_ayah": "5:109", "word_from": 8, "word_to": 15, "reciter": "s4",
                      "labels": {"separate_tide": 1, "the_tight_noon": 0, "concealment": 1,
                                 "target": 0},
                      "start_sample": 0, "end_sample": len(samples)}
    assert len(samples) == 8000  # resampled to 16 kHz mono


def test_the_loop_decodes_each_item_once_and_resumes(tmp_path, monkeypatch):
    rows = [_row("c1", _wav(0.25)), _row("c2", _wav(0.5))]
    monkeypatch.setattr(nd, "iter_rows", lambda dataset, shard, cache_dir: iter(rows))
    decoders = {"base": FakeDecoder(), "h448": FakeDecoder()}
    paths = {m: nd.decodes_path(tmp_path, "qdat", 0, m) for m in decoders}
    paths["h448"].write_text(json.dumps({"key": "c1#0"}) + "\n")  # an earlier run's record
    timing = decode_recordings(nd.recordings(QDAT, "s", None), decoders, paths, "qdat 0")
    assert timing["items"] == 2 and decoders["h448"].spans == 1
    lines = [json.loads(line) for line in paths["base"].read_text().splitlines()]
    assert [(line["key"], line["spans"], line["checks"]) for line in lines] == \
        [("c1#0", "4000", [{"hop": "x"}]), ("c2#0", "8000", [{"hop": "x"}])]
    again = decode_recordings(nd.recordings(QDAT, "s", None), decoders, paths, "qdat 0")
    assert again["items"] == 0  # every item is already decoded: no audio is read

"""Whole-clip exposure is recorded only for a clip that was read (#311 PR3, reviewer r7 R7.3).

The page's Whole control asks ``/api/audio?...&whole=1``; the server treats that as the owner
having heard the whole clip and rewrites an answer already given to say so. Reading the clip must
come first: a WAV that vanished mid-sitting returns 500 (S3.6) and must leave both the persisted
flag and the next save's flag exactly as they were, or the committed provenance claims an
exposure that never happened.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from tadabur.catch_loss_worklist import CatchLossClip, CatchLossRow
from tadabur.followalong_audit.s5_ui import SessionState
from tadabur.tashkeel_audit_ui import page_key
from tadabur.word_snips import ALIGNMENT, Snip

SAMPLE_RATE = 16_000
DIGEST = "a" * 64
SITE = "0000000000000001"


def _write_clip(path: Path, frames: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(SAMPLE_RATE)
        sink.writeframes(b"\x01\x00" * frames)


def _session(directory: Path, verdicts: Path) -> SessionState:
    """One served word whose snip is an excerpt, so ``whole=1`` is the path that records."""
    row = CatchLossRow(CatchLossClip(SITE, "2:3", "w", "clip.wav", DIGEST, 0, SAMPLE_RATE),
                       0, (0, 1))
    snip = Snip(SITE, DIGEST, 100, 900, ALIGNMENT)
    return SessionState([row], {SITE: snip}, wav_dir=directory, manifest={}, verdicts=verdicts,
                        stored={})


def _stored(verdicts: Path) -> dict:
    return json.loads(verdicts.read_text(encoding="utf-8"))


def test_a_failed_whole_read_records_no_exposure_and_the_next_save_keeps_it_false(tmp_path):
    source = tmp_path / f"{DIGEST}.wav"
    _write_clip(source)
    verdicts = tmp_path / "verdicts.jsonl"
    state = _session(tmp_path, verdicts)
    key = page_key(SITE)
    state.record(key, "correct", [], None, "first", None)
    answer = {"verdict": "correct", "kinds": [], "sub_reason": None, "note": "first"}
    assert _stored(verdicts)["whole_played"] is False

    source.unlink()
    with pytest.raises(OSError):
        state.audio(key, True)
    assert _stored(verdicts)["whole_played"] is False      # nothing was read, nothing recorded

    state.record(key, "correct", [], None, "second", answer)
    assert _stored(verdicts)["whole_played"] is False      # nor does the next save invent it

    _write_clip(source)
    assert state.audio(key, True).startswith(b"RIFF")      # a clip that reads does count
    assert _stored(verdicts)["whole_played"] is True

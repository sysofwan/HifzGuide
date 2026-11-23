"""The fetch's WAV digest manifest (#307).

The fetch writes a PCM_16 WAV outside the repo and records its SHA-256 and sample count; the run
verifies every WAV against that record, so a truncated rsync or a substituted file fails the clip
instead of grading different audio under the same clip's name.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from tadabur.followalong_audit.clips import verify_wav
from tadabur.tlog_index import sha256_file

FRAMES = 1_000


def write_wav(path: Path, fill: bytes = b"\x00\x00", frames: int = FRAMES) -> Path:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16_000)
        out.writeframes(fill * frames)
    return path


def test_the_wav_the_fetch_wrote_is_accepted_and_returns_its_digest(tmp_path):
    path = write_wav(tmp_path / "clip.wav")
    expected = {"wav_sha256": sha256_file(path), "samples": FRAMES}
    assert verify_wav(path, expected) == sha256_file(path)


def test_other_bytes_are_refused(tmp_path):
    path = write_wav(tmp_path / "clip.wav", fill=b"\x01\x00")
    with pytest.raises(ValueError, match="not the fetched"):
        verify_wav(path, {"wav_sha256": sha256_file(path), "samples": FRAMES - 1})


def test_a_different_length_than_the_fetch_recorded_is_refused(tmp_path):
    path = write_wav(tmp_path / "clip.wav")
    with pytest.raises(ValueError, match="samples"):
        verify_wav(path, {"wav_sha256": sha256_file(path), "samples": FRAMES + 1})


def test_a_clip_the_manifest_does_not_hold_is_refused(tmp_path):
    path = write_wav(tmp_path / "clip.wav")
    with pytest.raises(ValueError, match="no entry"):
        verify_wav(path, None)

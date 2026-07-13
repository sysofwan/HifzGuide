"""Tests for the eval harness's torch-free join/data-integrity logic (issue #7).

The model pass itself needs a GPU and audio, but the fixture↔reference↔audio join —
which must fail loudly rather than silently drop a clip and bias the metrics — is pure
and tested here."""

from __future__ import annotations

import pytest

from tadabur.audit_sampler import local_audio_path
from tadabur.eval_fixtures import ACCEPT, EvalFixtureEntry
from tadabur.eval_harness import _prepare_clips


def _entry(clip_id: str) -> EvalFixtureEntry:
    return EvalFixtureEntry(clip_id, clip_id, "2:255", "shadda", ACCEPT)


def _make_audio(audio_dir, clip_id: str) -> None:
    (audio_dir / local_audio_path(clip_id)).write_bytes(b"RIFFfake")


def test_prepare_pairs_clip_with_its_audio(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _make_audio(audio_dir, "a")
    prepared = _prepare_clips([_entry("a")], {"a": "\u0628\u0646\u0628"}, audio_dir)
    assert [e.clip_id for e, _ in prepared] == ["a"]
    assert prepared[0][1] == audio_dir / local_audio_path("a")


def test_missing_reference_fails_loudly(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _make_audio(audio_dir, "a")
    with pytest.raises(FileNotFoundError, match="reference"):
        _prepare_clips([_entry("a")], {}, audio_dir)


def test_missing_audio_fails_loudly(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="audio"):
        _prepare_clips([_entry("a")], {"a": "\u0628\u0646\u0628"}, audio_dir)

"""Unit tests for the counterfactual recording UI's pure logic.

Covers the take-filename contract the recording sheet already promises, the upload
validation that keeps unreadable audio out of the output directory, the directory-as-state
resume behaviour, and the item view the page renders — all without binding a socket.

The WAV fixtures here are byte-assembled with the *same* RIFF layout the page's
``encodeWav`` writes (16 kHz mono 16-bit PCM), rather than via ``soundfile``, so these tests
fail if the browser-side encoder's format contract ever drifts from what the scorer's loader
can read.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from training.counterfactual_script import CounterfactualItem

from .audio import decode_to_mono_16k
from .counterfactual_record_ui import (
    CONTROL,
    COUNTERFACTUAL,
    MIN_TAKE_SECONDS,
    RecordingSession,
    TakeStore,
    item_view,
    load_items,
    take_filename,
    validate_take,
)

FATHA, DAMMA = "\u064e", "\u064f"


def browser_wav(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """A WAV encoded exactly the way the page's ``encodeWav`` does — header written by hand."""
    count = int(seconds * sample_rate)
    tone = np.sin(2 * np.pi * 220 * np.arange(count) / sample_rate) * 0.5
    pcm = (tone * 0x7FFF).astype("<i2").tobytes()
    return b"".join([
        b"RIFF", struct.pack("<I", 36 + len(pcm)), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16),
        b"data", struct.pack("<I", len(pcm)), pcm,
    ])


def _item(item_id: str = "cf000", word_index: int = 1) -> CounterfactualItem:
    return CounterfactualItem(
        item_id=item_id,
        surah_ayah="12:94",
        segment_text="\u0642\u064e\u062f\u0652 \u0623\u064e\u0646 \u0642\u064e\u0627\u0644\u064e",
        word_index=word_index,
        target_word="\u0623" + FATHA + "\u0646",
        canonical_vowel=FATHA,
        spoken_vowel=DAMMA,
        reference_phonemes="\u0621\u064e\u06ba",
        audio_filename="tadabur_spk0075.wav",
    )


def _session(tmp_path, items=None) -> RecordingSession:
    return RecordingSession(items or [_item("cf000"), _item("cf001")], TakeStore(tmp_path / "out"))


def test_take_filename_matches_the_recording_sheet():
    # The sheet's take_1_file / take_2_file columns are these exact names, so the scorer
    # finds the audio by construction rather than by a recorded path.
    assert take_filename("cf000", CONTROL) == "cf000_control.wav"
    assert take_filename("cf000", COUNTERFACTUAL) == "cf000_counterfactual.wav"
    with pytest.raises(ValueError):
        take_filename("cf000", "take3")


def test_load_items_reads_the_frozen_manifest(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(
        '{"item_id": "cf000", "surah_ayah": "12:94", "segment_text": "a b", "word_index": 1,'
        ' "target_word": "b", "canonical_vowel": "\\u064e", "spoken_vowel": "\\u064f",'
        ' "reference_phonemes": "x", "audio_filename": "f.wav"}\n\n',
        encoding="utf-8",
    )
    items = load_items(path)
    assert [i.item_id for i in items] == ["cf000"]
    assert items[0].word_index == 1


def test_a_browser_encoded_wav_loads_through_the_scorers_loader():
    # The one property the whole UI hangs on: what the page writes is what the pipeline reads.
    waveform = decode_to_mono_16k(browser_wav(0.75))
    assert waveform.dtype == np.float32
    assert len(waveform) == pytest.approx(12000, abs=2)


def test_validate_take_returns_duration():
    assert validate_take(browser_wav(1.5)) == pytest.approx(1.5, abs=0.01)
    # A non-16 kHz source is fine: the loader resamples, so the duration is what matters.
    assert validate_take(browser_wav(1.0, sample_rate=48000)) == pytest.approx(1.0, abs=0.01)


def test_validate_take_rejects_audio_the_pipeline_cannot_read():
    # Chrome's MediaRecorder default. soundfile cannot open WebM and there is no ffmpeg here,
    # so this must fail at upload rather than after all 94 clips are recorded.
    webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 4096
    with pytest.raises(ValueError, match="cannot decode"):
        validate_take(webm)
    with pytest.raises(ValueError, match="empty"):
        validate_take(b"")


def test_validate_take_rejects_a_misclick():
    too_short = browser_wav(MIN_TAKE_SECONDS / 2)
    with pytest.raises(ValueError, match="recite the whole phrase"):
        validate_take(too_short)


def test_saving_a_take_writes_the_sheets_filename(tmp_path):
    session = _session(tmp_path)
    result = session.save_take("cf000", CONTROL, browser_wav(1.0))
    written = tmp_path / "out" / "cf000_control.wav"
    assert written.is_file()
    assert result["item"]["takes"][CONTROL]["recorded"] is True
    assert result["progress"]["takes_recorded"] == 1
    assert result["progress"]["items_done"] == 0  # the counterfactual take is still missing
    # No half-written staging file is left behind to be mistaken for a finished take.
    assert list((tmp_path / "out").glob("*.part")) == []


def test_an_item_is_done_only_with_both_takes(tmp_path):
    session = _session(tmp_path)
    session.save_take("cf000", CONTROL, browser_wav(1.0))
    result = session.save_take("cf000", COUNTERFACTUAL, browser_wav(1.0))
    assert result["item"]["done"] is True
    assert result["progress"]["items_done"] == 1


def test_a_rejected_take_is_not_written(tmp_path):
    session = _session(tmp_path)
    with pytest.raises(ValueError):
        session.save_take("cf000", CONTROL, b"\x1a\x45\xdf\xa3" + b"\x00" * 4096)
    assert list((tmp_path / "out").iterdir()) == []


def test_re_recording_overwrites_the_take(tmp_path):
    session = _session(tmp_path)
    session.save_take("cf000", CONTROL, browser_wav(1.0))
    result = session.save_take("cf000", CONTROL, browser_wav(2.0))
    assert result["item"]["takes"][CONTROL]["seconds"] == pytest.approx(2.0, abs=0.01)
    assert result["progress"]["takes_recorded"] == 1


def test_progress_resumes_from_the_output_directory(tmp_path):
    _session(tmp_path).save_take("cf000", CONTROL, browser_wav(1.0))
    # A fresh server over the same directory sees the same progress: the directory is the state.
    resumed = _session(tmp_path).state()
    assert resumed["progress"]["takes_recorded"] == 1
    assert resumed["items"][0]["takes"][CONTROL]["recorded"] is True
    assert resumed["items"][0]["takes"][COUNTERFACTUAL]["recorded"] is False
    assert resumed["items"][1]["done"] is False


def test_an_unknown_item_is_rejected(tmp_path):
    session = _session(tmp_path)
    with pytest.raises(KeyError):
        session.save_take("cf999", CONTROL, browser_wav(1.0))
    # An item id becomes a filename, so a traversal attempt must not write outside out-dir.
    with pytest.raises(KeyError):
        session.save_take("../escape", CONTROL, browser_wav(1.0))
    assert list((tmp_path / "out").iterdir()) == []


def test_an_unknown_take_is_rejected(tmp_path):
    session = _session(tmp_path)
    with pytest.raises(ValueError):
        session.save_take("cf000", "take3", browser_wav(1.0))


def test_item_view_carries_what_the_reciter_needs(tmp_path):
    view = item_view(_item(), TakeStore(tmp_path / "out"))
    # The phrase is pre-split on the same whitespace tokenization word_index is defined
    # against, so the page highlights the word the sheet points at.
    assert len(view["words"]) == 3
    assert view["words"][view["word_index"]] == view["target_word"]
    # The literal target for take 2: the same word with its one vowel replaced.
    assert view["spoken_word"] == "\u0623" + DAMMA + "\u0646"
    assert view["canonical_vowel"] == "fatha (a)"
    assert view["spoken_vowel"] == "damma (u)"
    assert view["done"] is False

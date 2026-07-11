"""Tests for the torch-free clip-staging half of the waqf pipeline.

The segmentation + scoring moved to :mod:`tadabur.segment_score` (it needs the model,
see :mod:`tadabur.waqf_detect`); what remains here is staging whole passing clips to
local disk and the shared realized-reference label vocabulary. The HF stream is
monkeypatched so no dataset is downloaded.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import tadabur.waqf_segments as ws
from tadabur.manifest import ManifestRecord
from tadabur.waqf_segments import SegmentRecord, _spaceless_word_offsets


def _passing(name: str, surah_ayah: str, predicted: str = "") -> ManifestRecord:
    return ManifestRecord(
        audio_filename=name,
        surah_ayah=surah_ayah,
        match_ratio=0.9,
        ayah_duration_s=8.0,
        reciter_id=106,
        predicted_phonemes=predicted,
    )


# --- SegmentRecord ----------------------------------------------------------


def test_segment_record_is_a_frozen_value():
    record = SegmentRecord(
        audio_filename="a.wav", surah_ayah="78:2", reciter_id=1, segment_index=0,
        word_start=0, word_end=2, start_s=0.0, end_s=1.5,
        realized_reference_phonemes="ءَب",
    )
    assert record.word_end == 2
    with pytest.raises(AttributeError):
        record.segment_index = 3  # frozen


# --- clip staging -----------------------------------------------------------


def test_stage_clips_decodes_and_saves_each_found_clip(tmp_path, monkeypatch):
    record = _passing("a.wav", "78:2")
    row = {"audio": {"bytes": b""}}
    saved: list = []
    monkeypatch.setattr(ws, "_stream_passing_rows", lambda *a, **k: iter([(row, record)]))
    monkeypatch.setattr(ws, "decode_to_mono_16k", lambda b: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(
        ws, "_save_local_clip", lambda audio_dir, name, wf: saved.append((name, len(wf)))
    )

    skips = ws.stage_clips([record], audio_dir=tmp_path)

    assert saved == [("a.wav", 16000)]
    assert "missing_due_to_limit" not in skips


def test_full_build_raises_when_a_passing_clip_is_never_streamed(tmp_path, monkeypatch):
    # A full build (no --limit) whose stream drops a passing clip would stage a
    # partial clip set; that must fail loudly, naming the misses.
    monkeypatch.setattr(ws, "_stream_passing_rows", lambda *a, **k: iter(()))
    passing = [_passing("a.wav", "78:2"), _passing("b.wav", "78:2")]
    with pytest.raises(ValueError, match="were not found"):
        ws.stage_clips(passing, audio_dir=tmp_path)


def test_limited_run_records_missing_due_to_limit_instead_of_raising(tmp_path, monkeypatch):
    # A --limit smoke run may legitimately stop before reaching every clip; the
    # shortfall is tallied, not raised.
    monkeypatch.setattr(ws, "_stream_passing_rows", lambda *a, **k: iter(()))
    passing = [_passing("a.wav", "78:2"), _passing("b.wav", "78:2")]
    skips = ws.stage_clips(passing, audio_dir=tmp_path, limit=1)
    assert skips["missing_due_to_limit"] == 2


def test_found_clips_are_not_reported_missing(tmp_path, monkeypatch):
    record = _passing("a.wav", "78:2")
    row = {"audio": {"bytes": b""}}
    monkeypatch.setattr(ws, "_stream_passing_rows", lambda *a, **k: iter([(row, record)]))
    monkeypatch.setattr(ws, "decode_to_mono_16k", lambda b: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(ws, "_save_local_clip", lambda *a, **k: None)

    skips = ws.stage_clips([record], audio_dir=tmp_path)
    assert "missing_due_to_limit" not in skips


def test_shard_staging_records_clips_outside_shards_without_raising(tmp_path, monkeypatch):
    # A --shards scan is deliberately partial (it stages only clips living in the given
    # shards), so a passing clip not in them is tallied, never raised — even though
    # ``limit`` is None (which would raise on a full stream build).
    staged = _passing("in_shard.wav", "78:2")
    absent = _passing("elsewhere.wav", "78:2")
    row = {"audio": {"bytes": b""}}

    captured = {}

    def fake_stream(passing, dataset_id, config_name, split, limit, row_source):
        captured["row_source"] = row_source
        return iter([(row, staged)])

    monkeypatch.setattr(ws, "_stream_passing_rows", fake_stream)
    monkeypatch.setattr(ws, "decode_to_mono_16k", lambda b: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(ws, "_save_local_clip", lambda *a, **k: None)
    monkeypatch.setattr("tadabur.shard_reader.iter_shard_rows",
                        lambda indices, **k: iter([{"marker": list(indices)}]))

    skips = ws.stage_clips([staged, absent], audio_dir=tmp_path, shards="0-1")
    assert skips["missing_outside_shards"] == 1
    assert "missing_due_to_limit" not in skips
    # A shard row source (not the datasets stream) was threaded into the streamer.
    assert next(captured["row_source"]) == {"marker": [0, 1]}


# --- _spaceless_word_offsets ------------------------------------------------


def test_spaceless_word_offsets_strips_spaces_and_remaps_boundaries():
    # Phonetizer kept a word-separating space ("XY ZW"); the model decode has none,
    # so the reference is stripped to "XYZW" and boundaries remapped to that string.
    # mappings is indexed by input char of "ab cd"; word starts are chars 0 and 3.
    words = ["ab", "cd"]
    mappings = [
        SimpleNamespace(pos=(0, 1)),  # 'a' -> phoneme index 0
        SimpleNamespace(pos=(1, 2)),  # 'b'
        SimpleNamespace(pos=(2, 3)),  # ' '
        SimpleNamespace(pos=(3, 4)),  # 'c' -> phoneme index 3 (after the space)
        SimpleNamespace(pos=(4, 5)),  # 'd'
    ]
    reference, boundaries = _spaceless_word_offsets("XY ZW", mappings, words)
    assert reference == "XYZW"
    assert boundaries == [0, 2, 4]


def test_spaceless_word_offsets_handles_wasl_merged_words():
    # A wasl merge leaves no separating space ("XYZW"); the per-word boundary still
    # comes from the second word's input-char mapping, not a split-on-space.
    words = ["ab", "cd"]
    mappings = [
        SimpleNamespace(pos=(0, 1)),  # 'a'
        SimpleNamespace(pos=(1, 2)),  # 'b'
        SimpleNamespace(pos=(2, 2)),  # ' ' elided at the wasl (zero-width)
        SimpleNamespace(pos=(2, 3)),  # 'c' -> phoneme index 2 (no gap)
        SimpleNamespace(pos=(3, 4)),  # 'd'
    ]
    reference, boundaries = _spaceless_word_offsets("XYZW", mappings, words)
    assert reference == "XYZW"
    assert boundaries == [0, 2, 4]


# --- torch-free offline stage -----------------------------------------------


def test_importing_waqf_segments_does_not_import_torch():
    # The stage must run in the plain macOS/CPU env without pulling in the GPU
    # inference path; a fresh interpreter proves nothing imports torch.
    code = (
        "import sys; import tadabur.waqf_segments; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr

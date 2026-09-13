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
    # surah_id is 0-indexed in Tadabur; canonical surah is surah_id + 1.
    assert clip == Clip("f.wav", "4:82", 88, b"xyz")


def test_canonical_surah_ayah_shifts_zero_indexed_surah():
    # Tadabur labels Al-Naba (78th surah, ayah 30) as surah_id=77; the reference
    # cache is canonical 1-indexed, so the surah must shift by one, ayah stays.
    assert filter_mod.canonical_surah_ayah(77, 30) == "78:30"
    assert filter_mod.canonical_surah_ayah(0, 1) == "1:1"
    assert filter_mod.canonical_surah_ayah(113, 6) == "114:6"


def test_parse_clip_falls_back_to_audio_path():
    # The fast ``preview`` config has no top-level audio_filename; the basename
    # comes from audio.path instead.
    clip = parse_clip(
        {
            "audio": {"bytes": b"xyz", "path": "tadabur_spk0106_S77_A30_x_000016.wav"},
            "surah_id": 77,
            "ayah_id": 30,
            "reciter_id": 106,
        }
    )
    assert clip == Clip("tadabur_spk0106_S77_A30_x_000016.wav", "78:30", 106, b"xyz")


def test_resolve_audio_filename_prefers_top_level_then_path_then_fails():
    assert filter_mod.resolve_audio_filename(
        {"audio_filename": "top.wav", "audio": {"path": "other.wav"}}
    ) == "top.wav"
    assert filter_mod.resolve_audio_filename(
        {"audio": {"path": "/nested/dir/only.wav"}}
    ) == "only.wav"
    with pytest.raises(ValueError):
        filter_mod.resolve_audio_filename({"audio": {"bytes": b"x"}})


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

    scored = score_batch(clips, model, REFERENCES, BALANCED_SCORER)

    assert [r.audio_filename for r in scored.passers] == ["pass.wav"]
    assert scored.passers[0].match_ratio == pytest.approx(1.0, abs=1e-3)
    assert scored.passers[0].ayah_duration_s == pytest.approx(1.0, abs=1e-6)
    assert scored.passers[0].surah_ayah == "3:82"
    assert scored.passers[0].reciter_id == 88
    # The failing clip lands in the reject list rather than vanishing.
    assert [r.audio_filename for r in scored.rejects] == ["fail.wav"]


def test_score_batch_skips_over_long_clips_before_decode():
    from tadabur.filter import MAX_AYAH_DURATION_S

    over_len = int((MAX_AYAH_DURATION_S + 1.0) * TARGET_SAMPLE_RATE)
    clips = [
        _clip("long.wav", over_len),
        _clip("ok.wav", TARGET_SAMPLE_RATE),
    ]
    # The fake model asserts it receives exactly one waveform: the over-long clip
    # must be dropped before the GPU decode so it never enters the batch.
    model = _FakeModel(["بتثج"])

    scored = score_batch(clips, model, REFERENCES, BALANCED_SCORER)

    assert [r.audio_filename for r in scored.passers] == ["ok.wav"]
    # A clip dropped before the decode was never gated, so it is not a reject either.
    assert scored.rejects == []


def test_score_batch_duration_reflects_actual_waveform():
    clips = [_clip("half.wav", TARGET_SAMPLE_RATE // 2)]
    model = _FakeModel(["بتثج"])

    (record,) = score_batch(clips, model, REFERENCES, BALANCED_SCORER).passers

    assert record.ayah_duration_s == pytest.approx(0.5, abs=1e-6)


def test_score_batch_attaches_contrasts_to_passers():
    # A soft-pair decode (ص for س) still passes and is tagged with that contrast;
    # a clean decode passes with no contrasts.
    references = {"3:82": "سلمن"}
    clips = [_clip("soft.wav"), _clip("clean.wav")]
    model = _FakeModel(["صلمن", "سلمن"])

    scored = score_batch(clips, model, references, BALANCED_SCORER)
    by_name = {r.audio_filename: r for r in scored.passers}

    assert by_name["soft.wav"].contrasts == ("\u0633\u2194\u0635",)  # س↔ص
    assert by_name["clean.wav"].contrasts == ()


def test_score_batch_fails_loudly_on_missing_reference():
    clips = [Clip("x.wav", "999:1", 88, _wav_bytes(TARGET_SAMPLE_RATE))]
    with pytest.raises(ValueError, match="No cached reference"):
        score_batch(clips, _FakeModel(["بتثج"]), REFERENCES, BALANCED_SCORER)


def test_score_batch_skips_unknown_refs_when_opted_in():
    # A non-canonical clip (e.g. preview's 1:77) is skipped, not fatal; the
    # canonical clip in the same batch still scores.
    clips = [
        Clip("bad.wav", "1:77", 88, _wav_bytes(TARGET_SAMPLE_RATE)),
        _clip("good.wav", TARGET_SAMPLE_RATE),
    ]
    model = _FakeModel(["محك", "بتثج"])  # bad ref (skipped), then a clean match
    scored = score_batch(clips, model, REFERENCES, BALANCED_SCORER, skip_unknown_refs=True)
    assert [r.audio_filename for r in scored.passers] == ["good.wav"]
    # The unknown-reference clip is skipped, not recorded as a gate reject.
    assert scored.rejects == []


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


def test_shard_clip_source_resumes_by_whole_shard(monkeypatch):
    from tadabur.filter import _shard_clip_source

    # Two shards of raw Tadabur-shaped rows (audio struct + id ints).
    def _row(surah_id, ayah_id, tag):
        return {
            "audio": {"bytes": _wav_bytes(TARGET_SAMPLE_RATE), "path": f"{tag}.wav"},
            "surah_id": surah_id,
            "ayah_id": ayah_id,
            "reciter_id": 3,
        }

    shards = {
        0: [_row(2, i, f"s0_c{i}") for i in range(3)],
        1: [_row(2, i, f"s1_c{i}") for i in range(3)],
    }

    def fake_iter(indices, **kwargs):
        for idx in indices:
            yield from shards[idx]

    monkeypatch.setattr("tadabur.shard_reader.iter_shard_rows", fake_iter)
    monkeypatch.setattr("tadabur.shard_reader.ROWS_PER_SHARD", 3)

    # 3 clips already processed = one finished shard, so resume skips shard 0.
    clips = list(_shard_clip_source(
        "0-1", clips_processed=3, dataset_id="d",
        shard_cache=None, delete_shards=False, limit=None,
    ))
    assert [c.audio_filename for c in clips] == ["s1_c0.wav", "s1_c1.wav", "s1_c2.wav"]
    # canonical surah shift (surah_id 2 -> "3:*") is applied by parse_clip.
    assert clips[0].surah_ayah == "3:0"


def test_shard_clip_source_applies_limit(monkeypatch):
    from tadabur.filter import _shard_clip_source

    rows = [
        {"audio": {"bytes": _wav_bytes(TARGET_SAMPLE_RATE), "path": f"c{i}.wav"},
         "surah_id": 0, "ayah_id": i, "reciter_id": 1}
        for i in range(10)
    ]
    monkeypatch.setattr("tadabur.shard_reader.iter_shard_rows",
                        lambda indices, **k: iter(rows))
    monkeypatch.setattr("tadabur.shard_reader.ROWS_PER_SHARD", 1000)

    clips = list(_shard_clip_source(
        "0", clips_processed=0, dataset_id="d",
        shard_cache=None, delete_shards=False, limit=4,
    ))
    assert [c.audio_filename for c in clips] == ["c0.wav", "c1.wav", "c2.wav", "c3.wav"]


# --- the reject sink (ADR-0016 decision 2) -----------------------------------------

# A 20-phoneme reference and a decode repeating its phonemes 5..10 in the middle — the
# clean-re-read shape; see tadabur.test_rejects for why these exact strings.
RE_READ_REFERENCES = {"3:82": "بتثجحخدذرزسشصضطظعغفق"}
RE_READ_DECODE = (
    RE_READ_REFERENCES["3:82"][:10]
    + RE_READ_REFERENCES["3:82"][5:10]
    + RE_READ_REFERENCES["3:82"][10:]
)


def test_score_batch_classifies_a_clean_re_read_and_hands_back_its_waveform():
    from tadabur.rejects import CAUSE_INSERTION_RUN

    clips = [_clip("reread.wav", TARGET_SAMPLE_RATE), _clip("junk.wav", TARGET_SAMPLE_RATE)]
    model = _FakeModel([RE_READ_DECODE, "مهنيول"])

    scored = score_batch(clips, model, RE_READ_REFERENCES, BALANCED_SCORER)

    assert scored.passers == []
    assert [r.audio_filename for r in scored.rejects] == ["reread.wav", "junk.wav"]
    assert scored.rejects[0].causes == (CAUSE_INSERTION_RUN,)
    # Only the clean re-read is handed back for staging, paired with the 16 kHz
    # waveform the gate actually scored.
    assert [name for name, _ in scored.clean_re_read_audio] == ["reread.wav"]
    assert len(scored.clean_re_read_audio[0][1]) == TARGET_SAMPLE_RATE


def test_run_filter_writes_rejects_and_stages_clean_re_read_audio(tmp_path, monkeypatch):
    from tadabur.rejects import read_reject_records

    clips = [_clip("reread.wav"), _clip("junk.wav"), _clip("pass.wav")]
    decodes = [RE_READ_DECODE, "مهنيول", RE_READ_REFERENCES["3:82"]]
    monkeypatch.setattr(
        filter_mod, "stream_clips", lambda *a, **k: iter(clips)
    )

    manifest_path = tmp_path / "subset.jsonl"
    rejects_path = tmp_path / "rejects.jsonl"
    audio_dir = tmp_path / "reject_audio"
    with FilterManifest.open(manifest_path, rejects_path=rejects_path) as manifest:
        run_filter(
            manifest,
            _FakeModel(decodes),
            RE_READ_REFERENCES,
            BALANCED_SCORER,
            batch_size=3,
            reject_audio_dir=audio_dir,
        )
        assert manifest.passers_written == 1
        assert manifest.rejects_written == 2

    assert [r.audio_filename for r in read_reject_records(rejects_path)] == [
        "reread.wav",
        "junk.wav",
    ]
    # Only the clean re-read is staged — the sink records every reject, the audio
    # directory holds only the ones the corpus will replay.
    assert sorted(p.name for p in audio_dir.iterdir()) == ["reread.wav"]


class _ScriptedModel:
    """Decodes clips in order from a script, across however many batches they arrive in."""

    def __init__(self, phonemes: list[str]) -> None:
        self._remaining = list(phonemes)

    def decode_batch(self, waveforms, sample_rate):
        assert sample_rate == TARGET_SAMPLE_RATE
        taken, self._remaining = (
            self._remaining[: len(waveforms)],
            self._remaining[len(waveforms):],
        )
        return [PhonemeDecode(p, 0, 0) for p in taken]


def test_omitting_rejects_leaves_the_manifest_byte_identical(tmp_path, monkeypatch):
    # The acceptance guarantee: adding the sink changes nothing for a run that does not
    # ask for it — same manifest bytes, same checkpoint, and no new files on disk.
    # Run over two batches, since the sink writes on the same per-batch commit path.
    clips = [_clip("reread.wav"), _clip("junk.wav"), _clip("pass.wav")]
    decodes = [RE_READ_DECODE, "مهنيول", RE_READ_REFERENCES["3:82"]]
    monkeypatch.setattr(filter_mod, "stream_clips", lambda *a, **k: iter(clips))

    def _run(directory, rejects_path):
        directory.mkdir()
        manifest_path = directory / "subset.jsonl"
        with FilterManifest.open(manifest_path, rejects_path=rejects_path) as manifest:
            run_filter(
                manifest,
                _ScriptedModel(decodes),
                RE_READ_REFERENCES,
                BALANCED_SCORER,
                batch_size=2,
            )
        return manifest_path

    without = _run(tmp_path / "without", None)
    with_sink = _run(tmp_path / "with", tmp_path / "with" / "rejects.jsonl")

    assert without.read_bytes() == with_sink.read_bytes()
    assert (
        (tmp_path / "without" / "subset.jsonl.progress.json").read_bytes()
        == (tmp_path / "with" / "subset.jsonl.progress.json").read_bytes()
    )
    # Nothing but the manifest and its checkpoint exists in the plain run.
    assert sorted(p.name for p in (tmp_path / "without").iterdir()) == [
        "subset.jsonl",
        "subset.jsonl.progress.json",
    ]


def test_reject_sink_resumes_without_duplicating_a_replayed_batch(tmp_path, monkeypatch):
    from tadabur.rejects import read_reject_records

    clips = [_clip("j0.wav"), _clip("j1.wav"), _clip("j2.wav")]
    manifest_path = tmp_path / "subset.jsonl"
    rejects_path = tmp_path / "rejects.jsonl"

    def _stream(dataset_id, config_name, split, start, limit):
        rows = clips[start:]
        return iter(rows[:limit] if limit is not None else rows)

    monkeypatch.setattr(filter_mod, "stream_clips", _stream)

    # First run stops after two clips; the resume must not re-record them.
    with FilterManifest.open(manifest_path, rejects_path=rejects_path) as manifest:
        run_filter(
            manifest, _FakeModel(["مهنيول"] * 2), RE_READ_REFERENCES,
            BALANCED_SCORER, batch_size=2, limit=2,
        )
    with FilterManifest.open(manifest_path, rejects_path=rejects_path) as manifest:
        assert manifest.clips_processed == 2
        assert manifest.rejects_written == 2
        run_filter(
            manifest, _FakeModel(["مهنيول"]), RE_READ_REFERENCES,
            BALANCED_SCORER, batch_size=2,
        )
        assert manifest.clips_processed == 3

    assert [r.audio_filename for r in read_reject_records(rejects_path)] == [
        "j0.wav", "j1.wav", "j2.wav",
    ]

"""Tests for the torch-free half of bleed staging (``tadabur.bleed_stage``).

Only the population selection is covered here — which clips the stage will decode, and
which it reports as unstaged. The decode and VAD passes need a GPU and the weights, and
are exercised by the corpus run itself.
"""

from __future__ import annotations

import json

from tadabur.bleed_stage import clean_re_reads_with_audio
from tadabur.rejects import CAUSE_INSERTION_RUN, CAUSE_LOW_RATIO, RejectRecord


def _reject(name: str, **overrides) -> RejectRecord:
    fields = {
        "audio_filename": name,
        "surah_ayah": "3:82",
        "reciter_id": 1,
        "ayah_duration_s": 10.0,
        "match_ratio": 0.8,
        "max_insertion_run": 7,
        "leading_trim": 0,
        "trailing_trim": 0,
        "added_shadda": False,
        "predicted_phonemes": "بتثج",
        "causes": (CAUSE_INSERTION_RUN,),
    }
    return RejectRecord(**{**fields, **overrides})


def _write(tmp_path, records, staged):
    sink = tmp_path / "rejects.jsonl"
    with sink.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({**record.__dict__, "causes": list(record.causes)}) + "\n")
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in staged:
        (clips / name).write_bytes(b"")
    return sink, clips


def test_only_clean_re_reads_are_staged(tmp_path):
    # A reject that failed on ratio alone carries no repeat, so it is not corpus
    # material and must not be decoded — the stage is not a re-run of the whole sink.
    records = [
        _reject("repeat.wav"),
        _reject("ratio.wav", max_insertion_run=0, match_ratio=0.4, causes=(CAUSE_LOW_RATIO,)),
        _reject("shadda.wav", added_shadda=True),
    ]
    sink, clips = _write(tmp_path, records, ["repeat.wav", "ratio.wav", "shadda.wav"])

    present, missing = clean_re_reads_with_audio(sink, clips)

    assert [r.audio_filename for r in present] == ["repeat.wav"]
    assert missing == []


def test_a_clean_re_read_with_no_staged_wav_is_reported_not_dropped_silently(tmp_path):
    # A resumed filter run stages audio only for the shards it walked, so the sink can
    # name clips whose WAV never landed. The corpus shrinks by exactly those; the
    # caller has to be told which.
    records = [_reject("here.wav"), _reject("gone.wav")]
    sink, clips = _write(tmp_path, records, ["here.wav"])

    present, missing = clean_re_reads_with_audio(sink, clips)

    assert [r.audio_filename for r in present] == ["here.wav"]
    assert missing == ["gone.wav"]

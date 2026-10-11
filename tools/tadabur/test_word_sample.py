"""Ship A's random word sample (#126): the frame, the draw, the file. No model involved."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tadabur.mining_pool import FRAME_PATH as POOL_FRAME_PATH
from tadabur.mining_pool import PoolClip, PoolSegment
from tadabur.ship_a import HELD_OUT, TRAIN, reciter_half
from tadabur.staged_audio import StagedClip
from tadabur.word_sample import (
    RECITERS,
    WORD_SAMPLE_PATH,
    WORD_SUMMARY_PATH,
    draw,
    frame_words,
    load_words,
    parse_word,
    uniform_probabilities,
    word_json,
    write_words,
)

HELD = [r for r in range(100) if reciter_half(r) == HELD_OUT][:3]
TRAINED = next(r for r in range(100) if reciter_half(r) == TRAIN)
REFERENCE = "aa bb cc"


def _clip(name: str, reciter: int, strata=("uniform",), segments=None) -> PoolClip:
    segment = PoolSegment(0, 2, 5, 16000, 160000, REFERENCE, (0, 3, 6, 8), True)
    return PoolClip(name, strata, 0.5, "2:2", reciter, 6, None, 0, 6, 0.0, 10.0,
                    (0.0, 0.5, 1.0, 3.0, 5.0, 7.0, 9.0), tuple(segments or (segment,)))


def _registry(clips) -> dict[str, StagedClip]:
    return {c.audio_filename: StagedClip(c.audio_filename, 77, 5, c.reciter_id, "2:2", 200000,
                                         "b" * 64, ("mining_pool",)) for c in clips}


def test_the_frame_is_the_held_out_uniform_clips_kept_words():
    dropped = PoolSegment(1, 5, 6, 160000, 190000, "dd", (0, 2), False)
    clips = [_clip("a.wav", HELD[0]), _clip("b.wav", TRAINED),
             _clip("c.wav", HELD[1], strata=("consonant_pair",)),
             replace(_clip("d.wav", HELD[2]), segments=(_clip("x", 0).segments[0], dropped))]
    words = frame_words(clips, frozenset())
    assert set(words) == {HELD[0], HELD[2]}
    assert [w.word_index for w in words[HELD[0]]] == [2, 3, 4]
    assert [w.reference_index for w in words[HELD[0]]] == [0, 3, 6]


def test_a_barred_reciter_is_refused():
    with pytest.raises(ValueError):
        frame_words([_clip("a.wav", HELD[0])], frozenset({HELD[0]}))


def test_one_word_per_drawn_reciter_with_its_inclusion_probability():
    clips = [_clip(f"{r}.wav", r) for r in HELD]
    words = frame_words(clips, frozenset())
    rows = draw(words, {r: 0.25 for r in HELD}, _registry(clips), reciters=2)
    assert len(rows) == 2 and len({row.site.reciter_id for row in rows}) == 2
    for row in rows:
        assert row.clip_inclusion_probability == 0.25
        assert row.draw_probability == pytest.approx(2 / 3 / 3)
        assert row.inclusion_probability == pytest.approx(0.25 * 2 / 9)
        assert row.site.stratum_population == 3
        start, end = row.word_span
        assert REFERENCE[start:end].strip() and row.excerpt_start_sample < row.excerpt_end_sample
    assert draw(words, {r: 0.25 for r in HELD}, _registry(clips), reciters=2) == rows


def test_the_draw_never_depends_on_frame_order():
    clips = [_clip(f"{r}.wav", r) for r in HELD]
    registry = _registry(clips)
    forward = draw(frame_words(clips, frozenset()), {r: 1.0 for r in HELD}, registry, 2)
    backward = draw(frame_words(clips[::-1], frozenset()), {r: 1.0 for r in HELD}, registry, 2)
    assert forward == backward


def test_a_word_row_round_trips_and_a_bad_one_is_refused(tmp_path):
    clips = [_clip(f"{r}.wav", r) for r in HELD]
    rows = draw(frame_words(clips, frozenset()), {r: 1.0 for r in HELD}, _registry(clips), 3)
    write_words(rows, tmp_path / "w.jsonl")
    assert load_words(tmp_path / "w.jsonl") == rows
    data = word_json(rows[0])
    for change in ({"word_index": 9}, {"inclusion_probability": 0.5},
                   {"site_id": "word_sample:other.wav#0@2"}, {"excerpt_end_sample": 0}):
        with pytest.raises(ValueError):
            parse_word({**data, **change}, "test")


def test_the_committed_sample_is_held_out_and_agrees_with_its_summary():
    rows = load_words(WORD_SAMPLE_PATH)
    summary = json.loads(WORD_SUMMARY_PATH.read_text(encoding="utf-8"))
    uniform = uniform_probabilities(json.loads(POOL_FRAME_PATH.read_text(encoding="utf-8")))
    assert len(rows) == summary["words"] == RECITERS
    assert len({row.site.reciter_id for row in rows}) == RECITERS
    for row in rows:
        assert reciter_half(row.site.reciter_id) == HELD_OUT
        assert row.clip_inclusion_probability == uniform[row.site.reciter_id]
        assert row.site.stratum_population == summary["frame"]["words_per_reciter"][
            str(row.site.reciter_id)]
    assert summary["listening_minutes"] == round(
        sum(row.listening_seconds() for row in rows) / 60, 1)

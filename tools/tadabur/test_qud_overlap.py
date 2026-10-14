"""Tests for tadabur.qud_overlap: the timing distance, comparable ayat and the decision lines."""

from __future__ import annotations

import json

import pytest

from tadabur import qud_overlap as qo
from tadabur.qud_source import QudRow

WORDS = ["ٱلرَّحْمَـٰنُ", "عَلَى", "ٱلْعَرْشِ", "ٱسْتَوَىٰ"]


def _row(words, starts_ms) -> QudRow:
    return QudRow("s", 20, 5, 4000, " ".join(words), ((1, len(words), 0, 4000),),
                  tuple((i + 1, s, s + 300) for i, s in enumerate(starts_ms)), "u", 0)


def test_timing_distance_ignores_the_offset_between_cuts():
    assert qo.timing_distance([0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]) == 0.0
    assert qo.timing_distance([0.0, 1.0, 2.0, 3.0], [0.0, 1.4, 2.0, 3.0]) == 0.0  # one outlier
    assert qo.timing_distance([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]) == 1.0
    with pytest.raises(ValueError):
        qo.timing_distance([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])


def test_only_an_ayah_recited_once_in_order_with_enough_words_is_compared():
    assert qo.qud_word_starts(_row(WORDS, [0, 1000, 2000, 3000]), WORDS) == [0.0, 1.0, 2.0, 3.0]
    repeated = [*WORDS[:2], WORDS[1], *WORDS[2:]]
    assert qo.qud_word_starts(_row(repeated, [0, 500, 1000, 2000, 3000]), WORDS) is None
    assert qo.qud_word_starts(_row(WORDS[:3], [0, 1000, 2000]), WORDS[:3]) is None


def _scores(positives, negatives, cross):
    return {"synthetic_positives": [{"ber": b} for b in positives],
            "hard_negatives": [{"ber": b} for b in negatives],
            "cross_aligner": cross}


def test_the_lines_come_from_the_controls_alone():
    cross = [
        {"ber": 0.05, "timing_distance": 0.04, "same_reciter": True, "tadabur_reciter": 1, "slug": "a"},
        {"ber": 0.08, "timing_distance": 0.06, "same_reciter": True, "tadabur_reciter": 1, "slug": "a"},
        {"ber": 0.42, "timing_distance": 0.05, "same_reciter": True, "tadabur_reciter": 2, "slug": "b"},
        {"ber": 0.45, "timing_distance": 0.30, "same_reciter": False, "tadabur_reciter": 1, "slug": "c"},
    ]
    lines = qo.decide_lines(_scores([0.02, 0.10], [0.30, 0.40], cross))
    assert lines["fingerprint"]["line"] == pytest.approx(0.2)
    assert lines["timing"]["line"] == pytest.approx(qo.TIMING_MARGIN * 0.06)
    assert lines["timing"]["positives"] == 2
    assert lines["timing"]["same_reciter_negatives_passing"] == 1
    assert lines["timing"]["other_reciter_negatives_passing"] == 0
    assert lines["aggregate"]["negative_pass_rate"] == [1, 2]
    assert lines["aggregate"]["control_overlaps"] == 1  # reciter 1 in mushaf a


def test_no_gap_between_positives_and_negatives_means_no_line():
    lines = qo.decide_lines(_scores([0.02, 0.35], [0.30], []))
    assert lines["fingerprint"]["line"] is None and lines["timing"]["line"] is None


def test_the_committed_controls_leave_the_pilot_reciters_out():
    controls = json.loads(qo.CONTROLS_PATH.read_text(encoding="utf-8"))
    left_out = set(controls["pilot_tadabur_ids_left_out"])
    assert left_out
    assert not {c["tadabur_reciter"] for c in controls["synthetic_positives"]} & left_out
    assert not {c["tadabur_reciter"] for c in controls["cross_aligner"]} & left_out
    slugs = {q["slug"] for c in controls["cross_aligner"] for q in [*c["qud"], c["other"]]}
    assert not slugs & set(qo.PILOT)


def test_the_binomial_tail_is_one_at_zero_and_small_far_above_the_rate():
    assert qo.binomial_tail(0, 10, 0.1) == 1.0
    assert qo.binomial_tail(10, 10, 0.1) == pytest.approx(1e-10)
    assert qo.binomial_tail(1, 1, 0.3) == pytest.approx(0.3)


def test_stage_one_flags_a_reciter_whose_pairs_keep_their_rhythm(monkeypatch):
    words = [f"w{i}" for i in range(6)]
    rows = [QudRow("s", 2, a, 1000, " ".join(words), ((1, 6, 0, 1000),),
                   tuple((i + 1, 1000 * i, 1000 * i + 900) for i in range(6)), "u", 0)
            for a in range(1, 21)]
    same = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    other = [0.0, 1.5, 2.2, 3.9, 4.1, 6.3]
    by_ayah = {f"2:{a}": [
        {"reciter_id": 7, "starts": same, "audio_filename": f"a{a}", "shard": a, "row_index": 0},
        {"reciter_id": 9, "starts": other, "audio_filename": f"b{a}", "shard": a, "row_index": 1},
    ] for a in range(1, 21)}
    monkeypatch.setattr(qo, "_canonical_words", lambda key: words)
    screened = qo.screen_mushaf(rows, by_ayah, 0.05, 0.12, named=[9])
    assert screened["flagged_reciters"] == [7]
    whys = {c["reciter_id"]: c["why"] for c in screened["candidates"]}
    assert whys == {7: "flagged", 9: "named"}
    assert len(screened["candidates"]) == 2 * qo.CONFIRM_PAIRS

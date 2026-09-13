"""Tests for the reject-yield arithmetic behind the shard budget."""

from __future__ import annotations

import math

from tadabur.rejects import (
    CAUSE_ADDED_SHADDA,
    CAUSE_INSERTION_RUN,
    CAUSE_LOW_RATIO,
    RejectRecord,
)
from tadabur.reject_yield import compute_yield, shards_for


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


def test_rates_are_taken_against_clips_processed_not_scored_clips():
    # Three clips were consumed; one passed, one was rejected, and one was dropped
    # before the gate (over-long). The third belongs in neither bucket but must still
    # be in the denominator, or the pass rate overstates itself.
    result = compute_yield(passers=1, rejects=[_reject("a.wav")], clips_processed=3)

    assert result.skipped_before_gate == 1
    assert result.pass_rate == round(1 / 3, 4)
    assert result.reject_rate == round(1 / 3, 4)


def test_clean_re_read_shares_are_reported_against_both_denominators():
    rejects = [
        _reject("clean.wav"),
        _reject("short.wav", max_insertion_run=2, causes=(CAUSE_LOW_RATIO,)),
        # A long repeat that dragged its own ratio down — admitted since #66 dropped the
        # floor, because the floor was filtering repeat length rather than correctness.
        _reject("longrepeat.wav", match_ratio=0.4, causes=(CAUSE_LOW_RATIO,)),
        _reject("shadda.wav", added_shadda=True, causes=(CAUSE_ADDED_SHADDA,)),
    ]
    result = compute_yield(passers=96, rejects=rejects, clips_processed=100)

    assert result.clean_re_reads == 2
    # Richness of the reject pile is what sizes a budget; share of all clips is what
    # sizes the stream.
    assert result.clean_re_read_share_of_rejects == 0.5
    assert result.clean_re_read_share_of_clips == 0.02


def test_causes_overlap_and_are_counted_independently():
    rejects = [
        _reject("both.wav", match_ratio=0.4, causes=(CAUSE_LOW_RATIO, CAUSE_INSERTION_RUN)),
        _reject("ratio.wav", match_ratio=0.3, causes=(CAUSE_LOW_RATIO,)),
    ]
    result = compute_yield(passers=0, rejects=rejects, clips_processed=2)

    assert result.cause_counts == {CAUSE_LOW_RATIO: 2, CAUSE_INSERTION_RUN: 1}
    # Shares sum past 100% because a clip can fail on more than one condition.
    assert result.cause_shares_of_rejects == {CAUSE_LOW_RATIO: 1.0, CAUSE_INSERTION_RUN: 0.5}


def test_causes_with_no_clips_are_omitted():
    result = compute_yield(passers=0, rejects=[_reject("a.wav")], clips_processed=1)
    assert result.cause_counts == {CAUSE_INSERTION_RUN: 1}


def test_both_argued_thresholds_get_a_histogram_over_all_rejects():
    rejects = [
        _reject("a.wav", match_ratio=0.66, max_insertion_run=5),
        _reject("b.wav", match_ratio=0.75, max_insertion_run=5),
        _reject("c.wav", match_ratio=0.79, max_insertion_run=12),
    ]
    result = compute_yield(passers=0, rejects=rejects, clips_processed=3)

    assert result.insertion_run_histogram == {"12": 1, "5": 2}
    # The two bars the predicate was argued over — the gate's 0.65 and the dropped 0.75
    # floor — land on bucket edges, so neither is straddled by the bucket meant to
    # justify it.
    assert result.match_ratio_histogram == {"0.65-0.70": 1, "0.75-0.80": 2}


def test_a_perfect_ratio_lands_in_the_top_bucket():
    result = compute_yield(passers=0, rejects=[_reject("a.wav", match_ratio=1.0)], clips_processed=1)
    assert result.match_ratio_histogram == {"0.95-1.00": 1}


def test_clean_re_read_spread_covers_reciters_ayat_and_duration():
    rejects = [
        _reject("a.wav", reciter_id=1, surah_ayah="3:82", ayah_duration_s=10.0),
        _reject("b.wav", reciter_id=1, surah_ayah="3:83", ayah_duration_s=12.5),
        _reject("c.wav", reciter_id=2, surah_ayah="3:82", ayah_duration_s=7.5),
        # Not a clean re-read: excluded from every spread figure below.
        _reject("d.wav", reciter_id=9, max_insertion_run=1, causes=(CAUSE_LOW_RATIO,)),
    ]
    result = compute_yield(passers=0, rejects=rejects, clips_processed=4)

    assert result.clean_re_read_reciters == 2
    assert result.clean_re_read_ayat == 2
    assert result.clean_re_read_audio_seconds == 30.0
    assert result.clean_re_read_top_reciters == [(1, 2), (2, 1)]


def test_shard_budget_extrapolates_the_measured_rate():
    # 5 clean re-reads from 1 shard ⇒ 100 shards for 500.
    assert shards_for(target=500, clean_re_reads=5, shards_run=1) == 100.0
    assert shards_for(target=500, clean_re_reads=20, shards_run=2) == 50.0


def test_a_zero_yield_run_cannot_size_a_budget():
    # Returning a large integer would dress a failed measurement up as a plan.
    assert math.isinf(shards_for(target=500, clean_re_reads=0, shards_run=1))


def test_an_empty_run_produces_zero_rates_rather_than_dividing_by_zero():
    result = compute_yield(passers=0, rejects=[], clips_processed=0)
    assert result.pass_rate == 0.0
    assert result.clean_re_read_share_of_rejects == 0.0


def test_the_corpus_repeat_lengths_are_reported_apart_from_every_reject_s():
    # The threshold argument reads every reject; the question "did the corpus keep the
    # long re-reads?" reads only the matching subset. A reject with no repeat at all
    # belongs in the first histogram and must not dilute the second.
    rejects = [
        _reject("no-repeat.wav", max_insertion_run=0, causes=(CAUSE_LOW_RATIO,)),
        _reject("short.wav", max_insertion_run=5),
        _reject("long.wav", max_insertion_run=18),
        _reject("shadda.wav", max_insertion_run=9, added_shadda=True,
                causes=(CAUSE_INSERTION_RUN, CAUSE_ADDED_SHADDA)),
    ]

    result = compute_yield(passers=0, rejects=rejects, clips_processed=4)

    assert result.insertion_run_histogram == {"0": 1, "18": 1, "5": 1, "9": 1}
    assert result.clean_re_read_run_histogram == {"18": 1, "5": 1}

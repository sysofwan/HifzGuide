"""Tests for the haraka-gap diagnosis (#85): classification, arm bookkeeping, statistics.

Torch-free and model-free: the clip, its segments and every decode are synthetic. The
fixture is one 9 s recitation of seven words in two waqf segments, which the frozen 5 s /
4 s-hop window build cuts into two windows that share word 3.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tadabur.mining_pool import PoolClip, PoolSegment
from training.decoding import (
    CONFIRM_TIMESTEPS,
    DEPLOYED_LOGIT_FRAMES,
    WINDOW_SAMPLES,
    clip_windows,
    commit_bounds,
)
from training.haraka_gap import WEIGHTS_DTYPE  # noqa: F401  (the CLI imports torch-free)
from training.haraka_gap_arms import (
    BLOCK_MID,
    CLIP_NOT_WINDOWED,
    EMPTY,
    FIRST_AT_PAUSE,
    FIRST_MID_SEGMENT,
    FLUSH,
    INTERIOR,
    LAST_AT_PAUSE,
    LAST_MID_SEGMENT,
    MATCHED,
    ONLY_WORD,
    SEAM_END,
    SEAM_START,
    SEGMENT,
    SEGMENT_ALL,
    STARTUP,
    STREAM,
    STREAM_CLASSES,
    TIME_NEIGHBOUR,
    TIME_OWN,
    UNDECODED,
    WINDOW,
    WINDOW_OCC,
    WINDOWED,
    JointKey,
    Observation,
    Site,
    build_arms,
    model_observations,
    plan_clip,
    stream_region,
    time_in_segment,
    unit_outcomes,
    window_position,
    windowed_population,
    word_position,
)
from training.haraka_gap_report import (
    ADR0005_SEGMENT_RECALL,
    ADR0007_WINDOW_RECALL,
    Ledger,
    build_report,
    contribution,
    rate,
    render_markdown,
)
from training.windowed_labels import WindowLabel

RATE = 16_000
FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
# Seven words; word 3 sits inside the two windows' 1 s overlap ([4, 5) s).
WORD_TIMES = (0.0, 1.4, 2.8, 4.1, 4.9, 6.5, 8.0, 9.0)
WORDS = (
    "ب" + FATHA + "ت" + FATHA + " ",
    "ث" + DAMMA + " ",
    "ج" + KASRA + "د ",
    "ر" + FATHA + "ز" + KASRA + " ",
    "س" + DAMMA + " ",
    "ش" + FATHA + " ",
    "ص" + KASRA,
)


def _segment(index: int, word_start: int, word_end: int, kept: bool = True) -> PoolSegment:
    words = WORDS[word_start:word_end]
    offsets = [0]
    for word in words:
        offsets.append(offsets[-1] + len(word))
    return PoolSegment(
        segment_index=index,
        word_start=word_start,
        word_end=word_end,
        start_sample=round(WORD_TIMES[word_start] * RATE),
        end_sample=round(WORD_TIMES[word_end] * RATE),
        reference="".join(words),
        raw_word_offsets=tuple(offsets),
        kept=kept,
    )


def _clip(name: str = "a.wav", reciter: int = 1, probability: float = 0.5,
          segments: tuple[PoolSegment, ...] | None = None, re_reads: int = 0) -> PoolClip:
    return PoolClip(
        audio_filename=name,
        strata=("uniform",),
        inclusion_probability=probability,
        surah_ayah="1:1",
        reciter_id=reciter,
        n_words=len(WORDS),
        skip_reason=None,
        re_reads=re_reads,
        recited_words=len(WORDS),
        recitation_start_s=0.0,
        recitation_end_s=WORD_TIMES[-1],
        word_times=WORD_TIMES,
        segments=segments if segments is not None else (_segment(0, 0, 5), _segment(1, 5, 7)),
    )


def _window(word_start: int, word_end: int) -> WindowLabel:
    return WindowLabel("a.wav", "1:1", 1, 0, 0, 80_000, 0, 250, 125, "", word_start, word_end, (0,))


# --- Classification ---------------------------------------------------------------------


def test_window_position_splits_edges_by_whether_they_are_segment_edges():
    seg = _segment(0, 2, 6)
    window = _window(3, 6)
    assert window_position(3, window, seg) == FIRST_MID_SEGMENT
    assert window_position(4, window, seg) == INTERIOR
    assert window_position(5, window, seg) == LAST_AT_PAUSE
    assert window_position(2, _window(2, 5), seg) == FIRST_AT_PAUSE
    assert window_position(4, _window(2, 5), seg) == LAST_MID_SEGMENT
    assert window_position(4, _window(4, 5), seg) == ONLY_WORD
    with pytest.raises(ValueError):
        window_position(6, window, seg)


def test_word_position_names_the_haraka_among_its_words_harakat():
    seg = _segment(0, 0, 3)  # "بَتَ ثُ جِد "
    assert [word_position(seg, r) for r in (1, 3, 6, 9)] == ["first", "last", "sole", "sole"]
    long_word = PoolSegment(0, 0, 1, 0, 1, "ب" + FATHA + "ت" + DAMMA + "ث" + KASRA, (0, 6), True)
    assert [word_position(long_word, r) for r in (1, 3, 5)] == ["first", "inner", "last"]


def _committing_window(seconds: float, num_samples: int, block: int) -> int:
    """The window whose :func:`commit_bounds` hold the timestep at ``seconds``."""
    windows = clip_windows(num_samples)
    owners = [
        w for w in range(len(windows))
        if commit_bounds(w, len(windows) - 1, block)[0]
        <= (seconds - w) / 0.04
        < commit_bounds(w, len(windows) - 1, block)[1]
    ]
    assert len(owners) <= 1
    return owners[0] if owners else -1


@pytest.mark.parametrize("block", [0, 1])
@pytest.mark.parametrize("seconds_long", [3.0, 5.0, 8.5, 12.3])
def test_stream_region_agrees_with_the_commit_rule(block, seconds_long):
    num_samples = round(seconds_long * RATE)
    last = len(clip_windows(num_samples)) - 1
    for step in range(round(seconds_long / 0.04)):
        t = (step + 0.5) * 0.04
        region = stream_region(t, num_samples, block)
        owner = _committing_window(t, num_samples, block)
        if owner == -1:
            assert region == UNDECODED and num_samples >= WINDOW_SAMPLES
        elif owner == 0 and t < block + 1:
            assert region == STARTUP
        elif owner == last and t >= last + block + 1:
            assert region == FLUSH
        else:
            assert region in {SEAM_START, BLOCK_MID, SEAM_END}
            # Steady state: the owner commits exactly its block ``b``.
            assert (t - owner) // 1 == block


def test_stream_region_seam_bands_sit_either_side_of_a_commit_boundary():
    eight_s = 8 * RATE  # windows 0..3, the last covering [3, 8)
    assert stream_region(0.5, eight_s, 0) == STARTUP
    assert stream_region(1.1, eight_s, 0) == SEAM_START
    assert stream_region(1.5, eight_s, 0) == BLOCK_MID
    assert stream_region(1.9, eight_s, 0) == SEAM_END
    assert stream_region(1.5, eight_s, 1) == STARTUP
    assert stream_region(3.99, eight_s, 0) == SEAM_END
    assert stream_region(4.0, eight_s, 0) == FLUSH
    assert stream_region(4.1, eight_s, 1) == SEAM_START
    assert stream_region(4.5, eight_s, 1) == BLOCK_MID
    assert stream_region(5.0, eight_s, 1) == FLUSH
    assert stream_region(8.2, round(8.5 * RATE), 0) == UNDECODED
    assert stream_region(2.9, 3 * RATE, 0) == FLUSH  # one padded window: nothing undecoded
    assert set(STREAM_CLASSES) >= {STARTUP, FLUSH, UNDECODED}


def test_time_in_segment_prefers_the_haraka_then_its_carrier():
    mids = [0.1, 0.2, 0.3, 0.4]
    assert time_in_segment(5, (4, 8), {5: 2}, mids) == (0.3, TIME_OWN)
    assert time_in_segment(5, (4, 8), {4: 1, 6: 3}, mids) == (0.2, TIME_NEIGHBOUR)
    assert time_in_segment(5, (4, 8), {6: 3}, mids) == (0.4, TIME_NEIGHBOUR)
    assert time_in_segment(5, (4, 8), {2: 0, 9: 1}, mids) is None


# --- Arm bookkeeping ----------------------------------------------------------------------


def _arms(*clips: PoolClip):
    plans = [plan_clip(clip) for clip in clips]
    population = windowed_population(plans)
    return plans, population, build_arms(plans, {site: 1.5 for site in population})


def test_every_arm_maps_its_reference_back_to_the_same_segment_sites():
    plans, population, arms = _arms(_clip())
    plan = plans[0]
    assert plan.eligible and [(w.word_start, w.word_end) for w in plan.windows] == [(0, 4), (3, 7)]
    by_index = {seg.segment_index: seg for seg in plan.segments}
    harakat = {
        Site("a.wav", seg.segment_index, r)
        for seg in plan.segments for r, c in enumerate(seg.reference) if c in (FATHA, DAMMA, KASRA)
    }
    assert population == harakat  # every word fits a window
    for arm, units in arms.units.items():
        seen = set()
        for unit in units:
            for index, site in unit.sites.items():
                assert unit.reference[index] == by_index[site.segment_index].reference[site.ref_index]
                seen.add(site)
        assert seen == harakat, arm


def test_window_classes_and_the_shared_word():
    _, _, arms = _arms(_clip())
    # Word 3 (reference indices 12-16 of segment 0) carries a fatha and a kasra.
    shared = {Site("a.wav", 0, 13), Site("a.wav", 0, 15)}
    assert shared <= arms.population
    classes = {}
    for unit in arms.units[WINDOW]:
        for index, site in unit.sites.items():
            classes.setdefault(site, []).append(unit.classes[index])
    assert all(classes[s] == [LAST_MID_SEGMENT, FIRST_MID_SEGMENT] for s in shared)
    first_word = Site("a.wav", 0, 1)
    assert classes[first_word] == [FIRST_AT_PAUSE]
    last_word = Site("a.wav", 1, 4)  # ص + kasra, the ayah's last word
    assert classes[last_word] == [LAST_AT_PAUSE]


def test_a_site_in_two_windows_counts_once_per_site_and_twice_per_occurrence():
    _, population, arms = _arms(_clip())
    outcomes = {
        unit.key: {i: "matched" for i in unit.sites}
        for units in arms.units.values() for unit in units
    }
    observations = model_observations("base", arms, outcomes)

    def shares(arm):
        return sum(o.share for o in observations if o.key.arm == arm)

    assert shares(SEGMENT) == shares(WINDOW) == shares(STREAM[0]) == len(population)
    assert shares(WINDOW_OCC) == len(population) + 2
    assert all(o.weight == pytest.approx(2 * o.share) for o in observations)  # 1 / 0.5


def test_ineligible_clips_enter_only_segment_all():
    re_read = _clip("b.wav", re_reads=1)
    plans, population, arms = _arms(_clip(), re_read)
    assert not plans[1].eligible
    assert {s.clip for s in population} == {"a.wav"}
    all_classes = {
        unit.classes[i] for unit in arms.units[SEGMENT_ALL] if unit.key.startswith("b.wav")
        for i in unit.sites
    }
    assert all_classes == {CLIP_NOT_WINDOWED}
    assert not any(unit.sites for unit in arms.units[SEGMENT] if unit.key.startswith("b.wav"))
    assert {
        unit.classes[i] for unit in arms.units[SEGMENT_ALL] if unit.key.startswith("a.wav")
        for i in unit.sites
    } == {WINDOWED}


def test_unit_outcomes_finds_an_empty_slot_and_requires_every_site():
    _, _, arms = _arms(_clip())
    unit = arms.units[SEGMENT][1]  # words 5-6: "شَ صِ"
    decode = unit.reference.replace(" ", "")
    assert set(unit_outcomes(decode, unit).values()) == {"matched"}
    dropped = decode.replace(KASRA, "")
    outcomes = unit_outcomes(dropped, unit)
    assert sorted(outcomes.values()) == ["matched", "omitted"]


# --- Statistics ------------------------------------------------------------------------


def _observations(reciters: int = 6) -> list[Observation]:
    """Per reciter: 10 sites; the window arm empties 2 edge sites the segment matched."""
    observations = []
    for reciter in range(reciters):
        for site in range(10):
            edge = site < 4
            seg_outcome = MATCHED
            win_outcome = EMPTY if site < 2 + (reciter % 2) else MATCHED
            for arm, outcome, cls in (
                (SEGMENT, seg_outcome, "whole"),
                (SEGMENT_ALL, seg_outcome, WINDOWED),
                (WINDOW_OCC, win_outcome, FIRST_AT_PAUSE if edge else INTERIOR),
            ):
                key = JointKey("base", arm, "fatha", cls, "first", seg_outcome, outcome)
                observations.append(Observation(reciter, key, 1.0 + reciter, 1.0))
    return observations


def test_contributions_partition_the_paired_difference():
    ledger = Ledger(_observations())
    whole = ledger.value(contribution("base", WINDOW_OCC, "all", MATCHED))
    parts = sum(
        ledger.value(contribution("base", WINDOW_OCC, "all", MATCHED, frozenset({c})))
        for c in (FIRST_AT_PAUSE, INTERIOR)
    )
    assert parts == pytest.approx(whole)
    # Weighted: reciters 0..5 weigh 1..6; odd reciters lose 3 of 10, even ones 2.
    lost = sum((1 + r) * (3 if r % 2 else 2) for r in range(6))
    assert whole == pytest.approx(-lost / sum(10 * (1 + r) for r in range(6)))
    assert ledger.value(contribution("base", WINDOW_OCC, "all", MATCHED), weighted=False) == \
        pytest.approx(-15 / 60)


def test_intervals_are_deterministic_and_bracket_the_estimate():
    ledger = Ledger(_observations())
    statistic = rate("base", WINDOW_OCC, "all", EMPTY)
    first, second = ledger.intervals([statistic]), ledger.intervals([statistic])
    assert first == second
    low, high = first[0]
    assert low <= ledger.value(statistic) <= high


def test_an_undefined_replicate_gives_no_interval():
    observations = _observations(reciters=6)
    lonely = JointKey("base", WINDOW_OCC, "kasra", INTERIOR, "first", MATCHED, MATCHED)
    observations.append(Observation(0, lonely, 1.0, 1.0))
    ledger = Ledger(observations)
    assert ledger.intervals([rate("base", WINDOW_OCC, "kasra", MATCHED)]) == [None]


def test_the_gap_terms_add_up_to_the_adr_difference():
    ledger = Ledger(_observations())
    report = build_report(ledger, ["base"], meta={
        "bootstrap": {"draws": 10_000, "seed": 20261008},
        "pool_clips": 1, "eligible_clips": 1, "reciters": 6, "exclusions": {},
        "windows": 1, "segments_eligible": 1, "segments_all": 1, "stream_seconds": 1,
        "population_sites": 60, "eligible_sites_in_no_window": 0, "site_time_sources": {},
        "base_segments_identical_to_pool_cache": [1, 1], "weights_dtype": "bf16",
        "decode_batch_size": 1, "model_refs": {},
    })
    gap = report["gap"]
    terms = ["adr0005_minus_S_all", "clips_not_windowed", "words_in_no_window",
             "overlap_recount", "window_decode", "W_occ_minus_adr0007"]
    assert sum(gap[t]["weighted"] for t in terms) == pytest.approx(
        ADR0005_SEGMENT_RECALL - ADR0007_WINDOW_RECALL, abs=1e-4)
    assert gap["window_decode:edge"]["weighted"] + gap["window_decode:interior"]["weighted"] \
        == pytest.approx(gap["window_decode"]["weighted"], abs=1e-4)
    assert not math.isnan(gap["window_decode"]["ci95"][0])
    text = render_markdown(report)
    assert "Gap arithmetic" in text and "window_decode:edge" in text


def test_the_ledger_sums_per_reciter():
    ledger = Ledger(_observations(reciters=2))
    assert ledger.reciters == [0, 1]
    assert np.isclose(ledger.unweighted.sum(), 2 * 10 * 3)
    assert ledger.total(lambda k: k.arm == SEGMENT) == pytest.approx(10 * 1 + 10 * 2)


def test_the_window_constants_match_the_protocol():
    assert DEPLOYED_LOGIT_FRAMES == 5 * CONFIRM_TIMESTEPS

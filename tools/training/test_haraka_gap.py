"""Tests for the haraka-gap diagnosis (#85): classification, arm bookkeeping, statistics.

Torch-free and model-free: the clip, its segments and every decode are synthetic. The
fixture is one 9 s recitation of seven words in two waqf segments, which the frozen 5 s /
4 s-hop window build cuts into two windows that share word 3.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from tadabur.mining_pool import PoolClip, PoolSegment
from training.decoding import (
    BLANK_ID,
    CONFIRM_TIMESTEPS,
    DEPLOYED_LOGIT_FRAMES,
    DecodeFingerprint,
    clip_windows,
    stream_emissions,
    stream_protocol,
)
from training.haraka_gap import (
    cache_files,
    validated_settings,
    verified_provenance,
    write_provenance,
)
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
    FROM_NEIGHBOUR,
    FROM_OWN,
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
    site_positions,
    step_in_segment,
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


@pytest.mark.parametrize("block", [0, 1])
@pytest.mark.parametrize("seconds_long", [3.0, 5.0, 8.5, 12.3])
def test_stream_region_names_the_window_the_protocol_commits_from(block, seconds_long):
    """Place one CTC run (odd and even lengths, so integer and half-step midpoints) in
    every window, replay :func:`stream_emissions`, and check the region of every run the
    protocol commits; a reachable position no window commits must be undecoded."""
    num_samples = round(seconds_long * RATE)
    windows = len(clip_windows(num_samples))
    last = windows - 1
    committed = set()
    for window in range(windows):
        for start in range(DEPLOYED_LOGIT_FRAMES):
            for end in (start, start + 1):
                if end >= DEPLOYED_LOGIT_FRAMES:
                    continue
                rows = [np.full(DEPLOYED_LOGIT_FRAMES, BLANK_ID) for _ in range(windows)]
                rows[window][start:end + 1] = 5
                if not stream_emissions(rows, block):
                    continue
                local = (start + end) / 2
                position = window * CONFIRM_TIMESTEPS + local
                committed.add(position)
                region = stream_region(position, num_samples, block)
                if window == 0 and local < (block + 1) * CONFIRM_TIMESTEPS:
                    assert region == STARTUP, (window, local)
                elif window == last and local >= (block + 1) * CONFIRM_TIMESTEPS:
                    assert region == FLUSH, (window, local)
                else:
                    assert region in {SEAM_START, BLOCK_MID, SEAM_END}, (window, local)
    # Every midpoint a run can have in some window, plus a few past the last window.
    reachable = {
        window * CONFIRM_TIMESTEPS + half / 2
        for window in range(windows) for half in range(2 * DEPLOYED_LOGIT_FRAMES - 1)
    }
    beyond = last * CONFIRM_TIMESTEPS + DEPLOYED_LOGIT_FRAMES
    for position in sorted(reachable - committed) + [beyond, beyond + 0.5, beyond + 10]:
        assert stream_region(position, num_samples, block) == UNDECODED, position


def test_stream_region_boundaries_in_ctc_steps():
    eight_s = 8 * RATE  # windows 0..3, the last covering steps [75, 200)
    assert stream_region(24.5, eight_s, 0) == STARTUP  # midpoint < 25: the first window's
    assert stream_region(25.0, eight_s, 0) == SEAM_START
    assert stream_region(29.5, eight_s, 0) == SEAM_START
    assert stream_region(30.0, eight_s, 0) == BLOCK_MID
    assert stream_region(44.5, eight_s, 0) == BLOCK_MID
    assert stream_region(45.0, eight_s, 0) == SEAM_END
    assert stream_region(49.5, eight_s, 1) == STARTUP
    assert stream_region(50.0, eight_s, 1) == SEAM_START
    assert stream_region(99.5, eight_s, 0) == SEAM_END
    assert stream_region(100.0, eight_s, 0) == FLUSH
    assert stream_region(124.5, eight_s, 1) == SEAM_END
    assert stream_region(125.0, eight_s, 1) == FLUSH
    eight_and_a_half = round(8.5 * RATE)
    assert stream_region(199.5, eight_and_a_half, 0) == FLUSH
    assert stream_region(200.0, eight_and_a_half, 0) == UNDECODED
    three_s = 3 * RATE  # one padded window: startup, then flush, nothing undecoded
    assert stream_region(24.5, three_s, 0) == STARTUP
    assert stream_region(25.0, three_s, 0) == FLUSH
    assert set(STREAM_CLASSES) >= {STARTUP, FLUSH, UNDECODED}


def test_step_in_segment_prefers_the_haraka_then_its_carrier():
    mids = [0.5, 2.0, 3.5, 5.0]
    assert step_in_segment(5, (4, 8), {5: 2}, mids) == (3.5, FROM_OWN)
    assert step_in_segment(5, (4, 8), {4: 1, 6: 3}, mids) == (2.0, FROM_NEIGHBOUR)
    assert step_in_segment(5, (4, 8), {6: 3}, mids) == (5.0, FROM_NEIGHBOUR)
    assert step_in_segment(5, (4, 8), {2: 0, 9: 1}, mids) is None


def test_site_positions_are_ctc_midpoints_on_the_stream_lattice():
    plans, population, _ = _arms(_clip())
    second = plans[0].segments[1]  # words 5-6, "شَ صِ", starting at 6.5 s = step 162.5
    decode = second.reference.replace(" ", "")
    base = {
        f"a.wav#{seg.segment_index}": {"decode": seg.reference.replace(" ", ""),
                                       "mid2": list(range(len(seg.reference)))}
        for seg in plans[0].segments
    }
    base["a.wav#1"] = {"decode": decode, "mid2": [0, 3, 4, 7]}  # midpoints 0, 1.5, 2, 3.5
    positions, sources = site_positions(plans, population, base)
    assert positions[Site("a.wav", 1, 1)] == pytest.approx(162.5 + 1.5)
    assert positions[Site("a.wav", 1, 4)] == pytest.approx(162.5 + 3.5)
    assert sources[FROM_OWN] == len(population)


# --- Arm bookkeeping ----------------------------------------------------------------------


def _arms(*clips: PoolClip):
    plans = [plan_clip(clip) for clip in clips]
    population = windowed_population(plans)
    return plans, population, build_arms(plans, {site: 30.0 for site in population})


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
        "population_sites": 60, "eligible_sites_in_no_window": 0, "site_position_sources": {},
        "base_segments_identical_to_pool_cache": [1, 1], "decode_settings": {}, "models": {},
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


# --- Fingerprints and provenance ----------------------------------------------------------


def _fingerprints(model_ref: str, **overrides) -> dict:
    modes = {"spans": "whole-spans", **{f"stream_b{b}": stream_protocol(b) for b in (0, 1)}}
    return {
        key: DecodeFingerprint(model_ref, mode, "bf16", 1, "cuda", True).as_dict()
        | overrides.get(key, {})
        for key, mode in modes.items()
    }


def _cache(model_ref: str, **overrides) -> dict:
    return {"model_ref": model_ref, "fingerprints": _fingerprints(model_ref, **overrides)}


def test_validated_settings_come_from_the_fingerprints():
    settings = validated_settings({"base": _cache("t"), "h448": _cache("c.pt")})
    assert settings["weights_dtype"] == "bf16" and settings["batch_size"] == 1


@pytest.mark.parametrize("caches", [
    {"base": {"model_ref": "t", "fingerprints": {}}},
    {"base": {"model_ref": "t"}},
    {"base": {"model_ref": "t", "fingerprints": {
        k: v for k, v in _fingerprints("t").items() if k != "stream_b1"}}},
    {"base": _cache("t", stream_b1={"mode": stream_protocol(0)})},
    {"base": _cache("t", spans={"weights_dtype": "fp32"})},
    {"base": _cache("t", stream_b0={"batch_size": 16})},
    {"base": _cache("t", stream_b0={"model": "other"})},
    {"base": _cache("t"), "h448": _cache("c.pt", spans={"weights_dtype": "fp32"},
                                          stream_b0={"weights_dtype": "fp32"},
                                          stream_b1={"weights_dtype": "fp32"})},
    {},
])
def test_validated_settings_refuse_missing_modes_and_mixed_numerics(caches):
    with pytest.raises(ValueError):
        validated_settings(caches)


def test_provenance_refuses_an_edited_cache(tmp_path):
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"weights")
    for name in cache_files(["h448"]):
        (tmp_path / name).write_bytes(name.encode())
    write_provenance(tmp_path, {"h448": str(checkpoint)})
    record = verified_provenance(tmp_path, ["h448"])
    assert record["models"]["h448"]["checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest()
    (tmp_path / "h448.json").write_bytes(b"edited")
    with pytest.raises(ValueError):
        verified_provenance(tmp_path, ["h448"])

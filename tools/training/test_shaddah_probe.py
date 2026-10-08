"""The shaddah probe's measures on synthetic posteriors (torch-free)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tadabur.phoneme_vocab import NUM_PHONEME_CLASSES, PHONEME_CHAR_TO_ID
from training import shaddah_probe as sp

R, B, D, FATHA, KASRA = "ر", "ب", "د", "َ", "ِ"
FLOOR = 1e-6


def posteriors(*frames: dict | str) -> np.ndarray:
    """Log-posteriors, one row per frame: ``"_"`` is blank, a char is that token at 0.99,
    a dict gives token probabilities with the rest on the blank."""
    rows = []
    for frame in frames:
        probs = np.full(NUM_PHONEME_CLASSES, FLOOR)
        spec = {} if frame == "_" else {frame: 0.99} if isinstance(frame, str) else frame
        for char, p in spec.items():
            probs[PHONEME_CHAR_TO_ID[char]] = p
        probs[0] = max(0.0, 1.0 - probs.sum() + FLOOR)
        rows.append(np.log(probs / probs.sum()))
    return np.array(rows, dtype=np.float32)


# ر َ ب(b) ِ with a geminate ب in the reference; the decode emits ب once at frame 4.
GEMINATE_REFERENCE = R + FATHA + B + B + KASRA


def _collapsed(second: dict | str = "_") -> np.ndarray:
    return posteriors(R, "_", FATHA, "_", B, "_", second, "_", KASRA)


def _measure(reference: str, lp: np.ndarray) -> dict[int, sp.SiteMeasure]:
    return {site.reference_index: site for site in sp.measure_segment(reference, lp).sites}


def test_reference_sites_are_consonant_runs():
    sites = sp.reference_sites("وَقُوو" + " " + "مَممممَ" + "اا" + "ںں")
    assert [(s.index, s.length, s.consonant) for s in sites] == [
        (0, 1, "و"), (2, 1, "ق"), (4, 2, "و"), (7, 1, "م"), (9, 4, "م"),
    ]
    assert [s.is_geminate for s in sites] == [False, False, True, False, True]


def test_a_collapsed_geminate_is_decoded_once():
    sites = _measure(GEMINATE_REFERENCE, _collapsed())
    assert sites[2].decoded == 1
    assert sites[0].decoded == 1 and not sites[0].is_geminate


def test_a_double_decoded_geminate_counts_both_tokens():
    lp = posteriors(R, "_", FATHA, "_", B, "_", B, "_", KASRA)
    site = _measure(GEMINATE_REFERENCE, lp)[2]
    assert site.decoded == 2
    assert site.log_ratio > 0  # the decode itself is the doubled transcript
    assert site.second_peak is None and site.extra_location is None


def test_an_inserted_double_at_a_single_is_counted_against_its_site():
    lp = posteriors(R, "_", FATHA, "_", B, "_", B, "_", KASRA)
    site = _measure(R + FATHA + B + KASRA, lp)[2]
    assert site.decoded == 2 and not site.is_geminate


def test_a_sub_argmax_second_spike_is_present_mass_after_the_run():
    site = _measure(GEMINATE_REFERENCE, _collapsed({B: 0.3}))[2]
    assert site.second_peak == pytest.approx(0.3, rel=1e-3)
    assert site.extra_location == sp.AFTER
    assert site.log_ratio >= sp.PRESENT_LOG_RATIO
    # the doubled transcript's best path has one extra spike of 0.3 against a blank of 0.7
    assert site.log_ratio == pytest.approx(math.log(0.3 / 0.7), abs=0.3)


def test_no_second_spike_is_absent_mass():
    site = _measure(GEMINATE_REFERENCE, _collapsed())[2]
    assert site.log_ratio < sp.ABSENT_LOG_RATIO
    assert site.second_peak == pytest.approx(FLOOR, rel=0.1)


def test_a_dip_inside_a_long_run_is_a_split():
    lp = posteriors(R, "_", FATHA, B, B, {B: 0.6}, B, B, "_", KASRA)
    lp[5, 0] = np.log(0.4)  # the blank rises to 0.4 mid-run but stays below the consonant
    site = _measure(GEMINATE_REFERENCE, lp)[2]
    assert site.decoded == 1 and site.run_frames == 5
    assert site.extra_location == sp.SPLIT
    assert site.second_peak == pytest.approx(FLOOR, rel=0.1)  # only the blank frame 8 is off-run


def test_the_interval_spans_the_frames_between_the_flanking_tokens():
    site = _measure(GEMINATE_REFERENCE, _collapsed())[2]
    assert site.interval_frames == 5  # frames 3..7, between the fatha (2) and the kasra (8)
    assert site.vv_context is True
    assert sp.site_interval(GEMINATE_REFERENCE, _collapsed(), 2) == (3, 8)


def test_a_site_at_the_edge_of_the_decode_has_no_interval():
    site = _measure(GEMINATE_REFERENCE, _collapsed())[0]
    assert site.interval_frames is None and site.vv_context is None


def test_word_spaces_do_not_shift_the_alignment():
    reference = R + FATHA + " " + B + B + KASRA
    assert _measure(reference, _collapsed())[3].decoded == 1


def _observation(segment="c.wav#0", reciter=1, weight=1.0, **site) -> sp.Observation:
    defaults = dict(
        reference_index=0, length=1, consonant=B, decoded=1, vv_context=True,
        interval_frames=4, run_frames=1, log_ratio=-10.0, second_peak=0.0,
        extra_location=sp.AFTER,
    )
    return sp.Observation(segment, reciter, weight, False, None, sp.SiteMeasure(**{**defaults, **site}))


def test_observations_normalize_by_the_segment_single_rate_and_flag_the_census():
    measure = sp.SegmentMeasure(
        "", tuple(
            [_observation(reference_index=i, interval_frames=f).site for i, f in [(0, 3), (5, 4), (9, 5)]]
            + [_observation(reference_index=12, length=2, interval_frames=8).site]
        )
    )
    obs = sp.segment_observations("c.wav#0", 7, 0.25, measure, frozenset({12}))
    assert [o.rate_normalized for o in obs] == [0.75, 1.0, 1.25, 2.0]
    assert [o.census_collapsed for o in obs] == [False, False, False, True]
    assert all(o.weight == 4.0 and o.reciter_id == 7 for o in obs)
    assert [o.population for o in obs] == [sp.SINGLE] * 3 + [sp.COLLAPSED]

    too_few = sp.SegmentMeasure("", measure.sites[2:])
    assert all(o.rate_normalized is None for o in sp.segment_observations("c#0", 7, 1.0, too_few, frozenset()))


def test_populations():
    assert _observation(length=2, decoded=1).population == sp.COLLAPSED
    assert _observation(length=2, decoded=2).population == sp.DOUBLE
    assert _observation(length=3, decoded=2).population is None
    assert _observation(length=1, decoded=2).population is None
    assert _observation(length=2, decoded=0).population is None


def test_weighted_share_and_reciter_clustered_interval():
    obs = [_observation(reciter=r, weight=w, log_ratio=v)
           for r, w, v in [(1, 1.0, 0.0), (1, 1.0, -20.0), (2, 3.0, 0.0), (3, 1.0, -20.0)]]
    present = lambda o: o.site.log_ratio >= sp.PRESENT_LOG_RATIO  # noqa: E731
    assert sp.share(obs, present) == {"n": 4, "unweighted": 0.5, "weighted": pytest.approx(4 / 6)}
    low, high = sp.clustered_interval(obs, present, resamples=2000)
    assert 0.0 <= low < 4 / 6 < high <= 1.0
    assert sp.clustered_interval(obs, present, resamples=2000) == [low, high]  # seeded
    assert sp.share([], present)["weighted"] is None


def test_weighted_quantiles():
    assert sp.weighted_quantiles([3, 1, 2], [1, 1, 1], [0.5]) == [2.0]
    assert sp.weighted_quantiles([1, 2], [1, 3], [0.5]) == [2.0]
    assert sp.weighted_quantiles([], [], [0.5]) is None


def test_rule_states_follow_the_preregistered_thresholds():
    assert sp.rule_state(_observation(decoded=2, length=2, log_ratio=-50.0)) == "held"
    assert sp.rule_state(_observation(log_ratio=sp.PRESENT_LOG_RATIO)) == "held"
    assert sp.rule_state(_observation(log_ratio=-4.0)) == "unsure"
    assert sp.rule_state(_observation(log_ratio=sp.ABSENT_LOG_RATIO - 0.01)) == "not_held"


def test_controls_are_the_nearest_unused_single_of_the_same_consonant():
    gem = lambda seg, i: _observation(segment=seg, reference_index=i, length=2)  # noqa: E731
    single = lambda seg, i, c=B: _observation(segment=seg, reference_index=i, consonant=c)  # noqa: E731
    obs = [
        gem("a.wav#0", 10), gem("a.wav#0", 30), gem("b.wav#0", 5),
        single("a.wav#0", 2), single("a.wav#0", 12), single("a.wav#0", 40, c=D),
        single("a.wav#1", 3), single("c.wav#0", 1),
    ]
    pairs = sp.match_controls(obs)
    found = [(g.segment, g.site.reference_index,
              None if c is None else (c.segment, c.site.reference_index)) for g, c in pairs]
    assert found == [
        ("a.wav#0", 10, ("a.wav#0", 12)),
        ("a.wav#0", 30, ("a.wav#0", 2)),
        ("b.wav#0", 5, None),
    ]
    assert sp.match_controls([gem("a.wav#0", 1), single("a.wav#1", 4)])[0][1].segment == "a.wav#1"


def _trial(population, factor, edited, decoy, reciter=1):
    site = lambda d: _observation(decoded=d).site  # noqa: E731
    return sp.StretchTrial("a#0", reciter, 1.0, population, 0, factor, site(edited), site(decoy), 0, 0)


def test_stretch_report_nets_the_decoy_out():
    trials = [
        _trial(sp.COLLAPSED, 1.5, 2, 1, reciter=1), _trial(sp.COLLAPSED, 1.5, 2, 2, reciter=2),
        _trial(sp.COLLAPSED, 1.5, 1, 1, reciter=3), _trial(sp.COLLAPSED, 1.5, 2, 1, reciter=4),
        _trial(sp.SINGLE, 1.5, 1, 1), _trial(sp.SINGLE, 1.5, 2, 1),
    ]
    report = sp.stretch_report(trials)["1.5"]
    assert report[sp.COLLAPSED]["double_after_stretch"]["weighted"] == 0.75
    assert report[sp.COLLAPSED]["double_on_decoy"]["weighted"] == 0.25
    assert report[sp.COLLAPSED]["net"]["weighted"] == 0.5
    assert report[sp.SINGLE]["net"]["weighted"] == 0.5
    assert sp.stretch_report(trials)["1.25"][sp.COLLAPSED]["trials"] == 0


def _verdict_inputs(geminate_like, collapsed_present, single_present, net_collapsed, net_single,
                    single_unsure=0.0):
    def states(present, unsure):
        return {
            "log_ratio_states": {"present": {"weighted": present}},
            "rule_states": {
                "held": {"weighted": present},
                "unsure": {"weighted": unsure},
                "not_held": {"weighted": 1 - present - unsure},
            },
        }
    mass = {sp.COLLAPSED: states(collapsed_present, 0.2), sp.SINGLE: states(single_present, single_unsure)}
    durations = {"geminate_like": {"weighted": geminate_like}}
    stretch = {"1.5": {sp.COLLAPSED: {"net": {"weighted": net_collapsed}},
                       sp.SINGLE: {"net": {"weighted": net_single}}}}
    return mass, durations, stretch


@pytest.mark.parametrize("inputs, call, viable", [
    ((0.7, 0.30, 0.01, 0.4, 0.05), "representation", True),
    ((0.7, 0.02, 0.01, 0.0, 0.0), "representation", False),
    ((0.2, 0.05, 0.01, 0.05, 0.02), "data", False),
    ((0.2, 0.30, 0.01, 0.0, 0.0), "mixed", True),
    ((0.2, 0.05, 0.01, 0.30, 0.0), "mixed", False),
    ((0.7, 0.30, 0.01, 0.4, 0.05, 0.2), "representation", False),  # too many unsure singles
])
def test_verdict(inputs, call, viable):
    result = sp.verdict(*_verdict_inputs(*inputs))
    assert result["call"] == call
    assert result["posterior_rule_looks_viable"] is viable


def test_model_report_runs_end_to_end_on_measured_segments():
    measures = {
        "a.wav#0": sp.measure_segment(GEMINATE_REFERENCE + R + FATHA + B + KASRA + D + FATHA,
                                      posteriors(R, "_", FATHA, "_", B, "_", {B: 0.3}, "_", KASRA,
                                                 R, "_", FATHA, B, "_", KASRA, D, FATHA)),
        "b.wav#0": sp.measure_segment(GEMINATE_REFERENCE, posteriors(
            R, "_", FATHA, "_", B, "_", B, "_", KASRA)),
    }
    observations = [
        o for key, m in measures.items()
        for o in sp.segment_observations(key, hash(key) % 3, 0.5, m, frozenset({2}))
    ]
    report = sp.model_report(observations, [])
    assert report["verdict"]["call"] == "insufficient_evidence"  # no stretch trials
    assert report["sites"]["by_population"][sp.COLLAPSED] == 1
    assert report["sites"]["by_population"][sp.DOUBLE] == 1
    (row,) = report["collapsed_sites"]
    assert row["census_collapsed"] and row["rule_state"] == "held"
    assert report["mass"][sp.COLLAPSED]["log_ratio_states"]["present"]["weighted"] == 1.0

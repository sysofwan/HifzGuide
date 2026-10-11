"""Ship A's sizing (#126): its simulators on known cases, and the committed results and page."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import pytest

from tadabur.power_frame import input_checksums
from tadabur.ship_a import SITES_PATH, listening_weights, load_sites, mine_all
from tadabur.test_power_frame import H448, REFERENCE, scored_pool
from tadabur.truth_sites import UNCLEAR
from tadabur.eval_harness import build as fixture_side_report
from tadabur.listening_session import Verdict, write_verdicts
from tadabur.exclusions import GUARD_PATHS, REASONS, Exclusions, current
from tadabur.word_sample import WORD_SAMPLE_PATH, load_words
from training import ship_a_sizing as z
from training.acceptance_stats import (
    BOOTSTRAP,
    CANNOT_CERTIFY,
    FAIL,
    LOWER,
    NO_BOUND,
    PASS,
    TANGO,
    WILSON,
    Interval,
    bound_verdict,
    ratio,
    unit_if_equal,
)
from training.ship_a_sizing_doc import DOC_PATH, load_inputs, render
from training.site_outcomes import CORRECT_SIDE
from training.truth_baseline import DEFAULT_MODELS
from training.truth_scorer import HEADLINE, item_key, reconcile, score


def _side(reciters, theta=0.0, rate=1.0, side=z.RECOVERY, rule=z.ONE_PER_RECITER,
          weights=None):
    weights = tuple(weights or [1.0] * len(reciters))
    return z.SideScenario("cell", tuple(reciters), weights, theta, rate, 0.15, side, rule)


def test_a_flawless_recovery_reaches_the_wilson_floor_of_its_reciters():
    """40 reciters, every site correct and recovered: degenerate, so Wilson, 40/43.84 = 91%."""
    result = z.side_bounds(_side(range(40)), 20)
    assert result["median_bound"] == pytest.approx(40 / (40 + 1.96**2), abs=1e-3)
    assert result["power"]["0.8"] == 1.0 and result["mean_reciters"] == 40


def test_a_sparse_cell_with_unequal_weights_cannot_certify():
    """§1: Wilson stands in only for independent, equally weighted observations."""
    weights = [1.0] * 10 + [4.0] * 10
    unequal = z.side_bounds(_side(range(20), weights=weights), 20)
    assert unequal["median_bound"] is None and unequal["power"]["0.2"] == 0.0
    equal = z.side_bounds(_side(range(15), weights=[3.0] * 15), 20)
    assert equal["median_bound"] == pytest.approx(15 / (15 + 1.96**2), abs=1e-3)


def test_every_site_without_one_per_reciter_cannot_certify_a_sparse_cell():
    """Six sites from two reciters: sparse, and not independent, so no bound at all."""
    result = z.side_bounds(_side([1, 1, 1, 2, 2, 2], rule=z.EVERY_SITE), 20)
    assert result["power"]["0.2"] == 0.0 and result["median_bound"] is None


def test_retention_reads_only_the_mistakes():
    assert z.side_bounds(_side(range(30), theta=0.0, side=z.RETENTION), 10)["mean_sites"] == 0
    assert z.side_bounds(_side(range(30), theta=1.0, side=z.RETENTION), 10)["mean_sites"] == 30


def test_no_change_bounds_no_new_error():
    scenario = z.HarmScenario((50, 80, 20), 12.0, 0.0, 1.0, 0.0, 2)
    result = z.harm_power(scenario, 10)
    assert result["median_bound_per_1000"] == 0.0 and result["mean_minutes"] == 0.0


def test_the_budget_counts_each_half_and_marks_sparse_cells():
    sites = mine_all(scored_pool(base=REFERENCE.replace("بب", "ب")))
    budget = z.budget(sites)
    family = budget["families"]["ذ↔ز"]
    assert family["sites"] == family["partner_swaps"] == 1 and family["held_out_sparse"]
    assert family["train"]["sites"] + family["held_out"]["sites"] == 1
    assert set(budget["cells"]) == {z.SOFT_PAIRS_POOLED, z.GEMINATES, "ت↔ط", "ح↔ه", "ذ↔ز",
                                    "س↔ص", "ض↔ظ", "ق↔ك", "ذ↔ظ", "س↔ش", "ط↔ق"}


def test_the_mistake_mix_pools_soft_pairs_and_bounds_the_reject_share():
    sides = {"sites": [
        {"family": "ق↔ك", "prescribed": "ق", "fixture_side": "should_accept",
         "arms": {"h448/stream_b0": {"committed": "ك"}}},
        {"family": "س↔ص", "prescribed": "س", "fixture_side": "should_reject",
         "arms": {"h448/stream_b0": {"committed": "ش"}}},
        {"family": "س↔ص", "prescribed": "س", "fixture_side": "should_reject",
         "arms": {"h448/stream_b0": {"committed": "س"}}},
    ]}
    mix = z.mistake_mix(sides, [])
    assert mix["p35_h448_off"] == {z.SOFT_PAIRS_POOLED: {
        "should_accept": 1, "should_reject": 1, "reject_share_upper": 0.5}}


def test_the_committed_sizing_reads_the_committed_inputs():
    sizing = json.loads(z.SIZING_PATH.read_text(encoding="utf-8"))
    assert sizing["inputs"] == {**input_checksums(), **{
        name: z._sha256(path) for name, path in (
            ("ship_a/sites.jsonl", SITES_PATH), ("ship_a/word_sample.jsonl", WORD_SAMPLE_PATH))}}
    excluded = Exclusions.of(**{reason: sizing[block][key]
                                for reason, (block, key) in z.GUARD_BLOCKS.items()})
    sites, words = z.guarded(load_sites(), load_words(), excluded)
    for reason, (block, _) in z.GUARD_BLOCKS.items():
        assert sizing[block]["guard"] == z._sha256(GUARD_PATHS[reason])
    assert sizing["budget"] == json.loads(json.dumps(z.budget(sites)))
    assert sizing["word_sample_effective"]["words"] == len(words)
    fixtures = fixture_side_report(DEFAULT_MODELS, None, excluded)
    for reason in REASONS:
        assert sizing["fixture_report"][reason] == fixtures[reason]
    assert sizing["mistake_mix"] == json.loads(json.dumps(z.mistake_mix(fixtures, sites)))
    assert sizing["repetitions"] == z.REPETITIONS


@pytest.mark.parametrize("note, change", [("waqf", -1), ("no waqf", +1)])
def test_an_owner_note_moves_question_4_both_ways(tmp_path, note, change):
    """Question 4 reads the fixture-side report rebuilt under the current guard and verdicts: a
    ``waqf`` note takes a wasl-dependent P3.5 shaddah the detector kept out of the mix, and a
    ``no waqf`` note brings back the one the detector realized."""
    kept = "p35_fixture:tadabur_spk0066_S26_A89_008098b6_000033__seg0.wav:shadda#69"
    realized = "p35_fixture:tadabur_spk0097_S20_A18_14e487d0_000013__seg0.wav:shadda#74"
    verdicts = tmp_path / "verdicts.jsonl"

    def shaddah_mix(notes: dict[str, str]) -> dict:
        write_verdicts({i: Verdict(i, "held", n) for i, n in notes.items()}, verdicts)
        guard = current(verdicts_path=verdicts)
        return z.mistake_mix(fixture_side_report(DEFAULT_MODELS, None, guard), [])[
            "p35_h448_off"]["shaddah"]

    before = shaddah_mix({})
    after = shaddah_mix({kept if note == "waqf" else realized: note})
    assert after["should_accept"] == before["should_accept"]
    assert after["should_reject"] == before["should_reject"] + change


def test_the_sizing_leaves_out_what_each_guard_takes_out():
    every_site, words = load_sites(), load_words()
    realized = {every_site[0].site.site_id, words[0].site.site_id}
    excluded = Exclusions.of(waqf_realized=realized,
                             onset_trimmed={every_site[0].site.site_id, every_site[1].site.site_id})
    sites, kept_words = z.guarded(every_site, words, excluded)
    assert len(sites) == len(every_site) - 2 and len(kept_words) == len(words) - 1
    block = z.guard_block("waqf_realized", every_site, words, excluded)
    assert block["sites"] == block["words"] == 1 and block["realized"] == sorted(realized)
    assert sum(block["families"].values()) == 1
    trimmed = z.guard_block("onset_trimmed", every_site, words, excluded)
    assert trimmed["sites"] == 1 and trimmed["words"] == 0  # the shared site counts as waqf
    assert len(trimmed["trimmed"]) == 2


@pytest.mark.parametrize("note, change", [("Trimming issues", -1), ("no trim", 0)])
def test_a_trim_note_takes_a_p35_site_out_of_question_4(tmp_path, note, change):
    """#138: question 4's P3.5 mix reads the onset guard at the current verdicts: a trim note
    takes a kept P3.5 site out of the mix; a negated one leaves it in."""
    kept = "p35_fixture:tadabur_spk0066_S26_A89_008098b6_000033__seg0.wav:shadda#69"
    verdicts = tmp_path / "verdicts.jsonl"

    def shaddah_mix(notes: dict[str, str]) -> dict:
        write_verdicts({i: Verdict(i, "held", n) for i, n in notes.items()}, verdicts)
        report = fixture_side_report(DEFAULT_MODELS, None, current(verdicts_path=verdicts))
        return report, z.mistake_mix(report, [])["p35_h448_off"]["shaddah"]

    _, before = shaddah_mix({})
    report, after = shaddah_mix({kept: note})
    assert (kept in report["onset_trimmed"]) == (change != 0)
    assert after["should_reject"] + after["should_accept"] == (
        before["should_reject"] + before["should_accept"] + change)


def test_the_committed_page_is_rendered_from_the_committed_results():
    assert DOC_PATH.read_text(encoding="utf-8") == render(*load_inputs())


def test_effective_sizes_read_the_listening_weights_only():
    sizes = z.effective_sizes(load_sites())
    for cell, entry in sizes.items():
        assert 0 <= entry["effective"] <= entry["sites"] + 1e-9, cell
    assert sizes[z.GEMINATES]["effective"] == sizes[z.GEMINATES]["sites"]  # a census


def test_the_word_sample_weighs_its_own_selection_not_the_pool():
    words = load_words()
    assert z.word_weights(words) == tuple(1 / w.draw_probability for w in words)
    flat = z.word_sample_power(z.WordScenario((1.0,) * 150, 0.0, 0.5), 5)
    assert all(p == 0.0 for p in flat["power"].values())  # no flag, nothing to reduce


def _verified_cell(chances, unclear=0):
    """A held-out cell of one swap site per reciter (reciter ``k``), each listened with chance
    ``chances[k]`` and recited as the mushaf prescribes, but the last ``unclear`` heard
    ``unclear`` (excluded, though they could belong to the cell)."""
    (swap,) = [s for s in mine_all(scored_pool(base=H448)) if s.kind == "pair"]
    heard = [swap.site.prescribed] * (len(chances) - unclear) + [UNCLEAR] * unclear
    return [replace(swap, listen_probability=p, row=replace(swap.row, site=replace(
        swap.site, site_id=f"{swap.site.site_id}#{k}", audio_filename=f"c{k}.wav",
        audio_sha256=f"{k:064x}", heard=heard[k],
        stratum_population=len(chances)))) for k, p in enumerate(chances)]


def _scored_cell(sites, recovered):
    """The #84 scorer's cell for ``sites`` under Ship A's listening weights: arm ``m`` reads
    each site right where ``recovered``, arm ``base`` never does."""
    truth = [s.site for s in sites]
    weights = listening_weights(sites)
    assert reconcile(truth, weights).weights == weights
    wrong = {item_key(t): t.reference.replace("ذ", "ز") for t in truth}
    decodes = {"m": {item_key(t): t.reference if ok else wrong[item_key(t)]
                     for t, ok in zip(truth, recovered)}, "base": wrong}
    report = score(truth, {t.audio_filename: k for k, t in enumerate(truth)}, decodes,
                   comparisons=[("m", "base")], design_weights=weights)
    return next(c for c in report["cells"] if (c["population"], c["side"], c["label"])
                == (HEADLINE, CORRECT_SIDE, "ذ↔ز:ذ"))


def _as_interval(entry) -> Interval:
    return Interval(*(entry[key] for key in ("lower", "upper", "method", "reason")))


def test_the_scorer_and_the_sizing_weigh_verified_sites_alike():
    """Estimator parity: the #84 scorer, given Ship A's listening weights, reads the same
    weighted rate the sizing simulates (Σw·A / Σw·C with w = 1 / listen_probability)."""
    sites, recovered = _verified_cell((1.0, 0.5, 0.25)), (True, False, True)
    rate = _scored_cell(sites, recovered)["arms"]["m"]["rates"]["committed_accuracy"]
    w = np.array(list(listening_weights(sites).values()))
    assert ratio(w * np.array(recovered, dtype=float), w) == pytest.approx(5 / 7)
    assert rate["point"] == pytest.approx(5 / 7)


FLOOR = 0.7
EQUAL_NOT_UNIT = (0.25,) * 13  # Astra's repro: 13 frozen س↔ش sites, one per reciter, weight 4
UNEQUAL = (1.0, 0.25) * 6 + (1.0,)


@pytest.mark.parametrize(("chances", "recovered", "method", "verdict"), [
    (EQUAL_NOT_UNIT, (True,) * 13, WILSON, PASS),
    (UNEQUAL, (True,) * 13, NO_BOUND, CANNOT_CERTIFY),
    ((1.0, 0.5) * 15, tuple(k % 5 > 0 for k in range(30)), BOOTSTRAP, FAIL),
])
def test_the_scorer_and_the_sizing_certify_a_cell_alike(chances, recovered, method, verdict):
    """One §1 rule on every Ship A path: the same cell gets the same method, bound and verdict
    from the sizing and from the scorer, a cell of one common weight (4, not 1) included."""
    sites = _verified_cell(chances)
    scored = _as_interval(_scored_cell(sites, recovered)["arms"]["m"]["rates"]
                          ["committed_accuracy"])
    weights = np.array(list(listening_weights(sites).values()))
    sized = z.cell_interval(np.arange(len(sites)), weights, np.array(recovered, dtype=float))
    assert sized.method == scored.method == method
    assert sized.lower == pytest.approx(scored.lower)
    assert (bound_verdict(sized, FLOOR, LOWER).verdict
            == bound_verdict(scored, FLOOR, LOWER).verdict == verdict)
    if all(recovered):  # flawless: the simulation draws this very cell every repetition
        simulated = z.side_bounds(_side(range(len(sites)), weights=weights.tolist()), 5)
        assert simulated["median_bound"] == pytest.approx(scored.lower)
        assert simulated["power"][str(FLOOR)] == float(verdict == PASS)


def test_the_scorer_and_the_word_sizing_read_a_common_weight_as_equal():
    """A sparse paired difference takes Tango under one common weight, whatever its value,
    and no bound under unequal weights: in the scorer and in the word-sample sizing alike."""
    differences = {name: _scored_cell(_verified_cell(chances), (True,) * 13)["differences"]
                   ["m - base"]["committed_accuracy"]["method"]
                   for name, chances in (("equal", EQUAL_NOT_UNIT), ("unequal", UNEQUAL))}
    assert differences == {"equal": TANGO, "unequal": NO_BOUND}
    power = {weight: z.word_sample_power(z.WordScenario((weight,) * 13, 1.0, 1.0), 20)["power"]
             for weight in (1.0, 4.0)}
    assert power[4.0] == power[1.0] and power[1.0]["0.0"] == 1.0
    unequal = z.word_sample_power(z.WordScenario(tuple(1 / c for c in UNEQUAL), 1.0, 1.0), 20)
    assert all(p == 0.0 for p in unequal["power"].values())


def test_a_common_weight_leaves_the_scorer_scale_free_with_an_excluded_site():
    """The unit scale reaches the intervals only: 13 verified sites and one ``unclear``, all
    at weight 1 or all at weight 4, give one point, interval, difference and exclusion
    sensitivity (Astra round 5: weight 4 once read the worst case as 13/17, not 13/14)."""
    at = {chance: _scored_cell(_verified_cell((chance,) * 14, unclear=1), (True,) * 14)
          for chance in (1.0, 0.25)}
    for cell in at.values():
        assert cell["excluded"] == 1
        assert cell["arms"]["m"]["sensitivity"]["committed_accuracy"] == pytest.approx(
            {"worst": 13 / 14, "best": 1.0})
    one, four = at[1.0], at[0.25]
    assert one["arms"]["m"]["sensitivity"] == four["arms"]["m"]["sensitivity"]
    assert one["differences"] == four["differences"]
    for name, rate in one["arms"]["m"]["rates"].items():
        scaled = four["arms"]["m"]["rates"][name]
        assert {k: scaled[k] for k in ("point", "lower", "upper", "method")} == {
            k: rate[k] for k in ("point", "lower", "upper", "method")}, name
        assert (scaled["num"], scaled["den"]) == (4 * rate["num"], 4 * rate["den"]), name


def test_one_common_weight_goes_on_the_unit_scale_and_unequal_weights_stay():
    assert unit_if_equal([4.0] * 3).tolist() == [1.0] * 3
    assert unit_if_equal([1.0, 4.0]).tolist() == [1.0, 4.0]
    assert unit_if_equal([]).size == 0

"""Ship A's sizing (#126): its simulators on known cases, and the committed results and page."""

from __future__ import annotations

import json

import numpy as np

import pytest

from tadabur.power_frame import input_checksums
from tadabur.ship_a import SITES_PATH, load_sites, mine_all
from tadabur.test_power_frame import REFERENCE, scored_pool
from tadabur.word_sample import WORD_SAMPLE_PATH
from training import ship_a_sizing as z
from training.ship_a_sizing_doc import DOC_PATH, load_inputs, render


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
            ("ship_a/sites.jsonl", SITES_PATH), ("ship_a/word_sample.jsonl", WORD_SAMPLE_PATH),
            ("truth_baseline/fixture_sides.json", z.FIXTURE_SIDES_PATH))}}
    assert sizing["budget"] == json.loads(json.dumps(z.budget(load_sites())))
    assert sizing["repetitions"] == z.REPETITIONS


def test_the_committed_page_is_rendered_from_the_committed_results():
    assert DOC_PATH.read_text(encoding="utf-8") == render(*load_inputs())


def test_effective_sizes_read_the_listening_weights_only():
    sizes = z.effective_sizes(load_sites())
    for cell, entry in sizes.items():
        assert 0 <= entry["effective"] <= entry["sites"] + 1e-9, cell
    assert sizes[z.GEMINATES]["effective"] == sizes[z.GEMINATES]["sites"]  # a census


def test_the_word_sample_weighs_its_own_selection_not_the_pool():
    from tadabur.word_sample import load_words

    words = load_words()
    assert z.word_weights(words) == tuple(1 / w.draw_probability for w in words)
    flat = z.word_sample_power(z.WordScenario((1.0,) * 150, 0.0, 0.5), 5)
    assert all(p == 0.0 for p in flat["power"].values())  # no flag, nothing to reduce


def test_the_scorer_and_the_sizing_weigh_verified_sites_alike():
    """Estimator parity: the #84 scorer, given Ship A's listening weights, reads the same
    weighted rate the sizing simulates (Σw·A / Σw·C with w = 1 / listen_probability)."""
    from dataclasses import replace

    from tadabur.ship_a import listening_weights
    from tadabur.test_power_frame import H448
    from training.acceptance_stats import ratio
    from training.site_outcomes import CORRECT_SIDE
    from training.truth_scorer import HEADLINE, item_key, reconcile, score

    (swap,) = [s for s in mine_all(scored_pool(base=H448)) if s.kind == "pair"]
    chances, recovered = (1.0, 0.5, 0.25), (True, False, True)
    sites = [replace(swap, listen_probability=p, row=replace(swap.row, site=replace(
        swap.site, site_id=f"{swap.site.site_id}#{k}", audio_filename=f"c{k}.wav",
        audio_sha256=f"{k}" * 64,
        heard=swap.site.prescribed, stratum_population=3))) for k, p in enumerate(chances)]
    truth = [s.site for s in sites]
    weights = listening_weights(sites)
    assert reconcile(truth, weights).weights == weights
    decodes = {"m": {item_key(t): t.reference if ok else t.reference.replace("ذ", "ز")
                     for t, ok in zip(truth, recovered)}}
    report = score(truth, {t.audio_filename: k for k, t in enumerate(truth)}, decodes,
                   design_weights=weights)
    cell = next(c for c in report["cells"] if (c["population"], c["side"], c["label"])
                == (HEADLINE, CORRECT_SIDE, "ذ↔ز:ذ"))
    w = np.array([weights[t.site_id] for t in truth])
    accurate = np.array(recovered, dtype=float)
    expected = ratio(w * accurate, w)
    assert expected == pytest.approx(5 / 7)
    assert cell["arms"]["m"]["rates"]["committed_accuracy"]["point"] == pytest.approx(expected)

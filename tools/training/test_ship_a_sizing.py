"""Ship A's sizing (#126): its simulators on known cases, and the committed results and page."""

from __future__ import annotations

import json

import pytest

from tadabur.power_frame import input_checksums
from tadabur.ship_a import SITES_PATH, load_sites, mine
from tadabur.test_power_frame import REFERENCE, scored_pool
from tadabur.word_sample import WORD_SAMPLE_PATH
from training import ship_a_sizing as z
from training.ship_a_sizing_doc import DOC_PATH, load_inputs, render


def _side(reciters, theta=0.0, rate=1.0, side=z.RECOVERY, rule=z.ONE_PER_RECITER):
    return z.SideScenario("cell", tuple(reciters), theta, rate, 0.15, side, rule)


def test_a_flawless_recovery_reaches_the_wilson_floor_of_its_reciters():
    """40 reciters, every site correct and recovered: degenerate, so Wilson, 40/43.84 = 91%."""
    result = z.side_bounds(_side(range(40)), 20)
    assert result["median_bound"] == pytest.approx(40 / (40 + 1.96**2), abs=1e-3)
    assert result["power"]["0.8"] == 1.0 and result["mean_reciters"] == 40


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


def test_the_word_sample_cannot_show_a_reduction_without_flags():
    result = z.word_sample_power(z.WordScenario(100, 0.0, 0.5), 5)
    assert all(p == 0.0 for p in result["power"].values())


def test_the_budget_counts_each_half_and_marks_sparse_cells():
    sites = mine(scored_pool(base=REFERENCE.replace("بب", "ب")))
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

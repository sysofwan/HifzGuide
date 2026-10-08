"""The truth-site scorer on synthetic sites and decodes: populations, sides, weights, clusters,
exclusions, the allowance view, required cells and the power-simulation inputs."""

from __future__ import annotations

import json

import pytest

from training.site_outcomes import CORRECT_SIDE, DHAL_ZAH, MISTAKE_SIDE
from training.test_site_outcomes import DHAKARA, DHAL, KAF, KATABA, QULHU, TAA, ZAI, FATHA, make
from training.truth_scorer import (
    ALL,
    DIAGNOSTIC,
    HEADLINE,
    PAUSE,
    REQUIRED_CELLS,
    SAFEGUARD,
    item_key,
    population,
    power_inputs,
    score,
    site_weights,
)

SOFT = "ذ↔ز"
WRONG_FATHA = KAF + FATHA + TAA + "ِ" + KATABA[4:]  # kasra on ت


def site(clip: str, reference: str, index: int, mark: str, prescribed: str, heard: str, **extra):
    return make(
        reference, index, mark, prescribed, heard,
        site_id=extra.pop("site_id", f"{clip}#{index}{mark}{heard}"),
        audio_filename=clip,
        **extra,
    )


@pytest.fixture
def sites():
    headline = dict(stratum="new_audit:fatha", stratum_population=10)
    return [
        # Headline correct fatha on three reciters; one pending site in the same stratum.
        site("a.wav", KATABA, 2, "fatha", "fatha", "fatha", **headline),
        site("b.wav", KATABA, 2, "fatha", "fatha", "fatha", **headline),
        site("c.wav", KATABA, 2, "fatha", "fatha", "kasra", **headline),
        site("d.wav", KATABA, 2, "fatha", "fatha", "pending", **headline),
        # A pause sukun and a weak-label wasl site.
        site("e.wav", QULHU, 2, "sukun", "sukun", "sukun", source="waqf_boundary",
             stratum="waqf_boundary:waqf"),
        site("e.wav", QULHU, 0, "damma", "damma", "damma", source="waqf_boundary",
             assumes_competent_reciter=True, stratum="waqf_boundary:wasl"),
        # A P3.5 safeguard on the soft pair.
        site("f.wav", DHAKARA, 2, SOFT, DHAL, DHAL, source="p35_fixture", stratum="p35_fixture:x"),
    ]


RECITERS = {"a.wav": 1, "b.wav": 2, "c.wav": 3, "d.wav": 3, "e.wav": 4, "f.wav": 5}


def decodes_for(sites, overrides=None):
    overrides = overrides or {}
    return {item_key(s): overrides.get(s.audio_filename, s.reference) for s in sites}


def cell(report, population_, side_, label):
    return next(
        c for c in report["cells"]
        if (c["population"], c["side"], c["label"]) == (population_, side_, label)
    )


def test_populations_come_from_the_truth_record(sites):
    assert [population(s) for s in sites] == [
        HEADLINE, HEADLINE, HEADLINE, HEADLINE, PAUSE, DIAGNOSTIC, SAFEGUARD,
    ]


def test_weights_count_sampled_sites_including_pending(sites):
    weights = site_weights(sites)
    assert weights["a.wav#2fathafatha"] == pytest.approx(10 / 4)
    assert weights["f.wav#2" + SOFT + DHAL] == 1


def test_sides_are_never_pooled_and_rates_are_weighted(sites):
    decodes = {
        "m/spans": decodes_for(sites, {"b.wav": KAF + FATHA + TAA + KATABA[4:]}),
        "m/stream": decodes_for(sites),
    }
    report = score(sites, RECITERS, decodes, [("m/spans", "m/stream")])
    correct = cell(report, HEADLINE, CORRECT_SIDE, "fatha")
    assert (correct["sites"], correct["reciters"], correct["sparse"]) == (2, 2, True)
    rates = correct["arms"]["m/spans"]["rates"]
    assert rates["commit_rate"]["point"] == 0.5  # b dropped the fatha
    assert rates["committed_accuracy"]["point"] == 1.0
    assert rates["false_flags"]["point"] == 0.5  # a dropped haraka on ت is flagged today
    assert rates["commit_rate"]["num"] == pytest.approx(2.5)  # weights 10/4
    mistake = cell(report, HEADLINE, MISTAKE_SIDE, "kasra")
    assert mistake["sites"] == 1
    m = mistake["arms"]["m/spans"]["rates"]
    assert m["committed_accuracy"]["point"] == 0.0  # it decoded the mushaf's fatha
    assert m["silent_corrections"]["point"] == 1.0 and m["missed_mistakes"]["point"] == 1.0
    pooled = cell(report, HEADLINE, CORRECT_SIDE, ALL)
    assert pooled["sites"] == 2  # the mistake is not in the correct-side pool
    delta = correct["differences"]["m/spans - m/stream"]["commit_rate"]
    assert delta["point"] == -0.5


def test_a_single_reciter_cell_has_a_degenerate_interval(sites):
    report = score(sites, RECITERS, {"m/spans": decodes_for(sites)})
    pause = cell(report, PAUSE, CORRECT_SIDE, "sukun")
    assert pause["reciters"] == 1
    spurious = pause["arms"]["m/spans"]["rates"]["spurious_haraka"]
    assert spurious["point"] == 0.0 and spurious["degenerate"]


def test_exclusions_and_their_sensitivity(sites):
    report = score(sites, RECITERS, {"m/spans": decodes_for(sites)})
    assert report["exclusions"] == [
        {"population": HEADLINE, "stratum": "new_audit:fatha", "mark": "fatha",
         "prescribed": "fatha", "heard": "pending", "sites": 1},
    ]
    correct = cell(report, HEADLINE, CORRECT_SIDE, "fatha")
    assert correct["excluded"] == 1  # the pending fatha could turn out correct
    sensitivity = correct["arms"]["m/spans"]["sensitivity"]["commit_rate"]
    assert sensitivity == {"worst": pytest.approx(2 / 3), "best": pytest.approx(1.0)}
    assert cell(report, HEADLINE, MISTAKE_SIDE, "kasra")["excluded"] == 1


def test_every_arm_needs_every_item(sites):
    partial = decodes_for(sites)
    partial.pop(item_key(sites[0]))
    with pytest.raises(ValueError, match="no decode"):
        score(sites, RECITERS, {"m/spans": partial})


def test_duplicate_site_ids_are_refused(sites):
    with pytest.raises(ValueError, match="duplicate"):
        score(sites + [sites[0]], RECITERS, {"m/spans": decodes_for(sites)})


def test_required_cells_are_directional_and_keep_unsupported_pairs(sites):
    labels = {(r.gate, r.rule, r.cell.side, r.cell.label) for r in REQUIRED_CELLS}
    for letter in DHAL_ZAH.split("↔"):
        assert ("§2 probe", "commit rate", CORRECT_SIDE, f"{DHAL_ZAH}:{letter}") in labels
        assert ("§2 probe", "committed accuracy", MISTAKE_SIDE, f"{DHAL_ZAH}:{letter}") in labels
    assert ("§2 probe", "sukun commit rate", CORRECT_SIDE, "sukun") in labels
    assert ("§3 ship", "false flags (relative change)", CORRECT_SIDE, ALL) in labels
    assert all(r.cell.population in (HEADLINE, PAUSE) for r in REQUIRED_CELLS)
    report = score(sites, RECITERS, {"m/spans": decodes_for(sites)})
    empty = cell(report, HEADLINE, CORRECT_SIDE, f"{DHAL_ZAH}:{DHAL}")
    assert empty["sites"] == 0 and "arms" not in empty
    assert len(report["required_cells"]) == len(REQUIRED_CELLS)


def test_allowance_view_switches_one_allowance_off(sites):
    swapped = DHAKARA.replace(DHAL, ZAI)
    report = score(sites, RECITERS, {"m/spans": decodes_for(sites, {"f.wav": swapped})})
    entry = next(
        a for a in report["allowances"]
        if a["population"] == SAFEGUARD and a["allowance"] == f"soft_pair {SOFT}"
    )
    block = entry["sides"]["false_flags"]
    assert block["sites"] == 1
    assert block["m/spans"]["on"]["point"] == 0.0 and block["m/spans"]["off"]["point"] == 1.0
    assert entry["sides"]["missed_mistakes"]["sites"] == 0


def test_power_inputs_hold_per_reciter_sums(sites):
    inputs = power_inputs(sites, RECITERS, {"h448/stream_b0": decodes_for(sites)})
    row = next(
        c for c in inputs["cells"]
        if c["population"] == HEADLINE and c["side"] == CORRECT_SIDE and c["heard"] == "fatha"
    )
    assert inputs["per_reciter_fields"][:3] == ["sites", "w", "w_commit"]
    assert row["per_reciter"]["1"][:3] == [1, 2.5, 2.5]
    assert inputs["excluded"] == {"new_audit:fatha": 1}
    json.dumps(inputs)


def test_the_report_is_plain_json(sites):
    report = score(sites, RECITERS, {"m/spans": decodes_for(sites, {"c.wav": WRONG_FATHA})})
    json.dumps(report, allow_nan=False)

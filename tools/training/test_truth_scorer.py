"""The truth-site scorer on synthetic sites and decodes: populations, sides, weights, clusters,
exclusions, the allowance view, required cells and the power-simulation inputs."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from training.muraja_policy import EVERY_WORD_ENDS, MURAJA_REVISION, RUN_ENDS
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
    reconcile,
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


def test_reported_intervals_follow_the_section_1_method_rules(sites):
    report = score(sites, RECITERS, {"m/spans": decodes_for(sites)})
    pause = cell(report, PAUSE, CORRECT_SIDE, "sukun")  # one site: independent, unit weight
    spurious = pause["arms"]["m/spans"]["rates"]["spurious_haraka"]
    assert spurious["point"] == 0.0 and spurious["method"] == "wilson"
    headline = cell(report, HEADLINE, CORRECT_SIDE, "fatha")  # weights 10/4: not unit
    rate = headline["arms"]["m/spans"]["rates"]["commit_rate"]
    assert rate["method"] == "none" and rate["lower"] is None and rate["upper"] is None


def test_three_flawless_independent_sites_get_the_wilson_bound():
    trio = [site(f"{c}.wav", KATABA, 2, "fatha", "fatha", "fatha", stratum="t",
                 stratum_population=3) for c in "abc"]
    report = score(trio, RECITERS, {"m/spans": decodes_for(trio)})
    rate = cell(report, HEADLINE, CORRECT_SIDE, "fatha")["arms"]["m/spans"]["rates"]
    assert rate["commit_rate"]["method"] == "wilson"
    assert rate["commit_rate"]["lower"] == pytest.approx(0.438493919551)


def test_one_physical_site_counts_once_and_keeps_its_frozen_weight(sites):
    twin = replace(sites[0], site_id="twin", source="p35_fixture", heard="pending",
                   stratum="p35_fixture:x")
    report = score(sites + [twin], RECITERS, {"m/spans": decodes_for(sites)})
    assert report["physical_sites_merged"] == [{"kept": sites[0].site_id, "dropped": ["twin"]}]
    correct = cell(report, HEADLINE, CORRECT_SIDE, "fatha")
    assert correct["sites"] == 2
    assert correct["arms"]["m/spans"]["rates"]["commit_rate"]["num"] == pytest.approx(5.0)
    safeguard = cell(report, SAFEGUARD, CORRECT_SIDE, ALL)
    assert safeguard["arms"]["m/spans"]["rates"]["commit_rate"]["den"] == pytest.approx(0.5)


def test_the_directly_adjudicated_verdict_wins_over_a_weak_label(sites):
    weak = replace(sites[0], site_id="weak", assumes_competent_reciter=True, heard="fatha")
    direct = replace(sites[0], site_id="direct", heard="kasra")
    reconciled = reconcile([weak, direct])
    assert [s.site_id for s in reconciled.sites] == ["direct"]


def test_a_direct_unclear_verdict_wins_over_a_weak_label_and_is_excluded(sites):
    weak = replace(sites[0], site_id="weak", assumes_competent_reciter=True, heard="fatha")
    unclear = replace(sites[0], site_id="unclear", heard="unclear")
    assert [s.site_id for s in reconcile([weak, unclear]).sites] == ["unclear"]
    report = score([weak, unclear], RECITERS, {"m/spans": decodes_for([weak])})
    assert [row["heard"] for row in report["exclusions"]] == ["unclear"]
    assert report["physical_sites_merged"] == [{"kept": "unclear", "dropped": ["weak"]}]


def test_pending_yields_to_an_adjudication_but_never_to_a_weak_label(sites):
    pending = replace(sites[0], site_id="pending", heard="pending")
    weak = replace(sites[0], site_id="weak", assumes_competent_reciter=True, heard="fatha")
    assert [s.site_id for s in reconcile([weak, pending]).sites] == ["pending"]
    unclear = replace(sites[0], site_id="unclear", heard="unclear")
    assert [s.site_id for s in reconcile([pending, unclear]).sites] == ["unclear"]
    direct = replace(sites[0], site_id="direct", heard="kasra")
    assert [s.site_id for s in reconcile([pending, unclear, direct]).sites] == ["direct"]


def test_conflicting_labels_of_one_physical_site_fail_loudly(sites):
    other = replace(sites[0], site_id="other", heard="kasra")
    with pytest.raises(ValueError, match="disagree"):
        reconcile([sites[0], other])
    with pytest.raises(ValueError, match="prescribed"):
        reconcile([sites[0], replace(sites[0], site_id="x", prescribed="kasra")])


def test_power_inputs_use_the_reconciled_sites(sites):
    twin = replace(sites[0], site_id="twin", heard="pending")
    inputs = power_inputs(sites + [twin], RECITERS, {"h448/stream_b0": decodes_for(sites)})
    assert inputs["excluded"] == {"new_audit:fatha": 1}  # the twin was dropped, d.wav stays


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
    assert ("§5 bias", "spurious haraka", MISTAKE_SIDE, "sukun") in labels  # its own row
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


def test_the_allowances_off_view_and_the_approximation_are_reported(sites):
    swapped = DHAKARA.replace(DHAL, ZAI)
    report = score(sites, RECITERS, {"m/spans": decodes_for(sites, {"f.wav": swapped})})
    rates = cell(report, SAFEGUARD, CORRECT_SIDE, ALL)["arms"]["m/spans"]["rates"]
    assert rates["false_flags"]["point"] == 0.0  # the soft pair is forgiven today
    assert rates["false_flags@allowances_off"]["point"] == 1.0
    assert rates["coverage@allowances_off"]["point"] == 1.0
    assert "false_flags@every_word_ends" in rates
    views = report["muraja_views"]
    assert views["today"]["approximation"] == RUN_ENDS and views["today"]["statement"]
    assert views["every_word_ends"]["approximation"] == EVERY_WORD_ENDS
    assert views["allowances_off"]["config"]["soft_pairs"] == []
    assert report["muraja_config"]["revision"] == MURAJA_REVISION


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


def test_required_cells_do_not_depend_on_string_hashing():
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = "from training.truth_scorer import REQUIRED_CELLS; print([r.cell.label for r in REQUIRED_CELLS])"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent.parent, env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(outputs) == 1

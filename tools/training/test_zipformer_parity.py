"""Tests for the sherpa-onnx parity gate's pure rule (#245); torch-free, no model.

``training.zipformer_parity`` holds the pre-declared rule (``.orch/tasks/zip-run/parity-gate.md``
as amended by principal r2 and astra r1): each benchmark source must decode identically on 99%
of its clips with at most 1 native-unit edit per thousand of our units, the phone table-hit
cohort must stay within 1 character per thousand reference characters, and the verdict-adapter
probe's cut decisions must agree. Only the arithmetic is tested here; the two phases that decode
need torch or sherpa-onnx and run on the box.
"""

from __future__ import annotations

import pytest

from training.zipformer_parity import (
    CHARACTER_PER_MILLE,
    EDITS_PER_MILLE,
    IDENTICAL_PERCENT,
    PROBE_CHECKS,
    ParityStop,
    cut_certainty,
    cut_decision,
    identical_clip,
    our_calls,
    phone_rule,
    probe_check,
    require_parity,
    sherpa_calls,
    source_rule,
)


def _row(key: str, source: str, *, units: int = 2, edits: int = 0,
         padded: int = 0) -> dict:
    """One clip's rule row: ``units`` units of ours, ``edits`` of them different in sherpa's."""
    ours = ["ب"] * units
    theirs = ["ب"] * (units - edits) + ["ت"] * edits
    return {"key": key, "source": source, "ours": ours, "sherpa": theirs,
            "padded_tokens": [17] * padded}


def _record(*, sources=None, phone=None, checks=None) -> dict:
    if sources is None:
        sources = {"phone": {"passes": True, "clips": 1, "identical": 1,
                             "identical_percent": 100.0, "unit_edits": 0, "units_ours": 2,
                             "edits_per_mille": 0.0}}
    if phone is None:
        phone = {"passes": True, "character_distance": 0, "reference_characters": 10,
                 "per_mille": 0.0}
    if checks is None:
        checks = [{"passes": True, "key": "a", "check": "first-window", "tokens_equal": True,
                   "mismatches": []}]
    return {"sources": sources, "phone_table_hits": phone, "probe": {"checks": checks}}


# --- the thresholds are the pre-registered ones ---------------------------------------------


def test_the_thresholds_are_the_ones_fixed_before_the_run():
    assert (IDENTICAL_PERCENT, EDITS_PER_MILLE, CHARACTER_PER_MILLE) == (99, 1, 1)
    assert PROBE_CHECKS == ("first-preview", "first-window", "later-window", "stop")


# --- the per-source clip and unit rules ------------------------------------------------------


def test_a_source_at_the_clip_floor_passes_and_one_clip_more_differing_fails():
    rows = [_row(f"s{index}", "studio", units=50,
                 edits=0 if index >= 2 else 1) for index in range(200)]
    totals = source_rule(rows, ["studio"])
    assert totals["studio"]["identical"] == 198
    assert totals["studio"]["passes"] is True

    rows[2] = _row("s2", "studio", units=50, edits=1)
    totals = source_rule(rows, ["studio"])
    assert totals["studio"]["identical"] == 197
    assert totals["studio"]["passes"] is False
    assert totals["studio"]["differing"][:3] == ["s0", "s1", "s2"]


def test_the_floor_is_a_share_of_a_smaller_source_too():
    rows = [_row(f"p{index}", "phone", units=50, edits=0 if index else 1)
            for index in range(177)]
    assert source_rule(rows, ["phone"])["phone"]["passes"] is True

    rows[1] = _row("p1", "phone", units=50, edits=1)
    assert source_rule(rows, ["phone"])["phone"]["passes"] is False


def test_the_edit_rule_is_one_edit_per_thousand_of_our_units():
    rows = [_row(f"s{index}", "studio", units=100) for index in range(200)]
    rows[0] = _row("s0", "studio", units=100, edits=20)
    totals = source_rule(rows, ["studio"])
    assert (totals["studio"]["unit_edits"], totals["studio"]["units_ours"]) == (20, 20_000)
    assert totals["studio"]["passes"] is True

    rows[0] = _row("s0", "studio", units=100, edits=21)
    assert source_rule(rows, ["studio"])["studio"]["passes"] is False


def test_a_padded_call_emission_differences_a_clip_whose_text_agrees():
    totals = source_rule([_row("a", "phone", units=10, padded=1)], ["phone"])
    assert totals["phone"]["identical"] == 0
    assert totals["phone"]["padded_clips"] == 1
    assert totals["phone"]["differing"] == ["a"]


def test_a_source_with_no_clips_cannot_pass():
    assert source_rule([], ["unseen"])["unseen"]["passes"] is False


def test_identical_means_the_same_units_and_no_padded_emission():
    assert identical_clip(["ب", "ت"], ["ب", "ت"], [])
    assert not identical_clip(["ب", "ت"], ["ب", "ت"], [17])
    assert not identical_clip(["ب", "تت"], ["ب", "ت"], [])


# --- the phone table-hit bound ---------------------------------------------------------------


def test_the_phone_bound_is_characters_over_the_references_own_characters():
    assert phone_rule(100, 100_000)["passes"] is True
    assert phone_rule(101, 100_000)["passes"] is False
    assert phone_rule(1, 100_000)["per_mille"] == pytest.approx(0.01)


def test_a_cohort_without_reference_characters_is_refused():
    with pytest.raises(ValueError):
        phone_rule(0, 0)


# --- the probe's cut decisions ---------------------------------------------------------------


def test_a_cut_keeps_an_emission_exactly_at_the_boundary():
    assert cut_decision(4, 6, 5)
    assert not cut_decision(4, 5, 5)


def test_sherpa_certainty_uses_both_ends_of_the_run_it_cannot_see():
    assert cut_certainty(4, 8, 4) == "kept"
    assert cut_certainty(4, 8, 5) == "ambiguous"
    assert cut_certainty(4, 8, 6) == "dropped"
    assert cut_certainty(4, None, 4) == "kept"


def _probe(ours, sherpa, boundaries, check="first-window") -> dict:
    return {"check": check, "boundaries": boundaries, "ours": ours, "sherpa": sherpa}


def test_a_probe_check_passes_when_the_units_and_cut_decisions_agree():
    result = probe_check(_probe(
        [{"text": "ب", "start": 0, "end": 2}, {"text": "ت", "start": 6, "end": 9}],
        [{"text": "ب", "start": 0}, {"text": "ت", "start": 6}], [5]))
    assert result["passes"] and result["tokens_equal"] and result["ambiguous"] == 0


def test_a_probe_check_fails_when_a_determined_cut_differs():
    result = probe_check(_probe(
        [{"text": "ب", "start": 0, "end": 12}], [{"text": "ب", "start": 0}], [5]))
    assert result["passes"] is False
    assert result["mismatches"] == [
        {"boundary": 5, "index": 0, "unit": "ب", "sherpa": "dropped", "ours": "kept"}]


def test_a_probe_check_reports_an_ambiguous_cut_instead_of_failing_on_it():
    result = probe_check(_probe(
        [{"text": "ب", "start": 5, "end": 7}, {"text": "ت", "start": 8, "end": 9}],
        [{"text": "ب", "start": 5}, {"text": "ت", "start": 8}], [6]))
    assert result["passes"] is True
    assert result["ambiguous"] == 1
    assert result["mismatches"] == []


def test_a_probe_check_fails_when_the_runtimes_emit_different_units():
    result = probe_check(_probe(
        [{"text": "ب", "start": 0, "end": 1}], [{"text": "ت", "start": 0}], [1]))
    assert result["passes"] is False
    assert result["tokens_equal"] is False


# --- the call counts the padded-chunk rule rests on ------------------------------------------


def test_our_loop_pads_the_last_call_and_sherpa_never_runs_a_padded_one():
    assert our_calls(61, chunk_frames=48) == 2
    assert sherpa_calls(61, input_frames=61, chunk_frames=48) == 0
    assert our_calls(605, chunk_frames=48) == 13
    assert sherpa_calls(605, input_frames=61, chunk_frames=48) == 12
    assert our_calls(852, chunk_frames=48) == 18
    assert sherpa_calls(852, input_frames=61, chunk_frames=48) == 17


# --- the gate itself -------------------------------------------------------------------------


def test_a_clean_record_passes():
    require_parity(_record())


def test_every_failing_rule_is_named_in_one_stop():
    failing = {"passes": False, "clips": 200, "identical": 190, "identical_percent": 95.0,
               "unit_edits": 4, "units_ours": 1000, "edits_per_mille": 4.0}
    with pytest.raises(ParityStop) as raised:
        require_parity(_record(
            sources={"studio": failing},
            phone={"passes": False, "character_distance": 9, "reference_characters": 100,
                   "per_mille": 90.0},
            checks=[{"passes": False, "key": "a", "check": "stop", "tokens_equal": False,
                     "mismatches": []}]))
    message = str(raised.value)
    assert "studio" in message
    assert "phone table hits" in message
    assert "probe a stop" in message


def test_a_probe_with_no_checks_cannot_pass():
    with pytest.raises(ParityStop):
        require_parity(_record(checks=[]))

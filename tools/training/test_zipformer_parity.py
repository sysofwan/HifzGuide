"""Tests for the sherpa-onnx parity gate's pure rule (#245); torch-free, no model.

``training.zipformer_parity`` holds the pre-declared rule (``.orch/tasks/zip-run/parity-gate.md``
as amended by principal r2 and astra r1): each benchmark source must decode identically on 99%
of its clips with at most 1 native-unit edit per thousand of our units, the phone table-hit
cohort must stay within 1 character per thousand reference characters, and the verdict-adapter
probe's cut decisions must agree. No call may be padded: the streams follow sherpa-onnx's
readiness rule, and the gate compares the two runtimes' call counts on every clip. Only the
arithmetic is tested here; the two phases that decode need torch or sherpa-onnx and run on the
box.
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
    chunk_calls,
    phone_rule,
    probe_check,
    require_parity,
    source_rule,
)


def _row(key: str, source: str, *, units: int = 2, edits: int = 0) -> dict:
    """One clip's rule row: ``units`` units of ours, ``edits`` of them different in sherpa's."""
    ours = ["ب"] * units
    theirs = ["ب"] * (units - edits) + ["ت"] * edits
    return {"key": key, "source": source, "ours": ours, "sherpa": theirs}


def _record(*, sources=None, phone=None, checks=None) -> dict:
    if sources is None:
        sources = {"phone": {"passes": True, "clips": 1, "identical": 1,
                             "identical_percent": 100.0, "unit_edits": 0, "units_ours": 2,
                             "edits_per_mille": 0.0}}
    if phone is None:
        phone = {"passes": True, "clips": 1, "missing": [], "character_distance": 0,
                 "reference_characters": 10, "per_mille": 0.0}
    if checks is None:
        checks = [{"passes": True, "key": "a", "check": "first-window", "tokens_equal": True,
                   "ambiguous": 0, "mismatches": []}]
    return {"gate": "sherpa-parity", "passed": True,
            "plan": {"clips": 1, "decoded": 1, "sources": {"phone": 1}, "partial": False},
            "sources": sources, "phone_table_hits": phone,
            "calls": {"agree": True, "mismatched": []},
            "audio": {"sample_rates": {16000: 1}}, "probe": {"checks": checks}}


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


def test_a_source_with_no_clips_cannot_pass():
    assert source_rule([], ["unseen"])["unseen"]["passes"] is False


def test_identical_means_the_same_units_in_the_same_order():
    assert identical_clip(["ب", "ت"], ["ب", "ت"])
    assert not identical_clip(["ب", "تت"], ["ب", "ت"])
    assert not identical_clip(["ب", "ت"], ["ت", "ب"])


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


def test_the_terminal_run_ends_at_the_crop_not_at_its_start():
    # the last token's run is [40, 64] (the crop's last step): its midpoint is 52
    assert cut_certainty(40, None, 40, last_step=64) == "kept"
    assert cut_certainty(40, None, 50, last_step=64) == "ambiguous"
    assert cut_certainty(40, None, 52, last_step=64) == "ambiguous"
    assert cut_certainty(40, None, 53, last_step=64) == "dropped"
    # assuming a one-frame run (no last_step) turns the same run's 50 into a determined drop,
    # which is the latent false STOP; and it can never rise to ambiguous where a real
    # disagreement sits, which is the latent false PASS
    assert cut_certainty(40, None, 50) == "dropped"


def _probe(ours, sherpa, boundaries, check="first-window", last_step=None,
           runs=None) -> dict:
    entry = {"check": check, "boundaries": boundaries, "ours": ours, "sherpa": sherpa,
             "last_step": last_step}
    if runs is not None:
        entry["runs"] = runs
    return entry


def test_a_terminal_cut_is_checked_on_the_crop_s_end():
    ours = [{"text": "ب", "start": 40, "end": 64}]
    sherpa = [{"text": "ب", "start": 40}]
    # with the crop's last step the run spans [40, 64], straddling 50: the cut is open, and an
    # open cut is refused rather than passed
    bounded = probe_check(_probe(ours, sherpa, [50], last_step=64))
    assert bounded["ambiguous"] == 1 and bounded["passes"] is False
    # without it the run is read as one frame, a determined drop: a mismatch at that boundary
    unread = probe_check(_probe(ours, sherpa, [50]))
    assert unread["mismatches"][0]["sherpa"] == "dropped"
    # the reference's own frame output ends the run at 64, which keeps it at 50
    traced = probe_check(_probe(ours, sherpa, [50], last_step=64,
                                runs=[{"text": "ب", "start": 40, "end": 64}]))
    assert traced["passes"] and traced["ambiguous"] == 0 and traced["resolved"] == 1


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


def test_an_unresolved_cut_is_refused_not_passed():
    # ours keeps `ب` over [5, 5]; sherpa starts at 5 and its next token starts at 8, so its run
    # ends at 5, 6 or 7: ending at 6 or later changes the field at boundary 6
    entry = _probe(
        [{"text": "ب", "start": 5, "end": 5}, {"text": "ت", "start": 8, "end": 9}],
        [{"text": "ب", "start": 5}, {"text": "ت", "start": 8}], [6])
    refused = probe_check(entry)
    assert refused["ambiguous"] == 1
    assert refused["mismatches"] == []
    assert refused["passes"] is False


def test_a_traced_run_end_resolves_the_open_cut_either_way():
    ours = [{"text": "ب", "start": 5, "end": 5}, {"text": "ت", "start": 8, "end": 9}]
    sherpa = [{"text": "ب", "start": 5}, {"text": "ت", "start": 8}]
    keeps = probe_check(_probe(ours, sherpa, [6], runs=[
        {"text": "ب", "start": 5, "end": 7}, {"text": "ت", "start": 8, "end": 9}]))
    assert keeps["mismatches"] == [{"boundary": 6, "index": 0, "unit": "ب",
                                    "sherpa": "kept", "ours": "dropped"}]
    assert keeps["passes"] is False and keeps["ambiguous"] == 0
    drops = probe_check(_probe(ours, sherpa, [6], runs=[
        {"text": "ب", "start": 5, "end": 5}, {"text": "ت", "start": 8, "end": 9}]))
    assert drops["passes"] and drops["resolved"] == 1 and drops["ambiguous"] == 0


def test_a_trace_that_is_not_the_reference_runtimes_output_is_refused():
    # the trace dropped a unit sherpa emitted: its runs are not that runtime's own output, so
    # the open cut stays open rather than being resolved from a different decoding pass
    result = probe_check(_probe(
        [{"text": "ب", "start": 5, "end": 5}, {"text": "ت", "start": 8, "end": 9}],
        [{"text": "ب", "start": 5}, {"text": "ت", "start": 8}], [6],
        runs=[{"text": "ب", "start": 5, "end": 5}]))
    assert result["ambiguous"] == 1 and result["passes"] is False
    assert result["trace"] == "refused: the trace is not the reference runtime's own output"


def test_a_probe_check_fails_when_the_runtimes_emit_different_units():
    result = probe_check(_probe(
        [{"text": "ب", "start": 0, "end": 1}], [{"text": "ت", "start": 0}], [1]))
    assert result["passes"] is False
    assert result["tokens_equal"] is False


# --- the call counts the padded-chunk rule rests on ------------------------------------------


def test_the_readiness_rule_never_runs_a_call_it_would_have_to_pad():
    # both runtimes stop while `processed + 61 < frames`: 61 frames is no call at all, and
    # 605 frames is 12 calls where the old loop padded a 13th
    assert chunk_calls(61, input_frames=61, chunk_frames=48) == 0
    assert chunk_calls(62, input_frames=61, chunk_frames=48) == 1
    assert chunk_calls(109, input_frames=61, chunk_frames=48) == 1
    assert chunk_calls(110, input_frames=61, chunk_frames=48) == 2
    assert chunk_calls(605, input_frames=61, chunk_frames=48) == 12
    assert chunk_calls(852, input_frames=61, chunk_frames=48) == 17


# --- the gate itself -------------------------------------------------------------------------


def test_a_clean_record_passes():
    require_parity(_record(), {"phone": 1})


def test_a_run_short_of_the_plan_cannot_pass():
    # one source, three clips, every one identical: the rule's 196/105/189 never held
    short = {"studio": {"passes": True, "clips": 3, "identical": 3,
                        "identical_percent": 100.0, "unit_edits": 0, "units_ours": 6,
                        "edits_per_mille": 0.0}}
    with pytest.raises(ParityStop, match="studio: 3 of the plan's 196 clips"):
        require_parity(_record(sources=short),
                       {"studio": 196, "phone": 105, "unseen": 189})


def test_a_source_the_plan_names_but_the_record_omits_cannot_pass():
    with pytest.raises(ParityStop, match="unseen: 0 of the plan's 189 clips"):
        require_parity(_record(), {"phone": 1, "unseen": 189})


def test_a_non_16_khz_clip_cannot_pass():
    record = _record()
    record["audio"]["sample_rates"] = {22050: 1}
    with pytest.raises(ParityStop, match="not 16000 Hz"):
        require_parity(record, {"phone": 1})


def test_a_phone_cohort_clip_one_runtime_skipped_cannot_pass():
    record = _record()
    record["phone_table_hits"]["missing"] = ["p1"]
    with pytest.raises(ParityStop, match="not decoded by both runtimes"):
        require_parity(record, {"phone": 1})


def test_every_failing_rule_is_named_in_one_stop():
    failing = {"passes": False, "clips": 200, "identical": 190, "identical_percent": 95.0,
               "unit_edits": 4, "units_ours": 1000, "edits_per_mille": 4.0}
    with pytest.raises(ParityStop) as raised:
        require_parity(_record(
            sources={"studio": failing},
            phone={"passes": False, "clips": 105, "missing": [], "character_distance": 9,
                   "reference_characters": 100, "per_mille": 90.0},
            checks=[{"passes": False, "key": "a", "check": "stop", "tokens_equal": False,
                     "mismatches": []}]), {"studio": 200})
    message = str(raised.value)
    assert "studio" in message
    assert "phone table hits" in message
    assert "probe a stop" in message


def test_a_probe_with_no_checks_cannot_pass():
    with pytest.raises(ParityStop):
        require_parity(_record(checks=[]), {"phone": 1})


def test_a_record_without_its_plan_cannot_pass():
    record = _record()
    record.pop("plan")
    with pytest.raises(ParityStop, match="does not carry the plan"):
        require_parity(record, {"phone": 1})


def test_a_partial_plan_cannot_pass_even_when_every_decoded_clip_is_identical():
    record = _record()
    record["plan"] = {"clips": 577, "decoded": 200, "sources": {"phone": 200},
                      "partial": True}
    with pytest.raises(ParityStop, match="plan: 200 of 577"):
        require_parity(record, {"phone": 200})


def test_a_plan_whose_per_source_counts_do_not_sum_to_its_size_cannot_pass():
    record = _record()
    record["plan"] = {"clips": 577, "decoded": 577, "sources": {"phone": 200},
                      "partial": False}
    with pytest.raises(ParityStop, match="per-source counts sum to 200"):
        require_parity(record, {"phone": 200})


def test_clips_that_made_a_different_number_of_calls_cannot_pass():
    record = _record()
    record["calls"] = {"agree": False, "mismatched": ["a"]}
    with pytest.raises(ParityStop, match="different number of calls"):
        require_parity(record, {"phone": 1})


def test_an_unresolved_probe_cut_is_named_in_the_stop():
    with pytest.raises(ParityStop, match="1 cut decisions are unresolved"):
        require_parity(_record(checks=[{
            "passes": False, "key": "a", "check": "first-window", "tokens_equal": True,
            "ambiguous": 1, "resolved": 0, "mismatches": []}]), {"phone": 1})


def test_the_committed_record_is_validated_not_trusted():
    from training.zipformer_parity import require_valid_record

    failed = {**_record(), "passed": False, "stop": "a source failed"}
    with pytest.raises(ParityStop, match="did not pass: a source failed"):
        require_valid_record(failed)
    unplanned = {**_record(), "plan": {}}
    with pytest.raises(ParityStop, match="no plan's per-source counts"):
        require_valid_record(unplanned)

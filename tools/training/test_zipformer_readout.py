"""Tests for the zipformer screen readout's rules (#245): the gap closure and the verdict.

The heavy sections (the harness, the benchmark references, the decodes) are proven by the
committed record they wrote; these tests pin the two rules a reader must not have to re-derive:
what "closes half the gap" means for a metric that is lower-is-better, and how the three
preregistered outcomes follow from the six numbers. Torch-free, no decodes, no harness.
"""

from __future__ import annotations

import json

import pytest

from training import zipformer_readout as readout
from training.zipformer_comparison_doc import render


# --- closing the gap -----------------------------------------------------------------------


def test_closing_half_the_gap_on_a_lower_is_better_metric():
    # teacher 10, candidate 20: the gap is 10, so zipformer must reach 15
    row = readout.closure(teacher=10.0, candidate=20.0, arm=15.0)
    assert row["gap"] == 10.0 and row["target"] == 15.0 and row["closed"] is True
    assert readout.closure(10.0, 20.0, 15.01)["closed"] is False
    assert readout.closure(10.0, 20.0, 14.0)["closed"] is True


def test_a_gap_the_candidate_already_closed_is_closed_by_matching_the_candidate():
    # the candidate is below the teacher (gap -2): anything at or below the candidate closes it
    assert readout.closure(10.0, 8.0, 8.0)["closed"] is True
    assert readout.closure(10.0, 8.0, 9.5)["closed"] is False
    assert readout.closure(10.0, 10.0, 10.0)["closed"] is True


# --- the verdict ---------------------------------------------------------------------------


def _record(*, qdat=20.0, consonant=2.0, haraka=2.0, heard=35.0, decoy=96.0, ship=35.0) -> dict:
    """A record shaped like the readout's, with only the numbers the verdict reads."""
    def errors(value):
        return {"consonant": {"error": value}, "haraka": {"error": value}}

    return {
        "qdat": {"models": {"base": {"point": 30.0}, "r2_candidate_ema": {"point": 37.0},
                            "zipformer-checks-w1": {"point": qdat}}},
        "benchmark": {"sources": {"tlog_holdout": {
            "base": errors(11.0), "r2_candidate_ema": errors(10.0),
            "zipformer": {"consonant": {"error": consonant},
                          "haraka": {"error": haraka}}}}},
        "edits": {"arms": {
            "edit": {"zipformer": {"point": heard}, "r2_candidate_ema": {"point": 35.0},
                     "base": {"point": 47.0}, "c95": {"point": 37.0}},
            "decoy": {"zipformer": {"point": decoy}, "r2_candidate_ema": {"point": 67.0},
                      "base": {"point": 99.0}, "c95": {"point": 71.0}}}},
        "ship_a": {"sites": 530, "mistakes": 75, "correct": 410, "without_side": 45,
                   "mistakes_flagged": {"models": {"zipformer-checks-w1": {"point": ship},
                                                   "r2_candidate_ema": {"point": 35.0},
                                                   "base": {"point": 33.0}}}},
        "gates": {"parity": {"validated": True, "stop": None}},
        "probe": {"stop": False, "projected_hours": 3.6, "horizon_hours": 12.0},
    }


def test_a_model_that_closes_the_gap_and_keeps_the_catch_side_is_go():
    verdict = readout.verdict(_record())
    assert verdict["correct_passed"] and verdict["catch_passed"]
    assert verdict["verdict"] == "GO"


def test_a_model_that_passes_the_correct_side_but_loses_the_catch_side_is_prior():
    verdict = readout.verdict(_record(heard=17.0))
    assert verdict["correct_passed"] and not verdict["catch_passed"]
    assert verdict["verdict"] == "PRIOR"


def test_decoys_under_the_floor_are_a_catch_failure():
    verdict = readout.verdict(_record(decoy=94.7))
    assert verdict["catch_side"]["edit_decoys_unchanged"]["passed"] is False
    assert verdict["verdict"] == "PRIOR"


def test_a_model_that_misses_the_correct_side_is_stop():
    verdict = readout.verdict(_record(qdat=40.0))
    assert verdict["correct_passed"] is False
    assert verdict["verdict"] == "STOP"


def test_the_ship_a_bar_is_the_candidate_minus_five_points():
    verdict = readout.verdict(_record(ship=30.5))
    bar = verdict["catch_side"]["ship_a_mistakes_flagged"]
    assert bar["bar"] == pytest.approx(30.0)
    assert bar["passed"] is True
    assert readout.verdict(_record(ship=29.0))["verdict"] == "PRIOR"


# --- the stop gates: no verdict from a gated-out run --------------------------------------


def test_a_failed_or_unvalidated_parity_gate_refuses_the_verdict():
    record = _record()
    record["gates"]["parity"] = {"validated": False,
                                 "stop": "probe a first-window: 1 cut decisions are unresolved"}
    with pytest.raises(readout.GateStop, match="parity gate: probe a first-window"):
        readout.verdict(record)


def test_a_stopped_timing_probe_refuses_the_verdict():
    record = _record()
    record["probe"] = {"stop": True, "projected_hours": 13.4, "horizon_hours": 12.0}
    with pytest.raises(readout.GateStop, match="13.4 h projected"):
        readout.verdict(record)


def test_the_report_refuses_before_scoring_anything_when_a_gate_failed(tmp_path, monkeypatch):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"decoded_audio_unit": "seconds",
                                 "decoded_audio_per_second": 50.0,
                                 "probe": {"wall_seconds": 60.0},
                                 "projected_seconds": 12960.0, "stop": False}),
                     encoding="utf-8")
    monkeypatch.setattr(readout, "PROBE_PATH", probe)
    monkeypatch.setattr(readout, "gates",
                        lambda: {"parity": {"validated": False, "stop": "a source failed"}})
    # the harness, the sections and the verdict are never reached: no comparison report from a
    # record whose decoder gate did not pass
    with pytest.raises(readout.GateStop, match="parity gate: a source failed"):
        readout.report()


def test_the_committed_parity_record_validates_and_names_its_resolved_cuts():
    parity = readout.gates()["parity"]
    assert parity["passed"] is True and parity["validated"] is True
    assert parity["plan"]["decoded"] == parity["plan"]["clips"]
    assert parity["probe"]["ambiguous_decisions"] == 0


# --- the committed page --------------------------------------------------------------------


def test_the_committed_doc_is_the_render_of_the_committed_record():
    record = json.loads(readout.REPORT_PATH.read_text(encoding="utf-8"))
    assert readout.DOC_PATH.read_text(encoding="utf-8") == render(record)


# --- the correct side's denominator, the carrier rules, the ratchet gain, exposure -----------


def test_a_site_without_a_human_side_leaves_both_denominators():
    from tadabur.truth_sites import TruthSite

    def site(name: str, heard: str, prescribed: str):
        return TruthSite(site_id=name, source="new_audit", assumes_competent_reciter=True,
                         audio_filename="a.wav", shard=0, start_sample=0, end_sample=1,
                         audio_sha256=None, surah_ayah="1:1", reference="بَ", reference_index=0,
                         mark="fatha", prescribed=prescribed, heard=heard, stratum="s",
                         stratum_population=3)

    sites = [site("m", "kasra", "fatha"), site("c", "fatha", "fatha"),
             site("p", "pending", "fatha"), site("u", "unclear", "fatha")]
    mistakes, correct, without_side = readout._sides(sites)
    assert [site.site_id for site in mistakes] == ["m"]
    assert [site.site_id for site in correct] == ["c"]
    assert without_side == 2


def test_a_wrong_carrier_or_two_marks_is_not_an_accurate_commitment():
    # the middle carrier is misheard (ط for ت): no slot, never an accurate haraka
    wrong_carrier = readout._mark_counts("بَطَبَ", "بَتَبَ")
    assert wrong_carrier["unaligned"] == 1
    assert wrong_carrier["committed"] == 2 and wrong_carrier["accurate"] == 2
    # fatha and kasra on one carrier: committed, but not the prescribed single mark
    two_marks = readout._mark_counts("بَتَِبَ", "بَتَبَ")
    assert two_marks["committed"] == 3 and two_marks["accurate"] == 2
    # a substituted mark is a commit, not an accurate one
    substituted = readout._mark_counts("بَتِبَ", "بَتَبَ")
    assert substituted["committed"] == 3 and substituted["accurate"] == 2
    # an omitted mark is an empty slot
    omitted = readout._mark_counts("بتَبَ", "بَتَبَ")
    assert omitted["haraka_slots"] == 3 and omitted["haraka_empty"] == 1
    assert omitted["committed"] == 2 and omitted["accurate"] == 2
    # the exact reference commits everything accurately
    exact = readout._mark_counts("بَتَبَ", "بَتَبَ")
    assert (exact["committed"], exact["accurate"], exact["unaligned"]) == (3, 3, 0)


def test_the_ratchet_gain_reads_the_final_grades_not_the_any_look_flags():
    from training.muraja_policy import CORRECT, FLAGGED

    # one bad look, kept correct: no final flag anywhere, and an any-look flag that was rescued
    graded = ({1: ["flagged"], 2: ["correct"]}, {1: {"quality": CORRECT},
                                                 2: {"quality": CORRECT}})
    assert readout._look_metrics(*graded)[1] == 1          # any-look: rescued, not displayed
    assert readout._final_flags(graded[1]) == 0
    # the single-decode approximation keeps it flagged: the real gain is 0 - 1 = -1
    approximation = ({1: ["flagged"]}, {1: {"quality": FLAGGED}})
    assert readout._final_flags(approximation[1]) == 1
    assert readout._final_flags(graded[1]) - readout._final_flags(approximation[1]) == -1
    assert readout._any_look_flags({1: graded}) == 1


def test_exposure_is_counted_at_recording_level_over_the_full_lineage():
    exposure = readout.exposure()
    assert exposure["inherited"]  # the shared-baseline uses every candidate inherits
    c95 = exposure["models"]["c95"]
    for corpus in ("synthetic_edit.source", "synthetic_edit.donor"):
        row = c95[corpus]
        # donor rows repeat one recording (114 rows, 77 recordings): recordings, not rows
        assert row["exposed"] <= row["recordings"]
        assert row["share"] == pytest.approx(100 * row["exposed"] / row["recordings"], abs=0.2)
    assert exposure["models"]["base"]["synthetic_edit.source"]["exposed"] is None
    assert exposure["lineage"]["r2_candidate_ema"][:2] == ["consonant1.training",
                                                           "consonant2.training"]


def test_ship_a_is_refused_when_an_arm_has_not_decoded_every_item(monkeypatch):
    from training import truth_baseline

    real = truth_baseline.read_cache

    def short(path):
        cache = real(path)
        if path.name == "c95.json":  # one arm's decodes and check sequences, one item short
            items = dict(cache["items"])
            items.pop(next(iter(items)))
            return {**cache, "items": items}
        return cache

    monkeypatch.setattr(truth_baseline, "read_cache", short)
    with pytest.raises(ValueError, match="never intersected away"):
        readout.ship_a_set()


# --- the report-only sets: one keying, one cluster, one arm set ------------------------------


def test_a_decode_line_keys_by_clip_and_span_and_its_transcript_is_the_committed_checks():
    from training import zipformer_sets

    line = {"audio_filename": "a.wav", "start_sample": 10, "end_sample": 20,
            "checks": [{"hop": "بَ"}, {"hop": "تَ"}]}
    assert zipformer_sets._item(line) == "a.wav@10:20"
    assert zipformer_sets._committed([line]) == {"a.wav@10:20": "بَتَ"}


def test_two_mushafs_do_not_share_a_bootstrap_cluster():
    from types import SimpleNamespace as NS

    from training import zipformer_sets

    def site(clip: str, stratum: str):
        return NS(audio_filename=clip, stratum=stratum, surah_ayah="1:1")

    clusters = zipformer_sets._clusters(
        [site("a.wav", "qud:anas_almiman_yt:ه→ح:agreed"),
         site("b.wav", "qud:anas_almiman_yt:ه→ح:agreed"),
         site("c.wav", "qud:abdul_hamid_ghraio_2026_yt:ه→ح:agreed")], mushaf=True)
    assert clusters["a.wav"] == clusters["b.wav"] != clusters["c.wav"]
    surahs = zipformer_sets._clusters(
        [site("a.wav", "tarteel_v1:tarteel_v1:geminate:agreed"),
         site("b.wav", "tarteel_v1:tarteel_v1:geminate:agreed")], mushaf=False)
    assert surahs["a.wav"] == surahs["b.wav"]

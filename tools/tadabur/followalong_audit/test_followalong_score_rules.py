"""The decision rules over attributed words (#311 PR2; prereg §6, amendment §5, §7, §8).

Committed before the PR2 code exists and **locked**: later turns may add tests but never weaken or
rewrite these.

Contract this file pins (``tadabur.followalong_audit.s5_score``)
-----------------------------------------------------------------
``evaluate_rules(attributions, *, oracle_pattern_words=0) -> dict`` over
:func:`attribute_listened` dicts (first verdicts only; ``item``, ``outcome``, ``bucket``,
``labels``, ``side``, ``teacher``, ``pi``, ``stratum`` and ``soft_or_target`` are read). It never
raises on no words. Keys:

* ``rule_1``: ``{"buckets": {name: {"instances", "clips", "triggered"}}, "f_ref": {"instances",
  "clips"}, "oracle_pattern_words", "oracle_pattern_triggered", "follow_along_first"}``. A bucket is
  counted by the word's first bucket over ``false_flag`` and ``miss`` words in every stratum, for
  ``F1, F1-align, M1r, M1, M1u, M-nograde`` (M-nograde only on the follow-along side); *triggered* is
  >= 3 instances in >= 3 distinct clips; the oracle pattern triggers at >= 10 words;
  ``follow_along_first`` is any trigger.
* ``rule_3``: ``{"label": "before any follow-along fix", "model_share": {"f_ref_follow_along",
  "f_ref_model"}, "unknown": {"words", "by_stratum"}, "extremes": {"unknown_follow_along",
  "unknown_model"}, "straddles_50", "owner_decides", "weighted_without_s4"}``. The share is the
  unweighted model-side fraction of attributed (``false_flag``/``miss``) words by ``side``, F-ref
  read on each side, unknown M-nograde words in neither numerator nor denominator; the extremes put
  every unknown on the follow-along / model side (F-ref follow-along); ``straddles_50`` is
  ``extremes.unknown_follow_along < 0.5 <= extremes.unknown_model`` and ``owner_decides`` equals it;
  ``weighted_without_s4`` is the pi-weighted (``1/pi``) model fraction without stratum S4, F-ref
  follow-along, unknown excluded. Shares are ``None`` with no known-side word.
* ``rule_4``: ``{"confirmed_misses", "threshold": 60, "decides", "status"}``: misses are the
  ``miss`` words (any stratum, unweighted); under 60 the status is ``"direction only"`` and
  ``decides`` False, else ``"decided"``.
* ``rule_5``: ``None`` with no F2/F3/M2/M3 word, else :func:`attribution.rule_five` over
  ``(bucket, teacher)`` pairs (a false flag's F2/F3 bucket; a miss's *last* label, M2/M3) plus
  ``"soft_or_target_share_of_f3"`` (F3 words with ``soft_or_target`` / all F3, ``None`` without F3)
  and ``"strict_mode_allowance"`` (that share >= 0.5).
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.attribution import FOLLOW_ALONG, MODEL, UNKNOWN
from tadabur.followalong_audit.s5_score import evaluate_rules

SIDES = {"F1": FOLLOW_ALONG, "F1-align": FOLLOW_ALONG, "F-ref": "f-ref", "F2": MODEL, "F3": MODEL,
         "M1r": FOLLOW_ALONG, "M1": FOLLOW_ALONG, "M1u": FOLLOW_ALONG, "M2": MODEL, "M3": MODEL}


def att(bucket, item="a#0", *, pi=0.1, stratum="S1", teacher="passed", soft=False, labels=None,
        side=None) -> dict:
    miss = bucket.startswith("M")
    return {"item": item, "id": f"{item}/{bucket}", "outcome": "miss" if miss else "false_flag",
            "bucket": bucket, "labels": labels or [bucket], "side": side or SIDES[bucket],
            "teacher": teacher, "pi": pi, "stratum": stratum, "soft_or_target": soft}


def words(bucket, count, **fields) -> list[dict]:
    return [att(bucket, item=f"{bucket}-{n}#0", **fields) for n in range(count)]


def nograde(*, graded: bool, stratum="S5", item="n#0") -> dict:
    return att("M-nograde", item, stratum=stratum, teacher="passed" if graded else "ungraded",
               labels=["M-nograde", "M3"], side=FOLLOW_ALONG if graded else UNKNOWN)


# --- rule 1 ---------------------------------------------------------------------------------------
def test_a_bucket_with_three_instances_in_three_clips_puts_follow_along_first():
    rules = evaluate_rules(words("F1", 3))["rule_1"]
    assert rules["buckets"]["F1"] == {"instances": 3, "clips": 3, "triggered": True}
    assert rules["follow_along_first"] is True


def test_three_instances_in_two_clips_do_not_trigger():
    rows = [att("M1r", "a#0"), att("M1r", "a#0"), att("M1r", "b#0")]
    rules = evaluate_rules(rows)["rule_1"]
    assert rules["buckets"]["M1r"] == {"instances": 3, "clips": 2, "triggered": False}
    assert rules["follow_along_first"] is False


def test_two_instances_in_two_clips_do_not_trigger():
    assert evaluate_rules(words("M1", 2))["rule_1"]["buckets"]["M1"]["triggered"] is False


def test_every_follow_along_bucket_is_counted_on_its_own_across_strata():
    rows = (words("F1", 3, stratum="S1") + words("F1-align", 3, stratum="S3")
            + words("M1u", 3, stratum="S4"))
    buckets = evaluate_rules(rows)["rule_1"]["buckets"]
    assert set(buckets) == {"F1", "F1-align", "M1r", "M1", "M1u", "M-nograde"}
    assert [buckets[name]["triggered"] for name in ("F1", "F1-align", "M1u")] == [True] * 3
    assert buckets["M1r"] == {"instances": 0, "clips": 0, "triggered": False}


def test_m_nograde_counts_only_on_the_follow_along_side():
    rows = [nograde(graded=True, item=f"g{n}#0") for n in range(3)] + [
        nograde(graded=False, item=f"u{n}#0") for n in range(5)]
    grade = evaluate_rules(rows)["rule_1"]["buckets"]["M-nograde"]
    assert grade == {"instances": 3, "clips": 3, "triggered": True}


def test_f_ref_is_reported_beside_the_buckets_and_never_triggers_follow_along_first():
    rules = evaluate_rules(words("F-ref", 4))["rule_1"]
    assert rules["f_ref"] == {"instances": 4, "clips": 4}
    assert rules["follow_along_first"] is False


def test_a_word_counts_only_under_its_first_bucket():
    rows = [att("M1r", f"x{n}#0", teacher="flagged", labels=["M1r", "M1", "M2"]) for n in range(3)]
    buckets = evaluate_rules(rows)["rule_1"]["buckets"]
    assert buckets["M1r"]["instances"] == 3 and buckets["M1"]["instances"] == 0


def test_a_catch_pass_or_excluded_word_is_not_an_instance():
    rows = [{**att("F1", f"x{n}#0"), "outcome": outcome, "bucket": None, "labels": [], "side": None}
            for n, outcome in enumerate(("correct_catch", "true_pass", "excluded"))]
    assert evaluate_rules(rows)["rule_1"]["buckets"]["F1"]["instances"] == 0


def test_the_oracle_pattern_triggers_at_ten_words():
    assert evaluate_rules([], oracle_pattern_words=9)["rule_1"]["follow_along_first"] is False
    rules = evaluate_rules([], oracle_pattern_words=10)["rule_1"]
    assert (rules["oracle_pattern_words"], rules["oracle_pattern_triggered"],
            rules["follow_along_first"]) == (10, True, True)


# --- rule 3 ---------------------------------------------------------------------------------------
def test_the_model_share_is_the_unweighted_model_fraction_of_the_known_side_words():
    rows = words("F1", 2) + words("M1r", 1) + words("F2", 3) + words("M3", 1)
    share = evaluate_rules(rows)["rule_3"]["model_share"]
    assert share == {"f_ref_follow_along": pytest.approx(4 / 7), "f_ref_model": pytest.approx(4 / 7)}


def test_f_ref_is_read_on_each_side_and_both_are_reported():
    rows = words("F1", 2) + words("F2", 2) + words("F-ref", 2)
    share = evaluate_rules(rows)["rule_3"]["model_share"]
    assert share["f_ref_follow_along"] == pytest.approx(2 / 6)
    assert share["f_ref_model"] == pytest.approx(4 / 6)


def test_unknown_words_are_outside_both_sides_counted_by_stratum_and_bounded_by_extremes():
    rows = (words("F1", 2) + words("M1r", 1) + words("F2", 3) + words("M3", 1)
            + [nograde(graded=False, stratum="S5", item="u1#0"),
               nograde(graded=False, stratum="S1", item="u2#0")])
    rules = evaluate_rules(rows)["rule_3"]
    assert rules["unknown"] == {"words": 2, "by_stratum": {"S1": 1, "S5": 1}}
    assert rules["model_share"]["f_ref_follow_along"] == pytest.approx(4 / 7)
    assert rules["extremes"] == {"unknown_follow_along": pytest.approx(4 / 9),
                                 "unknown_model": pytest.approx(6 / 9)}
    assert rules["straddles_50"] is True and rules["owner_decides"] is True


def test_extremes_on_one_side_of_half_leave_the_rule_decided():
    rows = words("F2", 5) + words("F1", 1) + [nograde(graded=False)]
    rules = evaluate_rules(rows)["rule_3"]
    assert rules["extremes"]["unknown_follow_along"] > 0.5
    assert rules["straddles_50"] is False and rules["owner_decides"] is False
    rows = words("F1", 5) + words("F2", 1) + [nograde(graded=False)]
    assert evaluate_rules(rows)["rule_3"]["straddles_50"] is False


def test_the_weighted_share_leaves_out_s4_and_weights_by_inverse_probability():
    rows = (words("F1", 2, pi=0.5) + words("F2", 2, pi=0.25, stratum="S3")
            + words("F2", 1, pi=0.01, stratum="S4"))
    rules = evaluate_rules(rows)["rule_3"]
    assert rules["weighted_without_s4"] == pytest.approx(8 / 12)
    assert rules["model_share"]["f_ref_follow_along"] == pytest.approx(3 / 5)


def test_the_share_is_labelled_as_computed_before_any_follow_along_fix():
    assert evaluate_rules(words("F2", 1))["rule_3"]["label"] == "before any follow-along fix"


def test_no_known_side_word_leaves_the_shares_none():
    for rows in ([], [nograde(graded=False)]):
        rules = evaluate_rules(rows)["rule_3"]
        assert rules["model_share"] == {"f_ref_follow_along": None, "f_ref_model": None}
        assert rules["weighted_without_s4"] is None
    assert evaluate_rules([])["rule_3"]["straddles_50"] is False


# --- rule 4 ---------------------------------------------------------------------------------------
def test_fewer_than_sixty_confirmed_misses_report_direction_only():
    rows = [att("M3", f"m{n}#0") for n in range(59)]
    assert evaluate_rules(rows)["rule_4"] == {"confirmed_misses": 59, "threshold": 60,
                                              "decides": False, "status": "direction only"}


def test_sixty_confirmed_misses_decide():
    rows = [att("M3", f"m{n}#0") for n in range(60)]
    assert evaluate_rules(rows)["rule_4"] == {"confirmed_misses": 60, "threshold": 60,
                                              "decides": True, "status": "decided"}


def test_misses_count_in_every_stratum_unweighted_and_catches_and_false_flags_do_not():
    rows = [att("M3", "a#0", stratum="S4", pi=0.001), att("M1r", "b#0", stratum="S5", pi=0.9),
            att("F2", "c#0"),
            {**att("M3", "d#0"), "outcome": "correct_catch", "bucket": None, "labels": [],
             "side": None}]
    assert evaluate_rules(rows)["rule_4"]["confirmed_misses"] == 2


def test_no_words_is_zero_misses_and_direction_only():
    assert evaluate_rules([])["rule_4"]["status"] == "direction only"


# --- rule 5 ---------------------------------------------------------------------------------------
def rule_five_rows() -> list[dict]:
    return ([att("F2", f"f2-{n}#0") for n in range(2)]
            + [att("F3", "f3-0#0", teacher="flagged", soft=True),
               att("F3", "f3-1#0", teacher="flagged", soft=True),
               att("F3", "f3-2#0", teacher="ungraded")]
            + [att("M1r", "m-0#0", teacher="flagged", labels=["M1r", "M2"]),
               att("M3", "m-1#0", teacher="passed", labels=["M3"])])


def test_rule_five_reads_f2_f3_and_the_last_label_of_a_miss_on_teacher_graded_words():
    rule = evaluate_rules(rule_five_rows())["rule_5"]
    assert (rule["learnable"], rule["beyond"], rule["ungraded"]) == (3, 3, 1)
    assert rule["direction"] == "beyond"
    assert rule["extremes"] == {"all_learnable": "learnable", "all_beyond": "beyond"}
    assert rule["extremes_agree"] is False


def test_soft_or_target_pairs_holding_half_of_f3_propose_a_strict_mode_allowance():
    rule = evaluate_rules(rule_five_rows())["rule_5"]
    assert rule["soft_or_target_share_of_f3"] == pytest.approx(2 / 3)
    assert rule["strict_mode_allowance"] is True
    exactly_half = [att("F3", "a#0", teacher="flagged", soft=True),
                    att("F3", "b#0", teacher="flagged")]
    assert evaluate_rules(exactly_half)["rule_5"]["strict_mode_allowance"] is True
    below = [att("F3", "a#0", teacher="flagged", soft=True)] + [
        att("F3", f"b{n}#0", teacher="flagged") for n in range(2)]
    assert evaluate_rules(below)["rule_5"]["strict_mode_allowance"] is False


def test_without_an_f3_word_there_is_no_soft_or_target_share_and_no_allowance():
    rule = evaluate_rules([att("F2", "a#0")])["rule_5"]
    assert rule["soft_or_target_share_of_f3"] is None and rule["strict_mode_allowance"] is False


def test_other_buckets_do_not_enter_rule_five_and_no_word_is_none():
    assert evaluate_rules(words("F1", 3) + words("M1", 2))["rule_5"] is None
    assert evaluate_rules([])["rule_5"] is None

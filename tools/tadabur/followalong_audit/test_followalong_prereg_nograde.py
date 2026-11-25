"""Amendments A2 and A3 to the attribution functions (#310, principal cross-check, owner-approved).

Committed before the change exists and **locked**: later turns may add tests but never weaken or
rewrite these. They extend ``tadabur/followalong_audit/attribution.py``; the earlier contract in
``test_followalong_prereg_attribution.py`` stands except where it says it was amended.

Contract this file pins
-----------------------
A2 -- a new miss label **M-nograde**: the App's kept grade is ungraded **and no check holds an App
status for the word** (no row at all, no checks, or every check's ``status`` is ``None``). Any
status at any check, even ``pending`` or a correct one, means the engine did see the word, so the
word is not M-nograde. Precedence ``M1r, M1, M1u, M-nograde, M2, M3``; ``miss_labels`` still ends
with M2 or M3, and a kept-ungraded word with a status below correct is M1u, never M-nograde.

* ``side("M-nograde", teacher_graded=...)`` is ``"follow-along"`` when the Teacher arm graded the
  word and ``"unknown"`` when it did not (``ValueError`` without ``teacher_graded``); every other
  bucket ignores ``teacher_graded``.
* ``model_share`` takes bucket strings as before, and the pair ``("M-nograde", teacher_graded)``
  for a no-grade word. A word on the ``"unknown"`` side is in neither the numerator nor the
  denominator (it is counted and reported by the caller); a share of only unknown words raises.

A3 -- rule 5 is computed on the words the Teacher arm graded, with the extremes reported.
``rule_five(words)`` takes ``(bucket, teacher_outcome)`` pairs, ``teacher_outcome`` being the
Teacher trace's ``kept_outcome`` (``passed``, ``flagged`` or ``ungraded``). Only F2, F3, M2 and M3
are read; the other buckets are ignored. F2 needs ``passed``, M2 ``flagged``, F3 ``flagged`` or
``ungraded``, M3 ``passed`` or ``ungraded`` (anything else is a ``ValueError``). It returns::

    {"learnable": F2 + M2,
     "beyond": F3 + M3 the Teacher graded,
     "ungraded": F3 + M3 the Teacher left ungraded,
     "direction": "beyond" if beyond >= learnable else "learnable"  (None when neither graded),
     "extremes": {"all_learnable": the direction with every ungraded word in learnable,
                  "all_beyond": the direction with every ungraded word in beyond},
     "extremes_agree": whether the two extremes name the same direction}

No F2, F3, M2 or M3 word at all raises ``ValueError``.
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.attribution import (
    miss_labels,
    model_share,
    rule_five,
    side,
)
from tadabur.followalong_audit.test_followalong_prereg_attribution import (
    FLAGGED,
    PASSED,
    PENDING,
    check,
    status,
    trace,
)


def labels(**evidence) -> list[str]:
    return miss_labels(**{"app": PASSED, "teacher": PASSED, **evidence})


# --- A2: M-nograde -----------------------------------------------------------------------------
@pytest.mark.parametrize("app", [
    None,
    trace(status("pending")),
    trace(status("pending", 0.9, withheld="minor")),
    trace(status("pending"), [check(cell=None), check(cell=None)]),
])
def test_an_ungraded_word_no_check_has_a_status_for_is_m_nograde(app):
    assert labels(app=app, teacher=PASSED) == ["M-nograde", "M3"]
    assert labels(app=app, teacher=FLAGGED) == ["M-nograde", "M2"]
    assert labels(app=app, teacher=None) == ["M-nograde", "M3"]


@pytest.mark.parametrize("cell", [status("pending"), status("correct"), status("correct", 0.9),
                                  status("pending", withheld="correct")])
def test_any_status_at_any_check_means_the_engine_saw_the_word_so_it_is_not_m_nograde(cell):
    app = trace(status("pending"), [check(cell=None), check(cell=cell)])
    assert "M-nograde" not in labels(app=app)


def test_a_status_below_correct_makes_an_ungraded_word_m1u_not_m_nograde():
    app = trace(status("pending"), [check(cell=status("minor", 0.4))])
    assert labels(app=app) == ["M1u", "M3"]
    assert labels(app=app, teacher=FLAGGED) == ["M1u", "M2"]


def test_a_graded_word_is_never_m_nograde_whatever_its_checks_hold():
    assert labels(app=PASSED) == ["M3"]
    assert labels(app=trace(status(), [check(cell=None)])) == ["M3"]


def test_m_nograde_ranks_after_m1u_and_before_the_teacher_labels():
    from tadabur.followalong_audit.attribution import BUCKETS
    order = [bucket for bucket in BUCKETS if bucket.startswith("M")]
    assert order == ["M1r", "M1", "M1u", "M-nograde", "M2", "M3"]


def test_the_missed_mistake_must_still_be_a_word_the_app_did_not_flag():
    with pytest.raises(ValueError):
        labels(app=FLAGGED)


def test_m_nograde_is_follow_along_when_the_teacher_arm_graded_the_word_and_unknown_when_not():
    assert side("M-nograde", teacher_graded=True) == "follow-along"
    assert side("M-nograde", teacher_graded=False) == "unknown"
    with pytest.raises(ValueError):
        side("M-nograde")


def test_the_other_buckets_ignore_teacher_graded():
    for bucket, expected in (("F2", "model"), ("M3", "model"), ("M1u", "follow-along"),
                             ("F1", "follow-along")):
        assert side(bucket, teacher_graded=False) == expected == side(bucket)


def test_the_model_share_counts_a_teacher_graded_no_grade_word_as_follow_along():
    assert model_share(["F2", ("M-nograde", True), "F3", "M1u"]) == pytest.approx(2 / 4)


def test_the_model_share_leaves_an_unknown_word_out_of_both_numerator_and_denominator():
    assert model_share(["F2", ("M-nograde", False), "F3", "M1u"]) == pytest.approx(2 / 3)
    assert model_share(["F2", ("M-nograde", False), ("M-nograde", False)]) == pytest.approx(1.0)


def test_the_model_share_of_only_unknown_words_is_refused():
    with pytest.raises(ValueError):
        model_share([("M-nograde", False)])


# --- A3: rule 5 on the Teacher-graded words ------------------------------------------------------
def test_rule_five_reads_only_teacher_graded_words_and_reports_both_extremes():
    result = rule_five([
        ("F2", "passed"), ("M2", "flagged"),
        ("F3", "flagged"), ("M3", "passed"),
        ("F3", "ungraded"), ("M3", "ungraded"), ("M3", "ungraded"),
        ("F1", "passed"), ("M1r", "passed"), ("F-ref", "ungraded"),
    ])
    assert result == {
        "learnable": 2, "beyond": 2, "ungraded": 3,
        "direction": "beyond",
        "extremes": {"all_learnable": "learnable", "all_beyond": "beyond"},
        "extremes_agree": False,
    }


def test_rule_five_names_the_direction_with_the_extremes_agreeing():
    result = rule_five([("F2", "passed")] * 5 + [("M3", "passed"), ("F3", "ungraded"),
                                                  ("M3", "ungraded")])
    assert result["learnable"] == 5 and result["beyond"] == 1 and result["ungraded"] == 2
    assert result["direction"] == "learnable"
    assert result["extremes"] == {"all_learnable": "learnable", "all_beyond": "learnable"}
    assert result["extremes_agree"] is True


def test_rule_five_ties_go_to_beyond_the_teacher_as_the_pre_registered_reading():
    result = rule_five([("F2", "passed"), ("F2", "passed"), ("M3", "passed"), ("F3", "flagged")])
    assert (result["learnable"], result["beyond"]) == (2, 2)
    assert result["direction"] == "beyond"


def test_rule_five_with_no_graded_word_has_no_direction_but_still_reports_the_extremes():
    result = rule_five([("F3", "ungraded"), ("M3", "ungraded")])
    assert result["direction"] is None
    assert result["extremes"] == {"all_learnable": "learnable", "all_beyond": "beyond"}
    assert result["extremes_agree"] is False


def test_rule_five_extremes_can_disagree_with_the_graded_direction():
    result = rule_five([("F2", "passed"), ("F3", "flagged"), ("F3", "flagged"),
                        ("M3", "ungraded"), ("M3", "ungraded"), ("M3", "ungraded")])
    assert result["direction"] == "beyond"
    assert result["extremes"] == {"all_learnable": "learnable", "all_beyond": "beyond"}


def test_rule_five_without_a_teacher_side_word_is_refused():
    for words in ([], [("F1", "passed"), ("M1", "passed")]):
        with pytest.raises(ValueError):
            rule_five(words)


@pytest.mark.parametrize("bucket, outcome", [
    ("F2", "ungraded"), ("F2", "flagged"), ("M2", "passed"), ("M2", "ungraded"),
    ("F3", "passed"), ("M3", "flagged"),
])
def test_rule_five_refuses_a_teacher_outcome_the_bucket_cannot_have(bucket, outcome):
    with pytest.raises(ValueError):
        rule_five([(bucket, outcome)])


def test_teacher_ungraded_false_flags_and_misses_stay_on_the_model_side_for_rule_two():
    assert side("F3") == "model" and side("M3") == "model"
    assert model_share(["F3", "M3", "F1"]) == pytest.approx(2 / 3)


def test_the_teacher_ungraded_pending_row_is_ungraded_for_rule_five_inputs():
    from tadabur.followalong_audit.attribution import kept_outcome
    assert kept_outcome(PENDING) == "ungraded" and kept_outcome(None) == "ungraded"

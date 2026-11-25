"""The attribution functions S5 will apply to the audit's verdicts (#310, pre-registration section 5).

Committed before ``tadabur/followalong_audit/attribution.py`` exists and **locked**: later turns
may add tests but never weaken or rewrite these. They pin the definitions of
``docs/followalong-audit-preregistration.md`` (architect decisions D3, D4, D5, D8, D9) with one
fixture per bucket.

Contract this file pins
-----------------------
Evidence is the ``training.muraja_policy.word_traces`` row of one word under one scoring
(``None`` when the arm kept no row for it)::

    {"quality", "score", "withheld", "errors",
     "checks": [{"hop", "overlap", "flush", "status": {"word", "quality", "score", "errors",
                                                        "withheld"?} | None}],
     "setting_checks": [check index, ...]}

* ``effective_quality(status)`` is ``withheld`` when present, else ``quality`` (D5);
  ``below_correct(status)`` is an effective quality outside {correct, pending}; ``None`` is not
  below correct.
* ``kept_outcome(trace, scoring=STRICT)`` is ``"ungraded"`` for no row, a ``pending`` row (with or
  without ``withheld``), else ``"flagged"`` when ``displayed(quality, scoring)`` is not correct,
  else ``"passed"``.
* ``false_flag_bucket(*, app, oracle, teacher, reference, suspect, scoring=STRICT)`` for a word
  the App flagged and the owner heard correct. Precedence F1, F1-align, F-ref, F2, F3:

  - F1: the oracle's kept outcome is ``flagged``;
  - F1-align: the oracle's is ``passed`` and a *setting check* of the App trace has
    ``hop + overlap`` containing, as an exact substring, ``previous + word`` or
    ``word + next`` (``reference`` is ``{"previous": str | None, "word": str, "next": str | None}``,
    the oracle's reference phonemes, joined without a separator); a clip with no neighbour at all
    needs ``word`` alone, with ``len(word) >= 4`` (D4). An ungraded oracle is neither F1 nor
    F1-align;
  - F-ref: ``suspect`` is true (computed by ``reference_suspects``);
  - F2: the Teacher arm's kept outcome is ``passed``; F3: it is ``flagged`` or ``ungraded`` (D8).

  It raises ``ValueError`` when the App's kept outcome is not ``flagged``.
* ``miss_labels(*, app, teacher, scoring=STRICT)`` for a word the owner heard as a mistake that the
  App did not flag: every label the word meets, in precedence order M1r, M1, M1u, M2, M3 (D9);
  the first is the primary. M1r: the App's kept outcome is ``passed`` and some check's status is
  below correct. M1: ``passed`` and some *setting* check's status has errors or a score below 1.
  M1u: ``ungraded`` and some check's status is below correct. M2: the Teacher's kept outcome is
  ``flagged``; M3: it is not (``passed`` or ``ungraded``, D8). It raises ``ValueError`` when the
  App flagged the word.
* ``side(bucket, *, f_ref="follow-along")`` is ``"follow-along"`` for F1, F1-align, M1r, M1, M1u,
  ``"model"`` for F2, F3, M2, M3, and ``f_ref`` for F-ref; ``model_share(buckets, *, f_ref=...)``
  is the model-side fraction (``ValueError`` on no buckets).
* ``error_key(errors)`` is the sorted tuple of ``(group_index, kind.type, kind.expected,
  kind.heard or None)``; ``reference_suspects(observations, minimum=3)`` is the set of
  ``(surah, ayah, word, error_key)`` seen in at least ``minimum`` distinct ``group_key``s
  (observation: ``{"group_key", "surah", "ayah", "word", "errors"}``); a word with no errors is
  never a suspect (D3).
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.attribution import (
    below_correct,
    effective_quality,
    error_key,
    false_flag_bucket,
    kept_outcome,
    miss_labels,
    model_share,
    reference_suspects,
    side,
)
from training.muraja_policy import ALLOWANCES_ON, SHIPPED_DEFAULT, STRICT, Scoring

WORD = "بِسمِ"
BEFORE = "مَلِكِ"
AFTER = "يَومِ"
SHORT = "مِن"
MISMATCH = {"group_index": 0, "kind": {"type": "mismatch", "expected": "س", "heard": "ص"}}


def status(quality: str = "correct", score: float = 1.0, errors=(), withheld=None) -> dict:
    cell = {"word": 3, "quality": quality, "score": score, "errors": list(errors)}
    return cell if withheld is None else {**cell, "withheld": withheld}


def check(text: str = "", cell: dict | None = None, overlap: str = "") -> dict:
    return {"hop": text, "overlap": overlap, "flush": False, "status": cell}


def trace(kept: dict, checks: list[dict] | None = None, setting: tuple[int, ...] = ()) -> dict:
    return {"quality": kept["quality"], "score": kept["score"], "withheld": kept.get("withheld"),
            "errors": kept["errors"], "checks": checks or [], "setting_checks": list(setting)}


FLAGGED = trace(status("minor", 0.5))
PASSED = trace(status())
PENDING = trace(status("pending", 0.9, withheld="minor"))
REFERENCE = {"previous": BEFORE, "word": WORD, "next": AFTER}


def app_hearing(text: str, *, setting: bool = True, overlap: str = "") -> dict:
    """An App-flagged trace whose one check decoded ``text`` (the setting check unless not)."""
    return trace(status("minor", 0.5), [check(text, status("minor", 0.5), overlap)],
                 (0,) if setting else ())


def f_bucket(**evidence) -> str:
    base = {"app": FLAGGED, "oracle": PASSED, "teacher": PASSED, "reference": REFERENCE,
            "suspect": False}
    return false_flag_bucket(**{**base, **evidence})


def m_labels(**evidence) -> list[str]:
    return miss_labels(**{"app": PASSED, "teacher": PASSED, **evidence})


# --- grades ----------------------------------------------------------------------------------------
def test_a_checks_effective_quality_is_its_withheld_value_when_it_has_one():
    assert effective_quality(status("pending", withheld="minor")) == "minor"
    assert effective_quality(status("pending")) == "pending"
    assert effective_quality(status("correct")) == "correct"


@pytest.mark.parametrize("cell, expected", [
    (status("correct"), False),
    (status("pending"), False),
    (status("pending", withheld="minor"), True),
    (status("pending", withheld="correct"), False),
    (status("minor"), True),
    (status("skipped"), True),
    (status("tashkeelError"), True),
    (None, False),
])
def test_below_correct_is_an_effective_quality_outside_correct_and_pending(cell, expected):
    assert below_correct(cell) is expected


@pytest.mark.parametrize("row, outcome", [
    (None, "ungraded"),
    (trace(status("pending")), "ungraded"),
    (PENDING, "ungraded"),
    (trace(status("correct")), "passed"),
    (trace(status("minor")), "flagged"),
    (trace(status("skipped")), "flagged"),
    (trace(status("tashkeelError")), "flagged"),
])
def test_the_kept_outcome_reads_the_displayed_grade_and_a_withheld_pending_word_is_ungraded(
        row, outcome):
    assert kept_outcome(row) == outcome


def test_the_kept_outcome_is_the_grade_as_the_scoring_shows_it():
    tashkeel = trace(status("tashkeelError"))
    assert kept_outcome(tashkeel, STRICT) == "flagged"
    assert kept_outcome(tashkeel, SHIPPED_DEFAULT) == "flagged"
    assert kept_outcome(tashkeel, ALLOWANCES_ON) == "passed"
    assert kept_outcome(trace(status("minor")), Scoring("lenient", mode="lenient")) == "passed"


# --- false flags: F1, F1-align, F-ref, F2, F3 ------------------------------------------------------
def test_f1_the_oracle_grades_the_word_below_correct():
    assert f_bucket(oracle=FLAGGED) == "F1"


def test_f1_wins_over_every_later_bucket():
    aligned = app_hearing(BEFORE + WORD)
    assert f_bucket(oracle=FLAGGED, app=aligned, suspect=True, teacher=PASSED) == "F1"


@pytest.mark.parametrize("text", [BEFORE + WORD, WORD + AFTER, "خ" + BEFORE + WORD + AFTER + "خ"])
def test_f1_align_the_oracle_passes_and_a_setting_check_holds_the_word_with_a_neighbour(text):
    assert f_bucket(app=app_hearing(text)) == "F1-align"


def test_f1_align_reads_the_hop_and_the_overlap_together():
    assert f_bucket(app=app_hearing(BEFORE, overlap=WORD)) == "F1-align"


def test_f1_align_needs_a_neighbour_the_word_alone_is_only_the_sensitivity_row():
    assert f_bucket(app=app_hearing(WORD)) == "F2"
    assert f_bucket(app=app_hearing(WORD), teacher=FLAGGED) == "F3"


def test_f1_align_needs_the_neighbour_to_touch_the_word_as_the_oracle_text_joins_them():
    assert f_bucket(app=app_hearing(BEFORE + "خ" + WORD)) == "F2"
    assert f_bucket(app=app_hearing(WORD + "خ" + AFTER)) == "F2"


def test_f1_align_reads_only_the_checks_that_set_the_kept_grade():
    assert f_bucket(app=app_hearing(BEFORE + WORD, setting=False)) == "F2"


def test_f1_align_is_not_met_when_the_oracle_flags_or_left_the_word_ungraded():
    assert f_bucket(app=app_hearing(BEFORE + WORD), oracle=FLAGGED) == "F1"
    assert f_bucket(app=app_hearing(BEFORE + WORD), oracle=None) == "F2"
    assert f_bucket(app=app_hearing(BEFORE + WORD), oracle=PENDING) == "F2"


def test_a_one_word_clip_matches_on_the_word_alone_when_it_is_at_least_four_characters():
    alone = {"previous": None, "word": WORD, "next": None}
    assert len(WORD) >= 4
    assert f_bucket(app=app_hearing("خ" + WORD + "خ"), reference=alone) == "F1-align"


def test_a_one_word_clip_does_not_match_a_word_shorter_than_four_characters():
    alone = {"previous": None, "word": SHORT, "next": None}
    assert len(SHORT) < 4
    assert f_bucket(app=app_hearing("خ" + SHORT + "خ"), reference=alone) == "F2"


def test_the_first_word_of_a_clip_matches_with_its_next_neighbour_only():
    first = {"previous": None, "word": WORD, "next": AFTER}
    assert f_bucket(app=app_hearing(WORD + AFTER), reference=first) == "F1-align"
    assert f_bucket(app=app_hearing(WORD), reference=first) == "F2"


def test_f_ref_is_a_reference_suspect_the_oracle_and_the_decode_did_not_explain():
    assert f_bucket(suspect=True) == "F-ref"


def test_f_ref_ranks_after_f1_align_and_before_the_teacher_buckets():
    assert f_bucket(app=app_hearing(BEFORE + WORD), suspect=True) == "F1-align"
    assert f_bucket(suspect=True, teacher=FLAGGED) == "F-ref"


def test_f2_the_teacher_arm_passes_the_word():
    assert f_bucket(teacher=PASSED) == "F2"


def test_f3_the_teacher_arm_flags_the_word():
    assert f_bucket(teacher=FLAGGED) == "F3"


@pytest.mark.parametrize("teacher", [None, trace(status("pending")), PENDING])
def test_f3_a_word_the_teacher_arm_left_ungraded_is_not_learnable(teacher):
    assert f_bucket(teacher=teacher) == "F3"


def test_a_false_flag_must_be_a_word_the_app_flagged():
    for app in (PASSED, None, PENDING):
        with pytest.raises(ValueError):
            f_bucket(app=app)


def test_the_false_flag_buckets_read_the_kept_grade_as_the_scoring_shows_it():
    tashkeel = trace(status("tashkeelError"))
    assert f_bucket(app=tashkeel, scoring=STRICT) == "F2"
    with pytest.raises(ValueError):
        f_bucket(app=tashkeel, scoring=ALLOWANCES_ON)


# --- misses: M1r, M1, M1u, M2, M3 ------------------------------------------------------------------
def test_m1r_some_check_graded_the_word_below_correct_and_the_kept_grade_is_correct():
    app = trace(status(), [check(cell=status("minor", 0.4)), check(cell=status())], setting=(1,))
    assert m_labels(app=app) == ["M1r", "M3"]


def test_m1r_counts_a_check_whose_below_correct_grade_is_withheld():
    app = trace(status(), [check(cell=status("pending", 0.9, withheld="skipped"))])
    assert m_labels(app=app)[0] == "M1r"


def test_m1_a_deviation_was_decoded_and_forgiven_by_the_setting_check():
    forgiven = status("correct", 0.9, [MISMATCH])
    app = trace(forgiven, [check(cell=forgiven)], setting=(0,))
    assert m_labels(app=app) == ["M1", "M3"]


def test_m1_a_setting_check_score_below_one_is_enough():
    forgiven = status("correct", 0.9)
    assert m_labels(app=trace(forgiven, [check(cell=forgiven)], setting=(0,))) == ["M1", "M3"]


def test_m1_reads_only_the_setting_checks():
    forgiven = status("correct", 0.9, [MISMATCH])
    app = trace(status(), [check(cell=forgiven)], setting=())
    assert m_labels(app=app) == ["M3"]


def test_a_word_that_meets_m1r_and_m1_records_both_with_m1r_first():
    forgiven = status("correct", 0.9, [MISMATCH])
    app = trace(forgiven, [check(cell=status("minor", 0.3)), check(cell=forgiven)], setting=(1,))
    assert m_labels(app=app) == ["M1r", "M1", "M3"]


def test_m1u_the_app_left_the_word_ungraded_though_a_check_graded_it_below_correct():
    app = trace(status("pending", 0.9, withheld="minor"),
                [check(cell=status("pending", 0.9, withheld="minor"))])
    assert m_labels(app=app, teacher=FLAGGED) == ["M1u", "M2"]


def test_m1u_counts_a_check_that_graded_below_correct_before_the_word_went_pending():
    app = trace(status("pending"), [check(cell=status("minor", 0.4)), check(cell=None)])
    assert m_labels(app=app) == ["M1u", "M3"]


def test_m1u_needs_a_check_below_correct_an_ungraded_word_with_a_status_but_none_is_a_plain_miss():
    # Amended by A2 (2026-11-24): an App word with no row at all is M-nograde, not a plain miss;
    # see test_followalong_prereg_nograde.py. A word with a status at some check stays M2/M3.
    app = trace(status("pending"), [check(cell=status("pending")), check(cell=None)])
    assert m_labels(app=app) == ["M3"]


def test_the_checks_of_an_ungraded_word_never_make_it_m1r_or_m1_and_a_passed_one_never_m1u():
    ungraded = trace(status("pending"), [check(cell=status("correct", 0.9, [MISMATCH]))],
                     setting=(0,))
    assert m_labels(app=ungraded) == ["M3"]
    graded = trace(status(), [check(cell=status("minor", 0.4))])
    assert "M1u" not in m_labels(app=graded)


def test_m2_the_teacher_arm_flags_the_word_the_app_passed():
    assert m_labels(teacher=FLAGGED) == ["M2"]


def test_m3_neither_arm_flags_the_word():
    assert m_labels(teacher=PASSED) == ["M3"]


@pytest.mark.parametrize("teacher", [None, trace(status("pending")), PENDING])
def test_m3_a_word_the_teacher_arm_left_ungraded_is_not_learnable(teacher):
    assert m_labels(teacher=teacher) == ["M3"]


def test_the_primary_miss_label_is_the_first_in_m1r_m1_m1u_m2_m3_order():
    app = trace(status(), [check(cell=status("minor", 0.3))])
    assert m_labels(app=app, teacher=FLAGGED) == ["M1r", "M2"]


def test_a_miss_must_be_a_word_the_app_did_not_flag():
    with pytest.raises(ValueError):
        m_labels(app=FLAGGED)


def test_the_miss_labels_read_the_kept_grade_as_the_scoring_shows_it():
    shown_correct = trace(status("tashkeelError", 0.8), [check(cell=status("tashkeelError", 0.8))])
    assert m_labels(app=shown_correct, scoring=ALLOWANCES_ON)[0] == "M1r"
    with pytest.raises(ValueError):
        m_labels(app=shown_correct, scoring=STRICT)


# --- sides and the model share ---------------------------------------------------------------------
@pytest.mark.parametrize("bucket", ["F1", "F1-align", "M1r", "M1", "M1u"])
def test_follow_along_buckets_are_the_follow_along_side(bucket):
    assert side(bucket) == "follow-along"


@pytest.mark.parametrize("bucket", ["F2", "F3", "M2", "M3"])
def test_teacher_arm_buckets_are_the_model_side(bucket):
    assert side(bucket) == "model"


def test_f_ref_is_a_suspect_that_rule_two_places_on_either_side():
    assert side("F-ref") == "follow-along"
    assert side("F-ref", f_ref="follow-along") == "follow-along"
    assert side("F-ref", f_ref="model") == "model"


def test_an_unknown_bucket_is_refused():
    with pytest.raises(ValueError):
        side("F4")


def test_the_model_share_is_the_model_side_fraction_with_f_ref_on_either_side():
    buckets = ["F1", "F1-align", "F-ref", "F2", "F3", "F3"]
    assert model_share(buckets) == pytest.approx(3 / 6)
    assert model_share(buckets, f_ref="follow-along") == pytest.approx(3 / 6)
    assert model_share(buckets, f_ref="model") == pytest.approx(4 / 6)


def test_the_model_share_of_no_words_is_refused():
    with pytest.raises(ValueError):
        model_share([])


# --- F-ref: the same deviation at the same word in three tranche groups -----------------------------
def kind(type_: str, expected: str | None, heard: str | None = None) -> dict:
    return {"type": type_, "expected": expected, "heard": heard}


def test_the_error_key_is_the_sorted_group_kind_expected_heard_tuples():
    errors = [{"group_index": 2, "kind": kind("tashkeel", "َ")},
              {"group_index": 0, "kind": kind("mismatch", "س", "ص")}]
    assert error_key(errors) == ((0, "mismatch", "س", "ص"), (2, "tashkeel", "َ", None))


def test_the_error_key_ignores_the_order_the_engine_listed_the_errors_in():
    one, two = ({"group_index": 0, "kind": kind("mismatch", "س", "ص")},
                {"group_index": 1, "kind": kind("tashkeel", "ِ")})
    assert error_key([one, two]) == error_key([two, one])


def observation(group: str, errors, *, word: int = 4, ayah: int = 7) -> dict:
    return {"group_key": group, "surah": 2, "ayah": ayah, "word": word, "errors": errors}


def test_the_same_deviation_at_the_same_word_in_three_groups_is_a_reference_suspect():
    errors = [{"group_index": 0, "kind": kind("mismatch", "س", "ص")}]
    suspects = reference_suspects([observation(f"g{n}", errors) for n in range(3)])
    assert suspects == {(2, 7, 4, error_key(errors))}


def test_two_groups_are_not_enough_however_many_clips_they_hold():
    errors = [{"group_index": 0, "kind": kind("mismatch", "س", "ص")}]
    rows = [observation("a", errors)] * 5 + [observation("b", errors)] * 5
    assert reference_suspects(rows) == set()


def test_the_threshold_is_a_parameter_with_the_pre_registered_default_of_three():
    errors = [{"group_index": 0, "kind": kind("mismatch", "س", "ص")}]
    rows = [observation("a", errors), observation("b", errors)]
    assert reference_suspects(rows, minimum=2) == {(2, 7, 4, error_key(errors))}
    assert reference_suspects(rows) == set()


def test_a_different_deviation_or_a_different_word_is_a_different_suspect():
    seen = [{"group_index": 0, "kind": kind("mismatch", "س", "ص")}]
    other = [{"group_index": 0, "kind": kind("mismatch", "س", "ث")}]
    rows = [observation("a", seen), observation("b", seen), observation("c", other)]
    assert reference_suspects(rows) == set()
    rows = [observation("a", seen), observation("b", seen, word=5), observation("c", seen, ayah=8)]
    assert reference_suspects(rows) == set()


def test_a_flagged_word_with_no_recorded_error_is_never_a_reference_suspect():
    assert reference_suspects([observation(f"g{n}", []) for n in range(5)]) == set()

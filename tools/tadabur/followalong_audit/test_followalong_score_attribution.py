"""One listened word through the locked attribution buckets (#311 PR2; prereg §5, amendment §7, §9).

Committed before the PR2 code exists and **locked**: later turns may add tests but never weaken or
rewrite these. The bucket functions themselves are pinned by ``test_followalong_prereg_attribution.py``
(the S4 fixtures); these pin how the scorer routes a verdict to them.

Contract this file pins (``tadabur.followalong_audit.s5_score``)
-----------------------------------------------------------------
An **arms row** (``s5_draw/arms.jsonl.gz``, one per drawn word) is::

    {"item", "word",
     "app": {"strict": trace | None, "shipped_default": trace | None},      # full word_traces row
     "causes": {"strict": [..], "shipped_default": [..]},                   # the App's flag causes
     "oracle" | "teacher" | "student": {"strict": row | None, "shipped_default": row | None},
     "reference": {"previous", "word", "next"} | None}     # oracle_reference(...); None: no F1-align

``row`` may be compact (``quality``, ``score``, ``withheld``, ``errors``); ``None`` is no row.

``attribute_listened(key_row, arms_row, verdict_row, *, suspects, scoring=STRICT) -> dict`` reads
the arm rows of ``scoring.name`` and returns (at least) ``id, item, word, stratum, cell, pi``
(the key's ``inclusion_probability``), ``verdict``, ``app`` and ``teacher`` (their kept outcomes),
``outcome``, ``bucket``, ``labels``, ``side`` and ``soft_or_target``:

* ``outcome``: ``false_flag`` (verdict ``correct``, App flagged), ``true_pass`` (``correct``, not
  flagged — passed or ungraded), ``correct_catch`` (``mistake``, App flagged: never a miss),
  ``miss`` (``mistake``, App not flagged), ``excluded`` (any other verdict);
* ``labels``: ``[F bucket]`` for a false flag (:func:`attribution.false_flag_bucket`),
  :func:`attribution.miss_labels` for a miss, else ``[]``; ``bucket`` is ``labels[0]`` or ``None``;
* ``side``: ``attribution.side`` of ``bucket`` (``"f-ref"`` for F-ref, M-nograde by whether the
  Teacher graded the word), ``None`` when there is no bucket;
* ``soft_or_target``: a false flag whose App causes (of ``scoring``) include ``soft_pair`` or
  ``target_pair``.

``is_reference_suspect(key_row, arms_row, suspects) -> bool``: the key's ``(surah, ayah, word)``
with :func:`attribution.error_key` of the App **strict** trace's errors is in ``suspects``
(:func:`attribution.reference_suspects`); a word with no errors never is.
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.attribution import FOLLOW_ALONG, MODEL, UNKNOWN, reference_suspects
from tadabur.followalong_audit.s5_score import attribute_listened, is_reference_suspect
from training.muraja_policy import SHIPPED_DEFAULT, STRICT

WORD, BEFORE, AFTER = "بِسمِ", "مَلِكِ", "يَومِ"
MISMATCH = {"group_index": 0, "kind": {"type": "mismatch", "expected": "س", "heard": "ص"}}
REFERENCE = {"previous": BEFORE, "word": WORD, "next": AFTER}


def status(quality="correct", score=1.0, errors=(), withheld=None) -> dict:
    cell = {"word": 3, "quality": quality, "score": score, "errors": list(errors)}
    return cell if withheld is None else {**cell, "withheld": withheld}


def trace(quality="correct", score=1.0, errors=(), checks=None, setting=(), withheld=None) -> dict:
    return {"quality": quality, "score": score, "withheld": withheld, "errors": list(errors),
            "checks": checks or [], "setting_checks": list(setting)}


def compact(quality) -> dict:
    return {"quality": quality, "score": 1.0, "withheld": None, "errors": []}


FLAGGED = trace("minor", 0.5, [MISMATCH])
PASSED = trace()


def hearing(text: str) -> dict:
    """An App-flagged trace whose setting check decoded ``text``."""
    return trace("minor", 0.5, [MISMATCH], checks=[
        {"hop": text, "overlap": "", "flush": False, "status": status("minor", 0.5, [MISMATCH])}],
        setting=(0,))


def key_row(**fields) -> dict:
    return {"id": "0123456789abcdef", "item": "1_2_123.flac#0", "word": 3, "surah_ayah": "1:2",
            "stratum": "S1", "cell": "other_consonant:no", "inclusion_probability": 0.04, **fields}


MISSING = object()


def arms(app=FLAGGED, *, oracle=MISSING, teacher=MISSING, reference=REFERENCE,
         causes=("other_consonant",), balanced=None) -> dict:
    def arm(row):
        return {"strict": row, "shipped_default": row}

    return {"item": "1_2_123.flac#0", "word": 3,
            "app": {"strict": app, "shipped_default": app if balanced is None else balanced},
            "causes": {"strict": list(causes), "shipped_default": list(causes)},
            "oracle": arm(compact("correct") if oracle is MISSING else oracle),
            "teacher": arm(compact("correct") if teacher is MISSING else teacher),
            "student": arm(compact("correct")), "reference": reference}


def verdict(kind="correct") -> dict:
    return {"id": "0123456789abcdef", "verdict": kind, "kinds": ["letter"] if kind == "mistake" else [],
            "sub_reason": None, "whole_played": False, "note": ""}


def run(arms_row, kind="correct", *, suspects=frozenset(), scoring=STRICT, **key):
    return attribute_listened(key_row(**key), arms_row, verdict(kind), suspects=suspects,
                              scoring=scoring)


# --- routing --------------------------------------------------------------------------------------
def test_a_correct_word_the_app_flagged_is_a_false_flag_with_one_bucket():
    result = run(arms(oracle=compact("minor")))
    assert (result["outcome"], result["bucket"], result["labels"]) == ("false_flag", "F1", ["F1"])
    assert (result["app"], result["id"], result["pi"]) == ("flagged", "0123456789abcdef", 0.04)
    assert (result["stratum"], result["cell"], result["item"], result["word"]) == (
        "S1", "other_consonant:no", "1_2_123.flac#0", 3)


@pytest.mark.parametrize("app", [PASSED, None, trace("pending", 0.9, withheld="minor")])
def test_a_correct_word_the_app_did_not_flag_is_a_true_pass_whether_passed_or_ungraded(app):
    result = run(arms(app))
    assert (result["outcome"], result["bucket"], result["labels"], result["side"]) == (
        "true_pass", None, [], None)


def test_a_mistake_the_app_flagged_is_a_correct_catch_never_a_miss():
    result = run(arms(FLAGGED), "mistake")
    assert (result["outcome"], result["bucket"], result["labels"]) == ("correct_catch", None, [])


@pytest.mark.parametrize("kind", ["not sure", "different text", "unusable audio"])
def test_the_other_verdicts_are_excluded_with_no_bucket(kind):
    for app in (FLAGGED, PASSED, None):
        result = run(arms(app), kind)
        assert (result["outcome"], result["bucket"], result["labels"]) == ("excluded", None, [])


# --- every false-flag bucket ----------------------------------------------------------------------
def test_f1_align_the_oracle_passes_and_a_setting_check_holds_the_word_with_a_neighbour():
    assert run(arms(hearing(BEFORE + WORD)))["bucket"] == "F1-align"


def test_f_ref_is_a_reference_suspect_the_app_deviation_repeats_in_three_groups():
    observations = [{"group_key": group, "surah": 1, "ayah": 2, "word": 3, "errors": [MISMATCH]}
                    for group in ("g1", "g2", "g3")]
    suspects = reference_suspects(observations)
    result = run(arms(FLAGGED), suspects=suspects)
    assert (result["bucket"], result["side"]) == ("F-ref", "f-ref")
    assert run(arms(FLAGGED), suspects=reference_suspects(observations[:2]))["bucket"] == "F2"


def test_f2_and_f3_follow_the_teachers_kept_outcome_and_ungraded_is_f3():
    assert run(arms(FLAGGED, teacher=compact("correct")))["bucket"] == "F2"
    assert run(arms(FLAGGED, teacher=compact("minor")))["bucket"] == "F3"
    assert run(arms(FLAGGED, teacher=None))["bucket"] == "F3"
    assert run(arms(FLAGGED, teacher=compact("pending")))["bucket"] == "F3"
    assert (run(arms(FLAGGED, teacher=compact("correct")))["side"],
            run(arms(FLAGGED, teacher=compact("minor")))["side"]) == (MODEL, MODEL)


def test_the_follow_along_false_flags_sit_on_the_follow_along_side():
    assert run(arms(hearing(BEFORE + WORD)))["side"] == FOLLOW_ALONG
    assert run(arms(FLAGGED, oracle=compact("minor")))["side"] == FOLLOW_ALONG


def test_an_engine_word_with_no_mapped_canonical_word_is_never_f1_align():
    text = BEFORE + WORD
    assert run(arms(hearing(text), reference=REFERENCE))["bucket"] == "F1-align"
    assert run(arms(hearing(text), reference=None))["bucket"] == "F2"
    assert run(arms(hearing(text), reference=None, teacher=compact("minor")))["bucket"] == "F3"


def test_an_ungraded_oracle_is_neither_f1_nor_f1_align():
    assert run(arms(hearing(BEFORE + WORD)))["bucket"] == "F1-align"
    ungraded = arms(hearing(BEFORE + WORD))
    ungraded["oracle"] = {"strict": None, "shipped_default": None}
    assert run(ungraded)["bucket"] == "F2"


# --- every miss label -----------------------------------------------------------------------------
def test_a_miss_is_labelled_by_miss_labels_and_its_first_label_is_its_bucket():
    withheld = trace("pending", 0.9, checks=[
        {"hop": "", "overlap": "", "flush": False, "status": status("minor", 0.5, [MISMATCH])}],
        withheld="minor")
    result = run(arms(withheld, teacher=compact("minor")), "mistake")
    assert (result["outcome"], result["labels"], result["bucket"]) == ("miss", ["M1u", "M2"], "M1u")
    assert result["side"] == FOLLOW_ALONG


def test_m1r_and_m1_the_app_passed_what_a_check_graded_or_forgave():
    graded = trace("correct", 1.0, checks=[
        {"hop": "", "overlap": "", "flush": False, "status": status("minor", 0.5, [MISMATCH])}])
    assert run(arms(graded), "mistake")["labels"] == ["M1r", "M3"]
    forgiven = trace("correct", 1.0, checks=[
        {"hop": "", "overlap": "", "flush": False, "status": status("correct", 0.9)}], setting=(0,))
    assert run(arms(forgiven), "mistake")["labels"] == ["M1", "M3"]


def test_m2_and_m3_are_the_teacher_arms_flag_when_no_app_label_applies():
    seen = trace("correct", 1.0, checks=[
        {"hop": "", "overlap": "", "flush": False, "status": status("correct")}])
    assert run(arms(seen, teacher=compact("minor")), "mistake")["labels"] == ["M2"]
    assert run(arms(seen, teacher=compact("correct")), "mistake")["labels"] == ["M3"]
    assert run(arms(seen, teacher=None), "mistake")["labels"] == ["M3"]
    assert run(arms(seen, teacher=compact("minor")), "mistake")["side"] == MODEL


def test_m_nograde_is_follow_along_when_the_teacher_graded_the_word_and_unknown_when_not():
    for app in (None, trace("pending", 0.9)):
        graded = run(arms(app, teacher=compact("correct")), "mistake")
        assert (graded["bucket"], graded["side"]) == ("M-nograde", FOLLOW_ALONG)
        silent = run(arms(app, teacher=None), "mistake")
        assert (silent["bucket"], silent["side"]) == ("M-nograde", UNKNOWN)


def test_a_miss_records_every_label_it_met_in_precedence_order():
    both = trace("correct", 1.0, checks=[
        {"hop": "", "overlap": "", "flush": False, "status": status("minor", 0.5, [MISMATCH])}],
        setting=(0,))
    assert run(arms(both, teacher=compact("minor")), "mistake")["labels"] == ["M1r", "M1", "M2"]


# --- scorings and causes --------------------------------------------------------------------------
def test_the_balanced_reading_uses_the_shipped_default_rows():
    both = arms(FLAGGED, balanced=PASSED)
    assert run(both, scoring=STRICT)["outcome"] == "false_flag"
    assert run(both, scoring=SHIPPED_DEFAULT)["outcome"] == "true_pass"
    flipped = arms(PASSED, balanced=FLAGGED)
    assert run(flipped, "mistake", scoring=STRICT)["outcome"] == "miss"
    assert run(flipped, "mistake", scoring=SHIPPED_DEFAULT)["outcome"] == "correct_catch"


@pytest.mark.parametrize("causes, expected", [
    (("soft_pair",), True), (("target_pair", "tashkeel"), True), (("other_consonant", "soft_pair"), True),
    (("other_consonant",), False), (("shaddah", "tashkeel"), False), ((), False)])
def test_a_false_flag_is_soft_or_target_when_any_of_its_causes_is(causes, expected):
    assert run(arms(FLAGGED, causes=causes))["soft_or_target"] is expected


def test_only_a_false_flag_can_be_soft_or_target():
    assert run(arms(PASSED, causes=("soft_pair",)))["soft_or_target"] is False
    assert run(arms(FLAGGED, causes=("soft_pair",)), "mistake")["soft_or_target"] is False


# --- reference suspects ---------------------------------------------------------------------------
def test_a_word_is_a_suspect_when_its_surah_ayah_word_and_strict_errors_are_in_the_set():
    suspects = {(1, 2, 3, ((0, "mismatch", "س", "ص"),))}
    assert is_reference_suspect(key_row(), arms(FLAGGED), suspects) is True
    assert is_reference_suspect(key_row(word=4), arms(FLAGGED), suspects) is False
    assert is_reference_suspect(key_row(surah_ayah="1:3"), arms(FLAGGED), suspects) is False
    other = trace("minor", 0.5, [{**MISMATCH, "kind": {**MISMATCH["kind"], "heard": "ث"}}])
    assert is_reference_suspect(key_row(), arms(other), suspects) is False


def test_a_word_with_no_errors_is_never_a_suspect_and_the_strict_trace_decides():
    assert is_reference_suspect(key_row(), arms(PASSED), {(1, 2, 3, ())}) is False
    strict_only = arms(FLAGGED, balanced=PASSED)
    assert is_reference_suspect(key_row(), strict_only,
                                {(1, 2, 3, ((0, "mismatch", "س", "ص"),))}) is True

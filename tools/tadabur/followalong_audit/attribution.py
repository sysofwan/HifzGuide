"""The audit's attribution buckets, applied to S5's verdicts (#310, pre-registration section 5).

Every bucket is a pure function of the evidence the audit holds for one word: the App arm's
``training.muraja_policy.word_traces`` row, the oracle's, the Teacher arm's, and the word's
reference phonemes. The kept grade is the one the reader sees under the scoring being read
(:func:`training.muraja_policy.displayed`); the audit reads **strict** and reports balanced the
same way.

A **false flag** is a word the App flagged and the owner heard correct. Its bucket, first match
wins (architect decisions D3, D4, D8)::

    F1 -> F1-align -> F-ref -> F2 -> F3

A **missed mistake** is a word the owner heard as a mistake that the App did not flag. Every label
the word meets is recorded, in precedence order (D9)::

    M1r -> M1 -> M1u -> M2 -> M3

:func:`reference_suspects` marks a word whose App strict deviation repeats at the same word in
three tranche groups; it is a *reference suspect* shown both ways, never settled by hand (D3).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from training.muraja_policy import CORRECT, PENDING, STRICT, Scoring, displayed

#: Who owns a bucket: the follow-along side or the teacher/model side (rule 2's model share).
FOLLOW_ALONG = "follow-along"
MODEL = "model"

#: The buckets, in the precedence order their sides are read.
BUCKETS = ("F1", "F1-align", "F-ref", "F2", "F3", "M1r", "M1", "M1u", "M2", "M3")

_SIDES = {
    "F1": FOLLOW_ALONG,
    "F1-align": FOLLOW_ALONG,
    "M1r": FOLLOW_ALONG,
    "M1": FOLLOW_ALONG,
    "M1u": FOLLOW_ALONG,
    "F2": MODEL,
    "F3": MODEL,
    "M2": MODEL,
    "M3": MODEL,
}


def effective_quality(status: Mapping | None) -> str | None:
    """A check's quality as it graded the word: the end-word holdback's value when it has one."""
    return None if status is None else status.get("withheld") or status["quality"]


def below_correct(status: Mapping | None) -> bool:
    """Whether a check graded the word below correct: its effective quality is neither correct
    nor pending, and a check with no status did not grade it."""
    return status is not None and effective_quality(status) not in (CORRECT, PENDING)


def kept_outcome(row: Mapping | None, scoring: Scoring = STRICT) -> str:
    """The word's kept grade as ``"flagged"``, ``"passed"`` or ``"ungraded"``.

    No row and a pending row are ungraded — a pending grade the holdback withheld is still not a
    verdict the reader got (D5). Otherwise the displayed quality decides.
    """
    if row is None or row["quality"] == PENDING:
        return "ungraded"
    return "passed" if displayed(row["quality"], scoring) == CORRECT else "flagged"


def _aligned(app: Mapping, reference: Mapping) -> bool:
    """Whether a setting check's ``hop + overlap`` holds the word's reference with a neighbour.

    The primary D4 rule: ``previous + word`` or ``word + next`` as an exact code-point substring.
    A clip with no neighbour at all must hold the word alone, and only from four code points up:
    shorter words match by chance somewhere in a check's seconds of text.
    """
    word, previous, following = reference["word"], reference["previous"], reference["next"]
    if previous is None and following is None:
        seeds = [word] if len(word) >= 4 else []
    else:
        seeds = [] if previous is None else [previous + word]
        if following is not None:
            seeds.append(word + following)
    checks = app["checks"]
    return any(seed in checks[i]["hop"] + checks[i]["overlap"]
               for i in app["setting_checks"] for seed in seeds)


def false_flag_bucket(*, app: Mapping, oracle: Mapping | None, teacher: Mapping | None,
                      reference: Mapping, suspect: bool,
                      scoring: Scoring = STRICT) -> str:
    """The bucket of a word the App flagged and the owner heard correct.

    F1 first: the oracle's own kept grade flags the word. Then F1-align (the oracle passed and
    the App's setting check decoded the reference with a neighbour), F-ref (a reference suspect
    computed by :func:`reference_suspects`, checked before the Teacher arm), and finally the
    Teacher arm: F2 when it passed the word, F3 when it flagged it or left it ungraded (D8).
    """
    if kept_outcome(app, scoring) != "flagged":
        raise ValueError("a false flag is a word the App flagged")
    if kept_outcome(oracle, scoring) == "flagged":
        return "F1"
    if kept_outcome(oracle, scoring) == "passed" and _aligned(app, reference):
        return "F1-align"
    if suspect:
        return "F-ref"
    return "F2" if kept_outcome(teacher, scoring) == "passed" else "F3"


def _forgiven(app: Mapping) -> bool:
    """Whether a setting check decoded the word away: errors, or a score below 1, kept correct."""
    for index in app["setting_checks"]:
        status = app["checks"][index]["status"]
        if status is not None and (status["errors"] or status["score"] < 1):
            return True
    return False


def miss_labels(*, app: Mapping | None, teacher: Mapping | None,
                scoring: Scoring = STRICT) -> list[str]:
    """Every label a missed mistake meets, in precedence order; the first is its primary.

    M1r: the kept grade passed and some check graded the word below correct. M1: the kept grade
    passed and a *setting* check decoded the deviation away with errors or a score below 1. M1u:
    the kept grade is ungraded and some check graded the word below correct. Then the Teacher arm:
    M2 when it flagged the word, M3 when it did not (D8).
    """
    outcome = kept_outcome(app, scoring)
    if outcome == "flagged":
        raise ValueError("a missed mistake is a word the App did not flag")
    checks = app["checks"] if app is not None else []
    labels = []
    if outcome == "passed":
        if any(below_correct(check["status"]) for check in checks):
            labels.append("M1r")
        if _forgiven(app):
            labels.append("M1")
    elif any(below_correct(check["status"]) for check in checks):
        labels.append("M1u")
    labels.append("M2" if kept_outcome(teacher, scoring) == "flagged" else "M3")
    return labels


def side(bucket: str, *, f_ref: str = FOLLOW_ALONG) -> str:
    """The side a bucket belongs to; F-ref is placed on ``f_ref``'s side, both ways reported."""
    if bucket == "F-ref":
        if f_ref not in (FOLLOW_ALONG, MODEL):
            raise ValueError(f"{f_ref!r} is not a side: {FOLLOW_ALONG!r} or {MODEL!r}")
        return f_ref
    if bucket not in _SIDES:
        raise ValueError(f"{bucket!r} is not a bucket: {BUCKETS}")
    return _SIDES[bucket]


def model_share(buckets: Iterable[str], *, f_ref: str = FOLLOW_ALONG) -> float:
    """Rule 2's model-side fraction of the buckets, with F-ref on the side ``f_ref`` names."""
    buckets = list(buckets)
    if not buckets:
        raise ValueError("the model share of no words is undefined")
    return sum(1 for bucket in buckets if side(bucket, f_ref=f_ref) == MODEL) / len(buckets)


def error_key(errors: Iterable[Mapping]) -> tuple[tuple[int, str, str | None, str | None], ...]:
    """The App strict kept-grade errors as a canonical key, whatever order the engine listed them.

    Each error is ``(group_index, kind.type, kind.expected, kind.heard or None)``, sorted, so two
    traces that describe the same deviation key equal (D3).
    """
    return tuple(sorted((error["group_index"], error["kind"]["type"], error["kind"]["expected"],
                         error["kind"].get("heard")) for error in errors))


def reference_suspects(observations: Iterable[Mapping], minimum: int = 3
                       ) -> set[tuple[int, int, int, tuple]]:
    """The ``(surah, ayah, word, error key)`` deviations seen in ``minimum`` distinct tranche
    groups (D3).

    ``observations`` is one row per App strict flagged word of the whole population:
    ``{"group_key", "surah", "ayah", "word", "errors"}``. A word with no recorded error is never
    a suspect: the deviation itself is the signal, not the flag.
    """
    groups: dict[tuple[int, int, int, tuple], set[str]] = defaultdict(set)
    for row in observations:
        key = error_key(row["errors"])
        if key:
            groups[(row["surah"], row["ayah"], row["word"], key)].add(row["group_key"])
    return {word for word, seen in groups.items() if len(seen) >= minimum}

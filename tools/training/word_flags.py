"""What the app flags on the random word sample, and why (#90, ADR-0012).

Ship A's word sample (:mod:`tadabur.word_sample`) is the one set of correctly recited words no
model chose. Each sampled word is replayed through Muraja's own engine with the rest of its
segment (:mod:`training.muraja_policy`), per arm and per scoring. A word the owner heard as
recited correctly that the app shows as not correct is a **false flag**; a word heard as a
mistake that the app shows as correct, or never grades, is a **missed mistake**. The two are
never pooled (ADR-0008).

**Why a word is flagged.** Each false flag is attributed to the errors of the grade the engine
kept for the word (Muraja's ``WordStatus``: its quality as shown and its ``WordError`` list,
``FollowAlong/FollowAlongTypes.swift:227-296``; the error kinds are built in
``QuranFollowAlong+WordScoring.swift:529-636`` and the quality at ``:711-746``), read with the
rule that produced the quality:

* a word shown as a **tashkeel error** is flagged for its tashkeel errors (:data:`TASHKEEL`).
  Each names the mark the mushaf prescribes on the letter (``fatha``, ``damma``, ``kasra``;
  ``sukun`` where the mushaf writes no haraka) and whether the decode left the slot empty
  (``dropped``) or put another mark there (``other``);
* a word shown as **minor, wrong or uncertain** is flagged for its consonant errors (a tashkeel
  error costs a word neither score nor the phoneme gate, so it is never why such a word fell). A
  mismatch is a :data:`SOFT_PAIR` when its two letters are one of Muraja's six soft pairs, a
  :data:`TARGET_PAIR` when they are ``ذ↔ظ``, ``س↔ش`` or ``ط↔ق`` (target pairs the app does not
  forgive), else :data:`OTHER_CONSONANT`. A deletion is :data:`SHADDAH_CAUSE` only when the
  deleted occurrence is a consonant geminate (:func:`deleted_run`): a consonant
  (``truth_sites.CONSONANTS``, so never a long vowel such as ``اا``) in a run of at least two
  in the word's Muraja reference. Anything else, or an occurrence that cannot be identified, is
  another consonant;
* a word whose kept grade is **skipped** (Muraja's skip marking, ``SkipMarking.swift``) is
  :data:`SKIPPED`;
* any other flagged word whose kept grade carries no error of its kind is :data:`UNRECORDED`: the
  engine reports none for what lowered it (a score below the threshold from coverage alone, or a
  collapsed geminate whose gap the engine folded into the group without a deletion,
  ``WordScoring.swift:478-489, 626-636``).

A word may carry several causes; each cause counts the false flags that carry it, so the causes
of one scoring can sum to more than its false flags.

**Rates.** Each word weighs the inverse of its selection probability within the sample's frame
(:attr:`tadabur.word_sample.VerifiedWord.weight`); a rate is ΣwF / Σw over the correctly recited
words, with the §1 interval of ``truth_scorer.estimate`` (one word per reciter, so the
reciter-clustered bootstrap resamples words; the weights are unequal, so a degenerate rate, such
as a cause no word carries, has no admissible bound). Counts are given next to every rate. With
one word per held-out reciter the sample is small, and the intervals are wide.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tadabur.exclusions import Exclusions
from tadabur.truth_sites import CONSONANTS, SOFT_PAIRS, SUKUN, UNFORGIVEN_PAIRS
from tadabur.word_sample import VerifiedWord
from training.acceptance_stats import Resample, unit_if_equal
from training.muraja_policy import (
    NOT_GRADED,
    TASHKEEL_ERROR,
    UNPLACED,
    WORD_FLAGGED,
    SiteGrade,
)
from training.site_outcomes import CORRECT_SIDE, MARK_NAMES, MISTAKE_SIDE
from training.truth_scorer import estimate

TASHKEEL = "tashkeel"
SOFT_PAIR = "soft_pair"
TARGET_PAIR = "target_pair"
OTHER_CONSONANT = "other_consonant"
SHADDAH_CAUSE = "shaddah"
SKIPPED = "skipped"
UNRECORDED = "unrecorded"
#: Every cause, in report order.
CAUSES = (TASHKEEL, SOFT_PAIR, TARGET_PAIR, OTHER_CONSONANT, SHADDAH_CAUSE, SKIPPED, UNRECORDED)
#: Muraja's quality for a word its skip marking passed over (``FollowAlongTypes.swift:369``).
SKIPPED_QUALITY = "skipped"
DROPPED = "dropped"
OTHER_MARK = "other"


@dataclass(frozen=True)
class Flag:
    """Why one flagged word is flagged: its causes, its tashkeel errors (``<mark> <dropped|
    other>``) and the kept grade (quality shown, every error as Muraja logs it)."""

    causes: tuple[str, ...]
    marks: tuple[str, ...]
    quality: str
    errors: tuple[str, ...]


def _pair(a: str, b: str) -> str:
    return "↔".join(sorted((a, b)))


def tashkeel_mark(error: dict) -> str:
    """``<prescribed mark> <dropped|other>`` for one tashkeel error."""
    kind = error["kind"]
    expected = SUKUN if kind["expected"] is None else MARK_NAMES.get(kind["expected"], kind["expected"])
    return f"{expected} {DROPPED if kind['heard'] is None else OTHER_MARK}"


#: What Muraja's word scorer skips before it numbers a word's groups (``maddScalars``,
#: ``QuranFollowAlong+WordScoring.swift:61-66``): the small waw and yeh of a long vowel, the
#: ghunna meem and the idgham marker.
SKIPPED_SCALARS = frozenset("\u06e6\u06e5\u06fe\u0687")


def letters(word: str) -> list[str]:
    """The base letter of each unit Muraja's scorer numbers in a word: one per Character
    (a letter with its marks), less :data:`SKIPPED_SCALARS`."""
    bases = []
    for char in word:
        if char in SKIPPED_SCALARS:
            continue
        if unicodedata.category(char) != "Mn":
            bases.append(char)
    return bases


def deleted_run(word: str, letter: str, group: int) -> int | None:
    """The length of the run of ``letter`` the deletion at ``group`` took a letter from.

    The deleted occurrence is the unit ``group`` names when that unit is ``letter`` (a
    collapsed geminate whose gap comes first is reported on its first half, at the group the
    scorer would give it, ``WordScoring.swift:478-489, 626-636``); otherwise, when every run of
    ``letter`` in the word has one length, that length; otherwise ``None`` (unidentified)."""
    bases = letters(word)

    def run_at(i: int) -> int:
        start, end = i, i
        while start > 0 and bases[start - 1] == letter:
            start -= 1
        while end + 1 < len(bases) and bases[end + 1] == letter:
            end += 1
        return end - start + 1

    if 0 <= group < len(bases) and bases[group] == letter:
        return run_at(group)
    runs = {run_at(i) for i, base in enumerate(bases) if base == letter}
    return runs.pop() if len(runs) == 1 else None


def consonant_cause(error: dict, word: str) -> str | None:
    """The cause of one consonant error (a mismatch or a deletion) in a word whose Muraja
    reference phonemes are ``word``; ``None`` for a tashkeel error."""
    kind = error["kind"]
    if kind["type"] == "mismatch":
        pair = _pair(kind["expected"], kind["heard"])
        return SOFT_PAIR if pair in SOFT_PAIRS else TARGET_PAIR if pair in UNFORGIVEN_PAIRS \
            else OTHER_CONSONANT
    if kind["type"] == "deletion":
        letter = kind["expected"]
        run = deleted_run(word, letter, error["group_index"]) if letter in CONSONANTS else None
        return SHADDAH_CAUSE if run is not None and run >= 2 else OTHER_CONSONANT
    return None


def log_description(error: dict) -> str:
    """An error as Muraja's ``WordError.logDescription`` writes it (``FollowAlongTypes.swift:286``)."""
    kind, group = error["kind"], error["group_index"]
    if kind["type"] == "mismatch":
        return f"g{group}:{kind['expected']}→{kind['heard']}"
    if kind["type"] == "deletion":
        return f"g{group}:del:{kind['expected']}"
    return f"g{group}:tash"


def flag(grade: SiteGrade, word: str) -> Flag:
    """Why a flagged word is flagged (module docstring)."""
    tashkeel = [e for e in grade.errors if e["kind"]["type"] == "tashkeel"]
    if grade.quality == SKIPPED_QUALITY:
        causes = {SKIPPED}
    elif grade.quality == TASHKEEL_ERROR:
        causes = {TASHKEEL} if tashkeel else set()
    else:
        causes = {c for e in grade.errors if (c := consonant_cause(e, word)) is not None}
    return Flag(
        causes=tuple(c for c in CAUSES if c in causes) or (UNRECORDED,),
        marks=tuple(sorted(tashkeel_mark(e) for e in tashkeel)) if TASHKEEL in causes else (),
        quality=grade.quality,
        errors=tuple(log_description(e) for e in grade.errors),
    )


def _rate(words: Sequence[VerifiedWord], hits: Sequence[bool]) -> dict:
    """The weighted share of ``words`` that are ``hits``, its §1 interval and its count."""
    clusters = [w.site.reciter_id for w in words]
    w = np.array([x.weight for x in words], dtype=float)
    unit = unit_if_equal(w)
    hit = np.array(hits, dtype=float)
    return {"words": int(hit.sum()),
            **estimate(clusters, (w * hit, w), unit, Resample.by_cluster(clusters),
                       (unit * hit, unit))}


def scoring_view(words: Sequence[VerifiedWord], grades: Mapping[str, SiteGrade],
                 texts: Mapping[str, str]) -> dict:
    """One arm under one scoring: the false flags on correctly recited words with their
    causes, and what the app did with the words heard as mistakes."""
    correct = [w for w in words if w.side == CORRECT_SIDE]
    flagged = {w.site.site_id: flag(grades[w.site.site_id], texts[w.site.site_id])
               for w in correct if grades[w.site.site_id].outcome in WORD_FLAGGED}
    marks = Counter(m for f in flagged.values() for m in set(f.marks))
    return {
        "correct_words": len(correct),
        "false_flags": _rate(correct, [w.site.site_id in flagged for w in correct]),
        "causes": {c: _rate(correct, [w.site.site_id in flagged
                                      and c in flagged[w.site.site_id].causes for w in correct])
                   for c in CAUSES},
        "tashkeel_marks": dict(sorted(marks.items())),
        "not_graded": sum(grades[w.site.site_id].outcome == NOT_GRADED for w in correct),
        "unplaced": sum(grades[w.site.site_id].outcome == UNPLACED for w in correct),
        "mistakes": {w.site.site_id: grades[w.site.site_id].outcome
                     for w in words if w.side == MISTAKE_SIDE},
        "flags": {word_id: {"causes": list(f.causes), "marks": list(f.marks),
                            "quality": f.quality, "errors": list(f.errors)}
                  for word_id, f in sorted(flagged.items())},
    }


def word_report(
    words: Sequence[VerifiedWord],
    grades: Mapping[str, Mapping[str, Mapping[str, SiteGrade]]],
    texts: Mapping[str, str],
    excluded_items: Exclusions,
) -> dict:
    """The word sample's app outcome: per arm and scoring (``grades[scoring][arm][word id]``),
    :func:`scoring_view` over the words the shared exclusion keeps. ``texts`` gives each placed
    word's Muraja reference phonemes."""
    kept = [w for w in words if w.site.site_id not in excluded_items]
    arms = sorted({arm for by_arm in grades.values() for arm in by_arm})
    weights = [w.weight for w in kept]
    return {
        "words": len(words),
        "verdicts": dict(sorted(Counter(w.heard for w in words).items())),
        "weight_range": [min(weights), max(weights)],
        **excluded_items.listed(w.site.site_id for w in words),
        "arms": {arm: {scoring: scoring_view(kept, by_arm[arm], texts)
                       for scoring, by_arm in grades.items()} for arm in arms},
    }

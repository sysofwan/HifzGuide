"""What the owner's free-text verdict notes claim (#134, #138).

A listening verdict (:mod:`tadabur.listening_session`) carries a note the owner types by hand.
The guards that take items out of evaluation read a claim from it, and the owner's claim
decides before any detector: the waqf guard (:mod:`tadabur.waqf_guard`) reads :data:`PAUSE`,
whether the reciter paused, and the onset guard (:mod:`tadabur.onset_guard`) reads
:data:`TRIM`, whether the clip's opening sound is cut. :func:`says` is the one reader.

The note is lower-cased with its whitespace and apostrophes normalized, and cut into clauses
at punctuation (``, ; . : ! ? ( )``) and at ``but`` and ``and``. Inside a clause it is read
word by word. A word of any claim (:data:`CLAIMS`) affirms or denies its claim, and is negated
when a negator (``no``, ``not``, ``never``, ``without``, ``nor``, ``cannot`` or a word ending
in ``n't``, or its apostrophe-less spelling, ``didnt``) is among the three words before it. A
negator governs only the next claim word, of whichever claim, so "no waqf, trimmed" and "no
waqf trimmed" deny a pause and affirm a trim. Two readings of one claim that disagree are
ambiguous: no claim.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

#: Words that negate the next claim word among the :data:`NEGATION_REACH` words after them.
NEGATORS = frozenset({"no", "not", "never", "without", "nor", "cannot", "didnt", "doesnt",
                      "dont", "hasnt", "havent", "hadnt", "isnt", "wasnt", "wont"})
NEGATION_REACH = 3
#: What ends a clause: punctuation, or a conjunction.
_CLAUSE_BREAK = re.compile(r"[,;.:!?()\n]|\bbut\b|\band\b")


def _never(word: str) -> bool:
    return False


@dataclass(frozen=True)
class Claim:
    """A claim a note can make: the words that affirm it, and those that deny it."""

    affirms: Callable[[str], bool]
    denies: Callable[[str], bool] = _never

    def names(self, word: str) -> bool:
        return self.affirms(word) or self.denies(word)


#: The reciter paused (``waqf``, ``pause(d)``, ``stop(ped)``), or continued (``wasl``).
PAUSE = Claim(frozenset({"waqf", "pause", "paused", "pauses", "pausing", "stop", "stopped",
                         "stops", "stopping"}).__contains__, frozenset({"wasl"}).__contains__)
#: The clip's opening sound is cut: a word starting ``trim``, ``cut`` or ``split``.
TRIM = Claim(lambda word: word.startswith(("trim", "cut", "split")))
CLAIMS = (PAUSE, TRIM)


def _negator(word: str) -> bool:
    return word in NEGATORS or word.endswith("n't")


def says(note: str, claim: Claim) -> bool | None:
    """``True`` when the note affirms ``claim``, ``False`` when it denies it, ``None`` when it
    says neither or is ambiguous (module docstring)."""
    text = " ".join(note.lower().replace("’", "'").split())
    readings = set()
    for clause in _CLAUSE_BREAK.split(text):
        words = re.findall(r"[a-z']+", clause)
        negator = None  # the position of the negator waiting for its claim word
        for k, word in enumerate(words):
            if _negator(word):
                negator = k
                continue
            if not any(c.names(word) for c in CLAIMS):
                continue
            negated = negator is not None and k - negator <= NEGATION_REACH
            negator = None
            if claim.names(word):
                readings.add(claim.affirms(word) != negated)
    return readings.pop() if len(readings) == 1 else None

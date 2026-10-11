"""What the owner's free-text verdict notes claim (#134, #138).

A listening verdict (:mod:`tadabur.listening_session`) carries a note the owner types by hand.
The guards that take items out of evaluation read a claim from it, and the owner's claim
decides before any detector: the waqf guard (:mod:`tadabur.waqf_guard`) reads :data:`PAUSE`,
whether the reciter paused, and the onset guard (:mod:`tadabur.onset_guard`) reads
:data:`TRIM`, whether the clip's opening sound is cut. :func:`says` is the one reader.

The note is cut into clauses at line breaks and at every Unicode punctuation mark (so the
Arabic ``،``, ``؛`` and ``؟`` too) but the apostrophe and a dash joining two letters
("not-trimmed", "no-waqf", read as two words); a spaced dash ("no waqf - trimmed") cuts. A
bracketed span, "no (audible) pause", is a qualifier and is skipped. Inside a clause the note
is read word by word, lower-cased. A word of a claim (:data:`CLAIMS`) affirms or denies it. A
negator (``no``, ``not``, ``never``, ``without``, ``nor``, ``cannot`` or a word ending in
``n't``, or its apostrophe-less spelling, ``didnt``) negates the first claim word it reaches
within three words (#134's window), and every later word of that same claim in the clause
("no pausing and stopping"); a word of another claim ends its scope, so "no waqf and trimmed"
and "no waqf trimmed" deny a pause and affirm a trim. ``hafs`` (the owner's "Not hafs",
"May not be hafs") is what a negator reaching it negates: the negator is spent there and
negates nothing after it. **This departs from #134's reader on purpose**: it read "Not hafs
paused" as no pause, which was a bug; no committed note reads differently. Two readings of
one claim that disagree are ambiguous: no claim.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

#: Words that negate the first claim word among the :data:`NEGATION_REACH` words after them.
NEGATORS = frozenset({"no", "not", "never", "without", "nor", "cannot", "didnt", "doesnt",
                      "dont", "hasnt", "havent", "hadnt", "isnt", "wasnt", "wont"})
NEGATION_REACH = 3
#: Line breaks (Unicode's), which end a clause like punctuation.
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
#: A bracketed qualifier, skipped.
_BRACKETED = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}")


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


def clauses(note: str) -> list[str]:
    """The note's clauses, lower-cased, bracketed qualifiers dropped (module docstring)."""
    text = _BRACKETED.sub(" ", note.lower().replace("\u2019", "'"))
    out, current = [], []
    for k, char in enumerate(text):
        category = unicodedata.category(char)
        joins = (category == "Pd" and 0 < k < len(text) - 1
                 and text[k - 1].isalpha() and text[k + 1].isalpha())
        if char in _LINE_BREAKS or (char != "'" and category.startswith("P") and not joins):
            out.append("".join(current))
            current = []
        else:
            current.append(char)
    return [*out, "".join(current)]


def says(note: str, claim: Claim) -> bool | None:
    """``True`` when the note affirms ``claim``, ``False`` when it denies it, ``None`` when it
    says neither or is ambiguous (module docstring)."""
    readings = set()
    for clause in clauses(note):
        negator: int | None = None  # a negator still looking for its first claim word
        governed: Claim | None = None  # the claim a negator's scope covers
        for k, word in enumerate(re.findall(r"[a-z']+", clause)):
            if _negator(word):
                negator, governed = k, None
                continue
            if word == "hafs" or word.startswith("hafs'"):
                negator = governed = None  # "not hafs": the negator is spent here
                continue
            owner = next((c for c in CLAIMS if c.names(word)), None)
            if owner is None:
                continue
            if negator is not None:
                governed = owner if k - negator <= NEGATION_REACH else None
                negator = None
            elif governed is not owner:
                governed = None
            if owner is claim:
                readings.add(claim.affirms(word) != (governed is claim))
    return readings.pop() if len(readings) == 1 else None

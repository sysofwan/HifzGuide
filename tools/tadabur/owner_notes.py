"""What the owner's free-text verdict notes claim (#134, #138).

A listening verdict (:mod:`tadabur.listening_session`) carries a note the owner types by hand.
The guards that take items out of evaluation read a claim from it, and the owner's claim
decides before any detector: the waqf guard (:mod:`tadabur.waqf_guard`) reads :data:`PAUSE`,
whether the reciter paused, and the onset guard (:mod:`tadabur.onset_guard`) reads
:data:`TRIM`, whether the clip's opening sound is cut. :func:`says` is the one reader.

Brackets and quotation marks are transparent: ``( ) [ ] { }``, every opening or closing
quote (``" “ ” « » „``, a backquote, and an apostrophe not between two letters) is read as a
space, so it neither cuts a clause nor hides a word from its negator ("no (audible) pause",
'no "pause"' and "not (trimmed)" deny; "clip (trimmed)" affirms). An apostrophe between two
letters stays ("didn't"). The note is then cut into clauses at line breaks and at every other
Unicode punctuation mark (so the Arabic ``،``, ``؛`` and ``؟`` too) but a dash joining two
letters ("not-trimmed", "no-waqf", read as two words); a spaced dash ("no waqf - trimmed")
cuts. Inside a clause the note
is read word by word, lower-cased. A word of a claim (:data:`CLAIMS`) affirms or denies it. A
negator (``no``, ``not``, ``never``, ``without``, ``nor``, ``cannot`` or a word ending in
``n't``, or its apostrophe-less spelling, ``didnt``) negates the first claim word it reaches
within three words (#134's window), and every later word of that same claim and polarity in
the clause ("no pausing and stopping"); a word of the opposite polarity ("no pause and wasl")
or of another claim ends its scope and is read on its own, so "no waqf and trimmed"
and "no waqf trimmed" deny a pause and affirm a trim. ``hafs`` (the owner's "Not hafs",
"May not be hafs") is what a negator reaching it negates: the negator is spent there and
negates nothing after it. **This departs from #134's reader on purpose**: it read "Not hafs
paused" as no pause, which was a bug; no committed note reads differently. Two readings of
one claim that disagree are ambiguous: no claim.

The listening page's **Trimmed** answer starts its note with :data:`TRIMMED_ANSWER`. That is
the listener's explicit action, so it affirms :data:`TRIM` whatever the rest of the note says
("trim; not trimmed" is trimmed); the rest is read as usual for every other claim.
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
#: Punctuation categories read as a space: brackets and opening or closing quotes.
_TRANSPARENT_CATEGORIES = frozenset({"Ps", "Pe", "Pi", "Pf"})
#: Quote marks of other categories ("'" between two letters is an apostrophe and stays).
_QUOTES = frozenset("\"'`\u2019\u2018")


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
#: How the listening page's Trimmed answer starts its note (``trimNote`` in
#: ``tashkeel_audit_session.mjs``).
TRIMMED_ANSWER = "trim;"


def _negator(word: str) -> bool:
    return word in NEGATORS or word.endswith("n't")


def _transparent(text: str, k: int) -> bool:
    char = text[k]
    if char in "'\u2019" and 0 < k < len(text) - 1 and text[k - 1].isalpha() \
            and text[k + 1].isalpha():
        return False  # an apostrophe inside a word
    return char in _QUOTES or unicodedata.category(char) in _TRANSPARENT_CATEGORIES


def clauses(note: str) -> list[str]:
    """The note's clauses, lower-cased, brackets and quotes read as spaces (module docstring)."""
    text = note.lower()
    text = "".join(" " if _transparent(text, k) else text[k] for k in range(len(text)))
    text = text.replace("\u2019", "'")
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
    if claim is TRIM and note.lstrip().lower().startswith(TRIMMED_ANSWER):
        return True
    readings = set()
    for clause in clauses(note):
        negator: int | None = None  # a negator still looking for its first claim word
        governed: tuple[Claim, bool] | None = None  # a negator's scope: claim and polarity
        for k, word in enumerate(re.findall(r"[a-z']+", clause)):
            if _negator(word):
                negator, governed = k, None
                continue
            if word == "hafs":
                negator = governed = None  # "not hafs": the negator is spent here
                continue
            owner = next((c for c in CLAIMS if c.names(word)), None)
            if owner is None:
                continue
            scope = (owner, owner.affirms(word))
            if negator is not None:
                governed = scope if k - negator <= NEGATION_REACH else None
                negator = None
            elif governed != scope:
                governed = None
            if owner is claim:
                readings.add(claim.affirms(word) != (governed == scope))
    return readings.pop() if len(readings) == 1 else None

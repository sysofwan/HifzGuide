"""What the owner's free-text verdict notes claim (#134, #138).

A listening verdict (:mod:`tadabur.listening_session`) carries a note the owner types by hand.
The guards that take items out of evaluation read a claim from it, and the owner's claim
decides before any detector: the waqf guard (:mod:`tadabur.waqf_guard`) reads whether the
reciter paused, the onset guard (:mod:`tadabur.onset_guard`) whether the clip's opening sound
is cut. :func:`claim` is the one reader both use.

The note is lower-cased with its whitespace and apostrophes normalized, then read word by
word. A word that affirms the claim says yes, a word that denies it says no, and either is
negated when ``no``, ``not``, ``never``, ``without``, ``nor``, ``cannot`` or a word ending in
``n't`` (or its apostrophe-less spelling, ``didnt``) is among the three words before it. Two
claims that disagree are ambiguous: no claim.
"""

from __future__ import annotations

import re
from collections.abc import Callable

#: Words that negate a claim word among the :data:`NEGATION_REACH` words before it.
NEGATORS = frozenset({"no", "not", "never", "without", "nor", "cannot", "didnt", "doesnt",
                      "dont", "hasnt", "havent", "hadnt", "isnt", "wasnt", "wont"})
NEGATION_REACH = 3


def _never(word: str) -> bool:
    return False


def claim(note: str, affirms: Callable[[str], bool],
          denies: Callable[[str], bool] = _never) -> bool | None:
    """``True`` when the note affirms the claim, ``False`` when it denies it, ``None`` when it
    says neither or is ambiguous (module docstring)."""
    words = re.findall(r"[a-z']+", " ".join(note.lower().replace("’", "'").split()))
    claims = set()
    for k, word in enumerate(words):
        if affirms(word):
            says = True
        elif denies(word):
            says = False
        else:
            continue
        negated = any(w in NEGATORS or w.endswith("n't")
                      for w in words[max(0, k - NEGATION_REACH):k])
        claims.add(says != negated)
    return claims.pop() if len(claims) == 1 else None

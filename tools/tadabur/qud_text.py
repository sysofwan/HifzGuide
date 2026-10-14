"""QUD's recited text onto quran-transcript's Uthmani words, and each item's reference (#144).

QUD's ``text_uthmani`` is the DigitalKhatt V2 spelling of **what was recited**: a word the
reciter repeated appears again where it was repeated. The repo's phonetizer reads only
quran-transcript's own Uthmani text (:mod:`hafs_phonetizer`), so every QUD word is mapped to
the canonical word it spells, keeping the recited order:

* both spellings are reduced to a **skeleton** (:func:`skeleton`): the base letters only, with
  the alif forms, alif maqsura and the hamza seats folded, so the two scripts' differences
  (waqf signs attached to a word, open-tanween code points, the tatweel before a dagger
  alif, ``ى`` for ``ي``, ``ؤ`` written as ``و`` with a combining hamza) vanish;
* the QUD words are walked in order against the ayah's canonical words (:func:`map_words`):
  a word that matches the next canonical word advances; one that matches an earlier canonical
  word is a **repeat** (the reciter went back) and restarts there. Anything else, or an ayah
  whose canonical words are not all reached, is **unmapped** and reported, never guessed.

An item is one of QUD's segments (:mod:`tadabur.qud_source`); its **realized reference** is
the phonetizer's output over the canonical words it recited, in recited order, its last word
in waqf (:func:`tadabur.waqf_segments.hafs_segment_reference`, as every realized reference in
the repo is built), with the per-word offsets into it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: Base Arabic letters (hamza to ya, less the tatweel) and alif wasla; everything else is a mark.
_LETTERS = re.compile("[\u0621-\u063f\u0641-\u064a\u0671]")
#: Folded: the alif forms, alif maqsura to ya, and a hamza's seat letters to the seat (QUD
#: writes ؤ and ئ as و and ي with a combining hamza above, which the letter filter drops).
_FOLD = str.maketrans({"\u0671": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ؤ": "و",
                       "ئ": "ي"})


def skeleton(word: str) -> str:
    """A word's base letters, alif forms and alif maqsura folded (module docstring)."""
    return "".join(_LETTERS.findall(word)).translate(_FOLD)


@dataclass(frozen=True)
class Mapping:
    """An ayah's QUD words onto its canonical words: per QUD word, the canonical index, and
    the QUD words that are repeats (start a run again at an earlier canonical word)."""

    indices: tuple[int, ...]
    repeats: tuple[int, ...]


class Unmapped(ValueError):
    """An ayah whose recited words cannot be put on its canonical words."""


def map_words(recited: Sequence[str], canonical: Sequence[str]) -> Mapping:
    """Each recited word's canonical index (module docstring); raises :class:`Unmapped`."""
    keys = [skeleton(w) for w in canonical]
    indices: list[int] = []
    repeats: list[int] = []
    reached = -1
    following = 0
    for position, word in enumerate(recited):
        key = skeleton(word)
        if following < len(keys) and keys[following] == key:
            index = following
        else:
            earlier = [i for i in range(following) if keys[i] == key]
            if not earlier:
                raise Unmapped(f"recited word {position + 1} {word!r} matches no canonical "
                               f"word at or before word {following + 1}")
            index = earlier[-1]
            repeats.append(position)
        indices.append(index)
        following = index + 1
        reached = max(reached, index)
    if reached != len(keys) - 1 or set(indices) != set(range(len(keys))):
        missing = sorted(set(range(len(keys))) - set(indices))
        raise Unmapped(f"canonical words {[m + 1 for m in missing]} are never recited")
    return Mapping(tuple(indices), tuple(repeats))


@dataclass(frozen=True)
class ItemReference:
    """One segment's canonical words (recited order) and its realized reference."""

    words: tuple[str, ...]
    word_indices: tuple[int, ...]
    reference: str
    offsets: tuple[int, ...]


def item_reference(
    segment: Sequence[int], mapping: Mapping, canonical: Sequence[str],
    reference_of: Callable[[list[str]], tuple[str, list[int]]],
) -> ItemReference:
    """A QUD segment's (``[word_from, word_to, ...]``, 1-based recited words) canonical words
    and realized reference. ``reference_of`` raises ``KeyError`` / ``IndexError`` where the
    phonetizer cannot handle the words."""
    word_from, word_to = segment[0], segment[1]
    if not 1 <= word_from <= word_to <= len(mapping.indices):
        raise Unmapped(f"segment {list(segment)} is outside the {len(mapping.indices)} "
                       "recited words")
    indices = mapping.indices[word_from - 1:word_to]
    words = [canonical[i] for i in indices]
    reference, offsets = reference_of(words)
    return ItemReference(tuple(words), tuple(indices), reference, tuple(offsets))

"""Planted mistakes (#308): one seeded edit to an ayah's spaceless phoneme words.

The oracle places its reference on the clock before the engine grades it
(:func:`tadabur.followalong_audit.oracle.oracle_checks` refuses a reference its checks do not
commit), so a planted mistake is a pure edit of the words — the checks are rebuilt from the
edited reference, never patched. A **Character** here is a consonant with its marks
(:func:`training.decoding.clusters`), and every kind states its edit on Characters:

* ``consonant_pair:ح↔ه`` — one letter of a target pair becomes its partner, marks kept;
* ``consonant_random`` — one plain consonant becomes another, never a madd letter;
* ``haraka_swap`` — one haraka on one consonant becomes a different haraka;
* ``haraka_drop`` — one haraka is removed; the drop is labelled ``protected`` when the letter
  is one of :data:`PROTECTED_LETTERS` (و ا ء ي, the app's haraka-drop exemptions) and ``other``
  otherwise;
* ``shaddah_drop`` — a geminate (a bare consonant followed by the same consonant) collapses to
  one consonant;
* ``skip_word`` — one word leaves the ayah;
* ``add_word`` — a copy of one word is inserted right after it.

Every draw comes from a caller-supplied :class:`random.Random`, so the same seed plants the
same mistake; a caller plants before placing, and the sweep records which word moved.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from tadabur.truth_sites import CONSONANTS, HARAKA_CHARS, TARGET_PAIRS
from training.decoding import clusters

#: The haraka characters a swap or a drop may move (the three short vowels).
HARAKAT = frozenset(HARAKA_CHARS.values())

#: The letters whose haraka drop the app forgives (CONTEXT.md, *haraka drop*).
PROTECTED_LETTERS = frozenset("واءي")

#: Every kind, in the order the report lists them: each target pair, then the six others.
KINDS = (
    *(f"consonant_pair:{pair}" for pair in sorted(TARGET_PAIRS)),
    "consonant_random",
    "haraka_swap",
    "haraka_drop",
    "shaddah_drop",
    "skip_word",
    "add_word",
)


@dataclass(frozen=True)
class Planted:
    """One planted mistake: the edited words, the word that moved, and how it moved."""

    kind: str
    words: tuple[str, ...]
    #: The 0-based word the edit was planted on. For ``add_word`` this is the copied word;
    #: the copy itself is at :attr:`inserted`.
    word: int
    label: str
    #: ``True``/``False`` for a ``haraka_drop`` on a protected letter / any other; ``None``
    #: for every other kind.
    protected: bool | None = None
    #: For ``add_word``, the 0-based index of the inserted copy; otherwise ``None``.
    inserted: int | None = None
    #: The 0-based Character (cluster) index the edit touched, or ``None`` for skip/add.
    character: int | None = None


def planting_seed(run_seed: int, item: str, kind: str) -> str:
    """The seed string a kind's draw for one clip uses: stable, and distinct per run/clip/kind."""
    return f"{run_seed}|{item}|{kind}"


def _pair(kind: str) -> tuple[str, str]:
    first, second = kind.split(":", 1)[1].split("↔")
    return first, second


def _geminates(characters: Sequence[str]) -> list[int]:
    """The indices of bare consonants that the next Character repeats (a shaddah)."""
    return [
        index for index, character in enumerate(characters)
        if character[0] in CONSONANTS
        and len(character) == 1
        and index + 1 < len(characters)
        and characters[index + 1][0] == character[0]
    ]


def _sites(kind: str, words: Sequence[str]) -> list[tuple[int, int]]:
    """Every ``(word, Character)`` a kind can be planted on, in word then Character order."""
    if kind == "skip_word":
        return [(word, 0) for word in range(len(words))] if len(words) > 1 else []
    if kind == "add_word":
        return [(word, 0) for word in range(len(words))]
    grid = [clusters(word) for word in words]
    sites: list[tuple[int, int]] = []
    for word, characters in enumerate(grid):
        for index, character in enumerate(characters):
            if kind.startswith("consonant_pair:"):
                first, second = _pair(kind)
                plantable = character[0] in (first, second)
            elif kind == "consonant_random":
                plantable = character[0] in CONSONANTS
            elif kind == "haraka_swap":
                plantable = bool(set(character[1:]) & HARAKAT)
            elif kind == "haraka_drop":
                plantable = sum(1 for mark in character[1:] if mark in HARAKAT) == 1
            elif kind == "shaddah_drop":
                plantable = index in _geminates(characters)
            else:
                raise ValueError(f"{kind!r} is not a planted kind: {KINDS}")
            if plantable:
                sites.append((word, index))
    return sites


def plantable_words(kind: str, words: Sequence[str]) -> list[int]:
    """The words a kind can be planted in, ascending."""
    return sorted({word for word, _ in _sites(kind, words)})


def plant(kind: str, words: Sequence[str], rng: random.Random,
          word: int | None = None) -> Planted:
    """Plant ``kind`` in one word of ``words``; at ``word`` when named, else the seeded draw.

    The draw picks a word uniformly among the words that can carry the kind and then a site
    uniformly inside it, so a name that cannot carry the kind raises rather than planting
    somewhere else.
    """
    sites = _sites(kind, words)
    if not sites:
        raise ValueError(f"{kind} is not plantable in {list(words)}")
    if word is None:
        chosen = rng.choice(sorted({site_word for site_word, _ in sites}))
    else:
        chosen = word
    in_word = [candidate for candidate in sites if candidate[0] == chosen]
    if not in_word:
        raise ValueError(f"{kind} is not plantable in word {word} of {list(words)}")
    character_index = rng.choice(in_word)[1]

    edited = list(words)
    protected: bool | None = None
    inserted: int | None = None
    character: int | None = character_index

    if kind == "skip_word":
        label = words[chosen]
        edited = edited[:chosen] + edited[chosen + 1:]
        character, inserted = None, None
    elif kind == "add_word":
        label = words[chosen]
        edited = edited[:chosen + 1] + [words[chosen]] + edited[chosen + 1:]
        inserted, character = chosen + 1, None
    else:
        characters = clusters(words[chosen])
        old = characters[character_index]
        if kind.startswith("consonant_pair:"):
            first, second = _pair(kind)
            new_base = second if old[0] == first else first
            characters[character_index] = new_base + old[1:]
            label = f"{old[0]}→{new_base}"
        elif kind == "consonant_random":
            new_base = rng.choice(sorted(CONSONANTS - {old[0]}))
            characters[character_index] = new_base + old[1:]
            label = f"{old[0]}→{new_base}"
        elif kind == "haraka_swap":
            marks = old[1:]
            positions = [index for index, mark in enumerate(marks) if mark in HARAKAT]
            position = rng.choice(positions)
            new_mark = rng.choice(sorted(HARAKAT - {marks[position]}))
            characters[character_index] = (
                old[0] + marks[:position] + new_mark + marks[position + 1:])
            label = f"{marks[position]}→{new_mark}"
        elif kind == "haraka_drop":
            characters[character_index] = old[0] + "".join(
                mark for mark in old[1:] if mark not in HARAKAT)
            protected = old[0] in PROTECTED_LETTERS
            label = f"-{old[0]}"
        else:  # shaddah_drop
            del characters[character_index]
            label = f"-{old[0]}"
        edited[chosen] = "".join(characters)

    return Planted(kind=kind, words=tuple(edited), word=chosen, label=label,
                   protected=protected, inserted=inserted, character=character)

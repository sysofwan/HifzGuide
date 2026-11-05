"""#267 step 2's text-only mapping: Imlaei letters, edit-distance alignment, site location.

The model writes Imlaei orthography, so its text is aligned to the Imlaei spelling of the
segment (``data/imlaei.json``, word by word), not to the realized phonetic reference. A truth
site names a carrier in that realized reference; mapping it onto the Imlaei spelling takes its
Uthmani word (the committed per-word offsets, or a rebuilt reference that must match exactly)
and then the letter the model's alignment puts there.

Nothing here reads audio or a model: a site is located from text alone, before any decode, so
the comparison set cannot be outcome-informed.

* :func:`letters_of` and :func:`glyphs` fold a text to its undiacritised letters (hamza seats
  to hamza) and keep each letter's shadda.
* :func:`align` maps each reference letter to a hypothesis index by edit distance.
* :func:`locate` maps a realized-reference index through its word to one Imlaei letter. A
  realized geminate (``تت``) collapses onto the word's one shadda letter; a repeated letter is
  chosen by occurrence order; a mark, a marker or a letter the word does not carry is refused.
* :func:`site_layout` / :func:`locate_site`, and :func:`locate_edit`, are the committed-site
  and edit entry points.
"""

from __future__ import annotations

import bisect
import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tadabur.truth_sites import CONSONANTS, HELD, NOT_HELD

IMLAEI_PATH = Path(__file__).resolve().parents[3] / "data" / "imlaei.json"
SHADDA = "\u0651"
#: The hamza seats: the Imlaei spelling and the model's text both get one hamza symbol.
_HAMZA_SEATS = str.maketrans({"أ": "ء", "إ": "ء", "آ": "ء", "ؤ": "ء", "ئ": "ء"})


@dataclass(frozen=True)
class Glyph:
    """One undiacritised letter of a word and whether it carries a shadda."""

    letter: str
    shadda: bool


@dataclass(frozen=True)
class Layout:
    """A segment's Imlaei words and where each starts in its realized reference."""

    words: tuple[str, ...]
    offsets: tuple[int, ...]
    reference: str

    @property
    def letters(self) -> str:
        """The segment's Imlaei letters, in order; the model's text aligns to this."""
        return segment_letters(self.words)


@dataclass(frozen=True)
class Located:
    """A located carrier: the segment's layout and the Imlaei letter index it maps to."""

    layout: Layout
    index: int

    @property
    def letters(self) -> str:
        return self.layout.letters

    @property
    def reference(self) -> str:
        return self.layout.reference


def _letter(char: str) -> str | None:
    """One character as an undiacritised letter (a hamza seat as hamza), or ``None`` for a
    mark, a pause sign, a digit, a space or any other non-letter."""
    if unicodedata.combining(char) or unicodedata.category(char)[0] != "L":
        return None
    return char.translate(_HAMZA_SEATS)


def letters_of(text: str) -> str:
    """The text's undiacritised letters, in order (``وَيَا قَوْمِ`` -> ``وياقوم``)."""
    return "".join(letter for char in text if (letter := _letter(char)) is not None)


def glyphs(text: str) -> tuple[Glyph, ...]:
    """The text's letters with their shaddas, read in either mark order (``تُّ`` and ``تَّ``
    are one ``ت`` with a shadda; ``دَدَ`` is two ``د`` without)."""
    out: list[tuple[str, bool]] = []
    for char in text:
        letter = _letter(char)
        if letter is not None:
            out.append((letter, False))
        elif char == SHADDA and out:
            out[-1] = (out[-1][0], True)
    return tuple(Glyph(letter, shadda) for letter, shadda in out)


def segment_letters(words: "list[str] | tuple[str, ...]") -> str:
    """The joined undiacritised letters of a segment's words (``["وَيَا قَوْمِ", "مَن"]`` ->
    ``وياقوممن``)."""
    return "".join(letters_of(word) for word in words)


def align_ops(reference: str, hypothesis: str) -> list[tuple[str, int | None, int | None]]:
    """The edit-distance alignment's operations, in order, insertions included.

    Each operation is ``(kind, reference index, hypothesis index)`` with ``kind`` one of
    ``match``, ``sub``, ``del`` (no hypothesis index) and ``ins`` (no reference index). A
    substitution is aligned (cost 1), a deletion has no hypothesis index, and a tie prefers the
    diagonal, so equal letters stay aligned.
    """
    reference, hypothesis = list(reference), list(hypothesis)
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            kind = "match" if reference[i - 1] == hypothesis[j - 1] else "sub"
            ops.append((kind, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    while i > 0:
        ops.append(("del", i - 1, None))
        i -= 1
    while j > 0:
        ops.append(("ins", None, j - 1))
        j -= 1
    ops.reverse()
    return ops


def align(reference: str, hypothesis: str) -> list[int | None]:
    """Each reference letter's hypothesis index under an edit-distance alignment.

    A substitution is aligned (cost 1), a deletion has no index (``None``), and an insertion
    shifts the indices after it. Ties prefer the diagonal, so equal letters stay aligned.
    """
    mapping: list[int | None] = [None] * len(reference)
    for _, position, index in align_ops(reference, hypothesis):
        if position is not None:
            mapping[position] = index
    return mapping


@lru_cache(maxsize=1)
def _imlaei() -> dict[str, dict]:
    return json.loads(IMLAEI_PATH.read_text(encoding="utf-8"))


def imlaei_words(surah_ayah: str, start: int, count: int) -> list[str]:
    """The segment's Imlaei words: ``count`` words from ``start`` (0-based), the ayah's
    number marker excluded. A range the ayah does not have is refused."""
    table = _imlaei()
    words: list[str] = []
    for word in range(1, 1000):
        entry = table.get(f"{surah_ayah}:{word}")
        if entry is None:
            break
        if letters_of(entry["text"]):
            words.append(entry["text"])
    if start < 0 or count < 0 or start + count > len(words):
        raise ValueError(f"{surah_ayah}: no words [{start}, {start + count}) in {len(words)}")
    return words[start:start + count]


def _consonant(char: str) -> str | None:
    """The carrier letter one realized-reference character stands for, if any."""
    letter = _letter(char)
    return letter if letter in CONSONANTS else None


def _collapsed_mapping(reference: list[tuple[str, int]],
                       hypothesis: tuple[Glyph, ...]) -> list[int | None]:
    """Map each reference letter to the Imlaei glyph aligned to it.

    A shadda glyph absorbs a realized geminate: consecutive equal reference letters all map to
    that one glyph (``تت`` -> the word's ``ت`` with a shadda), and a plain glyph keeps one
    reference letter each. The rest is :func:`align`'s edit distance, with the geminate move
    preferred on a tie so a realization that spells a geminate twice still lands on the one
    shadda letter.
    """
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(
                cost[i - 1][j - 1] + (reference[i - 1][0] != hypothesis[j - 1].letter),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )
            glyph = hypothesis[j - 1]
            if glyph.shadda and reference[i - 1][0] == glyph.letter:
                run = 1
                while i - run - 1 >= 0 and reference[i - run - 1][0] == glyph.letter:
                    run += 1
                for length in range(2, run + 1):
                    best = min(best, cost[i - length][j - 1])
            cost[i][j] = best
    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        glyph = hypothesis[j - 1]
        if glyph.shadda and reference[i - 1][0] == glyph.letter:
            run = 1
            while i - run - 1 >= 0 and reference[i - run - 1][0] == glyph.letter:
                run += 1
            collapsed = None
            for length in range(2, run + 1):
                if cost[i][j] == cost[i - length][j - 1]:
                    collapsed = length
            if collapsed is not None:
                for k in range(1, collapsed + 1):
                    mapping[i - k] = j - 1
                i, j = i - collapsed, j - 1
                continue
        if cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1][0] != glyph.letter):
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return mapping


def locate(words: "list[str] | tuple[str, ...]", offsets: "list[int] | tuple[int, ...]",
           reference: str, index: int) -> int | None:
    """The Imlaei letter index one realized-reference index maps to, or ``None``.

    ``offsets`` are the words' starts in ``reference`` (plus its end); ``index`` is the
    carrier's character position. A carrier that is no consonant, or one whose aligned Imlaei
    letter differs from it, is refused.
    """
    if index < 0 or index >= len(reference) or len(offsets) < 2:
        return None
    word = bisect.bisect_right(offsets, index) - 1
    if word < 0 or word + 1 >= len(offsets) or word >= len(words):
        return None
    carrier = _consonant(reference[index])
    if carrier is None:
        return None
    reference_letters = []
    for position in range(offsets[word], offsets[word + 1]):
        letter = _consonant(reference[position])
        if letter is not None:
            reference_letters.append((letter, position))
    positions = [position for _, position in reference_letters]
    if index not in positions:
        return None
    hypothesis = glyphs(words[word])
    mapping = _collapsed_mapping(reference_letters, hypothesis)
    glyph = mapping[positions.index(index)]
    if glyph is None or hypothesis[glyph].letter != carrier:
        return None
    return sum(len(glyphs(item)) for item in words[:word]) + glyph


#: One segment's realized reference, or ``None`` when the phonetizer refuses it.
def _segment_reference(words: tuple[str, ...]) -> tuple[str, tuple[int, ...]] | None:
    from tadabur.waqf_segments import hafs_segment_reference

    try:
        reference, offsets = hafs_segment_reference()(list(words))
    except (KeyError, IndexError, ValueError):
        return None
    return reference, tuple(offsets)


@lru_cache(maxsize=None)
def _whole_ayah(surah_ayah: str) -> tuple[tuple[str, ...], str, tuple[int, ...]] | None:
    """The ayah's Uthmani words and its realized reference with per-word offsets."""
    from tadabur.waqf_segments import _uthmani_words

    words = tuple(_uthmani_words(surah_ayah))
    built = _segment_reference(words)
    if built is None:
        return None
    return words, built[0], built[1]


@lru_cache(maxsize=None)
def _imlaei_layout(surah_ayah: str, start: int, count: int) -> Layout | None:
    from tadabur.waqf_segments import _uthmani_words

    built = _segment_reference(tuple(_uthmani_words(surah_ayah)[start:start + count]))
    if built is None:
        return None
    return Layout(tuple(imlaei_words(surah_ayah, start, count)), built[1], built[0])


#: A pausal final word's phonemes differ from its wasl slice by a few characters at most.
_PAUSAL_SLACK = 8


@lru_cache(maxsize=None)
def _derive_layout(surah_ayah: str, reference: str) -> Layout | None:
    """The word range whose rebuilt reference equals ``reference`` exactly, or ``None``.

    The whole ayah's phonetization supplies the word boundaries, so only a range whose words
    before the last one spell the reference's prefix, and whose slice is about as long as the
    reference, reaches a phonetizer call; a range whose slice already equals the reference is
    returned without one.
    """
    whole = _whole_ayah(surah_ayah)
    if whole is None:
        return None
    words, phonemes, offsets = whole
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            prefix = phonemes[offsets[start]:offsets[end - 1]]
            if not prefix or not reference.startswith(prefix):
                continue
            if reference == phonemes[offsets[start]:offsets[end]]:
                return Layout(tuple(imlaei_words(surah_ayah, start, end - start)),
                              tuple(offset - offsets[start] for offset in offsets[start:end + 1]),
                              reference)
            if abs(len(reference) - (offsets[end] - offsets[start])) > _PAUSAL_SLACK:
                continue
            layout = _imlaei_layout(surah_ayah, start, end - start)
            if layout is not None and layout.reference == reference:
                return layout
    return None


@lru_cache(maxsize=1)
def _stored_layouts() -> dict[str, dict]:
    path = Path(__file__).resolve().parent.parent / "ship_a" / "sites.jsonl"
    return {row["site_id"]: row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def site_layout(site) -> Layout | None:
    """The site's segment layout: the committed word offsets, else a rebuilt exact match."""
    stored = _stored_layouts().get(site.site_id)
    if stored is not None:
        if stored["reference"] != site.reference:
            return None
        count = len(stored["word_offsets"]) - 1
        return Layout(tuple(imlaei_words(site.surah_ayah, stored["word_start"], count)),
                      tuple(stored["word_offsets"]), site.reference)
    return _derive_layout(site.surah_ayah, site.reference)


def locate_site(site) -> Located | None:
    """The site's carrier in its segment's Imlaei letters, or ``None`` when unlocated."""
    layout = site_layout(site)
    if layout is None:
        return None
    index = locate(layout.words, layout.offsets, layout.reference, site.reference_index)
    return None if index is None else Located(layout, index)


def locate_edit(item: dict) -> Located | None:
    """The edit's carrier in its segment's Imlaei letters, or ``None`` when unlocated."""
    layout = _derive_layout(item["source"]["surah_ayah"], item["reference"])
    if layout is None:
        return None
    index = locate(layout.words, layout.offsets, layout.reference, item["reference_index"])
    return None if index is None else Located(layout, index)


def carried_at(letters: str, index: int, hypothesis: str, *, shaddah: bool) -> str | None:
    """What the hypothesis carries at Imlaei letter ``index``: its letter, or its gemination
    state when ``shaddah`` is set. ``None`` when nothing is aligned there, and on a shaddah
    site also when the aligned glyph is not the carrier (a shadda on another consonant is not
    this site's gemination state; the student rule requires the same consonant)."""
    hypothesis_letters = letters_of(hypothesis)
    aligned = align(letters, hypothesis_letters)[index]
    if aligned is None:
        return None
    if not shaddah:
        return hypothesis_letters[aligned]
    if hypothesis_letters[aligned] != letters[index]:
        return None
    return HELD if glyphs(hypothesis)[aligned].shadda else NOT_HELD

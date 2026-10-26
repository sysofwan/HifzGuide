"""Ayah identification for the tlog hold-out (#190): reference database, search, calibration.

The pre-registration's "Ayah identification" (``docs/tlog-holdout-preregistration.md``, frozen) as
amended by ``docs/tlog-ident-preregistration.md`` (S5, 2026-10-25, pre-outcome), which fills in the
procedures the frozen text leaves open.

**References.** Each ayah is phonetized on its own in its waqf form, made space-free, normalised
with :func:`tadabur.normalization.normalize_phonemes` (the scorer's canonical parity), and its
word offsets are remapped through the normalisation's ``offset_map``. The ayat are concatenated in
order per surah: 114 sequences whose spans are quran-transcript ``uthmani.split()`` words. Ayat with
an identical normalised reference share one **text class**, so a repeated ayah (55:13 and its 30
repeats) is never a disagreement. The 8 ayat the phonetizer cannot handle take their text from
``generate_phonemes.FALLBACK_PHONEMES``; their boundaries 0…n−2 come from phonetizing
``words[:-1]``, and the last word starts where an infix alignment of that prefix ends in the
fallback string.

**Decoys.** Each normalised ayah reference is reversed by grapheme cluster (a base plus its
combining marks), and the reversed ayat are concatenated in ayah order per surah.

**Search.** A consonant-skeleton 3-gram inverted index ranks the windows (each an ayah ± 1
neighbour) by the number of distinct query 3-grams they contain, ties by (surah, ayah) of the
centre ayah; the top k = 20 are scored with ``edlib`` 1.3.9.post1 in ``HW`` (infix) mode:
``score = 1 − distance / len(decode)``, the decode normalised first. The best location is a word
span; its ayah is the one with the most aligned words, and ``straddles`` says whether the span
crosses an ayah boundary. The best decoy score comes from the same shortlist over the decoy
sequences; the exhaustive decoy best is reported for sensitivity, never used.

**Thresholds.** Per decode-length bin the floor is the type-7 99th percentile of the best decoy
score over Tarteel v1 look decodes. ``b`` is the smallest grid value at which ≤ 0.5% of look clips
relabel and 0 of the 52 real-mistake clips relabel; ``b_u`` follows the owner-approved post-look
erratum (``docs/tlog-ident-bu-amendment.md``): the smallest bar at which ≥ 99% of the look clips
the identifier **names** are their own class. The literal share-of-all wording's result is
recorded as ``b_u_literal``; its coverage cannot rise with the bar. A length bin with no look
decode has no floor and no verdict. The bars and floors are committed in
``tools/tadabur/tlog/ident/ident_calibration.json`` with the generated page
``docs/tlog-ayah-identification.md``; every value is re-derived from the committed inputs by the
acceptance tests.

Committed inputs under ``tools/tadabur/tlog/ident/``: ``references.json.gz`` (the whole mushaf's
normalised references and offsets), ``real_mistake_clips.jsonl`` (the 52 clips holding the 56
owner-verified real-mistake sites), ``decodes/real_mistake.base.json`` and ``decodes/noise.base.json``
(the base teacher's whole-clip decodes), ``ident_calibration.json``.

Usage (from ``tools/``)::

  python -m tadabur.tlog_ident references    # build and commit references.json.gz (~5 min)
  python -m tadabur.tlog_ident calibrate     # thresholds, floors and the page
  python -m tadabur.tlog_ident report        # re-render the page from the committed report

The GPU decodes of the two calibration populations are produced by
``tadabur.tlog_ident_decode`` on cuda-dev; see ``scripts/issue-190-ident-decode.sh``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
import unicodedata
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import edlib

import generate_phonemes
import hafs_phonetizer

from training.tashkeel_eval import write_text_atomically

from . import normalization
from .normalization import map_char_offsets, normalize_phonemes
from .tlog_index import repo_relative, sha256_file
from .waqf_segments import spaceless_word_offsets

SCHEMA_VERSION = 1
#: The shortlist size the pre-registration fixes, and the gram size its S5 amendment adds.
K = 20
GRAM = 3
#: The bar grid: 0.00, 0.01, …, 0.50.
GRID: tuple[float, ...] = tuple(round(0.01 * index, 2) for index in range(51))
#: Decode-length bins (inclusive bounds, the last one open).
FLOOR_BINS = ("0-20", "21-40", "41-80", "81-160", "161-")
FLOOR_PERCENTILE = 99.0
#: The shortlist guard: seeded look clips and the recall it must reach.
RECALL_CLIPS = 1000
RECALL_MIN = 0.999
SALT = "issue-190-tlog-s5-v1"

BASE_MODEL = "obadx/muaalem-model-v3_2"
BASE_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"

#: ``label_source`` values.
FILENAME = "filename"
BOTH = "both"
IDENTIFIED = "identified"
NONE = "none"

IDENT_DIR = Path(__file__).parent / "tlog" / "ident"
REFERENCES_PATH = IDENT_DIR / "references.json.gz"
CALIBRATION_PATH = IDENT_DIR / "ident_calibration.json"
REAL_MISTAKE_CLIPS_PATH = IDENT_DIR / "real_mistake_clips.jsonl"
DECODES_DIR = IDENT_DIR / "decodes"
REAL_MISTAKE_DECODES_PATH = DECODES_DIR / "real_mistake.base.json"
NOISE_DECODES_PATH = DECODES_DIR / "noise.base.json"
TARTEEL_DIR = Path(__file__).parent / "nonpro_eval" / "tarteel_v1"
LOOK_DECODES_PATH = TARTEEL_DIR / "decodes.base.jsonl.gz"
SPLIT_PATH = TARTEEL_DIR / "split.json"
NOISE_POOL_PATH = Path(__file__).parent / "nonpro_gap" / "noise_pool.json"
PREREGISTRATION_PATH = (Path(__file__).resolve().parents[2]
                        / "docs" / "tlog-ident-preregistration.md")
BU_AMENDMENT_PATH = (Path(__file__).resolve().parents[2]
                     / "docs" / "tlog-ident-bu-amendment.md")
HOLDOUT_PREREGISTRATION_PATH = (Path(__file__).resolve().parents[2]
                                / "docs" / "tlog-holdout-preregistration.md")
DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "tlog-ayah-identification.md"

#: The alphabet the reference envelope records.
WORD_TOKENISATION = "quran-transcript uthmani.split"
#: The mode the Freesound pool is decoded in: a pool recording is too long for a whole pass.
NOISE_DECODE_MODE = "stream_b0"
LIBRARY = {"name": "edlib", "version": "1.3.9.post1", "mode": "HW"}
#: Relabel rate the look half must stay under, and the noise pass rate the pool must stay under.
RELABEL_LIMIT = 0.005
ASSIGNED_MIN = 0.99
LOOK_RETENTION_MIN = 0.90
NOISE_PASS_MAX = 0.01

#: Diacritics the consonant skeleton drops (normalised references hold none of them, but a raw
#: decode may).
_NON_CONSONANTS = frozenset("َُِ۪ڇۜـ")


def _split_label(label: str) -> tuple[int, int]:
    """``"surah:ayah"`` as two ints."""
    try:
        surah, ayah = (int(part) for part in label.split(":"))
    except ValueError as error:
        raise ValueError(f"{label!r} is not a surah:ayah label") from error
    return surah, ayah


def _normalise(text: str) -> str:
    """The scorer's normalisation, space-free: what both sides of every comparison use.

    Deliberately not idempotent: the normal form drops combining marks, so a doubled core the
    first pass produced from a shadda would collapse on a second pass. Every entry point
    therefore normalises the **raw** decode exactly once; the database's public methods take the
    raw decode and internal ones take the normalised query.
    """
    return normalize_phonemes(text).normalized.replace(" ", "")


def _skeleton(text: str) -> str:
    """A normalised string's consonant skeleton: its characters less the marks."""
    return "".join(char for char in text if char not in _NON_CONSONANTS)


# --- the data the search speaks ----------------------------------------------------------------


@dataclass(frozen=True)
class AyahReference:
    """One ayah's normalised, space-free reference and its ``len(words) + 1`` word offsets."""

    surah: int
    ayah: int
    reference: str
    offsets: tuple[int, ...]

    @property
    def words(self) -> int:
        return len(self.offsets) - 1


@dataclass(frozen=True)
class Match:
    """The best alignment of a decode to one window, as a word span."""

    surah: int
    ayah_from: int
    word_from: int
    ayah_to: int
    word_to: int
    text_class: int
    score: float

    @property
    def straddles(self) -> bool:
        return self.ayah_from != self.ayah_to


@dataclass(frozen=True)
class Window:
    """A scored window: its centre ayah and the alignment that won it."""

    surah: int
    ayah: int
    match: Match

    @property
    def score(self) -> float:
        return self.match.score


@dataclass(frozen=True)
class Identification:
    """One decode's identification: the span, its class, the decoy, Δ and the margin.

    ``class_label`` and ``filename_label`` are the label strings the class and the filename
    name; they let :func:`apply_floors` recompute the label fields without the database.
    """

    length: int
    match: Match | None
    decoy: float
    exhaustive_decoy: float | None
    filename_class: int | None
    filename_score: float | None
    delta: float | None
    margin: float | None
    margin_class: int | None
    floor_ok: bool | None
    overlaps_filename: bool | None
    label_source: str
    assigned_label: str | None
    class_label: str | None = None
    filename_label: str | None = None


@dataclass(frozen=True)
class Decode:
    """One committed Tarteel v1 look decode (its ``spans`` are the model's whole-clip output)."""

    key: str
    audio_filename: str
    surah_ayah: str
    word_from: int
    word_to: int
    cer_wav: float
    spans: str


@dataclass(frozen=True)
class _Constituent:
    """One ayah's slice of a window text."""

    surah: int
    ayah: int
    start: int
    end: int


@dataclass(frozen=True)
class _Candidate:
    """A window's text, its centre ayah and the ayat it is made of."""

    surah: int
    ayah: int
    text: str
    constituents: tuple[_Constituent, ...]


# --- scoring ----------------------------------------------------------------------------------


def score(decode: str, text: str) -> float:
    """``1 − infix_edit_distance(normalise(decode), text) / len(normalise(decode))``.

    ``text`` is a reference the database already normalised; the decode is normalised here.
    """
    query = _normalise(decode)
    if not query:
        raise ValueError("an empty decode has no score")
    return 1.0 - edlib.align(query, text, mode="HW")["editDistance"] / len(query)


def reverse_clusters(text: str) -> str:
    """``text`` with its grapheme clusters (a base plus trailing combining marks) reversed."""
    clusters: list[str] = []
    for char in text:
        if clusters and unicodedata.combining(char):
            clusters[-1] += char
        else:
            clusters.append(char)
    return "".join(reversed(clusters))


def _path(query: str, text: str) -> tuple[int, int, int, tuple[int, ...]]:
    """The best infix alignment: distance, target start (inclusive), end (exclusive), CIGAR.

    The CIGAR consumes the query whole; ``=``/``X`` align one target character, ``D`` consumes a
    target character with no query counterpart, ``I`` consumes a query character only.
    """
    result = edlib.align(query, text, mode="HW", task="path")
    start, end = result["locations"][0]
    if start is None:  # the empty query: no location
        return result["editDistance"], 0, 0, ()
    return result["editDistance"], int(start), int(end) + 1, _cigar_ops(result["cigar"])


def _cigar_ops(cigar: str | None) -> tuple[tuple[str, int], ...]:
    ops: list[tuple[str, int]] = []
    length = 0
    for char in cigar or "":
        if char.isdigit():
            length = length * 10 + int(char)
            continue
        ops.append((char, length))
        length = 0
    return tuple(ops)


# --- the reference database -------------------------------------------------------------------


class ReferenceDatabase:
    """114 real surah sequences, their decoys, the text classes and the 3-gram shortlists."""

    def __init__(self, ayat: Iterable[AyahReference], *, gram: int = GRAM) -> None:
        ordered = sorted(ayat, key=lambda reference: (reference.surah, reference.ayah))
        if not ordered:
            raise ValueError("a reference database needs at least one ayah")
        for reference in ordered:
            if reference.surah < 1 or reference.ayah < 1:
                raise ValueError(f"{reference.surah}:{reference.ayah} is not an ayah key")
            if len(reference.offsets) != reference.words + 1 or reference.offsets[0] != 0:
                raise ValueError(f"{reference.surah}:{reference.ayah} has malformed offsets")
        counts: dict[int, int] = defaultdict(int)
        for reference in ordered:
            counts[reference.surah] += 1
            if reference.ayah != counts[reference.surah]:
                raise ValueError(f"surah {reference.surah} is not contiguous at ayah "
                                 f"{reference.ayah}")
        if gram < 1:
            raise ValueError(f"gram must be positive, got {gram}")
        self.gram = gram
        self.ayat: dict[tuple[int, int], AyahReference] = {
            (reference.surah, reference.ayah): reference for reference in ordered}
        #: Window position of each ayah, so a per-window lookup is not a scan.
        self._position_of = {key: position for position, key in enumerate(self.ayat)}

        self._text: dict[int, str] = {}
        self._decoy: dict[int, str] = {}
        for surah in sorted(counts):
            self._text[surah] = "".join(
                reference.reference for (name, _), reference in self.ayat.items()
                if name == surah)
            self._decoy[surah] = "".join(
                reverse_clusters(reference.reference) for (name, _), reference in self.ayat.items()
                if name == surah)

        self._class_of: dict[tuple[int, int], int] = {}
        self._class_ayat: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._class_label: dict[int, str] = {}
        class_of_text: dict[str, int] = {}
        for key, reference in self.ayat.items():
            text_class = class_of_text.setdefault(reference.reference, len(class_of_text))
            self._class_of[key] = text_class
            self._class_ayat[text_class].append(key)
            self._class_label.setdefault(text_class, f"{key[0]}:{key[1]}")

        self._real = self._windows(decoy=False)
        self._decoys = self._windows(decoy=True)
        self._real_index = self._index(self._real)

    # --- construction ---

    @classmethod
    def build(cls, ayat: Iterable[AyahReference], *, gram: int = GRAM) -> ReferenceDatabase:
        return cls(ayat, gram=gram)

    def _windows(self, *, decoy: bool) -> tuple[_Candidate, ...]:
        windows: list[_Candidate] = []
        for (surah, ayah) in self.ayat:
            pieces: list[_Constituent] = []
            texts: list[str] = []
            cursor = 0
            for other in (ayah - 1, ayah, ayah + 1):
                neighbour = self.ayat.get((surah, other))
                if neighbour is None:
                    continue
                text = (reverse_clusters(neighbour.reference) if decoy
                        else neighbour.reference)
                pieces.append(_Constituent(surah, other, cursor, cursor + len(text)))
                texts.append(text)
                cursor += len(text)
            windows.append(_Candidate(surah, ayah, "".join(texts), tuple(pieces)))
        return tuple(windows)

    def _index(self, windows: Sequence[_Candidate]) -> dict[str, tuple[int, ...]]:
        postings: dict[str, list[int]] = defaultdict(list)
        for position, window in enumerate(windows):
            skeleton = _skeleton(window.text)
            for gram in {skeleton[index:index + self.gram]
                         for index in range(len(skeleton) - self.gram + 1)}:
                postings[gram].append(position)
        return {gram: tuple(found) for gram, found in postings.items()}

    # --- reading ---

    def text(self, surah: int) -> str:
        """The surah's concatenated real references."""
        return self._text[surah]

    def decoy(self, surah: int) -> str:
        """The surah's concatenated reversed references."""
        return self._decoy[surah]

    def window(self, surah: int, ayah: int) -> str:
        """The ayah ± 1 window's concatenated real references."""
        return self._candidate(surah, ayah, decoy=False).text

    def decoy_window(self, surah: int, ayah: int) -> str:
        """The ayah ± 1 window's concatenated reversed references."""
        return self._candidate(surah, ayah, decoy=True).text

    def _candidate(self, surah: int, ayah: int, *, decoy: bool) -> _Candidate:
        position = self._position_of.get((surah, ayah))
        if position is None:
            raise KeyError(f"{surah}:{ayah} is not an ayah of this database")
        return (self._decoys if decoy else self._real)[position]

    def text_class(self, surah: int, ayah: int) -> int:
        return self._class_of[(surah, ayah)]

    def class_ayat(self, text_class: int) -> tuple[tuple[int, int], ...]:
        return tuple(self._class_ayat[text_class])

    def class_label(self, text_class: int) -> str:
        return self._class_label[text_class]

    @property
    def class_count(self) -> int:
        return len(self._class_ayat)

    @property
    def surah_count(self) -> int:
        return len(self._text)

    def _ranked(self, query: str, *, k: int) -> tuple[int, ...]:
        """The shortlist as window positions, ranked by distinct shared 3-grams.

        A decode that hits no 3-gram has no signal to rank with, so the search is exhaustive
        for it — on the real side and, through the same positions, on the decoy side, which
        keeps the two comparable (the S5 amendment's target-decoy parity).
        """
        skeleton = _skeleton(query)
        grams = {skeleton[position:position + self.gram]
                 for position in range(len(skeleton) - self.gram + 1)}
        if not grams:
            return tuple(range(len(self._real)))[:k]
        counts: Counter[int] = Counter()
        for gram in grams:
            counts.update(self._real_index.get(gram, ()))
        ranked = sorted(counts, key=lambda position: (
            -counts[position], self._real[position].surah, self._real[position].ayah))
        if len(ranked) < k:  # the top k is filled with the windows no gram reached
            ranked += [position for position in range(len(self._real))
                       if position not in counts]
        return tuple(ranked[:k])

    def shortlist(self, decode: str, *, k: int = K) -> tuple[Window, ...]:
        """The k highest-ranked windows, aligned and ordered best score first."""
        query = _normalise(decode)
        if not query:
            return ()
        windows = [Window(candidate.surah, candidate.ayah, self._match(query, candidate))
                   for candidate in (self._real[position]
                                     for position in self._ranked(query, k=k))]
        return tuple(sorted(windows, key=lambda window: (
            -window.score, window.surah, window.ayah)))

    def all_windows(self, decode: str) -> tuple[Window, ...]:
        """Every window, aligned and ordered best score first."""
        query = _normalise(decode)
        if not query:
            return ()
        windows = [Window(candidate.surah, candidate.ayah, self._match(query, candidate))
                   for candidate in self._real]
        return tuple(sorted(windows, key=lambda window: (
            -window.score, window.surah, window.ayah)))

    def best(self, decode: str, *, k: int = K) -> Window | None:
        shortlisted = self.shortlist(decode, k=k)
        return shortlisted[0] if shortlisted else None

    def best_score(self, decode: str, *, k: int = K) -> float:
        best = self.best(decode, k=k)
        return best.score if best is not None else 0.0

    def exhaustive_best_score(self, decode: str) -> float:
        windows = self.all_windows(decode)
        return windows[0].score if windows else 0.0

    def decoy_best(self, decode: str, *, k: int = K) -> float:
        """The best score over the decoy text of the windows the real search shortlisted.

        The candidate pool is the real shortlist's, which is what makes the floor comparable
        with the score a non-recitation row is tested against (S5 amendment); only the text
        differs, and it is each window's reversed references.
        """
        query = _normalise(decode)
        if not query:
            return 0.0
        return max(1.0 - edlib.align(query, self._decoys[position].text,
                                     mode="HW")["editDistance"] / len(query)
                   for position in self._ranked(query, k=k))

    def decoy_exhaustive_best(self, decode: str) -> float:
        """The best score over every decoy window; reported, never used."""
        query = _normalise(decode)
        if not query:
            return 0.0
        return max(1.0 - edlib.align(query, candidate.text, mode="HW")["editDistance"]
                   / len(query) for candidate in self._decoys)

    def score_window(self, decode: str, surah: int, ayah: int) -> float:
        """The best score of a normalised decode against one ayah ± 1 window."""
        query = _normalise(decode)
        if not query:
            raise ValueError("an empty decode has no score")
        return 1.0 - edlib.align(query, self.window(surah, ayah),
                                 mode="HW")["editDistance"] / len(query)

    def _match(self, query: str, candidate: _Candidate) -> Match:
        distance, start, end, ops = _path(query, candidate.text)
        #: Per constituent, the words touched by a matched or substituted character.
        touched: list[set[int]] = [set() for _ in candidate.constituents]
        position = start
        for operation, length in ops:
            if operation in ("=", "X"):
                # Split the run at ayah boundaries. A run that crosses one belongs to each
                # ayah it touches, not whole to the ayah it starts in.
                run_start = position
                cursor = run_start
                while cursor < run_start + length:
                    index = _constituent_at(candidate.constituents, cursor)
                    constituent = candidate.constituents[index]
                    stop = min(run_start + length, constituent.end)
                    reference = self.ayat[(constituent.surah, constituent.ayah)]
                    touched[index].update(range(
                        _word_at(reference, constituent.start, cursor),
                        _word_at(reference, constituent.start, stop - 1) + 1))
                    cursor = stop
            if operation != "I":
                position += length
        chosen = _chosen_constituent(candidate.constituents, touched, start)
        first = candidate.constituents[_constituent_at(candidate.constituents, start)]
        last = candidate.constituents[_constituent_at(candidate.constituents, max(start,
                                                                                end - 1))]
        word_from = _word_at(self.ayat[(first.surah, first.ayah)], first.start, start)
        word_to = _word_at(self.ayat[(last.surah, last.ayah)], last.start, max(start, end - 1))
        return Match(surah=first.surah, ayah_from=first.ayah, word_from=word_from,
                     ayah_to=last.ayah, word_to=word_to,
                     text_class=self._class_of[(chosen.surah, chosen.ayah)],
                     score=1.0 - distance / len(query))


def _constituent_at(constituents: Sequence[_Constituent], position: int) -> int:
    for index, constituent in enumerate(constituents):
        if constituent.start <= position < constituent.end:
            return index
    return max(0, len(constituents) - 1)


def _chosen_constituent(constituents: Sequence[_Constituent], touched: Sequence[set[int]],
                        start: int) -> _Constituent:
    """The ayah with the most aligned words; ties go to the earlier ayah."""
    best = max(range(len(constituents)), key=lambda index: (len(touched[index]), -index))
    if not touched[best]:  # nothing aligned to a character: fall back on the location
        return constituents[_constituent_at(constituents, start)]
    return constituents[best]


def _word_at(reference: AyahReference, offset: int, position: int) -> int:
    """The 1-based word of ``reference`` that contains ``position`` (clamped to the ayah)."""
    local = min(max(position - offset, 0), len(reference.reference) - 1)
    word = bisect_right(reference.offsets, local) - 1
    return min(max(word, 0), reference.words - 1) + 1


def _overlaps(first: Match, second: Match) -> bool:
    if first.surah != second.surah:
        return False
    first_from = (first.ayah_from, first.word_from)
    first_to = (first.ayah_to, first.word_to)
    second_from = (second.ayah_from, second.word_from)
    second_to = (second.ayah_to, second.word_to)
    return first_from <= second_to and second_from <= first_to


def _overlaps_ayah(match: Match, key: tuple[int, int]) -> bool:
    """Whether the span holds at least one word of the ayah."""
    surah, ayah = key
    return match.surah == surah and match.ayah_from <= ayah <= match.ayah_to


# --- references: building, committing, loading ------------------------------------------------


def _spaceless_phonetization(words: Sequence[str]) -> tuple[str, list[int]]:
    """The words phonetized as one utterance: space-free text and its per-word offsets."""
    out = hafs_phonetizer.phonetize(" ".join(words))
    return spaceless_word_offsets(out.phonemes, out.mappings, list(words))


def _fallback_offsets(key: str, words: Sequence[str]) -> tuple[str, list[int]]:
    """The fallback string and the word boundaries the S5 amendment derives for it."""
    raw = generate_phonemes.FALLBACK_PHONEMES[key].replace(" ", "")
    prefix_words = list(words[:-1])
    if not prefix_words:
        return raw, [0, len(raw)]
    prefix, offsets = _spaceless_phonetization(prefix_words)
    _distance, _start, end, _ops = _path(prefix, raw)
    last_start = max(end, 1)
    boundaries = [*offsets[:-1], min(last_start, len(raw) - 1), len(raw)]
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1] + 1)
    boundaries[-1] = len(raw)
    return raw, boundaries


def ayah_reference(surah: int, ayah: int) -> AyahReference:
    """One ayah's reference: phonetized in waqf form, normalised, with remapped word offsets."""
    from quran_transcript import Aya

    words = Aya(surah, ayah).get().uthmani.split()
    key = f"{surah}:{ayah}"
    try:
        raw, offsets = _spaceless_phonetization(words)
    except (KeyError, IndexError):
        if key not in generate_phonemes.FALLBACK_PHONEMES:
            raise
        raw, offsets = _fallback_offsets(key, words)
    normalised = normalize_phonemes(raw)
    mapped = map_char_offsets(raw, normalised, offsets)
    return AyahReference(surah, ayah, normalised.normalized, tuple(mapped))


def build_references() -> ReferenceDatabase:
    """The whole mushaf's references, all 6,236 ayat in order."""
    return ReferenceDatabase.build([
        ayah_reference(surah, ayah)
        for surah, count in enumerate(generate_phonemes.HAFS_AYAH_COUNTS, start=1)
        for ayah in range(1, count + 1)])


def references_payload(database: ReferenceDatabase) -> dict:
    return {
        "schema": "tadabur.tlog_ident/references-v1",
        "phonetizer_revision": hafs_phonetizer.REVISION,
        "normalization_algorithm_version": normalization.ALGORITHM_VERSION,
        "word_tokenisation": WORD_TOKENISATION,
        "ayat": len(database.ayat),
        "classes": database.class_count,
        "surahs": database.surah_count,
        "items": [{"surah": reference.surah, "ayah": reference.ayah,
                   "reference": reference.reference, "offsets": list(reference.offsets)}
                  for reference in database.ayat.values()],
    }


def write_references(path: Path = REFERENCES_PATH,
                     database: ReferenceDatabase | None = None) -> dict:
    """Build (unless given) and write the reference database; return its envelope.

    The bytes are deterministic: gzipped without a timestamp (as ``tlog_index.write_shard``
    does), so a rebuild reproduces the file byte for byte on one interpreter.
    """
    payload = references_payload(database or build_references())
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(gzip.compress(data, mtime=0))
    os.replace(temporary, path)
    return payload


def load_references(path: Path = REFERENCES_PATH) -> ReferenceDatabase:
    """The committed reference database, refused unless it is this code's build."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "phonetizer_revision": hafs_phonetizer.REVISION,
        "normalization_algorithm_version": normalization.ALGORITHM_VERSION,
        "word_tokenisation": WORD_TOKENISATION,
        "ayat": sum(generate_phonemes.HAFS_AYAH_COUNTS),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path} records {key}={payload.get(key)!r}, not {value!r}")
    ayat = [AyahReference(item["surah"], item["ayah"], item["reference"],
                          tuple(item["offsets"])) for item in payload["items"]]
    database = ReferenceDatabase.build(ayat)
    if database.class_count != payload.get("classes"):
        raise ValueError(f"{path} records {payload.get('classes')} classes, "
                         f"the build has {database.class_count}")
    return database


# --- identification ---------------------------------------------------------------------------


def identify(decode: str, *, database: ReferenceDatabase, filename_ayah: str | None = None,
             floors: Mapping[str, float] | None = None,
             bars: Mapping[str, float] | None = None, k: int = K,
             exhaustive_decoy: bool = False) -> Identification:
    """One decode's identification (module docstring).

    ``floors`` maps a length bin to its floor; without it ``floor_ok`` is ``None`` rather than an
    assumption. ``bars`` carries ``b`` and ``b_u``; without them no row is assigned or relabelled.
    """
    name = _split_label(filename_ayah) if filename_ayah else None
    if name is not None and name not in database.ayat:
        raise ValueError(f"{filename_ayah!r} is not an ayah of this database")
    query = _normalise(decode)
    filename_class = database.text_class(*name) if name is not None else None
    filename_label = f"{name[0]}:{name[1]}" if name is not None else None

    if not query:
        identification = Identification(
            length=0, match=None, decoy=0.0,
            exhaustive_decoy=0.0 if exhaustive_decoy else None,
            filename_class=filename_class, filename_score=None, delta=None, margin=None,
            margin_class=None, floor_ok=False if floors is not None else None,
            overlaps_filename=None, label_source=NONE, assigned_label=None,
            filename_label=filename_label)
        return _with_label(identification, bars)

    shortlisted = database.shortlist(decode, k=k)
    best = shortlisted[0].match
    competitor = None
    for window in shortlisted[1:]:
        if window.match.text_class != best.text_class and not _overlaps(window.match, best):
            competitor = window.match
            break
    filename_score = (database.score_window(decode, *name) if name is not None else None)
    identification = Identification(
        length=len(query), match=best, decoy=database.decoy_best(decode, k=k),
        exhaustive_decoy=(database.decoy_exhaustive_best(decode)
                          if exhaustive_decoy else None),
        filename_class=filename_class, filename_score=filename_score,
        delta=(best.score - filename_score) if filename_score is not None else None,
        # With no different class at a non-overlapping location nothing contradicts the
        # match, so the margin is measured against the empty competitor (score 0).
        margin=best.score - (competitor.score if competitor is not None else 0.0),
        margin_class=competitor.text_class if competitor is not None else None,
        floor_ok=_floor_verdict(best.score, len(query), floors),
        overlaps_filename=(_overlaps_ayah(best, name) if name is not None else None),
        label_source=NONE, assigned_label=None,
        class_label=database.class_label(best.text_class),
        filename_label=filename_label)
    return _with_label(identification, bars)


def apply_floors(identification: Identification, floors: Mapping[str, float],
                 *, bars: Mapping[str, float] | None = None) -> Identification:
    """``identification`` with the floor verdict for its length, and the label it implies."""
    floor_ok = (False if identification.match is None else
                _floor_verdict(identification.match.score, identification.length, floors))
    return _with_label(replace(identification, floor_ok=floor_ok), bars)


def _with_label(identification: Identification,
                bars: Mapping[str, float] | None) -> Identification:
    """The label rule: ``both``/``identified``/``filename``/``none`` (pre-registration table)."""
    match = identification.match
    if match is None or identification.floor_ok is False:
        if identification.filename_label is not None:
            return replace(identification, label_source=FILENAME,
                           assigned_label=identification.filename_label)
        return replace(identification, label_source=NONE, assigned_label=None)
    if identification.filename_label is not None:
        if match.text_class == identification.filename_class:
            return replace(identification, label_source=BOTH,
                           assigned_label=identification.filename_label)
        if (bars is not None and bars.get("b") is not None
                and identification.delta is not None
                and identification.delta >= bars["b"]
                and not identification.overlaps_filename):
            return replace(identification, label_source=IDENTIFIED,
                           assigned_label=identification.class_label)
        return replace(identification, label_source=FILENAME,
                       assigned_label=identification.filename_label)
    if (bars is not None and bars.get("b_u") is not None
            and identification.margin is not None
            and identification.margin >= bars["b_u"]):
        return replace(identification, label_source=IDENTIFIED,
                       assigned_label=identification.class_label)
    return replace(identification, label_source=NONE, assigned_label=None)


def relabelled(identification: Identification, bar: float) -> bool:
    """Whether the relabel rule fires at ``bar``; the same class is never a disagreement."""
    return (identification.match is not None and identification.floor_ok is not False
            and identification.delta is not None and identification.delta >= bar
            and not identification.overlaps_filename
            and identification.match.text_class != identification.filename_class)


def assigned(identification: Identification, bar: float) -> bool:
    """Whether the assign rule fires at ``bar`` (labels hidden)."""
    return (identification.match is not None and identification.floor_ok is not False
            and identification.margin is not None and identification.margin >= bar)


# --- the rules over populations ---------------------------------------------------------------


def floor_bin(length: int) -> str:
    """The decode-length bin of a normalised decode."""
    if length <= 20:
        return FLOOR_BINS[0]
    if length <= 40:
        return FLOOR_BINS[1]
    if length <= 80:
        return FLOOR_BINS[2]
    if length <= 160:
        return FLOOR_BINS[3]
    return FLOOR_BINS[4]


def _floor_verdict(score: float, length: int,
                   floors: Mapping[str, float] | None) -> bool | None:
    """The bin's floor verdict; a bin with no floor (no look decode in it) has none."""
    if floors is None:
        return None
    floor = floors.get(floor_bin(length))
    return None if floor is None else score > floor


def _backoff_floor(bin_: str, floors: Mapping[str, float]) -> float | None:
    """The nearest present floor below ``bin_``, else the nearest present one above it."""
    index = FLOOR_BINS.index(bin_)
    for candidate in reversed(FLOOR_BINS[:index]):
        if candidate in floors:
            return floors[candidate]
    for candidate in FLOOR_BINS[index + 1:]:
        if candidate in floors:
            return floors[candidate]
    return None


def noise_pass_by_convention(rows: Sequence[Identification],
                             floors: Mapping[str, float]) -> dict:
    """The Freesound pass rate under the three empty-bin conventions.

    A: a row in a bin with no floor never passes. B: it is judged against the nearest
    calibrated bin's floor. C: it is excluded from the denominator. A row with no match never
    passes and stays in A's and B's denominator.
    """
    counts = {key: {"recordings": 0, "passed": 0} for key in ("A", "B", "C")}
    for row in rows:
        bin_ = floor_bin(row.length)
        floor = floors.get(bin_)
        passed = row.match is not None and floor is not None and row.match.score > floor
        counts["A"]["recordings"] += 1
        counts["A"]["passed"] += passed
        counts["B"]["recordings"] += 1
        if floor is None:
            backed = _backoff_floor(bin_, floors)
            counts["B"]["passed"] += (row.match is not None and backed is not None
                                      and row.match.score > backed)
        else:
            counts["B"]["passed"] += passed
        if floor is not None:
            counts["C"]["recordings"] += 1
            counts["C"]["passed"] += passed
    return {key: {**count, "pass": (count["passed"] / count["recordings"]
                                    if count["recordings"] else None)}
            for key, count in counts.items()}


def _quantile(values: Sequence[float], q: float) -> float:
    from training.acceptance_stats import quantile_type7

    return float(quantile_type7(list(values), q))


def derive_floors(identifications: Iterable[Identification]) -> dict[str, float]:
    """The type-7 ``FLOOR_PERCENTILE`` of the best decoy score, per present length bin."""
    by_bin: dict[str, list[float]] = defaultdict(list)
    for identification in identifications:
        by_bin[floor_bin(identification.length)].append(identification.decoy)
    return {bin_: round(_quantile(values, FLOOR_PERCENTILE / 100), 6)
            for bin_, values in sorted(by_bin.items(), key=lambda item: FLOOR_BINS.index(
                item[0]))}


def derive_bars(look: Sequence[Identification], mistake: Sequence[Identification],
                *, grid: Sequence[float] = GRID) -> dict:
    """The pre-registered ``b`` and the owner-approved amended ``b_u``.

    ``b`` is the smallest grid value at which at most ``RELABEL_LIMIT`` of look relabels and no
    real-mistake clip relabels. ``b_u`` (``docs/tlog-ident-bu-amendment.md``) is the smallest
    grid value at which at least ``ASSIGNED_MIN`` of the look clips the identifier **names**
    are their own class. The literal wording's share-of-all coverage is recorded as
    ``b_u_literal`` for transparency; that coverage can only fall as the bar rises.
    """
    if not look:
        raise ValueError("no look identification: the bars are undefined")
    labels = ("grid", "b", "b_u", "look_assigned_at_b_u", "b_u_literal",
              "look_assigned_at_b_u_literal", "look_relabel_at_b",
              "mistake_relabels_at_b", "no_bar", "precision")
    result = dict.fromkeys(labels)
    result["grid"] = list(grid)
    result["no_bar"] = True
    for value in grid:
        relabels = sum(relabelled(row, value) for row in look)
        rate = relabels / len(look)
        if rate <= RELABEL_LIMIT and sum(relabelled(row, value) for row in mistake) == 0:
            result.update(b=value, look_relabel_at_b=rate,
                          mistake_relabels_at_b=0, no_bar=False)
            break
    by_bar = []
    b_u = assigned_at = None
    literal = literal_coverage = None
    for value in grid:
        named = [row for row in look if assigned(row, value)]
        own = sum(row.match.text_class == row.filename_class for row in named)
        precision = own / len(named) if named else None
        coverage = own / len(look)
        by_bar.append({"bar": value, "assigned": len(named),
                       "precision": precision, "coverage": coverage})
        if b_u is None and precision is not None and precision >= ASSIGNED_MIN:
            b_u, assigned_at = value, precision
        if literal is None and coverage >= ASSIGNED_MIN:
            literal, literal_coverage = value, coverage
    result.update(b_u=b_u, look_assigned_at_b_u=assigned_at,
                  b_u_literal=literal, look_assigned_at_b_u_literal=literal_coverage,
                  precision={"used_for_b_u": True, "by_bar": by_bar})
    return result


# --- the committed populations ----------------------------------------------------------------


def recall_sample(clips: Iterable[str]) -> tuple[str, ...]:
    """The seeded look clips the shortlist guard scores (SALT), first ``RECALL_CLIPS`` by rank."""
    ranked = sorted(clips, key=lambda clip: hashlib.sha256(
        f"{SALT}:{clip}".encode("utf-8")).digest())
    return tuple(ranked[:RECALL_CLIPS])


def load_look_decodes(path: Path = LOOK_DECODES_PATH,
                      split_path: Path = SPLIT_PATH) -> list[Decode]:
    """The Tarteel v1 **look** decodes, and only them: the calibration's population."""
    look = set(json.loads(Path(split_path).read_text(encoding="utf-8"))["look"])
    decodes: list[Decode] = []
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["audio_filename"] not in look:
                continue
            decodes.append(Decode(row["key"], row["audio_filename"], row["surah_ayah"],
                                  int(row["word_from"]), int(row["word_to"]),
                                  float(row["labels"]["cer_wav"]), row["spans"]))
    return decodes


def load_decode_store(path: Path = REAL_MISTAKE_DECODES_PATH) -> dict:
    """One committed whole-clip decode store (identity, fingerprints, items)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def real_mistake_clips(path: Path = REAL_MISTAKE_CLIPS_PATH) -> list[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines()]


# --- the calibration --------------------------------------------------------------------------


def _look_measurements(database: ReferenceDatabase, look: Sequence[Decode]) -> tuple[list, float]:
    """Score every look decode, timed; returns the raw identifications and the elapsed seconds."""
    started = time.time()
    scored = [identify(decode.spans, database=database, filename_ayah=decode.surah_ayah)
              for decode in look]
    return scored, time.time() - started


def _shortlist_guard(database: ReferenceDatabase, sample: Sequence[str],
                     decodes: Mapping[str, Decode], *, k: int = K) -> dict:
    """The seeded guard: a clip is recalled when the shortlist's best equals the exhaustive best."""
    recalled = 0
    for clip in sample:
        spans = decodes[clip].spans
        if database.best_score(spans, k=k) == database.exhaustive_best_score(spans):
            recalled += 1
    return {"clips": len(sample), "salt": SALT, "recalled": recalled,
            "recall": recalled / len(sample), "k": k,
            "passed": recalled / len(sample) >= RECALL_MIN}


def _calibration(database: ReferenceDatabase) -> dict:
    look = load_look_decodes()
    by_clip = {decode.audio_filename: decode for decode in look}
    raw, seconds = _look_measurements(database, look)
    floors = derive_floors(raw)
    applied = [apply_floors(row, floors) for row in raw]

    clip_list = real_mistake_clips()
    truth = load_decode_store(REAL_MISTAKE_DECODES_PATH)
    mistakes = [apply_floors(identify(truth["items"][clip["audio_filename"]]["spans"],
                                      database=database, filename_ayah=clip["surah_ayah"]),
                             floors) for clip in clip_list]
    bars = derive_bars(applied, mistakes)

    sample = recall_sample(by_clip)
    guard = _shortlist_guard(database, sample, by_clip)
    while not guard["passed"]:
        guard = _shortlist_guard(database, sample, by_clip, k=guard["k"] * 2)

    exhaustive = [identify(decode.spans, database=database, filename_ayah=decode.surah_ayah,
                           exhaustive_decoy=True) for decode in look]
    exhaustive_floors = derive_floors(replace(row, decoy=row.exhaustive_decoy)
                                      for row in exhaustive) if exhaustive else {}
    exhaustive_applied = [apply_floors(row, exhaustive_floors) for row in exhaustive]
    exhaustive_retention = sum(row.floor_ok for row in exhaustive_applied) / len(
        exhaustive_applied)

    pool = json.loads(NOISE_POOL_PATH.read_text(encoding="utf-8"))["recordings"]
    noise_store = load_decode_store(NOISE_DECODES_PATH)
    noise = [identify(item["spans"], database=database)
             for item in noise_store["items"].values()]
    conventions = noise_pass_by_convention(noise, floors)
    # B (back off to the nearest calibrated bin) is the headline: the most faithful reading of
    # the binned rule. A and C are reported beside it (architect r16).
    noise_pass = conventions["B"]["pass"]
    floor_status = (f"stopped: noise {100 * noise_pass:.1f}% > {100 * NOISE_PASS_MAX:g}%"
                    if noise_pass > NOISE_PASS_MAX else "passed")
    bars["provisional"] = True
    bars["provisional_reason"] = floor_status

    from .tlog_overlap import iter_pool_rows  # heavy import, calibration only

    unlabelled = sum(1 for row in iter_pool_rows()
                     if row["split"] == "unclean" and row["filename_label"] is None
                     and row["usable"])

    retention = sum(row.floor_ok for row in applied) / len(applied)
    relabel_rate = bars["look_relabel_at_b"]
    tail = [(decode, row) for decode, row in zip(look, applied) if decode.cer_wav >= 0.2]
    tail_retention = sum(row.floor_ok for _, row in tail) / len(tail) if tail else None
    tail_relabel = (sum(relabelled(row, bars["b"]) for _, row in tail) / len(tail)
                    if tail and bars["b"] is not None else None)

    stop = {
        "look_retention": {"limit": LOOK_RETENTION_MIN, "value": retention,
                           "fired": retention < LOOK_RETENTION_MIN},
        "noise_pass": {"limit": NOISE_PASS_MAX, "value": noise_pass,
                       "fired": noise_pass > NOISE_PASS_MAX},
        "bar": {"limit": None, "value": bars["b"], "fired": bars["b"] is None},
    }
    return {
        "schema": "tadabur.tlog_ident/calibration-v1",
        "issue": 190,
        "preregistration": {
            "path": repo_relative(PREREGISTRATION_PATH),
            "sha256": sha256_file(PREREGISTRATION_PATH),
            "parent": {"path": repo_relative(HOLDOUT_PREREGISTRATION_PATH),
                       "sha256": sha256_file(HOLDOUT_PREREGISTRATION_PATH)},
        },
        "bu_amendment": {"path": repo_relative(BU_AMENDMENT_PATH),
                         "sha256": sha256_file(BU_AMENDMENT_PATH)},
        "floor_status": floor_status,
        "first_pass": {
            "engine_commit": "79acded",
            "defect": "each CIGAR run was credited whole to the ayah it started in, and "
                      "aligned characters were counted rather than words",
            "fix": "runs are split at ayah boundaries and the ayah with the most aligned "
                   "words is credited",
            "rules_changed": False,
        },
        "parameters": {
            "k": K, "gram": GRAM, "grid": list(GRID), "floor_bins": list(FLOOR_BINS),
            "floor_percentile": FLOOR_PERCENTILE,
            "library": LIBRARY,
            "normalization_algorithm_version": normalization.ALGORITHM_VERSION,
            "word_tokenisation": WORD_TOKENISATION,
            "phonetizer_revision": hafs_phonetizer.REVISION,
            "recall": {"clips": RECALL_CLIPS, "salt": SALT, "minimum": RECALL_MIN},
        },
        "reference": {"path": repo_relative(REFERENCES_PATH),
                      "sha256": sha256_file(REFERENCES_PATH), "ayat": len(database.ayat),
                      "classes": database.class_count, "surahs": database.surah_count},
        "decodes": {
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "look": {"path": repo_relative(LOOK_DECODES_PATH),
                     "sha256": sha256_file(LOOK_DECODES_PATH), "clips": len(look)},
            "real_mistake": {"path": repo_relative(REAL_MISTAKE_DECODES_PATH),
                             "sha256": sha256_file(REAL_MISTAKE_DECODES_PATH),
                             "clips": len(clip_list)},
            "noise": {"path": repo_relative(NOISE_DECODES_PATH),
                      "sha256": sha256_file(NOISE_DECODES_PATH), "recordings": len(noise),
                      "mode": NOISE_DECODE_MODE},
        },
        "real_mistake": {"path": repo_relative(REAL_MISTAKE_CLIPS_PATH),
                         "sha256": sha256_file(REAL_MISTAKE_CLIPS_PATH),
                         "clips": len(clip_list),
                         "sites": sum(len(clip["sites"]) for clip in clip_list)},
        "split": {"path": repo_relative(SPLIT_PATH), "sha256": sha256_file(SPLIT_PATH),
                  "look": len(look)},
        "shortlist_guard": guard,
        "floors": floors,
        "floors_source": {"percentile": {"method": "type-7",
                                         "q": FLOOR_PERCENTILE / 100},
                          "bin_counts": {bin_: sum(1 for row in raw if floor_bin(row.length)
                                                   == bin_) for bin_ in floors}},
        "exhaustive_decoy": {"floors": exhaustive_floors,
                             "look_retention": exhaustive_retention},
        "look": {"clips": len(applied), "retention": retention,
                 "relabel_rate_at_b": relabel_rate,
                 "bins": {bin_: sum(1 for row in applied if floor_bin(row.length) == bin_)
                          for bin_ in FLOOR_BINS},
                 "tail": {"clips": len(tail), "retention": tail_retention,
                          "relabel_rate": tail_relabel}},
        "mistake": {"clips": len(mistakes),
                    "relabelled_at_b": sum(relabelled(row, bars["b"])
                                           for row in mistakes)
                    if bars["b"] is not None else None},
        "unlabelled_unclean": {"source": "tadabur.tlog_overlap.iter_pool_rows",
                               "usable": unlabelled},
        "noise": {"recordings": len(noise), "pass": noise_pass,
                  "floor_passed": conventions["B"]["passed"],
                  "conventions": conventions},
        "bars": bars,
        "stop_conditions": stop,
        "throughput": {"clips": len(look), "seconds": round(seconds, 3),
                       "ms_per_clip": round(1000 * seconds / len(look), 3)},
        "pool": {"path": repo_relative(NOISE_POOL_PATH),
                 "sha256": sha256_file(NOISE_POOL_PATH), "recordings": len(pool)},
    }


def _pct(value: float | None) -> str:
    return "n/c" if value is None else f"{100 * value:.2f}%"


def _bar(value: float | None) -> str:
    return "none" if value is None else f"{value:.2f}"


def render(report: Mapping[str, Any]) -> str:
    """The generated page; every number is read from the report."""
    parameters, look, bars = report["parameters"], report["look"], report["bars"]
    stop = report["stop_conditions"]
    lines = [
        "# Ayah identification on Tarteel v1 look (#190)",
        "",
        "Generated by `python -m tadabur.tlog_ident calibrate` (from `tools/`); do not edit by "
        "hand. The machine-readable report is "
        "[`tools/tadabur/tlog/ident/ident_calibration.json`]"
        "(../tools/tadabur/tlog/ident/ident_calibration.json).",
        "",
        "> **Base teacher only.** No C95, student or candidate decode enters this calibration. "
        "No tlog audio was read.",
        "",
        "## Rules",
        "",
        f"Frozen by [`docs/tlog-holdout-preregistration.md`]"
        f"(tlog-holdout-preregistration.md) (sha256 "
        f"`{report['preregistration']['parent']['sha256'][:12]}`), with the S5 pre-outcome "
        f"amendment in [`docs/tlog-ident-preregistration.md`]"
        f"(tlog-ident-preregistration.md) (sha256 "
        f"`{report['preregistration']['sha256'][:12]}`), and the owner-approved post-look b_u "
        f"erratum in [`docs/tlog-ident-bu-amendment.md`]"
        f"(tlog-ident-bu-amendment.md) (sha256 "
        f"`{report['bu_amendment']['sha256'][:12]}`).",
        "",
        f"- alphabet: `normalize_phonemes` {parameters['normalization_algorithm_version']}, "
        f"words `{parameters['word_tokenisation']}`",
        f"- shortlist: consonant-skeleton {parameters['gram']}-grams, k = {parameters['k']}, "
        f"library `{parameters['library']['name']}` {parameters['library']['version']} "
        f"({parameters['library']['mode']})",
        f"- floors: type-7 {parameters['floor_percentile']}th percentile of the best decoy "
        f"score per length bin",
        "",
        "## Superseded first pass",
        "",
        f"The first look pass (engine `{report['first_pass']['engine_commit']}`) had a "
        f"span-attribution defect: {report['first_pass']['defect']}. Before this report the "
        f"engine was brought to the frozen rule: {report['first_pass']['fix']}. No rule, grid "
        "value or input changed, and the same committed look decodes were rescored; only the "
        "numbers below count.",
        "",
        "## Inputs",
        "",
        "| | |",
        "|---|---|",
        f"| references | {report['reference']['ayat']:,} ayat, "
        f"{report['reference']['classes']:,} text classes |",
        f"| Tarteel v1 look | {report['decodes']['look']['clips']:,} clips |",
        f"| real-mistake clips | {report['real_mistake']['clips']} clips over "
        f"{report['real_mistake']['sites']} sites |",
        f"| Freesound pool | {report['noise']['recordings']} recordings, "
        f"`{report['decodes']['noise']['mode']}` |",
        f"| b_u reach (unclean, no filename label, usable) | "
        f"{report['unlabelled_unclean']['usable']:,} rows |",
        "",
        "## Shortlist guard",
        "",
        f"{report['shortlist_guard']['recalled']} of {report['shortlist_guard']['clips']} "
        f"seeded look clips recalled ({_pct(report['shortlist_guard']['recall'])}) at "
        f"k = {report['shortlist_guard']['k']}; the bar is "
        f"{_pct(parameters['recall']['minimum'])}.",
        "",
        "## Floors and bars",
        "",
        "| bin | floor |",
        "|---|---|",
    ]
    lines += [f"| `{bin_}` | {value:.6f} |" for bin_, value in report["floors"].items()]
    lines += [
        "",
        f"**b = {bars['b']:.2f} (PROVISIONAL)**" if bars["b"] is not None
        else "**b: none on the grid**",
        f" ({_pct(bars['look_relabel_at_b'])} of look relabelled, "
        f"{bars['mistake_relabels_at_b']} of the real-mistake clips relabelled).",
        *([f"**b_u = {bars['b_u']:.2f} (PROVISIONAL)**: {_pct(bars['look_assigned_at_b_u'])} "
           "of the look clips it names are their own class."] if bars["b_u"] is not None else
          ["**b_u: none on the grid** (no bar names at least 99% own-class clips)."]),
        f"Literal-rule b_u (share of all clips): {_bar(bars['b_u_literal'])}; recorded for "
        "transparency.",
        "",
        "| bar | assigned | precision | coverage |",
        "|---|---|---|---|",
        *[f"| {row['bar']:.2f} | {row['assigned']} | {_pct(row['precision'])} | "
          f"{_pct(row['coverage'])} |" for row in bars["precision"]["by_bar"]],
        "",
        "_Precision is the amended b_u's quantity; coverage is the literal rule's, which can "
        "only fall as the bar rises._",
        "",
        "## Floor status",
        "",
        f"`{report['floor_status']}`",
        "",
        "| convention | recordings | passed | pass |",
        "|---|---|---|---|",
        *[f"| {key} | {value['recordings']} | {value['passed']} | {_pct(value['pass'])} |"
          for key, value in report["noise"]["conventions"].items()],
        "",
        "B backs a row in an empty bin off to the nearest calibrated floor, A fails it, C "
        "excludes it. The frozen noise stop fires under all three, so b and b_u are provisional "
        "and no replacement floor is applied here (parked for the owner).",
        "",
        "## Reported, never tuned on",
        "",
        f"| measure | value |",
        "|---|---|",
        f"| look retention | {_pct(look['retention'])} |",
        f"| look relabel rate at b | {_pct(look['relabel_rate_at_b'])} |",
        f"| `cer_wav ≥ 0.2` tail ({look['tail']['clips']} clips) retention | "
        f"{_pct(look['tail']['retention'])} |",
        f"| `cer_wav ≥ 0.2` tail relabel rate at b | {_pct(look['tail']['relabel_rate'])} |",
        f"| Freesound noise pass | {_pct(report['noise']['pass'])} |",
        f"| exhaustive-decoy floors (sensitivity) | retention "
        f"{_pct(report['exhaustive_decoy']['look_retention'])} |",
        f"| throughput | {report['throughput']['ms_per_clip']:.2f} ms per clip |",
        "",
        "## Stop conditions",
        "",
        "| condition | limit | value | fired |",
        "|---|---|---|---|",
        f"| look retention | {_pct(stop['look_retention']['limit'])} | "
        f"{_pct(stop['look_retention']['value'])} | {stop['look_retention']['fired']} |",
        f"| noise pass | {_pct(stop['noise_pass']['limit'])} | "
        f"{_pct(stop['noise_pass']['value'])} | {stop['noise_pass']['fired']} |",
        f"| bar b | — | {_bar(bars['b'])} | {stop['bar']['fired']} |",
        "",
    ]
    return "\n".join(lines)


def write_calibration(report: Mapping[str, Any], *, path: Path = CALIBRATION_PATH,
                      doc: Path = DOC_PATH) -> None:
    write_text_atomically(Path(path),
                          json.dumps(dict(report), ensure_ascii=False, indent=1,
                                     sort_keys=True) + "\n")
    write_text_atomically(Path(doc), render(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    references = commands.add_parser("references", help="build and write the reference database")
    references.add_argument("--out", type=Path, default=REFERENCES_PATH)
    commands.add_parser("calibrate", help="derive the floors and bars, write the report and page")
    commands.add_parser("report", help="re-render the page from the committed report")
    args = parser.parse_args()

    if args.command == "references":
        payload = write_references(args.out)
        print(f"wrote {args.out}: {payload['ayat']} ayat, {payload['classes']} classes")
    elif args.command == "calibrate":
        database = load_references()
        report = _calibration(database)
        write_calibration(report)
        print(json.dumps({"floors": report["floors"], "b": report["bars"]["b"],
                          "b_u": report["bars"]["b_u"],
                          "stop": report["stop_conditions"]}, indent=1))
    else:
        report = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        write_text_atomically(DOC_PATH, render(report))
        print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()

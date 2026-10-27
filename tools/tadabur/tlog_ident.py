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

**Verdicts (S5b).** A decode with speech is ``non_recitation`` when its normalised length is
under ``L_MIN`` (8) or its real best is within ``DELTA`` (0.30) of its own decoy best from the
same shortlisted search (``docs/tlog-ident-floor-preregistration.md``, #204). :func:`identify`
computes that verdict; no length bin is consulted. The retired binned floors — the type-7 99th
percentile of the best decoy score over Tarteel v1 look decodes, per length bin — are still
derived and committed as history, with the three empty-bin conventions of the S5 noise stop.

**Bars.** ``b`` is the smallest grid value at which ≤ 0.5% of look clips relabel and 0 of the 52
real-mistake clips relabel; ``b_u`` follows the owner-approved post-look erratum
(``docs/tlog-ident-bu-amendment.md``): the smallest bar at which ≥ 99% of the look clips the
identifier **names** are their own class. The literal share-of-all wording's result is recorded as
``b_u_literal``; its coverage cannot rise with the bar. The bars and floors are committed in
``tools/tadabur/tlog/ident/ident_calibration.json`` with the generated page
``docs/tlog-ayah-identification.md``; every value is re-derived from the committed inputs by the
acceptance tests.

**The five bars (S5b).** ``floor_status`` is the first fired confirmation bar
(:mod:`tadabur.tlog_ident_confirm`): the fresh Freesound, Common Voice ``ar`` and Common Voice
non-Arabic pass rates at ≤ 1%, all 52 real-mistake clips passing, and look retention at ≥ 90%.
The committed report carries them under ``confirm`` — the arms with Wilson 95% intervals, the
per-language rates, the rule's catches split by clause, the items that passed with their decodes
and spans, and the doc → manifests → stores provenance chain — so a stop is inspectable without a
rerun. ``b`` and ``b_u`` are ``provisional`` unless that status is exactly ``"confirmed"``, and
:func:`load_calibration` refuses anything else. It also refuses a report whose
``margin_definition.status`` is not ``"final"``: when the character-start oracle changes ``b_u``
or a look assignment the owner decides. And the mechanical outcome is not the owner's clearance:
the frozen floor document's §7 approval of the pinned margin definition is its own committed
record (``margin_approval.json``, :func:`margin_approval`), and ``load_calibration`` refuses until
one exists and cites the report's margin definition by SHA-256.

**The margin's bound (S5b).** The pinned competitor search is a word-span one, and
``docs/tlog-ident-margin-amendment.md`` discloses that it can overstate ``m``, never understate
it. :func:`oracle_margin` measures the bound once on the look half only: every character start of
every free word run is aligned in ``SHW`` mode and attributed by the same rule, one end and one
path per start. :func:`oracle_pass` records the result in ``tlog/ident/oracle_margin.json`` —
``b_u`` re-derived under the bound, the clips whose ``m`` changed and whether they cross ``b_u``.
It never chooses a bar or a label: the report's ``b_u``, precision table and assignments always
come from the pinned margins, and the mechanical consequence
(:func:`oracle_consequence`) is ``final`` only if the bound changed neither.

Committed inputs under ``tools/tadabur/tlog/ident/``: ``references.json.gz`` (the whole mushaf's
normalised references and offsets), ``real_mistake_clips.jsonl`` (the 52 clips holding the 56
owner-verified real-mistake sites), ``decodes/real_mistake.base.json`` and ``decodes/noise.base.json``
(the base teacher's whole-clip decodes), ``ident_calibration.json``.

Consumers use three functions, so S6 and S7 cannot fork the schema: :func:`identification_record`
names the committed row's fields, including the ``floor_rule`` that produced the verdict on every
row; :func:`load_calibration` returns ``(bars, k)`` and refuses any report whose
``floor_status`` is not ``"confirmed"``, whose ``margin_definition.status`` is not
``"final"``, or whose margin definition has no recorded owner approval
(:func:`margin_approval`, §7 of the frozen floor document, written by
``python -m tadabur.tlog_ident approve``); and :func:`load_identifier` is the entry point, which
additionally binds the reference database, the rule's values and the edlib version to the report.
The read path (a committed database and :func:`identify`) imports edlib only; the phonetizer is
imported where a reference is built.

Usage (from ``tools/``)::

  python -m tadabur.tlog_ident references    # build and commit references.json.gz (~5 min)
  python -m tadabur.tlog_ident calibrate     # thresholds, floors and the page (~40 min CPU:
                                             # look + guard + exhaustive decoy + pool walk)
  python -m tadabur.tlog_ident report        # re-render the page from the committed report
  python -m tadabur.tlog_ident oracle        # the character-start margin bound on look
                                             # (recorded only); --limit N times the projection
  python -m tadabur.tlog_ident approve       # the owner records their approval of the margin
                                             # definition; S6/S7 load nothing before it

The GPU decodes of the two calibration populations are produced by
``tadabur.tlog_ident_decode`` on cuda-dev; see ``scripts/ident-decode.sh``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import re
import time
import unicodedata
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import edlib

from training.tashkeel_eval import write_text_atomically

from . import normalization
from .normalization import map_char_offsets, normalize_phonemes
from .tarteel_split import DECODES_PATH, SPLIT_PATH, load_split
from .tlog_index import record_of, repo_relative, sha256_file

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

#: S5b (#204): the recitation rule is fixed in its own dated pre-registration, never selected on
#: the confirmation sets. A decode is non-recitation when it is shorter than ``L_MIN`` or its own
#: decoy is within ``DELTA`` of its real best, both from the same shortlisted search.
L_MIN = 8
DELTA = 0.30
FLOOR_SALT = "tlog-s5b-floor-confirm-v1"
FLOOR_PREREGISTRATION_PATH = (Path(__file__).resolve().parents[2]
                              / "docs" / "tlog-ident-floor-preregistration.md")
#: The pinned SHA-256 of that document; a test re-checks it against the committed file.
FLOOR_PREREGISTRATION_SHA256 = (
    "7bd2e1958ea8330df472030aefb6c97de7d0e9ccc2aa81ac133db682e1d58e3a")

BASE_MODEL = "obadx/muaalem-model-v3_2"
BASE_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"

#: ``label_source`` values.
FILENAME = "filename"
BOTH = "both"
IDENTIFIED = "identified"
NONE = "none"

#: The repository root and the one artifact directory; other modules import these, never rebuild.
ROOT = Path(__file__).resolve().parents[2]
IDENT_DIR = Path(__file__).parent / "tlog" / "ident"
REFERENCES_PATH = IDENT_DIR / "references.json.gz"
CALIBRATION_PATH = IDENT_DIR / "ident_calibration.json"
REAL_MISTAKE_CLIPS_PATH = IDENT_DIR / "real_mistake_clips.jsonl"
DECODES_DIR = IDENT_DIR / "decodes"
REAL_MISTAKE_DECODES_PATH = DECODES_DIR / "real_mistake.base.json"
NOISE_DECODES_PATH = DECODES_DIR / "noise.base.json"
#: The character-level margin oracle's result (look only; recorded, never a selection step).
ORACLE_PATH = IDENT_DIR / "oracle_margin.json"
#: The owner's clearance of the pinned margin definition — frozen floor document §7 requires it
#: at the S5b check-in, separately from the oracle's mechanical consequence. Committed only when
#: the owner approves (:func:`approve_margin` writes it); absent until then, and
#: :func:`load_calibration` refuses without it.
MARGIN_APPROVAL_PATH = IDENT_DIR / "margin_approval.json"
MARGIN_APPROVAL_SCHEMA = "tadabur.tlog_ident/margin-approval-v1"
#: The Freesound gap pool (the noise stop's population); the look decodes and the split come
#: from :mod:`tadabur.tarteel_split`, which owns their one definition.
NOISE_POOL_PATH = Path(__file__).parent / "nonpro_gap" / "noise_pool.json"
PREREGISTRATION_PATH = ROOT / "docs" / "tlog-ident-preregistration.md"
BU_AMENDMENT_PATH = ROOT / "docs" / "tlog-ident-bu-amendment.md"
MARGIN_AMENDMENT_PATH = ROOT / "docs" / "tlog-ident-margin-amendment.md"
HOLDOUT_PREREGISTRATION_PATH = ROOT / "docs" / "tlog-holdout-preregistration.md"
DOC_PATH = ROOT / "docs" / "tlog-ayah-identification.md"

#: The alphabet the reference envelope records.
WORD_TOKENISATION = "quran-transcript uthmani.split"
#: The mode the Freesound pool is decoded in: a pool recording is too long for a whole pass.
NOISE_DECODE_MODE = "stream_b0"
LIBRARY = {"name": "edlib", "version": "1.3.9.post1", "mode": "HW"}
#: Relabel rate the look half must stay under, and the negative arms' 1% limit (S5's noise stop
#: and S5b's three confirmation arms are one limit, so the two cannot drift apart).
RELABEL_LIMIT = 0.005
ASSIGNED_MIN = 0.99
LOOK_RETENTION_MIN = 0.90
PASS_LIMIT = 0.01

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
    name; they let the label rule recompute the fields without the database. ``floor_ok`` is the
    S5b recitation verdict, computed by :func:`identify`: ``length >= L_MIN`` and
    ``real - own decoy >= DELTA``.
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
    floor_ok: bool
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
        self._decoy_index = self._index(self._decoys)

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

    def _ranked(self, query: str, *, k: int, decoy: bool = False) -> tuple[int, ...]:
        """The shortlist as window positions, ranked by distinct shared 3-grams.

        The decoy side is ranked with its own index over the decoy sequences, never with the
        real one (the frozen rule and the S5 amendment). A decode that hits no 3-gram has no
        signal to rank with, so the search is exhaustive for it.
        """
        candidates = self._decoys if decoy else self._real
        index = self._decoy_index if decoy else self._real_index
        skeleton = _skeleton(query)
        grams = {skeleton[position:position + self.gram]
                 for position in range(len(skeleton) - self.gram + 1)}
        if not grams:
            return tuple(range(len(candidates)))
        counts: Counter[int] = Counter()
        for gram in grams:
            counts.update(index.get(gram, ()))
        ranked = sorted(counts, key=lambda position: (
            -counts[position], candidates[position].surah, candidates[position].ayah))
        if len(ranked) < k:  # the top k is filled with the windows no gram reached
            ranked += [position for position in range(len(candidates))
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
        """The best score over the decoy shortlist: the same construction as the real search,
        built over the decoy sequences (frozen rule; S5 amendment)."""
        query = _normalise(decode)
        if not query:
            return 0.0
        return max(1.0 - edlib.align(query, self._decoys[position].text,
                                     mode="HW")["editDistance"] / len(query)
                   for position in self._ranked(query, k=k, decoy=True))

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
        return self._attribute(query, candidate, distance, start, end, ops)

    def _attribute(self, query: str, candidate: _Candidate, distance: int, start: int,
                   end: int, ops: tuple[tuple[str, int], ...]) -> Match:
        """The word-span facts of one alignment of ``query`` inside ``candidate``.

        Shared by the pinned search (:meth:`_match`) and the character-start oracle
        (:meth:`_char_run_competitor`), so both attribute an alignment's words the same way.
        """
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

    def _sub_candidate(self, candidate: _Candidate, lo: int, hi: int) -> _Candidate | None:
        """The part of a window in ``[lo, hi)``.

        A constituent's extent stays anchored to its ayah's first character, so it may start
        before the slice: a suffix of an ayah must still resolve to that ayah's real words,
        not restart at word 1.
        """
        pieces = tuple(
            _Constituent(constituent.surah, constituent.ayah,
                         constituent.start - lo, constituent.end - lo)
            for constituent in candidate.constituents
            if constituent.start < hi and constituent.end > lo)
        if not pieces:
            return None
        return _Candidate(candidate.surah, candidate.ayah, candidate.text[lo:hi], pieces)

    def _free_word_runs(self, candidate: _Candidate, best: Match
                        ) -> tuple[tuple[tuple[int, int], ...], ...]:
        """The window's words outside the best span, split into maximal contiguous runs.

        Each word is its ``[start, end)`` character range in the window text. The best span's
        words are contiguous, so there are at most two runs; every location that does not
        overlap the winner lies inside one of them.
        """
        inside: list[bool] = []
        spans: list[tuple[int, int]] = []
        for constituent in candidate.constituents:
            reference = self.ayat[(constituent.surah, constituent.ayah)]
            for word in range(1, reference.words + 1):
                spans.append((constituent.start + reference.offsets[word - 1],
                              constituent.start + reference.offsets[word]))
                inside.append(
                    constituent.surah == best.surah
                    and best.ayah_from <= constituent.ayah <= best.ayah_to
                    and (constituent.ayah != best.ayah_from or word >= best.word_from)
                    and (constituent.ayah != best.ayah_to or word <= best.word_to))
        runs: list[list[tuple[int, int]]] = []
        run: list[tuple[int, int]] = []
        for span, taken in zip(spans, inside):
            if taken:
                if run:
                    runs.append(run)
                    run = []
            else:
                run.append(span)
        if run:
            runs.append(run)
        return tuple(tuple(word_run) for word_run in runs)

    def _run_competitor(self, query: str, candidate: _Candidate,
                        run: Sequence[tuple[int, int]], best: Match,
                        floor: float) -> Match | None:
        """The best different-class alignment inside one free word run.

        Branch and bound over word spans: a span's best score cannot exceed its parent's, so a
        span whose parent scores below the best competitor so far is skipped, and an
        admissible span prunes its own sub-spans, because its alignment is the earliest
        location at that score and so wins the frozen tie-break. This reaches a location that
        the whole run's single best alignment hides when that alignment is the best class.
        """
        found: Match | None = None
        seen: set[tuple[int, int]] = set()

        def visit(first: int, last: int, bound: float) -> None:
            nonlocal found
            if bound < max(floor, found.score if found is not None else 0.0):
                return
            if (first, last) in seen:
                return
            seen.add((first, last))
            sub = self._sub_candidate(candidate, run[first][0], run[last][1])
            match = self._match(query, sub)
            if match.text_class != best.text_class:
                if found is None or _rank(match) > _rank(found):
                    found = match
                return
            if first < last:
                visit(first + 1, last, match.score)
                visit(first, last - 1, match.score)

        visit(0, len(run) - 1, float("inf"))
        return found

    def competitor(self, query: str, windows: Sequence[Window], best: Match) -> Match | None:
        """The best different-class alignment at a location that does not overlap ``best``.

        Every shortlisted window is searched, including the winning one: each of its free word
        runs is searched span by span (branch and bound), so a different class beside the
        winner is found whether it scores the same as a same-class optimum or less.
        """
        def key(row: Match, winner: bool) -> tuple:
            return _rank(row, winner)

        found: tuple[tuple, Match] | None = None

        def offer(row: Match, winner: bool) -> None:
            nonlocal found
            rank = key(row, winner)
            if found is None or rank > found[0]:
                found = (rank, row)

        for window in windows:
            # A window's own winning alignment keeps the pre-existing tie-break: on equal
            # scores it is preferred over a location found inside another window.
            if window.match.text_class != best.text_class and not _overlaps(window.match, best):
                offer(window.match, True)
            candidate = self._candidate(window.surah, window.ayah, decoy=False)
            for run in self._free_word_runs(candidate, best):
                match = self._run_competitor(
                    query, candidate, run, best,
                    found[0][0] if found is not None else float("-inf"))
                if match is not None:
                    offer(match, False)
        return None if found is None else found[1]

    def char_competitor(self, query: str, windows: Sequence[Window], best: Match,
                        floor: float) -> Match | None:
        """The best different-class alignment that begins at a character outside ``best``.

        The character-start margin oracle (architect r18/r19): every character start of every
        free word run of every shortlisted window is aligned in ``SHW`` mode (the query must
        start at that character, the run's end is free) and attributed by :meth:`_attribute`.
        One end and one path per start, so it is a stricter bound on the margin, not the exact
        margin. Only a start whose score is strictly above ``floor`` (the pinned competitor's
        score) can lower the margin: equal scores leave it as it is.
        """
        found: Match | None = None
        for window in windows:
            candidate = self._candidate(window.surah, window.ayah, decoy=False)
            for run in self._free_word_runs(candidate, best):
                match = self._char_run_competitor(
                    query, candidate, run, best,
                    floor if found is None else max(floor, found.score))
                if match is not None and (found is None or _rank(match) > _rank(found)):
                    found = match
        return found

    def _char_run_competitor(self, query: str, candidate: _Candidate,
                             run: Sequence[tuple[int, int]], best: Match,
                             floor: float) -> Match | None:
        """The best different-class alignment starting at a character of one free run.

        Stage one scores every start's ``SHW`` distance and keeps the starts strictly above
        ``floor``; stage two visits them best first and stops as soon as no remaining start can
        beat the best different class found, so the paths are only computed where they can
        change the margin.
        """
        text = candidate.text
        stop = run[-1][1]
        starts: list[tuple[float, int]] = []
        for start in range(run[0][0], stop):
            distance = edlib.align(query, text[start:stop], mode="SHW")["editDistance"]
            score = 1.0 - distance / len(query)
            if score > floor:
                starts.append((-score, start))
        starts.sort()
        found: Match | None = None
        for negative, start in starts:
            if found is not None and -negative <= found.score:
                break
            result = edlib.align(query, text[start:stop], mode="SHW", task="path")
            local_start, local_end = result["locations"][0]
            match = self._attribute(query, candidate, result["editDistance"],
                                    start + int(local_start), start + int(local_end) + 1,
                                    _cigar_ops(result["cigar"]))
            if match.text_class != best.text_class and (
                    found is None or _rank(match) > _rank(found)):
                found = match
        return found


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


def _rank(row: Match, winner: bool = False) -> tuple:
    """The frozen competitor ranking: score, the window-winner flag, then the earlier span."""
    return (row.score, winner, -row.surah, -row.ayah_from, -row.word_from,
            -row.ayah_to, -row.word_to)


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
    """The words phonetized as one utterance: space-free text and its per-word offsets.

    The phonetizer is imported here, not at module level: the read path (a committed
    database and :func:`identify`) needs edlib alone.
    """
    import hafs_phonetizer
    from .waqf_segments import spaceless_word_offsets

    out = hafs_phonetizer.phonetize(" ".join(words))
    return spaceless_word_offsets(out.phonemes, out.mappings, list(words))


def _fallback_offsets(key: str, words: Sequence[str]) -> tuple[str, list[int]]:
    """The fallback string and the word boundaries the S5 amendment derives for it."""
    import generate_phonemes

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
    import generate_phonemes
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
    import generate_phonemes

    return ReferenceDatabase.build([
        ayah_reference(surah, ayah)
        for surah, count in enumerate(generate_phonemes.HAFS_AYAH_COUNTS, start=1)
        for ayah in range(1, count + 1)])


def references_payload(database: ReferenceDatabase) -> dict:
    import hafs_phonetizer

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
    """The committed reference database, from the file alone — no phonetizer import.

    Two kinds of check, both dependency-free. The envelope's records must be internally
    consistent (the ayah and class counts), and its preprocessing pins must be this code's:
    ``normalization_algorithm_version`` and ``word_tokenisation`` are the live constants, so a
    version bump that changes how a decode normalises cannot read references normalised by the
    old algorithm. Binding the envelope to the calibration that used it (the file's SHA-256 and
    counts) is :func:`load_identifier`'s job; only rebuilding references needs the phonetizer.
    """
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {"normalization_algorithm_version": normalization.ALGORITHM_VERSION,
                "word_tokenisation": WORD_TOKENISATION}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path} records {key}={payload.get(key)!r}, not {value!r}")
    ayat = [AyahReference(item["surah"], item["ayah"], item["reference"],
                          tuple(item["offsets"])) for item in payload["items"]]
    if len(ayat) != payload.get("ayat"):
        raise ValueError(f"{path} records {payload.get('ayat')} ayat, holds {len(ayat)}")
    database = ReferenceDatabase.build(ayat)
    if database.class_count != payload.get("classes"):
        raise ValueError(f"{path} records {payload.get('classes')} classes, "
                         f"the build has {database.class_count}")
    return database


# --- identification ---------------------------------------------------------------------------


def identify(decode: str, *, database: ReferenceDatabase, filename_ayah: str | None = None,
             bars: Mapping[str, float] | None = None, k: int = K,
             exhaustive_decoy: bool = False) -> Identification:
    """One decode's identification and its S5b recitation verdict (module docstring).

    The verdict is ``length >= L_MIN`` and ``real best - own decoy best >= DELTA``; both scores
    come from this decode's one shortlisted search. ``bars`` carries ``b`` and ``b_u``; without
    them no row is assigned or relabelled.
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
            margin_class=None, floor_ok=False,
            overlaps_filename=None, label_source=NONE, assigned_label=None,
            filename_label=filename_label)
        return _with_label(identification, bars)

    shortlisted = database.shortlist(decode, k=k)
    best = shortlisted[0].match
    competitor = database.competitor(query, shortlisted, best)
    filename_score = (database.score_window(decode, *name) if name is not None else None)
    decoy = database.decoy_best(decode, k=k)
    identification = Identification(
        length=len(query), match=best, decoy=decoy,
        exhaustive_decoy=(database.decoy_exhaustive_best(decode)
                          if exhaustive_decoy else None),
        filename_class=filename_class, filename_score=filename_score,
        delta=(best.score - filename_score) if filename_score is not None else None,
        # With no different class at a non-overlapping location nothing contradicts the
        # match, so the margin is measured against the empty competitor (score 0).
        margin=best.score - (competitor.score if competitor is not None else 0.0),
        margin_class=competitor.text_class if competitor is not None else None,
        floor_ok=len(query) >= L_MIN and best.score - decoy >= DELTA,
        overlaps_filename=(_overlaps_ayah(best, name) if name is not None else None),
        label_source=NONE, assigned_label=None,
        class_label=database.class_label(best.text_class),
        filename_label=filename_label)
    return _with_label(identification, bars)


def identification_record(identification: Identification, *, decode_mode: str) -> dict:
    """The committed row for one identification: the one place its field names are fixed.

    The keys are the issue's output list — span, ``straddles``, class, score, decoy, Δ,
    ``margin``, ``margin_class``, ``id_floor_ok``, ``label_source``, ``assigned_label`` — plus
    the inputs a consumer joins on, the ``floor_rule`` that produced ``id_floor_ok`` and the
    ``decode_mode`` that produced the decode (S5b I3.3: required, never defaulted, so S6/S7
    cannot omit which mode a verdict belongs to).
    """
    if not isinstance(decode_mode, str) or not decode_mode:
        raise ValueError("decode_mode is required on every identification record")
    match = identification.match
    span = ({"surah": match.surah, "ayah_from": match.ayah_from,
             "word_from": match.word_from, "ayah_to": match.ayah_to,
             "word_to": match.word_to, "straddles": match.straddles, "score": match.score}
            if match is not None else
            dict.fromkeys(("surah", "ayah_from", "word_from", "ayah_to", "word_to",
                           "straddles", "score")))
    return {
        **span,
        "length": identification.length,
        "class_label": identification.class_label,
        "decoy": identification.decoy,
        "exhaustive_decoy": identification.exhaustive_decoy,
        "delta": identification.delta,
        "margin": identification.margin,
        "margin_class": identification.margin_class,
        "id_floor_ok": identification.floor_ok,
        "overlaps_filename": identification.overlaps_filename,
        "filename_label": identification.filename_label,
        "label_source": identification.label_source,
        "assigned_label": identification.assigned_label,
        "floor_rule": floor_rule(),
        "decode_mode": decode_mode,
    }


def floor_rule() -> dict:
    """The recitation rule that sets ``floor_ok``: its pre-registration's path and SHA-256."""
    return {"path": repo_relative(FLOOR_PREREGISTRATION_PATH),
            "sha256": FLOOR_PREREGISTRATION_SHA256}


def oracle_consequence(b_u: float | None, oracle: Mapping[str, Any]) -> str:
    """The mechanical margin-pin consequence: ``final`` only if the oracle changed neither
    ``b_u`` nor a look assignment at ``b_u``; otherwise the owner decides (architect r18/r19)."""
    if oracle["b_u_oracle"] == b_u and oracle["assignments_changed_at_b_u"] == 0:
        return "final"
    return "owner"


def _with_label(identification: Identification,
                bars: Mapping[str, float] | None) -> Identification:
    """The label rule: ``both``/``identified``/``filename``/``none`` (pre-registration table).

    ``both`` is the table's class-equality fact and does not depend on the recitation rule; the
    rule gates only relabelling and assignment (``identified``). A non-recitation row is
    therefore never relabelled or assigned, but a matching filename class is still ``both``.
    """
    match = identification.match
    if (match is not None and identification.filename_label is not None
            and match.text_class == identification.filename_class):
        return replace(identification, label_source=BOTH,
                       assigned_label=identification.filename_label)
    if match is None or identification.floor_ok is not True:
        if identification.filename_label is not None:
            return replace(identification, label_source=FILENAME,
                           assigned_label=identification.filename_label)
        return replace(identification, label_source=NONE, assigned_label=None)
    if identification.filename_label is not None:
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
    return (identification.match is not None and identification.floor_ok is True
            and identification.delta is not None and identification.delta >= bar
            and not identification.overlaps_filename
            and identification.match.text_class != identification.filename_class)


def assigned(identification: Identification, bar: float) -> bool:
    """Whether the assign rule fires at ``bar`` (labels hidden)."""
    return (identification.match is not None and identification.floor_ok is True
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


# --- the character-start margin oracle --------------------------------------------------------


def oracle_margin(database: ReferenceDatabase, decode: str, *, pinned: float,
                  k: int = K) -> float:
    """One decode's character-start margin bound: ``min(pinned, best − char competitor)``.

    ``pinned`` is the margin :func:`identify` reports for the same decode. Only an alignment
    scoring strictly above the pinned competitor's (``best.score - pinned``) can lower the
    margin, so that score prunes the character starts. The bound is recorded, never a selection
    step: ``b_u`` and every assignment stay the pinned ones (architect r18/r19).
    """
    query = _normalise(decode)
    if not query:
        raise ValueError("an empty decode has no margin")
    shortlisted = database.shortlist(decode, k=k)
    best = shortlisted[0].match
    found = database.char_competitor(query, shortlisted, best, best.score - pinned)
    return pinned if found is None else min(pinned, best.score - found.score)


def oracle_artifact(k: int, path: Path = ORACLE_PATH) -> dict:
    """The committed margin-bound artifact, refused unless it measured this checkout's inputs.

    Every hash the artifact records — the margin definition, the look decodes, their split and
    the real-mistake clips — must be this checkout's, or the bound belongs to other data. ``k``
    is recorded from self-check r1 on; a future artifact must match the calibration's shortlist
    size, while the committed one predates the field.
    """
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"definition": sha256_file(MARGIN_AMENDMENT_PATH),
                "decodes": sha256_file(DECODES_PATH),
                "split": sha256_file(SPLIT_PATH),
                "mistakes": sha256_file(REAL_MISTAKE_CLIPS_PATH)}
    recorded = {"definition": artifact["definition"]["sha256"],
                **{key: artifact["population"][key]["sha256"]
                   for key in ("decodes", "split", "mistakes")}}
    stale = sorted(key for key, value in recorded.items() if expected[key] != value)
    if stale:
        raise ValueError(f"the oracle was measured on different inputs: {stale}")
    if "k" in artifact and artifact["k"] != k:
        raise ValueError(f"the oracle ran at k = {artifact['k']}, not {k}")
    return artifact


def oracle_pass(database: ReferenceDatabase, look: Sequence[Decode],
                mistakes: Sequence[Identification], *, k: int = K,
                progress: Callable[[str], None] | None = None) -> dict:
    """The character-start margin oracle's recorded result, over the look half only.

    Each look decode is identified under the pinned rule, then bounded by
    :func:`oracle_margin`. ``b_u`` is re-derived from the **pinned** margins and, separately,
    from the min(pinned, oracle) margins; ``assignments_changed_at_b_u`` counts the look rows
    whose assignment at the pinned ``b_u`` the bound would change. The oracle never chooses a
    bar: it records what an exact definition would have done (architect r18/r19).
    """
    emit = progress or _phase
    started = time.time()
    rows: list[Identification] = []
    bounds: list[float | None] = []
    for index, decode in enumerate(look, 1):
        row = identify(decode.spans, database=database, filename_ayah=decode.surah_ayah, k=k)
        rows.append(row)
        bounds.append(oracle_margin(database, decode.spans, pinned=row.margin, k=k)
                      if row.margin is not None else None)
        if index % 500 == 0:
            emit(f"oracle {index}/{len(look)}")
    seconds = time.time() - started

    pinned_bars = derive_bars(rows, mistakes)
    oracle_rows = [row if bound is None else replace(row, margin=bound)
                   for row, bound in zip(rows, bounds)]
    oracle_bars = derive_bars(oracle_rows, mistakes)
    b_u = pinned_bars["b_u"]
    if b_u is None:
        raise ValueError("the look population names no row at any bar: no b_u to check")

    # Each detail joins a decode with *its own* pinned and bounded rows: a zip, never a shared
    # index into another list (the r1 off-by-one read the following clip's margin).
    details = [{
        "key": decode.key, "audio_filename": decode.audio_filename,
        "surah_ayah": decode.surah_ayah, "length": row.length,
        "class_label": row.class_label, "margin_class": row.margin_class,
        "pinned_margin": row.margin, "oracle_margin": oracle.margin,
        "assigned_pinned": assigned(row, b_u),
        "assigned_oracle": assigned(oracle, b_u),
    } for decode, row, oracle in zip(look, rows, oracle_rows)
        if oracle.margin is not None and oracle.margin < row.margin]
    for item in details:
        item["crosses_b_u"] = item["assigned_pinned"] != item["assigned_oracle"]
    details.sort(key=lambda item: item["key"])

    return {
        "schema": "tadabur.tlog_ident/oracle-margin-v1",
        "k": k,
        "definition": {"path": repo_relative(MARGIN_AMENDMENT_PATH),
                       "sha256": sha256_file(MARGIN_AMENDMENT_PATH)},
        "disclosure": "one end and one path per character start: a stricter bound on the "
                      "margin, not the exact margin (docs/tlog-ident-margin-amendment.md)",
        "population": {
            "decodes": {"path": repo_relative(DECODES_PATH),
                        "sha256": sha256_file(DECODES_PATH), "clips": len(look)},
            "split": {"path": repo_relative(SPLIT_PATH), "sha256": sha256_file(SPLIT_PATH)},
            "mistakes": {"path": repo_relative(REAL_MISTAKE_CLIPS_PATH),
                         "sha256": sha256_file(REAL_MISTAKE_CLIPS_PATH),
                         "clips": len(mistakes)},
        },
        "clips": len(rows),
        "clips_m_changed": len(details),
        "share_m_changed": len(details) / len(rows),
        "b": pinned_bars["b"],
        "b_u": b_u,
        "b_oracle": oracle_bars["b"],
        "b_u_oracle": oracle_bars["b_u"],
        "assignments_at_b_u": {"pinned": sum(assigned(row, b_u) for row in rows),
                               "oracle": sum(assigned(row, b_u) for row in oracle_rows)},
        "assignments_changed_at_b_u": sum(
            item["crosses_b_u"] for item in details),
        "changed": details,
        "seconds": round(seconds, 3),
    }


# --- the committed populations ----------------------------------------------------------------


def recall_sample(clips: Iterable[str]) -> tuple[str, ...]:
    """The seeded look clips the shortlist guard scores (SALT), first ``RECALL_CLIPS`` by rank."""
    ranked = sorted(clips, key=lambda clip: hashlib.sha256(
        f"{SALT}:{clip}".encode("utf-8")).digest())
    return tuple(ranked[:RECALL_CLIPS])


_RULE_LENGTH = re.compile(r"L_min = (\d+)")
_RULE_DELTA = re.compile(r"δ = (\d+(?:\.\d+)?)")


def _documented_rule(path: Path = FLOOR_PREREGISTRATION_PATH) -> tuple[int, float]:
    """The ``(L_min, δ)`` the pinned pre-registration states, refused if the file drifted.

    The report records the rule as that document's SHA-256, so the values are checked against
    the frozen text whose hash the report cites — none of them is a separate report field.
    """
    digest = sha256_file(path)
    if digest != FLOOR_PREREGISTRATION_SHA256:
        raise ValueError(f"{repo_relative(path)} is {digest}, not the pinned "
                         f"{FLOOR_PREREGISTRATION_SHA256}")
    text = Path(path).read_text(encoding="utf-8")
    length, delta = _RULE_LENGTH.search(text), _RULE_DELTA.search(text)
    if length is None or delta is None:
        raise ValueError(f"{repo_relative(path)} no longer states L_min and δ")
    return int(length[1]), float(delta[1])


def margin_approval(path: Path = MARGIN_APPROVAL_PATH) -> dict:
    """The recorded owner approval of the pinned margin definition, or a refusal.

    The frozen floor pre-registration's §7 requires the owner to approve the margin definition
    at the S5b check-in before ``b_u`` is final — a condition the oracle's mechanical
    consequence cannot supply. The record is its own small file, written by
    :func:`approve_margin` when the owner approves: it names its schema, the approving owner and
    date, the margin definition by path and SHA-256, and the ``b_u`` it clears. An absent or
    malformed record is a refusal, never a default.
    """
    if not Path(path).exists():
        raise ValueError(
            f"the owner approval of {repo_relative(MARGIN_AMENDMENT_PATH)} is not recorded "
            f"({repo_relative(path)} is absent): b_u is not cleared for use")
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != MARGIN_APPROVAL_SCHEMA:
        raise ValueError(f"{path} is not a margin approval (schema "
                         f"{record.get('schema')!r}, not {MARGIN_APPROVAL_SCHEMA!r})")
    if record.get("approved_by") != "owner":
        raise ValueError(f"{path} is not the owner's approval (approved_by "
                         f"{record.get('approved_by')!r})")
    try:
        date.fromisoformat(str(record.get("approved_on")))
    except ValueError:
        raise ValueError(f"{path} records no approved_on date "
                         f"({record.get('approved_on')!r})") from None
    doc = record.get("doc") or {}
    if not all(isinstance(doc.get(key), str) and doc.get(key)
               for key in ("path", "sha256")):
        raise ValueError(f"{path} does not cite the approved margin definition's path and "
                         f"SHA-256")
    if not isinstance(record.get("b_u"), (int, float)):
        raise ValueError(f"{path} does not record the approved b_u")
    return record


def _cleared_report(path: Path, approval_path: Path) -> dict:
    """The calibration report, refused unless every consumer gate clears it.

    The S5b floor must be recorded as exactly ``"confirmed"``; a stopped, passed or provisional
    report is refused. The margin pin must be mechanically ``"final"``: when the character-start
    oracle changes ``b_u`` or a look assignment, the owner decides. And the owner must have
    approved the pinned margin definition (:func:`margin_approval`) *at the report's ``b_u``* —
    the mechanical outcome does not imply that, and a recalibration that moves ``b_u`` needs a
    new approval.
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    status = report.get("floor_status")
    if status != "confirmed":
        raise ValueError(f"{path} is not a confirmed calibration (floor_status {status!r})")
    if report["bars"].get("provisional"):
        raise ValueError(f"{path} is not a confirmed calibration (bars provisional: "
                         f"{report['bars'].get('provisional_reason')})")
    margin = report.get("margin_definition") or {}
    if margin.get("status") != "final":
        raise ValueError(f"{path} is not cleared for use (margin_definition.status "
                         f"{margin.get('status')!r}: the owner decides b_u)")
    approval = margin_approval(approval_path)
    approved = approval["doc"]
    if (approved["path"], approved["sha256"]) != (margin.get("path"), margin.get("sha256")):
        raise ValueError(
            f"{approval_path} approves {approved['path']} at {approved['sha256']}, not the "
            f"report's margin definition ({margin.get('path')} at {margin.get('sha256')}): "
            f"b_u is not cleared for use")
    if approval["b_u"] != report["bars"]["b_u"]:
        raise ValueError(
            f"{approval_path} approves b_u {approval['b_u']}, not the report's "
            f"{report['bars']['b_u']}: b_u is not cleared for use")
    return report


def load_calibration(path: Path = CALIBRATION_PATH, *,
                     approval_path: Path = MARGIN_APPROVAL_PATH) -> tuple[dict, int]:
    """The committed ``(bars, k)`` a consumer labels rows with, cleared only
    (:func:`_cleared_report`). The floors are history — the rule is inside :func:`identify` —
    so they are not returned."""
    report = _cleared_report(path, approval_path)
    return report["bars"], report["parameters"]["k"]


def load_identifier(path: Path = CALIBRATION_PATH, *,
                    approval_path: Path = MARGIN_APPROVAL_PATH,
                    references_path: Path = REFERENCES_PATH
                    ) -> tuple[ReferenceDatabase, dict, int]:
    """The consumer entry point: the cleared report's ``(database, bars, k)``, bound together.

    S6 and S7 must not assemble a database and a bars pair by hand. Beyond the three gates of
    :func:`load_calibration`, every input the report records is compared with this checkout:
    the references file's SHA-256 and its ayah, class and surah counts, together with its
    normalisation and tokenisation pins (:func:`load_references`, which refuses a cache built by
    a different algorithm); the floor rule (its
    document's SHA-256 through :func:`floor_rule`, and the ``L_min``/``δ`` the pinned document
    states, against the code's constants); and the edlib name, version and mode whose distances
    the floors and bars are percentiles of.
    """
    report = _cleared_report(path, approval_path)
    database = load_references(references_path)
    reference = report["reference"]
    digest = sha256_file(references_path)
    if digest != reference["sha256"]:
        raise ValueError(f"{references_path} is {digest}, not the calibrated "
                         f"{reference['sha256']}")
    if (len(database.ayat), database.class_count, database.surah_count) != (
            reference["ayat"], reference["classes"], reference["surahs"]):
        raise ValueError(
            f"{references_path} holds {len(database.ayat)} ayat, {database.class_count} "
            f"classes and {database.surah_count} surahs, not the calibrated "
            f"{reference['ayat']}, {reference['classes']} and {reference['surahs']}")
    if report["floor_rule"] != floor_rule():
        raise ValueError(f"{path} carries the floor rule {report['floor_rule']}, not "
                         f"{floor_rule()}")
    if (L_MIN, DELTA) != _documented_rule():
        raise ValueError(f"the code's rule (L_min {L_MIN}, DELTA {DELTA}) is not the pinned "
                         f"document's {_documented_rule()}")
    library = report["parameters"]["library"]
    installed = importlib.metadata.version("edlib")
    if library != LIBRARY or installed != LIBRARY["version"]:
        raise ValueError(f"the calibration's search is {library}; this checkout's is {LIBRARY} "
                         f"and its installed edlib is {installed}")
    return database, report["bars"], report["parameters"]["k"]


def approve_margin(issue: int = 204, *, path: Path = MARGIN_APPROVAL_PATH,
                   calibration_path: Path = CALIBRATION_PATH) -> dict:
    """Write the owner's approval of the pinned margin definition (the owner runs this).

    Only the owner approves, so nothing here decides *whether* to: the command records the live
    amendment's SHA-256, the committed calibration's ``b_u`` and today's date, after refusing a
    dirty tree or an existing record. The written file is committed; S6/S7 then load it.
    """
    from .tlog_ident_confirm import require_committed  # lazy: one import direction at load

    require_committed([MARGIN_AMENDMENT_PATH, FLOOR_PREREGISTRATION_PATH, calibration_path])
    if Path(path).exists():
        raise ValueError(f"{path} already records an approval; edit the repository, not this "
                         f"command, to supersede it")
    report = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if report.get("margin_definition", {}).get("status") != "final":
        raise ValueError(f"{calibration_path} has no mechanically final margin to approve")
    record = {
        "schema": MARGIN_APPROVAL_SCHEMA,
        "issue": issue,
        "doc": {"path": repo_relative(MARGIN_AMENDMENT_PATH),
                "sha256": sha256_file(MARGIN_AMENDMENT_PATH)},
        "b_u": report["bars"]["b_u"],
        "approved_on": date.today().isoformat(),
        "approved_by": "owner",
    }
    write_text_atomically(Path(path),
                          json.dumps(record, ensure_ascii=False, indent=1,
                                     sort_keys=True) + "\n")
    return record


def load_look_decodes(path: Path = DECODES_PATH,
                      split_path: Path = SPLIT_PATH) -> list[Decode]:
    """The Tarteel v1 **look** decodes, and only them: the calibration's population."""
    look = set(load_split(split_path)["look"])
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


def _mistake_identifications(database: ReferenceDatabase, k: int) -> list[Identification]:
    """The 52 real-mistake clips, identified from their committed whole-clip decodes."""
    truth = load_decode_store(REAL_MISTAKE_DECODES_PATH)
    return [identify(truth["items"][clip["audio_filename"]]["spans"],
                     database=database, filename_ayah=clip["surah_ayah"], k=k)
            for clip in real_mistake_clips()]


# --- the calibration --------------------------------------------------------------------------


def _look_measurements(database: ReferenceDatabase, look: Sequence[Decode], *,
                       k: int = K) -> tuple[list, float]:
    """Score every look decode, timed; returns the raw identifications and the elapsed seconds."""
    started = time.time()
    scored = [identify(decode.spans, database=database, filename_ayah=decode.surah_ayah, k=k)
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


def select_k(database: ReferenceDatabase, sample: Sequence[str],
             decodes: Mapping[str, Decode], *, k: int = K) -> dict:
    """The guard's k, doubling until the shortlist's best equals exhaustive search.

    The selected k is the search every calibration population is then scored with; the guard
    is not a separate measurement.
    """
    while True:
        _phase(f"shortlist guard: {len(sample)} clips at k = {k}")
        guard = _shortlist_guard(database, sample, decodes, k=k)
        _phase(f"guard recall {guard['recalled']}/{guard['clips']}")
        if guard["passed"]:
            return guard
        k *= 2


def _phase(message: str) -> None:
    print(f"[tlog_ident] {message}", flush=True)


def pool_summary(rows: Iterable[Mapping[str, Any]]) -> tuple[dict, int]:
    """The pool's length facts and b_u's reach, from :func:`tadabur.tlog_overlap.iter_pool_rows`.

    A row's length fact is ``speech_s``: the frozen row carries the pass-1 facts
    ``speech_s``/``speech_ratio`` (not the clip's ``duration_s``), and speech is what the decode's
    character length tracks. ``usable`` rows only, so a row the codec floor rejected (and whose
    ``speech_s`` is ``None``) is skipped before any conversion.
    """
    usable = unlabelled = 0
    over = {20: 0, 30: 0, 60: 0}
    longest = 0.0
    for row in rows:
        if not row["usable"]:
            continue
        usable += 1
        speech = float(row["speech_s"])
        for edge in over:
            over[edge] += speech > edge
        longest = max(longest, speech)
        if row["split"] == "unclean" and row["filename_label"] is None:
            unlabelled += 1
    return ({"source": "tadabur.tlog_overlap.iter_pool_rows", "usable": usable,
             "field": "speech_s", "over_20_s": over[20], "over_30_s": over[30],
             "over_60_s": over[60], "longest_s": round(longest, 3)}, unlabelled)


def _calibration(database: ReferenceDatabase) -> dict:
    import hafs_phonetizer

    look = load_look_decodes()
    by_clip = {decode.audio_filename: decode for decode in look}
    sample = recall_sample(by_clip)
    # The guard picks the search; every population below is scored with the same k.
    guard = select_k(database, sample, by_clip)
    k = guard["k"]

    _phase(f"scoring {len(look)} look decodes at k = {k}")
    raw, seconds = _look_measurements(database, look, k=k)
    floors = derive_floors(raw)
    _phase(f"floors (history) {floors}")
    # The rule's verdict is already on every row; the binned floors are committed as history.

    clip_list = real_mistake_clips()
    _phase(f"scoring {len(clip_list)} real-mistake clips")
    mistakes = _mistake_identifications(database, k)
    bars = derive_bars(raw, mistakes)
    retention = sum(row.floor_ok for row in raw) / len(raw)

    # The character-start margin bound, measured before this run and recorded beside the
    # pinned bars (architect r18/r19). It never chooses a bar: its own b_u is recorded for the
    # owner, and ``margin_definition.status`` is the mechanical consequence.
    bound = oracle_artifact(k)
    oracle = {"b_u_oracle": bound["b_u_oracle"],
              "clips_m_changed": bound["clips_m_changed"],
              "assignments_changed_at_b_u": bound["assignments_changed_at_b_u"],
              "artifact": record_of(ORACLE_PATH)}
    margin_status = oracle_consequence(bars["b_u"], oracle)

    # The S5b confirmation: the five bars are the floor's status, and the S5 binned stop below
    # is history.
    from .tlog_ident_confirm import confirm_block  # lazy: one import direction at load

    _phase("scoring the three confirmation stores (Freesound, CV ar, CV other)")
    confirmation = confirm_block(database, k=k, look_retention=retention,
                                 mistakes=[row.floor_ok for row in mistakes])
    floor_status = confirmation["status"]
    _phase(f"confirmation status {floor_status}")
    bars["provisional"] = floor_status != "confirmed"
    bars["provisional_reason"] = floor_status

    # History: the retired binned floors' sensitivity to scoring the exhaustive decoy. The
    # rule's verdict never reads it, but S5's report and the guard-k test pin the pass.
    _phase(f"exhaustive-decoy sensitivity on {len(look)} clips (history)")
    exhaustive = [identify(decode.spans, database=database, filename_ayah=decode.surah_ayah,
                           exhaustive_decoy=True, k=k) for decode in look]
    exhaustive_floors = derive_floors(replace(row, decoy=row.exhaustive_decoy)
                                      for row in exhaustive) if exhaustive else {}
    exhaustive_retention = sum(
        row.match is not None
        and (floor := exhaustive_floors.get(floor_bin(row.length))) is not None
        and row.match.score > floor
        for row in exhaustive) / len(exhaustive)

    pool = json.loads(NOISE_POOL_PATH.read_text(encoding="utf-8"))["recordings"]
    noise_store = load_decode_store(NOISE_DECODES_PATH)
    pool_samples = {str(record["freesound_id"]): record["samples_sha256"]
                    for record in pool}
    if set(noise_store["items"]) != set(pool_samples):
        raise ValueError("the noise decode store does not cover the committed pool")
    mismatched = [key for key, item in noise_store["items"].items()
                  if item["samples_sha256"] != pool_samples[key]]
    if mismatched:
        raise ValueError(f"{len(mismatched)} noise decodes are not the pool's samples: "
                         f"{sorted(mismatched)[:5]}")
    _phase(f"scoring {len(noise_store['items'])} noise recordings")
    noise = [identify(item["spans"], database=database, k=k)
             for item in noise_store["items"].values()]
    conventions = noise_pass_by_convention(noise, floors)
    # B (back off to the nearest calibrated bin) is the headline: the most faithful reading of
    # the binned rule. A and C are reported beside it (architect r16).
    noise_pass = conventions["B"]["pass"]
    rule_passed = sum(row.floor_ok for row in noise)

    from .tlog_overlap import iter_pool_rows  # heavy import, calibration only

    _phase("walking the S1+S4 pool for b_u's reach and the pool's length histogram")
    pool_lengths, unlabelled = pool_summary(iter_pool_rows())

    relabel_rate = bars["look_relabel_at_b"]
    tail = [(decode, row) for decode, row in zip(look, raw) if decode.cer_wav >= 0.2]
    tail_retention = sum(row.floor_ok for _, row in tail) / len(tail) if tail else None
    tail_relabel = (sum(relabelled(row, bars["b"]) for _, row in tail) / len(tail)
                    if tail and bars["b"] is not None else None)

    stop = {
        "look_retention": {"limit": LOOK_RETENTION_MIN, "value": retention,
                           "fired": retention < LOOK_RETENTION_MIN},
        "noise_pass": {"limit": PASS_LIMIT, "value": noise_pass,
                       "fired": noise_pass > PASS_LIMIT},
        "bar": {"limit": None, "value": bars["b"], "fired": bars["b"] is None},
    }
    return {
        "schema": "tadabur.tlog_ident/calibration-v1",
        "issue": 204,
        "preregistration": {
            "path": repo_relative(PREREGISTRATION_PATH),
            "sha256": sha256_file(PREREGISTRATION_PATH),
            "parent": {"path": repo_relative(HOLDOUT_PREREGISTRATION_PATH),
                       "sha256": sha256_file(HOLDOUT_PREREGISTRATION_PATH)},
        },
        "bu_amendment": {"path": repo_relative(BU_AMENDMENT_PATH),
                         "sha256": sha256_file(BU_AMENDMENT_PATH)},
        "floor_preregistration": {
            "path": repo_relative(FLOOR_PREREGISTRATION_PATH),
            "sha256": FLOOR_PREREGISTRATION_SHA256,
            "parents": {"holdout": sha256_file(HOLDOUT_PREREGISTRATION_PATH),
                        "ident": sha256_file(PREREGISTRATION_PATH),
                        "bu": sha256_file(BU_AMENDMENT_PATH)},
        },
        "floor_rule": floor_rule(),
        "margin_definition": {"path": repo_relative(MARGIN_AMENDMENT_PATH),
                              "sha256": sha256_file(MARGIN_AMENDMENT_PATH),
                              "status": margin_status},
        "floor_status": floor_status,
        "confirm": confirmation,
        "oracle": oracle,
        "first_pass": {
            "engine_commit": "79acded",
            "defect": "each CIGAR run was credited whole to the ayah it started in, and "
                      "aligned characters were counted rather than words",
            "fix": "runs are split at ayah boundaries and the ayah with the most aligned "
                   "words is credited",
            "rules_changed": False,
        },
        "margin_fix": {
            "engine_commit": "c993a79",
            "defect": "the runner-up search took one alignment per free range, so a location "
                      "beside a same-class optimum was lost whenever that optimum was the "
                      "best class: a tied location, or a worse straddling one",
            "fix": "Every free word run of every shortlisted window is now searched span by "
                   "span with branch and bound: a span's score cannot exceed its parent's, so "
                   "the search skips what cannot beat the best competitor yet and returns as "
                   "soon as a span is admissible. A sliced ayah keeps its origin, so its "
                   "words stay its own.",
            "affected": "Margins only: b_u, its precision table and the assigned counts. "
                        "Floors and b are unchanged, because they read the decoy score and Δ.",
        },
        "parameters": {
            "k": k, "gram": GRAM, "grid": list(GRID), "floor_bins": list(FLOOR_BINS),
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
            "look": {"path": repo_relative(DECODES_PATH),
                     "sha256": sha256_file(DECODES_PATH), "clips": len(look)},
            "real_mistake": {"path": repo_relative(REAL_MISTAKE_DECODES_PATH),
                             "sha256": sha256_file(REAL_MISTAKE_DECODES_PATH),
                             "clips": len(clip_list)},
            "noise": {"path": repo_relative(NOISE_DECODES_PATH),
                      "sha256": sha256_file(NOISE_DECODES_PATH), "recordings": len(noise),
                      "mode": NOISE_DECODE_MODE},
            # The S5b confirmation stores, exactly as the block that scored them recorded them.
            **{f"confirm_{arm}": store
               for arm, store in confirmation["stores"].items()},
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
        "look": {"clips": len(raw), "retention": retention,
                 "relabel_rate_at_b": relabel_rate,
                 "bins": {bin_: sum(1 for row in raw if floor_bin(row.length) == bin_)
                          for bin_ in FLOOR_BINS},
                 "tail": {"clips": len(tail), "retention": tail_retention,
                          "relabel_rate": tail_relabel}},
        "mistake": {"clips": len(mistakes),
                    "passed": sum(row.floor_ok for row in mistakes),
                    "relabelled_at_b": sum(relabelled(row, bars["b"])
                                           for row in mistakes)
                    if bars["b"] is not None else None},
        "unlabelled_unclean": {"source": "tadabur.tlog_overlap.iter_pool_rows",
                               "usable": unlabelled},
        "pool_lengths": pool_lengths,
        "noise": {"recordings": len(noise), "pass": noise_pass,
                  "floor_passed": conventions["B"]["passed"],
                  "rule_pass": {"clips": len(noise), "passed": rule_passed,
                                "pass": rule_passed / len(noise)},
                  "conventions": conventions,
                  "samples_sha256": {"recordings": len(noise_store["items"]),
                                     "mismatched": 0}},
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


def pool_length_lines(report: Mapping[str, Any]) -> list[str]:
    """The page's pool section: an indicative VAD speech histogram, not decode-bin membership.

    The pool's decode lengths do not exist until S7 runs, and `speech_s` is not one of them:
    characters per second vary between clips, so speech seconds say nothing certain about a
    row's floor bin (a short row can still decode past 160 characters).
    """
    pool = report["pool_lengths"]
    usable = pool["usable"]

    def share(count: int) -> str:
        return _pct(count / usable if usable else None)

    return [
        "The floors are keyed on the **decode's** character length. The look half's longest "
        "decode is 138 characters, so none reached `161-` and that bin has no floor. The pool's "
        "decode lengths do not exist until S7 runs. Its only committed length fact is the "
        "pass-1 VAD speech seconds (`speech_s`; the frozen row does not carry the clip's "
        "`duration_s`), which is **not** the decode's length: characters per second vary "
        "between clips, so a row with little speech can still decode past 160 characters. Read "
        "this as an indicative distribution of speech seconds, with no inference about a row's "
        "floor bin:",
        "",
        "| VAD speech | rows | share |",
        "|---|---|---|",
        f"| ≤ 20 s | {usable - pool['over_20_s']:,} | {share(usable - pool['over_20_s'])} |",
        f"| > 20 s | {pool['over_20_s']:,} | {share(pool['over_20_s'])} |",
        f"| > 30 s | {pool['over_30_s']:,} | {share(pool['over_30_s'])} |",
        f"| > 60 s | {pool['over_60_s']:,} | {share(pool['over_60_s'])} |",
        "",
        f"The share of the {usable:,} usable rows whose decode lands in `161-` is unknown until "
        "S7 decodes and measures normalised lengths (#192); this section claims nothing about "
        "it. Until then no row in that bin is assigned or relabelled.",
    ]


def confirm_lines(report: Mapping[str, Any]) -> list[str]:
    """The page's S5b sections: the five bars, their evidence and the margin's bound.

    Every number is read from the report; ``passing`` lists the items the rule let through, so
    a stop can be inspected without rerunning anything.
    """
    confirm, bars = report["confirm"], report["confirm"]["arms"]
    lines = [
        "## The confirmation (S5b, #204)",
        "",
        "The floor is the per-decode rule of "
        "[`docs/tlog-ident-floor-preregistration.md`](tlog-ident-floor-preregistration.md) "
        f"(sha256 `{report['floor_rule']['sha256'][:12]}`): a decode is `non_recitation` when "
        f"its normalised length is under `L_min = {L_MIN}` characters or its real best is "
        f"within `DELTA = {DELTA:.2f}` of its own decoy best, both from the same shortlisted "
        f"search. The three fresh sets were drawn and committed before decoding, under the salt "
        f"`{confirm['salt']}`.",
        "",
        "Three disclosures are corrected in "
        "[`docs/tlog-ident-floor-erratum.md`](tlog-ident-floor-erratum.md) (2026-10-27): the "
        "pre-registration's look figures are S5's measurement at `dc602df`, one search change "
        "(`659bce0`, the decoy shortlist ranked by its own index) away from the 99.63%/99.50% "
        "re-measured here; the Quran-text guard the draw used does not match Urdu/Persian "
        "letter variants, and the folded check excludes none of the 1,000 drawn sentences; and "
        "the provenance chain cites branch commits, reachable after a squash merge only "
        "through the PR ref.",
        "",
        "| arm | clips | passed | rate | Wilson 95% | limit | fired |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ("freesound", "cv_ar", "cv_other"):
        entry = bars[arm]
        low, high = entry["wilson"]
        lines.append(f"| {arm} | {entry['clips']:,} | {entry['passed']} | "
                     f"{_pct(entry['pass'])} | {_pct(low)}–{_pct(high)} | "
                     f"{_pct(entry['limit'])} | {entry['fired']} |")
    lines += [
        f"| real-mistake clips | {bars['mistakes']['clips']} | {bars['mistakes']['passed']} | "
        f"— | — | all | {bars['mistakes']['fired']} |",
        f"| look retention | {report['look']['clips']:,} | — | "
        f"{_pct(bars['look_retention']['pass'])} | — | "
        f"{_pct(bars['look_retention']['limit'])} | {bars['look_retention']['fired']} |",
        "",
        f"**Status: `{confirm['status']}`.**",
        "",
        "Per language (reported, not a bar):",
        "",
        "| arm | locale | clips | passed | rate | Wilson 95% |",
        "|---|---|---|---|---|---|",
    ]
    for primary in sorted(confirm["per_language"]):
        entry = confirm["per_language"][primary]
        wilson = entry["wilson"]
        interval = "—" if wilson is None else f"{_pct(wilson[0])}–{_pct(wilson[1])}"
        lines.append(f"| {primary} | {entry['locale']} | {entry['clips']} | "
                     f"{entry['passed']} | {_pct(entry['pass'])} | {interval} |")
    lines += [
        "",
        "Which clause caught the rejects (`length` = under `L_min`, `margin` = within `DELTA`):",
        "",
        "| arm | rejected | length | margin | both |",
        "|---|---|---|---|---|",
        *[f"| {arm} | {bars[arm]['clips'] - bars[arm]['passed']} | {catches['length']} | "
          f"{catches['margin']} | {catches['both']} |"
          for arm, catches in ((arm, confirm["catches"][arm])
                               for arm in ("freesound", "cv_ar", "cv_other"))],
        "",
        "The items that got through:",
        "",
        "| arm | key | length | class | span | decode |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ("freesound", "cv_ar", "cv_other"):
        items = confirm["passing"][arm]
        if not items:
            lines.append(f"| {arm} | — | — | — | — | — |")
        for item in items:
            span = item["span"]
            where = "—" if span is None else (f"{span['surah']}:{span['ayah_from']}."
                                              f"{span['word_from']}–{span['ayah_to']}."
                                              f"{span['word_to']}")
            decode = item["decode"].replace("|", "\\|") or "&nbsp;"
            lines.append(f"| {arm} | `{item['key']}` | {item['length']} | "
                         f"{item['class_label']} | {where} | `{decode}` |")
    return lines


def render(report: Mapping[str, Any]) -> str:
    """The generated page; every number is read from the report."""
    parameters, look, bars = report["parameters"], report["look"], report["bars"]
    stop = report["stop_conditions"]
    margin = report.get("margin_definition", {})
    provisional = " (PROVISIONAL)" if bars.get("provisional") else ""
    lines = [
        "# Ayah identification on Tarteel v1 look (#190, floor #204)",
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
        f"- floors (history): type-7 {parameters['floor_percentile']}th percentile of the best "
        f"decoy score per length bin",
        f"- floor: the per-decode rule of "
        f"[`docs/tlog-ident-floor-preregistration.md`](tlog-ident-floor-preregistration.md) "
        f"(sha256 `{report['floor_rule']['sha256'][:12]}`): under `L_min = {L_MIN}` characters, "
        f"or within `DELTA = {DELTA:.2f}` of the decode's own decoy best",
        f"- margin competitor: word spans, first optimal alignment, per "
        f"[`docs/tlog-ident-margin-amendment.md`](tlog-ident-margin-amendment.md) (sha256 "
        f"`{report['margin_definition']['sha256'][:12]}`); it can overstate the margin, never "
        f"understate it",
        "",
        "## Superseded calibration passes",
        "",
        f"The first look pass (engine `{report['first_pass']['engine_commit']}`) had a "
        f"span-attribution defect: {report['first_pass']['defect']}. Before this report the "
        f"engine was brought to the frozen rule: {report['first_pass']['fix']}. No rule, grid "
        "value or input changed, and the same committed look decodes were rescored; only the "
        "numbers below count.",
        "",
        f"A second look pass (engine `{report['margin_fix']['engine_commit']}`) overstated the "
        f"margin: {report['margin_fix']['defect']}. {report['margin_fix']['fix']} "
        f"{report['margin_fix']['affected']} The same committed inputs were rescored; only the "
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
        f"| Freesound confirmation | {report['decodes']['confirm_freesound']['items']:,} "
        f"recordings, `{report['decodes']['confirm_freesound']['mode']}` |",
        f"| Common Voice confirmation | "
        f"{report['decodes']['confirm_cv_ar']['items']:,} non-Quran Arabic + "
        f"{report['decodes']['confirm_cv_other']['items']:,} non-Arabic clips, "
        f"`{report['decodes']['confirm_cv_ar']['mode']}` |",
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
    lines += [f"| `{bin_}` | {report['floors'][bin_]:.6f} |"
              for bin_ in FLOOR_BINS if bin_ in report["floors"]]
    lines += [
        "",
        f"**b = {bars['b']:.2f}{provisional}**" if bars["b"] is not None
        else "**b: none on the grid**",
        f" ({_pct(bars['look_relabel_at_b'])} of look relabelled, "
        f"{bars['mistake_relabels_at_b']} of the real-mistake clips relabelled).",
        *([f"**b_u = {bars['b_u']:.2f}{provisional}**: {_pct(bars['look_assigned_at_b_u'])} "
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
        "## The margin's bound (S5b)",
        "",
        f"`margin_definition.status = \"{margin['status']}\"`. The pinned competitor can "
        f"overstate a margin, never understate it; the one measurement of that bound on the "
        f"look half (artifact sha256 `{report['oracle']['artifact']['sha256'][:12]}`) gives "
        f"`b_u_oracle = {_bar(report['oracle']['b_u_oracle'])}` against the pinned "
        f"`b_u = {_bar(bars['b_u'])}`, changed {report['oracle']['clips_m_changed']} clips' "
        f"margins and {report['oracle']['assignments_changed_at_b_u']} assignments at b_u. It "
        "never fed a bar or a label.",
        "",
        "## The pool's VAD speech duration (indicative)",
        "",
        *pool_length_lines(report),
        "",
        "## For S7: the row contract and the decode mode",
        "",
        "- **Row contract.** `tadabur.tlog_ident.identification_record` is the one place the "
        "committed row's field names are fixed: the span, `straddles`, `class_label`, `score`, "
        "`decoy`, `delta`, `margin`, `margin_class`, `id_floor_ok`, `overlaps_filename`, "
        "`filename_label`, `label_source`, `assigned_label`, plus `floor_rule` (the "
        "pre-registration's path and SHA-256). `id_floor_ok` is the S5b recitation verdict: the "
        "decode is at least `L_min = 8` characters and its real best is at least `DELTA = 0.30` "
        "above its own decoy best. Every row names the rule that produced it, so binned and "
        "rule verdicts cannot be mixed silently.",
        "- **Thresholds.** `tadabur.tlog_ident.load_calibration()` returns `(bars, k)` and "
        "refuses any report whose `floor_status` is not `\"confirmed\"`, whose "
        "`margin_definition.status` is not `\"final\"` (an undecided `b_u` waits for the "
        "owner), or whose margin definition has no recorded owner approval "
        "(`tlog/ident/margin_approval.json`, the frozen floor document's §7 check-in, written "
        "by `python -m tadabur.tlog_ident approve`). The retired binned floors stay in the "
        "report as history and are not returned. `load_identifier()` is what S6/S7 call: it "
        "applies those gates and binds the database to the report — the references file's "
        "SHA-256 and counts, the floor rule and the `L_min`/`DELTA` the pinned document states, "
        "and the edlib version.",
        f"- **Decode mode.** The frozen rule fixes a whole-clip greedy decode "
        f"(`docs/tlog-holdout-preregistration.md`), and both look decodes and these floors are "
        f"whole-clip. The pool is not: "
        f"{report['pool_lengths']['over_60_s']:,} usable rows carry over 60 s of speech, the "
        f"longest {report['pool_lengths']['longest_s']:,.0f} s, which a whole pass cannot "
        f"decode within "
        f"the VRAM budget (S5's Freesound pool already needed `{report['decodes']['noise']['mode']}`, "
        f"as its own row above says). `identify` takes a bare string and cannot tell the modes "
        f"apart. S7 must either obtain an amendment of the decode mode or record `decode_mode` "
        f"beside every identification and report the whole-clip and streamed populations "
        f"separately; S5 decides neither.",
        "",
        "## Floor status",
        "",
        f"`{report['floor_status']}`",
        "",
        *confirm_lines(report),
        "",
        "## S5's binned stop (history)",
        "",
        "The retired binned rule's record on the 445-recording pool. Its stop fired under all "
        "three conventions, which is what triggered the S5b re-pre-registration above.",
        "",
        "| convention | recordings | passed | pass |",
        "|---|---|---|---|",
        *[f"| {key} | {value['recordings']} | {value['passed']} | {_pct(value['pass'])} |"
          for key, value in report["noise"]["conventions"].items()],
        "",
        "B backs a row in an empty bin off to the nearest calibrated floor, A fails it, C "
        "excludes it.",
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
        f"| 445 pool, the rule's pass | {_pct(report['noise']['rule_pass']['pass'])} "
        f"({report['noise']['rule_pass']['passed']}/{report['noise']['rule_pass']['clips']}) |",
        f"| 445 pool, the binned rule's pass | {_pct(report['noise']['pass'])} |",
        *([f"| exhaustive-decoy floors (sensitivity) | retention "
           f"{_pct(report['exhaustive_decoy']['look_retention'])} |"]
          if "exhaustive_decoy" in report else []),
        f"| throughput | {report['throughput']['ms_per_clip']:.2f} ms per clip |",
        "",
        f"The noise population is decoded in `{report['decodes']['noise']['mode']}`, while look "
        "and the floors use whole-clip `spans`; the noise rate compares a different decode "
        "mode, not the calibration population's.",
        "",
        "## S5's binned stop conditions (history)",
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
    oracle = commands.add_parser(
        "oracle", help="run the character-start margin oracle on the look half (recorded only)")
    oracle.add_argument("--limit", type=int,
                        help="score only the first N look clips, print the projection and write "
                             "nothing (the cost check before the full run)")
    oracle.add_argument("--out", type=Path, default=ORACLE_PATH)
    approve = commands.add_parser(
        "approve", help="record the owner's approval of the pinned margin definition (the "
                        "owner runs this)")
    approve.add_argument("--issue", type=int, default=204)
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
    elif args.command == "oracle":
        database = load_references()
        # The committed calibration's k, read as data: load_calibration refuses a stopped
        # report and an unapproved margin, while the oracle must stay runnable before approval.
        k = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["parameters"]["k"]
        look = load_look_decodes()
        mistakes = _mistake_identifications(database, k)
        sampled = look[:args.limit] if args.limit else look
        payload = oracle_pass(database, sampled, mistakes, k=k)
        timing = {"clips": len(sampled), "seconds": payload["seconds"],
                  "ms_per_clip": round(1000 * payload["seconds"] / len(sampled), 3)}
        if args.limit:
            timing["projected_seconds"] = round(
                payload["seconds"] / len(sampled) * len(look), 1)
            print(json.dumps(timing, indent=1))
        else:
            write_text_atomically(Path(args.out),
                                  json.dumps(payload, ensure_ascii=False, indent=1,
                                             sort_keys=True) + "\n")
            print(json.dumps({**timing, "b_u": payload["b_u"],
                              "b_u_oracle": payload["b_u_oracle"],
                              "clips_m_changed": payload["clips_m_changed"],
                              "assignments_changed_at_b_u":
                                  payload["assignments_changed_at_b_u"],
                              "out": str(args.out)}, indent=1))
    elif args.command == "approve":
        record = approve_margin(args.issue)
        print(json.dumps({**record, "written": str(MARGIN_APPROVAL_PATH)}, indent=1))
    else:
        report = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        write_text_atomically(DOC_PATH, render(report))
        print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()

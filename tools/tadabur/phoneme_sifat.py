"""Articulatory phoneme similarity — Python port of Muraja's ``PhonemeSifat.swift``.

Classifies each of the 29 core Arabic consonants (including ``ا``) by *makhraj*
(place of articulation) and *sifat* (manner attributes), then scores how close
two consonants sound. The ``.balanced`` scorer feeds this into Smith-Waterman as
the graduated cost of substituting one consonant for another, so acoustically
close pairs (e.g. ``ذ↔ز``) cost far less than distant ones (e.g. ``ف↔ه``).

Ported verbatim from ``Muraja/ios/HifzGuide/FollowAlong/PhonemeSifat.swift`` and
validated against ``PhonemeSifatTests.swift`` — do not retune the tables here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Makhraj(IntEnum):
    """Place of articulation, grouped from classical makharij into 8 categories
    that align with the ASR model's confusion patterns."""

    JAWF = 0        # cavity (madd vowels): ا
    HALQ_DEEP = 1   # deep throat (glottal/pharyngeal): ء ع ه
    HALQ_UPPER = 2  # upper throat (uvular/pharyngeal fricatives): ح خ غ
    BACK_TONGUE = 3  # tongue root (uvular/velar stops): ق ك
    MID_TONGUE = 4  # mid tongue (palatal): ج ش ي
    TIP_TONGUE = 5  # tongue tip (dental/alveolar): ت د ط ن ل ر
    SIBILANT = 6    # blade/edge (interdental + sibilant fricatives): ذ ث ظ ز س ص ض
    SHAFAWI = 7     # lips (labial): ب ف م و


# Symmetric articulatory distance between makhraj groups (0.0 = same, ~1.0 =
# maximally distant), indexed by ``Makhraj`` raw values. Distances reflect
# articulatory proximity based on classical tajweed makharij.
#                jawf  halqD halqU backT midT  tipT  sibil shaf
_MAKHRAJ_DISTANCE: tuple[tuple[float, ...], ...] = (
    (0.0,  0.8,  0.8,  0.8,  0.8,  0.8,  0.8,  0.8),  # jawf
    (0.8,  0.0,  0.25, 0.4,  0.6,  0.7,  0.6,  0.8),  # halqDeep
    (0.8,  0.25, 0.0,  0.3,  0.5,  0.6,  0.5,  0.7),  # halqUpper
    (0.8,  0.4,  0.3,  0.0,  0.3,  0.4,  0.5,  0.6),  # backTongue
    (0.8,  0.6,  0.5,  0.3,  0.0,  0.3,  0.4,  0.6),  # midTongue
    (0.8,  0.7,  0.6,  0.4,  0.3,  0.0,  0.25, 0.5),  # tipTongue
    (0.8,  0.6,  0.5,  0.5,  0.4,  0.25, 0.0,  0.6),  # sibilant
    (0.8,  0.8,  0.7,  0.6,  0.6,  0.5,  0.6,  0.0),  # shafawi
)


def makhraj_distance(a: Makhraj, b: Makhraj) -> float:
    """Articulatory distance between two makhraj groups, in ``[0.0, 1.0]``."""
    return _MAKHRAJ_DISTANCE[a][b]


@dataclass(frozen=True)
class PhonemeSifa:
    """Articulatory attributes for a single consonant.

    ``makhraj`` is the place of articulation; the remaining ten fields are the
    binary/ternary sifat categories used for similarity (``CATEGORY_COUNT`` of
    them). Voicing/manner/emphasis/contact are small enums; the rest are flags.
    """

    makhraj: Makhraj
    hams: bool          # voiceless (True=hams) vs voiced (False=jahr)
    shadeed: str        # manner: "shadeed" | "between" | "rikhw"
    mofakham: bool      # emphatic (True) vs light (False)
    motbaq: bool        # tongue contact: closed (True=motbaq) vs open (monfateh)
    safeer: bool = False
    qalqala: bool = False
    tikraar: bool = False
    tafashie: bool = False
    istitala: bool = False
    ghonna: bool = False

    # Number of sifat categories (excluding makhraj) compared for similarity.
    CATEGORY_COUNT = 10

    def matching_count(self, other: PhonemeSifa) -> int:
        """How many of the ten sifat categories match between two phonemes."""
        return sum(
            getattr(self, field) == getattr(other, field)
            for field in (
                "hams", "shadeed", "mofakham", "motbaq", "safeer",
                "qalqala", "tikraar", "tafashie", "istitala", "ghonna",
            )
        )


def _sifa(
    makhraj: Makhraj,
    hams: bool,
    shadeed: str,
    mofakham: bool,
    motbaq: bool,
    *,
    safeer: bool = False,
    qalqala: bool = False,
    tikraar: bool = False,
    tafashie: bool = False,
    istitala: bool = False,
    ghonna: bool = False,
) -> PhonemeSifa:
    return PhonemeSifa(
        makhraj, hams, shadeed, mofakham, motbaq,
        safeer=safeer, qalqala=qalqala, tikraar=tikraar,
        tafashie=tafashie, istitala=istitala, ghonna=ghonna,
    )


# Static sifat classification for the 29 core consonant phonemes, keyed by the
# consonant character. Based on ``obadx/quran-transcript`` phonetic groups.
_SIFAT_TABLE: dict[str, PhonemeSifa] = {
    "\u0621": _sifa(Makhraj.HALQ_DEEP, False, "shadeed", False, False),  # ء hamza
    "\u0628": _sifa(Makhraj.SHAFAWI, False, "shadeed", False, False, qalqala=True),  # ب baa
    "\u062A": _sifa(Makhraj.TIP_TONGUE, True, "shadeed", False, False),  # ت taa
    "\u062B": _sifa(Makhraj.SIBILANT, True, "rikhw", False, False),  # ث thaa
    "\u062C": _sifa(Makhraj.MID_TONGUE, False, "shadeed", False, False, qalqala=True),  # ج jeem
    "\u062D": _sifa(Makhraj.HALQ_UPPER, True, "rikhw", False, False),  # ح haa
    "\u062E": _sifa(Makhraj.HALQ_UPPER, True, "rikhw", True, False),  # خ khaa
    "\u062F": _sifa(Makhraj.TIP_TONGUE, False, "shadeed", False, False, qalqala=True),  # د daal
    "\u0630": _sifa(Makhraj.SIBILANT, False, "rikhw", False, False),  # ذ thaal
    "\u0631": _sifa(Makhraj.TIP_TONGUE, False, "between", True, False, tikraar=True),  # ر raa
    "\u0632": _sifa(Makhraj.SIBILANT, False, "rikhw", False, False, safeer=True),  # ز zay
    "\u0633": _sifa(Makhraj.SIBILANT, True, "rikhw", False, False, safeer=True),  # س seen
    "\u0634": _sifa(Makhraj.MID_TONGUE, True, "rikhw", False, False, tafashie=True),  # ش sheen
    "\u0635": _sifa(Makhraj.SIBILANT, True, "rikhw", True, True, safeer=True),  # ص saad
    "\u0636": _sifa(Makhraj.SIBILANT, False, "rikhw", True, True, istitala=True),  # ض daad
    "\u0637": _sifa(Makhraj.TIP_TONGUE, True, "shadeed", True, True, qalqala=True),  # ط taa mofakhama
    "\u0638": _sifa(Makhraj.SIBILANT, False, "rikhw", True, True),  # ظ zaa mofakhama
    "\u0639": _sifa(Makhraj.HALQ_DEEP, False, "between", False, False),  # ع ayn
    "\u063A": _sifa(Makhraj.HALQ_UPPER, False, "rikhw", True, False),  # غ ghyn
    "\u0641": _sifa(Makhraj.SHAFAWI, True, "rikhw", False, False),  # ف faa
    "\u0642": _sifa(Makhraj.BACK_TONGUE, False, "shadeed", True, False, qalqala=True),  # ق qaf
    "\u0643": _sifa(Makhraj.BACK_TONGUE, True, "shadeed", False, False),  # ك kaf
    "\u0644": _sifa(Makhraj.TIP_TONGUE, False, "between", False, False),  # ل lam
    "\u0645": _sifa(Makhraj.SHAFAWI, False, "between", False, False, ghonna=True),  # م meem
    "\u0646": _sifa(Makhraj.TIP_TONGUE, False, "between", False, False, ghonna=True),  # ن noon
    "\u0647": _sifa(Makhraj.HALQ_DEEP, True, "rikhw", False, False),  # ه haa
    "\u0648": _sifa(Makhraj.SHAFAWI, False, "rikhw", False, False),  # و waw
    "\u064A": _sifa(Makhraj.MID_TONGUE, False, "rikhw", False, False),  # ي yaa
    "\u0627": _sifa(Makhraj.JAWF, False, "rikhw", False, False),  # ا alif
}


# Consonant pairs that are acoustically close and commonly confused by the ASR
# model. In balanced mode these bypass the phoneme gate (the word can still be
# graded "correct") but still incur a graduated-mismatch penalty. In strict mode
# no soft pairs exist — all mismatches are hard.
_BALANCED_SOFT_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("\u0630", "\u0632"),  # ذ ↔ ز
        ("\u062A", "\u0637"),  # ت ↔ ط
        ("\u0636", "\u0638"),  # ض ↔ ظ
        ("\u0643", "\u0642"),  # ك ↔ ق
        ("\u0633", "\u0635"),  # س ↔ ص
        ("\u062D", "\u0647"),  # ح ↔ ه
    )
)


def is_soft_mismatch(a: str, b: str, soft_pairs_enabled: bool) -> bool:
    """Whether ``a`` and ``b`` form a balanced-mode soft mismatch pair.

    Soft pairs bypass the phoneme gate but still reduce the score. Controlled by
    ``ScoringParameters.soft_pairs_enabled``: disabled in strict (all mismatches
    hard), enabled in balanced/lenient. Order-independent.
    """
    return soft_pairs_enabled and frozenset((a, b)) in _BALANCED_SOFT_PAIRS


def _contrast_label(pair: frozenset[str]) -> str:
    """Canonical, order-independent label for a soft pair: its two characters
    joined by ``↔`` in Unicode-codepoint order (e.g. ``ذ↔ز``)."""
    return "\u2194".join(sorted(pair))


def soft_pair_contrast(a: str, b: str) -> str | None:
    """The canonical contrast label if ``{a, b}`` is a balanced soft pair, else
    ``None``. Reuses ``_BALANCED_SOFT_PAIRS`` so the vocabulary is never
    re-derived. Order-independent; the label is codepoint-ordered."""
    pair = frozenset((a, b))
    if pair not in _BALANCED_SOFT_PAIRS:
        return None
    return _contrast_label(pair)


def soft_pair_contrasts() -> frozenset[str]:
    """The canonical labels of all six balanced soft-pair contrasts."""
    return frozenset(_contrast_label(pair) for pair in _BALANCED_SOFT_PAIRS)


def phoneme_similarity(a: str, b: str) -> float | None:
    """Articulatory similarity between two consonants, in ``[0.0, 1.0]``.

    ``1.0`` means identical phoneme, ``0.0`` maximally dissimilar. Combines
    makhraj (place) distance at 60% weight with sifat feature matching at 40%.
    Makhraj is weighted more heavily because many distant pairs share sparse
    "both false" sifat flags, which would otherwise inflate their similarity.
    Returns ``None`` if either character is not a core consonant.
    """
    sifa_a = _SIFAT_TABLE.get(a)
    sifa_b = _SIFAT_TABLE.get(b)
    if sifa_a is None or sifa_b is None:
        return None
    makhraj_sim = 1.0 - makhraj_distance(sifa_a.makhraj, sifa_b.makhraj)
    sifat_sim = sifa_a.matching_count(sifa_b) / PhonemeSifa.CATEGORY_COUNT
    return 0.6 * makhraj_sim + 0.4 * sifat_sim


def graduated_mismatch_score(
    a: str,
    b: str,
    worst_penalty: float = -0.5,
    best_mismatch: float = 0.0,
    fallback: float = -0.5,
    lenient: bool = False,
) -> float:
    """Graduated substitution score for a consonant pair, from articulatory
    similarity.

    For core consonant pairs the score interpolates linearly between
    ``worst_penalty`` (similarity 0) and ``best_mismatch`` (similarity 1). For
    non-consonant pairs it falls back to ``fallback``. In ``lenient`` mode a
    highly similar pair (similarity ≥ 0.9) instead earns 90% of a perfect match.
    """
    sim = phoneme_similarity(a, b)
    if sim is None:
        return fallback
    if lenient and sim >= 0.9:
        return 0.9
    return worst_penalty + sim * (best_mismatch - worst_penalty)

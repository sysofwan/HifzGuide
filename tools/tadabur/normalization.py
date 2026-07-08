"""Phoneme group normalization — Python port of Muraja's ``PhonemeNormalization.swift``.

Collapses runs of the same core phoneme character (a consonant or madd marker,
optionally carrying one diacritic) into a single representative character, and
maps tajweed ghunna variants (``۾→م``, ``ں→ن``) back to their base consonant.
This is the normalization the ``.balanced`` scorer applies to both the model's
decoded phonemes and the quran-transcript reference before alignment, so the two
sides are compared on the same footing.

Swift iterates over *Characters* (extended grapheme clusters), where a base
letter plus its combining diacritic form a single unit (e.g. ``بَ`` is one
Character, not two). We reproduce that by grouping each base scalar with the
combining marks that follow it (``unicodedata.combining`` is non-zero).

Shared with the ``.balanced`` scorer port (issue #3): keep this module the single
home for phoneme normalization.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Bump whenever the normalization behaviour changes (grouping rules, tajweed
# folds, residual set). Downstream caches key their validity on this so a cache
# produced by an older algorithm is rebuilt rather than silently reused.
ALGORITHM_VERSION = "1"

# Core phoneme scalars (consonants + madd markers) used for group classification.
_CORE_SCALARS: frozenset[str] = frozenset("ءبتثجحخدذرزسشصضطظعغفقكلمنهوياۥۦ۾ںـٲ")

# Residual diacritic scalars that can follow a core-char run:
# fatha, damma, kasra, ڇ, ؙ, ۪, ۜ.
_RESIDUAL_SCALARS: frozenset[str] = frozenset(
    "\u064E\u064F\u0650\u0687\u0619\u06EA\u06DC"
)

# Tajweed variant → base consonant. The phoneme data marks ghunna (nasal) sounds
# with tajweed-specific characters, but the Wav2Vec2-BERT model emits the regular
# consonant, so we fold the variants onto their base.
_TAJWEED_EQUIVALENTS: dict[str, str] = {
    "\u06FE": "\u0645",  # ۾ (ghunna mim) → م
    "\u06BA": "\u0646",  # ں (ghunna nun) → ن
}


@dataclass(frozen=True)
class PhonemeNormalization:
    """A normalized phoneme string plus the origin of each normalized character.

    ``offset_map[i]`` is the half-open range ``(start, end)`` of *grapheme-cluster*
    indices in the original string that the i-th normalized character represents,
    matching the Swift port's Character-index offsets.
    """

    normalized: str
    offset_map: list[tuple[int, int]]


def _grapheme_clusters(text: str) -> list[list[str]]:
    """Group ``text`` into Swift-style Characters: a base scalar plus trailing
    combining marks. A leading combining mark forms its own cluster."""
    clusters: list[list[str]] = []
    for ch in text:
        if clusters and unicodedata.combining(ch):
            clusters[-1].append(ch)
        else:
            clusters.append([ch])
    return clusters


def _folded_core(scalar: str) -> str:
    """The tajweed-folded core for ``scalar`` (identity if not a core scalar)."""
    return _TAJWEED_EQUIVALENTS.get(scalar, scalar)


def normalize_phonemes(text: str) -> PhonemeNormalization:
    """Collapse each phoneme group into a single representative core character.

    A *group* is quran-transcript's ``chunck_phonemes`` unit: one or more
    consecutive repetitions of the **same** core scalar (a shadda-style run such
    as ``ببُ``), optionally carrying **one** trailing residual/diacritic — the
    regex ``(c+)[residuals]?``. A diacritic ends the run, so cores separated by a
    diacritic (``بَبُ`` → ``بَ``,``بُ``) stay in distinct groups. Tajweed variants
    (``۾``,``ں``) fold onto their base consonant before grouping. Spaces are word
    boundaries and are preserved verbatim.
    """
    clusters = _grapheme_clusters(text)
    normalized: list[str] = []
    offset_map: list[tuple[int, int]] = []
    n = len(clusters)
    i = 0

    while i < n:
        base = clusters[i][0]

        if base == " ":
            normalized.append(" ")
            offset_map.append((i, i + 1))
            i += 1
            continue

        if base in _CORE_SCALARS:
            group_start = i
            core = _folded_core(base)
            # The residual is either a combining mark attached to a cluster in the
            # run, or a standalone residual scalar (e.g. non-combining ``ڇ``)
            # trailing it. Either terminates the group; the ``c+`` run consumes
            # only bare clusters of the same folded core until then.
            carried_residual = len(clusters[i]) > 1
            i += 1
            while not carried_residual and i < n:
                nxt = clusters[i][0]
                if nxt in _CORE_SCALARS and _folded_core(nxt) == core:
                    carried_residual = len(clusters[i]) > 1
                    i += 1
                elif nxt in _RESIDUAL_SCALARS:  # standalone residual (e.g. ڇ)
                    i += 1
                    break
                else:
                    break
            normalized.append(core)
            offset_map.append((group_start, i))
            continue

        # Stray residual without a preceding core, or unknown character — skip.
        i += 1

    return PhonemeNormalization("".join(normalized), offset_map)


def map_to_original(normalized_idx: int, offset_map: list[tuple[int, int]]) -> int:
    """Map a normalized character index back to the start of its original range."""
    if normalized_idx < 0:
        return 0
    if normalized_idx >= len(offset_map):
        return offset_map[-1][1] if offset_map else 0
    return offset_map[normalized_idx][0]


def map_to_original_end(normalized_idx: int, offset_map: list[tuple[int, int]]) -> int:
    """Map a normalized (exclusive) end index back to the original end index."""
    if normalized_idx <= 0:
        return 0
    if normalized_idx > len(offset_map):
        return offset_map[-1][1] if offset_map else 0
    return offset_map[normalized_idx - 1][1]

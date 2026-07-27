"""Phoneme group normalization — Python port of Muraja's ``PhonemeNormalization.swift``.

Collapses runs of the same **bare** core phoneme character (a consonant or madd
marker with no attached diacritic) into a single representative character, and
maps tajweed ghunna variants (``۾→م``, ``ں→ن``) back to their base consonant.
This is the normalization the ``.balanced`` scorer applies to both the model's
decoded phonemes and the quran-transcript reference before alignment, so the two
sides are compared on the same footing.

Swift iterates over *Characters* (extended grapheme clusters), where a base
letter plus its combining diacritic form a single unit (e.g. ``بَ`` is one
Character, not two). We reproduce that by grouping each base scalar with the
combining marks that follow it (``unicodedata.combining`` is non-zero).

Crucially — and this is where a faithful port differs from quran-transcript's
``chunck_phonemes`` — a *combining mark starts a NEW group*: when a run of bare
cores is followed by the **same** core carrying a diacritic, Swift **breaks
before consuming** that diacritic-bearing cluster and processes it as its own
group. So a shadda-style run such as ``رَببُ`` normalizes to ``ربب`` (not ``رب``)
and ``ببَ`` to ``بب`` (not ``ب``): shadda expansion stays DOUBLED here. It is the
downstream word scorer's ``shaddahSuppression`` (a gap-count exclusion in
``QuranFollowAlong+WordScoring.swift``), not this normalization, that neutralises
those doubled cores — so we must reproduce Swift's doubling verbatim or the
Smith-Waterman reference string, and thus the ``.balanced`` ``match_ratio``,
would diverge.

Shared with the ``.balanced`` scorer port (issue #3): keep this module the single
home for phoneme normalization.
"""

from __future__ import annotations

import bisect
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

# Bump whenever the normalization behaviour changes (grouping rules, tajweed
# folds, residual set). Downstream caches key their validity on this so a cache
# produced by an older algorithm is rebuilt rather than silently reused.
ALGORITHM_VERSION = "2"

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

    Mirrors Swift's ``normalizePhonemes`` loop verbatim. A core cluster opens a
    group; if that opening cluster is **bare** (no combining mark), the group
    greedily consumes following **bare** clusters of the same folded core, then
    absorbs at most one trailing standalone residual scalar (e.g. non-combining
    ``ڇ``). A same-core cluster that *carries* a combining mark does **not** join
    the run — Swift breaks before consuming it, so it becomes the start of a new
    group. That is why shadda-style expansions stay doubled: ``رَببُ`` → ``ربب``,
    ``ببَ`` → ``بب``, ``للَ`` → ``لل``. Tajweed variants (``۾``, ``ں``) fold onto
    their base consonant before grouping and comparison. Spaces are word
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
            # A combining mark on the opening cluster terminates the group
            # immediately (the diacritic acts as the residual); only a bare core
            # continues consuming its shadda-style run of same-folded-core bares.
            has_combining = len(clusters[i]) > 1
            i += 1
            if not has_combining:
                while i < n:
                    if _folded_core(clusters[i][0]) != core:
                        break
                    if len(clusters[i]) > 1:
                        # Same core but carrying a combining mark → NEW group.
                        # Swift breaks WITHOUT consuming it, keeping shadda
                        # expansion doubled. Leave it for the next iteration.
                        break
                    i += 1
                # Absorb one trailing standalone residual (e.g. non-combining ڇ).
                if i < n and clusters[i][0] in _RESIDUAL_SCALARS:
                    i += 1
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


def map_char_offsets(
    text: str, normalization: PhonemeNormalization, char_offsets: Sequence[int]
) -> list[int]:
    """Map character offsets in ``text`` to indices in its normalized form.

    ``offset_map`` is indexed by *grapheme cluster*, not character, so a raw offset
    (e.g. a word boundary in the phonetizer's output) must first be converted to a
    cluster index. Each returned index is the first normalized character whose source
    cluster range starts at or after that cluster — i.e. the normalized offset a slice
    should cut at. Offsets are clamped and forced non-decreasing so the returned
    sequence always describes a valid set of consecutive slices.
    """
    clusters = _grapheme_clusters(text)
    cluster_starts: list[int] = []
    pos = 0
    for cluster in clusters:
        cluster_starts.append(pos)
        pos += len(cluster)
    offset_map = normalization.offset_map

    mapped: list[int] = []
    previous = 0
    for raw in char_offsets:
        cluster_index = bisect.bisect_left(cluster_starts, min(max(raw, 0), len(text)))
        index = len(normalization.normalized)
        for i, (start, _end) in enumerate(offset_map):
            if start >= cluster_index:
                index = i
                break
        index = max(previous, min(index, len(normalization.normalized)))
        mapped.append(index)
        previous = index
    return mapped

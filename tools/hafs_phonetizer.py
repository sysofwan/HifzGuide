"""The repo's one entry point to quran-transcript's phonetizer: Hafs, in correct pausal form.

Every phonetized reference in the repo goes through :func:`phonetize` — the ayah phonemes
``generate_phonemes.py`` / ``generate_quran_db.py`` ship in ``quran.db``, the Tadabur filter's
reference cache, the waqf-segment references and the truth-site runs — so they share one
moshaf configuration and one set of corrections on top of ``quran_phonetizer``.
``test_hafs_phonetizer.py`` fails if any other module imports the phonetizer directly.

The phonetizer treats the end of its input as a waqf. quran-transcript 0.5.2 gets one
pausal form wrong: its ``MaddAlewad`` step turns *every* final tanween fatha into fatha +
alif before ``CleanEnd`` / ``NormalizeTaa`` run, so a final ``ةً`` comes out ``تَاا``
instead of ``ه`` (``رَحْمَةًۭ`` -> ``رَحمَتَاا``, not ``رَحمَه``). :func:`pausal_taa_marbuta`
rewrites that ending before phonetizing. Upstream fixed it in 0.6.0
(``docs/pausal-taa-marbuta.md`` records why the pin stays at 0.5.2 for now); on a fixed
version the rewrite is a no-op on the output, mappings included.

The input must be quran-transcript's own Uthmani text (``Aya(...).uthmani`` or its words).
Other encodings of the same mushaf — the QPC spelling's open tanween (U+08F0, U+0657), its
sukun (U+06E1), ayah-end markers, digits — are outside the alphabet the phonetizer
understands: it passes them through into the phonemes or raises ``IndexError``, at every
position, not just at a waqf. :func:`phonetize` rejects them with ``ValueError`` instead.

quran-transcript is imported lazily, so importing this module costs nothing where it is
absent.
"""

from __future__ import annotations

import dataclasses
import re
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes
    from quran_transcript.phonetics.phonetizer import QuranPhoneticScriptOutput

# Hafs recitation, matching the reference labels the Muaalem model was trained on.
HAFS_MOSHAF = dict(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=6,
    madd_aared_len=2,
)

# Bump when a change in this module alters any phonetized output, so that caches of
# phonetized references (``tadabur.reference_phonemes``) are rebuilt rather than trusted.
# "1": the pausal taa marbuta fix (#100).
REVISION = "1"

_FATHA = "َ"
# A word ending in taa marbuta + tanween fatha, optionally followed by the small low
# (U+06ED) or high (U+06E2) meem — every spelling of the ending in quran-transcript's text.
# Group 1 is what the pausal rewrite keeps.
_TAA_MARBUTA_TANWEEN_FATHA = re.compile("(ة)ً[ۭۢ]?\\Z")
# The phonetizer's last word: the final whitespace-separated run (quran_phonetizer itself
# collapses ``\s+`` to one space and strips the ends).
_LAST_WORD = re.compile(r"(\S+)\s*\Z")


def pausal_taa_marbuta(word: str) -> str:
    """``word`` with a final ``ةً`` rewritten so quran-transcript 0.5.2 gives its pausal form.

    Replacing the tanween (and the small meem after it) with a plain fatha lets ``CleanEnd``
    drop the fatha and ``NormalizeTaa`` produce ``ه``. The fatha takes the tanween's index;
    only the small meem, if any, is dropped. Any other word is returned unchanged.
    """
    match = _TAA_MARBUTA_TANWEEN_FATHA.search(word)
    if match is None:
        return word
    return word[: match.end(1)] + _FATHA


@cache
def _hafs_moshaf() -> MoshafAttributes:
    from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes

    return MoshafAttributes(**HAFS_MOSHAF)


@cache
def _uthmani_alphabet() -> frozenset[str]:
    """Every character of quran-transcript's Uthmani alphabet (the 62 its text uses)."""
    from quran_transcript import alphabet

    return frozenset(
        "".join(v for v in vars(alphabet.uthmani).values() if isinstance(v, str))
    )


def _require_uthmani(text: str) -> None:
    unsupported = sorted(
        {c for c in text if c not in _uthmani_alphabet() and not c.isspace()}
    )
    if unsupported:
        codepoints = ", ".join(f"U+{ord(c):04X}" for c in unsupported)
        raise ValueError(
            f"{text!r} has characters outside quran-transcript's Uthmani alphabet "
            f"({codepoints}); phonetize its own Aya(...).uthmani text instead"
        )


def phonetize(text: str) -> QuranPhoneticScriptOutput:
    """``quran_phonetizer(text)`` with the Hafs moshaf, its final word in correct waqf form.

    Only the last word is pausal, so only it goes through :func:`pausal_taa_marbuta`; a
    ``ةً`` anywhere earlier keeps its wasl form. Words are split on any whitespace, as the
    phonetizer does, and trailing whitespace is not a word. ``mappings`` index into ``text``
    exactly as given: a character the rewrite dropped maps to a deleted, empty span where
    the word's phonemes end, which is what a fixed quran-transcript (0.6.0+) returns for it.
    Raises ``ValueError`` on text outside quran-transcript's Uthmani alphabet, and whatever
    ``quran_phonetizer`` raises (``KeyError`` / ``IndexError`` on the ayat it cannot handle).
    """
    from quran_transcript import quran_phonetizer
    from quran_transcript.phonetics.conv_base_operation import MappingPos

    _require_uthmani(text)
    last = _LAST_WORD.search(text)
    if last is None:  # empty or all whitespace: nothing to make pausal
        return quran_phonetizer(text, _hafs_moshaf())
    word_start, word_end = last.span(1)
    pausal = pausal_taa_marbuta(last.group(1))
    cut = word_start + len(pausal)  # the rewrite only ever drops a suffix of the word
    out = quran_phonetizer(text[:word_start] + pausal + text[word_end:], _hafs_moshaf())
    if cut == word_end:
        return out
    end = out.mappings[cut - 1].pos[1]
    dropped = [MappingPos(pos=(end, end), deleted=True) for _ in range(word_end - cut)]
    return dataclasses.replace(
        out, mappings=[*out.mappings[:cut], *dropped, *out.mappings[cut:]]
    )

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

quran-transcript is imported lazily, so importing this module costs nothing where it is
absent.
"""

from __future__ import annotations

import dataclasses
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

_TAA_MARBUTA = "ة"
_TANWEEN_FATHA = "ً"
_FATHA = "َ"


def pausal_taa_marbuta(word: str) -> str:
    """``word`` with a final ``ةً`` rewritten so quran-transcript 0.5.2 gives its pausal form.

    Replacing the tanween (and the small mark after it) with a plain fatha lets ``CleanEnd``
    drop the fatha and ``NormalizeTaa`` produce ``ه``. The ``ة`` and the mark after it keep
    their input indices; only trailing characters are dropped. Any other word is returned
    unchanged.
    """
    taa = word.rfind(_TAA_MARBUTA)
    if taa >= 0 and _TANWEEN_FATHA in word[taa + 1 :]:
        return word[: taa + 1] + _FATHA
    return word


@cache
def _hafs_moshaf() -> MoshafAttributes:
    from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes

    return MoshafAttributes(**HAFS_MOSHAF)


def phonetize(text: str) -> QuranPhoneticScriptOutput:
    """``quran_phonetizer(text)`` with the Hafs moshaf, its final word in correct waqf form.

    Only the last word is pausal, so only it goes through :func:`pausal_taa_marbuta`; a
    ``ةً`` anywhere earlier keeps its wasl form. ``mappings`` index into ``text`` exactly as
    given: each character the rewrite dropped maps to a deleted, empty span at the end of
    the phonemes, which is what a fixed quran-transcript (0.6.0+) returns for them.
    Raises whatever ``quran_phonetizer`` raises (``KeyError`` / ``IndexError`` on the
    ayat it cannot handle).
    """
    from quran_transcript import quran_phonetizer
    from quran_transcript.phonetics.conv_base_operation import MappingPos

    head, space, last = text.rpartition(" ")
    pausal = head + space + pausal_taa_marbuta(last)
    out = quran_phonetizer(pausal, _hafs_moshaf())
    end = len(out.phonemes)
    dropped = [MappingPos(pos=(end, end), deleted=True) for _ in text[len(pausal) :]]
    return dataclasses.replace(out, mappings=[*out.mappings, *dropped])

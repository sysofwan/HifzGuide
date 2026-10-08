"""What one decode did at one truth site: the per-site outcomes of the acceptance rules (§1, §9).

A truth site (:mod:`tadabur.truth_sites`) names a carrier letter in an item's realized
reference, the mark under test, what the mushaf prescribes and what the reciter was heard to
say. Given the decode of that item, :func:`item_outcomes` reads what the decode **committed**
at the carrier, and :class:`SiteOutcome` derives the three outcomes every rate is built from:

* **C (commit)**: the decode committed an explicit output at the carrier;
* **A (correct)**: what it committed is what the human heard;
* **F (flagged, decode level)**: it committed an explicit output that differs from the
  prescribed (mushaf) mark or letter. Muraja plays no part (that is
  :mod:`training.muraja_policy`).

F implies C: an empty slot, an unaligned carrier and a wrong carrier are never flagged.
These are the operational definitions §9 freezes, pinned by ``test_site_outcomes.py``.

Tashkeel sites (fatha, damma, kasra, sukun)
-------------------------------------------
Read off :func:`training.tashkeel_eval.carrier_readings`, the carrier-anchored alignment of the
raw strings that the haraka eval uses, extended to every explicit mark on the carrier.

* The carrier is **aligned** when the decode character aligned to the reference's carrier is
  that same consonant. An unaligned carrier (a gap, or outside the local alignment) and a
  **wrong carrier** (a different consonant) have C = 0: the decode has no slot there.
* The **marks** are the explicit marks (a haraka, or the sukun mark once #93 adds the class)
  the alignment places on the carrier, whether aligned to the reference's mark or inserted.
* No mark: an **empty slot**, C = 0. Today's head has no sukun class, so at a sukun site the
  empty slot is the only thing it can emit and its commit rate is 0 by construction.
* One distinct mark: C = 1, and that mark is committed. A **substituted** mark is a commit of
  the other mark: A = 0, and F = 1 because it differs from the prescribed one.
* More than one distinct mark (a haraka plus an inserted one): C = 1 with the committed value
  :data:`MULTIPLE`, so A = 0 and F = 1. A repeat of one mark is that mark.

Consonant (pair) sites
----------------------
The **consonant-commitment rule** is :func:`tadabur.contrast_attribution.aligned_consonants`:
the normalized Smith-Waterman alignment the P3.5 sites were located with. The decode
character aligned to the carrier's normalized group is committed when it is a consonant
(C = 1, whatever the letter); a gap, a non-consonant or a carrier outside the alignment is
C = 0. A = the committed letter is the heard one; F = it is not the prescribed one, so a
substitution by a letter outside the pair is flagged too.

Shaddah sites (provisional until #92)
-------------------------------------
The carrier is aligned when any position of its run of the same consonant (the geminate,
which the realized reference may write ``ووو`` before an assimilated letter) is aligned to
that consonant. The
committed state is read from :func:`tadabur.contrast_attribution.contrast_sites` with the
raw-string gemination check: at a ``held`` site, a dropped gemination on the carrier commits
``not_held`` and anything else commits ``held``; at a ``not_held`` site an added gemination
commits ``held`` and anything else ``not_held``. Today's encoding has no "unsure" state, so
every aligned carrier commits; #92 decides whether a single consonant should instead read as
an empty slot, and until then every shaddah cell is descriptive (``cannot_certify``).

Sides and cells
---------------
A site whose heard value is ``unclear`` or ``pending`` has no side and leaves every
denominator. Otherwise heard = prescribed is **correct recitation** and heard ≠ prescribed a
**real mistake**; the two are never pooled. Cells are indexed by the human-heard value.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tadabur.contrast_attribution import (
    ADDED,
    DROPPED,
    SHADDA_CONTRAST,
    aligned_consonants,
    contrast_sites,
)
from tadabur.normalization import _folded_core
from tadabur.truth_sites import (
    CONSONANTS,
    HARAKA_CHARS,
    HELD,
    NO_VERDICT,
    NOT_HELD,
    SHADDAH,
    SOFT_PAIRS,
    SUKUN,
    TASHKEEL_MARKS,
    TruthSite,
)
from training.tashkeel_eval import SUKUN_MARK, carrier_readings

#: The three families of truth sites; a pair site's family is its pair label.
TASHKEEL = "tashkeel"
#: ذ↔ظ is a target pair (acceptance rules §7) without being one of Muraja's soft pairs.
DHAL_ZAH = "ذ↔ظ"
TARGET_PAIRS: tuple[str, ...] = tuple(sorted(SOFT_PAIRS | {DHAL_ZAH}))

CORRECT_SIDE = "correct"
MISTAKE_SIDE = "mistake"

#: The committed value when the decode put more than one distinct mark on a carrier.
MULTIPLE = "multiple"

#: Explicit mark character -> its name in the truth-site vocabulary.
MARK_NAMES: dict[str, str] = {char: name for name, char in HARAKA_CHARS.items()} | {
    SUKUN_MARK: SUKUN
}


def family(mark: str) -> str:
    """``tashkeel``, ``shaddah``, or the pair label itself."""
    if mark in TASHKEEL_MARKS:
        return TASHKEEL
    return mark


def side(site: TruthSite) -> str | None:
    """Correct recitation, real mistake, or ``None`` for a site with no verdict."""
    if site.heard in NO_VERDICT:
        return None
    return CORRECT_SIDE if site.heard == site.prescribed else MISTAKE_SIDE


@dataclass(frozen=True)
class SiteOutcome:
    """What one decode committed at one site. See the module docstring for the rules."""

    carrier_aligned: bool
    #: The committed mark, letter or gemination state; ``None`` for an empty or absent slot.
    committed: str | None
    #: The explicit mark names on the carrier, in decode order (tashkeel sites only).
    marks: tuple[str, ...] = ()

    @property
    def commits(self) -> bool:
        return self.committed is not None

    def correct(self, site: TruthSite) -> bool:
        return self.committed is not None and self.committed == site.heard

    def flagged(self, site: TruthSite) -> bool:
        return self.committed is not None and self.committed != site.prescribed


def _tashkeel_outcome(site: TruthSite, readings) -> SiteOutcome:
    reading = readings.get(site.reference_index)
    if reading is None or reading.decoded != site.reference[site.reference_index]:
        return SiteOutcome(False, None)
    marks = tuple(MARK_NAMES[char] for char in reading.marks)
    distinct = list(dict.fromkeys(marks))
    committed = None if not distinct else distinct[0] if len(distinct) == 1 else MULTIPLE
    return SiteOutcome(True, committed, marks)


def _pair_outcome(site: TruthSite, consonants: dict[int, str | None]) -> SiteOutcome:
    letter = consonants.get(site.reference_index)
    if letter not in CONSONANTS:
        return SiteOutcome(False, None)
    return SiteOutcome(True, letter)


def _shaddah_outcome(
    site: TruthSite, consonants: dict[int, str | None], changes: dict[int, str]
) -> SiteOutcome:
    letter = site.reference[site.reference_index]
    # A geminate is a run of the same consonant (``لل``, or ``ووو`` before an assimilated
    # letter); a decode that kept one consonant may align it to any position of the run.
    run = [site.reference_index]
    while run[-1] + 1 < len(site.reference) and site.reference[run[-1] + 1] == letter:
        run.append(run[-1] + 1)
    if all(consonants.get(index) != _folded_core(letter) for index in run):
        return SiteOutcome(False, None)
    change = changes.get(site.reference_index)
    if site.prescribed == HELD:
        return SiteOutcome(True, NOT_HELD if change == DROPPED else HELD)
    return SiteOutcome(True, HELD if change == ADDED else NOT_HELD)


def item_outcomes(sites: Iterable[TruthSite], decode: str) -> dict[str, SiteOutcome]:
    """Each site's outcome under one decode of the item they all sit on.

    The sites must share one realized reference (one item); each alignment is computed once.
    """
    sites = list(sites)
    references = {site.reference for site in sites}
    if len(references) > 1:
        raise ValueError("item_outcomes takes the sites of one item (one reference)")
    if not sites:
        return {}
    (reference,) = references
    readings = consonants = changes = None
    outcomes: dict[str, SiteOutcome] = {}
    for site in sites:
        kind = family(site.mark)
        if kind == TASHKEEL:
            if readings is None:
                readings = carrier_readings(decode, reference)
            outcomes[site.site_id] = _tashkeel_outcome(site, readings)
            continue
        if consonants is None:
            consonants = aligned_consonants(decode, reference)
        if kind == SHADDAH:
            if changes is None:
                changes = {
                    s.reference_index: s.change
                    for s in contrast_sites(decode, reference, SHADDA_CONTRAST)
                }
            outcomes[site.site_id] = _shaddah_outcome(site, consonants, changes)
        else:
            outcomes[site.site_id] = _pair_outcome(site, consonants)
    return outcomes

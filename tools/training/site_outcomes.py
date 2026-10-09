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
The site's **run** is the reference's run of its consonant (the geminate, which the
realized reference may write ``ووو`` before an assimilated letter, or the single consonant
of a ``not_held`` site). The decode's state is read off the raw strings:
:meth:`tadabur.contrast_attribution.CarrierAlignment.decoded_run` counts the consonants of
that core the decode holds there, from the groups aligned to the run plus any unaligned
copies next to them (an insertion, or an edge the local alignment trimmed), each group
counted on its raw characters so a bare ``دد`` merged by normalization is two. No consonant
is an unaligned carrier (C = 0); one commits ``not_held``; two or more commit ``held``.
Today's encoding has no "unsure" state, so every aligned carrier commits; #92 decides
whether a single consonant should instead read as an empty slot, and until then every
shaddah cell is descriptive (``cannot_certify``).

A tashkeel site whose carrier belongs to a reference geminate also records whether the
decode **collapsed** that geminate to one consonant (:attr:`SiteOutcome.geminate_collapsed`):
Muraja discards the collapsed group's tashkeel in every mode (ADR-0011 §2,
:mod:`training.muraja_policy`). It does not change C, A or F.

Sides and cells
---------------
A site whose heard value is ``unclear`` or ``pending`` has no side and leaves every
denominator. Otherwise heard = prescribed is **correct recitation** and heard ≠ prescribed a
**real mistake**; the two are never pooled. Cells are indexed by the human-heard value.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tadabur.contrast_attribution import CarrierAlignment, align_on_carriers
from tadabur.truth_sites import (
    CONSONANTS,
    HARAKA_CHARS,
    HELD,
    NO_VERDICT,
    NOT_HELD,
    SHADDAH,
    SOFT_PAIRS,
    SUKUN,
    TARGET_PAIRS,
    TASHKEEL_MARKS,
    TruthSite,
)
from training.tashkeel_eval import SUKUN_MARK, carrier_readings

#: The three families of truth sites; a pair site's family is its pair label.
TASHKEEL = "tashkeel"
#: The one target pair that is not a Muraja soft pair (acceptance rules §7).
(DHAL_ZAH,) = TARGET_PAIRS - SOFT_PAIRS

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
    #: The carrier sits in a reference geminate the decode holds as one consonant.
    geminate_collapsed: bool = False

    @property
    def commits(self) -> bool:
        return self.committed is not None

    def correct(self, site: TruthSite) -> bool:
        return self.committed is not None and self.committed == site.heard

    def flagged(self, site: TruthSite) -> bool:
        return self.committed is not None and self.committed != site.prescribed


def consonant_run(reference: str, index: int) -> list[int]:
    """The indices of the run of ``reference[index]`` that contains ``index``."""
    letter = reference[index]
    start, end = index, index
    while start > 0 and reference[start - 1] == letter:
        start -= 1
    while end + 1 < len(reference) and reference[end + 1] == letter:
        end += 1
    return list(range(start, end + 1))


def _tashkeel_outcome(site: TruthSite, readings, alignment: CarrierAlignment) -> SiteOutcome:
    reading = readings.get(site.reference_index)
    if reading is None or reading.decoded != site.reference[site.reference_index]:
        return SiteOutcome(False, None)
    marks = tuple(MARK_NAMES[char] for char in reading.marks)
    distinct = list(dict.fromkeys(marks))
    committed = None if not distinct else distinct[0] if len(distinct) == 1 else MULTIPLE
    run = consonant_run(site.reference, site.reference_index)
    collapsed = len(run) > 1 and alignment.decoded_run(run) == 1
    return SiteOutcome(True, committed, marks, collapsed)


def _pair_outcome(site: TruthSite, alignment: CarrierAlignment) -> SiteOutcome:
    letter = alignment.consonants().get(site.reference_index)
    if letter not in CONSONANTS:
        return SiteOutcome(False, None)
    return SiteOutcome(True, letter)


def _shaddah_outcome(site: TruthSite, alignment: CarrierAlignment) -> SiteOutcome:
    held = alignment.decoded_run(consonant_run(site.reference, site.reference_index))
    if not held:
        return SiteOutcome(False, None)
    return SiteOutcome(True, HELD if held > 1 else NOT_HELD)


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
    alignment = align_on_carriers(decode, reference)
    readings = None
    outcomes: dict[str, SiteOutcome] = {}
    for site in sites:
        kind = family(site.mark)
        if kind == TASHKEEL:
            if readings is None:
                readings = carrier_readings(decode, reference)
            outcomes[site.site_id] = _tashkeel_outcome(site, readings, alignment)
        elif kind == SHADDAH:
            outcomes[site.site_id] = _shaddah_outcome(site, alignment)
        else:
            outcomes[site.site_id] = _pair_outcome(site, alignment)
    return outcomes

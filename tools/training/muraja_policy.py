"""Today's Muraja grading, frozen at site level, with each allowance's ON/OFF state table (§9).

The ship criterion (acceptance rules §3) compares the candidate system with **today's
system**: ``h448`` at b = 0, no bias, and the Muraja configuration :data:`TODAY` pins here.
Muraja grades **words** (``QuranFollowAlong+WordScoring.swift``); this module models only
what it does with the one mark or letter a truth site tests, which is all an offline scorer
can say (ADR-0008: a word-flag claim needs a versioned Muraja replay). :func:`grade` returns
one of

* :data:`CORRECT`: Muraja reads the site as recited as prescribed;
* :data:`WRONG`: Muraja grades the site as an error (a **false flag** on correct recitation,
  a **caught** mistake on a real one);
* :data:`NOT_GRADED`: an allowance looks away, or the slot does not exist in the decode.
  A not-graded mistake is a missed one, and not-graded sites reduce **coverage**.

Source: ``sysofwan/Muraja`` at :data:`MURAJA_REVISION` (v1.0.27): ``ScoringParameters`` in
``FollowAlongTypes.swift``, ``computeWordStatuses`` and ``expandToOriginalSpace`` in
``QuranFollowAlong+WordScoring.swift``, the tashkeel toggle in ``GradeFilter+iOS.swift``,
and the defaults in ``AppSettings+iOS.swift`` (mode ``.balanced``, tashkeel error detection
on). ``.balanced`` sets ``softPairsEnabled`` and ``shaddahSuppression`` and leaves
``suppressHarakaDrop`` off; the per-letter dropped-haraka exemption and the end-word
exemption hold in every mode.

State tables (one per allowance; ``test_muraja_policy.py`` pins every row)
------------------------------------------------------------------------
**Tashkeel site** (the carrier aligned; a site whose carrier the decode missed is
``NOT_GRADED`` here, because the slot does not exist and the consonant error belongs to the
consonant). Muraja compares the reference group's trailing haraka with the decode group's
trailing haraka, so a sukun site expects none:

====================================  =============  ==========================================
decode at the carrier                 prescribed     grade
====================================  =============  ==========================================
the prescribed haraka                 haraka X       CORRECT
another haraka (or several, last ≠ X) haraka X       WRONG, in every mode
nothing (dropped haraka)              haraka X       WRONG; NOT_GRADED when the carrier is one of
                                                     و ا ء ي (``dropped_haraka_letters``) or
                                                     ``suppress_haraka_drop`` is on
nothing                               sukun          CORRECT
a haraka                              sukun          WRONG
====================================  =============  ==========================================

**End word** (``end_word``, kept by ADR-0011 §6): a site at a pause is ``NOT_GRADED`` for
tashkeel and for a consonant swap. A site is at a pause when its carrier is the last
consonant of the item's realized reference (a segment or ayah end) or it sits in the
human-heard pause stratum ``waqf_boundary:waqf``. Words carrying a mushaf waqf sign inside an
item are not modelled.

**Tashkeel toggle** (``tashkeel_errors``): off grades every tashkeel site ``NOT_GRADED``.

**Empty slot = not graded** (``empty_slot_not_graded``, the candidate system's rule,
ADR-0011 §2): on, an empty tashkeel slot is ``NOT_GRADED`` and any committed mark other than
the prescribed one (sukun included) is ``WRONG``; it supersedes the dropped-haraka exemptions.

**Shaddah site** (provisional until #92; carrier missed → ``NOT_GRADED``):

====================  ===========  ===============================================
decode                prescribed   grade
====================  ===========  ===============================================
doubled               held         CORRECT
single                held         NOT_GRADED with ``shaddah_suppression``, else WRONG
single                not_held     CORRECT
doubled (added)       not_held     NOT_GRADED in every mode: grading is indexed by the
                                   reference, which has no column for an insertion
====================  ===========  ===============================================

**Pair site**: no consonant at the carrier is a deletion, ``WRONG`` in every mode; the
prescribed letter is ``CORRECT``; the other letter of a soft pair in ``soft_pairs`` is
``NOT_GRADED`` (the pair bypasses the phoneme gate); any other letter is ``WRONG``. ``ذ↔ظ``
is not a Muraja soft pair, so its swap is always ``WRONG``.

Allowance-affected populations (§4, §9)
---------------------------------------
:data:`ALLOWANCES` lists each allowance with how to switch it off and which sites it
**affects**, decided from the truth record alone (never from a decode):

* ``soft_pair <a↔b>`` (one per soft pair, never pooled): the sites of that pair;
* ``shaddah_suppression``: shaddah sites prescribed ``held``;
* ``dropped_haraka_exemption``: haraka-prescribed sites whose carrier is و ء or ي (``ا`` is
  never a carrier in the phoneme vocabulary, so it affects nothing);
* ``suppress_haraka_drop``: haraka-prescribed sites on every other carrier. It is already
  off in :data:`TODAY`, so switching it off changes nothing.

The end-word allowance is kept by ADR-0011 §6 and is not up for retirement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from tadabur.phoneme_sifat import soft_pair_contrast
from tadabur.truth_sites import (
    CONSONANTS,
    HARAKA_CHARS,
    HELD,
    SHADDAH,
    SOFT_PAIRS,
    WAQF_BOUNDARY,
    TruthSite,
)
from tadabur.waqf_truth_sites import WAQF
from training.site_outcomes import TASHKEEL, SiteOutcome, family

CORRECT = "correct"
WRONG = "wrong"
NOT_GRADED = "not_graded"
GRADES = (CORRECT, WRONG, NOT_GRADED)

MURAJA_REVISION = "99c326f3fb7c6c53ff525568976c7e2442ccde3c"
#: و ا ء ي: the letters that double as madd carriers, exempt from a dropped haraka.
DROPPED_HARAKA_LETTERS = frozenset("واءي")
#: The stratum of the boundaries a human heard as a pause.
PAUSE_STRATUM = f"{WAQF_BOUNDARY}:{WAQF}"


@dataclass(frozen=True)
class MurajaConfig:
    """Everything that decides a site-level grade. See the module docstring."""

    revision: str
    mode: str
    tashkeel_errors: bool
    soft_pairs: frozenset[str]
    shaddah_suppression: bool
    dropped_haraka_letters: frozenset[str]
    suppress_haraka_drop: bool
    end_word: bool
    empty_slot_not_graded: bool

    def as_dict(self) -> dict:
        return {
            "revision": self.revision,
            "mode": self.mode,
            "tashkeel_errors": self.tashkeel_errors,
            "soft_pairs": sorted(self.soft_pairs),
            "shaddah_suppression": self.shaddah_suppression,
            "dropped_haraka_letters": sorted(self.dropped_haraka_letters),
            "suppress_haraka_drop": self.suppress_haraka_drop,
            "end_word": self.end_word,
            "empty_slot_not_graded": self.empty_slot_not_graded,
        }


#: Today's system: Muraja v1.0.27 defaults, ``.balanced``, tashkeel error detection on.
TODAY = MurajaConfig(
    revision=MURAJA_REVISION,
    mode="balanced",
    tashkeel_errors=True,
    soft_pairs=frozenset(SOFT_PAIRS),
    shaddah_suppression=True,
    dropped_haraka_letters=DROPPED_HARAKA_LETTERS,
    suppress_haraka_drop=False,
    end_word=True,
    empty_slot_not_graded=False,
)


def at_pause(site: TruthSite) -> bool:
    """The carrier ends the item's recitation, or a human heard a pause there."""
    after = site.reference[site.reference_index + 1 :]
    return site.stratum == PAUSE_STRATUM or not any(char in CONSONANTS for char in after)


def _grade_tashkeel(site: TruthSite, outcome: SiteOutcome, config: MurajaConfig) -> str:
    if not config.tashkeel_errors or not outcome.carrier_aligned:
        return NOT_GRADED
    if config.end_word and at_pause(site):
        return NOT_GRADED
    if config.empty_slot_not_graded:
        if outcome.committed is None:
            return NOT_GRADED
        return CORRECT if outcome.committed == site.prescribed else WRONG
    trailing = outcome.marks[-1] if outcome.marks and outcome.marks[-1] in HARAKA_CHARS else None
    expected = site.prescribed if site.prescribed in HARAKA_CHARS else None
    if trailing == expected:
        return CORRECT
    if trailing is None:  # a dropped haraka
        carrier = site.reference[site.reference_index]
        exempt = config.suppress_haraka_drop or carrier in config.dropped_haraka_letters
        return NOT_GRADED if exempt else WRONG
    return WRONG


def _grade_shaddah(site: TruthSite, outcome: SiteOutcome, config: MurajaConfig) -> str:
    if not outcome.carrier_aligned:
        return NOT_GRADED
    if outcome.committed == site.prescribed:
        return CORRECT
    if site.prescribed == HELD:  # the decode left the geminate single
        return NOT_GRADED if config.shaddah_suppression else WRONG
    return NOT_GRADED  # an added gemination is an insertion, which grading cannot see


def _grade_pair(site: TruthSite, outcome: SiteOutcome, config: MurajaConfig) -> str:
    if outcome.committed is None:
        return WRONG
    if outcome.committed == site.prescribed:
        return CORRECT
    if config.end_word and at_pause(site):
        return NOT_GRADED
    if soft_pair_contrast(outcome.committed, site.prescribed) in config.soft_pairs:
        return NOT_GRADED
    return WRONG


def grade(site: TruthSite, outcome: SiteOutcome, config: MurajaConfig = TODAY) -> str:
    """How ``config`` grades this site given the decode's outcome there."""
    kind = family(site.mark)
    if kind == TASHKEEL:
        return _grade_tashkeel(site, outcome, config)
    if kind == SHADDAH:
        return _grade_shaddah(site, outcome, config)
    return _grade_pair(site, outcome, config)


@dataclass(frozen=True)
class Allowance:
    """One allowance: how to switch it off, and which sites it affects."""

    name: str
    switch_off: Callable[[MurajaConfig], MurajaConfig]
    affects: Callable[[TruthSite], bool]


def _haraka_on(letters_in: bool) -> Callable[[TruthSite], bool]:
    def affects(site: TruthSite) -> bool:
        carrier = site.reference[site.reference_index]
        return site.prescribed in HARAKA_CHARS and (carrier in DROPPED_HARAKA_LETTERS) == letters_in

    return affects


def _soft_pair(pair: str) -> Allowance:
    return Allowance(
        name=f"soft_pair {pair}",
        switch_off=lambda config: replace(config, soft_pairs=config.soft_pairs - {pair}),
        affects=lambda site: site.mark == pair,
    )


ALLOWANCES: tuple[Allowance, ...] = (
    *(_soft_pair(pair) for pair in sorted(SOFT_PAIRS)),
    Allowance(
        "shaddah_suppression",
        lambda config: replace(config, shaddah_suppression=False),
        lambda site: site.mark == SHADDAH and site.prescribed == HELD,
    ),
    Allowance(
        "dropped_haraka_exemption",
        lambda config: replace(config, dropped_haraka_letters=frozenset()),
        _haraka_on(True),
    ),
    Allowance(
        "suppress_haraka_drop",
        lambda config: replace(config, suppress_haraka_drop=False),
        _haraka_on(False),
    ),
)

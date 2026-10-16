"""Whose error is each consonant false flag: the model's, or the reference's, tajweed's or grading's (#155).

The owner widened the consonant goal to any consonant the model mishears, prioritised by app
false-flag frequency (#77, 2026-10-15, option B). ``word_flags.substitution_tally`` counts the
consonant false flags by substitution; several of the top ones (``ن→ل``, ``ل→ر``, ``م→ب``,
``ا→∅``) look like tajweed assimilation rather than mishearing. Training cannot fix a flag where
Hafs does not pronounce the expected letter, so before training is planned each flag is
classified, mechanically, by whose error it is.

**Records.** Every consonant error (another consonant, a target or soft pair, a collapsed
geminate: :func:`training.word_flags.consonant_cause`) behind a false flag, per population,
model and scoring: the word (``surah:ayah:word``, Muraja's phoneme word), the substitution, where
it sits in the word, how the mushaf spells the letter there (:func:`spelling`), and, for QUD and
Tadabur audio only, the printed word, its neighbours across both boundaries and the query
around it. The device and owner sessions are private (:data:`PRIVATE`): their records and
checklists keep word references, substitutions and rules only.

**Where the letter is.** Muraja's word scorer numbers a word's groups after skipping what it
never scores (``QuranFollowAlong+WordScoring.swift:131-226`` at the pin: the leading tanween
``ں`` run and a leading assimilated letter, the connection ``وو`` / ``ںں`` / ``يي`` at the end,
a trailing madd run beyond its first letter, and the madd marks ``ۦ ۥ ۾ ڇ``; a run of one bare
letter is one group, as ``normalizePhonemes`` makes it). :func:`scored_units` replays that. A
reported group can only sit later than its number (a geminate whose two halves the alignment
gave one group, or an alignment that starts inside the word), so the letter is the first unit at
or after the group that holds the expected letter (:func:`candidates`). Without a group (the
owner table keeps substitutions only) every unit holding the letter is a candidate, and a flag
whose candidates are classified differently is ``unclear``.

**The classification** (:func:`classify`), on Muraja's reference for the ayah (``quran.db``: the
phonetizer's output for the whole ayah, wasl inside, waqf at its end; its letters are the
phonetizer's, ``generate_phonemes.py``):

1. **reference** (responsible: the **phonetizer**) where the mushaf's own orthography says the
   letter the phoneme comes from is not pronounced as itself, and the phonetizer wrote it as
   itself (:func:`mushaf_reason`): a letter marked silent (``۟``), a hamzat al-wasl inside the
   ayah, an unmarked letter before a letter with shaddah (complete idgham: mithlayn,
   mutajanisayn, mutaqaribayn, lam shamsiyyah), an unmarked ``ن`` (ikhfa, iqlab or idgham), an
   unmarked ``م`` before ``ب`` (ikhfa shafawi), a tanween marked as not izhar (idgham or ikhfa) or for iqlab;
2. **reference** (responsible: **Muraja's word scorer**) where what was recited is known (QUD,
   Tadabur: the item's realized reference, the phonetizer over its words, its last word in
   waqf) and it does not hold the letter at that place: the recitation paused or started where
   Muraja's ayah-wide wasl form does not (``waqf form`` when the item ends on the word);
3. **unclear** where what was recited is not known (device sessions) and the letter is
   **wasl-dependent**: it drops when the reader pauses on the word, starts at it, joins it to
   the previous ayah or carries it into the next (:data:`FORMS`, each phonetized). Whether the
   reader did is not in the data;
4. otherwise Hafs pronounces the letter there, and what the engine heard decides (:func:`heard`).
   The evidence is the **query of the check that set the kept grade**: for QUD and Tadabur the
   checks whose own grade of the word is the kept one
   (:attr:`training.muraja_policy.SiteGrade.setting_checks`); a device replay keeps no per-check
   grade, so there every check whose query can hold the letter (:func:`covering_checks`). A
   query is the last :data:`CONFIRMED_TAIL` confirmed Characters, the pending hops and the
   check's overlap or preview; which hops were pending is not recorded, so each count up to
   :data:`MAX_PENDING_HOPS` gives a possible query (:func:`possible_queries`), each aligned on the
   realized reference (or Muraja's ayah) by the truth-site scorer's rule
   (:func:`training.site_outcomes.pair_outcome`, :func:`~training.site_outcomes.shaddah_outcome`); a
   possible query whose alignment does not reach the letter is set aside, but a setting check
   none of whose queries reaches it leaves the flag ``unclear`` (a later check that reads well
   never masks it). It is **grading** (responsible: **Muraja's word scorer**) where every
   reading query holds the expected letter there (a geminate held, for a collapsed-geminate
   flag), **model** where none does, and **unclear** where they disagree or no setting check is
   found. The owner's sessions have
   no decode in the repo; the owner's verdict on a word the phone showed stands in (a model error
   read from the decode is ``model``, one explained by a word boundary is ``grading``); a flag with
   neither is ``unclear``.

A flag whose letter cannot be found in the word is ``unclear``. Each record names its rule.

**What the phonetizer audit checks.** Step 1 only finds a letter the reference keeps as itself
where the mushaf marks it silent or assimilated; it does not check where an assimilated letter
goes or that a kept letter is right. That a flagged letter is pronounced rests on the realized
reference holding it and on how the mushaf spells it there: as itself with a haraka, sukun or
shaddah (:data:`ITSELF_MARKED`) is the mushaf marking it pronounced.

**Rates.** Per 1,000 graded word occurrences (a word of one item, log or session the app graded,
the occurrence also the numerators' unit), by substitution and by expected letter
(:func:`training.word_flags.substitution_tally`: an occurrence counts once per key).
QUD's intervals are clustered by mushaf and surah (each mushaf is one reciter); the teacher's
device sessions' by log (#151's rule); the owner's three sessions and the two Tadabur
populations carry counts and points only. Nothing here sets a threshold.

Populations: the QUD pilot (#144; both models, every word of every item, regraded from the
committed check sequences), #151's teacher-era device sessions (replayed at the pin; the words
#151's docs explain left out), the owner's ``h448`` sessions (#153; its committed per-word table,
the owner's mistakes left out), #90's word sample and Ship A's held-out verified sites (correct
side), each on ``h448``'s recorded checks.

Usage (from ``tools/``; torch-free; ``build`` needs the Muraja harness at the pin)::

  python -m training.consonant_flags build
  python -m training.consonant_flags doc
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cache, cached_property
from itertools import accumulate
from pathlib import Path

import numpy as np

from tadabur.contrast_attribution import CarrierAlignment, align_on_carriers
from tadabur.truth_sites import HELD
from training.acceptance_stats import Resample, ratio_interval
from training.decoding import clusters, committed_text
from training.muraja_policy import NOT_FLAGGED, TASHKEEL_ERROR, WORD_FLAGGED, MurajaText, SiteGrade, counterparts_in
from training.site_outcomes import consonant_run, shaddah_outcome
from training.tashkeel_eval import write_text_atomically
from training.word_flags import (
    DELETED,
    SHADDAH_CAUSE,
    SKIPPED_QUALITY,
    consonant_cause,
    substitution,
    substitution_tally,
)

REPORT_DIR = Path(__file__).parent.parent / "consonant_flags"
REPORT_PATH = REPORT_DIR / "report.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "consonant-false-flags.md"

MODEL, GRADING, REFERENCE, UNCLEAR = "model", "grading", "reference", "unclear"
CLASSES = (MODEL, GRADING, REFERENCE, UNCLEAR)
PHONETIZER = "phonetizer"
MURAJA_SCORER = "Muraja word scorer"
#: The scorings every population is read under.
SCORING_NAMES = ("shipped_default", "allowances_off")
#: The owner-check list: so many substitutions per class, so many words per substitution.
CHECK_SUBSTITUTIONS, CHECK_WORDS = 5, 10

#: What Muraja's scorer skips (``maddScalars``) and folds (``tajweedEquivalents``).
MADD_MARKS = frozenset("\u06e6\u06e5\u06fe\u0687")  # small yaa, small waw, ghunna meem, qalqala
FOLDED = {"۾": "م", "ں": "ن"}
GHUNNA_NUN = "ں"
MADD_LETTERS = "يوا"
#: The kind of letter an expected phoneme is.
CONSONANT, MADD, HAMZA = "consonant", "madd", "hamza"


def letter_kind(letter: str) -> str:
    """A madd (``ا`` and the small waw and yaa), the hamza, or a consonant."""
    return MADD if letter in "اۥۦ" else HAMZA if letter == "ء" else CONSONANT


# --- where the letter is -----------------------------------------------------------------------


def _fold(unit: str) -> str:
    return FOLDED.get(unit[0], unit[0])


def scored_units(word: str) -> list[tuple[int, str]]:
    """Each group Muraja's word scorer numbers in ``word``, as ``(character offset, group)``
    (module docstring)."""
    units = clusters(word)
    offsets = list(accumulate((len(u) for u in units), initial=0))
    start = 0
    while start < len(units) and units[start] == GHUNNA_NUN:
        start += 1
    run = start
    while run < len(units) and len(units[run]) == 1 and _fold(units[run]) == _fold(units[start]):
        run += 1
    if start < run < len(units) and _fold(units[run]) == _fold(units[start]) and len(units[run]) > 1:
        start = run  # the leading assimilated letter (قُل لَهُ → للَهُ)
    end = len(units)
    while end > start and units[end - 1] in ("و", GHUNNA_NUN):
        end -= 1
    if end == len(units):
        ya = end
        while ya > start and units[ya - 1] == "ي":
            ya -= 1
        end = ya if len(units) - ya >= 2 else end
    if end > start and units[end - 1][0] in MADD_LETTERS:
        last = end - 1
        while last > start and units[last - 1][0] == units[end - 1][0]:
            last -= 1
        end = last + 1
    out, i = [], start
    while i < end:
        if units[i][0] in MADD_MARKS:
            i += 1
            continue
        j = i + 1
        if len(units[i]) == 1:
            while j < end and len(units[j]) == 1 and _fold(units[j]) == _fold(units[i]):
                j += 1
        out.append((offsets[i], "".join(units[i:j])))
        i = j
    return out


def candidates(word: str, expected: str, group: int | None) -> list[int]:
    """The character offsets in ``word`` the error may sit at (module docstring): with a group,
    the first unit at or after it that holds ``expected``; without, every unit that does."""
    units = scored_units(word)
    holding = [k for k, (_, unit) in enumerate(units) if unit[0] == expected]
    if group is not None:
        holding = [k for k in holding if k >= group][:1]
    return [units[k][0] for k in holding]


def word_start(ayah: str, word: int) -> int:
    """Where the 1-based phoneme ``word`` starts in ``ayah``'s space-separated phonemes."""
    return sum(len(w) + 1 for w in ayah.split(" ")[: word - 1])


# --- what the mushaf says -------------------------------------------------------------------------

SHADDA, SUKUN, SILENT = "\u0651", "\u0652", "\u06df"  # shaddah, sukun, the small rounded zero
HARAKAT = frozenset("\u064e\u064f\u0650")
TANWEEN = frozenset("\u064b\u064c\u064d")
#: The marks after a tanween for not-izhar (idgham or ikhfa) and, after a tanween or a noon, for iqlab.
NOT_IZHAR_MARK, IQLAB_MARK = "\u06ed", "\u06e2"
HAMZAT_WASL = "\u0671"
#: Letters an unmarked spelling leaves as madd or carrier, never an assimilated consonant.
UNMARKED_MADD = frozenset("اوىي" + HAMZAT_WASL)


def _marks(text: str, base: int) -> str:
    end = base + 1
    while end < len(text) and unicodedata.combining(text[end]):
        end += 1
    return text[base + 1:end]


def _next_letter(text: str, base: int) -> int | None:
    i = base + 1 + len(_marks(text, base))
    while i < len(text) and (text[i] in " \u0640" or unicodedata.combining(text[i])):
        i += 1
    return i if i < len(text) else None


def mushaf_reason(uthmani: str, index: int) -> tuple[str, str] | None:
    """Why the mushaf does not pronounce the character at ``index`` of an ayah's Uthmani text as
    its own letter, and the phoneme the phonetizer would wrongly keep for it; ``None`` when the
    orthography says nothing against it. The Madani mushaf writes these rules into its marks."""
    char = uthmani[index]
    if char in TANWEEN:
        after = uthmani[index + 1:index + 2]
        if after == NOT_IZHAR_MARK:
            return "tanween marked as not izhar (idgham or ikhfa)", "ن"
        if after == IQLAB_MARK:
            return "tanween marked for iqlab", "ن"
        return None
    if unicodedata.combining(char) or char == " ":
        return None
    marks = _marks(uthmani, index)
    if SILENT in marks:
        return "a letter marked silent", char
    if char == HAMZAT_WASL:
        return ("hamzat al-wasl inside the ayah", "ء") if index > 0 else None
    if char in UNMARKED_MADD:
        return None
    if char == "ن" and marks == IQLAB_MARK:
        return "noon marked for iqlab", char
    if marks:
        return None  # a haraka, sukun, shaddah or tanween: pronounced as written
    after = _next_letter(uthmani, index)
    following = uthmani[after] if after is not None else None
    if following is not None and SHADDA in _marks(uthmani, after) and following != char:
        return f"unmarked {char} before a shaddah {following} (complete idgham)", char
    if char == "ن" and following != "ن":
        return "unmarked noon (ikhfa, iqlab or idgham)", char
    if char == "م" and following == "ب":
        return "unmarked meem before baa (ikhfa shafawi)", char
    return None


def phonetizer_slips(uthmani: str, phonemes: str, spans: Sequence[tuple[int, int] | None]) -> dict[int, str]:
    """The phoneme positions (spaces not counted) where the phonetizer wrote a character's own
    letter although the mushaf marks it as not pronounced so (:func:`mushaf_reason`): ``spans``
    gives each Uthmani character's phoneme span (``None`` where deleted)."""
    nonspace = [i for i, c in enumerate(phonemes) if c != " "]
    ordinal = {i: k for k, i in enumerate(nonspace)}
    out: dict[int, str] = {}
    for index, span in enumerate(spans):
        if span is None:
            continue
        found = mushaf_reason(uthmani, index)
        if found is None:
            continue
        reason, letter = found
        for q in range(*span):
            if q in ordinal and phonemes[q] == letter:
                out[ordinal[q]] = reason
    return out


# --- the classification -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Form:
    """One realization of an ayah's context around a word (:data:`FORMS`): its phonemes, and the
    span ``[start, end)`` of Muraja's ayah reference it realizes (the rest it adds or leaves out)."""

    phonemes: str
    start: int
    end: int

    def keeps(self, ayah: str, position: int) -> bool:
        """Whether this realization holds Muraja's character at ``position``."""
        return position - self.start in counterparts_in(self.phonemes, ayah[self.start:self.end])


#: How the mushaf spells the letter an expected phoneme comes from (:func:`spelling`).
ITSELF_MARKED, ITSELF_UNMARKED, ANOTHER_LETTER, A_MARK = (
    "as itself, marked", "as itself, unmarked", "as another letter", "from a mark")
#: The Uthmani letters a phonetic letter may be spelled with (a hamza on its seat, a taa
#: marbuta, an alif maqsura, the noon and meem of ikhfa).
SPELLED_AS = {"ء": "ءأإؤئ" + HAMZAT_WASL, "ت": "تة", "ه": "هة", "ا": "اى", "ي": "يى", "ں": "ن", "۾": "م"}


def spelling(uthmani: str, index: int, letter: str) -> str:
    """How the Uthmani character at ``index`` spells the phonetic ``letter`` it produced: as
    that letter with a haraka, sukun, shaddah or tanween on it (or a hamza's seat) — the mushaf
    marking it pronounced —, as that letter with no mark, as another letter (an assimilation's
    source), or from a mark (a tanween's noon, a dagger alif, a small waw or yaa)."""
    char = uthmani[index]
    if unicodedata.combining(char) and char != SHADDA:
        return A_MARK
    base = index
    while unicodedata.combining(uthmani[base]):
        base -= 1
    if uthmani[base] not in SPELLED_AS.get(letter, letter):
        return ANOTHER_LETTER
    marked = uthmani[base] in SPELLED_AS["ء"] or set(_marks(uthmani, base)) & {*HARAKAT, SUKUN, SHADDA, *TANWEEN}
    return ITSELF_MARKED if marked else ITSELF_UNMARKED


def spellings(uthmani: str, phonemes: str, spans: Sequence[tuple[int, int] | None]) -> dict[int, str]:
    """:func:`spelling` of each phoneme position (by non-space ordinal), from the first Uthmani
    character whose phonetizer span holds it."""
    nonspace = {i: k for k, i in enumerate(i for i, c in enumerate(phonemes) if c != " ")}
    out: dict[int, str] = {}
    for index, span in enumerate(spans):
        for q in range(*span) if span else ():
            if q in nonspace and nonspace[q] not in out:
                out[nonspace[q]] = spelling(uthmani, index, phonemes[q])
    return out


@dataclass(frozen=True)
class Ayah:
    """Muraja's reference for one ayah and what can be said about each position of it (by
    non-space ordinal): where the phonetizer kept a letter the mushaf marks as not pronounced so,
    how the mushaf spells each phoneme's letter, and the ayah's other realizations around a word
    (:data:`FORMS`), for a recital not known."""

    phonemes: str
    slips: Mapping[int, str] = field(default_factory=dict)
    #: Whether the slips were looked for: the phonetizer reproduces Muraja's reference here.
    checked: bool = True
    spelled: Mapping[int, str] = field(default_factory=dict)
    forms: Callable[[int], Mapping[str, Form]] = field(default=lambda word: {}, compare=False)

    def ordinal(self, position: int) -> int:
        return position - self.phonemes.count(" ", 0, position)


#: The realizations of a word's context the ayah-wide reference does not show.
WAQF_HERE, IBTIDA_HERE, FROM_PREVIOUS, INTO_NEXT = (
    "a pause on this word", "a start at this word", "joined to the previous ayah", "carried into the next ayah")
FORMS = (WAQF_HERE, IBTIDA_HERE, FROM_PREVIOUS, INTO_NEXT)


# --- what the engine heard ---------------------------------------------------------------------------

#: The confirmed Characters the engine puts before the pending hops (``recentHeard``,
#: ``FollowAlongEngine.swift:3656-3659``).
CONFIRMED_TAIL = 20
#: The most hops a query can hold pending: the engine discards the oldest past
#: ``maxPendingHops`` (6, ``FollowAlongEngine.swift:265, 869-871``), and the check's own hop
#: joins them before it aligns.
MAX_PENDING_HOPS = 7


def kept_overlap(checks: Sequence[dict], index: int) -> str:
    """The overlap or preview check ``index`` aligns: its own, or for the app's stop the one
    the engine keeps from the update before it (``handleFinalFlush`` appends the stop's
    phonemes as a pending hop and aligns them with ``currentOverlap``,
    ``FollowAlongEngine.swift:899-913``)."""
    if not checks[index].get("stop"):
        return checks[index]["overlap"]
    return checks[index - 1]["overlap"] if index else ""


def possible_queries(checks: Sequence[dict], index: int) -> list[str]:
    """Every query check ``index`` may have aligned (``runFollowAlongCheck``,
    ``FollowAlongEngine.swift:1565-1628``): :data:`CONFIRMED_TAIL` confirmed Characters, the
    pending hops and :func:`kept_overlap`. Which hops were still pending is not recorded, so
    each count up to the last :data:`MAX_PENDING_HOPS` non-empty hops is a possible query, from
    none for a check that brings no hop and from one for a check that brings one (a window's
    hop or the stop's phonemes join the pending hops before it aligns,
    ``FollowAlongEngine.swift:625-628, 907-913``); a silence flush confirms every pending hop (``FollowAlongEngine.swift:613-624``), so
    no hop before the latest flush is pending, and a flush check aligns none. Distinct queries
    only, shortest pending run first."""
    flushes = [i for i, c in enumerate(checks[: index + 1]) if c["flush"]]
    after_flush = index + 1 if checks[index]["flush"] else (flushes[-1] + 1 if flushes else 0)
    hops = [clusters(c["hop"]) for c in checks[: index + 1] if c["hop"]]
    pendable = sum(1 for c in checks[after_flush: index + 1] if c["hop"])
    overlap = kept_overlap(checks, index)
    # a check that brings a hop appends it before it aligns (handleUpdate, handleFinalFlush): it
    # is pending, so at least one hop is
    fewest = 1 if checks[index]["hop"] and not checks[index]["flush"] else 0
    out: dict[str, None] = {}
    for k in range(fewest, min(MAX_PENDING_HOPS, pendable) + 1):
        confirmed = [c for hop in hops[: len(hops) - k] for c in hop][-CONFIRMED_TAIL:]
        pending = [c for hop in hops[len(hops) - k:] for c in hop]
        out["".join(confirmed + pending) + overlap] = None
    return list(out)


def _carrier(alignment: CarrierAlignment, index: int) -> int | None:
    """The raw-reference carrier of the normalized group that holds raw index ``index``."""
    starts = [*alignment.cluster_starts, len(alignment.raw_reference)]
    for lo, hi in alignment.reference.offset_map:
        if starts[lo] <= index < starts[hi]:
            return starts[lo]
    return None


def decode_holds(decode: str, reference: str, index: int, expected: str, geminate: bool) -> bool | None:
    """Whether ``decode``, aligned on ``reference`` by the truth-site scorer's rule, holds
    ``expected`` at ``index`` (the geminate there held, for a collapsed geminate); ``None`` where
    the alignment does not reach it."""
    alignment = align_on_carriers(decode, reference)
    if geminate:
        held = shaddah_outcome(alignment, consonant_run(reference, index)[0]).committed
        return None if held is None else held == HELD
    carrier = _carrier(alignment, index)
    consonants = alignment.consonants()
    if carrier is None or carrier not in consonants:
        return None
    return consonants[carrier] == FOLDED.get(expected, expected)


def check_evidence(queries: Sequence[str], reference: str, index: int, expected: str,
                   geminate: bool) -> frozenset[bool]:
    """What a check's possible queries say at the letter (:func:`decode_holds`): a query whose
    alignment does not reach the letter is taken not to be the one that graded it, and is set
    aside."""
    return frozenset(held for q in queries
                     if (held := decode_holds(q, reference, index, expected, geminate)) is not None)


def query_cluster(decode: str, reference: str, index: int) -> int | None:
    """The Character (cluster) of ``decode`` its alignment on ``reference`` pairs with the group
    holding raw index ``index`` (the nearest decode Character before it, at a gap)."""
    alignment = align_on_carriers(decode, reference)
    carrier = _carrier(alignment, index)
    starts = [*alignment.cluster_starts, len(reference)]
    last = None
    for pos, query in zip(alignment.positions, alignment.query_positions):
        last = query if query is not None else last
        if pos is not None and starts[alignment.reference.offset_map[pos][0]] == carrier:
            return None if last is None else alignment.decode.offset_map[last][0]
    return None


def covering_checks(checks: Sequence[dict], reference: str, index: int) -> tuple[int, ...]:
    """The checks of a session whose query can hold the letter at ``index`` of ``reference``:
    the letter's Character in the committed transcript (the hops, in order) lies in the
    check's longest possible query (:func:`possible_queries`) — its pending hops and confirmed
    tail, or the overlap that follows them."""
    at = query_cluster(committed_text(checks), reference, index)
    if at is None:
        return ()
    lengths = [len(clusters(c["hop"])) for c in checks]
    ends = list(accumulate(lengths))
    out = []
    for i, (check, end) in enumerate(zip(checks, ends)):
        pending = [n for n in lengths[: i + 1] if n][-MAX_PENDING_HOPS:]
        overlap = len(clusters(kept_overlap(checks, i)))
        if end - sum(pending) - CONFIRMED_TAIL <= at < end + overlap:
            out.append(i)
    return tuple(out)


@dataclass(frozen=True)
class Recital:
    """What is known of how a word was recited: the item's realized reference and whether the
    item ends on the word; the checks the engine graded it on, with the ones that can have set
    its kept grade (given the letter's position in Muraja's ayah: the checks whose own grade is
    the kept one, or for a device session every check whose query can hold the letter), read
    against ``realized`` (or Muraja's ayah when none is known); and the owner's verdict standing
    in for a decode the repo does not hold."""

    realized: str | None = None
    ends_here: bool = False
    checks: tuple[dict, ...] | None = None
    setting: Callable[[int], Sequence[int]] = field(default=lambda position: (), compare=False)
    verdict: str | None = None


@dataclass(frozen=True)
class Reading:
    """A flag's class, the rule that gave it, who is responsible and on what evidence."""

    klass: str
    rule: str
    responsible: str | None = None
    evidence: str | None = None


def heard(recital: Recital, reference: str, index: int, position: int, expected: str, geminate: bool) -> Reading:
    """Step 4: the class of a pronounced letter from what the engine heard (module docstring)."""
    if recital.checks is None:
        if recital.verdict is None:
            return Reading(UNCLEAR, "pronounced; no decode to read and no owner verdict")
        return Reading(recital.verdict, f"pronounced; the owner's verdict ({recital.verdict})",
                       MURAJA_SCORER if recital.verdict == GRADING else None, "owner")
    setting = recital.setting(position)
    if not setting:
        return Reading(UNCLEAR, "pronounced; no check that can have set the kept grade is identified")
    readings = [check_evidence(possible_queries(recital.checks, i), reference, index, expected, geminate)
                for i in setting]
    if not all(readings):
        return Reading(UNCLEAR, "pronounced; a check that can have set the kept grade has no query that "
                                "reaches the letter")
    said = frozenset().union(*readings)
    if said == {True}:
        return Reading(GRADING, "pronounced; the query that set the kept grade holds it", MURAJA_SCORER, "query")
    if said == {False}:
        return Reading(MODEL, "pronounced; the query that set the kept grade has another letter or none there",
                       None, "query")
    return Reading(UNCLEAR, "pronounced; the queries that can have set the kept grade disagree")


def read_position(ayah: Ayah, position: int, word: int, expected: str, geminate: bool,
                  recital: Recital) -> Reading:
    """The class of a flag at ``position`` of Muraja's ayah (module docstring, steps 1-4)."""
    slip = ayah.slips.get(ayah.ordinal(position))
    if slip is not None:
        return Reading(REFERENCE, f"orthography: {slip}", PHONETIZER)
    reference, index = ayah.phonemes, position
    if recital.realized is not None:
        index = counterparts_in(recital.realized, ayah.phonemes).get(position)
        if index is None:
            return Reading(REFERENCE, "waqf form" if recital.ends_here else "the realized form differs",
                           MURAJA_SCORER)
        reference = recital.realized
    else:
        dropped = [name for name, form in ayah.forms(word).items() if not form.keeps(ayah.phonemes, position)]
        if dropped:
            return Reading(UNCLEAR, f"wasl-dependent: dropped on {', '.join(dropped)}")
    return heard(recital, reference, index, position, expected, geminate)


def classify(ayah: Ayah, word: int, error: dict, recital: Recital, cause: str) -> tuple[Reading, int | None]:
    """A flag's reading and where in ``word`` its letter sits (``None`` when it cannot be
    placed, or when its candidates disagree)."""
    phonemes = ayah.phonemes.split(" ")[word - 1]
    expected = error["kind"]["expected"]
    group = error["group_index"]
    found = candidates(phonemes, expected, None if group == UNKNOWN_GROUP else group)
    if not found:
        return Reading(UNCLEAR, "the letter is not in the word"), None
    start = word_start(ayah.phonemes, word)
    readings = {offset: read_position(ayah, start + offset, word, expected, cause == SHADDAH_CAUSE, recital)
                for offset in found}
    if len({(r.klass, r.responsible) for r in readings.values()}) > 1:
        return Reading(UNCLEAR, "its candidate letters are classified differently"), None
    offset = found[0]
    return readings[offset], offset


# --- records -----------------------------------------------------------------------------------------

#: The populations whose recordings are private: their records keep word references only (no
#: Arabic beyond a substitution's or a letter's label).
QUD, DEVICE_LOGS, OWNER, WORD_SAMPLE, HELD_OUT_SITES = (
    "QUD pilot", "#151 device logs (teacher)", "#153 owner sessions (h448)", "#90 word sample (h448/checks)",
    "Ship A held-out verified sites (h448/checks)")
POPULATIONS = (QUD, DEVICE_LOGS, OWNER, WORD_SAMPLE, HELD_OUT_SITES)
PRIVATE = frozenset({DEVICE_LOGS, OWNER})


@dataclass(frozen=True)
class FlagRecord:
    """One consonant error behind one false flag, classified."""

    population: str
    model: str
    scoring: str
    word: str
    #: One recital of the word: an item's, a session's, a log's.
    occurrence: str
    cause: str
    substitution: str
    kind: str
    group: int | None
    unit: int | None
    #: How the mushaf spells the expected letter there (:func:`spelling`; ``None`` unplaced).
    spelled: str | None
    klass: str
    rule: str
    responsible: str | None
    evidence: str | None
    #: The word's interval cluster: a QUD mushaf and surah, a log, a session, an item.
    cluster: str
    #: Public populations only: the printed word and its context (:func:`context`).
    printed: str | None = None
    context: str | None = None
    #: QUD and Tadabur only: the query that set the kept grade, aligned to the word.
    decode: str | None = None


def context(ayah: str, word: int, offset: int | None) -> str:
    """The word with its neighbours across both boundaries, the flagged letter bracketed:
    ``<previous word's last letter>|<word>|<next word's first letter>`` (``#`` at the ayah's
    edge)."""
    words = ayah.split(" ")
    phonemes = words[word - 1]
    if offset is not None:
        width = len(next(u for o, u in scored_units(phonemes) if o == offset))
        phonemes = f"{phonemes[:offset]}[{phonemes[offset:offset + width]}]{phonemes[offset + width:]}"
    before = clusters(words[word - 2])[-1] if word > 1 else "#"
    after = clusters(words[word])[0] if word < len(words) else "#"
    return f"{before}|{phonemes}|{after}"


def decode_excerpt(decode: str, realized: str, word_span: tuple[int, int]) -> str:
    """The decode's characters aligned to the word's span of the realized reference."""
    alignment = align_on_carriers(decode, realized)
    starts = [*alignment.cluster_starts, len(realized)]
    lo, hi = word_span
    queries = [q for pos, q in zip(alignment.positions, alignment.query_positions)
               if q is not None and pos is not None
               and lo <= starts[alignment.reference.offset_map[pos][0]] < hi]
    if not queries:
        return ""
    chars = clusters(decode)
    first, last = alignment.decode.offset_map[min(queries)][0], alignment.decode.offset_map[max(queries)][1]
    return "".join(chars[first:last])


@dataclass(frozen=True)
class FlaggedWord:
    """A false flag to classify: where it is, its kept grade's errors, how it was recited."""

    population: str
    model: str
    scoring: str
    surah: int
    ayah: int
    word: int
    errors: tuple[dict, ...]
    recital: Recital
    cluster: str
    occurrence: str
    #: The word's span in the realized reference, for the decode excerpt.
    span: tuple[int, int] | None = None


def records(flagged: FlaggedWord, ayah: Ayah, printed: str) -> list[FlagRecord]:
    """Every consonant error of a flagged word, classified; a private population's without its
    Arabic."""
    phonemes = ayah.phonemes.split(" ")[flagged.word - 1]
    public = flagged.population not in PRIVATE
    recital = flagged.recital
    out = []
    for error in flagged.errors:
        cause = consonant_cause(error, phonemes)
        if cause is None:
            continue
        reading, offset = classify(ayah, flagged.word, error, recital, cause)
        units = [o for o, _ in scored_units(phonemes)]
        position = None if offset is None else word_start(ayah.phonemes, flagged.word) + offset
        excerpt = None
        if public and position is not None and recital.checks and recital.realized is not None and flagged.span:
            setting = recital.setting(position)
            if setting:
                query = possible_queries(recital.checks, setting[0])[-1]
                excerpt = decode_excerpt(query, recital.realized, flagged.span)
        out.append(FlagRecord(
            population=flagged.population, model=flagged.model, scoring=flagged.scoring,
            word=f"{flagged.surah}:{flagged.ayah}:{flagged.word}", occurrence=flagged.occurrence, cause=cause,
            substitution=substitution(error), kind=letter_kind(error["kind"]["expected"]),
            group=None if error["group_index"] == UNKNOWN_GROUP else error["group_index"],
            unit=None if offset is None else units.index(offset),
            spelled=None if position is None else ayah.spelled.get(ayah.ordinal(position)),
            klass=reading.klass, rule=reading.rule, responsible=reading.responsible, evidence=reading.evidence,
            cluster=flagged.cluster,
            printed=printed if public else None,
            context=context(ayah.phonemes, flagged.word, offset) if public else None,
            decode=excerpt))
    return out


#: The group of an error whose group the table did not keep: outside every word, so
#: ``word_flags.deleted_run`` reads the occurrence from the word's runs alone.
UNKNOWN_GROUP = -1


def error_without_group(label: str) -> dict:
    """A consonant error back from its substitution (``ل→ء``, ``ه→∅``), its group unknown."""
    expected, heard_ = label.split("→")
    kind = ({"type": "deletion", "expected": expected} if heard_ == DELETED
            else {"type": "mismatch", "expected": expected, "heard": heard_})
    return {"kind": kind, "group_index": UNKNOWN_GROUP}


# --- rates -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Population:
    """One population under one model and scoring: its graded words per cluster, whether the
    clusters carry an interval (QUD: mushaf and surah; device logs: log), and its classified
    flags."""

    name: str
    model: str
    scoring: str
    graded: Mapping[str, int]
    clustered: bool
    records: tuple[FlagRecord, ...]

    @property
    def words(self) -> int:
        return sum(self.graded.values())

    @cached_property
    def resample(self) -> Resample:
        """One bootstrap draw over the clusters, one observation per graded word, shared by
        every rate of the population."""
        return Resample.by_cluster(self.observations)

    @cached_property
    def observations(self) -> list[int]:
        """Each graded word's cluster, as an index into the sorted clusters."""
        return [k for k, c in enumerate(sorted(self.graded)) for _ in range(self.graded[c])]


def per_1000(hits: int, words: int) -> float | None:
    return round(1000 * hits / words, 2) if words else None


def _word_key(record: FlagRecord) -> tuple[str, str]:
    """A flagged word occurrence: its cluster and its recital, the unit graded words count in."""
    return record.cluster, record.occurrence


def interval(population: Population, hits: Mapping[str, int]) -> dict:
    """``hits`` (words per cluster) per 1,000 graded words, with §1's interval over the
    population's clusters when it carries one."""
    total = sum(hits.values())
    out = {"per_1000": per_1000(total, population.words)}
    if not population.clustered:
        return out
    num = np.array([1.0 if k < hits.get(c, 0) else 0.0
                    for c in sorted(population.graded) for k in range(population.graded[c])])
    bounds = ratio_interval(population.observations, num, np.ones_like(num), np.ones_like(num),
                            population.resample).as_dict()
    return out | {side: None if bounds[side] is None else round(1000 * bounds[side], 2) for side in ("lower", "upper")}


def rate_rows(population: Population, klass: str, by_expected: bool) -> list[dict]:
    """The words carrying each substitution (or expected letter) of ``klass``, most first, each
    per 1,000 graded words (:func:`interval`)."""
    by_word: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for r in population.records:
        if r.klass == klass:
            by_word[_word_key(r)].append((r.cause, r.substitution))
    key = "expected" if by_expected else "substitution"

    def label(substitution_: str) -> str:
        return substitution_.split("→")[0] if by_expected else substitution_

    return [row | interval(population, Counter(
        cluster for (cluster, _), errors in by_word.items()
        if any(cause == row["cause"] and label(s) == row[key] for cause, s in errors)))
        for row in substitution_tally(by_word.values(), by_expected)]


def class_counts(population: Population) -> dict:
    """Per class: the words carrying a flag of it and the flags, with the words per 1,000."""
    out = {}
    for klass in CLASSES:
        mine = [r for r in population.records if r.klass == klass]
        words = len({_word_key(r) for r in mine})
        out[klass] = {"flags": len(mine), "words": words, "per_1000": per_1000(words, population.words)}
    return out


def responsible_table(population: Population) -> list[dict]:
    """The non-model flags (reference and grading) by substitution, rule and responsible
    component, most words first."""
    groups: dict[tuple, set] = defaultdict(set)
    for r in population.records:
        if r.klass in (REFERENCE, GRADING):
            groups[(r.klass, r.responsible, r.substitution, r.rule)].add(_word_key(r))
    return [{"class": k, "responsible": who, "substitution": s, "rule": rule, "words": len(w),
             "per_1000": per_1000(len(w), population.words)}
            for (k, who, s, rule), w in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]


def owner_checks(flags: Sequence[FlagRecord]) -> dict[str, list[dict]]:
    """Per class, the top :data:`CHECK_SUBSTITUTIONS` substitutions (each word reference once
    per substitution, every substitution it carries counted) with up to :data:`CHECK_WORDS` of
    their words, in Quran order, for the owner to confirm. A private population's words carry
    no Arabic (:data:`PRIVATE`)."""
    out = {}
    for klass in CLASSES:
        mine = [r for r in flags if r.klass == klass]
        by_word: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for r in mine:
            by_word[r.word].add((r.cause, r.substitution))
        top = dict.fromkeys(row["substitution"] for row in substitution_tally(by_word.values()))
        lists = []
        for sub in list(top)[:CHECK_SUBSTITUTIONS]:
            words = sorted({r.word: r for r in mine if r.substitution == sub}.values(),
                           key=lambda r: tuple(int(x) for x in r.word.split(":")))
            lists.append({"substitution": sub, "words": [
                {"word": r.word, "rule": r.rule,
                 **({"printed": r.printed, "context": r.context} if r.population not in PRIVATE else {})}
                for r in words[:CHECK_WORDS]]})
        out[klass] = lists
    return out


def population_view(population: Population) -> dict:
    return {
        "population": population.name, "model": population.model, "scoring": population.scoring,
        "graded_words": population.words, "clusters": len(population.graded),
        "interval": population.clustered,
        "classes": class_counts(population),
        "model_by_substitution": rate_rows(population, MODEL, by_expected=False),
        "model_by_expected": rate_rows(population, MODEL, by_expected=True),
        "not_model": responsible_table(population),
        "unclear": sorted({(r.word, r.substitution, r.rule) for r in population.records if r.klass == UNCLEAR}),
    }


# --- running it --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="classify every population's consonant false flags (needs the harness)")
    commands.add_parser("doc", help=f"render {DOC_PATH.name} from the report")
    args = parser.parse_args()
    if args.command == "build":
        from training.consonant_flag_populations import build

        report = build()
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        write_text_atomically(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {REPORT_PATH}")
    from training.consonant_flags_doc import render

    write_text_atomically(DOC_PATH, render(json.loads(REPORT_PATH.read_text(encoding="utf-8"))))
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()

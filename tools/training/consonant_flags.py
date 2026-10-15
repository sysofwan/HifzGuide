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
it sits in the word, the word's neighbours across both boundaries, and, for QUD and Tadabur
audio only, the decode around it. Device and owner data keep word references only.

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
4. otherwise Hafs pronounces the letter there. It is **grading** (responsible: **Muraja's word
   scorer**) where the decode holds the expected letter at that place (a geminate held, for a
   collapsed-geminate flag) and the engine still flagged it, and **model** where the decode has
   something else or nothing there. The decode is the transcript the app commits (the recorded
   check sequence's hops, or a device session's logged hops), aligned on the realized reference
   (or, for a device session, on Muraja's ayah) by the truth-site scorer's rule
   (:func:`training.site_outcomes.pair_outcome`, :func:`~training.site_outcomes.shaddah_outcome`).
   The owner's sessions have no decode in the repo; the owner's verdict on a word the phone
   showed stands in (a model error read from the decode is ``model``, one explained by a word
   boundary is ``grading``). A flag with neither is ``model`` on the kept grade alone, and says so
   (``evidence``).

A flag whose letter cannot be found in the word is ``unclear``. Each record names its rule.

**Rates.** Per 1,000 graded words (correctly recited words the app graded), by substitution and
by expected letter (:func:`training.word_flags.substitution_tally`: a word counts once per key).
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
from training.decoding import clusters
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

SHADDA, SILENT = "\u0651", "\u06df"  # shaddah, the small rounded zero
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


@dataclass(frozen=True)
class Ayah:
    """Muraja's reference for one ayah and what can be said about each position of it: the
    phonetizer slips the mushaf contradicts (by non-space ordinal), and the ayah's other
    realizations around a word (:data:`FORMS`), each as phonemes, for a recital not known."""

    phonemes: str
    slips: Mapping[int, str] = field(default_factory=dict)
    #: Whether the slips were looked for: the phonetizer reproduces Muraja's reference here.
    checked: bool = True
    forms: Callable[[int], Mapping[str, Form]] = field(default=lambda word: {}, compare=False)

    def ordinal(self, position: int) -> int:
        return position - self.phonemes.count(" ", 0, position)


#: The realizations of a word's context the ayah-wide reference does not show.
WAQF_HERE, IBTIDA_HERE, FROM_PREVIOUS, INTO_NEXT = (
    "a pause on this word", "a start at this word", "joined to the previous ayah", "carried into the next ayah")
FORMS = (WAQF_HERE, IBTIDA_HERE, FROM_PREVIOUS, INTO_NEXT)


@dataclass(frozen=True)
class Recital:
    """What is known of how a word was recited: the item's realized reference, whether the item
    ends on the word, the committed decode (read against ``realized``, or against Muraja's ayah
    when no realized reference is known), and the owner's verdict standing in for a decode."""

    realized: str | None = None
    ends_here: bool = False
    decode: str | None = None
    verdict: str | None = None


@dataclass(frozen=True)
class Reading:
    """A flag's class, the rule that gave it, who is responsible and on what evidence."""

    klass: str
    rule: str
    responsible: str | None = None
    evidence: str | None = None


def _carrier(alignment: CarrierAlignment, index: int) -> int | None:
    """The raw-reference carrier of the normalized group that holds raw index ``index``."""
    starts = [*alignment.cluster_starts, len(alignment.raw_reference)]
    for lo, hi in alignment.reference.offset_map:
        if starts[lo] <= index < starts[hi]:
            return starts[lo]
    return None


def decode_holds(decode: str, reference: str, index: int, expected: str, geminate: bool) -> bool | None:
    """Whether ``decode``, aligned on ``reference``, holds ``expected`` at ``index`` (the geminate
    there held, for a collapsed geminate); ``None`` outside the alignment."""
    alignment = align_on_carriers(decode, reference)
    if geminate:
        held = shaddah_outcome(alignment, consonant_run(reference, index)[0]).committed
        return None if held is None else held == HELD
    carrier = _carrier(alignment, index)
    consonants = alignment.consonants()
    if carrier is None or carrier not in consonants:
        return None
    return consonants[carrier] == FOLDED.get(expected, expected)


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
    if recital.decode is not None:
        holds = decode_holds(recital.decode, reference, index, expected, geminate)
        if holds:
            return Reading(GRADING, "pronounced; the decode holds it", MURAJA_SCORER, "decode")
        return Reading(MODEL, "pronounced; the decode has another" if holds is False
                       else "pronounced; the decode has nothing aligned there", None, "decode")
    if recital.verdict is not None:
        return Reading(recital.verdict, f"pronounced; the owner's verdict ({recital.verdict})",
                       MURAJA_SCORER if recital.verdict == GRADING else None, "owner")
    return Reading(MODEL, "pronounced; kept grade only", None, "kept grade")


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


@dataclass(frozen=True)
class FlagRecord:
    """One consonant error behind one false flag, classified."""

    population: str
    model: str
    scoring: str
    word: str
    printed: str
    cause: str
    substitution: str
    kind: str
    group: int | None
    unit: int | None
    context: str
    klass: str
    rule: str
    responsible: str | None
    evidence: str | None
    #: The word's carrier group: a QUD mushaf and surah, a log, a session, a reciter.
    cluster: str
    #: QUD and Tadabur only: the committed decode aligned to the word.
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
    #: The word's span in the realized reference, for the decode excerpt.
    span: tuple[int, int] | None = None


def records(flagged: FlaggedWord, ayah: Ayah, printed: str) -> list[FlagRecord]:
    """Every consonant error of a flagged word, classified."""
    phonemes = ayah.phonemes.split(" ")[flagged.word - 1]
    out = []
    for error in flagged.errors:
        cause = consonant_cause(error, phonemes)
        if cause is None:
            continue
        reading, offset = classify(ayah, flagged.word, error, flagged.recital, cause)
        units = [o for o, _ in scored_units(phonemes)]
        excerpt = None
        if flagged.recital.decode is not None and flagged.recital.realized is not None and flagged.span:
            excerpt = decode_excerpt(flagged.recital.decode, flagged.recital.realized, flagged.span)
        out.append(FlagRecord(
            population=flagged.population, model=flagged.model, scoring=flagged.scoring,
            word=f"{flagged.surah}:{flagged.ayah}:{flagged.word}", printed=printed, cause=cause,
            substitution=substitution(error), kind=letter_kind(error["kind"]["expected"]),
            group=None if error["group_index"] == UNKNOWN_GROUP else error["group_index"], unit=units.index(offset) if offset is not None else None,
            context=context(ayah.phonemes, flagged.word, offset), klass=reading.klass, rule=reading.rule,
            responsible=reading.responsible, evidence=reading.evidence, cluster=flagged.cluster,
            decode=excerpt))
    return out


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
    return record.cluster, record.word


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
    """Per class, the top :data:`CHECK_SUBSTITUTIONS` substitutions with up to
    :data:`CHECK_WORDS` of their words (in Quran order) for the owner to confirm."""
    out = {}
    for klass in CLASSES:
        mine = [r for r in flags if r.klass == klass]
        top = [row["substitution"] for row in substitution_tally(
            [[(r.cause, r.substitution)] for r in {_word_key(r): r for r in mine}.values()])]
        lists = []
        for sub in list(dict.fromkeys(top))[:CHECK_SUBSTITUTIONS]:
            words = sorted({r.word: r for r in mine if r.substitution == sub}.values(),
                           key=lambda r: tuple(int(x) for x in r.word.split(":")))
            lists.append({"substitution": sub, "words": [
                {"word": r.word, "printed": r.printed, "context": r.context, "rule": r.rule}
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


# --- the populations -----------------------------------------------------------------------------------

QUD, DEVICE_LOGS, OWNER, WORD_SAMPLE, HELD_OUT_SITES = (
    "QUD pilot", "#151 device logs (teacher)", "#153 owner sessions (h448)", "#90 word sample (h448/checks)",
    "Ship A held-out verified sites (h448/checks)")
POPULATIONS = (QUD, DEVICE_LOGS, OWNER, WORD_SAMPLE, HELD_OUT_SITES)


class Ayat:
    """Each ayah's :class:`Ayah`: Muraja's reference, the phonetizer's slips against the
    mushaf, and the forms around a word, from quran-transcript (computed once per ayah)."""

    def __init__(self, text: MurajaText):
        self.text = text

    @cache
    def uthmani_words(self, surah: int, ayah: int) -> tuple[str, ...]:
        from tadabur.waqf_segments import _uthmani_words

        return tuple(_uthmani_words(f"{surah}:{ayah}"))

    def _phonetized(self, words: Sequence[str]) -> str | None:
        from hafs_phonetizer import phonetize

        try:
            return phonetize(" ".join(words)).phonemes
        except (KeyError, IndexError):
            return None  # quran-transcript 0.5.2's known gap

    @cache
    def get(self, surah: int, ayah: int) -> Ayah:
        from hafs_phonetizer import phonetize

        phonemes = self.text.phonemes(surah, ayah)
        words = self.uthmani_words(surah, ayah)
        slips: dict[int, str] = {}
        try:
            out = phonetize(" ".join(words))
        except (KeyError, IndexError):
            out = None
        checked = out is not None and out.phonemes.replace(" ", "") == phonemes.replace(" ", "")
        if checked:
            spans = [None if m.deleted else m.pos for m in out.mappings]
            slips = phonetizer_slips(" ".join(words), out.phonemes, spans)
        return Ayah(phonemes, slips, checked, lambda word: self.forms(surah, ayah, word))

    def every_ayah(self) -> Iterable[tuple[int, int]]:
        from quran_transcript import Aya

        for surah in range(1, 115):
            yield from ((surah, ayah) for ayah in range(1, Aya(surah, 1).get().num_ayat_in_sura + 1))

    @cache
    def forms(self, surah: int, ayah: int, word: int) -> dict[str, Form]:
        """The :data:`FORMS` a phoneme word's context can take (those the phonetizer handles)."""
        words = self.uthmani_words(surah, ayah)
        reference = self.text.phonemes(surah, ayah)
        start = word_start(reference, word)
        end = start + len(reference.split(" ")[word - 1])
        printed = self.text.text_words(surah, ayah, word)
        first, last = printed[0], printed[-1]
        spans: dict[str, tuple[list[str], int, int]] = {}
        if last < len(words):
            spans[WAQF_HERE] = (list(words[:last]), 0, end)
        if first > 1:
            spans[IBTIDA_HERE] = (list(words[first - 1:]), start, len(reference))
        if first == 1 and ayah > 1:
            spans[FROM_PREVIOUS] = ([self.uthmani_words(surah, ayah - 1)[-1], *words], 0, len(reference))
        if last == len(words) and self._has(surah, ayah + 1):
            spans[INTO_NEXT] = ([*words, self.uthmani_words(surah, ayah + 1)[0]], 0, len(reference))
        return {name: Form(phonemes, lo, hi) for name, (w, lo, hi) in spans.items()
                if (phonemes := self._phonetized(w)) is not None}

    @cache
    def _has(self, surah: int, ayah: int) -> bool:
        from quran_transcript import Aya

        return ayah <= Aya(surah, 1).get().num_ayat_in_sura


def phonetizer_check(ayat: Ayat) -> dict:
    """The orthography guard (:func:`mushaf_reason`) over every ayah, not only the flagged ones:
    the ayat checked, those it cannot check (the phonetizer does not reproduce Muraja's reference
    there: its fallback ayat), and every slip found."""
    readings = {(s, a): ayat.get(s, a) for s, a in ayat.every_ayah()}
    return {"ayat_checked": sum(r.checked for r in readings.values()),
            "not_checked": [f"{s}:{a}" for (s, a), r in readings.items() if not r.checked],
            "slips": [{"ayah": f"{s}:{a}", "ordinal": q, "reason": why}
                      for (s, a), r in readings.items() for q, why in sorted(r.slips.items())]}


def build_population(name: str, model: str, scoring: str, graded: Iterable[str],
                     flagged: Iterable[FlaggedWord], ayat: Ayat, clustered: bool) -> Population:
    """A population from its graded words' clusters and its flagged words, classified."""
    out: list[FlagRecord] = []
    for f in flagged:
        out += records(f, ayat.get(f.surah, f.ayah), ayat.text.printed(f.surah, f.ayah, f.word))
    return Population(name, model, scoring, dict(sorted(Counter(graded).items())), clustered, tuple(out))


def qud_populations(harness, ayat: Ayat) -> list[Population]:
    """The QUD pilot, both models, every word of every item regraded from the committed check
    sequences; clustered by mushaf and surah."""
    from training.decoding import committed_text
    from training.qud_pilot import MODELS, grade_words, load_pilot
    from training.truth_baseline import CHECKS

    pilot = load_pilot()
    words = grade_words(pilot.items, pilot.decodes, harness)
    out = []
    for model in MODELS:
        decodes = {item.key: committed_text(pilot.decodes[model][item.key][CHECKS]) for item in pilot.items}
        for scoring in SCORING_NAMES:
            graded, flagged = [], []
            for item in pilot.items:
                offsets = item.reference.offsets
                last = words.placed[item.key][-1][0].site_id
                for word, site in words.placed[item.key]:
                    grade = words.grade(scoring, model, item, word)
                    if grade.outcome not in (*WORD_FLAGGED, NOT_FLAGGED):
                        continue
                    cluster = f"{item.slug}:{item.row.surah}"
                    graded.append(cluster)
                    if grade.outcome not in WORD_FLAGGED or grade.quality in (TASHKEEL_ERROR, SKIPPED_QUALITY):
                        continue
                    k = item.reference.word_indices.index(word.word_index)
                    flagged.append(FlaggedWord(
                        QUD, model, scoring, site.surah, site.ayah, site.word, grade.errors,
                        Recital(item.reference.reference, word.site_id == last, decodes[item.key]),
                        cluster, (offsets[k], offsets[k + 1])))
            out.append(build_population(QUD, model, scoring, graded, flagged, ayat, clustered=True))
    return out


def device_log_populations(ayat: Ayat) -> list[Population]:
    """#151's included teacher sessions replayed at the pin, the words its docs explain left out;
    each log's committed transcript is the decode (read here, never written)."""
    from training import device_logs as dl
    from training.muraja_policy import MURAJA_COMMIT, SCORINGS

    pin = dl.require(MURAJA_COMMIT)
    entries = [e for e in dl.load_inventory() if e.included and e.model == dl.TEACHER]
    paths = {e.log: dl.log_dir(MURAJA_COMMIT) / f"{e.log}.log" for e in entries}
    scorings = [s for s in SCORINGS if s.name in SCORING_NAMES]
    _, replays = dl.replay_logs(pin, paths, scorings)
    transcripts = {log: dl.committed_transcript(path.read_text(encoding="utf-8")) for log, path in paths.items()}
    out = []
    for scoring in scorings:
        graded, flagged = [], []
        for e in entries:
            for g in dl.word_grades(e, replays[e.log], [scoring], ayat.text):
                if g.explained:
                    continue
                graded.append(e.log)
                if g.flagged and g.consonants:
                    surah, ayah, word = dl.position(g.word)
                    flagged.append(FlaggedWord(DEVICE_LOGS, dl.TEACHER, scoring.name, surah, ayah, word, g.errors,
                                               Recital(decode=transcripts[e.log]), e.log))
        out.append(build_population(DEVICE_LOGS, dl.TEACHER, scoring.name, graded, flagged, ayat, clustered=True))
    return out


#: The owner's verdict on a word the phone showed, as a class (``owner_sessions.json``).
OWNER_VERDICTS = {"decode": MODEL, "boundary": GRADING}


#: The group of an error whose group the table did not keep: outside every word, so
#: ``word_flags.deleted_run`` reads the occurrence from the word's runs alone.
UNKNOWN_GROUP = -1


def error_without_group(label: str) -> dict:
    """A consonant error back from its substitution (``ل→ء``, ``ه→∅``), its group unknown."""
    expected, heard = label.split("→")
    kind = ({"type": "deletion", "expected": expected} if heard == DELETED
            else {"type": "mismatch", "expected": expected, "heard": heard})
    return {"kind": kind, "group_index": UNKNOWN_GROUP}


def owner_populations(ayat: Ayat) -> list[Population]:
    """#153's committed per-word table (no group index: every unit holding the letter is a
    candidate); the owner's mistakes left out; no interval (three sessions)."""
    from training.device_exports import MODEL_ERROR, REPORT_PATH as OWNER_REPORT
    from training.device_logs import UNEXPLAINED, position

    report = json.loads(OWNER_REPORT.read_text(encoding="utf-8"))
    out = []
    for scoring in SCORING_NAMES:
        words = report["rates"][scoring][UNEXPLAINED]["false_flags"]["words"]
        flagged = []
        for row in report["words"]:
            if row["scoring"] != scoring or row["explained"] is not None or not row["substitutions"]:
                continue
            surah, ayah, word = position(row["word"])
            verdict = row["verdict"]
            evidence = OWNER_VERDICTS.get(verdict.get("explained_by")) if verdict["status"] == MODEL_ERROR else None
            errors = tuple(error_without_group(label) for label in row["substitutions"])
            flagged.append(FlaggedWord(OWNER, "h448", scoring, surah, ayah, word, errors,
                                       Recital(verdict=evidence), row["log"]))
        # three sessions carry no interval: the words are counted, not clustered
        out.append(build_population(OWNER, "h448", scoring, ["sessions"] * words, flagged, ayat, clustered=False))
    return out


def tadabur_populations(harness, ayat: Ayat) -> list[Population]:
    """#90's word sample and Ship A's held-out verified sites (correct side) on ``h448``'s
    recorded checks, regraded from the committed check sequences; a word is a site's word."""
    from tadabur.exclusions import current
    from tadabur.listening_session import adjudicated, read_verdicts
    from tadabur.ship_a import HELD_OUT
    from tadabur.ship_a_verified import SHIP_A, load_designs
    from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips
    from tadabur.word_sample import load_words, verified_words
    from training import truth_baseline as tb
    from training.decoding import committed_text
    from training.site_outcomes import CORRECT_SIDE
    from training.truth_scorer import item_key

    arm = tb.arm("h448", tb.CHECKS)
    verdicts = read_verdicts()
    designs = load_designs()
    excluded = current()
    sites = [s for s in adjudicated(tb.load_sites(tb.truth_site_files()), verdicts)
             if s.site_id in designs and (designs[s.site_id].frame, designs[s.site_id].half) == (SHIP_A, HELD_OUT)]
    words = verified_words(load_words(), verdicts)
    registry = load_staged_clips(REGISTRY_PATH)
    items = tb.items_of([*sites, *(w.site for w in words)], registry)
    _, recorded = tb.model_caches("h448", tb.DEFAULT_MODELS["h448"], items, registry, None)
    sequences = {arm: {k: v[tb.CHECKS] for k, v in recorded["items"].items() if k in items}}
    decodes = {arm: {k: committed_text(c) for k, c in sequences[arm].items()}}
    _, kept = tb.app_outcomes(sites, words, decodes, sequences, harness)
    correct_sites = [s for s in sites if s.heard == s.prescribed and s.site_id not in excluded]
    correct_words = [w.site for w in words if w.side == CORRECT_SIDE and w.site.site_id not in excluded]
    out = []
    for name, located in ((WORD_SAMPLE, correct_words), (HELD_OUT_SITES, correct_sites)):
        for scoring in SCORING_NAMES:
            grades: Mapping[str, SiteGrade] = kept.grades[scoring][arm]
            graded, flagged, seen = [], [], set()
            for site in located:
                grade = grades[site.site_id]
                where = ayat.text.locate(site)
                if where is None or grade.outcome not in (*WORD_FLAGGED, NOT_FLAGGED):
                    continue
                graded.append(item_key(site))
                key = (item_key(site), where.word)
                if (grade.outcome not in WORD_FLAGGED or grade.quality in (TASHKEEL_ERROR, SKIPPED_QUALITY)
                        or key in seen):
                    continue
                seen.add(key)
                span = _word_span(site.reference, site.reference_index)
                flagged.append(FlaggedWord(
                    name, "h448", scoring, where.surah, where.ayah, where.word, grade.errors,
                    Recital(site.reference, span[1] >= len(site.reference.rstrip()), decodes[arm][item_key(site)]),
                    item_key(site), span))
            out.append(build_population(name, "h448", scoring, graded, flagged, ayat, clustered=False))
    return out


def _word_span(reference: str, index: int) -> tuple[int, int]:
    """The space-delimited word of ``reference`` around ``index``."""
    start = reference.rfind(" ", 0, index) + 1
    end = reference.find(" ", index)
    return start, len(reference) if end < 0 else end


# --- running it --------------------------------------------------------------------------------------


def build() -> dict:
    from training.muraja_policy import MURAJA_COMMIT, require_harness

    harness = require_harness()
    ayat = Ayat(MurajaText(harness.quran_db))
    populations = [*qud_populations(harness, ayat), *device_log_populations(ayat), *owner_populations(ayat),
                   *tadabur_populations(harness, ayat)]
    return {
        "issue": 155,
        "muraja_commit": MURAJA_COMMIT,
        "phonetizer_check": phonetizer_check(ayat),
        "populations": [population_view(p) for p in populations],
        # each population and model, its scorings' flags together (a word once)
        "owner_checks": [{"population": name, "model": model, "checks": owner_checks(
            [r for p in populations if (p.name, p.model) == (name, model) for r in p.records])}
            for name, model in dict.fromkeys((p.name, p.model) for p in populations)],
        "records": [{k: v for k, v in asdict(r).items() if v is not None or k != "decode"}
                    for p in populations for r in p.records],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="classify every population's consonant false flags (needs the harness)")
    commands.add_parser("doc", help=f"render {DOC_PATH.name} from the report")
    args = parser.parse_args()
    if args.command == "build":
        report = build()
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        write_text_atomically(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {REPORT_PATH}")
    from training.consonant_flags_doc import render

    write_text_atomically(DOC_PATH, render(json.loads(REPORT_PATH.read_text(encoding="utf-8"))))
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()

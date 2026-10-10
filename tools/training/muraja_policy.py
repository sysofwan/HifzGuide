"""Muraja's grading, replayed at truth sites: per-word grades over a sequence of decodes.

The ship criterion (acceptance rules §3) compares a candidate system with **today's system**:
``h448`` at b = 0, no bias, and the Muraja configuration :data:`TODAY` pins here. This module
replays what Muraja does with each word that holds a truth site, over the decodes of the
cycles that grade it, and reads off what that means for each site.

Every rule below was read from ``sysofwan/Muraja`` at :data:`MURAJA_REVISION` (v1.0.27,
``main`` on 2026-10-09). Paths are under ``ios/HifzGuide/``; ``WS`` is
``FollowAlong/QuranFollowAlong+WordScoring.swift`` and ``FAT`` is
``FollowAlong/FollowAlongTypes.swift``.

Cycles and the ratchet
----------------------
* The model sees a 5 s window that advances 1 s per step; CTC segments whose midpoint lies in
  the first second are the hop, the rest the overlap (``Models/RealtimeTranscriber+iOS.swift:
  169-188``, ``Models/MuaalemInference+iOS.swift:128-149``). Between steps a **preview**
  decode runs whenever 200 ms of new audio has arrived (``RealtimeTranscriber+iOS.swift:
  238-240, 617-627``).
* Every hop and every preview runs a check (``FollowAlong/FollowAlongEngine.swift:625-639``)
  over the last 20 confirmed characters plus the pending hops plus the current overlap or
  preview (``FollowAlongEngine.swift:237-239, 1565-1628, 3656-3658``). A check grades every
  word its alignment overlaps, up to its **end word**: the word the alignment ends on, or the
  one before when the alignment covers less than 30% of it (``FollowAlong/
  QuranFollowAlong.swift:385-413, 489-506``; ``WS:362-382``).
* ``GradeStore`` keeps each word's best grade. A new grade replaces the kept one only if its
  rank is higher, or it is the same quality scoring more than 0.01 higher (``FollowAlong/
  GradeStore.swift:175-207, 295-411``). Ranks: pending 0, skipped 1, uncertain and wrong 2,
  tashkeelError 3, minor 4, correct 5 (``FAT:367-392``). Wrong, uncertain and skipped grades
  wait 4 cycles before they show and are then promoted only over a lower rank
  (``GradeStore.swift:130, 543-600``): that changes when a grade shows, not which one is kept.
  Nothing finalizes a grade, so one clean cycle locks a word correct and a mistake decoded
  clean once is never flagged.
* The end word passes a correct grade through and holds any other back as pending
  (``WS:748-766``). A session that ends on a word settles its withheld grade
  (``GradeStore.swift:126-134``, #141); that is not replayed here.

One cycle's grade of one word (``computeWordStatuses``, ``WS:335-771``)
-----------------------------------------------------------------------
* **Scored span.** Madd markers ``ۦ ۥ ۾ ڇ`` (``WS:58-66``), a trailing run of one madd letter
  beyond its first (``WS:110-131``), leading ghunna ``ں`` (``WS:133-149``) and the bare copies
  of a word-initial **assimilated geminate** (``WS:151-191``) are not scored. At the end word a
  trailing tanween ``ن``/``ں`` (or a doubled ``ي``) is trimmed, and in every word a trailing
  connection ``و``/``ں`` run (``WS:79-108, 219-252, 395-406``).
* **Alignment.** Both strings are normalized into groups; a reference geminate is two groups
  (bare, then voweled) and a decoded bare double merges into one (``FollowAlong/
  PhonemeNormalization.swift:67-137``). Each reference character takes its group's outcome:
  match, tashkeel (same consonant, the trailing haraka differs, ``WS:283-307``), mismatch or
  gap. A trailing haraka is fatha, damma or kasra only; class 35 (U+06EA, "sukun" in
  ``Models/PhonemeVocabulary.swift:49``) is a residual (``PhonemeNormalization.swift:12-29``),
  so **a sukun the reciter said and nothing emitted both arrive as** ``heard: nil``.
* **Tashkeel** (``WS:529-562``) is counted unless the position is the final consonant of a
  **waqf word** (the end word, a word with a waqf sign, or the last word of the ayah:
  ``WS:388-396, 427-436, 532``), or nothing was heard and the letter is one of ``و ا ء ي``
  (``WS:42-56, 544``) or ``suppressHarakaDrop`` is on (``FAT:75-80``, ``WS:541``). A wrong
  haraka is counted in every mode. A group that holds a gap discards its tashkeel
  (``WS:492-508, 640-646``).
* **Mismatch** (``WS:563-627``): the final consonant of a waqf word gets full credit
  (``WS:598-603``). Otherwise it costs the graduated sifat penalty (``FollowAlong/
  PhonemeSifat.swift:259-275``) and is **hard** unless the pair is a soft pair under
  ``softPairsEnabled``: ``ذ↔ز ت↔ط ض↔ظ ك↔ق س↔ص ح↔ه`` (``PhonemeSifat.swift:205-225``).
  ``ذ↔ظ`` is not one.
* **Gap**: one per reference character. A group that holds a gap and a decoded copy of the
  same consonant (a collapsed geminate) has one **shaddah gap** (``WS:509-522``).
* **Score** (``WS:648-708``): (matches + Σ mismatch credit) / scored characters, times
  coverage; a local re-alignment may raise a score below ``reAlignThreshold`` (``WS:672-694,
  783-852``); a word of at most two groups scoring ≥ 0.30 gains ×1.5, the last word of an ayah
  ×1.3 (the larger only).
* **Quality** (``WS:714-746``): score ≥ ``correctThreshold`` → correct, or tashkeelError if any
  tashkeel counted; every consonant right but a tashkeel counted → tashkeelError; score ≥
  ``minorThreshold`` → minor; else uncertain (coverage < 0.5) or wrong. With
  ``phonemeGateEnabled`` a hard mismatch or a gap (shaddah gaps excluded under
  ``shaddahSuppression``) turns correct or tashkeelError into minor.
* **Display** (``FollowAlong/GradeFilter+iOS.swift:58-87``): with tashkeel detection off a
  tashkeelError shows as correct; in ``.lenient`` a minor shows as correct.

Modes (``FAT:109-147``): ``strict`` (thresholds 0.75 / 0.50, no soft pairs, no shaddah
suppression); ``balanced`` (0.65 / 0.40, soft pairs, shaddah suppression); ``lenient``
(0.55 / 0.30, mismatch best credit 0.2, re-align below 0.55, the lenient sifat boost, no
phoneme gate, ``suppressHarakaDrop``, minor shown as correct). The app defaults to
``balanced`` (``Data/AppSettings+iOS.swift:89-104``, ``GradeFilter+iOS.swift:52``) with tashkeel
detection on (``AppSettings+iOS.swift:63``). Each of :data:`PRESETS` is a composition of the
toggles of :class:`MurajaConfig`, and each of :data:`ALLOWANCES` switches one toggle off.

What it means for a site
------------------------
A truth site tests one mark or letter. :func:`site_edit` turns a decode's
:class:`training.site_outcomes.SiteOutcome` into what Muraja's alignment holds at the site's
groups; every other group of the word is taken as decoded like the reference (**isolation**:
truth sites are the only positions with a known outcome). :func:`grade_item` grades each word
that holds a site in every :class:`Cycle` that reaches it, folds the grades through the
ratchet, shows the kept one through the display filters, and gives each site one of

* :data:`CORRECT`: the kept grade checked the site and the decode had it as prescribed;
* :data:`WRONG`: the word shows as not correct and this site's deviation is why (Muraja counted
  it, or the word would show correct without it): a **false flag** on correct recitation, a
  **caught** mistake on a real one;
* :data:`NOT_GRADED`: nothing showed against the site. An allowance or exemption let its
  deviation pass; or its carrier was not decoded (a tashkeel or shaddah site has no slot then,
  and the consonant error belongs to the consonant); or the deviation is invisible (an inserted
  gemination, a geminate held as one group); or the position is never scored; or no cycle
  produced a grade. A not-graded mistake is a missed one; not-graded sites reduce **coverage**.

The single-decode approximation
-------------------------------
HifzGuide decodes each item once per protocol (whole span, or the stream at b = 0), not once
per Muraja cycle. :func:`single_decode_cycles` lets that one decode stand for every cycle and
varies only which word a cycle ends on (:data:`APPROXIMATIONS`, recorded in each report):

* :data:`RUN_ENDS`: each word that ends a recited run (the item's last word, or a word a human
  heard a pause after) is the end word of one cycle, and a last cycle ends past the item. The
  ratchet never sees a different decode, so false flags a cleaner cycle would clear stay, and
  mistakes a clean-decoding cycle would hide count as caught.
* :data:`EVERY_WORD_ENDS`: every word is the end word of one cycle, as when a check ends on
  each word while the reciter passes it, so every word-final consonant gets the waqf exemption.
  A bound on what ``run_ends`` leaves out, not a measurement.

Mushaf waqf signs and the last-word-of-ayah boost are not known per word here and are not
modelled; both can only add flags. A real sequence of decodes goes through
:func:`grade_decode_sequence`, which aligns each check's decode with the item's reference to
find the words it grades and its end word.

The candidate rule
------------------
``empty_slot_not_graded`` (ADR-0011 §2, the candidate system of acceptance rules §3) is not in
Muraja yet: an empty tashkeel slot is never counted, and a committed mark other than the
prescribed one (sukun included) is a tashkeel error.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache

from tadabur.normalization import cluster_offsets, normalize_phonemes
from tadabur.phoneme_sifat import graduated_mismatch_score, soft_pair_contrast
from tadabur.smith_waterman import local_alignment_score, smith_waterman
from tadabur.truth_sites import HARAKA_CHARS, HELD, SHADDAH, SOFT_PAIRS, WAQF_BOUNDARY, TruthSite
from tadabur.waqf_truth_sites import WAQF
from training.site_outcomes import TASHKEEL, SiteOutcome, family, item_outcomes

MURAJA_REVISION = "99c326f3fb7c6c53ff525568976c7e2442ccde3c"

# --- a site's grade ------------------------------------------------------------------------
CORRECT = "correct"
WRONG = "wrong"
NOT_GRADED = "not_graded"
GRADES = (CORRECT, WRONG, NOT_GRADED)

#: The stratum of the boundaries a human heard as a pause.
PAUSE_STRATUM = f"{WAQF_BOUNDARY}:{WAQF}"


class Quality(Enum):
    """Muraja's ``WordQuality`` and its ratchet rank (``FAT:367-392``)."""

    PENDING = "pending"
    SKIPPED = "skipped"
    UNCERTAIN = "uncertain"
    WRONG = "wrong"
    TASHKEEL_ERROR = "tashkeelError"
    MINOR = "minor"
    CORRECT = "correct"

    @property
    def rank(self) -> int:
        return _RANKS[self]


_RANKS = {
    Quality.PENDING: 0,
    Quality.SKIPPED: 1,
    Quality.UNCERTAIN: 2,
    Quality.WRONG: 2,
    Quality.TASHKEEL_ERROR: 3,
    Quality.MINOR: 4,
    Quality.CORRECT: 5,
}
#: A same-quality grade replaces the kept one only when it scores more than this much higher.
SCORE_UPGRADE_MARGIN = 0.01

# --- characters Muraja treats specially (WS:29-75, PhonemeNormalization.swift:8-38) -------
MADD_SCALARS = frozenset("ۦۥ۾ڇ")  # ۦ ۥ ۾ ڇ
MADD_LETTERS = frozenset("يوا")
#: و ا ء ي: the letters that double as long-vowel carriers, exempt from a dropped haraka.
HARAKA_DROP_LETTERS = frozenset("واءي")
TANWEEN_SCALARS = frozenset("ںن")  # ں ن
CONNECTION_SCALARS = frozenset("وں")  # و ں
GHUNNA_NUN = "ں"
YA = "ي"
_TAJWEED_EQUIVALENTS = {"۾": "م", GHUNNA_NUN: "ن"}
_CORE_SCALARS = frozenset("ءبتثجحخدذرزسشصضطظعغفقكلمنهوياۥۦ۾ںـٲ")
_HARAKA_NAMES = {char: name for name, char in HARAKA_CHARS.items()}


def _fold(scalar: str) -> str:
    return _TAJWEED_EQUIVALENTS.get(scalar, scalar)


# --- configuration ------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreParameters:
    """The numeric half of ``ScoringParameters`` (``FAT:53-81``)."""

    mismatch_best_credit: float
    mismatch_worst_penalty: float
    correct_threshold: float
    minor_threshold: float
    realign_threshold: float
    lenient_sifat_boost: bool
    phoneme_gate_enabled: bool


STRICT_SCORES = ScoreParameters(0.0, -0.5, 0.75, 0.50, 0.75, False, True)  # FAT:111-122
BALANCED_SCORES = replace(STRICT_SCORES, correct_threshold=0.65, minor_threshold=0.40)  # :124-129
LENIENT_SCORES = replace(  # FAT:131-139
    BALANCED_SCORES,
    mismatch_best_credit=0.2,
    correct_threshold=0.55,
    minor_threshold=0.30,
    realign_threshold=0.55,
    lenient_sifat_boost=True,
    phoneme_gate_enabled=False,
)


@dataclass(frozen=True)
class MurajaConfig:
    """A mode's scores plus one toggle per allowance and the display filters.

    Allowance toggles (each switched off by one of :data:`ALLOWANCES`):

    * ``soft_pairs``: the pairs that bypass the phoneme gate; ``softPairsEnabled`` sets all
      six (``PhonemeSifat.swift:205-225``), kept per pair so each can be retired alone.
    * ``shaddah_suppression``: ``shaddahSuppression`` (``FAT:69-71``, ``WS:740``).
    * ``suppress_haraka_drop``: ``suppressHarakaDrop`` (``FAT:75-80``, ``WS:541``).
    * ``haraka_drop_letters``: the letters exempt from a dropped haraka (``WS:42-56, 544``).
    * ``final_at_waqf_tashkeel`` and ``final_at_waqf_consonant``: ``isFinalAtWaqf`` for the
      final consonant's tashkeel (``WS:532, 551``) and for the consonant (``WS:598-603``).
    * ``group_has_gap``: a group holding a gap discards its tashkeel (``WS:492-508``).
    * ``leading_assimilation``: the bare copies of a word-initial assimilated geminate are not
      scored (``WS:151-191, 470``).

    Display filters: ``tashkeel_errors`` (``GradeFilter+iOS.swift:68-73``) and ``mask_minor``
    (``:80-85``). ``empty_slot_not_graded`` is the candidate rule, off in every preset.
    """

    revision: str
    mode: str
    scores: ScoreParameters
    soft_pairs: frozenset[str]
    shaddah_suppression: bool
    suppress_haraka_drop: bool
    haraka_drop_letters: frozenset[str]
    final_at_waqf_tashkeel: bool
    final_at_waqf_consonant: bool
    group_has_gap: bool
    leading_assimilation: bool
    tashkeel_errors: bool = True
    mask_minor: bool = False
    empty_slot_not_graded: bool = False

    def as_dict(self) -> dict:
        return {
            "revision": self.revision,
            "mode": self.mode,
            "scores": dict(vars(self.scores)),
            "soft_pairs": sorted(self.soft_pairs),
            "shaddah_suppression": self.shaddah_suppression,
            "suppress_haraka_drop": self.suppress_haraka_drop,
            "haraka_drop_letters": sorted(self.haraka_drop_letters),
            "final_at_waqf_tashkeel": self.final_at_waqf_tashkeel,
            "final_at_waqf_consonant": self.final_at_waqf_consonant,
            "group_has_gap": self.group_has_gap,
            "leading_assimilation": self.leading_assimilation,
            "tashkeel_errors": self.tashkeel_errors,
            "mask_minor": self.mask_minor,
            "empty_slot_not_graded": self.empty_slot_not_graded,
        }


def _preset(mode: str, scores: ScoreParameters, *, soft: bool, suppression: bool, lenient: bool):
    return MurajaConfig(
        revision=MURAJA_REVISION,
        mode=mode,
        scores=scores,
        soft_pairs=frozenset(SOFT_PAIRS) if soft else frozenset(),
        shaddah_suppression=suppression,
        suppress_haraka_drop=lenient,
        haraka_drop_letters=HARAKA_DROP_LETTERS,
        final_at_waqf_tashkeel=True,
        final_at_waqf_consonant=True,
        group_has_gap=True,
        leading_assimilation=True,
        mask_minor=lenient,
    )


#: Muraja's three modes (``FAT:109-147``), tashkeel detection on.
PRESETS: dict[str, MurajaConfig] = {
    "strict": _preset("strict", STRICT_SCORES, soft=False, suppression=False, lenient=False),
    "balanced": _preset("balanced", BALANCED_SCORES, soft=True, suppression=True, lenient=False),
    "lenient": _preset("lenient", LENIENT_SCORES, soft=True, suppression=True, lenient=True),
}
#: Today's system: Muraja v1.0.27 defaults, ``.balanced``, tashkeel error detection on.
TODAY = PRESETS["balanced"]


# --- the words of an item -----------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Word:
    """One space-separated word of an item's realized reference, as Muraja groups it.

    ``clusters`` are Swift Characters (a base scalar and its combining marks), ``groups`` the
    normalized groups as cluster ranges, ``item`` every cluster of the item. Compared by
    identity: :func:`item_words` builds each word once per reference.
    """

    index: int
    is_last: bool
    start: int  # code-point offset of the word in the reference
    clusters: tuple[str, ...]
    groups: tuple[tuple[int, int], ...]
    item: tuple[str, ...]
    item_offset: int  # index in ``item`` of the word's first cluster

    def group_of(self, cluster: int) -> int:
        return next(g for g, (lo, hi) in enumerate(self.groups) if lo <= cluster < hi)


def _clusters(text: str) -> tuple[str, ...]:
    offsets = cluster_offsets(text)
    return tuple(text[a:b] for a, b in zip(offsets, [*offsets[1:], len(text)]))


@lru_cache(maxsize=4096)
def item_words(reference: str) -> tuple[Word, ...]:
    """The words of a realized reference, in order."""
    item, pieces = _clusters(reference), reference.split(" ")
    words, start, offset = [], 0, 0
    for index, text in enumerate(pieces):
        clusters = _clusters(text)
        groups = tuple(normalize_phonemes(text).offset_map)
        words.append(Word(index, index == len(pieces) - 1, start, clusters, groups, item, offset))
        start += len(text) + 1
        offset += len(clusters) + 1
    return tuple(words)


@dataclass(frozen=True)
class Location:
    """A site's word, its carrier's cluster and group, and the groups of the carrier's run of
    one consonant (a geminate's two groups, or the carrier's own)."""

    word: Word
    cluster: int
    group: int
    run: tuple[int, ...]


@lru_cache(maxsize=65536)
def _locate(reference: str, index: int) -> Location:
    word = next(w for w in reversed(item_words(reference)) if w.start <= index)
    position = index - word.start
    offsets = cluster_offsets("".join(word.clusters))
    cluster = max(i for i, offset in enumerate(offsets) if offset <= position)
    base = word.clusters[cluster][0]
    lo, hi = cluster, cluster
    while lo > 0 and word.clusters[lo - 1][0] == base:
        lo -= 1
    while hi + 1 < len(word.clusters) and word.clusters[hi + 1][0] == base:
        hi += 1
    run = tuple(sorted({word.group_of(c) for c in range(lo, hi + 1)}))
    return Location(word, cluster, word.group_of(cluster), run)


def locate(site: TruthSite) -> Location:
    return _locate(site.reference, site.reference_index)


def ends_run(site: TruthSite) -> bool:
    """The site's word ends a recited run: the item's last word, or a heard pause follows it."""
    return locate(site).word.is_last or site.stratum == PAUSE_STRATUM


@dataclass(frozen=True)
class Span:
    """The part of a word Muraja scores in one cycle (``WS:395-436``), in cluster indices:
    ``[skip, scored_end)`` minus madd markers; ``end`` is the end after the trims and
    ``last_consonant`` the start of the waqf-final position."""

    end: int
    skip: int
    scored_end: int
    last_consonant: int
    tanween: int


def _waqf_trim(clusters: Sequence[str]) -> int:
    """``WS:85-108``."""
    if len(clusters[-1]) == 1 and clusters[-1] in TANWEEN_SCALARS:
        return 1
    run = _bare_run(clusters, {YA})
    return run if run >= 2 else 0


def _bare_run(clusters: Sequence[str], letters: Collection[str]) -> int:
    run = 0
    while run < len(clusters) and len(clusters[-1 - run]) == 1 and clusters[-1 - run] in letters:
        run += 1
    return run


def _connection_trim(clusters: Sequence[str]) -> int:
    """``WS:219-252``."""
    run = _bare_run(clusters, CONNECTION_SCALARS)
    if run:
        return run
    run = _bare_run(clusters, {YA})
    return run if run >= 2 else 0


def _trailing_elongation(clusters: Sequence[str], end: int) -> int:
    """``WS:113-131``."""
    if end <= 0 or clusters[end - 1][0] not in MADD_LETTERS:
        return 0
    last, count = clusters[end - 1][0], 0
    while end - 1 - count > 0 and clusters[end - 2 - count][0] == last:
        count += 1
    return count


def _leading_tanween(clusters: Sequence[str], end: int) -> int:
    """``WS:137-149``."""
    count = 0
    while count < end and clusters[count][0] == GHUNNA_NUN:
        count += 1
    return count


def _leading_assimilation(clusters: Sequence[str], start: int, end: int) -> int:
    """``WS:156-191``."""
    if end - start < 2 or len(clusters[start]) != 1:
        return 0
    base = clusters[start][0]
    if base not in _CORE_SCALARS or base == GHUNNA_NUN:
        return 0
    count = start
    while count < end and _fold(clusters[count][0]) == _fold(base) and len(clusters[count]) == 1:
        count += 1
    if count < end and _fold(clusters[count][0]) == _fold(base):
        return count - start
    return 0


def scoring_span(word: Word, end_word: bool, leading_assimilation: bool = True) -> Span:
    """Which clusters Muraja scores in a cycle where ``word`` is, or is not, the end word."""
    clusters = word.clusters
    trim = _connection_trim(clusters)
    if end_word:
        trim = max(_waqf_trim(clusters), trim)
    end = max(len(clusters) - trim, 0)
    elongation = _trailing_elongation(clusters, end)
    tanween = _leading_tanween(clusters, end)
    assimilation = _leading_assimilation(clusters, tanween, end) if leading_assimilation else 0
    scored_end = end - elongation
    last_consonant = scored_end - 1
    while last_consonant > 0 and clusters[last_consonant][0] in MADD_SCALARS:
        last_consonant -= 1
    return Span(end, tanween + assimilation, scored_end, last_consonant, tanween)


# --- what the alignment holds at a group --------------------------------------------------
@dataclass(frozen=True)
class Gap:
    """The group was not decoded."""


@dataclass(frozen=True)
class Tashkeel:
    """Same consonant, another trailing mark: a haraka name (``sukun`` or ``multiple`` too,
    under the candidate rule), or ``None`` for nothing heard."""

    heard: str | None


@dataclass(frozen=True)
class Swap:
    """Another consonant."""

    letter: str


GAP = Gap()
State = Gap | Tashkeel | Swap


@dataclass(frozen=True)
class SiteEdit:
    """What one decode did at a site, as the groups of its word that differ from the reference.

    ``deviates``: the decode differs from the prescribed mark or letter. ``observable``: the
    difference can reach Muraja's grade (an unaligned tashkeel or shaddah carrier, an inserted
    gemination, or a geminate held in one group cannot).
    """

    states: Mapping[int, State]
    deviates: bool
    observable: bool


_AS_PRESCRIBED = SiteEdit({}, False, True)


def site_edit(site: TruthSite, outcome: SiteOutcome, config: MurajaConfig = TODAY) -> SiteEdit:
    """The groups a decode's outcome at ``site`` changes, every other group as the reference."""
    loc = locate(site)
    kind = family(site.mark)
    if kind == TASHKEEL:
        if not outcome.carrier_aligned:
            return SiteEdit({loc.group: GAP}, True, False)
        collapsed = {g: GAP for g in loc.run if g != loc.group} if outcome.geminate_collapsed else {}
        if config.empty_slot_not_graded:
            heard, expected = outcome.committed, site.prescribed
        else:
            heard = outcome.marks[-1] if outcome.marks and outcome.marks[-1] in HARAKA_CHARS else None
            expected = _HARAKA_NAMES.get(loc.word.clusters[loc.cluster][-1])
        mark = {loc.group: Tashkeel(heard)} if heard != expected else {}
        return SiteEdit({**collapsed, **mark}, bool(mark), True)
    if kind == SHADDAH:
        if not outcome.carrier_aligned:
            return SiteEdit({g: GAP for g in loc.run}, True, False)
        if outcome.committed == site.prescribed:
            return _AS_PRESCRIBED
        if site.prescribed == HELD and len(loc.run) > 1:  # held as one consonant: one group gaps
            return SiteEdit({loc.run[0]: GAP}, True, True)
        return SiteEdit({}, True, False)  # an inserted gemination, or a geminate in one group
    # A pair site's outcome is the consonant aligned to its carrier's group; a geminate's other
    # copy keeps the reference (isolation).
    if outcome.committed is None:
        return SiteEdit({loc.group: GAP}, True, True)
    if outcome.committed == site.prescribed:
        return _AS_PRESCRIBED
    return SiteEdit({loc.group: Swap(outcome.committed)}, True, True)


# --- one cycle's grade of one word ----------------------------------------------------------
@dataclass(frozen=True)
class Evaluation:
    """One cycle's grade of a word, before the end-word hold and the display filters.

    ``counted``: groups whose deviation Muraja counted against the word by rule (a tashkeel
    error, or a hard mismatch or gap the phoneme gate fired on). ``unscored``: groups outside
    the scored span. ``tashkeel_exempt`` and ``swap_exempt``: groups where no tashkeel
    deviation, or no consonant swap, would count in this cycle (unscored, final at waqf, or a
    tashkeel group that held a gap). A gap is exempt only where unscored.
    """

    quality: Quality
    score: float
    counted: frozenset[int]
    unscored: frozenset[int]
    tashkeel_exempt: frozenset[int]
    swap_exempt: frozenset[int]


def evaluate(word: Word, states: Mapping[int, State], end_word: bool, config: MurajaConfig) -> Evaluation | None:
    """One cycle's grade of ``word`` with ``states`` at its groups and every other group
    decoded as the reference; ``None`` when Muraja scores nothing of the word."""
    return _evaluate(word, tuple(sorted(states.items())), end_word, config)


@lru_cache(maxsize=262144)
def _evaluate(word: Word, states: tuple, end_word: bool, config: MurajaConfig) -> Evaluation | None:
    return _WordScoring(word, dict(states), end_word, config).run()


class _WordScoring:
    """``computeWordStatuses`` (``WS:384-746``) for one word in isolation."""

    def __init__(self, word: Word, states: dict, end_word: bool, config: MurajaConfig):
        self.word, self.states, self.end_word, self.config = word, states, end_word, config
        self.matches = self.mismatches = self.hard = self.gaps = self.shaddah_gaps = 0
        self.aligned = self.tashkeel_count = 0
        self.mismatch_credit = 0.0
        self.counted: set[int] = set()
        self.gap_groups: set[int] = set()  # groups in a Muraja group that held a gap
        self.gate_gaps: list[tuple[int, int]] = []  # (group, Muraja group id) per gap
        self.shaddah_group_ids: set[int] = set()
        self.group_id = 0
        self._new_group()

    def _new_group(self) -> None:
        self.members: set[int] = set()
        self.pending, self.pending_groups = 0, set()
        self.has_gap = self.has_non_gap = self.shaddah_gap_counted = False

    def _flush(self) -> None:
        """The group's deferred tashkeel counts unless the group held a gap (``WS:492-508``)."""
        if self.has_gap:
            self.gap_groups |= self.members
        if not (self.has_gap and self.config.group_has_gap):
            self.tashkeel_count += self.pending
            self.counted |= self.pending_groups

    def run(self) -> Evaluation | None:
        word, config, params = self.word, self.config, self.config.scores
        clusters = word.clusters
        span = scoring_span(word, self.end_word, config.leading_assimilation)
        madd = sum(c[0] in MADD_SCALARS for c in clusters[: span.end])
        scoring_length = span.scored_end - span.skip - sum(
            c[0] in MADD_SCALARS for c in clusters[span.skip : span.scored_end]
        )
        if span.end <= 0 or scoring_length <= 0:
            return None
        last_query, last_base = -999, None
        for c in range(span.skip, span.scored_end):
            base = clusters[c][0]
            if base in MADD_SCALARS:
                continue
            self.aligned += 1
            g = word.group_of(c)
            state = self.states.get(g)
            is_gap = isinstance(state, Gap)
            query = -1 if is_gap else g
            same = (query == last_query and query >= 0) or (
                last_base is not None and base == last_base and (query < 0 or last_query < 0)
            )
            if not same:
                self._flush()
                self.group_id += 1
                self._new_group()
            self.members.add(g)
            if same and not self.shaddah_gap_counted and (self.has_non_gap if is_gap else self.has_gap):
                self.shaddah_gaps += 1
                self.shaddah_gap_counted = True
                self.shaddah_group_ids.add(self.group_id)
            self.has_gap |= is_gap
            self.has_non_gap |= not is_gap
            last_query, last_base = query, base
            self._score(c, g, base, state, final=self.end_word and c >= span.last_consonant)
        self._flush()

        coverage = (span.end - madd) / scoring_length
        accuracy = max(0.0, (self.matches + self.mismatch_credit) / self.aligned)
        score = accuracy * min(1.0, coverage)
        if score < params.realign_threshold and scoring_length >= 2:
            realigned = _realign_score(word, self.states, span)
            if realigned is not None and realigned > score:
                score = realigned
        if _group_count(clusters, span) <= 2 and score >= 0.30:
            score = min(1.0, score * 1.5)
        quality = self._quality(score, coverage)

        scored = {word.group_of(c) for c in range(span.skip, span.scored_end) if clusters[c][0] not in MADD_SCALARS}
        unscored = set(range(len(word.groups))) - scored
        final = {word.group_of(c) for c in range(span.last_consonant, span.scored_end)} if self.end_word else set()
        tashkeel_exempt = set(unscored)
        if config.final_at_waqf_tashkeel:
            tashkeel_exempt |= final
        if config.group_has_gap:
            tashkeel_exempt |= self.gap_groups
        swap_exempt = unscored | (final if config.final_at_waqf_consonant else set())
        return Evaluation(
            quality,
            score,
            frozenset(self.counted),
            frozenset(unscored),
            frozenset(tashkeel_exempt),
            frozenset(swap_exempt),
        )

    def _score(self, c: int, g: int, base: str, state, final: bool) -> None:
        config, params = self.config, self.config.scores
        if state is None:
            self.matches += 1
        elif isinstance(state, Tashkeel):  # WS:529-562
            self.matches += 1
            dropped = state.heard is None and (
                config.suppress_haraka_drop or config.empty_slot_not_graded or base in config.haraka_drop_letters
            )
            if not (final and config.final_at_waqf_tashkeel) and not dropped:
                self.pending += 1
                self.pending_groups.add(g)
        elif isinstance(state, Swap):  # WS:597-626
            if final and config.final_at_waqf_consonant:
                self.matches += 1
                return
            self.mismatches += 1
            self.mismatch_credit += graduated_mismatch_score(
                base, state.letter,
                worst_penalty=params.mismatch_worst_penalty,
                best_mismatch=params.mismatch_best_credit,
                fallback=params.mismatch_worst_penalty,
                lenient=params.lenient_sifat_boost,
            )
            if soft_pair_contrast(base, state.letter) not in config.soft_pairs:
                self.hard += 1
                if params.phoneme_gate_enabled:
                    self.counted.add(g)
        else:
            self.gaps += 1
            self.gate_gaps.append((g, self.group_id))

    def _quality(self, score: float, coverage: float) -> Quality:
        """``WS:714-746``."""
        config, params = self.config, self.config.scores
        if score >= params.correct_threshold:
            quality = Quality.TASHKEEL_ERROR if self.tashkeel_count else Quality.CORRECT
        elif self.tashkeel_count and not self.mismatches and not self.gaps and coverage >= 0.5:
            quality = Quality.TASHKEEL_ERROR
        elif score >= params.minor_threshold:
            quality = Quality.MINOR
        elif coverage < 0.5:
            quality = Quality.UNCERTAIN
        else:
            quality = Quality.WRONG
        if not params.phoneme_gate_enabled:
            return quality
        suppressed = self.shaddah_group_ids if config.shaddah_suppression else set()
        effective = self.gaps - (self.shaddah_gaps if config.shaddah_suppression else 0)
        if effective:
            self.counted |= {g for g, gid in self.gate_gaps if gid not in suppressed}
        if (self.hard or effective) and quality in (Quality.CORRECT, Quality.TASHKEEL_ERROR):
            return Quality.MINOR
        return quality


def _group_count(clusters: Sequence[str], span: Span) -> int:
    """``countNormalizedGroups`` (``WS:197-217``)."""
    count, last = 0, None
    for cluster in clusters[span.skip : span.scored_end]:
        if cluster[0] in MADD_SCALARS:
            continue
        if _fold(cluster[0]) != last:
            count, last = count + 1, _fold(cluster[0])
    return count


def _decoded_item(word: Word, states: Mapping[int, State]) -> tuple[list[str], list[int]]:
    """The item's clusters with ``word`` decoded as ``states`` says, and where each of the
    word's groups starts in it (−1 for a gap)."""
    decoded = list(word.item[: word.item_offset])
    starts = []
    for g, (lo, hi) in enumerate(word.groups):
        state = states.get(g)
        if isinstance(state, Gap):
            starts.append(-1)
            continue
        starts.append(len(decoded))
        for cluster in word.clusters[lo:hi]:
            decoded.append(state.letter + cluster[1:] if isinstance(state, Swap) else cluster)
    decoded.extend(word.item[word.item_offset + len(word.clusters) :])
    return decoded, starts


def _realign_score(word: Word, states: Mapping[int, State], span: Span) -> float | None:
    """``localReAlignScore`` (``WS:783-852``) against the item decoded in isolation."""
    decoded, starts = _decoded_item(word, states)
    positions = [starts[g] for g in {word.group_of(c) for c in range(span.end)} if starts[g] >= 0]
    phonemes = [c for c in word.clusters[span.tanween : span.scored_end] if c[0] not in MADD_SCALARS]
    if not positions or not phonemes:
        return None
    padding = max(8, len(phonemes) * 3)
    window = decoded[max(0, min(positions) - padding) : min(len(decoded), max(positions) + padding + 1)]
    word_norm = normalize_phonemes("".join(phonemes)).normalized
    window_norm = normalize_phonemes("".join(window)).normalized
    if not word_norm or not window_norm:
        return None
    best = local_alignment_score(word_norm, window_norm)
    return min(1.0, best / len(word_norm)) if best > 0 else None


def displayed(quality: Quality, config: MurajaConfig) -> Quality:
    """What the reader sees of a kept grade (``GradeFilter+iOS.swift:58-87``)."""
    if quality == Quality.TASHKEEL_ERROR and not config.tashkeel_errors:
        return Quality.CORRECT
    if quality == Quality.MINOR and config.mask_minor:
        return Quality.CORRECT
    return quality


# --- cycles and the ratchet -------------------------------------------------------------------
@dataclass(frozen=True)
class Cycle:
    """One Muraja check as seen at an item's sites.

    ``outcomes``: every site's outcome under the check's decode. The check grades the words
    from ``first_word`` to ``end_word``, the word it ends on; ``end_word`` is ``None`` when it
    ends past the item, so every word from ``first_word`` on is graded and none is the end word.
    """

    outcomes: Mapping[str, SiteOutcome]
    end_word: int | None
    first_word: int = 0

    @classmethod
    def of(cls, sites: Sequence[TruthSite], decode: str, end_word: int | None, first_word: int = 0) -> Cycle:
        return cls(item_outcomes(sites, decode), end_word, first_word)

    def grades(self, word: Word) -> bool:
        return self.first_word <= word.index and (self.end_word is None or word.index <= self.end_word)


@dataclass(frozen=True)
class CycleGrade:
    """One cycle's grade of one word (``pending`` when the end word held it back), with what it
    was computed from."""

    cycle: int
    quality: Quality
    score: float
    end_word: bool
    evaluation: Evaluation
    edits: Mapping[str, SiteEdit]


def upgrades(kept: CycleGrade | None, new: CycleGrade) -> bool:
    """Whether ``GradeStore`` replaces ``kept`` with ``new`` (``GradeStore.swift:175-207``).
    A pending grade never replaces anything, and anything replaces nothing (``:306-332,
    377-394``)."""
    if new.quality == Quality.PENDING:
        return False
    if kept is None or new.quality.rank > kept.quality.rank:
        return True
    return new.quality == kept.quality and new.score > kept.score + SCORE_UPGRADE_MARGIN


def ratchet(grades: Iterable[CycleGrade]) -> CycleGrade | None:
    """The grade ``GradeStore`` keeps from a word's grades in cycle order; ``None`` while every
    grade was pending."""
    kept = None
    for grade in grades:
        if upgrades(kept, grade):
            kept = grade
    return kept


@dataclass(frozen=True)
class WordResult:
    """A word's final grade as the reader sees it (``pending`` if no grade was kept)."""

    word: int
    quality: Quality
    kept: CycleGrade | None


@dataclass(frozen=True)
class ItemGrades:
    words: dict[int, WordResult]
    sites: dict[str, str]


def _site_grade(site: TruthSite, kept: CycleGrade | None, config: MurajaConfig) -> str:
    """What the kept grade means for ``site`` (module docstring)."""
    if kept is None:
        return NOT_GRADED
    edit, loc, evaluation = kept.edits[site.site_id], locate(site), kept.evaluation
    if family(site.mark) == TASHKEEL:
        exempt, groups = evaluation.tashkeel_exempt, {loc.group}
    elif any(isinstance(state, Gap) for state in edit.states.values()):
        # a gap is never final-at-waqf credit (WS:628-637)
        exempt, groups = evaluation.unscored, set(edit.states)
    else:
        exempt, groups = evaluation.swap_exempt, set(edit.states) or {loc.group}
    if not edit.observable or groups <= exempt:
        return NOT_GRADED
    if not edit.deviates:
        return CORRECT
    if displayed(kept.quality, config) == Quality.CORRECT:
        return NOT_GRADED  # an allowance let the deviation pass
    if set(edit.states) & kept.evaluation.counted:
        return WRONG
    others = {g: s for other_id, other in kept.edits.items() if other_id != site.site_id for g, s in other.states.items()}
    without = evaluate(loc.word, others, kept.end_word, config)
    if without is not None and displayed(without.quality, config) == Quality.CORRECT:
        return WRONG  # the deviation's score cost alone took the word below correct
    return NOT_GRADED


def grade_item(sites: Sequence[TruthSite], cycles: Sequence[Cycle], config: MurajaConfig = TODAY) -> ItemGrades:
    """Each word that holds one of ``sites`` (one item) graded in every cycle that reaches it,
    ratcheted and shown, and what that means for each site (module docstring)."""
    by_word: dict[int, list[TruthSite]] = {}
    for site in sites:
        by_word.setdefault(locate(site).word.index, []).append(site)
    words: dict[int, WordResult] = {}
    grades: dict[str, str] = {}
    for index, word_sites in sorted(by_word.items()):
        word = locate(word_sites[0]).word
        cycle_grades = []
        for number, cycle in enumerate(cycles):
            if not cycle.grades(word):
                continue
            edits = {s.site_id: site_edit(s, cycle.outcomes[s.site_id], config) for s in word_sites}
            states = {g: state for edit in edits.values() for g, state in edit.states.items()}
            is_end = cycle.end_word == word.index
            evaluation = evaluate(word, states, is_end, config)
            if evaluation is None:
                continue
            quality = evaluation.quality
            if is_end and quality != Quality.CORRECT:  # WS:753-766
                quality = Quality.PENDING
            cycle_grades.append(CycleGrade(number, quality, evaluation.score, is_end, evaluation, edits))
        kept = ratchet(cycle_grades)
        words[index] = WordResult(index, displayed(kept.quality, config) if kept else Quality.PENDING, kept)
        for site in word_sites:
            grades[site.site_id] = _site_grade(site, kept, config)
    return ItemGrades(words, grades)


# --- where the cycles come from ---------------------------------------------------------------
RUN_ENDS = "single_decode:run_ends"
EVERY_WORD_ENDS = "single_decode:every_word_ends"
#: Each single-decode approximation (module docstring) and the sentence a report states for it.
APPROXIMATIONS = {
    RUN_ENDS: (
        "One decode per item stands for every Muraja cycle. Each word that ends a recited run "
        "(the item's last word, or a word followed by a heard pause) is the end word of one "
        "cycle, and a last cycle ends past the item. The ratchet never sees a different decode: "
        "false flags a cleaner cycle would clear stay, and mistakes a clean-decoding cycle would "
        "hide count as caught. Mushaf waqf signs and the last-word-of-ayah boost are not modelled."
    ),
    EVERY_WORD_ENDS: (
        "As run_ends, but every word is the end word of one cycle, as when a check ends on each "
        "word while the reciter passes it, so every word-final consonant gets the waqf "
        "exemption. A bound on what run_ends leaves out, not a measurement."
    ),
}


def single_decode_cycles(
    sites: Sequence[TruthSite], outcomes: Mapping[str, SiteOutcome], approximation: str = RUN_ENDS
) -> tuple[Cycle, ...]:
    """The cycles one decode of an item stands for (its sites' ``outcomes``)."""
    if approximation == RUN_ENDS:
        ends = {locate(s).word.index for s in sites if ends_run(s)}
    elif approximation == EVERY_WORD_ENDS:
        ends = {locate(s).word.index for s in sites}
    else:
        raise ValueError(f"unknown approximation {approximation!r}")
    return (*(Cycle(outcomes, end) for end in sorted(ends)), Cycle(outcomes, None))


#: :func:`decode_span`'s answer for a decode that aligns with nothing: no word is graded.
NO_SPAN = (0, -1)


def decode_span(reference: str, decode: str) -> tuple[int, int | None]:
    """The first word a check's decode reaches and its end word (``QuranFollowAlong.swift:
    385-413``): the word holding the alignment's end, or the one before when the alignment
    covers less than 30% of it. Aligned on haraka-free groups (``tadabur.smith_waterman``);
    the stricter rule for a short previous word (``:403-410``) and the jump cap (``:436-443``)
    need the engine's position and are not replayed."""
    ref = normalize_phonemes(reference)
    alignment = smith_waterman(normalize_phonemes(decode).normalized, ref.normalized)
    if alignment.ref_end <= alignment.ref_start:
        return NO_SPAN
    words = item_words(reference)

    def word_at(cluster: int) -> Word:
        return next(w for w in reversed(words) if w.item_offset <= cluster)

    first = word_at(ref.offset_map[alignment.ref_start][0]).index
    end_cluster = ref.offset_map[alignment.ref_end - 1][1]
    last = word_at(end_cluster - 1)
    covered = (end_cluster - last.item_offset) / len(last.clusters)
    if covered < 0.30 and last.index > 0:
        return first, last.index - 1
    return first, last.index


def grade_decode_sequence(
    sites: Sequence[TruthSite],
    decodes: Sequence[str],
    config: MurajaConfig = TODAY,
    reciter_moved_on: bool = True,
) -> ItemGrades:
    """Grade an item's sites from Muraja's checks in order.

    Each decode is one check's query (the last 20 confirmed characters, the pending hops and
    the current overlap or preview) as phoneme text over the part of the item it heard;
    :func:`decode_span` places it. With ``reciter_moved_on`` the recitation goes on past the
    item: the last check is graded once more with no end word among the item's words, as the
    checks after the item would grade it.
    """
    if not sites:
        return ItemGrades({}, {})
    cycles = []
    for decode in decodes:
        first, end = decode_span(sites[0].reference, decode)
        if (first, end) != NO_SPAN:
            cycles.append(Cycle.of(sites, decode, end, first))
    if reciter_moved_on and cycles:
        cycles.append(replace(cycles[-1], end_word=None))
    return grade_item(sites, cycles, config)


def grade(
    site: TruthSite, outcome: SiteOutcome, config: MurajaConfig = TODAY, approximation: str = RUN_ENDS
) -> str:
    """One site's grade from one decode of its item, under a single-decode approximation."""
    cycles = single_decode_cycles([site], {site.site_id: outcome}, approximation)
    return grade_item([site], cycles, config).sites[site.site_id]


# --- allowances ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Allowance:
    """One allowance: how to switch it off, and which sites it affects (decided from the truth
    record alone, never from a decode)."""

    name: str
    switch_off: Callable[[MurajaConfig], MurajaConfig]
    affects: Callable[[TruthSite], bool]


def _soft_pair(pair: str) -> Allowance:
    return Allowance(
        name=f"soft_pair {pair}",
        switch_off=lambda config: replace(config, soft_pairs=config.soft_pairs - {pair}),
        affects=lambda site: site.mark == pair,
    )


def _haraka_on(letters_in: bool) -> Callable[[TruthSite], bool]:
    def affects(site: TruthSite) -> bool:
        carrier = site.reference[site.reference_index]
        return site.prescribed in HARAKA_CHARS and (carrier in HARAKA_DROP_LETTERS) == letters_in

    return affects


def _waqf_final(site: TruthSite) -> bool:
    """The carrier is the final consonant of a word that ends a recited run."""
    if not ends_run(site):
        return False
    loc = locate(site)
    span = scoring_span(loc.word, end_word=True)
    return span.last_consonant <= loc.cluster < span.scored_end


def _is_pair(site: TruthSite) -> bool:
    return family(site.mark) not in (TASHKEEL, SHADDAH)


def _word_initial_geminate(site: TruthSite) -> bool:
    if site.mark != SHADDAH or site.prescribed != HELD:
        return False
    loc = locate(site)
    return loc.cluster < scoring_span(loc.word, end_word=False).skip


ALLOWANCES: tuple[Allowance, ...] = (
    *(_soft_pair(pair) for pair in sorted(SOFT_PAIRS)),
    Allowance(
        "shaddah_suppression",
        lambda config: replace(config, shaddah_suppression=False),
        lambda site: site.mark == SHADDAH and site.prescribed == HELD,
    ),
    Allowance(
        "haraka_drop_letters",
        lambda config: replace(config, haraka_drop_letters=frozenset()),
        _haraka_on(True),
    ),
    Allowance(
        "suppress_haraka_drop",
        lambda config: replace(config, suppress_haraka_drop=False),
        _haraka_on(False),
    ),
    Allowance(
        "final_at_waqf_tashkeel",
        lambda config: replace(config, final_at_waqf_tashkeel=False),
        lambda site: family(site.mark) == TASHKEEL and _waqf_final(site),
    ),
    Allowance(
        "final_at_waqf_consonant",
        lambda config: replace(config, final_at_waqf_consonant=False),
        lambda site: _is_pair(site) and _waqf_final(site),
    ),
    Allowance(
        "group_has_gap",
        lambda config: replace(config, group_has_gap=False),
        lambda site: family(site.mark) == TASHKEEL and len(locate(site).run) > 1,
    ),
    Allowance(
        "leading_assimilation",
        lambda config: replace(config, leading_assimilation=False),
        _word_initial_geminate,
    ),
)


def allowances_off(config: MurajaConfig = TODAY) -> MurajaConfig:
    """``config`` with every allowance switched off; its scores and display filters kept."""
    for allowance in ALLOWANCES:
        config = allowance.switch_off(config)
    return config

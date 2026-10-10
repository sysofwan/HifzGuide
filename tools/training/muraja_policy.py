"""Muraja's app outcome at truth sites, graded by Muraja's own Swift engine (ADR-0012).

The app outcome is never re-implemented here. ``tools/muraja_harness`` compiles the
FollowAlong sources of a pinned Muraja checkout, kept outside this repo, together with a small
driver of our own, and replays each item's checks through ``FollowAlongEngine``: query assembly,
placement, word scoring, and the ``GradeStore`` ratchet with its hold buffer and the end-word
holdback. This module holds what is ours:

* the **scorings** to replay (:class:`Scoring`): a Muraja mode, the two allowance flags of
  ``ScoringParameters`` (``softPairsEnabled``, ``shaddahSuppression``,
  ``FollowAlong/FollowAlongTypes.swift:66-71``) and the Tashkeel toggle;
* the **checks** each item's one decode stands for (:func:`single_decode_checks`, the
  single-decode approximation);
* where each truth site sits in Muraja's reference (:class:`MurajaText`), and what the grade the
  engine keeps means for it (:func:`site_outcome`).

Muraja paths are under ``ios/HifzGuide/`` of ``sysofwan/Muraja`` at :data:`MURAJA_COMMIT`.

Scorings and allowances (ADR-0012)
----------------------------------
ADR-0012 judges a model with an allowance switched off. Three are in scope, pooled as Muraja
pools them, each expressed through Muraja's own flags:

* **soft-pair forgiveness**: ``softPairsEnabled``, one switch for all six pairs
  (``FollowAlong/PhonemeSifat.swift:214-225``);
* **shaddah suppression**: ``shaddahSuppression`` (``QuranFollowAlong+WordScoring.swift:740``);
* **the Tashkeel toggle**: the display filter that shows a tashkeelError as correct when the
  reader turned tashkeel detection off (``FollowAlong/GradeFilter+iOS.swift:68-73``). That
  source is iOS-only, so :func:`displayed` applies it to the grade the engine keeps, together
  with ``.lenient``'s minor mask (``:80-85``).

The protected exemptions (a dropped haraka on و ا ء ي, the waqf-final consonant, the geminate-gap
tashkeel discard, the word-initial assimilation skip, ``.lenient``'s ``suppressHarakaDrop``) are
scoring rules, not switches, and stay on in every scoring. The two ``ScoringParameters`` flags
have no runtime setter (``ScoringParameters.forMode``, ``FollowAlongTypes.swift:141-147``, is a
fixed switch), so the harness build renames that one function and wraps it in Muraja's own
``ScoringParameters.with`` (``:83-107``); no scoring logic is edited.

:data:`ALLOWANCES_ON` is balanced with all three allowances on: tashkeel hidden, as readers who
turn the toggle off see it. The shipped default (:data:`SHIPPED_DEFAULT`: tashkeel shown,
``Data/AppSettings+iOS.swift:63``) is the same engine run with the Tashkeel allowance off.
:data:`SCORINGS` adds each allowance off alone and all three off.

The single-decode approximation
-------------------------------
Muraja checks every hop (the first second of a full 5 s window) and every preview of the pending
audio, run each time 200 ms of new audio has arrived once 2 s are pending
(``Models/RealtimeTranscriber+iOS.swift:566-660``); each check regrades the words it reaches.
HifzGuide decodes each item once per protocol, so :func:`single_decode_checks` replays that
cadence over the item's span with every check cut from the one decode, its characters spread
evenly over the span's duration (whole Swift Characters, so a haraka never leaves its
consonant), on exact sample counts. The session starts on the item's first word (the harness
moves the engine there after ``setPage``, as the app's explicit navigation does). The engine
then places, grades and ratchets as in the app; what the approximation cannot show is a check
decoding differently from the whole-item decode. Each item is one session: it ends with a
silence flush of the pending audio, including what follows the last whole 200 ms step, and the
session's end (``handleFinalFlush``, ``settleReadersWord``).

What a kept grade means for a site
----------------------------------
A site is placed on Muraja's reference for its ayah (:class:`MurajaText`) in two steps kept
apart: its **word** (from the carrier, or else from the carrier's immediate anchored
neighbours, never from whitespace; a pausal ه maps to the wasl ت it stands for), and its
**letter**:
the printed graphemes the app marks for the carrier's phoneme group. The app maps an error's
``WordError.groupIndex`` to printed graphemes through ``QuranDatabase.phonemeGroupCharIndices``
(``Data/QuranDatabase.swift:676-735``, which the harness runs and reports) and then scalar to
grapheme (``App/MistakeSnippetRenderer.swift:140-150, 350-363``, :func:`highlights`), and so
does this module: two groups that print on one letter mark the same letter, and a group whose
scalars fall past the printed text marks none. That mapping counts the ۥ/ۦ madd groups
the word scorer skips, so after one the app marks the letter after the erring one
(sysofwan/Muraja#260); the outcome reproduces what the app shows. A site's outcome
(:func:`site_outcome`) is

* :data:`FLAGGED`: the word shows as not correct and the app marks the site's letter;
* :data:`FLAGGED_ELSEWHERE`: the word shows as not correct; the app marks other letters only;
* :data:`FLAGGED_UNATTRIBUTED`: the word shows as not correct, and the site's carrier has no
  counterpart on Muraja's reference, so whether its letter is marked is unknown;
* :data:`NOT_FLAGGED`: the word shows as correct;
* :data:`NOT_GRADED`: the engine shows no grade for the word (never reached, or still pending
  at the session's end);
* :data:`UNPLACED`: no character of the site's word has a counterpart on Muraja's reference (a
  mapping failure, not an engine outcome).
"""

from __future__ import annotations

import difflib
import json
import os
import sqlite3
import subprocess
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from tadabur.truth_sites import HELD, SHADDAH, SOFT_PAIRS, TruthSite
from training.site_outcomes import TASHKEEL, family

MURAJA_COMMIT = "99c326f3fb7c6c53ff525568976c7e2442ccde3c"
#: Where ``tools/muraja_harness/build.sh`` puts the checkout, the binary and ``harness.json``.
HARNESS_ROOT = Path(os.environ.get("MURAJA_HARNESS_DIR", Path.home() / ".cache/hifzguide/muraja-harness"))
BUILD_SCRIPT = Path(__file__).parent.parent / "muraja_harness" / "build.sh"

FLAGGED = "flagged"
FLAGGED_ELSEWHERE = "flagged_elsewhere"
FLAGGED_UNATTRIBUTED = "flagged_unattributed"
NOT_FLAGGED = "not_flagged"
NOT_GRADED = "not_graded"
UNPLACED = "unplaced"
OUTCOMES = (FLAGGED, FLAGGED_ELSEWHERE, FLAGGED_UNATTRIBUTED, NOT_FLAGGED, NOT_GRADED, UNPLACED)
#: The outcomes in which the site's word shows as not correct.
WORD_FLAGGED = frozenset({FLAGGED, FLAGGED_ELSEWHERE, FLAGGED_UNATTRIBUTED})

CORRECT = "correct"
MINOR = "minor"
TASHKEEL_ERROR = "tashkeelError"
PENDING = "pending"


# --- scorings ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Scoring:
    """One Muraja configuration to replay; ``None`` keeps the mode's own flag."""

    name: str
    mode: str = "balanced"
    soft_pairs_enabled: bool | None = None
    shaddah_suppression: bool | None = None
    tashkeel_shown: bool = True

    @property
    def engine_name(self) -> str:
        """The engine run this scoring reads; the Tashkeel toggle is display only."""
        return f"{self.mode}|soft_pairs={self.soft_pairs_enabled}|shaddah={self.shaddah_suppression}"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "muraja_commit": MURAJA_COMMIT,
            "mode": self.mode,
            "soft_pairs_enabled": self.soft_pairs_enabled,
            "shaddah_suppression": self.shaddah_suppression,
            "tashkeel_shown": self.tashkeel_shown,
        }


#: The app as shipped: balanced, tashkeel detection on.
SHIPPED_DEFAULT = Scoring("shipped_default")
#: Balanced with ADR-0012's three allowances on: soft pairs, shaddah suppression, tashkeel hidden.
ALLOWANCES_ON = Scoring("allowances_on", tashkeel_shown=False)


@dataclass(frozen=True)
class Allowance:
    """One ADR-0012 allowance: how to switch it off, and the sites it affects (decided from the
    truth record alone)."""

    name: str
    switch_off: Callable[[Scoring], Scoring]
    affects: Callable[[TruthSite], bool]


def _in_geminate(site: TruthSite) -> bool:
    index, letter = site.reference_index, site.reference[site.reference_index]
    return letter in (site.reference[index - 1 : index], site.reference[index + 1 : index + 2])


ALLOWANCES: tuple[Allowance, ...] = (
    Allowance(
        "soft_pair_forgiveness",
        lambda s: replace(s, soft_pairs_enabled=False),
        lambda site: site.mark in SOFT_PAIRS,
    ),
    Allowance(
        "shaddah_suppression",
        lambda s: replace(s, shaddah_suppression=False),
        # held geminates, and the tashkeel an un-collapsed geminate exposes (ADR-0012)
        lambda site: (site.mark == SHADDAH and site.prescribed == HELD)
        or (family(site.mark) == TASHKEEL and _in_geminate(site)),
    ),
    Allowance(
        "tashkeel_toggle",
        lambda s: replace(s, tashkeel_shown=True),
        lambda site: family(site.mark) == TASHKEEL,
    ),
)


def all_off(scoring: Scoring) -> Scoring:
    for allowance in ALLOWANCES:
        scoring = allowance.switch_off(scoring)
    return scoring


#: Every scoring a report replays: the baseline, each allowance off alone (the Tashkeel one is
#: the shipped default), and all three off.
SCORINGS: tuple[Scoring, ...] = (
    ALLOWANCES_ON,
    replace(ALLOWANCES[0].switch_off(ALLOWANCES_ON), name="soft_pair_forgiveness_off"),
    replace(ALLOWANCES[1].switch_off(ALLOWANCES_ON), name="shaddah_suppression_off"),
    SHIPPED_DEFAULT,
    replace(all_off(ALLOWANCES_ON), name="allowances_off"),
)


def allowance_off(allowance: Allowance) -> Scoring:
    """The scoring of :data:`SCORINGS` that is :data:`ALLOWANCES_ON` with ``allowance`` off."""
    off = allowance.switch_off(ALLOWANCES_ON)
    return next(s for s in SCORINGS if replace(s, name=off.name) == off)


def displayed(quality: str, scoring: Scoring) -> str:
    """The kept quality as the reader sees it (``GradeFilter+iOS.swift:68-73, 80-85``)."""
    if quality == TASHKEEL_ERROR and not scoring.tashkeel_shown:
        return CORRECT
    if quality == MINOR and scoring.mode == "lenient":
        return CORRECT
    return quality


# --- the checks one decode stands for --------------------------------------------------------------
SAMPLE_RATE = 16_000
#: The cadence in samples (``RealtimeTranscriber+iOS.swift:169-188, 238-240, 566-569``): a 5 s
#: window, a 1 s hop, no inference below 2 s pending, a preview per 3,200 new samples (200 ms).
WINDOW_SAMPLES, HOP_SAMPLES, MIN_PENDING_SAMPLES, PREVIEW_SAMPLES = 80_000, 16_000, 32_000, 3_200


#: The single-decode approximation (module docstring) and the statement a report carries.
SINGLE_DECODE = "single_decode_on_app_cadence"
APPROXIMATIONS = {
    SINGLE_DECODE: (
        "Each item has one decode per arm, not one per Muraja check. The checks Muraja would run "
        "over the item's audio (a hop per second once 5 s are pending, a preview per 200 ms once "
        "2 s are) are replayed through Muraja's engine with each check's text cut from that one "
        "decode, its characters spread evenly over the item's samples; the audio after the last "
        "whole 200 ms step goes to the closing flush. The session starts on the item's first "
        "word. Placement, grading, the "
        "ratchet, the hold buffer and the end-word holdback are Muraja's own; a check that would "
        "decode differently from the whole-item decode is not represented. Each item is one "
        "session, ended by a silence flush."
    ),
}


def clusters(text: str) -> list[str]:
    """Swift Characters: a base scalar and the combining marks after it."""
    out: list[str] = []
    for char in text:
        if out and unicodedata.combining(char):
            out[-1] += char
        else:
            out.append(char)
    return out


def single_decode_checks(decode: str, n_samples: int) -> list[dict]:
    """The checks Muraja would run over an item of ``n_samples``, each cut from one decode.

    The audio arrives in steps of 3,200 samples (200 ms), counted in samples so nothing is
    rounded: with a full window pending, its hop (the first second) is confirmed with the rest
    as overlap and the window advances a second, after which a preview runs at once; otherwise,
    with at least 2 s pending, a preview of all pending audio runs. The audio after the last
    whole step and everything still pending go to the closing silence flush. Updates with
    nothing in them are not sent, as the transcriber sends none. Every window start stays on the
    200 ms grid (the window and hop are whole numbers of steps), so this is the transcriber's
    schedule up to its 10 ms polling.
    """
    chars = clusters(decode)
    per_sample = len(chars) / n_samples if n_samples else 0.0

    def text(start: int, end: int) -> str:
        return "".join(chars[round(start * per_sample) : round(end * per_sample)])

    checks: list[dict] = []

    def send(hop: str, overlap: str, flush: bool) -> None:
        if hop or overlap or flush:
            checks.append({"hop": hop, "overlap": overlap, "flush": flush})

    start = 0
    for now in range(PREVIEW_SAMPLES, n_samples + 1, PREVIEW_SAMPLES):
        if now - start >= WINDOW_SAMPLES:
            send(text(start, start + HOP_SAMPLES), text(start + HOP_SAMPLES, start + WINDOW_SAMPLES), False)
            start += HOP_SAMPLES
        if now - start >= MIN_PENDING_SAMPLES:
            send("", text(start, now), False)
    send(text(start, n_samples), "", True)
    return checks


# --- truth sites on Muraja's reference --------------------------------------------------------------
@dataclass(frozen=True)
class MurajaSite:
    """A site on Muraja's reference: the phoneme word (1-based) its carrier sits in, and the
    ordinal of the carrier's phoneme group in that word, ``None`` when the carrier itself has no
    counterpart on Muraja's reference (the word is still known from its neighbours)."""

    surah: int
    ayah: int
    word: int
    group: int | None


def highlights(letters: dict) -> tuple[frozenset[int], ...]:
    """Per phoneme group of a word, the printed graphemes the app marks for an error there: the
    harness's ``phonemeGroupCharIndices`` (scalar indices of the printed word) taken to grapheme
    indices as ``toGraphemeIndices`` does (``App/MistakeSnippetRenderer.swift:350-363``), an
    index past the printed text marking nothing."""
    grapheme_of = graphemes(letters["text"])
    return tuple(
        frozenset(grapheme_of[i] for i in group if 0 <= i < len(grapheme_of))
        for group in letters["group_chars"]
    )


def _ayah(site: TruthSite) -> tuple[int, int]:
    surah, ayah = site.surah_ayah.split(":")
    return int(surah), int(ayah)


def graphemes(text: str) -> list[int]:
    """The grapheme index of each code point of ``text``: a base and the marks that extend it
    (Swift's ``Character``, for the Arabic script's Mn/Me marks)."""
    index, out = -1, []
    for char in text:
        if index < 0 or unicodedata.category(char) not in ("Mn", "Me"):
            index += 1
        out.append(index)
    return out


#: The shortest run of identical characters a site's word may be placed by when its carrier
#: has no counterpart: shorter runs match by coincidence.
ANCHOR_RUN = 3


@lru_cache(maxsize=1024)
def _alignment(text: str, reference: str) -> tuple[dict[int, int], frozenset[int]]:
    """Every index of ``text`` with a counterpart in ``reference`` (spaces ignored on both
    sides), and those inside a run of at least :data:`ANCHOR_RUN` identical characters."""
    a = [i for i, c in enumerate(text) if c != " "]
    b = [i for i, c in enumerate(reference) if c != " "]
    left, right = [text[i] for i in a], [reference[i] for i in b]
    mapping: dict[int, int] = {}
    anchored: set[int] = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[a[i1 + k]] = b[j1 + k]
                if i2 - i1 >= ANCHOR_RUN:
                    anchored.add(a[i1 + k])
        elif tag == "replace" and _pausal_taa(left[i1:i2], right[j1:j2]):
            mapping[a[i1]] = b[j1]
    return mapping, frozenset(anchored)


def counterpart(text: str, index: int, reference: str) -> int | None:
    """The index in ``reference`` that ``text[index]`` aligns to, spaces ignored on both sides.

    Matched characters map one to one. The one spelling difference mapped across is a taa
    marbuta at a pause: a pausal ``ه`` against the wasl ``ت`` group Muraja's reference holds
    (``تَںںں``, ``تِن``, ``تِوو``) maps to that ``ت``. Anything else spelled differently maps to
    nothing (``None``)."""
    return _alignment(text, reference)[0].get(index)


def _pausal_taa(realized: Sequence[str], reference: Sequence[str]) -> bool:
    """A pausal ``ه`` (with its marks) against a ``ت`` with its marks and at most a run of one
    repeated letter, the tanween's spelling."""
    def bare(chars: Sequence[str]) -> list[str]:
        return [c for c in chars if not unicodedata.combining(c)]

    left, right = bare(realized), bare(reference)
    return left == ["ه"] and right[:1] == ["ت"] and len(set(right[1:])) <= 1


class MurajaText:
    """Muraja's reference phonemes and phoneme groups, read from the harness's ``quran.db``."""

    def __init__(self, quran_db: Path):
        self._db = sqlite3.connect(f"file:{quran_db}?mode=ro", uri=True)

    def phonemes(self, surah: int, ayah: int) -> str:
        (text,) = self._db.execute(
            "SELECT phonemes FROM ayahs WHERE surah=? AND ayah=?", (surah, ayah)
        ).fetchone()
        return text

    def _text_word(self, surah: int, ayah: int, word: int) -> int:
        """The printed word a phoneme word is drawn on (``MistakeSnippetRenderer.swift:142``)."""
        row = self._db.execute(
            "SELECT MIN(text_word) FROM word_map WHERE surah=? AND ayah=? AND phoneme_word=?",
            (surah, ayah, word),
        ).fetchone()
        return row[0] if row and row[0] is not None else word

    def word_groups(self, surah: int, ayah: int, word: int) -> list[tuple[int, int]]:
        """A phoneme word's groups as code-point spans of the ayah's phonemes, through the first
        printed word the phoneme word is drawn on."""
        return list(
            self._db.execute(
                "SELECT ph_start, ph_end FROM phoneme_groups WHERE surah=? AND ayah=? AND uthmani_word=? "
                "ORDER BY ph_start",
                (surah, ayah, self._text_word(surah, ayah, word)),
            )
        )

    def locate(self, site: TruthSite) -> MurajaSite | None:
        """The site on Muraja's reference: its word from the carrier, or, when the carrier has
        no counterpart, from its anchored neighbours (:meth:`_word_of`); ``None`` when neither
        places it."""
        surah, ayah = _ayah(site)
        reference = self.phonemes(surah, ayah)
        position = counterpart(site.reference, site.reference_index, reference)
        if position is not None:
            word = reference[:position].count(" ") + 1
        else:
            word = self._word_of(site, reference)
            if word is None:
                return None
        group = None
        if position is not None:
            groups = self.word_groups(surah, ayah, word)
            group = next((g for g, (lo, hi) in enumerate(groups) if lo <= position < hi), None)
        return MurajaSite(surah, ayah, word, group)

    @staticmethod
    def _word_of(site: TruthSite, reference: str) -> int | None:
        """The Muraja word an unmatched carrier belongs to, read off its immediate neighbours and
        never off whitespace, which the phonetizer drops where it runs two words together (#129).

        The character just before the carrier and the first one after its own marks (spaces
        skipped) are its neighbours. A neighbour counts only when it is anchored (in a matched
        run of at least :data:`ANCHOR_RUN`). The carrier takes the word of its anchored
        neighbours when they agree; with none, or two that disagree, it is not placed (``None``):
        a word recited twice, or one whose letters around the carrier match nothing, never
        borrows a neighbouring word's grade."""
        text, index = site.reference, site.reference_index
        mapping, anchored = _alignment(text, reference)
        before = index - 1
        while before >= 0 and text[before] == " ":
            before -= 1
        after = index + 1
        while after < len(text) and (text[after] == " " or unicodedata.combining(text[after])):
            after += 1
        words = {reference[: mapping[i]].count(" ") + 1 for i in (before, after) if i in anchored}
        if len(words) != 1:
            return None
        (word,) = words
        return word

    def start_word(self, site: TruthSite) -> int:
        """The phoneme word the site's item starts on (word 1 when its start has no counterpart)."""
        surah, ayah = _ayah(site)
        reference = self.phonemes(surah, ayah)
        first = next(i for i, c in enumerate(site.reference) if c != " ")
        position = counterpart(site.reference, first, reference)
        return 1 if position is None else reference[:position].count(" ") + 1


# --- the harness -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Harness:
    """A built harness (``tools/muraja_harness/build.sh``), as its ``harness.json`` records it."""

    binary: Path
    quran_db: Path
    layout_db: Path
    muraja_commit: str
    compiler: str

    @classmethod
    def locate(cls, root: Path = HARNESS_ROOT) -> Harness | None:
        """The harness built for :data:`MURAJA_COMMIT`, or ``None`` when there is none."""
        manifest = root / MURAJA_COMMIT / "harness.json"
        if not manifest.exists():
            return None
        record = json.loads(manifest.read_text(encoding="utf-8"))
        harness = cls(
            Path(record["binary"]), Path(record["quran_db"]), Path(record["layout_db"]),
            record["muraja_commit"], record["compiler"],
        )
        present = all(p.exists() for p in (harness.binary, harness.quran_db, harness.layout_db))
        return harness if present and harness.muraja_commit == MURAJA_COMMIT else None

    def run(self, requests: Iterable[dict]) -> tuple[dict, list[dict]]:
        """The build record the binary prints, and one result per request and scoring."""
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests)
        completed = subprocess.run(
            [str(self.binary), str(self.quran_db)],
            input=payload, capture_output=True, text=True, check=True,
            env={**os.environ, "QURAN_LAYOUT_DB_PATH": str(self.layout_db)},
        )
        build, *results = (json.loads(line) for line in completed.stdout.splitlines() if line)
        if build["muraja_commit"] != MURAJA_COMMIT:
            raise ValueError(f"the harness was built from {build['muraja_commit']}, not {MURAJA_COMMIT}")
        return build, results


def require_harness() -> Harness:
    harness = Harness.locate()
    if harness is None:
        raise SystemExit(f"no Muraja harness under {HARNESS_ROOT}; build it with `bash {BUILD_SCRIPT}`")
    return harness


# --- items in, site outcomes out -------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    """One decoded item: its sites (one realized reference in one ayah), a decode, its length."""

    key: str
    sites: tuple[TruthSite, ...]
    decode: str
    n_samples: int


@dataclass(frozen=True)
class SiteGrade:
    """A site's outcome under one scoring, and the kept grade it was read from."""

    outcome: str
    quality: str | None = None
    errors: tuple = field(default=(), compare=False)


def site_outcome(
    where: MurajaSite | None, kept: dict | None, marks: Sequence[frozenset[int]], scoring: Scoring
) -> SiteGrade:
    """What the kept grade of the site's word means for the site (module docstring)."""
    if where is None:
        return SiteGrade(UNPLACED)
    if kept is None or kept["quality"] == PENDING:
        return SiteGrade(NOT_GRADED, kept["quality"] if kept else None)
    shown = displayed(kept["quality"], scoring)
    errors = tuple(kept["errors"])
    if shown == CORRECT:
        return SiteGrade(NOT_FLAGGED, shown, errors)
    if where.group is None or where.group >= len(marks):
        return SiteGrade(FLAGGED_UNATTRIBUTED, shown, errors)
    marked = set().union(*(marks[e["group_index"]] for e in errors if e["group_index"] < len(marks)))
    return SiteGrade(FLAGGED if marked & marks[where.group] else FLAGGED_ELSEWHERE, shown, errors)


def build_request(item: Item, text: MurajaText, scorings: Sequence[Scoring]) -> tuple[dict, dict]:
    """The harness request for an item (one engine run per distinct engine configuration), and
    where each of its sites sits on Muraja's reference."""
    where = {site.site_id: text.locate(site) for site in item.sites}
    surah, ayah = _ayah(item.sites[0])
    engines = {s.engine_name: s for s in scorings}
    return {
        "item": item.key,
        "surah": surah,
        "ayah": ayah,
        "start_word": text.start_word(item.sites[0]),
        "report_words": sorted({w.word for w in where.values() if w is not None}),
        "checks": single_decode_checks(item.decode, item.n_samples),
        "scorings": [
            {
                "name": name,
                "mode": s.mode,
                "soft_pairs_enabled": s.soft_pairs_enabled,
                "shaddah_suppression": s.shaddah_suppression,
            }
            for name, s in sorted(engines.items())
        ],
    }, where


@dataclass(frozen=True)
class AppOutcomes:
    """Every site's outcome per scoring, and the harness build that graded them."""

    #: ``[scoring name][(item key, site id)]``
    grades: dict[str, dict[tuple[str, str], SiteGrade]]
    build: dict
    #: Site ids whose word has no counterpart on Muraja's reference (always ``unplaced``).
    unplaced: tuple[str, ...]
    #: Site ids whose word is placed but whose carrier has no counterpart.
    unattributed: tuple[str, ...]


def read_results(
    items: Sequence[Item], where: dict, results: Iterable[dict], scorings: Sequence[Scoring], build: dict
) -> AppOutcomes:
    """Parse the harness's results into every site's outcome under every scoring."""
    results = list(results)
    kept = {(r["item"], r["scoring"]): {s["word"]: s for s in r["final"]} for r in results}
    marks = {(r["item"], w["word"]): highlights(w) for r in results for w in r["letters"]}
    grades: dict[str, dict[tuple[str, str], SiteGrade]] = {s.name: {} for s in scorings}
    for item in items:
        for site in item.sites:
            loc = where[item.key][site.site_id]
            word_marks = marks.get((item.key, loc.word), ()) if loc else ()
            for scoring in scorings:
                status = kept[(item.key, scoring.engine_name)].get(loc.word) if loc else None
                grades[scoring.name][(item.key, site.site_id)] = site_outcome(loc, status, word_marks, scoring)
    placed = [(s.site_id, where[i.key][s.site_id]) for i in items for s in i.sites]
    unplaced = sorted({site_id for site_id, loc in placed if loc is None})
    unattributed = sorted({site_id for site_id, loc in placed if loc is not None and loc.group is None})
    return AppOutcomes(grades, build, tuple(unplaced), tuple(unattributed))


def grade_items(items: Sequence[Item], scorings: Sequence[Scoring], harness: Harness) -> AppOutcomes:
    """Replay every item under every scoring through the harness."""
    text = MurajaText(harness.quran_db)
    requests, where = [], {}
    for item in items:
        req, where[item.key] = build_request(item, text, scorings)
        requests.append(req)
    build, results = harness.run(requests)
    return read_results(items, where, results, scorings, build)

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
consonant). The engine then places, grades and ratchets as in the app; what the approximation
cannot show is a check decoding differently from the whole-item decode. Each item is one
session: it ends with a silence flush of the pending audio and the session's end
(``handleFinalFlush``, ``settleReadersWord``).

What a kept grade means for a site
----------------------------------
A site is mapped onto Muraja's reference for its ayah (:class:`MurajaText`): its phoneme word,
and the phoneme group of that word the app marks for an error there. The app maps
``WordError.groupIndex`` to letters through ``QuranDatabase.phonemeGroupCharIndices``
(``App/MistakeSnippetRenderer.swift:140-150``, ``Data/QuranDatabase.swift:676-716``), and so
does this module. A site's outcome (:func:`site_outcome`) is

* :data:`FLAGGED`: the word shows as not correct and the kept grade has an error on the site's
  group (the app marks the site's letter);
* :data:`FLAGGED_ELSEWHERE`: the word shows as not correct, with no error on the site's group;
* :data:`NOT_FLAGGED`: the word shows as correct;
* :data:`NOT_GRADED`: no grade shows for the word (never reached, or still pending at the
  session's end), or the site has no counterpart on Muraja's reference.
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
from pathlib import Path

from tadabur.truth_sites import HELD, SHADDAH, SOFT_PAIRS, TruthSite
from training.site_outcomes import TASHKEEL, family

MURAJA_COMMIT = "99c326f3fb7c6c53ff525568976c7e2442ccde3c"
#: Where ``tools/muraja_harness/build.sh`` puts the checkout, the binary and ``harness.json``.
HARNESS_ROOT = Path(os.environ.get("MURAJA_HARNESS_DIR", Path.home() / ".cache/hifzguide/muraja-harness"))
BUILD_SCRIPT = Path(__file__).parent.parent / "muraja_harness" / "build.sh"

FLAGGED = "flagged"
FLAGGED_ELSEWHERE = "flagged_elsewhere"
NOT_FLAGGED = "not_flagged"
NOT_GRADED = "not_graded"
OUTCOMES = (FLAGGED, FLAGGED_ELSEWHERE, NOT_FLAGGED, NOT_GRADED)

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
#: The cadence, in 200 ms ticks (``RealtimeTranscriber+iOS.swift:169-188, 238-240, 566-569``):
#: a 5 s window, a 1 s hop, no inference below 2 s pending, a preview per 200 ms of new audio.
TICK_S = 0.2
WINDOW_TICKS, HOP_TICKS, MIN_PENDING_TICKS = 25, 5, 10


#: The single-decode approximation (module docstring) and the statement a report carries.
SINGLE_DECODE = "single_decode_on_app_cadence"
APPROXIMATIONS = {
    SINGLE_DECODE: (
        "Each item has one decode per arm, not one per Muraja check. The checks Muraja would run "
        "over the item's audio (a hop per second once 5 s are pending, a preview per 200 ms once "
        "2 s are) are replayed through Muraja's engine with each check's text cut from that one "
        "decode, its characters spread evenly over the item's duration. Placement, grading, the "
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

    One tick per 200 ms of audio. With a full window pending, its hop (the first second) is
    confirmed with the rest as overlap and the window advances a second, after which a preview
    runs at once; otherwise, with at least 2 s pending, a preview of all pending audio runs. At
    the end a silence flush confirms what is pending. Updates with nothing in them are not sent,
    as the transcriber sends none.
    """
    chars = clusters(decode)
    total = round(n_samples / SAMPLE_RATE / TICK_S)
    per_tick = len(chars) / total if total else 0.0

    def text(start: int, end: int) -> str:
        return "".join(chars[round(start * per_tick) : round(end * per_tick)])

    checks: list[dict] = []

    def send(hop: str, overlap: str, flush: bool) -> None:
        if hop or overlap or flush:
            checks.append({"hop": hop, "overlap": overlap, "flush": flush})

    start = 0
    for now in range(1, total + 1):
        if now - start >= WINDOW_TICKS:
            send(text(start, start + HOP_TICKS), text(start + HOP_TICKS, start + WINDOW_TICKS), False)
            start += HOP_TICKS
        if now - start >= MIN_PENDING_TICKS:
            send("", text(start, now), False)
    send(text(start, total), "", True)
    return checks


# --- truth sites on Muraja's reference --------------------------------------------------------------
@dataclass(frozen=True)
class MurajaSite:
    """A site on Muraja's reference: its ayah, phoneme word (1-based), and the ordinal of the
    phoneme group the app marks for an error there."""

    surah: int
    ayah: int
    word: int
    group: int


def _ayah(site: TruthSite) -> tuple[int, int]:
    surah, ayah = site.surah_ayah.split(":")
    return int(surah), int(ayah)


def counterpart(text: str, index: int, reference: str) -> int | None:
    """The index in ``reference`` that ``text[index]`` aligns to, spaces ignored on both sides;
    ``None`` when it aligns to nothing."""
    a = [i for i, c in enumerate(text) if c != " "]
    b = [i for i, c in enumerate(reference) if c != " "]
    matcher = difflib.SequenceMatcher(None, [text[i] for i in a], [reference[i] for i in b], autojunk=False)
    target = a.index(index)
    for block in matcher.get_matching_blocks():
        if block.a <= target < block.a + block.size:
            return b[block.b + target - block.a]
    return None


class MurajaText:
    """Muraja's reference phonemes and phoneme groups, read from the harness's ``quran.db``."""

    def __init__(self, quran_db: Path):
        self._db = sqlite3.connect(f"file:{quran_db}?mode=ro", uri=True)

    def phonemes(self, surah: int, ayah: int) -> str:
        (text,) = self._db.execute(
            "SELECT phonemes FROM ayahs WHERE surah=? AND ayah=?", (surah, ayah)
        ).fetchone()
        return text

    def word_groups(self, surah: int, ayah: int, word: int) -> list[tuple[int, int]]:
        """A phoneme word's groups as code-point spans of the ayah's phonemes, read as the app
        reads them: through the first text word the phoneme word maps to."""
        row = self._db.execute(
            "SELECT MIN(text_word) FROM word_map WHERE surah=? AND ayah=? AND phoneme_word=?",
            (surah, ayah, word),
        ).fetchone()
        text_word = row[0] if row and row[0] is not None else word
        return list(
            self._db.execute(
                "SELECT ph_start, ph_end FROM phoneme_groups WHERE surah=? AND ayah=? AND uthmani_word=? "
                "ORDER BY ph_start",
                (surah, ayah, text_word),
            )
        )

    def locate(self, site: TruthSite) -> MurajaSite | None:
        """The site's carrier on Muraja's reference; ``None`` when it has no counterpart there."""
        surah, ayah = _ayah(site)
        reference = self.phonemes(surah, ayah)
        position = counterpart(site.reference, site.reference_index, reference)
        if position is None:
            return None
        word = reference[:position].count(" ") + 1
        groups = self.word_groups(surah, ayah, word)
        ordinal = next((g for g, (lo, hi) in enumerate(groups) if lo <= position < hi), None)
        return None if ordinal is None else MurajaSite(surah, ayah, word, ordinal)

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


def site_outcome(where: MurajaSite | None, kept: dict | None, scoring: Scoring) -> SiteGrade:
    """What the kept grade of the site's word means for the site (module docstring)."""
    if where is None or kept is None or kept["quality"] == PENDING:
        return SiteGrade(NOT_GRADED, kept["quality"] if kept else None)
    shown = displayed(kept["quality"], scoring)
    errors = tuple(kept["errors"])
    if shown == CORRECT:
        return SiteGrade(NOT_FLAGGED, shown, errors)
    on_site = any(error["group_index"] == where.group for error in errors)
    return SiteGrade(FLAGGED if on_site else FLAGGED_ELSEWHERE, shown, errors)


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
    #: Site ids with no counterpart on Muraja's reference (always ``not_graded``).
    unmapped: tuple[str, ...]


def read_results(
    items: Sequence[Item], where: dict, results: Iterable[dict], scorings: Sequence[Scoring], build: dict
) -> AppOutcomes:
    """Parse the harness's results into every site's outcome under every scoring."""
    kept = {(r["item"], r["scoring"]): {s["word"]: s for s in r["final"]} for r in results}
    grades: dict[str, dict[tuple[str, str], SiteGrade]] = {s.name: {} for s in scorings}
    for item in items:
        for site in item.sites:
            loc = where[item.key][site.site_id]
            for scoring in scorings:
                status = kept[(item.key, scoring.engine_name)].get(loc.word) if loc else None
                grades[scoring.name][(item.key, site.site_id)] = site_outcome(loc, status, scoring)
    unmapped = sorted({s.site_id for i in items for s in i.sites if where[i.key][s.site_id] is None})
    return AppOutcomes(grades, build, tuple(unmapped))


def grade_items(items: Sequence[Item], scorings: Sequence[Scoring], harness: Harness) -> AppOutcomes:
    """Replay every item under every scoring through the harness."""
    text = MurajaText(harness.quran_db)
    requests, where = [], {}
    for item in items:
        req, where[item.key] = build_request(item, text, scorings)
        requests.append(req)
    build, results = harness.run(requests)
    return read_results(items, where, results, scorings, build)

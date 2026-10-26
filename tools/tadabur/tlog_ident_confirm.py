"""S5b (#204): the fresh confirmation draws, the five bars and the oracle's consequence.

Everything here is a pure function of committed inputs, the salt and the fixed sizes; the fetch
route (full archive or two-pass streaming) only supplies rows. The draws follow
``docs/tlog-ident-floor-preregistration.md``, whose SHA-256 the decode stores and the report cite,
and whose values are never selected on the confirmation data.

* **Freesound.** Rank the 100 shared shards of ``obadx/freesound-commercial-50k-noise-only`` by
  salted hash, take them until 1,000 recordings are eligible (at most :data:`MAX_SHARDS`), drop
  anything in either committed pool, then take the first 1,000 by salted rank of
  ``freesound_id``.
* **Common Voice.** One speaker draw per locale: pass 1 at most one clip per ``client_id``, pass 2
  at most two, else stop and report. A locale below 100 eligible clips at cap 2 fires its one
  recorded substitute, which must itself reach 100.
* **Quran-text guard.** Sentences are normalised with ``normalize_imlai`` (dropping the dagger
  alef, tatweel and non-letters) and excluded if any four consecutive words occur contiguously in
  a surah's normalised ``data/imlaei.json`` text.
* **Bars.** Five bars, each with a Wilson 95% interval where it is a rate; the first to fire names
  the status. A failing bar is reported, never retuned or redrawn.

Usage (from ``tools/``): the draw and decode entry points live beside the committed manifests;
this module holds the rules they share with the acceptance tests.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from generate_quran_db import normalize_imlai
from training.counterfactual_eval import wilson_interval

ROOT = Path(__file__).resolve().parents[2]
IDENT_DIR = Path(__file__).parent / "tlog" / "ident"

#: The one salt every S5b draw ranks with; it is written in the pre-registration.
SALT = "tlog-s5b-floor-confirm-v1"
FREESOUND_SIZE = 1000
CV_AR_SIZE = 500
CV_OTHER_PER_LOCALE = 100
PRIMARY_LOCALES = ("en", "ur", "fa", "tr", "id")
SUBSTITUTES = {"en": "es", "ur": "hi", "fa": "ckb", "tr": "sw", "id": "bn"}
#: Locales whose clips are written in Arabic script, so the Quran-text guard applies.
ARABIC_SCRIPT = frozenset({"ar", "ur", "fa", "ckb"})

#: The Freesound source and revision of the S5 pool (``tools/training/perturbation.py``).
FREESOUND_REPO = "obadx/freesound-commercial-50k-noise-only"
FREESOUND_REVISION = "a5f48ab83b64ba1d35ee4ec363dae75d4e2c69d3"
TOTAL_SHARDS = 104
#: Shards the two committed pools were drawn from, and the S5 decoding pool: never reused.
HELD_OUT_SHARDS = frozenset({0, 26, 52, 78})
MAX_SHARDS = 10

CV_RELEASE = "cv-corpus-27.0-2026-09-11"

CONFIRM_DIR = IDENT_DIR / "confirm"
FREESOUND_MANIFEST_PATH = CONFIRM_DIR / "freesound.json"
CV_MANIFEST_PATH = CONFIRM_DIR / "common_voice.jsonl"
CV_COUNTS_PATH = CONFIRM_DIR / "common_voice.json"
CONFIRM_DECODES = {
    "freesound": IDENT_DIR / "decodes" / "confirm_freesound.base.json",
    "cv_ar": IDENT_DIR / "decodes" / "confirm_cv_ar.base.json",
    "cv_other": IDENT_DIR / "decodes" / "confirm_cv_other.base.json",
}

#: The five bars: the three negative arms and look retention are rates; mistakes is a count.
PASS_LIMIT = 0.01
MISTAKE_CLIPS = 52
LOOK_RETENTION_MIN = 0.90

_SHARD_PATTERN = re.compile(r"train-(\d+)-of-\d+")


class ConfirmDrawError(ValueError):
    """A draw cannot reach its pre-registered size without changing the rule."""


def salted_rank(*parts: object) -> bytes:
    """The draw key: ``sha256("<salt>:<part>:<part>…")`` over the parts, as bytes."""
    payload = f"{SALT}:" + ":".join(map(str, parts))
    return hashlib.sha256(payload.encode("utf-8")).digest()


# --- Freesound: shards and recordings ----------------------------------------------------------


def shard_name(index: int) -> str:
    """The collection's shard name for ``index`` (104 equal shards)."""
    return f"data/train-{index:05d}-of-{TOTAL_SHARDS:05d}.parquet"


def shard_index(shard: str) -> int:
    """The shard's index in the collection's order."""
    match = _SHARD_PATTERN.search(shard)
    if match is None:
        raise ValueError(f"{shard!r} is not a shard of this collection")
    return int(match.group(1))


def candidate_shards() -> tuple[str, ...]:
    """The 100 shared shards: every shard except the four the pools already used."""
    return tuple(shard_name(index) for index in range(TOTAL_SHARDS)
                 if index not in HELD_OUT_SHARDS)


def shard_ranking(shards: Iterable[str]) -> tuple[str, ...]:
    """The shards in salted-rank order (the draw takes a prefix of this)."""
    return tuple(sorted(shards, key=lambda shard: salted_rank("shard", shard)))


def freesound_select(eligible: Sequence[Mapping[str, Any]],
                     *, count: int = FREESOUND_SIZE) -> list[Mapping[str, Any]]:
    """The first ``count`` eligible recordings by salted rank of ``freesound_id``."""
    ranked = sorted(eligible, key=lambda row: salted_rank("fs", row["freesound_id"]))
    return ranked[:count]


# --- Common Voice: the Quran-text guard --------------------------------------------------------


def _is_arabic_letter(char: str) -> bool:
    return unicodedata.name(char, "").startswith("ARABIC LETTER")


def guard_tokens(text: str) -> tuple[str, ...]:
    """The comparison tokens of one sentence: Imlaei-normalised Arabic letters only.

    The dagger alef is dropped before normalisation so modern spelling and the Quran's Uthmani
    forms meet on the same word (``الرحمن``), and tatweel, tashkeel, punctuation and every digit
    are removed. ``normalize_imlai`` unifies the alef, ya and ta-marbuta forms.
    """
    normalised = normalize_imlai(text.replace("\u0670", ""))
    tokens = []
    for chunk in normalised.split():
        token = "".join(char for char in chunk if _is_arabic_letter(char))
        if token:
            tokens.append(token)
    return tuple(tokens)


def quran_ngrams(imlaei: Mapping[str, Mapping[str, str]]) -> set[tuple[str, ...]]:
    """Every contiguous four-word run of each surah's normalised Imlaei text.

    Consecutive words within a surah, so a quote crossing an ayah boundary is still caught;
    verse-number entries are dropped.
    """
    by_surah: dict[str, list[tuple[int, int, tuple[str, ...]]]] = {}
    for entry in imlaei.values():
        text = entry["text"].strip()
        if not text or all(char.isdigit() for char in text):
            continue
        tokens = guard_tokens(text)
        if not tokens:
            continue
        by_surah.setdefault(entry["surah"], []).append(
            (int(entry["ayah"]), int(entry["word"]), tokens))
    ngrams: set[tuple[str, ...]] = set()
    for words in by_surah.values():
        words.sort()
        # An entry can hold more than one whitespace word; the token stream is the text.
        stream = [token for _, _, tokens in words for token in tokens]
        for start in range(len(stream) - 3):
            ngrams.add(tuple(stream[start:start + 4]))
    return ngrams


def fails_quran_guard(sentence: str, ngrams: Iterable[tuple[str, ...]]) -> bool:
    """Whether four consecutive words of ``sentence`` occur contiguously in the Quran text."""
    tokens = guard_tokens(sentence)
    for start in range(len(tokens) - 3):
        if tuple(tokens[start:start + 4]) in ngrams:
            return True
    return False


# --- Common Voice: the speaker draw and the substitute plan ------------------------------------


def cv_draw(clips: Sequence[Mapping[str, Any]], *, locale: str,
            count: int) -> dict[str, Any]:
    """``count`` clips of one locale under the two-pass speaker cap.

    Pass 1 walks the salted ranking and takes a clip when its speaker has none taken; pass 2
    walks the same ranking again over the clips not yet taken, at a cap of two. The draw stops at
    ``count``; if pass 2 still ends short, ``cap_reached`` is ``None`` and ``short`` says by how
    many. The result is independent of the input order, and the drawn list is rank-ordered.
    """
    ranked = sorted(clips, key=lambda row: salted_rank("cv", locale, row["path"]))
    per_speaker: Counter[str] = Counter()
    taken: list[Mapping[str, Any]] = []
    taken_paths: set[str] = set()
    for row in ranked:
        if len(taken) == count:
            break
        if per_speaker[row["client_id"]] < 1:
            taken.append(row)
            taken_paths.add(row["path"])
            per_speaker[row["client_id"]] += 1
    cap_reached = 1 if len(taken) == count else 2
    if len(taken) < count:
        for row in ranked:
            if len(taken) == count:
                break
            if row["path"] in taken_paths:
                continue
            if per_speaker[row["client_id"]] < 2:
                taken.append(row)
                taken_paths.add(row["path"])
                per_speaker[row["client_id"]] += 1
    if len(taken) < count:
        cap_reached = None
    drawn = sorted(taken, key=lambda row: salted_rank("cv", locale, row["path"]))
    return {"drawn": drawn, "cap_reached": cap_reached, "short": count - len(taken)}


def locale_plan(eligible: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """Each non-Arabic primary's locale: itself, or its one recorded substitute.

    ``eligible`` holds the clip counts after the guard at cap 2. A primary below
    :data:`CV_OTHER_PER_LOCALE` fires its substitute once; the substitute must reach the same
    count, otherwise the draw stops (:class:`ConfirmDrawError`).
    """
    planned: dict[str, dict[str, Any]] = {}
    for primary in PRIMARY_LOCALES:
        if eligible.get(primary, 0) >= CV_OTHER_PER_LOCALE:
            planned[primary] = {"locale": primary, "substituted": False}
            continue
        substitute = SUBSTITUTES[primary]
        if eligible.get(substitute, 0) < CV_OTHER_PER_LOCALE:
            raise ConfirmDrawError(
                f"{primary} has {eligible.get(primary, 0)} eligible clips at cap 2, and its "
                f"substitute {substitute} has {eligible.get(substitute, 0)}")
        planned[primary] = {"locale": substitute, "substituted": True}
    return planned


# --- the five bars -----------------------------------------------------------------------------


def pass_rate(passed: int, clips: int, *, limit: float) -> dict[str, Any]:
    """One negative arm: its rate, Wilson 95% interval and whether it exceeds the limit."""
    rate = passed / clips if clips else None
    return {"clips": clips, "passed": passed, "pass": rate,
            "wilson": list(wilson_interval(passed, clips)), "limit": limit,
            "fired": rate is None or rate > limit}


def five_bars(*, freesound: Sequence[bool], cv_ar: Sequence[bool],
              cv_other: Sequence[bool], mistakes: Sequence[bool],
              look_retention: float) -> dict[str, Any]:
    """The five bars and the status the first fired one names.

    Every arm takes the floor verdict itself (``floor_ok``): ``True`` = the clip passed the
    rule. For ``freesound``, ``cv_ar`` and ``cv_other`` a pass is the failure each 1% limit
    counts; the mistakes bar holds only when all 52 clips pass.
    ``look_retention`` is the share of look clips the rule keeps.
    """
    arms = {
        "freesound": pass_rate(sum(freesound), len(freesound), limit=PASS_LIMIT),
        "cv_ar": pass_rate(sum(cv_ar), len(cv_ar), limit=PASS_LIMIT),
        "cv_other": pass_rate(sum(cv_other), len(cv_other), limit=PASS_LIMIT),
    }
    mistake_passed = sum(1 for passed in mistakes if passed)
    arms["mistakes"] = {"clips": len(mistakes), "passed": mistake_passed,
                        "fired": mistake_passed != len(mistakes)}
    # A retention rate, not a count: ``passed`` has no meaning here and stays null.
    look = {"clips": None, "passed": None, "pass": look_retention,
            "limit": LOOK_RETENTION_MIN, "fired": look_retention < LOOK_RETENTION_MIN}

    status = "confirmed"
    for arm in ("freesound", "cv_ar", "cv_other"):
        if arms[arm]["fired"]:
            rate = arms[arm]["pass"]
            status = (f"stopped: {arm} {rate:g} > {arms[arm]['limit']:g}"
                      if rate is not None else f"stopped: {arm} no clips")
            break
    else:
        if arms["mistakes"]["fired"]:
            status = f"stopped: mistakes {mistake_passed}/{len(mistakes)}"
        elif look["fired"]:
            status = f"stopped: look_retention {look_retention:g} < {LOOK_RETENTION_MIN:g}"
    return {"status": status, **arms, "look_retention": look}


# --- the commit gate ---------------------------------------------------------------------------


def require_committed(paths: Iterable[Path], *, repo: Path = ROOT) -> None:
    """Refuse unless every path is tracked in ``repo`` and has no staged or unstaged change.

    The decode script may run only on committed inputs: the pre-registration and the manifests
    it cites.
    """
    root = Path(repo).resolve()
    for path in paths:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError(f"{path} is outside {root}") from error
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", "--",
                                  relative], capture_output=True)
        if tracked.returncode != 0:
            raise RuntimeError(f"{relative} is not committed")
        for staged in (False, True):
            command = ["git", "-C", str(root), "diff", "--quiet"]
            if staged:
                command.append("--cached")
            if subprocess.run([*command, "--", relative],
                              capture_output=True).returncode != 0:
                raise RuntimeError(f"{relative} has uncommitted changes")

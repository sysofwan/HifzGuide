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
import json
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from generate_quran_db import normalize_imlai
from training.counterfactual_eval import wilson_interval

from . import tlog_ident
from .tlog_index import head_sha, record_of, repo_relative, sha256_file

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


#: Urdu's heh goal / yeh barree and Persian's farsi yeh, which ``normalize_imlai`` leaves alone;
#: folding them onto the Arabic letters lets a quote in either orthography meet the Quran's MSA
#: spelling. The committed draw used the unfolded guard and is unchanged
#: (``docs/tlog-ident-floor-erratum.md`` §b).
_SCRIPT_FOLD = str.maketrans({"ھ": "ه", "ہ": "ه", "ۂ": "ه", "ی": "ي", "ے": "ي"})


def folded_guard_tokens(text: str) -> tuple[str, ...]:
    """The guard tokens with the Urdu/Persian letter variants folded onto the Arabic ones."""
    return guard_tokens(text.translate(_SCRIPT_FOLD))


def _contains_quran_ngram(tokens: Sequence[str],
                          ngrams: Iterable[tuple[str, ...]]) -> bool:
    return any(tuple(tokens[start:start + 4]) in ngrams
               for start in range(len(tokens) - 3))


def fails_quran_guard(sentence: str, ngrams: Iterable[tuple[str, ...]]) -> bool:
    """Whether four consecutive words of ``sentence`` occur contiguously in the Quran text."""
    return _contains_quran_ngram(guard_tokens(sentence), ngrams)


def fails_folded_quran_guard(sentence: str, ngrams: Iterable[tuple[str, ...]]) -> bool:
    """``fails_quran_guard`` with the ur/fa fold: the check a future draw must use.

    It changes no decision on the committed 1,000 sentences, so it is not the guard they were
    drawn under; that draw stands as committed (``docs/tlog-ident-floor-erratum.md`` §b).
    """
    return _contains_quran_ngram(folded_guard_tokens(sentence), ngrams)


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
    # The bar is 52/52, unconditionally: a short or empty list would silently restate it as
    # "all N" and let a missing population confirm the floor.
    if len(mistakes) != MISTAKE_CLIPS:
        raise ValueError(f"the real-mistake bar is {MISTAKE_CLIPS} clips, got {len(mistakes)}")
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


# --- the committed confirmation block ----------------------------------------------------------

#: The decode mode of each arm: the Freesound set repeats the 445's ``stream_b0``; the Common
#: Voice clips are whole-clip, the mode S6/S7 use for tlog rows.
CONFIRM_MODES = {"freesound": tlog_ident.NOISE_DECODE_MODE, "cv_ar": "whole-spans",
                 "cv_other": "whole-spans"}


def caught_by(identification: Any) -> dict[str, bool]:
    """The rule's two clauses on one row: below ``L_MIN``, or within ``DELTA`` of its decoy."""
    length = identification.length < tlog_ident.L_MIN
    margin = (identification.match is not None
              and identification.match.score - identification.decoy < tlog_ident.DELTA)
    return {"length": length, "margin": margin}


def _passing(key: str, spans: str, identification: Any) -> dict[str, Any]:
    match = identification.match
    return {"key": key, "decode": spans,
            "score": None if match is None else match.score,
            "length": identification.length,
            "class_label": identification.class_label,
            "span": None if match is None else {
                "surah": match.surah, "ayah_from": match.ayah_from,
                "word_from": match.word_from, "ayah_to": match.ayah_to,
                "word_to": match.word_to}}


def _evidence(rows: Sequence[tuple[str, Mapping[str, Any], Any]]
              ) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """One arm's rejects by clause and the items the rule let through, decode and span."""
    catches = {"length": 0, "margin": 0, "both": 0}
    passing: list[dict[str, Any]] = []
    for key, item, identification in rows:
        caught = caught_by(identification)
        catches["length"] += caught["length"]
        catches["margin"] += caught["margin"]
        catches["both"] += caught["length"] and caught["margin"]
        if identification.floor_ok:
            passing.append(_passing(key, item["spans"], identification))
    return catches, passing


def _locale_rate(locale: str, rows: Sequence[tuple[str, Mapping[str, Any], Any]]) -> dict:
    passed = sum(identification.floor_ok for _, _, identification in rows)
    return {"locale": locale, "clips": len(rows), "passed": passed,
            "pass": passed / len(rows) if rows else None,
            "wilson": list(wilson_interval(passed, len(rows))) if rows else None}


def confirm_block(database: Any, *, k: int, look_retention: float,
                  mistakes: Sequence[bool]) -> dict[str, Any]:
    """The five bars and the committed confirmation record, from the committed stores.

    Each store is scored once, at the calibration's ``k``, label-blind (no filename): a fresh
    negative that passes the rule is the failure each 1% limit counts. ``catches`` splits the
    rows the rule rejected by clause; ``passing`` records the decodes and best spans of the
    ones it let through. The provenance chain is the doc -> manifests -> stores order the
    calibration cites, with the scoring commit it ran from.
    """
    scored: dict[str, list[tuple[str, Mapping[str, Any], Any]]] = {}
    for arm, path in CONFIRM_DECODES.items():
        store = tlog_ident.load_decode_store(path)
        scored[arm] = [(key, item, tlog_ident.identify(item["spans"], database=database, k=k))
                       for key, item in store["items"].items()]
    verdicts = {arm: [identification.floor_ok for _, _, identification in rows]
                for arm, rows in scored.items()}
    bars = five_bars(freesound=verdicts["freesound"], cv_ar=verdicts["cv_ar"],
                     cv_other=verdicts["cv_other"], mistakes=mistakes,
                     look_retention=look_retention)

    catches: dict[str, dict[str, int]] = {}
    passing: dict[str, list[dict[str, Any]]] = {}
    for arm, rows in scored.items():
        catches[arm], passing[arm] = _evidence(rows)

    counts = json.loads(CV_COUNTS_PATH.read_text(encoding="utf-8"))
    locale_of = {row["path"]: row["locale"] for row in
                 map(json.loads, CV_MANIFEST_PATH.read_text(encoding="utf-8").splitlines())}
    per_language = {}
    for primary in PRIMARY_LOCALES:
        locale = counts["locales"][primary]["locale"]
        per_language[primary] = _locale_rate(
            locale, [row for row in scored["cv_other"] if locale_of.get(row[0]) == locale])

    freesound = json.loads(FREESOUND_MANIFEST_PATH.read_text(encoding="utf-8"))
    if counts["provenance"] != freesound["provenance"]:
        raise ValueError("the two confirmation manifests cite different provenance")
    provenance = freesound["provenance"]
    return {
        "salt": SALT,
        "status": bars["status"],
        "arms": {arm: bars[arm] for arm in
                 ("freesound", "cv_ar", "cv_other", "mistakes", "look_retention")},
        "per_language": per_language,
        "catches": catches,
        "passing": passing,
        "stores": {arm: {"path": repo_relative(path), "sha256": sha256_file(path),
                         "items": len(scored[arm]), "mode": CONFIRM_MODES[arm]}
                   for arm, path in CONFIRM_DECODES.items()},
        "provenance": {
            "doc": provenance["doc"],
            "manifests": {"freesound": record_of(FREESOUND_MANIFEST_PATH),
                          "common_voice": record_of(CV_MANIFEST_PATH)},
            "decodes": {arm: record_of(path) for arm, path in CONFIRM_DECODES.items()},
            "scoring": {"commit": head_sha()},
            "pull_request": provenance["pull_request"],
            "pull_request_ref": provenance["pull_request_ref"],
        },
    }


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

"""Re-grade the four detectors at the audit's commit and assign every word its stratum (#309).

The candidate pools are read from the detectors' committed whole-clip decodes (#300's gating
decodes: base, C95, round-2 and round-3 candidate EMA), re-graded here at
:data:`training.muraja_policy.CURRENT_COMMIT` in strict — a single-decode check set per clip, the
same replay #300 used — and joined to the App arm's committed word table. Neither the decodes nor
#300's own ``word_grades.jsonl.gz`` is trusted raw: the former are read through the candidate
inventory's pins and same-key-set check, the latter is not read at all (it was graded at the old
pin under ``allowances_off``, so its outcomes are not this audit's).

Round 2's 150 catch-loss words are marked from its committed worklist (A5b), keyed by the source
row and the 0-based canonical word the worklist names. Those words are look-tranche rows, so a
confirm.1/confirm.2 population matches none of them: the pools record ``marked`` and ``matched``
together so a zero is visible rather than implied.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from tadabur.followalong_audit.allocation import (
    PRIMARY_CAUSES, S1_CELL_FLOOR, S1_SNIPS, allocate, primary_cause,
)
from tadabur.followalong_audit.atomic import write_text_atomically
from tadabur.followalong_audit.coverage import ayah_coverage
from tadabur.followalong_audit.strata import (
    S1, S1_PENDING, S3, S4, S5, covered, pool_sizes, s1_records, stratify,
)
from tadabur.tlog_mining.candidate_inventory import DETECTORS, GATING, load_decode_lines
from training.aug_readout_gate import harness_identity
from training.external_corpus import word_grades
from training.muraja_policy import CURRENT_COMMIT, STRICT, MurajaText, require_harness
from training.nonpro_decode import item_key
from training.tlog_eval import gating_items

#: The tools root the committed names resolve under (``tadabur/…``, ``training/…``).
ROOT = Path(__file__).resolve().parents[2]

#: The ``tadabur`` package root the decode files sit under.
TADABUR_ROOT = ROOT / "tadabur"

#: Round 2's catch-loss wordlist: one line per word the candidate missed and base caught.
CATCH_LOSS_PATH = ROOT / "tadabur" / "consonant2_readout" / "catch_loss" / "worklist.jsonl"

#: How many words round 2's catch-loss audit marked (pinned by its own 150-line worklist).
CATCH_LOSS_WORDS = 150


def detector_decodes() -> tuple[dict, dict]:
    """The gating decodes: the base's full records (for the items) and each model's checks."""
    lines = load_decode_lines(TADABUR_ROOT)[GATING]
    base = lines[DETECTORS[0]]
    return base, {model: {key: {"checks": record["checks"]}
                          for key, record in lines[model].items()} for model in DETECTORS}


def catch_loss_marks(path: Path = CATCH_LOSS_PATH) -> set[tuple[str, int]]:
    """Round 2's catch-loss words as ``(item key, word_index)``, the coordinate its worklist names."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != CATCH_LOSS_WORDS:
        raise ValueError(f"{path} holds {len(rows)} words, not {CATCH_LOSS_WORDS}")
    return {(item_key(row["audio_filename"]), row["word_index"]) for row in rows}


def strata_words(rows: Sequence[Mapping], app_grades: Sequence[Mapping], *,
                 harness=None, oracle: Sequence[Mapping] | None = None,
                 failed: Sequence[str] = ()) -> tuple[list[dict], dict]:
    """Every graded population word with its App, oracle and detector outcomes.

    A clip the App run failed has no word table; its words are **excluded** (they are not S5,
    which means the App graded the clip and left that word ungraded) and counted in the metadata.
    """
    harness = require_harness(commit=CURRENT_COMMIT) if harness is None else harness
    base, decodes = detector_decodes()
    from tadabur.waqf_segments import hafs_segment_reference

    failed_keys = set(failed)
    items, unsupported = gating_items([dict(row) for row in rows], base,
                                      reference_of=hafs_segment_reference())
    if unsupported:
        raise ValueError(f"{len(unsupported)} population clips have no item: "
                         f"{sorted(unsupported)[:3]}")
    _, grades = word_grades(items, decodes, harness, models=DETECTORS, scorings=[STRICT])
    words, meta = words_from_grades(grades, app_grades, oracle=oracle, failed=failed_keys)
    coverage = _transcript_coverage(words, base, MurajaText(harness.quran_db))
    words = with_transcript_coverage(words, coverage)
    return words, {"items": len(items), **meta, "harness": harness_identity(harness)}


def _transcript_coverage(words: Sequence[Mapping], teacher_rows: Mapping,
                         text: MurajaText) -> dict[tuple[str, int], bool]:
    """The Teacher transcript's verdict on every App-ungraded word, keyed ``(clip, word)``."""
    by_clip: dict[str, list[int]] = {}
    for row in words:
        if row["app"] == "ungraded":
            by_clip.setdefault(row["key"], []).append(row["word"])
    coverage = {}
    for key, clip_words in by_clip.items():
        row = teacher_rows[key]
        surah, ayah = (int(part) for part in row["surah_ayah"].split(":"))
        listed = ayah_coverage(row, text, surah, ayah, clip_words)
        coverage.update({(key, word): holds for word, holds in listed.items()})
    return coverage


def with_transcript_coverage(words: Iterable[Mapping],
                             coverage: Mapping[tuple[str, int], bool]) -> list[dict]:
    """Each word with ``transcript_covered``: the gate's verdict for an App-ungraded word, else
    ``None``. A word the gate never assessed is refused; the input is not mutated."""
    return [{**row,
             "transcript_covered": (coverage[(row["key"], row["word"])]
                                    if row["app"] == "ungraded" else None)}
            for row in words]


def words_from_grades(grades: Sequence, app_grades: Sequence[Mapping], *,
                      oracle: Sequence[Mapping] | None = None,
                      failed: Sequence[str] = ()) -> tuple[list[dict], dict]:
    """The strata's per-word rows from the detectors' grades and the App arm's table.

    A word is ``{"key", "position", "word", "word_index", "app", "oracle", "detectors",
    "detector_graded", "causes"}``: ``detector_graded`` is whether each detector graded the word
    (an ungraded detector never flags), which S4's bound records. A clip the App run failed is
    excluded — its words are not S5, which means the App graded the clip and left that word
    ungraded — and counted in the metadata.
    """
    failed_keys = set(failed)
    app_by_word = {(row["item"], row["word"]): row["strict"] for row in app_grades}
    oracle_by_word = {(row["item"], row["word"]): _oracle_outcome(row["strict"])
                      for row in (oracle or [])}
    by_position: dict[tuple[str, int], dict[str, object]] = {}
    for grade in grades:
        by_position.setdefault((grade.item.key, grade.position), {})[grade.model] = grade
    words, ungraded, excluded_words = [], 0, 0
    item_keys = set()
    for (key, position), per_model in sorted(by_position.items()):
        item_keys.add(key)
        if key in failed_keys:
            excluded_words += 1
            continue
        grade = per_model[DETECTORS[0]]
        cell = app_by_word.get((key, grade.word))
        ungraded += sum(1 for model in DETECTORS if not per_model[model].graded)
        words.append({
            "key": key,
            "position": position,
            "word": grade.word,
            "word_index": grade.word_index,
            "app": _outcome(cell),
            "oracle": oracle_by_word.get((key, grade.word)),
            "detectors": {model: bool(per_model[model].flagged) for model in DETECTORS},
            "detector_graded": {model: bool(per_model[model].graded) for model in DETECTORS},
            "causes": list(cell["causes"]) if cell is not None else [],
        })
    return words, {"words": len(words), "detector_ungraded": ungraded,
                   "excluded_failed_clips": len(failed_keys & item_keys),
                   "excluded_failed_words": excluded_words}


def _outcome(cell: Mapping | None) -> str | None:
    if cell is None or not cell["graded"]:
        return "ungraded"
    return "flagged" if cell["flagged"] else "passed"


def _oracle_outcome(cell: Mapping) -> str | None:
    """S2's outcome for a word: the strata model has no "oracle ungraded", so that is unknown."""
    outcome = _outcome(cell)
    return None if outcome == "ungraded" else outcome


def stratified_run(rows: Sequence[Mapping], app_grades: Sequence[Mapping], *,
                   harness=None, oracle: Sequence[Mapping] | None = None,
                   failed: Sequence[str] = ()) -> tuple[list[dict], dict, set]:
    """The graded population with its strata, its metadata, and round 2's catch-loss marks."""
    words, meta = strata_words(rows, app_grades, harness=harness, oracle=oracle, failed=failed)
    marks = catch_loss_marks()
    return stratify(words, catch_loss=marks), meta, marks


def pools_from_run(stratified: Sequence[Mapping], meta: Mapping,
                   marks: set) -> dict:
    """The pools summary: pool sizes, the S1 pool by cause, and the catch-loss marks."""
    matched = sum(1 for row in stratified if row["catch_loss"])
    return {
        "clips": meta["items"] - meta["excluded_failed_clips"],
        "words": meta["words"],
        "detector_ungraded": meta["detector_ungraded"],
        "excluded_failed_clips": meta["excluded_failed_clips"],
        "excluded_failed_words": meta["excluded_failed_words"],
        "harness": meta["harness"],
        "pools": pool_sizes(stratified),
        "s1_by_cause_and_base": s1_records(stratified),
        "catch_loss": {"marked": len(marks), "matched": matched,
                       "worklist": "consonant2_readout/catch_loss/worklist.jsonl"},
    }


def candidate_pools(rows: Sequence[Mapping], app_grades: Sequence[Mapping], *,
                    harness=None, oracle: Sequence[Mapping] | None = None,
                    failed: Sequence[str] = ()) -> dict:
    """The strata table's summary: pool sizes, the S1 pool by cause, and the catch-loss marks.

    ``oracle`` is S2's word table (the same line shape as the App arm's, read under ``strict``),
    joined on ``(item, word)``: with it, ``assign_stratum`` splits S1 from ``ORACLE_EXPLAINED``
    and the S1 pool becomes final; without it the pool renders pending.
    """
    return pools_from_run(*stratified_run(rows, app_grades, harness=harness, oracle=oracle,
                                          failed=failed))


#: S5's coverage classes: where the word sits in its clip.
S5_CLASSES = ("S5-leading", "S5-trailing", "S5-interior")

#: The coverage classes: every sampled stratum, S5 split by place in the clip.
COVERAGE_CLASSES = ("S1", "S3", "S4", *S5_CLASSES)


def _edges(stratified: Sequence[Mapping]) -> dict[tuple[str, int], str]:
    """Each word's S5 edge class, decided over *every* word of its clip, covered or not.

    A word before the clip's first App-graded word is ``leading`` (every word of a wholly
    ungraded clip is), after its last is ``trailing``, and between them is ``interior``.
    """
    by_clip: dict[str, list[tuple[int, bool]]] = {}
    for row in stratified:
        by_clip.setdefault(row["key"], []).append((row["position"], row["app"] != "ungraded"))
    edges = {}
    for key, rows in by_clip.items():
        graded = [position for position, is_graded in rows if is_graded]
        if not graded:
            edges.update({(key, position): "leading" for position, _ in rows})
            continue
        first, last = min(graded), max(graded)
        for position, _ in rows:
            edges[(key, position)] = ("leading" if position < first
                                      else "trailing" if position > last else "interior")
    return edges


def _coverage_class(row: Mapping, edges: Mapping[tuple[str, int], str]) -> str | None:
    """The coverage class a row belongs to; ``None`` for a stratum the gate does not sample."""
    if row["stratum"] == S5:
        return f"S5-{edges[(row['key'], row['position'])]}"
    return {S1: "S1", S1_PENDING: "S1", S3: "S3", S4: "S4"}.get(row["stratum"])


def draw_frame(stratified: Sequence[Mapping]) -> list[dict]:
    """The words the draw may pick from (A1), sorted by clip then position.

    Every S1 (pending included), S3 and S4 word, and S5's *covered* words. Each row is
    ``{"key", "position", "stratum", "edge", "whole_played"}``; S5 edges are served whole.
    Residual, oracle-explained and uncovered S5 words are not in the frame.
    """
    edges = _edges(stratified)
    frame = []
    for row in stratified:
        name = _coverage_class(row, edges)
        if name is None or (name.startswith("S5") and not covered(row)):
            continue
        edge = edges[(row["key"], row["position"])] if name.startswith("S5") else None
        frame.append({"key": row["key"], "position": row["position"],
                      "stratum": "S5" if edge is not None else name, "edge": edge,
                      "whole_played": edge in ("leading", "trailing")})
    return sorted(frame, key=lambda row: (row["key"], row["position"]))


def coverage_record(stratified: Sequence[Mapping]) -> dict:
    """The covered frame and the not-covered pool, by coverage class.

    Only S5's gate drops words; S1, S3 and S4 are whole. ``frame_words`` is the sampled frame and
    ``not_covered_words`` the reported pool with π = 0.
    """
    frame = dict.fromkeys(COVERAGE_CLASSES, 0)
    not_covered = dict.fromkeys(COVERAGE_CLASSES, 0)
    s5_engine_graded = dict.fromkeys(S5_CLASSES, 0)
    edges = _edges(stratified)
    for row in stratified:
        name = _coverage_class(row, edges)
        if name is None:
            continue
        if name.startswith("S5") and not covered(row):
            not_covered[name] += 1
            continue
        frame[name] += 1
        if name.startswith("S5") and row["detector_graded"]["base"]:
            s5_engine_graded[name] += 1
    return {"frame": frame, "not_covered": not_covered,
            "s5_engine_graded": s5_engine_graded,
            "frame_words": sum(frame.values()),
            "not_covered_words": sum(not_covered.values())}


def primary_cells(stratified: Sequence[Mapping]) -> list[dict]:
    """S1's 14 primary-cause × base-flag cells with their pre-registered allocation.

    A word counts once, in its primary cause's cell (D2). The 200 snips are allocated by
    :func:`tadabur.followalong_audit.allocation.allocate` with the pre-registered floor. The
    per-cell inclusion probability (§3) is its ``n`` over its ``words``.
    """
    counts = {(cause, base): 0 for cause in PRIMARY_CAUSES for base in (False, True)}
    for row in stratified:
        if row["stratum"] in (S1, S1_PENDING):
            counts[(primary_cause(row["causes"]), bool(row["detectors"]["base"]))] += 1
    split = allocate(counts, S1_SNIPS, floor=S1_CELL_FLOOR)
    return [{"cause": cause, "base_flags": base, "words": counts[(cause, base)],
             "n": split[(cause, base)]}
            for cause in PRIMARY_CAUSES for base in (False, True)]


def s4_ungraded(stratified: Sequence[Mapping]) -> dict:
    """S4's words and how many detectors graded nothing at them (D11)."""
    rows = [row for row in stratified if row["stratum"] == S4]
    return {
        "words": len(rows),
        "at_least_one_detector_ungraded": sum(
            1 for row in rows if not all(row["detector_graded"].values())),
        "all_four_detectors_ungraded": sum(
            1 for row in rows if not any(row["detector_graded"].values())),
        "ungraded_pairs": sum(1 for row in rows for graded in row["detector_graded"].values()
                              if not graded),
    }


def cells_record(stratified: Sequence[Mapping], detector_ungraded: int) -> dict:
    """The coverage record, the primary-cause cells and S4's ungraded split, as the committed
    ``app_arm/cells.json``."""
    return {"coverage": coverage_record(stratified),
            "detector_ungraded": detector_ungraded,
            "s1_primary_cells": primary_cells(stratified),
            "s4": s4_ungraded(stratified)}


def write_pools(pools: Mapping, path: Path) -> None:
    """The pools summary as sorted, indented JSON, atomically."""
    write_text_atomically(Path(path),
                          json.dumps(pools, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_cells(cells: Mapping, path: Path) -> None:
    """The cells record as sorted, indented JSON, atomically."""
    write_text_atomically(Path(path),
                          json.dumps(cells, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

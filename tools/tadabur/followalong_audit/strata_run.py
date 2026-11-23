"""Re-grade the four detectors at the audit's commit and assign every word its stratum (#309).

The candidate pools are read from the detectors' committed whole-clip decodes (#300's gating
decodes: base, C95, round-2 and round-3 candidate EMA), re-graded here at
:data:`training.muraja_policy.CURRENT_COMMIT` in strict — a single-decode check set per clip, the
same replay #300 used — and joined to the App arm's committed word table. Nothing is read from
#300's own ``word_grades.jsonl.gz``: it was graded at the old pin under ``allowances_off``, so its
outcomes are not this audit's.

Round 2's 150 catch-loss words are marked from its committed worklist (A5b), by source row and
word index, so the pools show how much of each candidate stratum the previous round could not
catch.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.followalong_audit.pinned import pinned_records
from tadabur.followalong_audit.strata import pool_sizes, s1_records, stratify
from tadabur.tlog_mining.candidate_inventory import DETECTORS
from training.aug_readout_gate import harness_identity
from training.external_corpus import word_grades
from training.muraja_policy import CURRENT_COMMIT, STRICT, require_harness
from training.nonpro_decode import item_key
from training.tlog_eval import gating_items

#: The tools root the committed names resolve under (``tadabur/…``, ``training/…``).
ROOT = Path(__file__).resolve().parents[2]

#: The ``tadabur`` package root the pinned decode names are relative to.
TADABUR_ROOT = ROOT / "tadabur"

#: Round 2's catch-loss wordlist: one line per word the candidate missed and base caught.
CATCH_LOSS_PATH = ROOT / "tadabur" / "consonant2_readout" / "catch_loss" / "worklist.jsonl"

#: How many words round 2's catch-loss audit marked (pinned by its own 150-line worklist).
CATCH_LOSS_WORDS = 150


def decode_records(name: str) -> list[dict]:
    """A pinned gating decode's records, refused unless its bytes hash to the pin."""
    return pinned_records(name, TADABUR_ROOT)


def detector_decodes(names: Mapping[str, str] | None = None) -> tuple[dict, dict]:
    """The four gating decodes: the base's full records (for the items) and each model's checks."""
    names = names or {model: _decode_name(model) for model in DETECTORS}
    records = {model: decode_records(names[model]) for model in DETECTORS}
    base = {line["key"]: line for line in records[DETECTORS[0]]}
    if any(set(line["key"] for line in records[model]) != set(base) for model in DETECTORS):
        raise ValueError("the four detector decodes do not name the same clips")
    return base, {model: {line["key"]: {"checks": line["checks"]}
                          for line in records[model]} for model in DETECTORS}


def _decode_name(model: str) -> str:
    from tadabur.tlog_mining.candidate_inventory import GATING, decode_name

    return decode_name(GATING, model)


def catch_loss_marks(path: Path = CATCH_LOSS_PATH) -> set[tuple[str, int]]:
    """Round 2's catch-loss words as ``(item key, word index)``."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != CATCH_LOSS_WORDS:
        raise ValueError(f"{path} holds {len(rows)} words, not {CATCH_LOSS_WORDS}")
    return {(item_key(row["audio_filename"]), row["word_index"]) for row in rows}


def strata_words(rows: Sequence[Mapping], app_grades: Sequence[Mapping],
                 *, harness=None) -> tuple[list[dict], dict]:
    """Every population word with its App outcome and the four detectors' flags."""
    harness = require_harness(commit=CURRENT_COMMIT) if harness is None else harness
    base, decodes = detector_decodes()
    from tadabur.waqf_segments import hafs_segment_reference

    items, unsupported = gating_items([dict(row) for row in rows], base,
                                      reference_of=hafs_segment_reference())
    if unsupported:
        raise ValueError(f"{len(unsupported)} population clips have no item: "
                         f"{sorted(unsupported)[:3]}")
    _, grades = word_grades(items, decodes, harness, models=DETECTORS, scorings=[STRICT])
    app_by_word = {(row["item"], row["word"]): row["strict"] for row in app_grades}
    by_position: dict[tuple[str, int], dict[str, object]] = {}
    for grade in grades:
        by_position.setdefault((grade.item.key, grade.position), {})[grade.model] = grade
    words, ungraded = [], 0
    for (key, position), per_model in sorted(by_position.items()):
        grade = per_model[DETECTORS[0]]
        cell = app_by_word.get((key, grade.word))
        ungraded += sum(1 for model in DETECTORS if not per_model[model].graded)
        words.append({
            "key": key,
            "position": position,
            "word": grade.word,
            "app": _app_outcome(cell),
            "oracle": None,
            "detectors": {model: bool(per_model[model].flagged) for model in DETECTORS},
            "causes": list(cell["causes"]) if cell is not None else [],
        })
    return words, {"items": len(items), "words": len(words),
                   "detector_ungraded": ungraded,
                   "harness": harness_identity(harness)}


def _app_outcome(cell: Mapping | None) -> str:
    if cell is None or not cell["graded"]:
        return "ungraded"
    return "flagged" if cell["flagged"] else "passed"


def candidate_pools(rows: Sequence[Mapping], app_grades: Sequence[Mapping],
                    *, harness=None) -> dict:
    """The strata table's summary: pool sizes, the S1 pool by cause, and the catch-loss marks."""
    words, meta = strata_words(rows, app_grades, harness=harness)
    marks = catch_loss_marks()
    stratified = stratify(words, catch_loss=marks)
    matched = sum(1 for row in stratified if row["catch_loss"])
    return {
        "clips": meta["items"],
        "words": meta["words"],
        "detector_ungraded": meta["detector_ungraded"],
        "harness": meta["harness"],
        "pools": pool_sizes(stratified),
        "s1_by_cause_and_base": s1_records(stratified),
        "catch_loss": {"marked": len(marks), "matched": matched},
    }


def write_pools(pools: Mapping, path: Path) -> None:
    """The pools summary as sorted, indented JSON, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(pools, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)

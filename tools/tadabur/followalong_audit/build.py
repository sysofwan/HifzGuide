"""Build the committed S5 draw (#311, PR 1): frame, gate, key, worklist, serve orders, snips.

The draw is re-derived, never hand-edited. The **frame** is :func:`strata_run.draw_frame`'s covered
words over the committed App table and detectors; the **gate** compares a second full App run with
``app_arm/word_grades.jsonl.gz`` and removes the words that did not reproduce; the **key**,
**worklist**, serve orders and **snips** come from the seed alone. Every file lands in
``s5_draw/`` before any listening.

The gate needs the regenerated table and the per-clip traces that carry its error sets (the
committed table holds only ``{graded, flagged, causes}``). Pass them with ``--regen-grades`` and
``--traces``; ``--reuse-frame`` re-reads a frame already built, which is the slow step.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from tadabur.followalong_audit.app_arm import read_grades, read_jsonl, write_jsonl
from tadabur.followalong_audit.atomic import write_bytes_atomically, write_text_atomically
from tadabur.followalong_audit.clips import app_arm_rows
from tadabur.followalong_audit.draw import (
    AMENDMENT_COMMIT, EDGE_WHOLE_REASON, MAIN_REPEATS, PILOT_REPEATS, S5_EDGES, derive_seed,
    design_totals, frame_rows, item_id, non_reproducing, plan_sitting, repeat_id,
    require_commit_hex, select_words, snip_placements,
)
from tadabur.followalong_audit.strata_run import stratified_run
from tadabur.word_snips import SnipPlacement, write_snips
from training.nonpro_decode import item_key

AUDIT = Path(__file__).resolve().parent
APP_ARM = AUDIT / "app_arm"
DRAW_DIR = AUDIT / "s5_draw"
ORACLE_GRADES = AUDIT / "oracle_sweep" / "word_grades.jsonl.gz"
WORD_TIMES = AUDIT.parent / "tlog_mining" / "word_times"
DOCS = AUDIT.parents[2] / "docs"
PREREGISTRATION = DOCS / "followalong-audit-preregistration.md"
AMENDMENT = DOCS / "followalong-audit-draw-amendment.md"

TRANCHES = ("confirm.1", "confirm.2")


def build(*, out: Path = DRAW_DIR, draw_commit: str, regen_grades: Path, traces: Path,
          reuse_frame: bool = False) -> dict:
    """Write every committed draw file under ``out`` and return the run's counts."""
    require_commit_hex(draw_commit)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    population = app_arm_rows()
    frame = (_read_jsonl_gz(out / "frame.jsonl.gz") if reuse_frame
             else build_frame(population))
    _check_frame(frame)
    _write_jsonl_gz(frame, out / "frame.jsonl.gz")

    committed = read_grades(APP_ARM / "word_grades.jsonl.gz")
    regenerated = read_grades(Path(regen_grades))
    reproduction = build_reproduction(frame, committed, regenerated,
                                      regen_grades=Path(regen_grades), traces=Path(traces))
    _write_json(reproduction, out / "reproduction.json")
    _write_jsonl_gz(build_app_errors(Path(traces), regenerated, population),
                    out / "app_errors.jsonl.gz")

    seed = derive_seed(AMENDMENT_COMMIT)
    removed = {(entry["item"], entry["word"]) for entry in reproduction["did_not_reproduce"]}
    draws = select_words(frame, seed=seed, removed=removed)
    clips = {item_key(clip["audio_path"]): clip for clip in population}
    manifest = {row["audio_sha256"]: row
                for row in _read_jsonl_gz(APP_ARM / "audio_manifest.jsonl.gz")}
    originals = _key_rows(draws, clips, seed)
    plans, repeats = _plans_and_repeats(originals, seed)
    key = sorted(originals + repeats, key=lambda row: row["id"])
    designed = sum(design_totals(frame).values())
    if len(originals) != designed:
        raise ValueError(f"the key holds {len(originals)} drawn words, not the design's {designed}")
    if len({row["id"] for row in key}) != len(key):
        raise ValueError("two served items share an id")
    if len(key) != len(originals) + PILOT_REPEATS + MAIN_REPEATS:
        raise ValueError(f"the key holds {len(key)} served items, not the design's")
    worklist = _worklist_rows(key, clips, manifest)
    write_jsonl(key, out / "key.jsonl")
    write_jsonl(worklist, out / "worklist.jsonl")
    for sitting, plan in plans.items():
        write_jsonl(_serve_rows(plan, seed), out / f"{sitting}.jsonl")

    records = [record for tranche in TRANCHES
               for record in _read_jsonl_gz(WORD_TIMES / f"{tranche}.jsonl.gz")]
    placed = snip_placements(worklist, key, records)
    write_snips((placement.snip for placement in placed), out / "snips.jsonl")
    report = _snip_report(placed)
    _write_json(report, out / "snips.report.json")

    _write_json({
        "amendment_commit": AMENDMENT_COMMIT,
        "seed": seed,
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "amendment_sha256": _sha256(AMENDMENT),
        "draw_commit": draw_commit,
    }, out / "draw_seed.json")
    return {"frame": len(frame), "flips": reproduction["flips"],
            "drawn": len(originals), "served": len(key), "snips": report}


# --- the frame ------------------------------------------------------------------------------------

def build_frame(population: Sequence[Mapping]) -> list[dict]:
    """The draw frame, re-derived from the committed App table and detectors.

    :func:`build` checks it against the committed pools before anything is written.
    """
    stratified, _meta, _marks = stratified_run(
        population, read_grades(APP_ARM / "word_grades.jsonl.gz"),
        oracle=read_grades(ORACLE_GRADES), failed=_failed_items(APP_ARM / "clip_runs.jsonl"))
    return frame_rows(stratified)


def _failed_items(path: Path) -> set[str]:
    return {row["item"] for row in read_jsonl(path) if row["failure"] is not None}


def _check_frame(frame: Sequence[Mapping]) -> None:
    """Refuse a frame that is not the committed pools: a drift would silently redraw."""
    counts = Counter((row["stratum"], row["cell"]) for row in frame)
    cells = json.loads((APP_ARM / "cells.json").read_text(encoding="utf-8"))
    expected = {(stratum, "all"): cells["coverage"]["frame"][stratum]
                for stratum in ("S3", "S4")}
    expected.update({("S5", edge): cells["coverage"]["frame"][f"S5-{edge}"] for edge in S5_EDGES})
    expected.update({("S1", f"{cell['cause']}:{'yes' if cell['base_flags'] else 'no'}"):
                     cell["words"] for cell in cells["s1_primary_cells"]})
    if dict(counts) != expected:
        differing = sorted(set(counts) ^ set(expected))
        differing += [key for key in set(counts) & set(expected) if counts[key] != expected[key]]
        raise ValueError(f"the frame is not the committed pools: {differing[:5]}")


# --- the gate -------------------------------------------------------------------------------------

def build_reproduction(frame: Sequence[Mapping], committed: Sequence[Mapping],
                       regenerated: Sequence[Mapping], *, regen_grades: Path,
                       traces: Path) -> dict:
    """The F-ref gate: every word's outcome compared, and the frame words that did not reproduce.

    The regenerated table and the per-clip traces are the second App run's, on the GPU box and
    uncommitted, so the record pins them by digest: a "0 flips" claim is only re-verifiable
    against the same run.
    """
    flips = non_reproducing(committed, regenerated)
    frame_words = {(row["key"], row["word"]) for row in frame}
    gone = [word for word in flips if word in frame_words]
    gone_set = set(gone)
    counts = Counter((row["stratum"], row["cell"]) for row in frame)
    removed = Counter((row["stratum"], row["cell"]) for row in frame
                      if (row["key"], row["word"]) in gone_set)
    cells = [{"stratum": stratum, "cell": cell, "frame": counts[(stratum, cell)],
              "did_not_reproduce": removed[(stratum, cell)],
              "population": counts[(stratum, cell)] - removed[(stratum, cell)]}
             for (stratum, cell) in sorted(counts)]
    return {
        "committed_words": len(committed),
        "regenerated_words": len(regenerated),
        "regen_grades_sha256": _sha256(regen_grades),
        "traces": _traces_identity(traces),
        "flips": len(flips),
        "flips_in_s1_cells": sum(entry["did_not_reproduce"] for entry in cells
                                 if entry["stratum"] == "S1"),
        "did_not_reproduce": [{"item": item, "word": word} for item, word in sorted(gone)],
        "cells": cells,
    }


def _traces_identity(traces: Path) -> dict:
    """The gate's per-clip traces: how many, and one digest over their names and bytes."""
    paths = sorted(Path(traces).glob("*.jsonl.gz"))
    listing = "".join(f"{path.name} {_sha256(path)}\n" for path in paths)
    return {"files": len(paths), "sha256": sha256(listing.encode("utf-8")).hexdigest()}


def build_app_errors(traces: Path, regenerated: Sequence[Mapping],
                     population: Sequence[Mapping]) -> list[dict]:
    """The regenerated App strict flagged words as ``reference_suspects`` observations.

    A flagged word's error set is the deviation itself; the committed word table keeps only the
    causes, so the gate's own per-clip traces are the source. Every flagged word must have its
    strict trace: a missing one would silently shrink the reference-suspect count.
    """
    flagged = {(row["item"], row["word"]) for row in regenerated if row["strict"]["flagged"]}
    clips = {item_key(clip["audio_path"]): clip for clip in population}
    rows: list[dict] = []
    for path in sorted(Path(traces).glob("*.jsonl.gz")):
        for trace in _read_jsonl_gz(path):
            if trace["scoring"] != "strict" or (trace["item"], trace["word"]) not in flagged:
                continue
            clip = clips[trace["item"]]
            surah, ayah = (int(part) for part in clip["filename_label"].split(":"))
            rows.append({"item": trace["item"], "group_key": clip["group_key"],
                         "surah": surah, "ayah": ayah, "word": trace["word"],
                         "errors": trace["errors"]})
    found = {(row["item"], row["word"]) for row in rows}
    missing = sorted(flagged - found)
    if missing:
        raise ValueError(f"{len(missing)} regenerated flagged words have no strict trace "
                         f"({missing[:3]})")
    return sorted(rows, key=lambda row: (row["item"], row["word"]))


# --- the key, the worklist and the serve orders ---------------------------------------------------

def _key_rows(draws: Mapping, clips: Mapping[str, Mapping], seed: int) -> list[dict]:
    """One key row per drawn word: its blind id and the design facts the page never sees."""
    rows = []
    for draw in draws.values():
        for index, word in enumerate(draw.rows):
            clip = clips[word["key"]]
            rows.append({
                "id": item_id(seed, word["key"], word["word"]),
                "item": word["key"], "word": word["word"], "word_index": word["word_index"],
                "surah_ayah": clip["filename_label"], "stratum": draw.stratum,
                "cell": draw.cell, "edge": word["edge"], "whole_played": word["whole_played"],
                "n": draw.n, "population": draw.population,
                "inclusion_probability": draw.inclusion,
                "in_pilot": index < draw.pilot_n, "repeat_of": None,
            })
    return rows


def _plans_and_repeats(originals: Sequence[Mapping], seed: int
                       ) -> tuple[dict[str, list], list[dict]]:
    """The pilot and main sittings, and the key row a repeat is served under."""
    by_id = {row["id"]: row for row in originals}
    repeats = []
    plans = {}
    for sitting, count in (("pilot", PILOT_REPEATS), ("main", MAIN_REPEATS)):
        items = [row["id"] for row in originals if row["in_pilot"] == (sitting == "pilot")]
        plan = plan_sitting(items, seed=seed, sitting=sitting, repeats=count)
        plans[sitting] = plan
        for served in plan:
            if served.repeat:
                repeats.append({**by_id[served.item], "id": repeat_id(seed, served.item),
                                "repeat_of": served.item, "inclusion_probability": None})
    return plans, repeats


def _serve_rows(plan: Sequence, seed: int) -> list[dict]:
    return [{"id": repeat_id(seed, served.item) if served.repeat else served.item,
             "position": position}
            for position, served in enumerate(plan, 1)]


def _worklist_rows(key: Sequence[Mapping], clips: Mapping[str, Mapping],
                   manifest: Mapping[str, Mapping]) -> list[dict]:
    """The blind file: an opaque id, the clip's provenance, the ayah and the word's text.

    One line per served item, a repeat's line identical to its original's but for the id, so the
    page can serve both without knowing which is which.
    """
    rows = []
    for row in key:
        clip = clips[row["item"]]
        words = _uthmani(row["surah_ayah"])
        if not 0 <= row["word_index"] < len(words):
            raise ValueError(f"{row['item']}: word {row['word_index']} is outside "
                             f"{row['surah_ayah']}")
        fetched = manifest.get(clip["audio_sha256"])
        if fetched is None:
            raise ValueError(f"{row['item']}: no decoded clip in the audio manifest")
        rows.append({
            "id": row["id"], "split": clip["split"], "shard_file": clip["shard_file"],
            "row": clip["row"], "audio_filename": clip["audio_path"],
            "audio_sha256": clip["audio_sha256"], "start_sample": 0,
            "end_sample": fetched["samples"], "surah_ayah": row["surah_ayah"],
            "word_index": row["word_index"], "word": words[row["word_index"]],
        })
    return sorted(rows, key=lambda row: row["id"])


@lru_cache(maxsize=None)
def _uthmani(surah_ayah: str) -> tuple[str, ...]:
    from tadabur.waqf_segments import _uthmani_words

    return tuple(_uthmani_words(surah_ayah))


# --- the files ------------------------------------------------------------------------------------

def _snip_report(placed: Sequence[SnipPlacement]) -> dict:
    """The snips' counts: how many play whole, why, and the alignment fallback rate.

    ``fallback_rate`` is the whole-clip share (the sidecar convention, and the number the locked
    test pins); the S5 edge words play whole by rule, so the alignment's own failures are
    ``alignment_fallbacks`` / ``alignment_fallback_rate``, which is what the run note quotes.
    """
    reasons = Counter(placement.reason for placement in placed if placement.reason is not None)
    rows = len(placed)
    whole = sum(reasons.values())
    edge = reasons.get(EDGE_WHOLE_REASON, 0)
    fallbacks = whole - edge
    return {"rows": rows,
            "alignment": rows - whole,
            "whole_clip": whole,
            "edge_whole": edge,
            "fallback_rate": whole / rows if rows else 0.0,
            "alignment_fallbacks": fallbacks,
            "alignment_fallback_rate": fallbacks / rows if rows else 0.0,
            "reasons": dict(sorted(reasons.items()))}


def _sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_jsonl_gz(rows: Iterable[Mapping], path: Path) -> None:
    """One sorted-key JSON object per line, gzipped with a fixed mtime so builds are identical."""
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    write_bytes_atomically(Path(path), gzip.compress(body.encode("utf-8"), mtime=0))


def _write_json(data: Mapping, path: Path) -> None:
    write_text_atomically(Path(path),
                          json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

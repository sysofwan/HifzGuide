"""Freeze the adjudicated waqf audit into reciter-disjoint calibration + test sets (#30).

The correction-based audit UI (:mod:`tadabur.waqf_audit_ui`) leaves three local files:
a candidate baseline (:mod:`tadabur.waqf_candidates`), an **overrides-only** event store
(``waqf_events.jsonl`` — only the boundaries the human corrected), and a
``waqf_reviewed_clips.json`` roster of the clips explicitly marked reviewed. The frozen
eval set ADR-0004's F0 owes F1/F2 is the *materialized* ground truth over those reviewed
clips — every candidate boundary carried with its human ``verdict`` (an override where one
exists, else the detector's ``predicted`` class) — split **reciter-disjoint** into a
calibration partition (F2 tunes the inference threshold on it) and a test partition
(reported once), so no reciter — hence no clip — straddles the two.

The gate is **binary**: a boundary is scored as ``waqf`` (a real stop) versus *not-waqf*
(``wasl`` continuation or ``mid_word_closure``). ``mid_word_closure`` — a VAD silence the
segmenter did **not** treat as a segment boundary, i.e. a within-word articulation closure
(qalqala on ق/ط, hamza in شَيء, madd elongation) — is not a distinct product outcome but a
*diagnostic tag* on the not-waqf class, kept only to report the hard-negative rejection
rate. It is best-effort: the human reviewer collapsed some interior silences straight to
``wasl``, so the tag under-counts, but every ``mid_word_closure`` is unambiguously not-waqf
and so never moves the binary ground truth. Only ``verdict == "waqf"`` is a positive; the
report's ``binary`` block records this split per partition.

Because the D2/D3 fine-tune has not yet fixed a training-reciter set, disjointness from
training is guaranteed the other way round: this freeze emits ``must_exclude_reciters``
(every reciter in either eval partition) for the eventual training run to hold out, so
calibration/test stay leak-free by construction.

An override whose ``(clip_id, boundary_index)`` no longer names a boundary in the current
candidate baseline (e.g. a false-negative recorded against an earlier candidate version,
before the early-stop/re-read fixes trimmed that clip's boundaries) is **stale**: it cannot
be placed on the baseline, so it is dropped from the frozen set and listed under
``stale_overrides`` in the report rather than silently distorting the ground truth.

Usage:
  python -m tadabur.waqf_freeze \\
    --candidates audit_run/waqf_candidates.jsonl \\
    --events     waqf_event_fixtures/waqf_events.jsonl \\
    --reviewed   waqf_event_fixtures/waqf_reviewed_clips.json \\
    --out-dir    waqf_event_fixtures \\
    [--test-fraction 0.5] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .waqf_event_fixtures import WAQF, WaqfEventEntry, write_waqf_events
from .waqf_partition import partition, reciter_of


def load_reviewed(path: Path) -> set[str]:
    """The set of clip ids explicitly marked reviewed (the eval-set membership)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data["reviewed"])


def load_overrides(path: Path) -> dict[tuple[str, int], dict]:
    """Human overrides keyed by ``(clip_id, boundary_index)`` — the audit's only edits."""
    overrides: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return overrides
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            overrides[(row["clip_id"], row["boundary_index"])] = row
    return overrides


def materialize(
    candidates_path: Path,
    reviewed: set[str],
    overrides: dict[tuple[str, int], dict],
) -> tuple[dict[str, list[WaqfEventEntry]], list[dict]]:
    """Materialize per-clip ground truth (baseline ⊕ overrides) for the reviewed clips.

    Streams the candidate baseline and, for every candidate boundary of a reviewed clip,
    emits a :class:`WaqfEventEntry` whose ``verdict`` is the human override for that
    ``(clip_id, boundary_index)`` if one exists, else the detector's ``predicted`` class.
    Returns ``(entries_by_clip, stale_overrides)`` where ``stale_overrides`` are override
    rows that matched no current candidate boundary (see the module docstring).
    """
    by_clip: dict[str, list[WaqfEventEntry]] = {}
    consumed: set[tuple[str, int]] = set()
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            clip = cand["clip_id"]
            if clip not in reviewed:
                continue
            key = (clip, cand["boundary_index"])
            override = overrides.get(key)
            if override is not None:
                consumed.add(key)
                verdict = override["verdict"]
                note = override.get("note", "")
            else:
                verdict = cand["predicted"]
                note = ""
            by_clip.setdefault(clip, []).append(
                WaqfEventEntry(
                    clip_id=clip,
                    audio_ref=cand["audio_ref"],
                    surah_ayah=cand["surah_ayah"],
                    boundary_index=cand["boundary_index"],
                    word_index=cand["word_index"],
                    start_s=cand["start_s"],
                    end_s=cand["end_s"],
                    predicted=cand["predicted"],
                    verdict=verdict,
                    note=note,
                )
            )
    stale = [overrides[k] for k in overrides if k not in consumed]
    return by_clip, stale


def _binary_counts(entries: list[WaqfEventEntry]) -> dict:
    """Binary waqf-vs-not scoring summary for one partition.

    Positives are ``verdict == "waqf"``; everything else is not-waqf. ``mid_word_closure``
    is reported only as a diagnostic tag on the not-waqf class (the hard-negative subset),
    never as a distinct outcome.
    """
    positives = sum(1 for e in entries if e.verdict == WAQF)
    negatives = len(entries) - positives
    closure_tag = sum(1 for e in entries if e.verdict == "mid_word_closure")
    return {
        "boundaries": len(entries),
        "waqf": positives,
        "not_waqf": negatives,
        "closure_tag": closure_tag,
    }


def freeze(
    by_clip: dict[str, list[WaqfEventEntry]],
    reviewed: set[str],
    test_fraction: float,
    seed: int,
) -> tuple[list[WaqfEventEntry], list[WaqfEventEntry], dict]:
    """Split the materialized ground truth reciter-disjoint into (calibration, test, report).

    The reviewed clips are partitioned at the reciter level via
    :func:`tadabur.waqf_partition.partition` (no training exclusion — none exists yet), and
    every clip's entries follow its reciter. The report records the reciter→partition
    assignment and ``must_exclude_reciters`` (the union) for the future training split.
    """
    rows = [{"clip_id": c} for c in sorted(reviewed)]
    _, _, split = partition(rows, excluded=set(), test_fraction=test_fraction, seed=seed)
    calibration_reciters = set(split["calibration_reciters"])
    test_reciters = set(split["test_reciters"])

    calibration: list[WaqfEventEntry] = []
    test: list[WaqfEventEntry] = []
    calibration_clips: list[str] = []
    test_clips: list[str] = []
    for clip in sorted(by_clip):
        entries = sorted(by_clip[clip], key=lambda e: e.boundary_index)
        if reciter_of(clip) in test_reciters:
            test.extend(entries)
            test_clips.append(clip)
        else:
            calibration.extend(entries)
            calibration_clips.append(clip)

    report = {
        "seed": seed,
        "test_fraction": test_fraction,
        "calibration_reciters": sorted(calibration_reciters),
        "test_reciters": sorted(test_reciters),
        "must_exclude_reciters": sorted(calibration_reciters | test_reciters),
        "calibration_clips": calibration_clips,
        "test_clips": test_clips,
        "counts": {
            "reviewed_clips": len(reviewed),
            "calibration_clips": len(calibration_clips),
            "test_clips": len(test_clips),
            "calibration_boundaries": len(calibration),
            "test_boundaries": len(test),
            "calibration_reciters": len(calibration_reciters),
            "test_reciters": len(test_reciters),
        },
        "binary": {
            "note": (
                "Gate is waqf vs not-waqf; verdict=='waqf' is the only positive. "
                "mid_word_closure is a diagnostic tag on not-waqf (best-effort, "
                "under-counts), not a separate class."
            ),
            "calibration": _binary_counts(calibration),
            "test": _binary_counts(test),
        },
    }
    return calibration, test, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Candidate baseline JSONL (tadabur.waqf_candidates output).")
    parser.add_argument("--events", type=Path, required=True,
                        help="Overrides-only event store (waqf_events.jsonl).")
    parser.add_argument("--reviewed", type=Path, required=True,
                        help="Reviewed-clip roster JSON (waqf_reviewed_clips.json).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to write the frozen partitions and report into.")
    parser.add_argument("--test-fraction", type=float, default=0.5,
                        help="Fraction of eligible reciters in the test set (default: 0.5).")
    parser.add_argument("--seed", type=int, default=0, help="Reciter-shuffle seed (default: 0).")
    args = parser.parse_args()

    reviewed = load_reviewed(args.reviewed)
    overrides = load_overrides(args.events)
    by_clip, stale = materialize(args.candidates, reviewed, overrides)

    missing = reviewed - set(by_clip)
    if missing:
        raise SystemExit(
            f"{len(missing)} reviewed clip(s) have no candidate boundaries "
            f"(candidate manifest out of date?): {sorted(missing)[:5]}"
        )

    calibration, test, report = freeze(by_clip, reviewed, args.test_fraction, args.seed)
    report["stale_overrides"] = stale

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_waqf_events(calibration, args.out_dir / "waqf_events.calibration.jsonl")
    write_waqf_events(test, args.out_dir / "waqf_events.test.jsonl")
    (args.out_dir / "waqf_partition.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    counts = report["counts"]
    binary = report["binary"]
    print(
        f"Froze {counts['reviewed_clips']} reviewed clips "
        f"({counts['calibration_boundaries'] + counts['test_boundaries']} boundaries): "
        f"calibration {counts['calibration_clips']} clips / {counts['calibration_reciters']} reciters, "
        f"test {counts['test_clips']} clips / {counts['test_reciters']} reciters; "
        f"{len(stale)} stale override(s) dropped; "
        f"{len(report['must_exclude_reciters'])} reciters to exclude from training."
    )
    print(
        "  binary waqf/not-waqf: "
        f"calibration {binary['calibration']['waqf']}/{binary['calibration']['not_waqf']} "
        f"(mwc tag {binary['calibration']['closure_tag']}), "
        f"test {binary['test']['waqf']}/{binary['test']['not_waqf']} "
        f"(mwc tag {binary['test']['closure_tag']})."
    )


if __name__ == "__main__":
    main()

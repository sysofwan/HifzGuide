"""Split the waqf candidate manifest into reciter-disjoint calibration + test sets (#30).

ADR-0004's event-level eval must be reported **once** on a held-out set, after tuning the
inference threshold on a *separate* one — and both must be free of any reciter the
phoneme+waqf fine-tune (D2/D3) trained on, or the numbers leak. This module makes those
two partitions from a candidate manifest (:mod:`tadabur.waqf_candidates`):

* **reciter-disjoint** — every candidate of a given reciter goes to exactly one side, so
  no reciter is split across calibration and test (a reciter's idiosyncratic waqf habits
  can't be memorised on one side and scored on the other).
* **disjoint from training** — reciters named by ``--train-reciters`` / ``--train-manifest``
  (the D2/D3 fine-tune set) are dropped from *both* partitions before the split.

The reciter of a candidate is read from its clip id (Tadabur names clips
``tadabur_spk<NNNN>_...``), so no schema change to :class:`~tadabur.waqf_event_sampler.WaqfCandidate`
is needed. The split is deterministic in ``--seed``: eligible reciters are shuffled once and
the first ``--test-fraction`` of them form the test set, the rest calibration. A
``--report`` JSON records the exact reciter→partition assignment so the frozen eval set is
auditable and reproducible for F1/F2.

Usage:
  python -m tadabur.waqf_partition --candidates candidates.jsonl \\
    --train-manifest passing_subset_full.jsonl \\
    --calibration candidates.calibration.jsonl --test candidates.test.jsonl \\
    [--test-fraction 0.5] [--seed 0] [--report partition.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# Tadabur clip ids embed the reciter as ``spk<NNNN>`` (== the manifest ``reciter_id``).
_RECITER_RE = re.compile(r"spk(\d+)")

DEFAULT_TEST_FRACTION = 0.5


def reciter_of(clip_id: str) -> int:
    """The reciter id encoded in a Tadabur clip id (the ``spk<NNNN>`` field)."""
    match = _RECITER_RE.search(clip_id)
    if match is None:
        raise ValueError(f"clip id {clip_id!r} has no spk<NNNN> reciter field")
    return int(match.group(1))


def read_rows(path: Path) -> list[dict]:
    """Read a candidate manifest (JSONL) as raw dict rows, in file order."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def train_reciters(
    reciter_ids: list[int] | None,
    manifest: Path | None,
) -> set[int]:
    """The set of training (D2/D3) reciter ids to exclude, from ids and/or a manifest.

    ``manifest`` is any JSONL carrying a ``reciter_id`` field per row (e.g. the fine-tune
    ``passing_subset``) or, failing that, an ``audio_filename`` / ``clip_id`` a reciter can
    be parsed from. Both sources are unioned so training reciters can be named directly or
    lifted straight from the training manifest.
    """
    reciters: set[int] = set(reciter_ids or [])
    if manifest is not None:
        for row in read_rows(manifest):
            if "reciter_id" in row:
                reciters.add(int(row["reciter_id"]))
            else:
                clip = row.get("clip_id") or row.get("audio_ref") or row.get("audio_filename", "")
                reciters.add(reciter_of(clip))
    return reciters


def partition(
    rows: list[dict],
    excluded: set[int],
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = 0,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    """Split candidate ``rows`` into (calibration, test, report), reciter-disjoint.

    Reciters in ``excluded`` (the training set) are dropped first. The remaining reciters
    are shuffled deterministically by ``seed`` and the leading ``test_fraction`` fraction
    assigned to test, the rest to calibration; every row follows its reciter, so the two
    partitions share no reciter and neither shares one with training. The report captures
    the exact assignment and the drop/kept tallies for freezing.
    """
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError(f"test_fraction must be in [0, 1], got {test_fraction}")

    reciters = sorted({reciter_of(row["clip_id"]) for row in rows})
    eligible = [r for r in reciters if r not in excluded]
    dropped = [r for r in reciters if r in excluded]

    shuffled = list(eligible)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    test_reciters = set(shuffled[:n_test])
    calibration_reciters = set(shuffled[n_test:])

    calibration = [r for r in rows if reciter_of(r["clip_id"]) in calibration_reciters]
    test = [r for r in rows if reciter_of(r["clip_id"]) in test_reciters]

    report: dict[str, object] = {
        "seed": seed,
        "test_fraction": test_fraction,
        "train_reciters": sorted(excluded),
        "dropped_train_reciters": dropped,
        "calibration_reciters": sorted(calibration_reciters),
        "test_reciters": sorted(test_reciters),
        "counts": {
            "total_rows": len(rows),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "dropped_train_rows": len(rows) - len(calibration) - len(test),
            "eligible_reciters": len(eligible),
            "calibration_reciters": len(calibration_reciters),
            "test_reciters": len(test_reciters),
        },
    }
    return calibration, test, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Candidate manifest JSONL (tadabur.waqf_candidates output).")
    parser.add_argument("--calibration", type=Path, required=True, help="Output calibration partition (JSONL).")
    parser.add_argument("--test", type=Path, required=True, help="Output test partition (JSONL).")
    parser.add_argument("--train-manifest", type=Path, default=None,
                        help="Training (D2/D3) manifest JSONL to lift excluded reciters from.")
    parser.add_argument("--train-reciters", type=int, nargs="*", default=None,
                        help="Explicit training reciter ids to exclude (unioned with --train-manifest).")
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION,
                        help=f"Fraction of eligible reciters in the test set (default: {DEFAULT_TEST_FRACTION}).")
    parser.add_argument("--seed", type=int, default=0, help="Reciter-shuffle seed (default: 0).")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional JSON path recording the reciter→partition assignment.")
    args = parser.parse_args()

    rows = read_rows(args.candidates)
    excluded = train_reciters(args.train_reciters, args.train_manifest)
    calibration, test, report = partition(rows, excluded, args.test_fraction, args.seed)

    _write_rows(calibration, args.calibration)
    _write_rows(test, args.test)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = report["counts"]
    print(
        f"Partitioned {counts['total_rows']} candidates: "
        f"{counts['calibration_rows']} calibration ({counts['calibration_reciters']} reciters), "
        f"{counts['test_rows']} test ({counts['test_reciters']} reciters); "
        f"dropped {counts['dropped_train_rows']} rows from {len(report['dropped_train_reciters'])} "
        f"training reciters."
    )


if __name__ == "__main__":
    main()

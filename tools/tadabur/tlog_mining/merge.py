"""Gather phase A's parts into one file per tranche (#263), and report the fallback rate.

Every part is a shard's records, in whatever order the batches appended them. The merge checks
them against the tranches' **gating** rows before writing anything: each gating row exactly
once, no record for a row that is not gating, no clip in two tranches. A partial run therefore
never looks complete — a missing row refuses the merge and no output file is written.

The output is ``<out_dir>/<tranche>.jsonl.gz`` (sorted by item key, the tlog writers' gzip
contract) and ``<out_dir>/meta.json``: the caller's provenance plus, per tranche, the row,
aligned and fallback counts and the fallback rate. A clip the aligner could not place is a
record with ``aligned: false`` and its ``reason``, so the rate counts listed clips, not dropped
ones.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.tlog_index import write_json, write_shard
from tadabur.tlog_tranches import GATING
from training.nonpro_decode import item_key

#: The tranches phase A commits, in the launcher's order.
PHASE_A_TRANCHES = ("look", "confirm.1", "confirm.2")


def merge_word_times(parts_dir: Path, out_dir: Path, *, manifests: Mapping[str, Sequence[Mapping]],
                     provenance: Mapping) -> dict:
    """Write one sorted file per tranche and ``meta.json``; returns the meta record.

    ``manifests`` is exactly :data:`PHASE_A_TRANCHES` mapped to their rows; only each tranche's
    ``gating`` rows are expected. A missing, repeated, foreign or cross-tranche key is a
    ``ValueError`` before any output file is written.
    """
    if set(manifests) != set(PHASE_A_TRANCHES):
        raise ValueError(f"the merge takes exactly the phase A tranches {list(PHASE_A_TRANCHES)}, "
                         f"not {sorted(manifests)}")
    expected: dict[str, set[str]] = {}
    owner: dict[str, str] = {}
    for tranche in PHASE_A_TRANCHES:
        keys = {item_key(row["audio_path"]) for row in manifests[tranche]
                if row["stratum"] == GATING}
        for key in keys:
            if key in owner:
                raise ValueError(f"{key} is a gating row of both {owner[key]} and {tranche}")
            owner[key] = tranche
        expected[tranche] = keys
    records = _read_parts(Path(parts_dir))
    foreign = sorted(set(records) - set(owner))
    if foreign:
        raise ValueError(f"{len(foreign)} records are for rows that are not gating rows of "
                         f"phase A (first {foreign[0]})")
    tranches = {tranche: _tranche_records(tranche, expected[tranche], records)
                for tranche in PHASE_A_TRANCHES}
    counts = {}
    for tranche in PHASE_A_TRANCHES:
        aligned = sum(1 for record in tranches[tranche] if record["aligned"])
        rows = len(tranches[tranche])
        counts[tranche] = {"rows": rows, "aligned": aligned, "fallback": rows - aligned,
                           "fallback_rate": (rows - aligned) / rows if rows else 0.0}
    out_dir = Path(out_dir)
    for tranche in PHASE_A_TRANCHES:
        write_shard(tranches[tranche], out_dir / f"{tranche}.jsonl.gz")
    meta = {**provenance, "tranches": counts}
    write_json(meta, out_dir / "meta.json")
    return meta


def _read_parts(parts_dir: Path) -> dict[str, dict]:
    """Every part's records by key; a key in two parts is a repeated clip."""
    records: dict[str, dict] = {}
    for path in sorted(Path(parts_dir).glob("word_times.*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number} is not a word-times record") from error
                key = record.get("key")
                if key in records:
                    raise ValueError(f"{key} appears twice: {path} repeats a record")
                records[key] = record
    return records


def _tranche_records(tranche: str, expected: set[str], records: Mapping[str, dict]) -> list[dict]:
    """``tranche``'s records in key order, refusing a missing or foreign key."""
    missing = sorted(expected - set(records))
    if missing:
        raise ValueError(f"{tranche}: {len(missing)} gating rows have no record "
                         f"(first {missing[0]}): the run is not complete")
    return [records[key] for key in sorted(expected)]

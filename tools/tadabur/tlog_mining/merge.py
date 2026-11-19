"""Gather a phase's parts into its committed files (#263): phase A's word times, phase B's decodes.

Phase A: every part is a shard's records, in whatever order the batches appended them. The merge
checks them against the tranches' **gating** rows before writing anything: each gating row
exactly once, no record for a row that is not gating, no clip in two tranches. A partial run
therefore never looks complete — a missing row refuses the merge and no output file is written.

The output is ``<out_dir>/<tranche>.jsonl.gz`` (sorted by item key, the tlog writers' gzip
contract) and ``<out_dir>/meta.json``: the caller's provenance plus, per tranche, the row,
aligned and fallback counts and the fallback rate. A clip the aligner could not place is a
record with ``aligned: false`` and its ``reason``, so the rate counts listed clips, not dropped
ones.

Phase B (:func:`merge_mined_decodes`): the same contract over the mined rows, pooled per model
across both tranches, with the un-phonetizable rows listed in the meta instead of decoded.

Phase C (:func:`merge_phase_c_decodes`): the round-3 weights' parts over the gating rows and
phase B's rows, written as ``tlog.mined.<model>.jsonl.gz`` beside
``tlog.gating.<model>.jsonl.gz`` — the mined file in phase B's shape, the gating file in round
2's — plus its **own** ``meta.<model>.json``. Phase B's committed ``meta.json`` is never
touched.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.tlog_index import read_jsonl, write_json, write_shard
from tadabur.tlog_tranches import GATING
from training.nonpro_decode import item_key

from .mined_decode import MINED_PREFIX
from .selection import PHASE_A_TRANCHES, PHASE_B_TRANCHES, PHASE_C_TRANCHES

#: Phase C's two pooled files: the mined rows under phase B's prefix, the gating rows under their
#: own, both named for the model that decoded them.
POOL_PREFIXES = {"mined": MINED_PREFIX, "gating": "tlog.gating"}


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


def merge_mined_decodes(parts_dir: Path, out_dir: Path, *,
                        rows: Mapping[str, Sequence[Mapping]],
                        unphonetizable: Mapping[str, Sequence[Mapping]],
                        provenance: Mapping, models: Sequence[str]) -> dict[str, int]:
    """Pool phase B's parts into one sorted file per model plus ``meta.json`` (#263).

    ``rows`` is each mined tranche's decodable rows and ``unphonetizable`` the rows listed instead
    of decoded (#248). Every model must hold each decodable row exactly once: a missing, repeated
    or foreign row is a ``ValueError`` before anything is written, so a partial run never looks
    complete. Each pooled line gains its ``tranche`` and the reference source its manifest row
    recorded (``label_source``), so the shared round-2 record shape stays untouched.

    ``meta.json`` carries the provenance, the per-model row counts, the row count per
    ``label_source``, the un-phonetizable listing and the producers' identity records — the
    weights that actually decoded, keyed as ``tlog.mined/<model>``.

    Phase B consumes none of phase A's committed ``word_times``: phase A reads the gating rows
    and phase B the reported rows, and a clip belongs to exactly one stratum, so the two sets
    share no key and no join exists. Phase A's timing is therefore never recomputed here — there
    is nothing here to recompute it for (brief Q2 read literally; the row sets are disjoint).
    """
    if set(rows) != set(PHASE_B_TRANCHES):
        raise ValueError(f"the mined merge takes exactly the phase B tranches "
                         f"{list(PHASE_B_TRANCHES)}, not {sorted(rows)}")
    manifests = {tranche: {item_key(row["audio_path"]): row for row in rows[tranche]}
                for tranche in PHASE_B_TRANCHES}
    _require_one_tranche_per_clip(manifests, PHASE_B_TRANCHES)
    pooled: dict[str, list[dict]] = {}
    for model in models:
        lines: list[dict] = []
        for tranche in PHASE_B_TRANCHES:
            found = _validated_lines(parts_dir, tranche, model, set(manifests[tranche]))
            lines.extend({**line, "tranche": tranche,
                          "label_source": manifests[tranche][line["key"]]["label_source"]}
                         for line in found)
        pooled[model] = sorted(lines, key=lambda line: line["key"])
    identities = _mined_identities(parts_dir, models)
    label_sources = Counter(row["label_source"] for tranche in PHASE_B_TRANCHES
                            for row in manifests[tranche].values())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, lines in pooled.items():
        write_shard(lines, out_dir / f"{MINED_PREFIX}.{model}.jsonl.gz")
    write_json({**provenance, "tranches": list(PHASE_B_TRANCHES), "models": list(models),
                "rows": {model: len(lines) for model, lines in pooled.items()},
                "label_sources": dict(label_sources),
                "unphonetizable": _listing(unphonetizable, PHASE_B_TRANCHES),
                "identities": identities},
               out_dir / "meta.json")
    return {model: len(lines) for model, lines in pooled.items()}


def merge_phase_c_decodes(parts_dir: Path, out_dir: Path, *,
                          rows: Mapping[str, Sequence[Mapping]],
                          unphonetizable: Mapping[str, Sequence[Mapping]],
                          provenance: Mapping, models: Sequence[str]) -> dict[str, dict]:
    """Pool phase C's parts into the round-3 weights' files and its own meta (#263).

    ``rows`` maps each phase C tranche to :func:`~tadabur.tlog_mining.selection.phase_c_rows`'
    decodable rows — the gating rows and phase B's — and ``unphonetizable`` to its listing. Phase
    C decodes one model: the round-3 chosen weights. Every decodable row must be held exactly
    once: a missing, repeated, foreign or listed row is a ``ValueError`` before anything is
    written, and so is a clip in both tranches.

    Writes ``tlog.mined.<model>.jsonl.gz`` (phase B's record shape: ``tranche`` and the
    manifest's ``label_source`` added), ``tlog.gating.<model>.jsonl.gz`` (round 2's confirm
    record shape plus ``spans`` — the gating stratum decodes both stores; the accepted shape,
    compared on ``key`` with ``spans`` ignored — sorted by key, each line gaining its
    ``tranche``) and ``meta.<model>.json``.
    The meta is never ``meta.json``: that is phase B's committed file, and phase C must leave it
    byte-identical. Returns each output's row count.
    """
    if set(rows) != set(PHASE_C_TRANCHES):
        raise ValueError(f"the phase C merge takes exactly the phase C tranches "
                         f"{list(PHASE_C_TRANCHES)}, not {sorted(rows)}")
    if len(models) != 1:
        raise ValueError(f"phase C decodes exactly one model, not {list(models)}")
    manifests = {tranche: {item_key(row["audio_path"]): row for row in rows[tranche]}
                 for tranche in PHASE_C_TRANCHES}
    _require_one_tranche_per_clip(manifests, PHASE_C_TRANCHES)
    pooled: dict[str, list[dict]] = {"mined": [], "gating": []}
    for tranche in PHASE_C_TRANCHES:
        for line in _validated_lines(parts_dir, tranche, models[0], set(manifests[tranche])):
            row = manifests[tranche][line["key"]]
            if row["stratum"] == GATING:
                pooled["gating"].append({**line, "tranche": tranche})
            else:
                pooled["mined"].append({**line, "tranche": tranche,
                                        "label_source": row["label_source"]})
    identities = _mined_identities(parts_dir, models)
    label_sources = Counter(row["label_source"] for tranche in PHASE_C_TRANCHES
                            for row in manifests[tranche].values()
                            if row["stratum"] != GATING)
    counts = {models[0]: {kind: len(lines) for kind, lines in pooled.items()}}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, prefix in POOL_PREFIXES.items():
        write_shard(sorted(pooled[kind], key=lambda line: line["key"]),
                    out_dir / f"{prefix}.{models[0]}.jsonl.gz")
    write_json({**provenance, "tranches": list(PHASE_C_TRANCHES), "models": list(models),
                "rows": counts, "label_sources": dict(label_sources),
                "unphonetizable": _listing(unphonetizable, PHASE_C_TRANCHES),
                "identities": identities},
               out_dir / f"meta.{models[0]}.json")
    return counts


def _require_one_tranche_per_clip(manifests: Mapping[str, Mapping[str, Mapping]],
                                  tranches: Sequence[str]) -> None:
    """Refuse a clip pooled under two tranches: it is decoded once, against one reference."""
    owner: dict[str, str] = {}
    for tranche in tranches:
        for key in manifests[tranche]:
            other = owner.setdefault(key, tranche)
            if other != tranche:
                raise ValueError(f"{key} is in both {other} and {tranche}: a clip is pooled once")


def _validated_lines(parts_dir: Path, tranche: str, model: str,
                     expected: set[str]) -> list[dict]:
    """``model``'s part lines for ``tranche``: every expected key exactly once, nothing else."""
    found = _tranche_lines(parts_dir, tranche, model)
    keys = [line["key"] for line in found]
    repeated = [key for key, count in Counter(keys).items() if count > 1]
    if repeated:
        raise ValueError(f"{model}: {repeated[0]} appears twice in {tranche}")
    missing = sorted(expected - set(keys))
    extra = sorted(set(keys) - expected)
    if missing or extra:
        raise ValueError(f"{model}: {len(missing)} {tranche} rows missing "
                         f"(first {missing[:1]}), {len(extra)} not in the selection "
                         f"(first {extra[:1]})")
    return found


def _listing(unphonetizable: Mapping[str, Sequence[Mapping]],
             tranches: Sequence[str]) -> list[dict]:
    """The un-phonetizable rows as the meta lists them: keyed, named, and never decoded."""
    return [{"tranche": tranche, "key": item_key(row["audio_path"]),
             "audio_filename": row["audio_path"], "assigned_label": row["assigned_label"],
             "stratum": row["stratum"]}
            for tranche in tranches for row in unphonetizable.get(tranche, ())]


def _tranche_lines(parts_dir: Path, tranche: str, model: str) -> list[dict]:
    """One tranche's and model's part records, every shard file in name order."""
    return [line for path in sorted(Path(parts_dir).glob(f"tlog.{tranche}.*.{model}.jsonl"))
            for line in read_jsonl(path)]


def _mined_identities(parts_dir: Path, models: Sequence[str]) -> dict[str, dict]:
    """The producers' identity records beside the parts, keyed ``tlog.mined/<model>``."""
    identities = {}
    for model in models:
        path = Path(parts_dir) / f"{model}.identity.json"
        if not path.exists():
            raise ValueError(f"no identity record for {model} in {parts_dir}: the parts do not "
                             f"say which weights decoded them")
        identities[f"{MINED_PREFIX}/{model}"] = json.loads(path.read_text(encoding="utf-8"))
    return identities


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
    """``tranche``'s records in key order, refusing a gating row with no record."""
    missing = sorted(expected - set(records))
    if missing:
        raise ValueError(f"{tranche}: {len(missing)} gating rows have no record "
                         f"(first {missing[0]}): the run is not complete")
    return [records[key] for key in sorted(expected)]

"""The candidate inventory of the tlog confirm.1/2 decodes (#300, #261 T1).

A no-listening count over the committed phase A-C decodes: 5,762 gating rows (5,742 after #248)
and 10,642 mined rows (the 26 phase B listed left out), each word graded by the four detectors
(``base``, ``c95``, ``r2_candidate_ema``, ``r3_candidate_ema``) under the two scorings the
harness records — the strict ``allowances_off`` of
``docs/tlog-mining-preregistration.md`` §3 and the shipped default. It counts where the
detectors flag words, of what kind, and how far each mined ``label_source`` can be trusted
against the gating rows before any of it reaches the owner's ears. It reads no human verdict and
no look, confirm.3/4 or train row.

Every input is pinned by SHA-256 (:data:`PINS`); a row outside confirm.1/2, an unpinned or
changed input, a listed row and an r3 identity that is not
:data:`training.aug_readout_spec.R3_CANDIDATE_SHA256` are refused. The r3 gating decode carries
an extra ``spans`` field, so the four detectors are joined on ``key`` and only ``checks`` are
read.

Outputs (committed):

- ``inventory/word_grades.jsonl.gz``: every word, both scorings, four detectors — or, if it
  exceeds :data:`WORD_GRADES_LIMIT_BYTES`, every word any detector flags plus
  ``inventory/clip_word_counts.jsonl.gz``.
- ``inventory/summary.json``: the pins, the harness build, the population, the counts, the
  per-``label_source`` trust and the catch-loss context.
- ``docs/tlog-mining-inventory.md``: rendered from ``summary.json``, never hand-edited.

Usage, from ``tools/``::

    python -m tadabur.tlog_mining.candidate_inventory             # grade and write everything
    python -m tadabur.tlog_mining.candidate_inventory regenerate  # counts/doc from the export
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path

from tadabur.tlog_index import write_json
from training.aug_readout_spec import R3_CANDIDATE_MODEL, R3_CANDIDATE_SHA256
from training.decode_evalset import cluster_bootstrap
from training.nonpro_decode import item_key
from training.tashkeel_eval import write_text_atomically
from training.word_flags import CAUSES

#: ``tools/tadabur``: the directory every pinned input name is relative to.
ROOT = Path(__file__).resolve().parents[1]
#: The committed artifact directory, beside the decodes it reads.
INVENTORY_DIR = ROOT / "tlog_mining" / "inventory"
#: The generated doc, at the repository root.
DOC_PATH = ROOT.parents[1] / "docs" / "tlog-mining-inventory.md"

BASE, C95, R2, R3 = "base", "c95", "r2_candidate_ema", "r3_candidate_ema"
DETECTORS = (BASE, C95, R2, R3)
STRICT, SHIPPED = "allowances_off", "shipped_default"
#: The scorings every word is recorded under: the strict one first, the shipped one beside it.
SCORINGS = (STRICT, SHIPPED)
GATING, MINED = "gating", "mined"
POPULATIONS = (GATING, MINED)
#: The only tranches whose decodes this readout reads.
CONFIRM_TRANCHES = ("confirm.1", "confirm.2")
#: The manifest stratum that makes a clip a gating row; every other row is a mined row.
GATING_STRATUM = "gating"

#: The complete word export's ceiling; over it, only flagged words are committed (#300).
WORD_GRADES_LIMIT_BYTES = 25_000_000
#: The group-clustered trust intervals' seed and draws: fixed, so a rerun is byte-identical.
TRUST_BOOTSTRAP_SEED = 300_000
TRUST_BOOTSTRAP_ITERATIONS = 4000
#: How many group keys the concentration table names.
CONCENTRATION_TOP = 20

#: Every input this readout reads, pinned by its SHA-256. A name absent here is refused, and so
#: are bytes that do not hash to the pin. The first eight are the pre-registration's §2.
PINS = {
    "consonant2_readout/decodes/tlog.confirm.base.jsonl.gz":
        "91b66205beb25e36044558e91cd96876b6e439c8d33d24ca4165cb896abaf172",
    "consonant2_readout/decodes/tlog.confirm.c95.jsonl.gz":
        "644a1bd1a3e25ceb320ff32e3d3125729e3d3dc14f5270a82627772d0418a5fc",
    "consonant2_readout/decodes/tlog.confirm.r2_candidate_ema.jsonl.gz":
        "4de59c3c8dbceb630a854fb00982b16874885414839c7bc63919c8bc53d1c46a",
    "tlog_mining/decodes/tlog.gating.r3_candidate_ema.jsonl.gz":
        "b6f383bb825502f1e99b006d4eade61bdc26bde90273692f1b2ea7c8d706be80",
    "tlog_mining/decodes/tlog.mined.base.jsonl.gz":
        "a31c4f0535c0bd3a4cd8ff6e7d1b6578fb03e2757e0530a7263a1b6784e49448",
    "tlog_mining/decodes/tlog.mined.c95.jsonl.gz":
        "b9b1e3526a77201a91725943062d3bf4faf15a30bf1b1bc2622321cff0444607",
    "tlog_mining/decodes/tlog.mined.r2_candidate_ema.jsonl.gz":
        "4630d0902c96c884f01357432036522d5d64ebc636f7ca85ffe157f6c9c12150",
    "tlog_mining/decodes/tlog.mined.r3_candidate_ema.jsonl.gz":
        "427e76e7f0c5637ac6c9954ebbbd34f41c5e89081e64b847a7a561c5abdf4066",
    "consonant2_readout/report.json":
        "2149ea0f2f5dc4ba125f718f89f6a942fd2c30a39850f1d259e2823e338b1117",
    "tlog_mining/decodes/meta.r3_candidate_ema.json":
        "abea387728cde8043beabdb5dbab3026f8162e50e34c0c0ddc149352a713fcb0",
    "consonant2_readout/catch_loss/result.json":
        "1babf43ce3bfd650548440834f5163d8cca40a81cf072dcc600336f8a3288ea3",
    "consonant3_readout/catch_loss/result.json":
        "9f23f6cb5ecec789fb30e0417e9bf3be41b6c78f83556dfc4e904e1641b1088d",
    "tlog/tranches/confirm.1.jsonl":
        "b46b96209546a3d67952797a2e6628fe7280ad13bf25ccc68d763a7a743ac7a4",
    "tlog/tranches/confirm.2.jsonl":
        "c7521a584da7321c4bf13e5cc99b4dae5ae476d7c484465b7b7af0a918d44493",
}


# --- inputs: the pinned decodes, manifests, report and audits -----------------------------------


def decode_name(population: str, model: str) -> str:
    """The committed decode file of one population's detector."""
    if population == GATING:
        if model == R3:
            return "tlog_mining/decodes/tlog.gating.r3_candidate_ema.jsonl.gz"
        return f"consonant2_readout/decodes/tlog.confirm.{model}.jsonl.gz"
    if population == MINED:
        return f"tlog_mining/decodes/tlog.mined.{model}.jsonl.gz"
    raise ValueError(f"{population!r} is not a population: {list(POPULATIONS)}")


def read_pinned(name: str, root: Path = ROOT) -> bytes:
    """The pinned input ``name``'s bytes: refused unless the name is pinned and the bytes hash
    to the pin."""
    if name not in PINS:
        raise ValueError(f"{name} is not pinned: every input this readout reads is listed in "
                         f"candidate_inventory.PINS")
    data = (Path(root) / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PINS[name]:
        raise ValueError(f"{name} is not the pinned SHA-256 {PINS[name]}: it is {digest}, so "
                         f"the readout refuses to count it")
    return data


def _gzip_lines(data: bytes, name: str) -> dict[str, dict]:
    """A decode file's records by key, refusing a corrupt line and a repeated key."""
    try:
        text = gzip.decompress(data).decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is not a gzipped UTF-8 decode file") from error
    lines: dict[str, dict] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            line = json.loads(raw)
            key = line["key"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{name}:{number} is not a decode record") from error
        if key in lines:
            raise ValueError(f"{name}:{number} repeats the key {key}")
        lines[key] = line
    return lines


def _decode_lines(root: Path = ROOT) -> dict[str, dict[str, dict[str, dict]]]:
    """Every decode line, by population, model and key: each input read through
    :func:`read_pinned` and refused when it holds a row outside confirm.1/2 or a key set the
    population's base did not decode."""
    decodes: dict[str, dict[str, dict[str, dict]]] = {}
    for population in POPULATIONS:
        decodes[population] = {}
        base_keys = None
        for model in DETECTORS:
            name = decode_name(population, model)
            lines = _gzip_lines(read_pinned(name, root), name)
            for key, line in lines.items():
                tranche = line.get("tranche")
                if tranche not in CONFIRM_TRANCHES:
                    raise ValueError(f"{name}: {key} is a {tranche!r} row, not a confirm.1/"
                                     f"confirm.2 row")
            if base_keys is None:
                base_keys = set(lines)
            elif set(lines) != base_keys:
                raise ValueError(f"{name}: model {model} decoded {len(lines)} clips, not the "
                                 f"base's {len(base_keys)}")
            decodes[population][model] = lines
    return decodes


def load_decodes(root: Path = ROOT) -> dict[str, dict[str, dict[str, Sequence[dict]]]]:
    """Each population's detector decodes by key, holding only the recorded ``checks`` (the r3
    gating decode's ``spans`` play no part in the join)."""
    return {population: {model: {key: line["checks"] for key, line in lines.items()}
                         for model, lines in models.items()}
            for population, models in _decode_lines(root).items()}


def excluded_248(root: Path = ROOT) -> set[str]:
    """The 20 gating clips round 2's report excluded as unphonetizable (#248)."""
    report = json.loads(read_pinned("consonant2_readout/report.json", root))
    return set(report["coverage"]["graded"]["excluded_clips"]["keys"])


def _r3_meta(root: Path = ROOT) -> dict:
    meta = json.loads(read_pinned("tlog_mining/decodes/meta.r3_candidate_ema.json", root))
    require_r3_identity(meta)
    return meta


def listed_unphonetizable(root: Path = ROOT) -> set[str]:
    """The 26 rows phase C listed because their ``assigned_label`` cannot be phonetized."""
    return {row["key"] for row in _r3_meta(root)["unphonetizable"]}


def require_r3_identity(meta: Mapping) -> None:
    """Refuse a phase C meta whose r3 checkpoint is not the round's chosen one."""
    entry = f"tlog.mined/{R3_CANDIDATE_MODEL}"
    identities = meta.get("identities") if isinstance(meta, Mapping) else None
    pinned = identities.get(entry) if isinstance(identities, Mapping) else None
    identity = pinned.get("identity") if isinstance(pinned, Mapping) else None
    checkpoint = identity.get("checkpoint_sha256") if isinstance(identity, Mapping) else None
    if checkpoint is None:
        raise ValueError(f"the r3 meta names no {entry!r} identity with a checkpoint: the "
                         f"readout grades the round's chosen {R3_CANDIDATE_MODEL} only")
    if checkpoint != R3_CANDIDATE_SHA256:
        raise ValueError(f"the r3 meta's checkpoint is {checkpoint}, not the pinned "
                         f"{R3_CANDIDATE_SHA256}: the readout grades the weights round 3 "
                         f"chose only")


def _manifest_rows(root: Path, tranche: str) -> list[dict]:
    """A confirm tranche's committed manifest rows, read at its pinned bytes."""
    name = f"tlog/tranches/{tranche}.jsonl"
    try:
        text = read_pinned(name, root).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not UTF-8") from error
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- the population -----------------------------------------------------------------------------


def clip_table(rows_by_tranche: Mapping[str, Sequence[Mapping]], *, excluded: Iterable[str],
               listed: Iterable[str]) -> dict[str, dict]:
    """One record per clip the decodes hold: its population (``gating`` for the gating stratum,
    else ``mined``), tranche, stratum, ``label_source`` and ``group_key``. The #248 exclusions
    drop gating clips; a listed row is refused."""
    listed = set(listed)
    excluded = set(excluded)
    clips: dict[str, dict] = {}
    for tranche, rows in rows_by_tranche.items():
        if tranche not in CONFIRM_TRANCHES:
            raise ValueError(f"{tranche!r} is not a confirm.1/confirm.2 tranche: the readout "
                             f"reads the gating and mined rows of {list(CONFIRM_TRANCHES)} only")
        for row in rows:
            key = item_key(row["audio_path"])
            if key in listed:
                raise ValueError(f"{key} is one of the rows phase C listed as unphonetizable: "
                                 f"the readout leaves every listed row out")
            population = GATING if row["stratum"] == GATING_STRATUM else MINED
            if population == GATING and key in excluded:
                continue
            clips[key] = {"population": population, "tranche": tranche,
                          "stratum": row["stratum"], "label_source": row["label_source"],
                          "group_key": row["group_key"]}
    return clips


def population(rows: Sequence[Mapping], *, clips: Mapping[str, Mapping] | None = None) -> dict:
    """Clips, group keys and words, per population and over all. ``clips`` is the pool's clip
    table (:func:`clip_table`): the clip and group-key counts are its, because a clip whose
    audio the harness could not place (a partial recitation, a different ayah) contributes no
    word row but is still part of the pool. Without it the counts come from the rows alone."""
    counts = {}
    for name in (*POPULATIONS, "all"):
        chosen = rows if name == "all" else [row for row in rows if row["population"] == name]
        if clips is None:
            keys = {row["key"] for row in chosen}
            groups = {row["group_key"] for row in chosen}
        else:
            pool = [(key, clip) for key, clip in clips.items()
                    if name == "all" or clip["population"] == name]
            keys = {key for key, _ in pool}
            groups = {clip["group_key"] for _, clip in pool}
        counts[name] = {"clips": len(keys), "group_keys": len(groups), "words": len(chosen)}
    return counts


# --- grades -> word rows ------------------------------------------------------------------------


def word_rows(grades: Iterable, clips: Mapping[str, Mapping]) -> list[dict]:
    """One row per word the four detectors all graded under ``allowances_off``, paired on
    ``(key, position)`` and sorted the same way: the clip's fields, ``key``, ``position``,
    ``word_index`` and, per scoring, per detector, its flag, causes and consonant pairs."""
    by_word: dict[tuple[str, int], dict[tuple[str, str], object]] = {}
    for grade in grades:
        key = grade.item.key
        if key not in clips:
            raise ValueError(f"{key}: the grade's clip is not in the clip table")
        by_word.setdefault((key, grade.position), {})[(grade.model, grade.scoring)] = grade
    rows = []
    for (key, position), graded in sorted(by_word.items()):
        strict = [graded.get((model, STRICT)) for model in DETECTORS]
        if any(grade is None or not grade.graded for grade in strict):
            continue
        indexes = {grade.word_index for grade in graded.values()}
        if len(indexes) != 1:
            raise ValueError(f"{key} word {position}: the detectors graded different words "
                             f"{sorted(indexes)}")
        row = {**clips[key], "key": key, "position": position, "word_index": indexes.pop()}
        for scoring in SCORINGS:
            row[scoring] = {model: _reason_of(graded.get((model, scoring)))
                            for model in DETECTORS}
        rows.append(row)
    return rows


def _reason_of(grade) -> dict:
    """A grade's flag as the export records it: unflagged unless it graded and flagged."""
    if grade is None or not grade.graded or not grade.flagged:
        return {"flagged": False, "causes": [], "consonants": []}
    return {"flagged": True, "causes": list(grade.reason.causes),
            "consonants": [list(pair) for pair in grade.reason.consonants]}


# --- the counts ---------------------------------------------------------------------------------


def _flagged(row: Mapping, scoring: str) -> bool:
    return any(row[scoring][model]["flagged"] for model in DETECTORS)


def _level(row: Mapping, scoring: str) -> int:
    return sum(1 for model in DETECTORS if row[scoring][model]["flagged"])


def _pattern(row: Mapping, scoring: str) -> str:
    flagging = [model for model in DETECTORS if row[scoring][model]["flagged"]]
    return "+".join(flagging) if flagging else "none"


def _table(rows: Sequence[Mapping], scoring: str, name_of: Callable[[Mapping], str]) -> dict:
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = table.setdefault(name_of(row), {"words": 0, "flagged_words": 0})
        entry["words"] += 1
        entry["flagged_words"] += 1 if _flagged(row, scoring) else 0
    return {name: table[name] for name in sorted(table)}


def counts(rows: Sequence[Mapping], scoring: str) -> dict:
    """The inventory's counts for one scoring: agreement levels, all sixteen detector
    patterns, causes and substitution pairs (a word counts once per cause or pair over every
    detector that flags it), the tranche/stratum/label-source splits and the concentration."""
    levels = Counter({str(level): 0 for level in range(len(DETECTORS) + 1)})
    patterns = Counter({name: 0 for name in
                        ["none"] + ["+".join(combo) for n in range(1, len(DETECTORS) + 1)
                                    for combo in combinations(DETECTORS, n)]})
    causes: Counter = Counter()
    pairs: Counter = Counter()
    for row in rows:
        levels[str(_level(row, scoring))] += 1
        patterns[_pattern(row, scoring)] += 1
        if not _flagged(row, scoring):
            continue
        found: set[str] = set()
        seen_pairs: set[str] = set()
        for model in DETECTORS:
            entry = row[scoring][model]
            if not entry["flagged"]:
                continue
            found.update(entry["causes"])
            seen_pairs.update(pair[1] for pair in entry["consonants"])
        unknown = found - set(CAUSES)
        if unknown:
            raise ValueError(f"{row['key']} word {row['position']}: unknown causes {unknown}")
        causes.update(found)
        pairs.update(seen_pairs)
    flagged_by_group: Counter = Counter()
    for row in rows:
        if _flagged(row, scoring):
            flagged_by_group[row["group_key"]] += 1
    top = sorted(flagged_by_group.items(), key=lambda item: (-item[1], item[0]))
    return {
        "scoring": scoring,
        "words": len(rows),
        "flagged_words": sum(1 for row in rows if _flagged(row, scoring)),
        "by_level": {level: levels[level] for level in sorted(levels)},
        "by_pattern": {name: patterns[name] for name in
                       sorted(patterns, key=_pattern_order)},
        "by_cause": {cause: causes[cause] for cause in CAUSES},
        "substitutions": [{"pair": pair, "words": words}
                          for pair, words in sorted(pairs.items(),
                                                    key=lambda item: (-item[1], item[0]))],
        "by_tranche": _table(rows, scoring, lambda row: row["tranche"]),
        "by_stratum": _table(rows, scoring, lambda row: row["stratum"]),
        "by_label_source": _table(rows, scoring,
                                  lambda row: f"{row['population']}:{row['label_source']}"),
        "concentration": {"flagged_words": sum(flagged_by_group.values()),
                          "group_keys": len(flagged_by_group),
                          "top": [{"group_key": group, "flagged_words": words}
                                  for group, words in top[:CONCENTRATION_TOP]]},
    }


def _pattern_order(name: str) -> tuple[int, int, str]:
    """``none`` first, then the smaller patterns, then detector order (the table's order)."""
    if name == "none":
        return (0, 0, "")
    flagging = name.split("+")
    return (len(flagging), 0, f"{DETECTORS.index(flagging[0]):02d}")


# --- reference trust ----------------------------------------------------------------------------


def _aggregates(rows: Sequence[Mapping]) -> list[dict]:
    """Each ``group_key``'s strict-scoring words and flagged words, sorted by key: the cluster
    rows the trust interval resamples."""
    groups: dict[str, dict] = {}
    for row in rows:
        group = groups.setdefault(row["group_key"], {"cluster": row["group_key"], "words": 0,
                                                     "flagged": 0})
        group["words"] += 1
        group["flagged"] += 1 if _flagged(row, STRICT) else 0
    return [groups[name] for name in sorted(groups)]


def _rate(clusters: Sequence[Mapping]) -> float:
    words = sum(cluster["words"] for cluster in clusters)
    return sum(cluster["flagged"] for cluster in clusters) / words if words else 0.0


def _interval(clusters: Sequence[Mapping]) -> list[float]:
    """The ``group_key``-clustered 95% percentile interval of :func:`_rate`."""
    if not clusters:
        return [0.0, 0.0]
    low, high = cluster_bootstrap(list(clusters), lambda cluster: cluster["cluster"], _rate,
                                  seed=TRUST_BOOTSTRAP_SEED,
                                  iterations=TRUST_BOOTSTRAP_ITERATIONS)
    return [low, high]


def _difference_interval(gating: Sequence[Mapping], source: Sequence[Mapping]) -> list[float]:
    """The interval of ``rate(source) - rate(gating)``, resampling the **original** group keys:
    a key shared by both populations is drawn once and its gating and mined words contribute to
    both rates together. Population membership is not an independent sampling unit, so giving
    the two a cluster of their own would treat one group's two observations as independent."""
    combined = [{**cluster, "population": GATING} for cluster in gating] + \
               [{**cluster, "population": MINED} for cluster in source]

    def statistic(sample):
        totals = {GATING: [0, 0], MINED: [0, 0]}
        for cluster in sample:
            total = totals[cluster["population"]]
            total[0] += cluster["words"]
            total[1] += cluster["flagged"]
        return (totals[MINED][1] / totals[MINED][0] if totals[MINED][0] else 0.0) - \
               (totals[GATING][1] / totals[GATING][0] if totals[GATING][0] else 0.0)

    low, high = cluster_bootstrap(combined, lambda cluster: cluster["cluster"], statistic,
                                  seed=TRUST_BOOTSTRAP_SEED,
                                  iterations=TRUST_BOOTSTRAP_ITERATIONS)
    return [low, high]


def label_source_trust(rows: Sequence[Mapping]) -> dict:
    """The strict-scoring flag rate per ``label_source``, against the gating rows (all ``both``)
    as the baseline: each mined source's rate, its ``group_key``-clustered interval and its
    difference from gating with that difference's interval."""
    gating = _aggregates([row for row in rows if row["population"] == GATING])
    baseline = _rate(gating)
    trust = {GATING: {"words": sum(c["words"] for c in gating),
                      "flagged_words": sum(c["flagged"] for c in gating),
                      "rate": baseline, "ci95": _interval(gating)}}
    mined = [row for row in rows if row["population"] == MINED]
    for source in sorted({row["label_source"] for row in mined}):
        clusters = _aggregates([row for row in mined if row["label_source"] == source])
        rate = _rate(clusters)
        trust[source] = {"words": sum(c["words"] for c in clusters),
                         "flagged_words": sum(c["flagged"] for c in clusters),
                         "rate": rate, "ci95": _interval(clusters),
                         "difference_from_gating": rate - baseline,
                         "difference_ci95": _difference_interval(gating, clusters)}
    return trust


# --- the prior evidence, as context only --------------------------------------------------------


def catch_loss_context(root: Path = ROOT) -> dict:
    """The committed round-2 and round-3 catch-loss audits' real-mistake shares by cause: each
    round's ``removed`` and ``added`` words summed, with the round's real total. Context only:
    their strata differ from this inventory's, so no yield is projected from them."""
    context = {}
    for round_name, name in (("round2", "consonant2_readout/catch_loss/result.json"),
                             ("round3", "consonant3_readout/catch_loss/result.json")):
        result = json.loads(read_pinned(name, root))
        by_cause = {}
        for cause in CAUSES:
            sides = [result["by_cause"][side].get(cause, {}) for side in ("removed", "added")]
            by_cause[cause] = {"real": sum(side.get("real", 0) for side in sides),
                               "decisive": sum(side.get("decisive", 0) for side in sides)}
        context[round_name] = {"real_total": result["removed"]["real"]
                               + result["added"]["real"], "by_cause": by_cause}
    return context


# --- the artifacts ------------------------------------------------------------------------------


def _jsonl_gz(rows: Iterable[Mapping]) -> bytes:
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return gzip.compress(body.encode("utf-8"), mtime=0)


def _write_bytes(path: Path, data: bytes) -> None:
    """Bytes to a sibling temporary file, then over ``path``: a run never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _any_flagged(row: Mapping) -> bool:
    return any(row[scoring][model]["flagged"] for scoring in SCORINGS for model in DETECTORS)


def write_word_grades(rows: Sequence[Mapping], directory: Path, *,
                      limit_bytes: int = WORD_GRADES_LIMIT_BYTES) -> dict:
    """The per-word export: every word with both scorings, or — when that exceeds
    ``limit_bytes`` — every word any detector flags plus each clip's word count. Returns what
    was written and its size."""
    directory = Path(directory)
    ordered = sorted(rows, key=lambda row: (row["key"], row["position"]))
    complete = _jsonl_gz(ordered)
    grades_path = directory / "word_grades.jsonl.gz"
    counts_path = directory / "clip_word_counts.jsonl.gz"
    if len(complete) <= limit_bytes:
        _write_bytes(grades_path, complete)
        counts_path.unlink(missing_ok=True)
        return _export_record(grades_path, complete, complete=True)
    written = _jsonl_gz([row for row in ordered if _any_flagged(row)])
    _write_bytes(grades_path, written)
    words: Counter = Counter(row["key"] for row in ordered)
    _write_bytes(counts_path, _jsonl_gz([{"key": key, "words": number}
                                         for key, number in sorted(words.items())]))
    return _export_record(grades_path, written, complete=False,
                          clip_word_counts_file=counts_path.name)


def _export_record(path: Path, data: bytes, **extra) -> dict:
    """What the summary records of the export it just wrote: its name, size and SHA-256. The
    hash is what :func:`regenerate` authenticates the bytes by; size and row count alone do not
    catch a changed flag."""
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(data).hexdigest(), **extra}


def build_summary(rows: Sequence[Mapping], *, inputs: Mapping[str, str],
                  harness: Mapping, exclusions: Mapping, catch_loss: Mapping,
                  word_grades: Mapping, clips: Mapping[str, Mapping] | None = None) -> dict:
    """``summary.json``: the pins, the harness, the population, the two scorings' counts, the
    label-source trust, the catch-loss context and the word export's record. ``clips`` is the
    pool's clip table, so the population's clip and group-key counts include a clip that no
    word row could be graded from."""
    return {
        "inputs": dict(inputs),
        "harness": dict(harness),
        "population": population(rows, clips=clips),
        "exclusions": dict(exclusions),
        "strict": counts(rows, STRICT),
        "shipped_default": counts(rows, SHIPPED),
        "label_source_trust": label_source_trust(rows),
        "catch_loss_context": dict(catch_loss),
        "word_grades": dict(word_grades),
    }


def render(summary: Mapping) -> str:
    """The doc from ``summary.json``: a plain-language summary first, then every count."""
    strict, shipped = summary["strict"], summary["shipped_default"]
    total = summary["population"]["all"]
    trust = summary["label_source_trust"]
    lines = [
        "# Tlog candidate inventory",
        "",
        "Generated from `tools/tadabur/tlog_mining/inventory/summary.json` by "
        "`python -m tadabur.tlog_mining.candidate_inventory`; never hand-edit it.",
        "",
        "## Summary",
        "",
    ]
    flagged = strict["flagged_words"]
    share = f" ({flagged / strict['words']:.1%})" if strict["words"] else ""
    lines += [
        f"- {total['clips']} clips in {total['group_keys']} group keys carry {total['words']} "
        f"words all four detectors graded; a clip whose audio they could not place (a partial "
        f"recitation, a different ayah) has no word row but stays in the pool.",
        f"- {flagged} words{share} are flagged by at least one of the four detectors under the "
        f"strict `allowances_off` scoring; the shipped default flags "
        f"{shipped['flagged_words']}.",
        f"- All four detectors agree on {strict['by_level']['4']} words; "
        f"{strict['by_level']['0']} words are flagged by none.",
        f"- The gating rows flag {_format_rate(trust['gating'])} of their words; the mined "
        f"label sources flag "
        + "; ".join(f"`{name}` {_format_rate(entry)}"
                    for name, entry in sorted(trust.items()) if name != GATING) + ".",
        f"- Prior evidence only: the round-2 catch-loss audit found "
        f"{summary['catch_loss_context']['round2']['real_total']} real mistakes and round 3's "
        f"{summary['catch_loss_context']['round3']['real_total']}, on strata that differ from "
        f"this pool, so no yield is projected.",
        "",
        "## Word export",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    lines += [f"| `{field}` | {value} |" for field, value in
              sorted(summary["word_grades"].items())]
    lines += [""]
    if not summary["word_grades"].get("complete", True):
        lines += ["The complete export exceeded the 25 MB ceiling, so only the words any "
                  "detector flags are committed; `clip_word_counts.jsonl.gz` holds every "
                  "clip's word count.", ""]
    lines += [
        "## Inputs",
        "",
        "| input | SHA-256 |",
        "| --- | --- |",
    ]
    lines += [f"| `{name}` | `{digest}` |" for name, digest in sorted(summary["inputs"].items())]
    lines += ["", "Harness:", "", "| field | value |", "| --- | --- |"]
    lines += [f"| `{field}` | `{value}` |" for field, value in sorted(summary["harness"].items())]
    lines += [
        "",
        "## Population",
        "",
        "| population | clips | group keys | words |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {name} | {entry['clips']} | {entry['group_keys']} | {entry['words']} |"
              for name, entry in sorted(summary["population"].items())]
    lines += ["", "## Exclusions", "", "| exclusion | clip rows |", "| --- | ---: |"]
    lines += [f"| {name} | {number} |" for name, number in sorted(summary["exclusions"].items())]
    for title, scoring in (("Agreement (allowances_off)", strict),
                           ("Shipped default", shipped)):
        lines += ["", f"## {title}", "", "| detectors flagging | words |", "| --- | ---: |"]
        lines += [f"| {level} | {number} |"
                  for level, number in sorted(scoring["by_level"].items())]
        lines += ["", "| detector pattern | words |", "| --- | ---: |"]
        lines += [f"| {pattern} | {scoring['by_pattern'][pattern]} |"
                  for pattern in sorted(scoring["by_pattern"], key=_pattern_order)]
        lines += ["", "| cause | words |", "| --- | ---: |"]
        lines += [f"| {cause} | {scoring['by_cause'][cause]} |" for cause in CAUSES]
        lines += ["", "| substitution | words |", "| --- | ---: |"]
        lines += ([f"| {pair['pair']} | {pair['words']} |" for pair in
                   scoring["substitutions"]] or ["| (none) | 0 |"])
        for name, table in (("tranche", "by_tranche"), ("stratum", "by_stratum"),
                            ("label source", "by_label_source")):
            lines += ["", f"| {name} | words | flagged words |", "| --- | ---: | ---: |"]
            lines += [f"| {key} | {entry['words']} | {entry['flagged_words']} |"
                      for key, entry in sorted(scoring[table].items())]
        concentration = scoring["concentration"]
        lines += ["", f"Concentration: {concentration['flagged_words']} flagged words in "
                      f"{concentration['group_keys']} group keys.", "",
                  "| group key | flagged words |", "| --- | ---: |"]
        lines += ([f"| {entry['group_key']} | {entry['flagged_words']} |"
                   for entry in concentration["top"]] or ["| (none) | 0 |"])
    lines += [
        "",
        "## Reference trust",
        "",
        "The strict-scoring flag rate per label source, against the gating rows (all `both`) as "
        "the baseline; intervals resample whole group keys.",
        "",
        "| label source | words | flagged words | rate | 95% interval | vs gating | difference "
        "95% interval |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, entry in sorted(trust.items()):
        difference = ("" if "difference_from_gating" not in entry else
                      f"{entry['difference_from_gating']:+.3f}")
        interval = ("" if "difference_ci95" not in entry else
                    f"[{entry['difference_ci95'][0]:+.3f}, {entry['difference_ci95'][1]:+.3f}]")
        lines.append(f"| {name} | {entry['words']} | {entry['flagged_words']} | "
                     f"{entry['rate']:.3f} | [{entry['ci95'][0]:.3f}, {entry['ci95'][1]:.3f}] "
                     f"| {difference} | {interval} |")
    lines += ["", "## Prior evidence: the catch-loss audits", "",
              "Context only: these audits read the look tranche, on strata that differ from "
              "this pool.", ""]
    for round_name, context in sorted(summary["catch_loss_context"].items()):
        lines += [f"### {round_name}", "",
                  f"Real mistakes: {context['real_total']}.", "",
                  "| cause | real | decisive |", "| --- | ---: | ---: |"]
        for cause in CAUSES:
            entry = context["by_cause"].get(cause, {"real": 0, "decisive": 0})
            lines.append(f"| {cause} | {entry['real']} | {entry['decisive']} |")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _format_rate(entry: Mapping) -> str:
    return f"{entry['rate']:.1%} ({entry['flagged_words']}/{entry['words']})"


def write_inventory(rows: Sequence[Mapping], *, inputs: Mapping[str, str],
                    harness: Mapping, exclusions: Mapping, catch_loss: Mapping,
                    out_dir: Path = INVENTORY_DIR, doc_path: Path = DOC_PATH,
                    clips: Mapping[str, Mapping] | None = None) -> dict:
    """Write the word export, ``summary.json`` and the doc rendered from it; returns the
    summary."""
    out_dir = Path(out_dir)
    report = write_word_grades(rows, out_dir)
    summary = build_summary(rows, inputs=inputs, harness=harness, exclusions=exclusions,
                            catch_loss=catch_loss, word_grades=report, clips=clips)
    write_json(summary, out_dir / "summary.json")
    write_text_atomically(Path(doc_path), render(summary))
    return summary


# --- the real run -------------------------------------------------------------------------------


def _progress(message: str, since: float | None = None) -> float:
    """A run's phase marker on stderr: the phonetizer and the harness take minutes, and a run
    that prints nothing for that long is indistinguishable from a hung one."""
    now = time.time()
    spent = "" if since is None else f" after {now - since:.0f}s"
    print(f"[inventory] {message}{spent}", file=sys.stderr, flush=True)
    return now


def build(*, harness=None) -> dict:
    """Grade the committed confirm.1/2 decodes with the four detectors and write the committed
    inventory. Needs the Mac harness only: no GPU and no audio."""
    from tadabur.waqf_segments import hafs_segment_reference
    from training.aug_readout_gate import harness_identity
    from training.external_corpus import app_scorings, word_grades
    from training.muraja_policy import require_harness
    from training.tlog_eval import gating_items, reported_item_sets

    harness = require_harness() if harness is None else harness
    clock = _progress("build started")
    decodes = _decode_lines(ROOT)
    excluded = excluded_248(ROOT)
    listed = listed_unphonetizable(ROOT)
    manifests = {tranche: _manifest_rows(ROOT, tranche) for tranche in CONFIRM_TRANCHES}
    decoded = {**decodes[GATING][BASE], **decodes[MINED][BASE]}
    rows_by_tranche = {tranche: [row for row in rows
                                 if item_key(row["audio_path"]) in decoded]
                       for tranche, rows in manifests.items()}
    clips = clip_table(rows_by_tranche, excluded=excluded, listed=listed)
    clock = _progress(f"inputs read: {len(clips)} clips", clock)

    def chosen(population: str) -> list[dict]:
        return [row for rows in rows_by_tranche.values() for row in rows
                if item_key(row["audio_path"]) in clips
                and clips[item_key(row["audio_path"])]["population"] == population]

    lines = {**decodes[GATING][BASE], **decodes[MINED][BASE]}
    reference_of = hafs_segment_reference()
    gating, unsupported = gating_items(chosen(GATING), lines, reference_of=reference_of)
    reported = reported_item_sets(chosen(MINED), lines, reference_of=reference_of,
                                  label_of=_assigned_label)
    if unsupported:
        raise ValueError(f"{len(unsupported)} rows the readout expected to grade have no item: "
                         f"{sorted(unsupported)[:5]}")
    items = [*gating, *(item for name in sorted(reported) for item in reported[name])]
    clock = _progress(f"{len(items)} items built", clock)

    merged = {model: {**decodes[GATING][model], **decodes[MINED][model]} for model in DETECTORS}
    _, grades = word_grades(items, merged, harness, models=DETECTORS,
                            scorings=app_scorings())
    clock = _progress(f"{len(grades)} word grades", clock)
    rows = word_rows(grades, clips)
    clock = _progress(f"{len(rows)} words paired", clock)

    clock = _progress("counting and writing", clock)
    summary = write_inventory(
        rows, inputs=dict(PINS), harness=harness_identity(harness),
        exclusions={"gating_decoded": len(decodes[GATING][BASE]),
                    "gating_excluded_248": len(excluded),
                    "mined_decoded": len(decodes[MINED][BASE]),
                    "listed_unphonetizable": len(listed)},
        catch_loss=catch_loss_context(ROOT), out_dir=INVENTORY_DIR, doc_path=DOC_PATH,
        clips=clips)
    _progress("artifacts written", clock)
    return summary


def _assigned_label(row: Mapping) -> str | None:
    """The ayah a mined row is read against: its ``assigned_label`` (#263), never its filename."""
    return row["assigned_label"]


def _export_rows(data: bytes, name: str) -> list[dict]:
    """The export's JSON rows, refusing bytes that are not gzipped UTF-8 JSON lines."""
    try:
        text = gzip.decompress(data).decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is not a gzipped UTF-8 word export") from error
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{name}:{number} is not a word row") from error
    return rows


def regenerate(*, inventory_dir: Path = INVENTORY_DIR, doc_path: Path = DOC_PATH) -> dict:
    """Rewrite ``summary.json`` and the doc from the committed word export: the counts, the
    intervals and the population are recomputed from the exported rows, so a counting or
    statistics fix needs no harness and no regrade.

    The export is an input, so it is authenticated: its bytes must hash to the SHA-256 the
    committed summary recorded when it was written, and every row must name a clip of the
    reconstructed confirm.1/2 pool with that clip's provenance. An export holding only the
    flagged words is refused, because the counts cannot be rebuilt from it."""
    inventory_dir = Path(inventory_dir)
    summary_path = inventory_dir / "summary.json"
    committed = json.loads(summary_path.read_text(encoding="utf-8"))
    report = committed["word_grades"]
    if not report.get("complete"):
        raise ValueError(f"{summary_path}: the committed export is the flagged-word fallback "
                         f"({report.get('clip_word_counts_file')}); the counts need every word")
    recorded = report.get("sha256")
    if recorded is None:
        raise ValueError(f"{summary_path}: the word export records no SHA-256; rebuild the "
                         f"inventory so the export it reads is authenticated")
    data = (inventory_dir / report["file"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != recorded:
        raise ValueError(f"{report['file']} is the SHA-256 {digest}, not the recorded "
                         f"{recorded}: the export changed after the summary recorded it")
    rows = _export_rows(data, report["file"])
    decodes = _decode_lines(ROOT)
    excluded = excluded_248(ROOT)
    listed = listed_unphonetizable(ROOT)
    manifests = {tranche: _manifest_rows(ROOT, tranche) for tranche in CONFIRM_TRANCHES}
    decoded = {**decodes[GATING][BASE], **decodes[MINED][BASE]}
    rows_by_tranche = {tranche: [row for row in manifest
                                 if item_key(row["audio_path"]) in decoded]
                       for tranche, manifest in manifests.items()}
    clips = clip_table(rows_by_tranche, excluded=excluded, listed=listed)
    for row in rows:
        key = row["key"]
        clip = clips.get(key)
        if clip is None:
            raise ValueError(f"{key} is not a clip of the confirm.1/confirm.2 pool: the export "
                             f"does not belong to the inputs it is counted under")
        if row["tranche"] not in CONFIRM_TRANCHES:
            raise ValueError(f"{key} is a {row['tranche']!r} row, not a confirm.1/confirm.2 row")
        fields = ("population", "tranche", "stratum", "label_source", "group_key")
        if any(row[field] != clip[field] for field in fields):
            raise ValueError(f"{key}: the export's {fields} are not the clip table's "
                             f"{[clip[field] for field in fields]}")
    summary = build_summary(
        rows, inputs=dict(PINS), harness=committed["harness"],
        exclusions={"gating_decoded": len(decodes[GATING][BASE]),
                    "gating_excluded_248": len(excluded),
                    "mined_decoded": len(decodes[MINED][BASE]),
                    "listed_unphonetizable": len(listed)},
        catch_loss=catch_loss_context(ROOT), word_grades=report, clips=clips)
    write_json(summary, summary_path)
    write_text_atomically(Path(doc_path), render(summary))
    _progress(f"summary and doc regenerated from {len(rows)} committed word rows")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("build", "regenerate"), default="build",
                        help="build grades the decodes and writes everything; regenerate "
                             "rewrites summary.json and the doc from the committed word export")
    args = parser.parse_args()
    summary = build() if args.command == "build" else regenerate()
    print(json.dumps({name: summary[name] for name in
                      ("population", "exclusions", "word_grades")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

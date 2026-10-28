"""Decode the tlog ``look`` tranche's gating rows for round 2 (#212); no grading.

Round 2 reads tlog look gating and bar 4's teacher check sequence, but S6/S7 committed only
whole-clip ``spans``. This module streams each shard once and decodes its gating rows with the
three models the narrowed run needs — the base teacher, C95 and the #178 candidate's averaged
weights — into the Tarteel v1 record shape ``training.aug_readout`` already grades, under the
one store that readout reads: the app's ``checks``. It decodes **only**: nothing here scores,
counts a flag or compares a model.

The look tranche is the only input. Any other role is refused before a file is opened, and a
look row whose row key also appears in a committed ``confirm.<k>.jsonl`` is refused too, so no
confirm row is ever opened. Each wanted shard is streamed once, its audio's SHA-256 checked
before it is decoded, and the shard deleted once read; audio is never written.

The run is several processes over disjoint shard subsets, each with ``OMP_NUM_THREADS=1``
(the work is CPU-bound; round 1 measured that configuration at ~3x this box's serial
per-model rate). A gate reads the first :data:`GATE_ROWS` timing lines and projects the
selection from the wall the run has taken; over :data:`RUN_LIMIT_HOURS` it stops the run and
writes ``<parts-dir>/gate_timing.json`` (exit 3), so nobody spends a day on a run the
orchestrator has not sized. The fallback the architect named is the teacher and C95 alone.

Model identities are the round's pins, checked here rather than through
``training.aug_readout_decode``'s #95 gate: C95 and the #178 candidate by their checkpoint
SHA-256, the teacher by the hub revision its loaded config resolved to.

Parts go to ``<parts-dir>/tlog.look.<shard_file>.<model>.jsonl``, one file per shard and
model, so a killed run resumes by key. ``merge`` gathers them into
``tools/tadabur/consonant2_readout/decodes/tlog.look.<model>.jsonl.gz`` with a ``meta.json``;
every model must cover every selected row exactly once, and a labelled row's ``word_from``/
``word_to`` become S6's identified span of the clip (the decode wrote the whole ayah).

Usage, from ``tools/`` (GPU box, under ``flock /root/scratch/gpu.lock``)::

    OMP_NUM_THREADS=1 python -m tadabur.tlog_eval_decode decode --parts-dir DIR \\
        --cache-dir DIR --models base c95 candidate_ema --shard-index 0 --shard-count 6
    python -m tadabur.tlog_eval_decode gate   --parts-dir DIR --elapsed-seconds 600 \\
        --processes 6 --threads 1 --limit-hours 11
    python -m tadabur.tlog_eval_decode merge  --parts-dir DIR --out-dir DIR \\
        --models base c95 candidate_ema --processes 6 --threads 1 --pr 213
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from .nonpro_sources import audio_sha256
from .tlog_ident import BASE_REVISION, load_references
from .tlog_index import (commit_of, head_sha, provenance, read_jsonl, sha256_file, write_json,
                         write_shard)
from .tlog_pass1 import read_jsonl_gz, shard_rows
from .tlog_pool import default_audio, row_key
from .tlog_tranches import CONFIRM_ROLES, GATING, TRANCHES_DIR
from training.aug_readout_gate import model_spec
from training.aug_readout_spec import C95_SHA256, CANDIDATE_SHA256, CONSONANT1
from training.external_decode import (ModelSpec, Recording, append_timing, decode_recordings,
                                      load_decoders)
from training.nonpro_decode import item_key
from training.truth_baseline import CHECK_STORE, DECODES_DIR, DEFAULT_MODELS, PROTOCOLS, Store

#: The tranche this decoder reads; every other role is refused.
LOOK = "look"
#: The four models round 2 can read, in the order their merged files and meta list them.
MODELS = ("base", "c95", "candidate_ema", "h448")
#: The narrowed run: the teacher, C95 and the #178 candidate (architect r26 drops ``h448``).
RUN_MODELS = ("base", "c95", "candidate_ema")
#: The three checkpoints, by the SHA-256 of their bytes (#174's C95, #178's candidate, h448).
H448_SHA256 = "eadc3f7504653cf4f88f4fb77482e1eb0f307e786ca5844d54ad72bc5483aa7a"
PINNED_SHA256 = {"c95": C95_SHA256, "candidate_ema": CANDIDATE_SHA256, "h448": H448_SHA256}
#: The teacher's hub commit: the loaded config must resolve to it, checked after the load.
PINNED_BASE_REVISION = BASE_REVISION
#: The locked 8 h slice gate's default limit (``slice_verdict``); the run's gate uses 11 h.
LIMIT_HOURS = 8.0
#: The narrowed run's parallelism (architect r27): round 1 measured OMP_NUM_THREADS=1 run as
#: several processes at ~3x this box's serial per-model rate, so each process is single-threaded.
RUN_PROCESSES = 6
RUN_THREADS = 1
#: Shard timing lines the gate waits for before it projects the rest of the selection.
GATE_ROWS = 12
RUN_LIMIT_HOURS = 11.0
DECODES_ROOT = Path(__file__).parent / "consonant2_readout" / "decodes"
#: S6's identified look tree: the per-row audio durations the gate's projection sums.
IDENT_LOOK_DIR = Path(__file__).parent / "tlog" / "ident" / "look"


def _stream_b0(decoder, span) -> dict:
    return {"stream_b0": decoder.decode_stream(span, block=0, flush_tail=True)}


#: The two stores the round-2 readout reads: the app's check sequence and the flushed b = 0
#: stream. The whole-clip ``spans`` decode is never taken (it is bounded by D_max; streaming
#: is not), and no store here grades anything.
STORES = {"checks": CHECK_STORE,
          "stream_b0": Store(DECODES_DIR, {"stream_b0": PROTOCOLS["stream_b0"]}, _stream_b0)}
#: The narrowed run takes the check sequence only (architect r26).
RUN_STORES = {"checks": CHECK_STORE}


# --- the models and their pins ------------------------------------------------------------------


def model_specs() -> dict[str, ModelSpec]:
    """The four models' decode specs, in :data:`MODELS` order.

    The teacher and ``h448`` are the truth baseline's; C95 and the #178 candidate are the
    consonant round's pinned checkpoints at step 10,000 under ``aug94-mixture-v2``, the
    candidate's averaged weights. This is the spec, not #95's gate: no committed choice is
    read (#212).
    """
    return {name: model_spec(name, CONSONANT1) for name in MODELS}


def require_pinned_identities(identities: Mapping[str, Mapping[str, object]],
                              models: Sequence[str] = MODELS) -> None:
    """Refuse an identity record that is not one of ``models`` at its pin.

    ``identities`` is the ``identity`` record ``load_decoders`` wrote for each model. The
    teacher is its reference plus the hub revision the loaded config resolved to — the check
    that pins a moving ``main`` after the load; a checkpoint is the SHA-256 of its bytes. A
    missing or extra model is a refusal naming it, never a silently smaller run.
    """
    missing = set(models) - set(identities)
    if missing:
        raise ValueError(f"no identity for {sorted(missing)[0]}; the run needs {list(models)}")
    extra = set(identities) - set(models)
    if extra:
        raise ValueError(f"{sorted(extra)[0]} is not one of the run's models {list(models)}")
    if "base" in models:
        base = identities["base"]
        if (base.get("model_ref"), base.get("hub_revision")) != (
                DEFAULT_MODELS["base"], PINNED_BASE_REVISION):
            raise ValueError(f"base is {base.get('model_ref')}@{base.get('hub_revision')}, "
                             f"not {DEFAULT_MODELS['base']}@{PINNED_BASE_REVISION}")
    for name in (name for name in models if name in PINNED_SHA256):
        found = identities[name].get("checkpoint_sha256")
        if found != PINNED_SHA256[name]:
            raise ValueError(f"{name} is {found}, not the pinned {PINNED_SHA256[name]}")


# --- the look tranche only ----------------------------------------------------------------------


def read_manifest(role: str, *, tranches_dir: Path = TRANCHES_DIR) -> list[dict]:
    """The ``look`` manifest's rows, refused if ``role`` is not look or a row is a confirm row.

    The role is checked before ``tranches_dir`` is touched, so a mistyped role can never open a
    ``confirm.<k>.jsonl``. Every look row's key ``(split, shard_file, row)`` must then be
    absent from each committed confirm manifest: a planted row is refused, not decoded.
    """
    if role != LOOK:
        raise ValueError(f"this decode reads the {LOOK!r} tranche only, not {role!r}")
    rows = read_jsonl(Path(tranches_dir) / f"{role}.jsonl")
    confirm: dict[tuple[str, str, int], str] = {}
    for name in CONFIRM_ROLES:
        path = Path(tranches_dir) / f"{name}.jsonl"
        if path.exists():
            confirm.update({row_key(row): name for row in read_jsonl(path)})
    planted = [(row_key(row), confirm[row_key(row)]) for row in rows
               if row_key(row) in confirm]
    if planted:
        key, name = planted[0]
        raise ValueError(f"look row {key} is also in {name}: a confirm row is never decoded "
                         f"here ({len(planted)} of {len(rows)} rows)")
    return rows


def gating_rows(rows: Sequence[Mapping]) -> list[dict]:
    """The rows the narrowed run decodes: the manifest's ``gating`` stratum (architect r26).

    Gating is the owner-approved rule's own stratum (``docs/tlog-gating-amendment.md`` §1), so
    it is read off the frozen manifest, never recomputed from a decode.
    """
    return [row for row in rows if row["stratum"] == GATING]


# --- one shard, every model ---------------------------------------------------------------------


def decodes_path(out_dir: Path, shard_file: str, model: str) -> Path:
    return Path(out_dir) / f"tlog.look.{shard_file}.{model}.jsonl"


def recordings(rows: Sequence[Mapping], *, stream, word_counts: Callable[[str], int],
               load_audio: Callable[[bytes], object] | None = None) -> Iterator[Recording]:
    """One shard's wanted manifest rows as recordings, in the shard's own row order.

    ``stream(name)`` yields the parquet rows of ``data/<shard_file>.parquet``; it is called
    once, and only the rows ``rows`` names are turned into recordings (each the whole clip,
    one item). A streamed row is a wanted row when its position is that row's index; a source
    that does not number its rows as the shard does is matched by the clip's recorded
    checksum, which is its identity either way. ``load_audio`` turns the row's bytes into
    16 kHz mono samples and is called only when the item is still to be decoded. The
    recording verifies the audio's SHA-256 before it decodes anything, and a wanted row the
    shard does not hold is an error.
    """
    rows = list(rows)
    if not rows:
        return
    shard_files = {row["shard_file"] for row in rows}
    if len(shard_files) != 1:
        raise ValueError(f"recordings takes one shard's rows, got {sorted(shard_files)}")
    shard_file = shard_files.pop()
    load = default_audio if load_audio is None else load_audio
    by_index = {row["row"]: row for row in rows}
    if len(by_index) != len(rows):
        raise ValueError(f"{shard_file}: a manifest row index appears twice")
    by_sha256 = {row["audio_sha256"]: row for row in rows}

    def record(row: Mapping, data: bytes) -> tuple[object, list[dict]]:
        checksum = audio_sha256(data)
        if checksum != row["audio_sha256"]:
            raise ValueError(f"{row_key(row)} is {checksum}, not the recorded "
                             f"{row['audio_sha256']}")
        samples = load(data)
        label = row["filename_label"]
        return samples, [{
            "key": item_key(row["audio_path"]),
            "audio_filename": row["audio_path"],
            "audio_sha256": checksum,
            "surah_ayah": label,
            "word_from": None if label is None else 1,
            "word_to": None if label is None else word_counts(label),
            "start_sample": 0,
            "end_sample": len(samples),
            "split": row["split"],
            "shard_file": row["shard_file"],
            "row": row["row"],
            "stratum": row["stratum"],
            "assigned_label": row["assigned_label"],
        }]

    for index, item in enumerate(stream(f"data/{shard_file}.parquet")):
        data = item["audio"]["bytes"]
        row = by_index.pop(index, None)
        if row is None:
            # A source that does not number its rows as the shard does: the clip's own
            # checksum is still its identity, so a wanted row is matched by that.
            row = by_sha256.pop(audio_sha256(data), None)
            if row is None:
                continue
            by_index.pop(row["row"], None)
        else:
            by_sha256.pop(row["audio_sha256"], None)
        yield Recording((item_key(row["audio_path"]),),
                        lambda row=row, data=data: record(row, data))
    if by_index:
        raise ValueError(f"{shard_file}: rows {sorted(by_index)} were not in the streamed shard")


def run_shard(shard_file: str, rows: Sequence[Mapping], out_dir: Path, decoders: dict, *,
              stream, word_counts: Callable[[str], int],
              load_audio: Callable[[bytes], object] | None = None,
              stores: Mapping = STORES) -> dict:
    """Read ``shard_file`` once and decode each wanted row with every decoder in ``decoders``.

    The records are appended to :func:`decodes_path` (one file per shard and model), so a
    rerun decodes only what is missing; returns ``decode_recordings``' timing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: decodes_path(out_dir, shard_file, name) for name in decoders}
    return decode_recordings(
        recordings(rows, stream=stream, word_counts=word_counts, load_audio=load_audio),
        decoders, paths, f"tlog look {shard_file}", stores=stores)


# --- merging the parts --------------------------------------------------------------------------


def ident_spans(directory: Path = IDENT_LOOK_DIR) -> dict[str, tuple[int, int]]:
    """S6's identified word span of each look clip, keyed by its item.

    The identified tree names the clip's own ``word_from``/``word_to`` (words of its ayah),
    which is what the grader builds its reference from; a gating row's decode records the whole
    ayah instead. The join is the row key the tree and the manifest share.
    """
    return {item_key(row["audio_path"]): (row["word_from"], row["word_to"])
            for row in ident_rows(directory)}


def merge_decodes(parts_dir: Path, out_dir: Path, *, manifest_path: Path,
                  provenance: Mapping, models: Sequence[str] = MODELS) -> dict:
    """Gather the per-shard parts into one sorted file per model, and write ``meta.json``.

    The expected rows are the **selection this run decodes** — the manifest's gating stratum,
    never the whole look tranche — so a merge checks the parts against what was asked for.
    Every model must hold each selected row exactly once: a missing, repeated or foreign row
    is a ``ValueError`` naming the model, so a partial run can never look complete. The merged
    files are gzipped without a timestamp (:func:`tlog_index.write_shard`), sorted by item key.

    A labelled row's ``word_from``/``word_to`` become the clip's identified span
    (:func:`ident_spans`), not the whole ayah's; a labelled row the identified tree does not
    cover keeps the span its decode recorded, and ``meta["word_spans"]["fallback"]`` counts it.
    """
    selected = gating_rows(read_jsonl(Path(manifest_path)))
    expected = {item_key(row["audio_path"]) for row in selected}
    labelled = {item_key(row["audio_path"]) for row in selected
                if row["filename_label"] is not None}
    spans = ident_spans()
    identified = sorted(labelled & set(spans))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for model in models:
        parts = sorted(Path(parts_dir).glob(f"tlog.look.*.{model}.jsonl"))
        lines = sorted((line for path in parts for line in read_jsonl(path)),
                       key=lambda line: line["key"])
        keys = [line["key"] for line in lines]
        repeated = [key for key, count in Counter(keys).items() if count > 1]
        if repeated:
            raise ValueError(f"{model}: {repeated[0]} appears twice in the parts")
        missing, extra = sorted(expected - set(keys)), sorted(set(keys) - expected)
        if missing or extra:
            raise ValueError(f"{model}: {len(missing)} look rows missing "
                             f"(first {missing[:1]}), {len(extra)} not in the manifest "
                             f"(first {extra[:1]})")
        for line in lines:
            span = spans.get(line["key"]) if line.get("surah_ayah") is not None else None
            if span is not None:
                line["word_from"], line["word_to"] = span
        write_shard(lines, out_dir / f"tlog.look.{model}.jsonl.gz")
        counts[model] = len(lines)
    write_json({**provenance, "manifest_sha256": sha256_file(Path(manifest_path)),
                "manifest_rows": len(expected), "models": list(models), "rows": counts,
                "word_spans": {"rule": "the clip's S6 identified word_from/word_to "
                                       "(tlog/ident/look), joined by the row key",
                               "source": "tlog/ident/look",
                               "identified": len(identified),
                               "fallback": len(labelled) - len(identified)}},
               out_dir / "meta.json")
    return counts


# --- the slice gate -----------------------------------------------------------------------------


def slice_verdict(slice_audio_s: float, slice_wall_s: float, total_audio_s: float,
                  limit_hours: float = LIMIT_HOURS) -> dict:
    """The selection's wall time projected from the shards covered so far, and whether it fits.

    The projection is linear in audio seconds, which is what the models' work tracks; the
    budget is inclusive at the limit.
    """
    if slice_audio_s <= 0:
        raise ValueError("the slice decoded no audio; it cannot project the full pass")
    projected = slice_wall_s / slice_audio_s * total_audio_s / 3600.0
    return {"slice_audio_s": slice_audio_s, "slice_wall_s": slice_wall_s,
            "total_audio_s": total_audio_s, "projected_hours": projected,
            "limit_hours": limit_hours, "within_budget": projected <= limit_hours}


def ident_rows(directory: Path = IDENT_LOOK_DIR) -> Iterator[dict]:
    """S6's identified look rows, from the committed tree's shards (one per shard)."""
    files = sorted(Path(directory).glob("*.jsonl.gz"))
    if not files:
        raise ValueError(f"{directory} holds no identified look tree")
    return (row for path in files for row in read_jsonl_gz(path))


def selection_audio_seconds(rows: Sequence[Mapping],
                            directory: Path = IDENT_LOOK_DIR) -> float:
    """The audio seconds of ``rows``, from S6's identified look tree's durations.

    A selected row the tree does not cover is refused: a short total would understate the
    projection the run gates on.
    """
    durations = {row_key(row): row["duration_s"] for row in ident_rows(directory)}
    keys = {row_key(row) for row in rows}
    missing = keys - set(durations)
    if missing:
        raise ValueError(f"{len(missing)} of {len(keys)} selected rows are not in {directory}: "
                         f"first {sorted(missing)[0]}")
    return sum(durations[key] for key in keys)


# --- the run ------------------------------------------------------------------------------------


def shard_stream(cache_dir: Path, on_download: Callable[[str], None] | None = None):
    """The row source :func:`recordings` streams: tlog shards, downloaded into ``cache_dir``
    and deleted once read. ``name`` is the dataset-relative shard path
    (``data/<shard_file>.parquet``), as :func:`tadabur.tlog_pass1.shard_rows` names it."""
    def stream(name: str) -> Iterator[dict]:
        return shard_rows(name, Path(cache_dir), on_download=on_download)
    return stream


def ayah_word_counts() -> Callable[[str], int]:
    """The words in one ``"surah:ayah"`` label, from the committed reference database."""
    database = load_references()

    def word_counts(label: str) -> int:
        surah, ayah = label.split(":")
        return database.ayat[(int(surah), int(ayah))].words
    return word_counts


def run_identities(parts_dir: Path, models: Sequence[str] = RUN_MODELS) -> dict[str, dict]:
    """The identity records ``load_decoders`` wrote for ``models`` into ``parts_dir``."""
    return {name: json.loads((Path(parts_dir) / f"{name}.identity.json").read_text(
        encoding="utf-8"))["identity"] for name in models}


def load_models(parts_dir: Path, models: Sequence[str] = RUN_MODELS) -> dict:
    """The run's decoders, with every identity checked against its pin after the load."""
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"{sorted(unknown)[0]} is not a round-2 model {list(MODELS)}")
    specs = {name: spec for name, spec in model_specs().items() if name in models}
    decoders = load_decoders(lambda name: parts_dir / f"{name}.identity.json", specs)
    require_pinned_identities(run_identities(parts_dir, models), models)
    return decoders


def _decode_shards(shards: Sequence[str], by_shard: Mapping[str, list[Mapping]], *,
                   parts_dir: Path, cache_dir: Path, decoders: dict,
                   word_counts: Callable[[str], int], stores: Mapping,
                   on_download: Callable[[str], None] | None = None) -> list[dict]:
    """Decode ``shards`` in order through :func:`run_shard`, one timing line per shard."""
    stream = shard_stream(cache_dir, on_download)
    timings = []
    for shard_file in shards:
        timing = run_shard(shard_file, by_shard[shard_file], parts_dir, decoders,
                           stream=stream, word_counts=word_counts, stores=stores)
        append_timing(Path(parts_dir) / "timing.jsonl", {"shard_file": shard_file, **timing})
        timings.append(timing)
    return timings


def _totals(timings: Sequence[Mapping], models: Sequence[str], stores: Mapping) -> dict:
    """The shards' timings summed: shards, items, audio, wall and decode seconds.

    ``wall_seconds_summed_over_processes`` adds every shard's own wall, so with several
    processes it is not the run's wall (the gate's ``slice_wall_s`` and the run's
    ``elapsed_seconds`` are).
    """
    return {"shards": len(timings),
            "items": sum(timing["items"] for timing in timings),
            "audio_seconds": round(sum(timing["audio_seconds"] for timing in timings), 1),
            "wall_seconds_summed_over_processes": round(
                sum(timing["wall_seconds"] for timing in timings), 1),
            "decode_seconds": {model: {store: round(sum(
                timing["decode_seconds"][model][store] for timing in timings), 1)
                for store in stores}
                for model in models}}


def decode_shards(*, parts_dir: Path, cache_dir: Path, models: Sequence[str] = RUN_MODELS,
                  stores: Mapping = RUN_STORES, shard_index: int = 0, shard_count: int = 1,
                  on_download: Callable[[str], None] | None = None) -> dict:
    """Decode one process's share of the look gating rows: ``shards[shard_index::shard_count]``.

    Each shard is streamed once for every model in ``models`` under ``stores``; processes
    write disjoint part files and resume by key, so they share no state but the parts.
    """
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"--shard-index must be in [0, {shard_count})")
    rows = gating_rows(read_manifest(LOOK))
    by_shard: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)
    shards = sorted(by_shard)[shard_index::shard_count]
    timings = _decode_shards(shards, by_shard, parts_dir=parts_dir, cache_dir=cache_dir,
                             decoders=load_models(parts_dir, models),
                             word_counts=ayah_word_counts(), stores=stores,
                             on_download=on_download)
    return _totals(timings, models, stores)


def gate_verdict(parts_dir: Path, *, elapsed_seconds: float,
                 limit_hours: float = RUN_LIMIT_HOURS, processes: int = RUN_PROCESSES,
                 threads: int = RUN_THREADS,
                 total_audio_s: float | None = None) -> dict:
    """Project the whole selection from the timing lines written so far, and record it.

    ``elapsed_seconds`` is the wall the run itself has taken (all processes together, not the
    summed shard walls, which double-count), so the projection is in this configuration's
    terms. The record lands in ``<parts-dir>/gate_timing.json``.
    """
    lines = read_jsonl(Path(parts_dir) / "timing.jsonl")
    total = (selection_audio_seconds(gating_rows(read_manifest(LOOK)))
             if total_audio_s is None else total_audio_s)
    record = {**slice_verdict(sum(line["audio_seconds"] for line in lines), elapsed_seconds,
                              total, limit_hours),
              "shards": len(lines), "items": sum(line["items"] for line in lines),
              "total_audio_s": total, "processes": processes, "threads": threads}
    write_json(record, Path(parts_dir) / "gate_timing.json")
    return record


def timing_totals(parts_dir: Path, models: Sequence[str] = RUN_MODELS,
                  stores: Mapping = RUN_STORES) -> dict:
    """The parts' per-shard timing lines summed, as :func:`_totals` reports a run."""
    path = Path(parts_dir) / "timing.jsonl"
    return _totals(read_jsonl(path) if path.exists() else [], models, stores)


def run_provenance(parts_dir: Path, *, models: Sequence[str] = RUN_MODELS,
                   stores: Mapping = RUN_STORES, processes: int = RUN_PROCESSES,
                   threads: int = RUN_THREADS, pull_request: int | None = None,
                   elapsed_seconds: float | None = None,
                   manifest_commit: str | None = None) -> dict:
    """What ``meta.json`` records of the run: code, models, the selection, its inputs, the
    parallelism, the gate's verdict and the timings.

    The model pins are read from the parts' own identity records and checked against the
    module's pins, so a parts dir carried from another run cannot cite weights this merge never
    saw. ``manifest_commit`` is the commit main holds the frozen manifest at (the launcher
    resolves it); without one the branch's last commit touching the file is recorded.
    ``elapsed_seconds`` is the run's own wall, when the launcher measured it.
    """
    identities = run_identities(parts_dir, models)
    require_pinned_identities(identities, models)
    gate = Path(parts_dir) / "gate_timing.json"
    return {**provenance(head_sha(), pull_request),
            "selection": f"{LOOK}.{GATING}",
            "stores": list(stores),
            "processes": processes,
            "threads": threads,
            "elapsed_seconds": elapsed_seconds,
            "base_revision": (identities["base"]["hub_revision"]
                              if "base" in models else None),
            "checkpoint_sha256": {name: identities[name]["checkpoint_sha256"]
                                  for name in models if name in PINNED_SHA256},
            "manifest_commit": (manifest_commit
                                or commit_of(TRANCHES_DIR / f"{LOOK}.jsonl")),
            "tranche_meta_sha256": sha256_file(TRANCHES_DIR / "meta.json"),
            "gate": json.loads(gate.read_text(encoding="utf-8")) if gate.exists() else None,
            "timings": timing_totals(parts_dir, models, stores)}


# --- CLI ----------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    decode_ = commands.add_parser("decode", help="decode one process's share of the gating rows")
    decode_.add_argument("--parts-dir", type=Path, required=True)
    decode_.add_argument("--cache-dir", type=Path, required=True)
    decode_.add_argument("--models", nargs="+", default=list(RUN_MODELS))
    decode_.add_argument("--shard-index", type=int, default=0)
    decode_.add_argument("--shard-count", type=int, default=RUN_PROCESSES)
    gate_ = commands.add_parser("gate", help="project the selection from the timing lines so far")
    gate_.add_argument("--parts-dir", type=Path, required=True)
    gate_.add_argument("--elapsed-seconds", type=float, required=True)
    gate_.add_argument("--limit-hours", type=float, default=RUN_LIMIT_HOURS)
    gate_.add_argument("--processes", type=int, default=RUN_PROCESSES)
    gate_.add_argument("--threads", type=int, default=RUN_THREADS)
    merge_ = commands.add_parser("merge", help="merge the parts into the committed decodes")
    merge_.add_argument("--parts-dir", type=Path, required=True)
    merge_.add_argument("--out-dir", type=Path, default=DECODES_ROOT)
    merge_.add_argument("--models", nargs="+", default=list(RUN_MODELS))
    merge_.add_argument("--processes", type=int, default=RUN_PROCESSES)
    merge_.add_argument("--threads", type=int, default=RUN_THREADS)
    merge_.add_argument("--pr", type=int, default=None,
                        help="the PR whose refs/pull/<N>/head keeps the cited code SHA")
    merge_.add_argument("--elapsed-seconds", type=float, default=None,
                        help="the run's own wall, for meta.json")
    merge_.add_argument("--manifest-commit", default=None,
                        help="the commit main holds the frozen manifest at")
    args = parser.parse_args()
    if args.command == "decode":
        result = decode_shards(parts_dir=args.parts_dir, cache_dir=args.cache_dir,
                               models=args.models, shard_index=args.shard_index,
                               shard_count=args.shard_count)
    elif args.command == "gate":
        result = gate_verdict(args.parts_dir, elapsed_seconds=args.elapsed_seconds,
                              limit_hours=args.limit_hours, processes=args.processes,
                              threads=args.threads)
    else:
        result = merge_decodes(args.parts_dir, args.out_dir,
                               manifest_path=TRANCHES_DIR / f"{LOOK}.jsonl",
                               provenance=run_provenance(
                                   args.parts_dir, models=args.models,
                                   processes=args.processes, threads=args.threads,
                                   pull_request=args.pr,
                                   elapsed_seconds=args.elapsed_seconds,
                                   manifest_commit=args.manifest_commit),
                               models=args.models)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    # 3 is the launcher's "the gate stopped the run; post it" code.
    if result.get("within_budget") is False:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

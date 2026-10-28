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
before it is decoded, and the shard deleted once read.

The selected clips are kept in a box-local clip cache (never the repository, never the fetched
output), each file named by the checksum it must hold: ``cache`` populates it for the gating
rows and ``coverage`` re-verifies and counts it. A later pass over the same rows decodes from
the cache without reading the source shards again; the shard download directory stays a
throwaway and its parquet files are deleted after use.

The run is several processes over disjoint shard subsets, each with ``OMP_NUM_THREADS=1``
(the work is CPU-bound; round 1 measured that configuration at ~3x this box's serial
per-model rate). A gate reads the first :data:`GATE_ROWS` timing lines and projects the
selection from the wall the run has taken; over :data:`RUN_LIMIT_HOURS` it stops the run and
writes ``<parts-dir>/gate_timing.json`` (exit 3), so nobody spends a day on a run the
orchestrator has not sized. The fallback the architect named is the teacher and C95 alone.

A check's features do not depend on the model, so a ``decode`` pass shares them: each item is
featurised once and every model still pending on it runs on those features
(``--no-share-features`` reproduces the per-model path, which is a benchmark's control). The
records are unchanged by sharing -- ``parity`` re-decodes the first :data:`PARITY_SHARDS`
gating shards and compares its parts with the committed decodes of exactly those shards,
line for line, writing ``tools/tadabur/consonant2_readout/decodes/shared_parity.json``. It
compares decode text only: nothing here grades.

Model identities are the round's pins, checked here rather than through
``training.aug_readout_decode``'s #95 gate: C95 and the #178 candidate by their checkpoint
SHA-256, the teacher by the hub revision its loaded config resolved to.

Parts go to ``<parts-dir>/tlog.look.<shard_file>.<model>.jsonl``, one file per shard and
model, so a killed run resumes by key. ``merge`` gathers them into
``tools/tadabur/consonant2_readout/decodes/tlog.look.<model>.jsonl.gz`` with a ``meta.json``;
every model must cover every selected row exactly once, and a labelled row's ``word_from``/
``word_to`` become S6's identified span of the clip (the decode wrote the whole ayah).

Usage, from ``tools/`` on the GPU box (``decode`` and ``merge`` under
``flock /root/scratch/gpu.lock``; ``cache``/``coverage`` are CPU and network only)::

    OMP_NUM_THREADS=1 python -m tadabur.tlog_eval_decode cache  --clip-cache DIR \\
        --shard-cache DIR --shard-index 0 --shard-count 6
    python -m tadabur.tlog_eval_decode coverage --clip-cache DIR
    OMP_NUM_THREADS=1 python -m tadabur.tlog_eval_decode decode --parts-dir DIR \\
        --cache-dir DIR --clip-cache DIR --models base c95 candidate_ema \\
        --shard-index 0 --shard-count 6
    python -m tadabur.tlog_eval_decode gate   --parts-dir DIR --elapsed-seconds 600 \\
        --processes 6 --threads 1 --limit-hours 11
    python -m tadabur.tlog_eval_decode merge  --parts-dir DIR --out-dir DIR \\
        --models base c95 candidate_ema --processes 6 --threads 1 --pr 213
    python -m tadabur.tlog_eval_decode parity --parts-dir DIR --out shared_parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
#: The shards the shared-feature re-decode is compared with the committed decodes on: the
#: first 20 gating shards in name order (#213).
PARITY_SHARDS = 20
#: The committed record of that comparison, written by ``parity``.
PARITY_PATH = DECODES_ROOT / "shared_parity.json"
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


def shard_items(rows: Sequence[Mapping], *, stream) -> Iterator[tuple[Mapping, bytes]]:
    """One shard's wanted manifest rows and their clip bytes, in the shard's own row order.

    ``stream(name)`` yields the parquet rows of ``data/<shard_file>.parquet``; it is called
    once, and only the rows ``rows`` names are yielded. A streamed row is a wanted row when
    its position is that row's index; a source that does not number its rows as the shard
    does is matched by the clip's recorded checksum, which is its identity either way. A
    wanted row the shard does not hold is an error.
    """
    rows = list(rows)
    if not rows:
        return
    shard_files = {row["shard_file"] for row in rows}
    if len(shard_files) != 1:
        raise ValueError(f"one shard's rows at a time, got {sorted(shard_files)}")
    shard_file = shard_files.pop()
    by_index = {row["row"]: row for row in rows}
    if len(by_index) != len(rows):
        raise ValueError(f"{shard_file}: a manifest row index appears twice")
    by_sha256 = {row["audio_sha256"]: row for row in rows}
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
        yield row, data
    if by_index:
        raise ValueError(f"{shard_file}: rows {sorted(by_index)} were not in the streamed shard")


def _record(row: Mapping, data: bytes, load: Callable[[bytes], object],
            word_counts: Callable[[str], int]) -> tuple[object, list[dict]]:
    """One verified clip's samples and its Tarteel v1 record."""
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


def recordings(rows: Sequence[Mapping], *, stream, word_counts: Callable[[str], int],
               load_audio: Callable[[bytes], object] | None = None,
               clip_cache: Path | None = None) -> Iterator[Recording]:
    """One shard's wanted manifest rows as recordings, in the shard's own row order.

    ``load_audio`` turns the row's bytes into 16 kHz mono samples and is called only when the
    item is still to be decoded. The recording verifies the audio's SHA-256 before it decodes
    anything. With ``clip_cache``, a clip already cached under its recorded checksum
    (:func:`clip_path`) is read on load rather than streamed, so a pass whose rows are all
    cached never calls ``stream``.
    """
    load = default_audio if load_audio is None else load_audio
    pending = []
    for row in rows:
        path = None if clip_cache is None else clip_path(clip_cache, row["audio_sha256"])
        if path is None or not path.is_file():
            pending.append(row)
            continue
        yield Recording((item_key(row["audio_path"]),),
                        lambda row=row, path=path: _record(row, path.read_bytes(), load,
                                                           word_counts))
    for row, data in shard_items(pending, stream=stream):
        yield Recording((item_key(row["audio_path"]),),
                        lambda row=row, data=data: _record(row, data, load, word_counts))


def run_shard(shard_file: str, rows: Sequence[Mapping], out_dir: Path, decoders: dict, *,
              stream, word_counts: Callable[[str], int],
              load_audio: Callable[[bytes], object] | None = None,
              stores: Mapping = STORES, clip_cache: Path | None = None,
              share_features: bool = False) -> dict:
    """Read ``shard_file`` once and decode each wanted row with every decoder in ``decoders``.

    The records are appended to :func:`decodes_path` (one file per shard and model), so a
    rerun decodes only what is missing; returns ``decode_recordings``' timing. Clips in
    ``clip_cache`` are decoded without reading the shard. ``share_features`` is
    ``decode_recordings``' flag: one feature pass per item for a store that declares one.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: decodes_path(out_dir, shard_file, name) for name in decoders}
    return decode_recordings(
        recordings(rows, stream=stream, word_counts=word_counts, load_audio=load_audio,
                   clip_cache=clip_cache),
        decoders, paths, f"tlog look {shard_file}", stores=stores,
        share_features=share_features)


# --- the on-box clip cache ----------------------------------------------------------------------


def clip_path(directory: Path, sha256: str) -> Path:
    """A cached clip's path: named by the checksum it must hold."""
    return Path(directory) / f"{sha256}.audio"


def cache_shard(shard_file: str, rows: Sequence[Mapping], clip_cache: Path, *,
                stream) -> dict:
    """Persist one shard's selected clips, each verified against its recorded checksum.

    A row already cached is not fetched, so a second population pass reads no shard. The
    stored bytes are the source clip's own (written atomically), never samples or decodes.
    """
    pending = [row for row in rows
               if not clip_path(clip_cache, row["audio_sha256"]).is_file()]
    written = cached_bytes = 0
    for row, data in shard_items(pending, stream=stream):
        checksum = audio_sha256(data)
        if checksum != row["audio_sha256"]:
            raise ValueError(f"{row_key(row)} is {checksum}, not the recorded "
                             f"{row['audio_sha256']}")
        path = clip_path(clip_cache, checksum)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        written += 1
        cached_bytes += len(data)
    return {"shard_file": shard_file, "rows": len(rows), "written": written,
            "bytes": cached_bytes}


def cache_selection(clip_cache: Path, shard_cache: Path, *,
                    rows: Sequence[Mapping] | None = None, shard_index: int = 0,
                    shard_count: int = 1, stream: Callable | None = None,
                    on_download: Callable[[str], None] | None = None) -> dict:
    """Populate the box's clip cache with the selected rows, one pass over their shards.

    The rows are the manifest's gating stratum unless given; each clip is stored under its
    verified checksum, and the shard is deleted once read (``shard_rows``'s contract), so the
    selection can be decoded again without the source. Nothing is decoded or graded.
    """
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"--shard-index must be in [0, {shard_count})")
    rows = gating_rows(read_manifest(LOOK)) if rows is None else list(rows)
    by_shard: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)
    if stream is None:
        stream = shard_stream(shard_cache, on_download)
    shards = sorted(by_shard)[shard_index::shard_count]
    written = cached_bytes = 0
    for shard_file in shards:
        timing = cache_shard(shard_file, by_shard[shard_file], clip_cache, stream=stream)
        written += timing["written"]
        cached_bytes += timing["bytes"]
    return {"shards": len(shards), "rows": sum(len(by_shard[name]) for name in shards),
            "written": written, "bytes": cached_bytes}


def cache_coverage(rows: Sequence[Mapping], directory: Path) -> dict:
    """How much of ``rows`` the clip cache holds, each cached clip re-verified."""
    missing = [row_key(row) for row in rows
               if not clip_path(directory, row["audio_sha256"]).is_file()]
    absent = set(missing)
    held = [row for row in rows if row_key(row) not in absent]
    stored = 0
    for row in held:
        path = clip_path(directory, row["audio_sha256"])
        if sha256_file(path) != row["audio_sha256"]:
            raise ValueError(f"{path} is not the clip its name records")
        stored += path.stat().st_size
    return {"rows": len(rows), "cached": len(held), "missing": len(missing),
            "missing_first": missing[:1], "bytes": stored}


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
                   clip_cache: Path | None = None, share_features: bool,
                   on_download: Callable[[str], None] | None = None) -> list[dict]:
    """Decode ``shards`` in order through :func:`run_shard`, one timing line per shard.

    ``share_features`` is required: it is the one place where the run's sharing choice is
    made, so a new entry point cannot inherit one silently. ``decode_shards`` shares;
    ``run_shard`` stays off by default as the per-shard control the tests decode.
    """
    stream = shard_stream(cache_dir, on_download)
    timings = []
    for shard_file in shards:
        timing = {"shard_file": shard_file, **run_shard(
            shard_file, by_shard[shard_file], parts_dir, decoders, stream=stream,
            word_counts=word_counts, stores=stores, clip_cache=clip_cache,
            share_features=share_features)}
        append_timing(Path(parts_dir) / "timing.jsonl", timing)
        timings.append(timing)
    return timings


def _totals(timings: Sequence[Mapping], models: Sequence[str], stores: Mapping) -> dict:
    """The shards' timings summed: shards, items, audio, wall and decode seconds.

    A shard counts once however often a resume visited it; its visits' work is summed, and a
    model that decoded nothing in a visit has no store seconds there — a legitimate zero, not
    a missing key. A record naming another model or store is refused. A visit that shared a
    store's feature pass adds its feature seconds under ``shared_feature_seconds``; the key is
    absent when no visit shared one.

    ``wall_seconds_summed_over_processes`` adds every shard's own wall, so with several
    processes it is not the run's wall (the gate's ``slice_wall_s`` and the run's
    ``elapsed_seconds`` are).
    """
    seconds = {(model, store): 0.0 for model in models for store in stores}
    features: dict[str, float] = defaultdict(float)
    shared = False
    for timing in timings:
        for model, arms in timing["decode_seconds"].items():
            if model not in models:
                raise ValueError(f"{timing['shard_file']}: timing names {model!r}, "
                                 f"not one of {list(models)}")
            for store, value in arms.items():
                if store not in stores:
                    raise ValueError(f"{timing['shard_file']}: timing names {store!r}, "
                                     f"not one of {list(stores)}")
                seconds[(model, store)] += value
        if "shared_feature_seconds" in timing:
            shared = True
            for store, value in timing["shared_feature_seconds"].items():
                if store not in stores:
                    raise ValueError(f"{timing['shard_file']}: shared features name {store!r}, "
                                     f"not one of {list(stores)}")
                features[store] += value
    totals = {"shards": len({timing["shard_file"] for timing in timings}),
              "items": sum(timing["items"] for timing in timings),
              "audio_seconds": round(sum(timing["audio_seconds"] for timing in timings), 1),
              "wall_seconds_summed_over_processes": round(
                  sum(timing["wall_seconds"] for timing in timings), 1),
              "decode_seconds": {model: {store: round(seconds[(model, store)], 1)
                                         for store in stores}
                                 for model in models}}
    if shared:
        totals["shared_feature_seconds"] = {store: round(value, 1)
                                            for store, value in features.items()}
    return totals


def decode_shards(*, parts_dir: Path, cache_dir: Path, models: Sequence[str] = RUN_MODELS,
                  stores: Mapping = RUN_STORES, shard_index: int = 0, shard_count: int = 1,
                  clip_cache: Path | None = None,
                  on_download: Callable[[str], None] | None = None,
                  shard_files: Sequence[str] | None = None,
                  share_features: bool = True) -> dict:
    """Decode one process's share of the look gating rows: ``shards[shard_index::shard_count]``.

    Each shard is streamed once for every model in ``models`` under ``stores``; processes
    write disjoint part files and resume by key, so they share no state but the parts. Clips
    already in ``clip_cache`` are decoded without fetching their shard. ``shard_files``
    restricts the selection to the named gating shards (the parity run's first 20), each of
    which must hold a gating row; index and count then split that selection. With
    ``share_features`` (on by default, off only to reproduce the per-model path) the checks
    store featurises each item once for every model pending on it.
    """
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"--shard-index must be in [0, {shard_count})")
    rows = gating_rows(read_manifest(LOOK))
    by_shard: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)
    if shard_files is None:
        chosen = sorted(by_shard)
    else:
        unknown = sorted(set(shard_files) - set(by_shard))
        if unknown:
            raise ValueError(f"{unknown[0]} is not a gating shard of the {LOOK!r} tranche")
        chosen = sorted(set(shard_files))
    shards = chosen[shard_index::shard_count]
    timings = _decode_shards(shards, by_shard, parts_dir=parts_dir, cache_dir=cache_dir,
                             decoders=load_models(parts_dir, models),
                             word_counts=ayah_word_counts(), stores=stores,
                             clip_cache=clip_cache, share_features=share_features,
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


# --- the shared-feature parity (#213) -----------------------------------------------------------


def parity_shards(rows: Sequence[Mapping] | None = None) -> list[str]:
    """The shards the parity run re-decodes and compares: the first :data:`PARITY_SHARDS`
    gating shards in name order."""
    rows = gating_rows(read_manifest(LOOK)) if rows is None else rows
    return sorted({row["shard_file"] for row in rows})[:PARITY_SHARDS]


#: Fields a decode line did not write: ``merge`` joins the identified tree's word span in, so
#: a part's span is not part of what the decode computed.
PARITY_IGNORED = ("word_from", "word_to")


def _parity_line(line: Mapping) -> dict:
    return {key: value for key, value in line.items() if key not in PARITY_IGNORED}


def _keyed_lines(lines: Iterable[Mapping], *, model: str, side: str) -> dict[str, dict]:
    """The lines by row key, refusing a repeated key rather than keeping the last one.

    A decode claims each row once (``merge_decodes`` refuses repeats the same way); a part
    concatenated twice must not be able to put a wrong decode behind a right one.
    """
    by_key: dict[str, dict] = {}
    for line in lines:
        if line["key"] in by_key:
            raise ValueError(f"{model}: {line['key']} appears twice in the {side}")
        by_key[line["key"]] = _parity_line(line)
    return by_key


def parity_digest(lines: Iterable[Mapping]) -> str:
    """The SHA-256 of a set of decode lines, key-sorted and without ``word_from``/``word_to``.

    Two sets of lines are byte-identical decodes exactly when their digests agree: the same
    records, whatever order they were written or read in.
    """
    canonical = sorted((_parity_line(line) for line in lines), key=lambda line: line["key"])
    text = "".join(json.dumps(line, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n" for line in canonical)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parity_record(parts_dir: Path, *, models: Sequence[str] = RUN_MODELS,
                  shards: Sequence[str] | None = None, decodes_dir: Path = DECODES_ROOT,
                  processes: int = RUN_PROCESSES, threads: int = RUN_THREADS,
                  pull_request: int | None = None) -> dict:
    """Compare a shared-feature re-decode's parts with the committed decodes, shard by shard.

    ``shards`` defaults to :func:`parity_shards`; the parts are the per-shard files
    :func:`decodes_path` wrote for those shards, and the committed lines are the same shards'
    lines of ``decodes_dir/tlog.look.<model>.jsonl.gz``. A line pair is compared whole, never
    graded. The record names the comparison's rows, mismatches and digests, plus what shows the
    parts were decoded rather than put there: the timing lines' items and seconds
    (:func:`timing_totals`) and each part file's SHA-256. A parts directory seeded from the
    committed decodes compares identical but records no items. A row key that appears twice on
    either side is refused: a concatenated part could otherwise hide a wrong decode behind a
    right one, and the line count would hide the extra row.
    """
    shards = parity_shards() if shards is None else sorted(set(shards))
    selected = set(shards)
    timing = timing_totals(parts_dir, models)
    record = {**provenance(head_sha(), pull_request), "shards": shards,
              "models": list(models), "processes": processes, "threads": threads,
              "share_features": True, "items": timing["items"], "timing": timing,
              "rows": {}, "mismatches": {}, "identical": {},
              "shared_sha256": {}, "committed_sha256": {}, "part_sha256": {}}
    for model in models:
        shared = [line for shard_file in shards
                  for line in read_jsonl(decodes_path(parts_dir, shard_file, model))]
        committed = [line for line in read_jsonl_gz(
            Path(decodes_dir) / f"tlog.look.{model}.jsonl.gz")
            if line["shard_file"] in selected]
        expected = _keyed_lines(committed, model=model, side="committed decodes")
        found = _keyed_lines(shared, model=model, side="shared parts")
        record["rows"][model] = len(found)
        record["mismatches"][model] = (
            len(set(expected) ^ set(found))
            + sum(found[key] != expected[key] for key in set(found) & set(expected)))
        record["identical"][model] = record["mismatches"][model] == 0
        record["shared_sha256"][model] = parity_digest(shared)
        record["committed_sha256"][model] = parity_digest(committed)
        record["part_sha256"][model] = {
            shard_file: hashlib.sha256(
                decodes_path(parts_dir, shard_file, model).read_bytes()).hexdigest()
            for shard_file in shards}
    return record


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
    decode_.add_argument("--clip-cache", type=Path, default=None,
                         help="the box cache of selected clips, keyed by their checksum")
    decode_.add_argument("--shard-files", nargs="+", default=None,
                         help="decode only the named gating shards (the parity selection)")
    decode_.add_argument("--no-share-features", action="store_true",
                         help="featurise every check once per model, as before #213")
    cache_ = commands.add_parser("cache", help="persist the gating rows' clips into the box cache")
    cache_.add_argument("--clip-cache", type=Path, required=True)
    cache_.add_argument("--shard-cache", type=Path, required=True)
    cache_.add_argument("--shard-index", type=int, default=0)
    cache_.add_argument("--shard-count", type=int, default=1)
    coverage_ = commands.add_parser("coverage", help="verify and count the cached gating clips")
    coverage_.add_argument("--clip-cache", type=Path, required=True)
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
    parity_ = commands.add_parser("parity", help="compare a shared re-decode with the decodes")
    parity_.add_argument("--parts-dir", type=Path, required=True)
    parity_.add_argument("--out", type=Path, default=PARITY_PATH)
    parity_.add_argument("--models", nargs="+", default=list(RUN_MODELS))
    parity_.add_argument("--processes", type=int, default=RUN_PROCESSES)
    parity_.add_argument("--threads", type=int, default=RUN_THREADS)
    parity_.add_argument("--pr", type=int, default=None,
                         help="the PR whose refs/pull/<N>/head keeps the cited code SHA")
    args = parser.parse_args()
    if args.command == "decode":
        result = decode_shards(parts_dir=args.parts_dir, cache_dir=args.cache_dir,
                               models=args.models, shard_index=args.shard_index,
                               shard_count=args.shard_count, clip_cache=args.clip_cache,
                               shard_files=args.shard_files,
                               share_features=not args.no_share_features)
    elif args.command == "gate":
        result = gate_verdict(args.parts_dir, elapsed_seconds=args.elapsed_seconds,
                              limit_hours=args.limit_hours, processes=args.processes,
                              threads=args.threads)
    elif args.command == "cache":
        result = cache_selection(args.clip_cache, args.shard_cache,
                                 shard_index=args.shard_index, shard_count=args.shard_count)
    elif args.command == "coverage":
        result = cache_coverage(gating_rows(read_manifest(LOOK)), args.clip_cache)
    elif args.command == "parity":
        result = parity_record(args.parts_dir, models=args.models, shards=parity_shards(),
                               processes=args.processes, threads=args.threads,
                               pull_request=args.pr)
        write_json(result, args.out)
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

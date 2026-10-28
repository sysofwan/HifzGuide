"""Decode the tlog ``look`` tranche with the four round-2 models (#212); no grading.

Round 2 reads tlog look gating and bar 4's teacher check sequence, but S6/S7 committed only
whole-clip ``spans``. This module streams every look row once and decodes it with all four
models at once — the base teacher, C95, the #178 candidate's averaged weights and ``h448`` —
into the Tarteel v1 record shape ``training.aug_readout`` already grades, under the two stores
that readout needs: the app's ``checks`` and the flushed ``stream_b0``. It decodes **only**:
nothing here scores, counts a flag or compares a model.

The look tranche is the only input. Any other role is refused before a file is opened, and a
look row whose row key also appears in a committed ``confirm.<k>.jsonl`` is refused too, so no
confirm row is ever opened. Each wanted shard is streamed once, its audio's SHA-256 checked
before it is decoded, and the shard deleted once read; audio is never written.

Model identities are the round's pins, checked here rather than through
``training.aug_readout_decode``'s #95 gate: C95, the #178 candidate and ``h448`` by their
checkpoint SHA-256, the teacher by the hub revision its loaded config resolved to.

Parts go to ``<parts-dir>/tlog.look.<shard_file>.<model>.jsonl``, one file per shard and
model, so shards decode side by side and a killed run resumes by key. ``merge`` gathers them
into ``tools/tadabur/consonant2_readout/decodes/tlog.look.<model>.jsonl.gz`` with a
``meta.json``; every model must cover every manifest row exactly once.

Usage, from ``tools/`` (GPU box, under ``flock /root/scratch/gpu.lock``)::

    python -m tadabur.tlog_eval_decode slice  --parts-dir DIR --cache-dir DIR --rows 500
    python -m tadabur.tlog_eval_decode decode --parts-dir DIR --cache-dir DIR \\
        --shard-index 0 --shard-count 2
    python -m tadabur.tlog_eval_decode merge  --parts-dir DIR --out-dir DIR --pr 213

``slice`` exits 3 when the projected full pass is over the 8 h budget, so the launcher stops
and posts instead of continuing.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from .nonpro_sources import audio_sha256
from .tlog_ident import BASE_REVISION, load_references
from .tlog_index import (commit_of, head_sha, provenance, read_jsonl, sha256_file, write_json,
                         write_shard)
from .tlog_pass1 import read_jsonl_gz, shard_rows
from .tlog_pool import default_audio, row_key
from .tlog_tranches import CONFIRM_ROLES, TRANCHES_DIR
from training.aug_readout_gate import model_spec
from training.aug_readout_spec import C95_SHA256, CANDIDATE_SHA256, CONSONANT1
from training.external_decode import (ModelSpec, Recording, append_timing, decode_recordings,
                                      load_decoders)
from training.truth_baseline import CHECK_STORE, DECODES_DIR, DEFAULT_MODELS, PROTOCOLS, Store

#: The tranche this decoder reads; every other role is refused.
LOOK = "look"
#: The four models round 2 reads, in the order their merged files and meta list them.
MODELS = ("base", "c95", "candidate_ema", "h448")
#: The three checkpoints, by the SHA-256 of their bytes (#174's C95, #178's candidate, h448).
H448_SHA256 = "eadc3f7504653cf4f88f4fb77482e1eb0f307e786ca5844d54ad72bc5483aa7a"
PINNED_SHA256 = {"c95": C95_SHA256, "candidate_ema": CANDIDATE_SHA256, "h448": H448_SHA256}
#: The teacher's hub commit: the loaded config must resolve to it, checked after the load.
PINNED_BASE_REVISION = BASE_REVISION
#: Rows the timed slice decodes before the full pass; the launcher gates on its projection.
SLICE_ROWS = 500
LIMIT_HOURS = 8.0
DECODES_ROOT = Path(__file__).parent / "consonant2_readout" / "decodes"
#: S6's identified look tree: the per-row audio durations the slice's projection sums.
IDENT_LOOK_DIR = Path(__file__).parent / "tlog" / "ident" / "look"


def _stream_b0(decoder, span) -> dict:
    return {"stream_b0": decoder.decode_stream(span, block=0, flush_tail=True)}


#: The two stores the round-2 readout reads: the app's check sequence and the flushed b = 0
#: stream. The whole-clip ``spans`` decode is never taken (it is bounded by D_max; streaming
#: is not), and no store here grades anything.
STORES = {"checks": CHECK_STORE,
          "stream_b0": Store(DECODES_DIR, {"stream_b0": PROTOCOLS["stream_b0"]}, _stream_b0)}


# --- the models and their pins ------------------------------------------------------------------


def model_specs() -> dict[str, ModelSpec]:
    """The four models' decode specs, in :data:`MODELS` order.

    The teacher and ``h448`` are the truth baseline's; C95 and the #178 candidate are the
    consonant round's pinned checkpoints at step 10,000 under ``aug94-mixture-v2``, the
    candidate's averaged weights. This is the spec, not #95's gate: no committed choice is
    read (#212).
    """
    return {name: model_spec(name, CONSONANT1) for name in MODELS}


def require_pinned_identities(identities: Mapping[str, Mapping[str, object]]) -> None:
    """Refuse an identity record that is not one of the four models at its pin.

    ``identities`` is the ``identity`` record ``load_decoders`` wrote for each model. The
    teacher is its reference plus the hub revision the loaded config resolved to — the check
    that pins a moving ``main`` after the load; a checkpoint is the SHA-256 of its bytes. A
    missing or extra model is a refusal naming it, never a silently smaller run.
    """
    missing = set(MODELS) - set(identities)
    if missing:
        raise ValueError(f"no identity for {sorted(missing)[0]}; the run needs {list(MODELS)}")
    extra = set(identities) - set(MODELS)
    if extra:
        raise ValueError(f"{sorted(extra)[0]} is not one of the round's models {list(MODELS)}")
    base = identities["base"]
    if (base.get("model_ref"), base.get("hub_revision")) != (
            DEFAULT_MODELS["base"], PINNED_BASE_REVISION):
        raise ValueError(f"base is {base.get('model_ref')}@{base.get('hub_revision')}, not "
                         f"{DEFAULT_MODELS['base']}@{PINNED_BASE_REVISION}")
    for name, sha256 in PINNED_SHA256.items():
        found = identities[name].get("checkpoint_sha256")
        if found != sha256:
            raise ValueError(f"{name} is {found}, not the pinned {sha256}")


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


# --- one shard, every model ---------------------------------------------------------------------


def item_key(audio_path: str) -> str:
    """A clip's one item, named as a QUD segment is (``<clip>#<segment>``)."""
    return f"{audio_path}#0"


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
              load_audio: Callable[[bytes], object] | None = None) -> dict:
    """Read ``shard_file`` once and decode each wanted row with every decoder in ``decoders``.

    The records are appended to :func:`decodes_path` (one file per shard and model), so a
    rerun decodes only what is missing; returns ``decode_recordings``' timing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: decodes_path(out_dir, shard_file, name) for name in decoders}
    return decode_recordings(
        recordings(rows, stream=stream, word_counts=word_counts, load_audio=load_audio),
        decoders, paths, f"tlog look {shard_file}", stores=STORES)


# --- merging the parts --------------------------------------------------------------------------


def merge_decodes(parts_dir: Path, out_dir: Path, *, manifest_path: Path,
                  provenance: Mapping, models: Sequence[str] = MODELS) -> dict:
    """Gather the per-shard parts into one sorted file per model, and write ``meta.json``.

    Every model must hold each manifest row exactly once: a missing, repeated or foreign row
    is a ``ValueError`` naming the model, so a partial run can never look complete. The merged
    files are gzipped without a timestamp (:func:`tlog_index.write_shard`), sorted by item key.
    """
    expected = {item_key(row["audio_path"]) for row in read_jsonl(Path(manifest_path))}
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
        write_shard(lines, out_dir / f"tlog.look.{model}.jsonl.gz")
        counts[model] = len(lines)
    write_json({**provenance, "manifest_sha256": sha256_file(Path(manifest_path)),
                "manifest_rows": len(expected), "models": list(models), "rows": counts},
               out_dir / "meta.json")
    return counts


# --- the slice gate -----------------------------------------------------------------------------


def slice_verdict(slice_audio_s: float, slice_wall_s: float, total_audio_s: float,
                  limit_hours: float = LIMIT_HOURS) -> dict:
    """The full pass's wall time projected from the slice's, and whether it fits the budget.

    The projection is linear in audio seconds, which is what the four models' work tracks; the
    budget is inclusive at the limit.
    """
    if slice_audio_s <= 0:
        raise ValueError("the slice decoded no audio; it cannot project the full pass")
    projected = slice_wall_s / slice_audio_s * total_audio_s / 3600.0
    return {"slice_audio_s": slice_audio_s, "slice_wall_s": slice_wall_s,
            "total_audio_s": total_audio_s, "projected_hours": projected,
            "limit_hours": limit_hours, "within_budget": projected <= limit_hours}


def total_audio_seconds(directory: Path = IDENT_LOOK_DIR) -> float:
    """The look tranche's audio seconds, summed from S6's identified tree's durations."""
    files = sorted(Path(directory).glob("*.jsonl.gz"))
    if not files:
        raise ValueError(f"{directory} holds no identified look tree")
    return sum(row["duration_s"] for path in files for row in read_jsonl_gz(path))


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


def load_models(parts_dir: Path) -> dict:
    """The four decoders, with every identity checked against its pin after the load."""
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs()
    decoders = load_decoders(lambda name: parts_dir / f"{name}.identity.json", specs)
    identities = {name: json.loads((parts_dir / f"{name}.identity.json").read_text(
        encoding="utf-8"))["identity"] for name in specs}
    require_pinned_identities(identities)
    return decoders


def _decode_rows(rows: Sequence[Mapping], *, parts_dir: Path, cache_dir: Path, decoders: dict,
                 word_counts: Callable[[str], int],
                 on_download: Callable[[str], None] | None = None) -> list[dict]:
    """Decode ``rows`` shard by shard through :func:`run_shard`, one timing line per shard."""
    by_shard: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)
    stream = shard_stream(cache_dir, on_download)
    timings = []
    for shard_file in sorted(by_shard):
        timing = run_shard(shard_file, by_shard[shard_file], parts_dir, decoders,
                           stream=stream, word_counts=word_counts)
        append_timing(Path(parts_dir) / "timing.jsonl", {"shard_file": shard_file, **timing})
        timings.append(timing)
    return timings


def time_slice(*, parts_dir: Path, cache_dir: Path, rows: int = SLICE_ROWS,
               total_audio_s: float | None = None, limit_hours: float = LIMIT_HOURS,
               on_download: Callable[[str], None] | None = None) -> dict:
    """Decode the first ``rows`` look rows by row key, time them and project the full pass.

    The slice writes through the same parts the full pass reads, so no row is decoded twice.
    Its record lands in ``<parts_dir>/slice_timing.json``.
    """
    if rows <= 0:
        raise ValueError("the slice needs a positive --rows")
    wanted = sorted(read_manifest(LOOK), key=row_key)[:rows]
    decoders = load_models(parts_dir)
    word_counts = ayah_word_counts()
    started = time.time()
    timings = _decode_rows(wanted, parts_dir=parts_dir, cache_dir=cache_dir,
                           decoders=decoders, word_counts=word_counts, on_download=on_download)
    verdict = slice_verdict(sum(timing["audio_seconds"] for timing in timings),
                            round(time.time() - started, 1),
                            total_audio_seconds() if total_audio_s is None else total_audio_s,
                            limit_hours)
    record = {"rows": sum(timing["items"] for timing in timings),
              "shards": len(timings), **verdict}
    write_json(record, Path(parts_dir) / "slice_timing.json")
    return record


def decode_shards(*, parts_dir: Path, cache_dir: Path, shard_index: int = 0,
                  shard_count: int = 1,
                  on_download: Callable[[str], None] | None = None) -> dict:
    """Decode every look row of the shards ``sorted(shards)[shard_index::shard_count]``.

    Each shard is streamed once for all four models; processes with disjoint shard subsets
    write disjoint part files, so they share no state but the resume.
    """
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"--shard-index must be in [0, {shard_count})")
    rows = read_manifest(LOOK)
    shards = sorted({row["shard_file"] for row in rows})
    chosen = set(shards[shard_index::shard_count])
    selected = [row for row in rows if row["shard_file"] in chosen]
    timings = _decode_rows(selected, parts_dir=parts_dir, cache_dir=cache_dir,
                           decoders=load_models(parts_dir),
                           word_counts=ayah_word_counts(), on_download=on_download)
    return {"shards": len(timings), "items": sum(timing["items"] for timing in timings),
            "audio_seconds": sum(timing["audio_seconds"] for timing in timings),
            "decode_seconds": {name: {store: round(sum(timing["decode_seconds"][name][store]
                                                       for timing in timings), 1)
                                      for store in STORES}
                               for name in MODELS}}


def timing_totals(parts_dir: Path) -> dict:
    """The parts' per-shard timing lines summed: shards, items, audio, and decode seconds."""
    path = Path(parts_dir) / "timing.jsonl"
    lines = read_jsonl(path) if path.exists() else []
    return {"shards": len(lines),
            "items": sum(line["items"] for line in lines),
            "audio_seconds": round(sum(line["audio_seconds"] for line in lines), 1),
            "summed_wall_seconds": round(sum(line["wall_seconds"] for line in lines), 1),
            "decode_seconds": {model: {store: round(sum(line["decode_seconds"][model][store]
                                                        for line in lines), 1)
                                       for store in STORES}
                               for model in MODELS}}


def run_provenance(parts_dir: Path, pull_request: int | None = None) -> dict:
    """What ``meta.json`` records of the run: code, models, the look manifest and timings."""
    return {**provenance(head_sha(), pull_request),
            "base_revision": PINNED_BASE_REVISION,
            "checkpoint_sha256": dict(PINNED_SHA256),
            "manifest_commit": commit_of(TRANCHES_DIR / f"{LOOK}.jsonl"),
            "tranche_meta_sha256": sha256_file(TRANCHES_DIR / "meta.json"),
            "timings": timing_totals(parts_dir)}


# --- CLI ----------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    slice_ = commands.add_parser("slice", help="time the look decode on a slice of rows")
    slice_.add_argument("--parts-dir", type=Path, required=True)
    slice_.add_argument("--cache-dir", type=Path, required=True)
    slice_.add_argument("--rows", type=int, default=SLICE_ROWS)
    slice_.add_argument("--limit-hours", type=float, default=LIMIT_HOURS)
    decode_ = commands.add_parser("decode", help="decode a shard subset of the look tranche")
    decode_.add_argument("--parts-dir", type=Path, required=True)
    decode_.add_argument("--cache-dir", type=Path, required=True)
    decode_.add_argument("--shard-index", type=int, default=0)
    decode_.add_argument("--shard-count", type=int, default=1)
    merge_ = commands.add_parser("merge", help="merge the parts into the committed decodes")
    merge_.add_argument("--parts-dir", type=Path, required=True)
    merge_.add_argument("--out-dir", type=Path, default=DECODES_ROOT)
    merge_.add_argument("--pr", type=int, default=None,
                        help="the PR whose refs/pull/<N>/head keeps the cited code SHA")
    args = parser.parse_args()
    if args.command == "slice":
        record = time_slice(parts_dir=args.parts_dir, cache_dir=args.cache_dir,
                            rows=args.rows, limit_hours=args.limit_hours)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        # 3 is the launcher's "over budget, post and wait for a decision" code.
        raise SystemExit(0 if record["within_budget"] else 3)
    if args.command == "decode":
        result = decode_shards(parts_dir=args.parts_dir, cache_dir=args.cache_dir,
                               shard_index=args.shard_index, shard_count=args.shard_count)
    else:
        result = merge_decodes(args.parts_dir, args.out_dir,
                               manifest_path=TRANCHES_DIR / f"{LOOK}.jsonl",
                               provenance=run_provenance(args.parts_dir, args.pr))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

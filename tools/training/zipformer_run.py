"""The #245 zipformer screen's steps on the run box: fetch, disjointness, card gate.

The code layer (:mod:`training.zipformer_decode`, :mod:`training.zipformer_gate`,
:mod:`tadabur.benchmark_overlap`) is pure and torch-free; this module is the driver that
fetches the card's files, runs the pass over the audio and writes the provenance the later
steps and the report read. Every step runs on cuda-dev, where the gated Hub token and the
tlog fingerprint store live, and every step writes its result inside the checkout:

* **fetch** — the v3.2 fp32 ONNX (chunk 24), its int8 file (a numerics line only), the card's
  ``quran_text2phoneme.json``, the benchmark manifest and its 600 clips, and Tadabur's
  ``sheikh_dict.json``, at the Hub revisions read now. Every file's SHA-256 is pinned in
  ``fetch.json`` in the work directory; the model files' hashes reach the committed identity
  through :meth:`training.zipformer_decode.ZipformerFiles.identity`. The benchmark audio
  stays in the work directory and is never committed (its licence is research/evaluation).
* **disjointness** — all 600 benchmark clips are fingerprinted
  (:func:`tadabur.qud_fingerprint.fingerprint`) and queried against **every** tlog row, not
  only ``tlog.training``, with :class:`tadabur.tlog_overlap.EvalIndex`, its line and vote
  threshold. Each hit names the row's pre-registered role; `tadabur.benchmark_overlap` turns
  them into the plan every later decode follows: a clip matching a sealed row is dropped
  before any decode, a clip matching a row ``tlog.training`` consumed is decoded but excluded
  from the verdict, everything else counts. The per-source reciter-name screen and the
  "Tadabur audio not read" note are recorded beside the table (#156's precedent).
* **gate** — the card's own unit PER (its table, its normalization, the units' greedy
  longest-match stand-in) on every clip the plan allows, at both pre-declared front-end
  scales. The scale is pinned by the pre-declared rule, both scales' PER are kept, and the
  round trip must hold before the identity is written; a failure writes the evidence and stops
  with exit code 3. Nothing outside the plan's decode list is ever decoded.

Usage (from ``tools/``, on the run box)::

    python -m training.zipformer_run fetch --work-dir /root/scratch/zipformer
    python -m training.zipformer_run disjointness --work-dir /root/scratch/zipformer
    python -m training.zipformer_run gate --work-dir /root/scratch/zipformer --workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tadabur.benchmark_overlap import plan_disjointness
from tadabur.tlog_index import sha256_file, write_json
from training.zipformer_decode import (
    BENCHMARK_REPO,
    IDENTITY_PATH,
    INT8_FILE,
    MANIFEST_FILE,
    MODEL_REPO,
    ONNX_FILE,
    TEXT2PHONEME_FILE,
    ZIPFORMER_DIR,
    ZipformerFiles,
    load_zipformer,
)
from training.zipformer_gate import (
    CARD_PER,
    GATE_LIMIT_POINTS,
    GateStop,
    card_table,
    character_per,
    pin_scale,
    reference_units,
    require_card_reproduction,
    require_roundtrip,
    roundtrip_identical,
    roundtrip_share,
    source_per,
)

#: The run box's scratch: the fetched files, the benchmark audio and the fetch record. Nothing
#: here is committed (the card's benchmark is research/evaluation-use only).
DEFAULT_WORK_DIR = Path("/root/scratch/zipformer")
#: The fetch record in the work directory; every later step refuses a changed file.
FETCH_NAME = "fetch.json"
#: The committed outputs of the disjointness and gate steps.
DISJOINTNESS_PATH = ZIPFORMER_DIR / "disjointness.json"
GATE_PATH = ZIPFORMER_DIR / "card_gate.json"
#: The card's PER sources, by the benchmark's own ``source`` column (its README).
CARD_SOURCES = {"everyayah_heldout": "studio", "tlog_holdout": "phone",
                "qul_alnufais": "unseen"}
#: The two front-end scales the gate decides between, pre-declared (architect r1, Q4).
SCALES = (1.0, 32_768.0)
#: The tlog fingerprint store (``tadabur.tlog_overlap.STORE_DIR`` on cuda-dev).
DEFAULT_STORE_DIR = Path("/root/scratch/tlog/fp")
#: Tadabur's name dictionary, its dataset and the benchmark reciter it can look up. The
#: benchmark publishes only "Al-Nufais" for its QUL source; the dictionary's only name
#: carrying النفيس is احمد النفيس.
TADABUR_DATASET = "FaisaI/tadabur"
QUL_RECITER_NAME = "احمد النفيس"
#: One ONNX thread per worker: the gate parallelises whole clips across processes.
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
#: The benchmark's composition (its README): 200 clips per source, 600 in all.
CLIPS_PER_SOURCE = 200


# --- the fetch record ------------------------------------------------------------------------


def manifest_rows(path: Path) -> list[dict]:
    """The benchmark manifest's rows, each with an id, a known source, a text and audio path.

    The published v1.1 manifest repeats one row verbatim (``tlog_holdout/11_90_2170378964``,
    same audio and text): the clip is counted once, and ``fetch`` records the collapse. The
    same id with *different* content is refused: that is not a repeated clip but a collision
    between two.
    """
    rows: dict[str, dict] = {}
    for row in _printed_rows(path):
        missing = {"id", "source", "reference_text", "audio"} - set(row)
        if missing:
            raise ValueError(f"{path}: {row.get('id')!r} lacks {sorted(missing)}")
        if row["source"] not in CARD_SOURCES:
            raise ValueError(f"{path}: {row['id']} has the unknown source {row['source']!r}")
        seen = rows.get(row["id"])
        if seen is None:
            rows[row["id"]] = row
        elif seen != row:
            raise ValueError(f"{path} names {row['id']!r} twice with different rows")
    return list(rows.values())


def _printed_rows(path: Path) -> list[dict]:
    """The manifest's rows as printed, exact duplicates included."""
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_entry(work_dir: Path, path: Path) -> dict:
    """One fetched file's work-dir-relative path, byte count and SHA-256."""
    resolved = Path(path).resolve()
    return {"path": str(resolved.relative_to(Path(work_dir).resolve())),
            "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _verify_fetch(work_dir: Path, record: Mapping) -> None:
    """Refuse a work directory whose fetched files no longer match the pinned bytes."""
    entries = [*record["files"].values(), record["manifest"], record["sheikh_dict"]]
    for entry in entries:
        path = Path(work_dir) / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path} does not match the SHA-256 the fetch pinned")


def load_fetch(work_dir: Path) -> dict:
    """The fetch record, verified file by file; a missing one names the step to run first."""
    path = Path(work_dir) / FETCH_NAME
    if not path.is_file():
        raise ValueError(f"{path} is missing; run the fetch step first")
    record = json.loads(path.read_text(encoding="utf-8"))
    _verify_fetch(work_dir, record)
    return record


def files_from_record(work_dir: Path, record: Mapping) -> ZipformerFiles:
    """The card's files a fetch record names, as the loader takes them."""
    base = Path(work_dir)
    return ZipformerFiles(
        onnx=base / record["files"][ONNX_FILE]["path"],
        int8=base / record["files"][INT8_FILE]["path"],
        text2phoneme=base / record["files"][TEXT2PHONEME_FILE]["path"],
        manifest=base / record["manifest"]["path"],
        revision=record["model"]["revision"],
        benchmark_revision=record["benchmark"]["revision"],
    )


def _check_committed_card_files(model_dir: Path) -> None:
    """The committed symbol table and unit inventory must be the fetched revision's own."""
    for name, committed in (("tokens.txt", ZIPFORMER_DIR / "tokens.txt"),
                            ("phoneme_units.json", ZIPFORMER_DIR / "phoneme_units.json")):
        if sha256_file(model_dir / name) != sha256_file(committed):
            raise ValueError(f"{committed} is not the card's {name} at the pinned revision")


def fetch(work_dir: Path) -> dict:
    """Fetch and pin the card's files, the benchmark and Tadabur's name dictionary.

    A work directory that already holds a fetch record is verified rather than refreshed: the
    revisions are pinned on the first run, so a card update never moves a running screen.
    """
    work_dir = Path(work_dir)
    fetch_path = work_dir / FETCH_NAME
    if fetch_path.is_file():
        record = load_fetch(work_dir)
        print(f"fetch: {fetch_path} is pinned; every file still matches its SHA-256")
        return record

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    model = api.model_info(MODEL_REPO)
    benchmark = api.dataset_info(BENCHMARK_REPO)
    tadabur = api.dataset_info(TADABUR_DATASET)
    model_dir = work_dir / "model"
    benchmark_dir = work_dir / "benchmark"

    files = {}
    for name in (ONNX_FILE, INT8_FILE, TEXT2PHONEME_FILE, "tokens.txt", "phoneme_units.json"):
        path = Path(hf_hub_download(MODEL_REPO, name, revision=model.sha, local_dir=model_dir))
        files[name] = _file_entry(work_dir, path)
    _check_committed_card_files(model_dir)

    snapshot_download(BENCHMARK_REPO, repo_type="dataset", revision=benchmark.sha,
                      local_dir=benchmark_dir, allow_patterns=[MANIFEST_FILE, "audio/**"])
    manifest = benchmark_dir / MANIFEST_FILE
    printed = _printed_rows(manifest)
    rows = manifest_rows(manifest)
    counts = Counter(row["source"] for row in printed)
    wanted = {source: CLIPS_PER_SOURCE for source in CARD_SOURCES}
    if dict(counts) != wanted:
        raise ValueError(f"{manifest} holds {dict(counts)} clips; the benchmark is {wanted}")
    for row in rows:
        audio = manifest.parent / row["audio"]
        if not audio.is_file():
            raise ValueError(f"{row['id']}: the fetched benchmark has no {audio}")

    sheikh = Path(hf_hub_download(TADABUR_DATASET, "sheikh_dict.json", repo_type="dataset",
                                  revision=tadabur.sha, local_dir=work_dir / "tadabur"))

    record = {
        "schema_version": 1,
        "issue": 245,
        "model": {"repo": MODEL_REPO, "revision": model.sha},
        "benchmark": {"repo": BENCHMARK_REPO, "revision": benchmark.sha},
        "tadabur": {"repo": TADABUR_DATASET, "revision": tadabur.sha},
        "files": files,
        "manifest": {**_file_entry(work_dir, manifest), "clips": len(rows),
                     "rows": len(printed), "duplicate_rows": len(printed) - len(rows),
                     "by_source": dict(sorted(counts.items()))},
        "sheikh_dict": _file_entry(work_dir, sheikh),
    }
    write_json(record, fetch_path)
    print(f"fetch: pinned {len(files)} model files, {len(rows)} benchmark clips, "
          f"sheikh_dict.json -> {fetch_path}")
    return record


# --- disjointness ----------------------------------------------------------------------------


def _row_id(row: Mapping) -> str:
    """A tlog row's identity, in the form ``training.tlog_pool.rows_digest`` hashes."""
    return f"{row['split']}:{row['shard_file']}:{row['row']}"


def name_screen(rows: Sequence[Mapping], sheikh: Mapping[str, int],
                uses_of: Mapping[int, Mapping[str, int]]) -> dict:
    """The benchmark's reciter names against Tadabur's dictionary and exposure registry.

    Only the QUL source publishes a name the dictionary can hold. The studio source's three
    EveryAyah reciters and the phone source's tlog recordings are unnamed in the benchmark, so
    they get no name screen: names prove nothing anyway, and no absence of a match is claimed.
    """
    from tadabur.qud_selection import tadabur_screen

    screen = {}
    for source in sorted({row["source"] for row in rows}):
        if source == "qul_alnufais":
            screen[source] = {
                "reciter": "Al-Nufais (the source's name)",
                "query_name": QUL_RECITER_NAME,
                "tadabur": tadabur_screen(QUL_RECITER_NAME, sheikh, uses_of),
            }
        elif source == "everyayah_heldout":
            screen[source] = {
                "reciter": "3 EveryAyah reciters, unnamed in the benchmark",
                "tadabur": None,
                "note": "no published name to look up; names prove nothing (qud_overlap.py)",
            }
        elif source == "tlog_holdout":
            screen[source] = {
                "reciter": "anonymous tlog recordings (recording ids only)",
                "tadabur": None,
                "note": "no published name to look up; names prove nothing (qud_overlap.py)",
            }
        else:
            raise ValueError(f"no name screen for the source {source!r}")
    return screen


def _clip_prints(rows: Sequence[Mapping], manifest: Path) -> dict[str, tuple[str, object]]:
    """Each clip's audio SHA-256 and fingerprint (the bytes stay in the work directory)."""
    from tadabur.audio import decode_to_mono_16k, read_audio_bytes
    from tadabur.qud_fingerprint import fingerprint

    prints = {}
    for row in rows:
        raw = read_audio_bytes(manifest.parent / row["audio"])
        prints[row["id"]] = (hashlib.sha256(raw).hexdigest(),
                             fingerprint(decode_to_mono_16k(raw)))
    return prints


def disjointness(work_dir: Path, *, store_dir: Path = DEFAULT_STORE_DIR) -> dict:
    """Fingerprint every benchmark clip against every tlog row and write the decode plan.

    The tlog index is built one split at a time (the store is 2.3 GB of fingerprints; one
    index over both splits would hold three arrays of every frame at once), and every clip is
    queried against each. A row's role comes from the committed freeze, not from a re-derived
    hash, and ``tlog.training``'s consumed rows are the verified registry row set.
    """
    from tadabur.exposure import load_registry
    from tadabur.qud_selection import tadabur_uses_of
    from tadabur.tlog_overlap import EvalIndex, iter_pool_rows, load_tlog_prints
    from tadabur.tlog_pool import load_training_rows, rows_digest

    fetch_record = load_fetch(work_dir)
    files = files_from_record(work_dir, fetch_record)
    rows = manifest_rows(files.manifest)
    clip_prints = _clip_prints(rows, files.manifest)

    roles: dict[str, str] = {}
    pool_rows: list[dict] = []
    matched: dict[str, list[dict]] = {row["id"]: [] for row in rows}

    def query(prints: Mapping[str, object], split: str) -> None:
        if not prints:
            return
        index = EvalIndex.build(prints)
        print(f"disjointness: {split}: indexed {len(index)} frames of {len(prints)} rows",
              flush=True)
        for row in rows:
            for hit in index.query(clip_prints[row["id"]][1]):
                matched[row["id"]].append({"role": roles[hit.clip], "row": hit.clip,
                                           "ber": hit.ber})

    prints: dict[str, object] = {}
    indexed_rows = 0
    shard = None
    split = None
    store: Mapping[str, object] = {}
    for row in iter_pool_rows():
        current = (row["split"], row["shard_file"])
        if current != shard:
            if row["split"] != split:
                query(prints, split)
                prints = {}
                split = row["split"]
            store = load_tlog_prints(Path(store_dir) / f"{current[1]}.npz")
            shard = current
        key = _row_id(row)
        roles[key] = row["role"]
        pool_rows.append({"split": row["split"], "shard_file": row["shard_file"],
                          "row": row["row"], "role": row["role"], "usable": row["usable"]})
        found = store.get(row["audio_path"])
        if found is not None and len(found):
            prints[key] = found
            indexed_rows += 1
    query(prints, split)
    del prints

    training = load_training_rows(pool_rows=pool_rows)
    del pool_rows
    consumed = {_row_id(row) for row in training}
    for hits in matched.values():
        hits.sort(key=lambda hit: (hit["ber"], hit["row"]))
    hits = {clip: [(hit["role"], hit["row"]) for hit in found]
            for clip, found in matched.items()}
    plan = plan_disjointness([row["id"] for row in rows], hits, consumed)

    sheikh = json.loads((Path(work_dir) / fetch_record["sheikh_dict"]["path"])
                        .read_text(encoding="utf-8-sig"))
    screen = name_screen(rows, sheikh, tadabur_uses_of(load_registry()))

    sealed = set(plan.sealed)
    trained = set(plan.trained)
    clip_table = [{
        "id": row["id"],
        "source": row["source"],
        "audio_sha256": clip_prints[row["id"]][0],
        "hits": matched[row["id"]],
        "outcome": "sealed" if row["id"] in sealed else
                   ("trained" if row["id"] in trained else "verdict"),
    } for row in rows]

    printed_rows = _printed_rows(files.manifest)
    record = {
        "schema_version": 1,
        "issue": 245,
        "benchmark": {"repo": BENCHMARK_REPO, "revision": files.benchmark_revision,
                      "manifest_sha256": sha256_file(files.manifest), "clips": len(rows),
                      "rows": len(printed_rows),
                      "duplicate_rows": len(printed_rows) - len(rows)},
        "tlog": {
            "store_dir": str(store_dir),
            "rows": len(roles),
            "indexed_rows": indexed_rows,
            "unfingerprinted_rows": len(roles) - indexed_rows,
            "training": {"use": "tlog.training", "rows": len(training),
                         "rows_sha256": rows_digest(training)},
        },
        "plan": {"decode": list(plan.decode), "verdict": list(plan.verdict),
                 "trained": list(plan.trained),
                 "sealed": {clip: list(hit_roles) for clip, hit_roles in plan.sealed.items()},
                 "by_role": plan.by_role},
        "name_screen": screen,
        "tadabur_audio": {
            "read": False,
            "reason": "professional reciters against app audio: #156's precedent, and the "
                      "Tadabur shards are not fingerprinted in this store",
        },
        "clips": clip_table,
    }
    write_json(record, DISJOINTNESS_PATH)
    print(f"disjointness: {len(plan.decode)} decodable ({len(plan.verdict)} verdict, "
          f"{len(plan.trained)} trained), {len(plan.sealed)} sealed -> {DISJOINTNESS_PATH}")
    return record


# --- the card gate ---------------------------------------------------------------------------


def _card_inventory() -> list[str]:
    """The card's non-blank units, checked equal to the symbol table's emitted strings."""
    from training.zipformer_decode import read_tokens

    units = json.loads((ZIPFORMER_DIR / "phoneme_units.json").read_text(encoding="utf-8"))
    if units.get("<blank>") != 0:
        raise ValueError("phoneme_units.json does not put <blank> at 0")
    tokens = read_tokens(ZIPFORMER_DIR / "tokens.txt")
    inventory = {unit for unit in units if unit != "<blank>"}
    emitted = set(tokens.values()) - {"<blank>"}
    if inventory != emitted:
        raise ValueError("tokens.txt and phoneme_units.json hold different unit strings")
    return sorted(inventory)


_WORKER: dict[str, object] = {}


def _start_gate_worker(files: ZipformerFiles, scale: float) -> None:
    """One worker's session and decoder; the gate parallelises whole clips across processes."""
    import torch

    torch.set_num_threads(1)
    _WORKER["decoder"] = load_zipformer(files, scale=scale)


def _decode_clip(item: tuple[str, str]) -> tuple[str, list[str]]:
    """One clip's emitted units, strings from ``tokens.txt`` (the arm the gate scores)."""
    from tadabur.audio import decode_to_mono_16k, read_audio_bytes

    clip_id, audio = item
    decoder = _WORKER["decoder"]
    samples = decode_to_mono_16k(read_audio_bytes(Path(audio)))
    return clip_id, [emission.text for emission in decoder.stream(samples)]


def _decode_items(files: ZipformerFiles, scale: float, items: Sequence[tuple[str, str]],
                  workers: int, label: str) -> dict[str, list[str]]:
    """Every item's units at ``scale``, decoded in parallel processes."""
    decoded: dict[str, list[str]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_start_gate_worker,
                             initargs=(files, scale)) as pool:
        for done, (clip_id, units) in enumerate(pool.map(_decode_clip, items, chunksize=2), 1):
            decoded[clip_id] = units
            if done % 100 == 0:
                print(f"  {label}: {done}/{len(items)} clips", flush=True)
    if len(decoded) != len(items):
        raise ValueError(f"{label}: decoded {len(decoded)} of {len(items)} clips")
    return decoded


def _read_plan(files: ZipformerFiles) -> dict:
    """The disjointness plan, checked against this manifest; no plan means no decode."""
    if not DISJOINTNESS_PATH.is_file():
        raise ValueError(f"{DISJOINTNESS_PATH} is missing; run the disjointness step first")
    record = json.loads(DISJOINTNESS_PATH.read_text(encoding="utf-8"))
    if record["benchmark"]["manifest_sha256"] != sha256_file(files.manifest):
        raise ValueError(f"{DISJOINTNESS_PATH} screened another benchmark manifest")
    plan = record["plan"]
    if set(plan["decode"]) & set(plan["sealed"]):
        raise ValueError("the plan both decodes and drops a sealed clip")
    return plan


def gate(work_dir: Path, *, workers: int = DEFAULT_WORKERS) -> dict:
    """Reproduce the card's PER, pin the front-end scale, and write the identity.

    Every clip the plan allows is decoded at both scales. The scale is pinned first, then the
    round trip is checked before any PER is trusted (the golden units are the stand-in
    tokenizer's, so a PER against a different segmentation means nothing), then the card's
    unit PER is reproduced: pooled edits over pooled reference units per source, a clip whose
    reference the card's table lacks counted as a miss. The character PER is recorded beside
    it so a tokenizer mismatch can be told apart from a broken decoder. A failure writes the
    evidence and is a stop: :class:`training.zipformer_gate.GateStop`.
    """
    files = files_from_record(work_dir, load_fetch(work_dir))
    rows = manifest_rows(files.manifest)
    plan = _read_plan(files)
    by_id = {row["id"]: row for row in rows}
    decode_ids = set(plan["decode"])
    unknown = decode_ids - set(by_id)
    if unknown:
        raise ValueError(f"the plan names clips this manifest lacks: {sorted(unknown)}")
    decoded_rows = [row for row in rows if row["id"] in decode_ids]

    raw = json.loads(files.text2phoneme.read_text(encoding="utf-8"))
    table = card_table(raw)
    inventory = _card_inventory()

    items = [(row["id"], str((files.manifest.parent / row["audio"]).resolve()))
             for row in decoded_rows]
    decoded: dict[float, dict[str, list[str]]] = {}
    scales: dict[str, dict] = {}
    pinned: float | None = None
    base = {
        "schema_version": 1,
        "issue": 245,
        "card": CARD_PER,
        "limit_points": GATE_LIMIT_POINTS,
        "sources": CARD_SOURCES,
        "clips": {"manifest": len(rows), "decoded": len(decoded_rows),
                  "sealed_dropped": len(plan["sealed"]), "trained": len(plan["trained"])},
        "identity": str(IDENTITY_PATH.relative_to(ZIPFORMER_DIR.parents[2])),
    }
    try:
        for scale in SCALES:
            print(f"gate: decoding {len(items)} clips at scale {scale}", flush=True)
            predictions = _decode_items(files, scale, items, workers, f"scale {scale}")
            decoded[scale] = predictions
            scoring = [{"source": CARD_SOURCES[row["source"]],
                        "gold": reference_units(table, row["reference_text"], inventory),
                        "pred": predictions[row["id"]]} for row in decoded_rows]
            scales[str(scale)] = {
                "unit_per": source_per(scoring),
                "character_per": character_per(scoring),
                "roundtrip": roundtrip_share(predictions.values(), inventory),
            }
            measured = scales[str(scale)]["unit_per"]
            print(f"  scale {scale}: " + ", ".join(
                f"{source} {entry['per']:.2f}" if entry["per"] is not None else
                f"{source} n/a" for source, entry in sorted(measured.items())), flush=True)

        pinned = pin_scale({
            scale: {source: entry["per"]
                    for source, entry in scales[str(scale)]["unit_per"].items()}
            for scale in SCALES
        })
        scales[str(pinned)]["roundtrip"]["failed_clips"] = sorted(
            clip for clip, units in decoded[pinned].items()
            if not roundtrip_identical(units, inventory))
        require_roundtrip(decoded[pinned].values(), inventory)
        require_card_reproduction(scales[str(pinned)]["unit_per"])
    except GateStop as stop:
        write_json({**base, "scales": scales, "pinned_scale": pinned, "passed": False,
                    "stop": str(stop)}, GATE_PATH)
        raise

    load_zipformer(files, scale=pinned, identity_path=IDENTITY_PATH)
    record = {**base, "scales": scales, "pinned_scale": pinned, "passed": True}
    write_json(record, GATE_PATH)
    print(f"gate: PASSED at scale {pinned}; identity pinned at {IDENTITY_PATH}")
    return record


# --- the command line ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step; the card gate's own refusal exits 3, any other failure propagates."""
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", parents=[common])
    disjoint = commands.add_parser("disjointness", parents=[common])
    disjoint.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    card = commands.add_parser("gate", parents=[common])
    card.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)

    try:
        if args.command == "fetch":
            fetch(args.work_dir)
        elif args.command == "disjointness":
            disjointness(args.work_dir, store_dir=args.store_dir)
        else:
            gate(args.work_dir, workers=args.workers)
    except GateStop as stop:
        print(f"STOP: {stop}", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

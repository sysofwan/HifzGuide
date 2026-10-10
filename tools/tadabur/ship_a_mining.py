"""Ship A's mining extension (#126, D9): more ``h448`` partner swaps from audio it never saw.

The mining pool (#83) holds 73 carriers where ``h448`` commits a soft pair's partner letter
(more on the owner's two added pairs, ``س↔ش`` and ``ط↔ق``). Too few for a claim per pair, so
this module decodes more ``h448``-unseen audio and keeps the clips with a partner swap.

**Frame.** Every row of the shards ``h448`` never trained on (every shard outside the
exposure registry's ``h448.training`` use: the held-out block 0-20 and the strided reserve)
that the pool's own rule admits
(:func:`tadabur.mining_pool.eligible`: not in ``decode_evalset``, 1.5-50 s, a phonetizable
ayah), less: clips already staged for any use (the pool among them); the sealed panel's
reciters (:func:`tadabur.exposure.excluded_reciters`); and recordings the exposure registry
ties to a row of ``h448``'s training shards (a checksum alias or a ``copy-screen-v2``
probable copy), which ``h448`` may have trained on. No decode decides the frame.

**Scan.** Shards are taken in ``sha256("<salt>:shard:<n>")`` order and each is scanned
whole: every frame row decoded from exactly the samples its staged file would hold, by
``h448`` streamed at b = 0 (fp32 weights, as :mod:`tadabur.pool_stream`) and by the base
teacher (whole span, bf16, as #83's scan). A clip is a **hit** when ``h448`` commits the
partner letter of a target pair (the nine, :data:`tadabur.truth_sites.ALL_TARGET_PAIRS`)
at a carrier of the clip's whole-ayah reference
(:func:`tadabur.contrast_attribution.contrast_sites`). Hits are staged (WAV and registry
row, use ``ship_a_mining``). The scan stops between shards once a time budget is spent, so
the scanned shards are a simple random sample of the frame's: a scanned clip's inclusion
probability is ``S / N`` (``S`` of the ``N`` frame shards), conditional on the frame.

**Build.** The hits are segmented as the pool was (``tadabur.resegment``), then
:func:`build` writes a manifest of the pool's shape (:class:`tadabur.mining_pool.PoolClip`,
stratum :data:`HIT`) with the base teacher's segment decodes, and :func:`stream` the
``h448`` stream cut per kept segment. Ship A (:mod:`tadabur.ship_a`) mines its partner
swaps from them like the pool's.

Files (``tools/tadabur/ship_a/mining/``): ``frame.json`` (eligible and excluded rows per
shard, the shard order, which shards were scanned, per reciter its frame and scanned
clips), ``scan.jsonl`` (one row per scanned clip: provenance and the partner letters each
decode committed per pair), ``clips.jsonl``, ``base_decodes.json``,
``h448_stream_decodes.json`` and ``summary.json`` (yield per pair, GPU time). A hit that
is byte for byte a recording already staged (a pool clip, or an earlier hit) is left out of
the manifest, so each recording is mined once.

Usage (on the GPU box, from ``tools/``, under the shared GPU lock; stage dir of this run)::

  flock /root/scratch/gpu.lock python -m tadabur.ship_a_mining scan --index <shard index> \\
      --evalset-manifest tadabur/exposure/sources/decode_evalset.manifest.json \\
      --checkpoint <h448> --shard-cache <dir> --stage-dir <dir> --max-hours 3
  python -m tadabur.ship_a_mining register --stage-dir <dir>      # on the Mac: registry rows
  flock … python -m tadabur.resegment --registry tadabur/staged_audio/clips.jsonl \\
      --use ship_a_mining --audio-dir <dir>/clips --out-dir <dir>/seg
  python -m tadabur.ship_a_mining build --stage-dir <dir>
  flock … python -m tadabur.ship_a_mining stream --checkpoint <h448> --audio-dir <dir>/clips
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .contrast_attribution import contrast_sites
from .mining_pool import (
    PoolClip,
    Selected,
    build_manifest,
    eligible,
    load_manifest,
    read_evalset_rows,
    segment_key,
    whole_ayah_references,
    write_manifest,
)
from .power_frame import SHIP_A_DIR, PoolFiles
from .staged_audio import (
    REGISTRY_PATH,
    SHIP_A_MINING,
    IndexRow,
    StagedClip,
    load_staged_clips,
    read_shard_index,
    read_staged_clips,
    write_staged_clips,
)
from .truth_sites import ALL_TARGET_PAIRS

MINING_DIR = SHIP_A_DIR / "mining"
FRAME_PATH = MINING_DIR / "frame.json"
SCAN_PATH = MINING_DIR / "scan.jsonl"
CLIPS_PATH = MINING_DIR / "clips.jsonl"
BASE_DECODES_PATH = MINING_DIR / "base_decodes.json"
H448_DECODES_PATH = MINING_DIR / "h448_stream_decodes.json"
SUMMARY_PATH = MINING_DIR / "summary.json"

SALT = "ship-a-mining-v1"
#: The manifest's one stratum: a scanned clip where ``h448`` committed a partner letter.
HIT = "h448_partner_swap"
FILES = PoolFiles(CLIPS_PATH, BASE_DECODES_PATH, H448_DECODES_PATH, SHIP_A_MINING, (HIT,))

#: A frame exclusion beyond the pool's own rule.
ALREADY_STAGED = "already_staged"
PANEL_RECITER = "panel_reciter"
COPY_IN_H448_TRAINING = "copy_in_h448_training"


def shard_order(shards: Iterable[int]) -> list[int]:
    """The order shards are scanned in: a salted hash, so any prefix is a random sample."""
    return sorted(shards, key=lambda s: hashlib.sha256(f"{SALT}:shard:{s}".encode()).hexdigest())


def h448_training_copies(registry) -> set[tuple[int, int]]:
    """Every ``(shard, row)`` the exposure registry ties to a row of ``h448``'s training
    shards: a checksum alias or a probable copy (the use is shard level, membership exact)."""
    from .exposure import H448_TRAINING

    training = set(registry.shard_uses[H448_TRAINING].shards)
    return {source for source in registry.related_sources()
            if any(shard in training for shard, _ in registry.same_recording(source) - {source})}


def unseen_shards(registry) -> list[int]:
    """Every Tadabur shard outside ``h448``'s training shards (an exact shard use)."""
    from .exposure import H448_TRAINING
    from .truth_sites import NUM_SHARDS

    return sorted(set(range(NUM_SHARDS)) - set(registry.shard_uses[H448_TRAINING].shards))


def mining_frame(
    rows: Iterable[IndexRow], evalset_rows: set[tuple[int, int]], staged: Iterable[str],
    barred: frozenset[int], copies: set[tuple[int, int]], shards: Iterable[int],
) -> tuple[list[IndexRow], list[tuple[IndexRow, str]]]:
    """The frame (module docstring) and every row of ``shards`` left out, with why."""
    kept, excluded = eligible(rows, evalset_rows, shards)
    staged = set(staged)
    frame = []
    for row in kept:
        if row.audio_filename in staged:
            excluded.append((row, ALREADY_STAGED))
        elif row.reciter_id in barred:
            excluded.append((row, PANEL_RECITER))
        elif (row.shard, row.row_index) in copies:
            excluded.append((row, COPY_IN_H448_TRAINING))
        else:
            frame.append(row)
    return frame, excluded


def partner_commits(decode: str, reference: str) -> dict[str, int]:
    """Per target pair, the carriers where ``decode`` has the pair's other letter."""
    found = {pair: len(contrast_sites(decode, reference, pair)) for pair in sorted(ALL_TARGET_PAIRS)}
    return {pair: n for pair, n in found.items() if n}


def frame_record(
    frame: list[IndexRow], excluded: list[tuple[IndexRow, str]], order: list[int],
    scanned: list[int],
) -> dict:
    """Per shard its eligible and excluded rows, the scan order and the scanned shards, and
    per reciter its frame and scanned clips."""
    per_shard: dict[int, Counter] = {shard: Counter() for shard in order}
    for row in frame:
        per_shard[row.shard]["eligible"] += 1
    for row, reason in excluded:
        per_shard[row.shard][reason] += 1
    per_reciter: dict[int, Counter] = {}
    for row in frame:
        counts = per_reciter.setdefault(row.reciter_id, Counter())
        counts["frame"] += 1
        counts["scanned"] += row.shard in scanned
    return {
        "salt": SALT,
        "shard_order": order,
        "scanned_shards": scanned,
        "inclusion_probability": len(scanned) / len(order),
        "per_shard": {str(s): dict(sorted(c.items())) for s, c in per_shard.items()},
        "per_reciter": {str(r): dict(c) for r, c in sorted(per_reciter.items())},
        "frame_clips": len(frame),
        "frame_reciters": len(per_reciter),
        "excluded": dict(sorted(Counter(reason for _, reason in excluded).items())),
    }


# --- scan (GPU) ------------------------------------------------------------------------------


def _scan(args) -> None:
    """Scan whole shards in order until ``--max-hours`` is spent (module docstring).

    Resumable: ``scan.jsonl`` and ``staged.jsonl`` take a shard's rows only once the shard
    is done, and ``run.json`` lists the done shards with the GPU seconds spent."""
    from training.decoding import DEPLOYED_BLOCK, SPANS, Decoder, stream_protocol

    from .audio import decode_to_mono_16k
    from .dataset_source import resolve_audio_filename
    from .exposure import SHIP_A_MINING as EXPOSURE_USE
    from .exposure import excluded_reciters, load_registry
    from .resegment import BASE_TEACHER, DECODE_BATCH_SIZE, WEIGHTS_DTYPE
    from .shard_reader import iter_shard_rows
    from .staged_audio import as_staged, stage_row

    started = time.monotonic()
    registry = load_registry()
    shards = unseen_shards(registry)
    frame, excluded = mining_frame(
        read_shard_index(args.index).values(), read_evalset_rows(args.evalset_manifest),
        load_staged_clips(), excluded_reciters(EXPOSURE_USE, registry),
        h448_training_copies(registry), shards,
    )
    order = shard_order(shards)
    references = whole_ayah_references(row.surah_ayah for row in frame)
    h448 = Decoder.load(args.checkpoint, args.device, weights_dtype="fp32", batch_size=16)
    base = Decoder.load(BASE_TEACHER, args.device, weights_dtype=WEIGHTS_DTYPE,
                        batch_size=DECODE_BATCH_SIZE)
    stage = args.stage_dir
    (stage / "clips").mkdir(parents=True, exist_ok=True)
    run_path = stage / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {
        "fingerprints": {"h448": h448.fingerprint(stream_protocol(DEPLOYED_BLOCK)).as_dict(),
                         "base": base.fingerprint(SPANS).as_dict()},
        "done_shards": [], "gpu_seconds": 0.0, "wall_seconds": 0.0,
    }
    wall_before = run["wall_seconds"]
    print(f"frame {len(frame)} clips; {len(run['done_shards'])} shards done", flush=True)
    by_shard: dict[int, dict[int, IndexRow]] = {}
    for row in frame:
        by_shard.setdefault(row.shard, {})[row.row_index] = row
    for shard in order:
        if shard in run["done_shards"]:
            continue
        if run["gpu_seconds"] > args.max_hours * 3600:
            break
        shard_start = time.monotonic()
        scanned, staged = [], []
        wanted = by_shard.get(shard, {})
        for row_index, row in enumerate(iter_shard_rows(
                [shard], cache_dir=args.shard_cache, delete_after=True,
                columns=["audio", "reciter_id"])):
            entry = wanted.get(row_index)
            if entry is None:
                continue
            if resolve_audio_filename(row) != entry.audio_filename:
                raise ValueError(f"shard {shard} row {row_index} is not {entry.audio_filename}")
            reference = references[entry.surah_ayah]
            if reference is None:
                continue
            samples = as_staged(decode_to_mono_16k(row["audio"]["bytes"]))
            found = {"h448": partner_commits(h448.decode_stream(samples, DEPLOYED_BLOCK),
                                             reference),
                     "base": partner_commits(base.decode_spans([samples])[0], reference)}
            scanned.append({"audio_filename": entry.audio_filename, "shard": shard,
                            "row_index": row_index, "reciter_id": entry.reciter_id,
                            "surah_ayah": entry.surah_ayah, **found})
            if found["h448"]:
                staged.append(stage_row(row, entry, stage / "clips", frozenset({SHIP_A_MINING})))
        with open(stage / "scan.jsonl", "a", encoding="utf-8") as out:
            out.writelines(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                           for r in scanned)
        previous = read_staged_clips(stage / "staged.jsonl") if (stage / "staged.jsonl").exists() \
            else {}
        write_staged_clips([*previous.values(), *staged], stage / "staged.jsonl")
        run["done_shards"].append(shard)
        run["gpu_seconds"] += time.monotonic() - shard_start
        run["wall_seconds"] = wall_before + time.monotonic() - started
        write_text_atomically(run_path, json.dumps(run, indent=1, sort_keys=True) + "\n")
        print(f"  shard {shard}: {len(scanned)} clips, {len(staged)} hits, "
              f"{run['gpu_seconds'] / 3600:.2f} GPU-h", flush=True)
    record = frame_record(frame, excluded, order, run["done_shards"])
    write_text_atomically(stage / "frame.json", json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"done: {len(run['done_shards'])} of {len(order)} shards", flush=True)


# --- after the scan --------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(raw) for raw in f if raw.strip()]


def _register(args) -> None:
    """Merge the scan's staged hits into the shared registry (refusing a changed row) and
    record them as the ``ship_a.mining`` exposure use."""
    from .exposure import SHIP_A_MINING as EXPOSURE_USE
    from .exposure import whole_recording, write_use

    registry = load_staged_clips()
    hits = read_staged_clips(args.stage_dir / "staged.jsonl")
    clashes = sorted(n for n, c in hits.items() if n in registry and registry[n] != c)
    if clashes:
        raise SystemExit(f"{len(clashes)} hits disagree with the registry, e.g. {clashes[0]}")
    write_staged_clips([*registry.values(), *hits.values()], REGISTRY_PATH)
    write_use(EXPOSURE_USE, [whole_recording(c, c.audio_sha256) for c in hits.values()])
    print(f"{len(hits)} hits in {REGISTRY_PATH} and the exposure registry")


def duplicate_recordings(registry: Mapping[str, StagedClip]) -> set[str]:
    """Hits that are byte for byte a recording staged before them: one held by a clip of
    another use, or by a hit earlier in file-name order. Each recording is mined once."""
    seen = {c.audio_sha256 for c in registry.values() if SHIP_A_MINING not in c.uses}
    duplicates = set()
    for name in sorted(n for n, c in registry.items() if SHIP_A_MINING in c.uses):
        if registry[name].audio_sha256 in seen:
            duplicates.add(name)
        seen.add(registry[name].audio_sha256)
    return duplicates


def summarize(scan: list[dict], frame: Mapping, clips: list[PoolClip], run: Mapping,
              duplicates: set[str]) -> dict:
    """Per target pair: the partner commits each decode made in the scanned clips and the
    clips holding one; the scanned and hit clips; the GPU time."""
    pairs = {}
    for pair in sorted(ALL_TARGET_PAIRS):
        pairs[pair] = {
            model: {"partner_commits": sum(row[model].get(pair, 0) for row in scan),
                    "clips": sum(pair in row[model] for row in scan)}
            for model in ("h448", "base")
        }
    gpu_hours = run["gpu_seconds"] / 3600
    hits = sum(bool(row["h448"]) for row in scan)
    return {
        "scanned_shards": len(frame["scanned_shards"]),
        "scanned_clips": len(scan),
        "scanned_reciters": len({row["reciter_id"] for row in scan}),
        "hit_clips": hits,
        "duplicate_recordings_dropped": sorted(duplicates),
        "kept_hit_clips": sum(any(s.kept for s in clip.segments) for clip in clips),
        "scan_gpu_hours": round(gpu_hours, 2),
        "scan_wall_hours": round(run["wall_seconds"] / 3600, 2),
        "hit_clips_per_gpu_hour": round(hits / gpu_hours, 1) if gpu_hours else None,
        "pairs": pairs,
        "fingerprints": run["fingerprints"],
    }


def _build(args) -> None:
    """The committed manifest, base decodes, compact scan, frame and summary."""
    stage, seg = args.stage_dir, args.stage_dir / "seg"
    frame = json.loads((stage / "frame.json").read_text(encoding="utf-8"))
    run = json.loads((stage / "run.json").read_text(encoding="utf-8"))
    scan = sorted(_jsonl(stage / "scan.jsonl"), key=lambda row: row["audio_filename"])
    registry = load_staged_clips()
    duplicates = duplicate_recordings(registry)
    selected = {row["audio_filename"]: Selected((HIT,), frame["inclusion_probability"])
                for row in scan if row["h448"] and row["audio_filename"] not in duplicates}
    statuses = [r for r in _jsonl(seg / "clip_status.jsonl") if r["audio_filename"] in selected]
    clips = build_manifest(statuses, _jsonl(seg / "segmentation.jsonl"), selected, registry,
                           SHIP_A_MINING)
    seg_run = json.loads((seg / "run.json").read_text(encoding="utf-8"))
    decodes = {
        segment_key(row["clip_audio_filename"], row["segment_index"]): row["predicted_phonemes"]
        for row in _jsonl(seg / "segment_manifest.jsonl")
    }
    MINING_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(clips, CLIPS_PATH)
    load_manifest(CLIPS_PATH, registry, SHIP_A_MINING, (HIT,))
    write_text_atomically(BASE_DECODES_PATH, json.dumps(
        {"decode_fingerprint": seg_run["decode_fingerprint"], "decodes": decodes},
        ensure_ascii=False, indent=0, sort_keys=True) + "\n")
    write_text_atomically(SCAN_PATH, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scan))
    write_text_atomically(FRAME_PATH, json.dumps(frame, indent=1, sort_keys=True) + "\n")
    summary = summarize(scan, frame, clips, run, duplicates)
    write_text_atomically(SUMMARY_PATH, json.dumps(summary, indent=1, ensure_ascii=False,
                                                   sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "fingerprints"},
                     indent=1, ensure_ascii=False))


def _stream(args) -> None:
    from .pool_stream import stream_record, write_stream_record

    registry = load_staged_clips()
    clips = load_manifest(CLIPS_PATH, registry, SHIP_A_MINING, (HIT,))
    record = stream_record(clips, registry, args.checkpoint, args.audio_dir, args.device)
    write_stream_record(record, H448_DECODES_PATH)
    print(f"{len(record['decodes'])} segments -> {H448_DECODES_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="decode whole shards and stage the hits")
    scan.add_argument("--index", type=Path, required=True)
    scan.add_argument("--evalset-manifest", type=Path, required=True)
    scan.add_argument("--checkpoint", type=Path, required=True)
    scan.add_argument("--shard-cache", type=Path, required=True)
    scan.add_argument("--max-hours", type=float, required=True,
                      help="no new shard is started once the scan's GPU time exceeds this")
    for name, text in (("register", "merge the staged hits into the registry"),
                       ("build", "write the committed manifest, decodes, scan and summary")):
        commands.add_parser(name, help=text)
    for command in commands.choices.values():
        command.add_argument("--stage-dir", type=Path, required=True)
    stream = commands.add_parser("stream", help="h448's stream cut per kept segment")
    stream.add_argument("--checkpoint", type=Path, required=True)
    stream.add_argument("--audio-dir", type=Path, required=True)
    for command in (scan, stream):
        command.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"scan": _scan, "register": _register, "build": _build, "stream": _stream}[args.command](args)


if __name__ == "__main__":
    main()

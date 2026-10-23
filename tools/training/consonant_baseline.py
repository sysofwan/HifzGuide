"""C95's baseline decodes on #176's two never-decoded sets, and the numbers the round reports.

The consonant round compares a future candidate with **C95** on two sets C95 has never been
decoded on (#174): the QUD half #95 did not read (G1) and Ship A's held-out verified sites (G6).
This module produces those baseline decodes, before any candidate exists:

* **The QUD other half** (:func:`decode_qother_shard`): every segment of #175's complement
  selection, decoded through :mod:`training.aug_readout_decode`'s path into its ``checks``
  protocol only, one line per item (the same fingerprint as #95: bf16, batch 1, CUDA). One shard
  is one mushaf parquet, read from the Hub and deleted, so the ten runs share the box.
  :func:`merge_qother` validates the run's identity with
  :func:`training.aug_readout_gate.validate_identity` (C95's pinned EMA weights), checks that
  every key is the complement's, and writes the round's one gzipped file. It also appends
  ``qud_other/c95`` to the round's ``models.json`` — append-only, never rewriting an entry.
* **Ship A's held-out verified sites** (:func:`update_held_out`): the ship_a frame's held_out
  half (G6's eight real mistakes and the marks reported only), decoded into the truth baseline's
  stores under the truth baseline's own protocol, so the harness replays each item's **check
  sequence** (:func:`assert_held_out_caches` proves the caches are the pin and carry them). The
  baselines' caches already hold every item.
* **The read** (:func:`read`, on the Mac with the harness): the QUD other half's coverage and
  C95's false flags per 1,000, and G6's frozen site count and C95's flagged count — through
  #175's readout (:func:`training.consonant_readout.g6_scope` pins G6 to its frozen ids).

Usage (on the GPU box from a copy of ``tools/``, under ``flock /root/scratch/gpu.lock``; the
launcher and its done-file check are ``scripts/issue-176-decode.sh``)::

  python -m training.consonant_baseline qother --shard 3 --out-dir <decodes> --cache-dir <hf>
  python -m training.consonant_baseline held-out --audio-dir <clips>

Then on the Mac, from ``tools/``::

  python -m training.consonant_baseline merge --decodes-dir <fetched decodes>
  python -m training.consonant_baseline read
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.ship_a import HELD_OUT
from tadabur.ship_a_verified import SHIP_A, load_verified
from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips
from training import aug_readout_decode as ad
from training.aug_readout_gate import (
    QUD_OTHER_SET,
    identity_record,
    model_spec,
    round_decodes_file,
    validate_identity,
)
from training.aug_readout_metrics import C95
from training.aug_readout_spec import CONSONANT1, C95_REF
from training import consonant_readout as cr
from training.external_corpus import merge_decode_files
from training.external_decode import append_timing, decode_recordings, load_decoders
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import (
    CHECK_PROTOCOLS,
    CHECK_STORE,
    CHECKS,
    DECODE_STORE,
    PROTOCOLS,
    check_cache,
    file_sha256,
    items_of,
    model_caches,
    read_cache,
)

#: The round's paired baseline: the one model #176 decodes.
BASELINE = C95
QOTHER = QUD_OTHER_SET
#: Where the merged QUD decode is committed (the round's own root, as the readout resolves it).
QOTHER_PATH = round_decodes_file(CONSONANT1, QOTHER, BASELINE)


def pin() -> dict:
    """C95's identity as the round fixed it (#174): the checkpoint, its SHA-256, EMA."""
    return identity_record(CONSONANT1.models[BASELINE])


# --- the QUD other half -------------------------------------------------------------------------


def qother_identity(record: Mapping) -> dict:
    """One QUD decode run's identity record, complete and the round's pin (``validate_identity``
    checks the frozen fingerprints and the pinned EMA checkpoint)."""
    return validate_identity(QOTHER, BASELINE, record, spec=CONSONANT1)


def qother_recordings(shard_index: int, cache_dir: Path):
    """One shard's recordings of the complement half, each holding only that half's items."""
    return ad.recordings(QOTHER, ad.shards(QOTHER)[shard_index], cache_dir, None)


def decode_qother_shard(shard_index: int, out_dir: Path, cache_dir: Path) -> dict:
    """Decode one shard of the complement half with C95's EMA weights, at the round's
    fingerprint, appending each item's check sequence as one line; resumable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = ad.shards(QOTHER)[shard_index]
    identity_path = out_dir / f"{QOTHER}.{BASELINE}.identity.json"
    decoders = load_decoders(lambda model: identity_path,
                             {BASELINE: model_spec(BASELINE, spec=CONSONANT1)})
    identity = qother_identity(json.loads(identity_path.read_text(encoding="utf-8")))
    print(f"{QOTHER}: {shard} decodes {BASELINE} as {identity}", flush=True)
    paths = {BASELINE: out_dir / f"{QOTHER}.{shard_index}.{BASELINE}.jsonl"}
    timing = decode_recordings(qother_recordings(shard_index, cache_dir), decoders, paths,
                               f"{QOTHER} {shard}", {CHECKS: CHECK_STORE})
    append_timing(out_dir / f"{QOTHER}.{shard_index}.timing.jsonl",
                  {"set": QOTHER, "shard": shard, "identity": identity, **timing})
    return timing


def merge_qother(decodes_dir: Path, destination: Path, models_path: Path = cr.MODELS_PATH,
                 expected: int | None = None) -> dict:
    """The shard decodes as the round's one committed file, under C95's pinned identity.

    Refuses a key outside the complement selection, a record count other than the selection's,
    and any run whose identity is not the pin. The round's ``models.json`` gains this set's
    entry and is never otherwise rewritten."""
    decodes_dir, destination = Path(decodes_dir), Path(destination)
    identity_path = decodes_dir / f"{QOTHER}.{BASELINE}.identity.json"
    if not identity_path.exists():
        raise ValueError(f"{identity_path}: no decode identity to validate")
    record = json.loads(identity_path.read_text(encoding="utf-8"))
    identity = qother_identity(record)
    complement = cr.load_qud_complement()
    paths = sorted(decodes_dir.glob(f"{QOTHER}.*.{BASELINE}.jsonl"))
    if not paths:
        raise ValueError(f"{decodes_dir}: no {QOTHER} shard decoded")
    for path in paths:
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                key = json.loads(raw)["key"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"{path}:{number} is not a decode record") from error
            if key not in complement:
                raise ValueError(f"{path}:{number} {key} is not in the complement selection")
    merge_decode_files(paths, identity_path, destination,
                       len(complement) if expected is None else expected)
    _record_models_entry(models_path, record)
    return identity


def _record_models_entry(path: Path, record: Mapping) -> None:
    """Append this decode's identity to the round's ``models.json``; never change an entry."""
    models = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    name = f"{QOTHER}/{BASELINE}"
    if name in models:
        if models[name] != record:
            raise ValueError(f"{path}: {name} is {models[name]}, not this decode's {record}")
        return
    models[name] = dict(record)
    write_text_atomically(path, json.dumps(models, ensure_ascii=False, indent=1) + "\n")


# --- Ship A's held-out verified sites -----------------------------------------------------------


def held_out_sites() -> list:
    """Ship A's verified sites the ship_a frame holds out: G6's and the report-only marks."""
    sites, designs = load_verified()
    return [site for site in sites
            if site.site_id in designs
            and (designs[site.site_id].frame, designs[site.site_id].half) == (SHIP_A, HELD_OUT)]


def held_out_items(registry: Mapping | None = None) -> dict:
    """One item per held-out clip and span, every one staged with its recorded checksum."""
    registry = registry if registry is not None else load_staged_clips(REGISTRY_PATH)
    return items_of(held_out_sites(), registry)


def update_held_out(audio_dir: Path) -> dict:
    """Decode C95's decodes and check sequences of every held-out item, then prove the caches
    are the round's pin; the baselines' caches already hold the same items."""
    if not Path(C95_REF).is_file():
        raise SystemExit(f"{C95_REF}: the round's checkpoint is not on this machine")
    found = file_sha256(C95_REF)
    if found != pin()["checkpoint_sha256"]:
        raise SystemExit(f"{C95_REF} is {found}, not the round's pin")
    registry = load_staged_clips(REGISTRY_PATH)
    items = held_out_items(registry)
    model_caches(BASELINE, C95_REF, items, registry, Path(audio_dir), use_ema=True)
    return assert_held_out_caches(items)


def assert_held_out_caches(items: Sequence, check_dir: Path | None = None,
                           decode_dir: Path | None = None) -> dict:
    """Refuse anything but the round's pin, holding every item's decode and check sequence."""
    check_dir = Path(check_dir or CHECK_STORE.directory)
    decode_dir = Path(decode_dir or DECODE_STORE.directory)
    expected = pin()
    for store, directory, protocols in ((CHECK_STORE, check_dir, CHECK_PROTOCOLS),
                                        (DECODE_STORE, decode_dir, PROTOCOLS)):
        path = directory / f"{BASELINE}.json"
        if not path.exists():
            raise SystemExit(f"{path}: no {BASELINE} cache; decode the held-out clips first")
        cache = read_cache(path)
        check_cache(path, cache, C95_REF, protocols)
        if not cr.binds_cache(cache["identity"] or {}, expected):
            raise ValueError(f"{path}: {cache['identity']} is not the round's pin {expected}")
        missing = set(items) - set(cache["items"])
        if missing:
            raise ValueError(f"{path}: {len(missing)} held-out item(s) not decoded")
        for key in items:
            absent = [protocol for protocol in protocols if protocol not in cache["items"][key]]
            if absent:
                raise ValueError(f"{path}: {key} carries no {', '.join(absent)}")
    return expected


# --- the read (baseline numbers) ----------------------------------------------------------------


def g6_baseline(cache_dir: Path | None = None) -> dict | None:
    """G6's table for C95 alone, its sites from #175's frozen list; ``None`` without caches."""
    scope = cr.g6_scope()
    return cr.catch_arm(scope["sites"], (BASELINE,), spec=CONSONANT1, cache_dir=cache_dir,
                     scope=scope)


def read() -> dict:
    """The PR's numbers: the QUD other half's coverage and C95's false flags per 1,000, G6's
    site count and C95's flagged count, and the report-only marks. The round's pins are
    enforced before any input is read: a same-commit harness with another binary, or a moved
    pre-registration, refuses here as #175's confirm path refuses it."""
    from training import aug_readout as ar
    from training.aug_readout_gate import sha256_file
    from training.aug_readout_metrics import PER_1000, compare, false_flag_terms
    from training.aug_readout_spec import PREREG_PATH, check_pins
    from training.muraja_policy import SHIPPED_DEFAULT

    sources = ar.Sources(CONSONANT1)
    check_pins(CONSONANT1, sources.harness_identity(),
               sha256_file(PREREG_PATH) if PREREG_PATH.exists() else "")
    data = sources.qud_other((BASELINE,))
    grades = sources.grade(data.items, data.decodes, (BASELINE,))
    flags = compare(false_flag_terms(grades, (BASELINE,), SHIPPED_DEFAULT.name), PER_1000,
                    comparisons=())
    row = flags["models"][BASELINE]
    population = cr.coverage()["populations"][QOTHER]
    table = g6_baseline()
    return {
        "qud_other": {"decoded": population["decoded"], "evaluated": population["evaluated"],
                      "excluded": population["excluded"],
                      "graded_words": int(row["den"]), "flagged_words": int(row["num"]),
                      "false_flags_per_1000": row["point"]},
        "g6": {"sites": len(cr.load_g6_sites()),
               "flagged": None if table is None else table["models"][BASELINE]["flagged"]},
        "descriptive": cr.descriptive_catch((BASELINE,)),
    }


# --- running it ---------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    qother = commands.add_parser("qother", help="decode one shard of the QUD other half")
    qother.add_argument("--shard", type=int)
    qother.add_argument("--out-dir", type=Path)
    qother.add_argument("--cache-dir", type=Path)
    qother.add_argument("--list-shards", action="store_true")

    held = commands.add_parser("held-out", help="decode Ship A's held-out verified sites")
    held.add_argument("--audio-dir", type=Path)
    held.add_argument("--list-clips", action="store_true")

    merged = commands.add_parser("merge", help="the shard decodes as the round's one file")
    merged.add_argument("--decodes-dir", type=Path, required=True)
    merged.add_argument("--destination", type=Path, default=QOTHER_PATH)
    merged.add_argument("--models", type=Path, default=cr.MODELS_PATH)
    merged.add_argument("--expected", type=int)

    commands.add_parser("read", help="the baseline numbers, through the harness")

    args = parser.parse_args()
    if args.command == "qother":
        if args.list_shards:
            print("\n".join(ad.shards(QOTHER)))
            return
        if None in (args.shard, args.out_dir, args.cache_dir):
            parser.error("--shard, --out-dir and --cache-dir are required")
        decode_qother_shard(args.shard, args.out_dir, args.cache_dir)
    elif args.command == "held-out":
        if args.list_clips:
            print("\n".join(sorted({site.audio_filename for site in held_out_sites()})))
            return
        if args.audio_dir is None:
            parser.error("--audio-dir is required")
        print(json.dumps(update_held_out(args.audio_dir), ensure_ascii=False, indent=1))
    elif args.command == "merge":
        print(json.dumps(merge_qother(args.decodes_dir, args.destination,
                                      models_path=args.models, expected=args.expected),
                         ensure_ascii=False, indent=1))
    else:
        print(json.dumps(read(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Diagnose ``h448``'s non-professional gap (#159): D1 clone agreement, D2 acoustic drift.

#156 found ``h448`` 2-3x worse than the base teacher on app users' and amateurs' audio, and
teacher-on-window targets explain only about a fifth of it. Two diagnostics separate what is
left before any training is spent:

* **D1, domain shift.** ``h448``'s decode against the teacher's, under identical windows, on
  the audio ``h448`` was distilled on (Tadabur's ``decode_evalset`` dev half, the population of
  its 94.87% headline) and on audio neither model heard: QUD's professional mushafs (#144),
  Tarteel v1's app users and qdat's amateurs (#156). Their committed decodes are reused; only
  the dev half is decoded here.
* **D2, acoustic fragility.** A fixed sample of dev clips ``h448`` never trained on, under the
  grid of :mod:`training.perturbation` (committed before any decode), decoded by both models;
  each perturbed decode is read against the model's own clean decode (**drift**) and against
  the teacher's clean decode.

**The metric** is #73's, the decode compared phoneme to phoneme with the reference decode, never
through a scorer: pooled ``1 - edits / reference phonemes`` (``phoneme``), and, through the same
alignment, the share of the reference decode's slots the other decode matches: every consonant,
the consonants #159 names (:data:`FOCUS`), every haraka and every geminate's shaddah
(:func:`tally`; reflexive, so a decode agrees with itself in every class). Every interval is a 95% percentile bootstrap over clusters
(:func:`training.decode_evalset.cluster_bootstrap`): Tadabur's and qdat's reciters, QUD's mushaf
and surah, Tarteel v1's surah (its users are unnamed, as in #156).

Every decode here is the truth baseline's protocol (bf16 weights, batch 1, CUDA; ``stream_b0``
the deployed b = 0 stream, tail flushed; ``checks`` the transcript Muraja commits from its
recorded checks), so a difference between corpora is the audio, not the numerics. ``h448``'s
94.87% was fp32 weights at batch 32 against a teacher decoded at batch 32
(:func:`numerics`).

**Never used:** the ``decode_evalset`` test half, the sealed panel and Ship A's held-out half
(:func:`sample_record` refuses a dev clip that is, or is a copy of, one of their recordings).
Nothing here trains, sets a threshold or recommends one.

Files (``tadabur/nonpro_gap/``): ``grid.json`` (the grid and the D2 sample, committed first),
``noise_pool.json`` (the noise recordings' provenance; their audio stays on the GPU box),
``decodes/<condition>.<model>.jsonl.gz`` and ``report.json``.

Usage (from ``tools/``; torch-free but the decode)::

  python -m training.nonpro_gap grid
  python -m training.nonpro_gap noise-pool --cache-dir <scratch> --out <scratch>/noise_pool.npz
  python -m training.nonpro_gap_decode --condition <name> ...        # the GPU box
  python -m training.nonpro_gap merge --decodes-dir <its decodes> --logs-dir <its logs>
  python -m training.nonpro_gap report
  python -m training.nonpro_gap doc
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tadabur.exposure import (
    EVALSET_DEV,
    EVALSET_MANIFEST_PATH,
    EVALSET_TEST,
    H448_INIT_VALIDATION,
    H448_TRAINING,
    SEALED_PANEL,
    SHIP_A_HELD_OUT,
    ExposureRegistry,
    evalset_records,
    linked_sources,
    load_registry,
)
from tadabur.listening_session import held_geminates, run_start
from tadabur.truth_sites import CONSONANTS
from training.decode_evalset import align, cluster_bootstrap
from training.decoding import committed_text
from training.external_corpus import merge_decode_files, read_decode_file
from training.perturbation import GRID, SEED, Condition, grid_record
from training.tashkeel_eval import SHORT_VOWELS, write_text_atomically
from training.truth_baseline import CHECKS

GAP_DIR = Path(__file__).parent.parent / "tadabur" / "nonpro_gap"
GRID_PATH = GAP_DIR / "grid.json"
POOL_PATH = GAP_DIR / "noise_pool.json"
DECODES_DIR = GAP_DIR / "decodes"
#: Decodes a grid amendment replaced, kept to show what moved: the codec condition under grid
#: v1, whose output lagged its input by 1-2 samples.
SUPERSEDED_DIR = DECODES_DIR / "superseded"
SUPERSEDED = ("opus_12k_nb_v1",)
REPORT_PATH = GAP_DIR / "report.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "nonpro-gap.md"

MODELS = ("base", "h448")
STREAM = "stream_b0"
#: Dev clips drawn for D2: ~1.4 h of audio per condition, about one GPU-hour for the grid.
D2_SAMPLE_SIZE = 400
#: The consonants #159 asks for as a group: ``h448``'s commonest substitutions on Tarteel v1
#: and qdat (#156: ``ن→ل``, ``ل→ن``, ``ب→ف``, ``د→ت``, ``ع→ء``).
FOCUS = frozenset("لنءبد")
#: What a tally counts, in its order: (agreeing, all) for each.
CLASSES = ("phoneme", "consonant", "focus", "haraka", "shaddah")
#: The uses a D2 clip must hold no recording (or copy) of: ``h448``'s training and validation.
SEEN_BY_H448 = (H448_TRAINING, H448_INIT_VALIDATION)
#: The uses no clip here may come from at all.
NEVER_USED = (EVALSET_TEST, SEALED_PANEL, SHIP_A_HELD_OUT)
BOOTSTRAP_SEED = 7


# --- the dev half and the D2 sample -------------------------------------------------------------


@dataclass(frozen=True)
class DevClip:
    """One clip of the ``decode_evalset`` dev half, with the teacher decode its manifest caches
    (bf16 weights, batch 32)."""

    filename: str
    reciter_id: int
    source: tuple[int, int]
    duration_s: float
    cached_teacher: str


def dev_clips(manifest_path: Path = EVALSET_MANIFEST_PATH) -> list[DevClip]:
    """The manifest's dev half (the committed copy of the GPU box's ``gate_eval`` manifest)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [DevClip(c["filename"], c["reciter_id"], source, c["duration_s"], c["teacher_text"])
            for c, (source, use) in zip(manifest["clips"], evalset_records(manifest_path))
            if use == EVALSET_DEV]


def linked(clips: Iterable[DevClip], uses: Sequence[str],
           registry: ExposureRegistry) -> set[str]:
    """The clips whose recording is, sits in a shard of, or is tied to a recording of ``uses``."""
    sources = linked_sources(uses, registry)
    shards = {s for u in uses if u in registry.shard_uses for s in registry.shard_uses[u].shards}
    return {c.filename for c in clips if c.source in sources or c.source[0] in shards}


def _rank(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, (SEED, *parts))).encode("utf-8")).hexdigest()


def sample_record(clips: Sequence[DevClip], registry: ExposureRegistry) -> dict:
    """D2's sample: :data:`D2_SAMPLE_SIZE` dev clips ``h448`` holds no copy of, by salted rank."""
    if barred := linked(clips, NEVER_USED, registry):
        raise ValueError(f"{len(barred)} dev clips are recordings of {NEVER_USED}")
    seen = linked(clips, SEEN_BY_H448, registry)
    eligible = [c for c in clips if c.filename not in seen]
    chosen = sorted(sorted(eligible, key=lambda c: _rank("d2-sample", c.filename))
                    [:D2_SAMPLE_SIZE], key=lambda c: c.filename)
    return {
        "size": D2_SAMPLE_SIZE, "dev_clips": len(clips), "h448_seen": sorted(seen),
        "eligible": len(eligible), "reciters": len({c.reciter_id for c in chosen}),
        "hours": round(sum(c.duration_s for c in chosen) / 3600, 3),
        "rule": "eligible dev clips ranked by sha256(seed:d2-sample:filename), first size",
        "clips": [c.filename for c in chosen],
    }


def grid_file(registry: ExposureRegistry | None = None) -> dict:
    return {"grid": grid_record(),
            "sample": sample_record(dev_clips(), registry or load_registry())}


def committed_grid() -> dict:
    """The committed grid, refused when the code's grid has moved since it was committed."""
    committed = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    if committed["grid"] != json.loads(json.dumps(grid_record())):
        raise ValueError(f"{GRID_PATH} is not the grid training.perturbation now applies")
    return committed


def decode_plan(condition: Condition) -> list[DevClip]:
    """What a condition decodes: every dev clip clean (D1), the committed sample otherwise."""
    clips = dev_clips()
    if condition.name == "clean":
        return clips
    chosen = set(committed_grid()["sample"]["clips"])
    return [c for c in clips if c.filename in chosen]


# --- the metric -------------------------------------------------------------------------------


def tally(decode: str, reference: str) -> tuple[int, ...]:
    """``decode`` against ``reference``, another decode: (agreeing, all) for each of
    :data:`CLASSES`, flat. Both are read through one alignment, the Levenshtein backtrace
    (:func:`training.decode_evalset.align`, diagonal first); a reference slot agrees when its
    characters are matches:

    * ``phoneme``: ``len(reference) - edits``, so pooled it is #73's agreement;
    * ``consonant``: every run of a consonant, agreeing when any letter of it is matched (the
      letter was heard, held or not); ``focus`` those of :data:`FOCUS`;
    * ``haraka``: every fatha, damma and kasra;
    * ``shaddah``: every geminate (a doubled consonant), agreeing when its whole run is.

    Reflexive by construction: a decode read against itself agrees in every slot. (The
    transcript reading, :func:`training.nonpro_eval.letter_outcomes`, is not: it assumes a
    reference's shape, one haraka per consonant, that a decode need not have.)"""
    ops = align(reference, decode)
    matched = {i for op, i, _ in ops if op == "match"}

    def run(i: int) -> set[int]:
        end = i
        while end + 1 < len(reference) and reference[end + 1] == reference[i]:
            end += 1
        return set(range(i, end + 1))

    def count(slots: Iterable[set[int]], agrees: Callable[[set[int]], bool]) -> tuple[int, int]:
        slots = list(slots)
        return sum(map(agrees, slots)), len(slots)

    def heard(slot: set[int]) -> bool:
        return bool(slot & matched)

    def held(slot: set[int]) -> bool:
        return slot <= matched

    consonants = [i for i, c in enumerate(reference)
                  if c in CONSONANTS and run_start(reference, i) == i]
    edits = len(ops) - len(matched)
    return (len(reference) - edits, len(reference),
            *count((run(i) for i in consonants), heard),
            *count((run(i) for i in consonants if reference[i] in FOCUS), heard),
            *count(({i} for i, c in enumerate(reference) if c in SHORT_VOWELS), held),
            *count((run(i) for i in held_geminates(reference)), held))


def _tally_pair(pair: tuple[str, str]) -> tuple[int, ...]:
    return tally(*pair)


def tallies(pairs: Sequence[tuple[str, str]], workers: int) -> list[tuple[int, ...]]:
    """:func:`tally` of each ``(decode, reference)``, in order, across ``workers`` processes."""
    if workers <= 1:
        return [tally(*p) for p in pairs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_tally_pair, pairs, chunksize=64))


def _add(vectors: Iterable[Sequence[int]]) -> tuple[int, ...]:
    return tuple(map(sum, zip(*vectors)))


def interval(rows: Sequence[tuple[object, Sequence[int]]],
             statistic: Callable[[tuple[int, ...]], float]) -> dict:
    """``statistic`` of the summed vectors, and its cluster-bootstrap 95% interval; ``rows`` are
    ``(cluster, vector)``, summed per cluster first (the statistic reads only sums)."""
    per_cluster: dict = defaultdict(list)
    for cluster, vector in rows:
        per_cluster[cluster].append(vector)
    summed = [(k, _add(v)) for k, v in sorted(per_cluster.items(), key=lambda kv: str(kv[0]))]
    low, high = cluster_bootstrap(summed, cluster_of=lambda row: row[0],
                                  statistic=lambda s: statistic(_add(v for _, v in s)),
                                  seed=BOOTSTRAP_SEED)
    return {"point": round(statistic(_add(v for _, v in summed)), 5),
            "low": round(low, 5), "high": round(high, 5), "clusters": len(summed)}


def _ratio(k: int, offset: int = 0) -> Callable[[tuple[int, ...]], float]:
    """Class ``k``'s agreement, of the tally starting at ``offset`` in a joined vector."""
    return lambda s: s[offset + 2 * k] / max(1, s[offset + 2 * k + 1])


def rates(rows: Sequence[tuple[object, tuple[int, ...]]]) -> dict:
    """Each class's pooled agreement, interval and slot count (none where there is no slot)."""
    total = _add(v for _, v in rows)
    return {name: {**interval(rows, _ratio(k)), "slots": total[2 * k + 1]}
            if total[2 * k + 1] else None for k, name in enumerate(CLASSES)}


def paired(rows: Sequence[tuple[object, tuple[int, ...]]],
           difference: Callable[[Callable[[int], float]], float]) -> dict:
    """Per class, ``difference(ratio_at)``, ``ratio_at(offset)`` being that class's pooled
    agreement in the tally at ``offset`` of the rows' joined vectors; a paired interval, as both
    tallies of a row resample together."""
    return {name: interval(rows, lambda s, k=k: difference(lambda o: _ratio(k, o)(s)))
            for k, name in enumerate(CLASSES)}


# --- D1 -----------------------------------------------------------------------------------------


def _committed(line: dict, protocol: str) -> str:
    return committed_text(line[CHECKS]) if protocol == CHECKS else line[protocol]


def external_pairs() -> dict[str, dict[str, list[tuple[object, str, str]]]]:
    """``(cluster, h448 decode, base decode)`` per item, per committed corpus and protocol."""
    from tadabur.nonpro_sources import QDAT, TARTEEL_V1
    from tadabur.qud_overlap import PILOT
    from training.nonpro_eval import clusters, decodes_file
    from training.qud_pilot import read_decodes

    out = {}
    corpora = {"qud": [({k: (PILOT.index(slug), int(l["surah_ayah"].split(":")[0]))
                         for k, l in read_decodes(slug, "base").items()},
                        {m: read_decodes(slug, m) for m in MODELS}) for slug in PILOT]}
    for dataset in (TARTEEL_V1, QDAT):
        by_model = {m: read_decode_file(decodes_file(dataset, m)) for m in MODELS}
        corpora[dataset.name] = [(clusters(dataset, list(by_model["base"].values())), by_model)]
    for name, parts in corpora.items():
        out[name] = {p: [(cluster[k], _committed(d["h448"][k], p), _committed(d["base"][k], p))
                         for cluster, d in parts for k in sorted(cluster)]
                     for p in (STREAM, CHECKS)}
    return out


def gap_decodes(condition: str, directory: Path = DECODES_DIR) -> dict[str, dict[str, dict]]:
    """A condition's committed decodes, per model, by clip."""
    return {m: read_decode_file(directory / f"{condition}.{m}.jsonl.gz") for m in MODELS}


def d1(workers: int) -> dict:
    """Clone agreement (``h448`` against the base teacher) per corpus and protocol; the dev
    half whole (the headline's population) and without the clips ``h448`` holds a copy of."""
    clips = dev_clips()
    clean = gap_decodes("clean")
    seen = set(committed_grid()["sample"]["h448_seen"])

    def dev(keep: Callable[[DevClip], bool]) -> list[tuple[object, str, str]]:
        return [(c.reciter_id, clean["h448"][c.filename][STREAM],
                 clean["base"][c.filename][STREAM]) for c in clips if keep(c)]

    corpora = {"tadabur_dev": {STREAM: dev(lambda c: True)},
               "tadabur_dev_unseen": {STREAM: dev(lambda c: c.filename not in seen)},
               **external_pairs()}
    out = {}
    for name, protocols in corpora.items():
        out[name] = {}
        for protocol, rows in protocols.items():
            counted = tallies([(h, b) for _, h, b in rows], workers)
            out[name][protocol] = {"items": len(rows), "rates": rates(
                [(cluster, t) for (cluster, _, _), t in zip(rows, counted)])}
        print(f"D1 {name} done", flush=True)
    return out


def numerics(workers: int) -> dict:
    """The dev half's teacher decode here (bf16, batch 1) and ``h448``'s, each against the
    teacher decode the manifest caches (bf16, batch 32): how far this protocol sits from the
    one ``h448``'s 94.87% headline was measured under."""
    clips = dev_clips()
    clean = gap_decodes("clean")
    out = {}
    for model in MODELS:
        counted = tallies([(clean[model][c.filename][STREAM], c.cached_teacher) for c in clips],
                          workers)
        out[f"{model}_vs_cached_teacher"] = rates([(c.reciter_id, t)
                                                   for c, t in zip(clips, counted)])
    return out


# --- D2 -----------------------------------------------------------------------------------------


def d2(workers: int) -> dict:
    """Per perturbed condition: each model's agreement with its own clean decode (drift) and
    with the teacher's; ``h448``'s agreement with the teacher on the same perturbed audio
    (**clone**, D1's metric under the condition) and its change from clean; and two paired
    differences, ``h448``'s minus the teacher's (positive: ``h448`` moves further): **excess
    drift**, away from its own clean decode, and **excess clone drop**, the fall of its
    agreement with the teacher's clean decode."""
    sample = committed_grid()["sample"]["clips"]
    reciter = {c.filename: c.reciter_id for c in dev_clips()}
    clean = gap_decodes("clean")
    width = 2 * len(CLASSES)
    clone = tallies([(clean["h448"][k][STREAM], clean["base"][k][STREAM]) for k in sample],
                    workers)
    out = {"clean": {"teacher": {"h448": rates([(reciter[k], t)
                                                for k, t in zip(sample, clone)])}}}
    #: Each perturbed decode against each reference, in a row's order (then h448 clean against
    #: the teacher's clean, the same in every condition).
    readings = (("h448", "h448"), ("h448", "base"), ("base", "base"), ("h448", "base_perturbed"))
    own_h, teacher_h, own_b, clone_c, clone_clean = range(5)

    def part(rows: list, index: int) -> list:
        return [(c, v[index * width:(index + 1) * width]) for c, v in rows]

    def block(perturbed: dict[str, dict[str, dict]]) -> dict:
        def reference(name: str, key: str) -> str:
            source = perturbed["base"] if name == "base_perturbed" else clean[name]
            return source[key][STREAM]

        counted = tallies([(perturbed[m][k][STREAM], reference(ref, k))
                           for k in sample for m, ref in readings], workers)
        n = len(readings)
        rows = [(reciter[k], sum(counted[n * i:n * (i + 1)], ()) + clone[i])
                for i, k in enumerate(sample)]
        return {
            "own": {"h448": rates(part(rows, own_h)), "base": rates(part(rows, own_b))},
            "teacher": {"h448": rates(part(rows, teacher_h)), "base": rates(part(rows, own_b))},
            "clone": rates(part(rows, clone_c)),
            "clone_change": paired(rows, lambda r: r(clone_c * width) - r(clone_clean * width)),
            "excess_drift": paired(rows, lambda r: r(own_b * width) - r(own_h * width)),
            "excess_clone_drop": paired(rows, lambda r: (r(clone_clean * width)
                                                         - r(teacher_h * width))
                                        - (1.0 - r(own_b * width))),
        }

    for condition in GRID[1:]:
        out[condition.name] = block(gap_decodes(condition.name))
        print(f"D2 {condition.name} done", flush=True)
    out["superseded"] = {name: block(gap_decodes(name, SUPERSEDED_DIR)) for name in SUPERSEDED}
    return out


def noise_summary() -> dict:
    """The pool, and the licences of the recordings the sample drew."""
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    by_id = {r["freesound_id"]: r for r in pool["recordings"]}
    drawn = Counter(line["freesound_id"]
                    for line in gap_decodes(GRID[1].name)["base"].values())
    return {"recordings": len(pool["recordings"]), "left_out": pool["left_out"],
            "hours": round(sum(r["seconds"] for r in pool["recordings"]) / 3600, 2),
            "recordings_drawn": len(drawn),
            "licences_drawn": dict(sorted(Counter(by_id[i]["licence"] for i in drawn).items()))}


# --- merge, noise pool -------------------------------------------------------------------------


def merge(decodes_dir: Path) -> dict:
    """The GPU box's per-condition decodes as one gzipped file per condition and model, each
    checked against its plan's count; the models' identity records."""
    out = {}
    for condition in GRID:
        for model in MODELS:
            out[f"{condition.name}/{model}"] = merge_decode_files(
                [decodes_dir / f"{condition.name}.{model}.jsonl"],
                decodes_dir / f"{model}.identity.json",
                DECODES_DIR / f"{condition.name}.{model}.jsonl.gz", len(decode_plan(condition)))
    return out


def noise_pool(cache_dir: Path, out: Path) -> dict:
    """Build the noise pool on scratch and its committed provenance record."""
    from training.perturbation import build_pool, save_pool

    pool, left_out = build_pool(cache_dir)
    save_pool(pool, out)
    return {"left_out": left_out, "recordings": [r.as_dict() for r in pool]}


def pass_hours(logs_dir: Path) -> float:
    """The GPU lock's wall time over every decode pass (``decode[N]_start`` to its ``_end``):
    passes held the lock one at a time, with gaps between them that held nothing."""
    from datetime import datetime

    def stamp(path: Path) -> float:
        return datetime.fromisoformat(path.read_text(encoding="utf-8").strip()
                                      .replace("Z", "+00:00")).timestamp()

    return round(sum(stamp(start.with_name(start.name.replace("_start", "_end"))) - stamp(start)
                     for start in sorted(logs_dir.glob("decode*_start"))) / 3600, 2)


def _write_json(path: Path, payload: dict) -> None:
    write_text_atomically(path, json.dumps(payload, ensure_ascii=False, indent=1,
                                           sort_keys=True) + "\n")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("grid", help="write grid.json: the grid and the D2 sample")
    pooled = commands.add_parser("noise-pool", help="download, decode and filter the noise")
    pooled.add_argument("--cache-dir", type=Path, required=True)
    pooled.add_argument("--out", type=Path, required=True, help="the pool's .npz (scratch)")
    merged = commands.add_parser("merge", help="gather the GPU box's decodes")
    merged.add_argument("--decodes-dir", type=Path, required=True)
    merged.add_argument("--logs-dir", type=Path, required=True)
    merged.add_argument("--gpu", default="NVIDIA GeForce RTX 5060 Ti")
    reported = commands.add_parser("report", help="D1 and D2 from the committed decodes")
    reported.add_argument("--workers", type=int, default=8)
    commands.add_parser("doc", help="render docs/nonpro-gap.md from the report")
    args = parser.parse_args()

    if args.command == "grid":
        _write_json(GRID_PATH, grid_file())
        return
    if args.command == "noise-pool":
        _write_json(POOL_PATH, noise_pool(args.cache_dir, args.out))
        return
    if args.command == "doc":
        from training.nonpro_gap_doc import render

        write_text_atomically(DOC_PATH, render(json.loads(REPORT_PATH.read_text(
            encoding="utf-8"))))
        print(f"wrote {DOC_PATH}")
        return
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    if args.command == "merge":
        from training.nonpro_eval import compute_record

        report["models"] = merge(args.decodes_dir)
        report["compute"] = compute_record(args.decodes_dir, args.logs_dir, args.gpu,
                                           parallel_runs=3)
        # It counts timing lines, one per finished condition run; the first pass wrote none.
        del report["compute"]["shard_runs_completed"]
        # It reads one start to the last decode; the passes left the lock between them.
        report["compute"]["gpu_hours"] = pass_hours(args.logs_dir)
        notes = args.logs_dir / "run_notes"
        report["compute"]["run_notes"] = (notes.read_text(encoding="utf-8").strip()
                                          if notes.exists() else None)
    else:
        grid = committed_grid()
        report.update({"grid": grid["grid"],
                       "sample": {k: v for k, v in grid["sample"].items() if k != "clips"},
                       "noise": noise_summary(), "numerics": numerics(args.workers),
                       "d1": d1(args.workers), "d2": d2(args.workers)})
    _write_json(REPORT_PATH, report)


if __name__ == "__main__":
    main()

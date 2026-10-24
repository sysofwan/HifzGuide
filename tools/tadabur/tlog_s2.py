"""Tadabur-only validation of the pre-registered speaker clustering (#186, tlog S2).

`docs/tlog-holdout-preregistration.md` ("Groups and roles", items 2-3) fixes the procedure
(:mod:`tadabur.speaker_cluster`) and the validation it must pass. This module is the run:

``sample``
    the fixed-seed draw from the training-complement shards, written to
    ``tools/tadabur/tlog/s2/sample.jsonl`` with its census in ``selection.json``;
``embed``
    the GPU pass: the two :data:`tadabur.speaker_embed.MODELS` are pinned, then every sample
    clip's first 20 s is embedded. The encoder files are scratch, never committed;
``report``
    the τ grid over the fetched embeddings, the pre-registered choice, and the generated
    ``tools/tadabur/tlog/s2/report.json`` and ``docs/tlog-speaker-clustering.md``.

**Tadabur only.** The sample is drawn from the 20 strided training-complement shards below and
:func:`assert_tadabur_sample` refuses anything else, so no tlog audio is ever read here.

The sample (approved 2026-10-24):
- the 20 shards are every 17th shard of the complement of
  :func:`training.distill_stream.held_out_shards`, so the set is derived, never hand-typed;
- the frame is every row of theirs whose ``ayah_duration_s`` falls inside the Tarteel v1 look
  range; the ten deciles are that pool's equal-count cut points, and the draw fills exactly
  :data:`PER_BIN` clips per decile;
- :data:`COVERAGE_AYAT` ayat with the most distinct reciters in the frame get
  :data:`COVERAGE_RECITERS` clips each, one per reciter, so the text-independence check has
  same-ayah pairs to measure;
- the rest is filled in clip-id rank order (``sha256(salt:...)``), at most
  :data:`RECITER_CAP` clips per reciter and one clip per ``(reciter, ayah)`` pair.

Usage (from ``tools/``)::

  python -m tadabur.tlog_s2 sample                  # draw and write the committed sample
  python -m tadabur.tlog_s2 embed --out-dir /root/scratch/issue-186/embeddings
  python -m tadabur.tlog_s2 report --embeddings-dir /root/scratch/issue-186/embeddings
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import speaker_embed
from .shard_reader import NUM_SHARDS
from .speaker_cluster import GRID, RECALL_FLOOR, cluster, decide, metrics, model_outcome
from .tlog_index import head_sha
from training.tashkeel_eval import write_text_atomically

#: The draw's fixed seed: every rank below is ``sha256("<salt>:<key>")``, so a rerun ranks
#: the same clips the same way on any interpreter (no RNG state to drift).
SALT = "issue-186-tlog-s2-v1"
#: Clips per duration decile (ten bins, so the sample is :data:`PER_BIN` x 10).
PER_BIN = 200
#: Ayat given a guaranteed spread of reciters, and how many reciters each.
COVERAGE_AYAT = 60
COVERAGE_RECITERS = 5
#: The most clips one reciter may contribute.
RECITER_CAP = 20
#: The embedding excerpt: the first 20 s at 16 kHz, as the pre-registration states.
SPAN_SAMPLES = speaker_embed.SPAN_SAMPLES
#: Every 17th shard of the complement: 345 shards / 17 -> 20 shards, spanning the corpus.
SAMPLE_STRIDE = 17
SAMPLE_SHARD_COUNT = 20

#: Where the committed data lives.
S2_DIR = Path(__file__).parent / "tlog" / "s2"
SAMPLE_PATH = S2_DIR / "sample.jsonl"
SELECTION_PATH = S2_DIR / "selection.json"
REPORT_PATH = S2_DIR / "report.json"
DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "tlog-speaker-clustering.md"
PREREGISTRATION_PATH = Path(__file__).resolve().parents[2] / "docs" / "tlog-holdout-preregistration.md"
TARTEEL_DIR = Path(__file__).parent / "nonpro_eval" / "tarteel_v1"
DECODES_PATH = TARTEEL_DIR / "decodes.base.jsonl.gz"
SPLIT_PATH = TARTEEL_DIR / "split.json"


def sample_shards() -> tuple[int, ...]:
    """The sample's frame: every 17th training-complement shard, the first 20."""
    from training.distill_stream import held_out_shards

    reserved = held_out_shards()
    complement = [shard for shard in range(NUM_SHARDS) if shard not in reserved]
    return tuple(complement[::SAMPLE_STRIDE][:SAMPLE_SHARD_COUNT])


#: Derived from the reservations, so a changed held-out block moves the sample with it.
SAMPLE_SHARDS = sample_shards()


# --- the Tarteel v1 look duration deciles ----------------------------------------------------


@lru_cache(maxsize=1)
def tarteel_look() -> tuple[tuple[float, ...], float, int]:
    """The look pool's decile cut points, its longest clip and its clip count.

    Durations are the committed ``start_sample``/``end_sample`` spans over 16 kHz, as
    ``tadabur.tarteel_excess`` reads them for the same decodes.
    """
    look = set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["look"])
    durations = []
    with gzip.open(DECODES_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["audio_filename"] in look:
                durations.append((row["end_sample"] - row["start_sample"])
                                 / speaker_embed.SAMPLE_RATE)
    if len(durations) < 10:
        raise ValueError(f"{len(look)} look clips, too few for deciles")
    durations.sort()
    count = len(durations)
    cuts = (durations[0], *(durations[round(count * index / 10)] for index in range(1, 10)))
    return cuts, durations[-1], count


def bin_edges() -> tuple[float, ...]:
    """The ten bins' eleven edges: the look decile cuts, the top open."""
    cuts, _, _ = tarteel_look()
    return (*cuts, float("inf"))


def bin_of(seconds: float) -> int | None:
    """The Tarteel look decile a duration falls in, or ``None`` outside the look range."""
    cuts, longest, _ = tarteel_look()
    if not cuts[0] <= seconds <= longest:
        return None
    edges = (*cuts, float("inf"))
    for decile in range(10):
        if edges[decile] <= seconds < edges[decile + 1]:
            return decile
    return None  # unreachable inside the range; keeps the contract total


# --- the draw --------------------------------------------------------------------------------


def _rank(text: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()


def _sample_row(row: dict) -> dict:
    """One committed row: the clip's identity, its true first-20-s span, and its duration."""
    return {
        "audio_filename": row["audio_filename"],
        "shard": row["shard"],
        "row": row["row_index"],
        "reciter_id": row["reciter_id"],
        "surah_ayah": f"{row['surah_id'] + 1}:{row['ayah_id']}",
        "start_sample": 0,
        "end_sample": min(SPAN_SAMPLES,
                          round(row["ayah_duration_s"] * speaker_embed.SAMPLE_RATE)),
        "ayah_duration_s": row["ayah_duration_s"],
    }


def draw_sample(frame, *, per_bin: int = PER_BIN, coverage_ayat: int = COVERAGE_AYAT,
                coverage_reciters: int = COVERAGE_RECITERS, cap: int = RECITER_CAP,
                salt: str = SALT) -> list[dict]:
    """The fixed-seed sample from a metadata frame: the constraints, not a preference order."""
    rows = [(bin_of(float(row["ayah_duration_s"])), row) for row in frame]
    rows = [(decile, row) for decile, row in rows if decile is not None]
    if not rows:
        raise ValueError("no frame row has an ayah duration inside the Tarteel look range")

    def rank(text: str) -> str:
        return _rank(text, salt)

    def clip_rank(entry) -> str:
        return rank(f"clip:{entry[1]['audio_filename']}")

    chosen: list[dict] = []
    seen: set[tuple] = set()
    counts: Counter = Counter()
    filled: Counter = Counter()

    def take(decile: int, row: dict) -> bool:
        key = (row["reciter_id"], row["surah_id"], row["ayah_id"])
        if key in seen or counts[row["reciter_id"]] >= cap or filled[decile] >= per_bin:
            return False
        seen.add(key)
        counts[row["reciter_id"]] += 1
        filled[decile] += 1
        chosen.append(row)
        return True

    # 1. Coverage: the ayat the frame can spread over the most reciters.
    by_ayah: dict[tuple[int, int], list] = defaultdict(list)
    for decile, row in rows:
        by_ayah[(row["surah_id"], row["ayah_id"])].append((decile, row))
    candidates = sorted(
        ((-len({row["reciter_id"] for _, row in entries}), rank(f"ayah:{ayah}"), ayah)
         for ayah, entries in by_ayah.items()
         if len({row["reciter_id"] for _, row in entries}) >= coverage_reciters))
    if len(candidates) < coverage_ayat:
        raise ValueError(f"the frame has {len(candidates)} ayat with {coverage_reciters} "
                         f"reciters, fewer than {coverage_ayat}")
    targeted = []
    for _, _, ayah in candidates[:coverage_ayat]:
        reciters = sorted({row["reciter_id"] for _, row in by_ayah[ayah]},
                          key=lambda reciter: rank(f"reciter:{reciter}"))
        for reciter in reciters[:coverage_reciters]:
            clips = sorted((entry for entry in by_ayah[ayah]
                            if entry[1]["reciter_id"] == reciter), key=clip_rank)
            if not take(clips[0][0], clips[0][1]):
                raise ValueError(f"ayah {ayah}, reciter {reciter}: its decile is already full")
        targeted.append(ayah)

    # 2. Fill every decile to the quota in clip-rank order, one clip per reciter per pass.
    by_reciter: dict[int, list] = defaultdict(list)
    for decile, row in rows:
        by_reciter[row["reciter_id"]].append((decile, row))
    for entries in by_reciter.values():
        entries.sort(key=clip_rank)
    reciters = sorted(by_reciter, key=lambda reciter: rank(f"reciter:{reciter}"))
    while True:
        missing = [decile for decile in range(10) if filled[decile] < per_bin]
        if not missing:
            break
        added = 0
        for reciter in reciters:
            if counts[reciter] >= cap:
                continue
            used = set()
            for decile, row in by_reciter[reciter]:
                if decile in used or filled[decile] >= per_bin:
                    continue
                if take(decile, row):
                    used.add(decile)
                    added += 1
        if not added:
            raise ValueError(f"the frame cannot fill deciles {missing}: "
                             f"{sorted(filled.items())}")

    per_ayah: dict[tuple[int, int], set] = defaultdict(set)
    for row in chosen:
        per_ayah[(row["surah_id"], row["ayah_id"])].add(row["reciter_id"])
    for ayah in targeted:
        if len(per_ayah[ayah]) < coverage_reciters:
            raise ValueError(f"ayah {ayah} lost its reciter spread during the fill")

    chosen.sort(key=lambda row: (row["shard"], row["row_index"]))
    return [_sample_row(row) for row in chosen]


def _frame_sha256(frame) -> str:
    ordered = sorted(frame, key=lambda row: (row["shard"], row["row_index"]))
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in ordered]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def selection(rows: list[dict], frame) -> dict:
    """The census that makes the committed sample checkable and its draw reproducible."""
    decile_of = {row["audio_filename"]: bin_of(float(row["ayah_duration_s"])) for row in frame}
    bins = [0] * 10
    for row in rows:
        bins[decile_of[row["audio_filename"]]] += 1
    per_ayah: dict[str, set] = defaultdict(set)
    for row in rows:
        per_ayah[row["surah_ayah"]].add(row["reciter_id"])
    return {
        "salt": SALT,
        "shards": list(SAMPLE_SHARDS),
        "frame_rows": len(frame),
        "frame_sha256": _frame_sha256(frame),
        "clips": len(rows),
        "reciters": len({row["reciter_id"] for row in rows}),
        "ayat_with_five_reciters": sum(1 for reciters in per_ayah.values()
                                       if len(reciters) >= COVERAGE_RECITERS),
        "bins": bins,
        "per_bin": PER_BIN,
        "coverage_ayat": COVERAGE_AYAT,
        "coverage_reciters": COVERAGE_RECITERS,
        "reciter_cap": RECITER_CAP,
        "span_samples": SPAN_SAMPLES,
    }


def build_sample(shards: tuple[int, ...] = SAMPLE_SHARDS) -> tuple[list[dict], list[dict]]:
    """Stream the frame's metadata, draw the sample, and return both."""
    from .shard_reader import iter_shard_metadata

    frame = list(iter_shard_metadata(shards))
    return draw_sample(frame), frame


# --- the sample file and its guard -----------------------------------------------------------


def read_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    """The committed sample, in file order."""
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_sample(rows: list[dict], path: Path = SAMPLE_PATH) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    write_text_atomically(Path(path), text)


def assert_tadabur_sample(rows: list[dict]) -> None:
    """Refuse anything that is not Tadabur audio from the training-complement shards."""
    from training.distill_stream import held_out_shards

    reserved = held_out_shards()
    for row in rows:
        name = row["audio_filename"]
        if not name.startswith("tadabur_") or not name.endswith(".wav"):
            raise ValueError(f"{name}: not a Tadabur clip; S2 never reads tlog audio")
        if row["shard"] in reserved:
            raise ValueError(f"{name}: shard {row['shard']} is held out of training")


# --- the pinned models and the embedding pass ------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


#: The encoder stack whose versions the pins and the report record: candidate B's numbers
#: depend on ``transformers``' x-vector implementation, exactly as A's depend on ECAPA's.
LIBRARIES = ("torch", "transformers", "speechbrain")


def library_versions() -> dict[str, str | None]:
    """The installed encoder-stack versions, ``None`` where a package is absent."""
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in LIBRARIES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def write_pins(out_dir: Path, *, sample_sha256: str, code_sha: str,
               models=speaker_embed.MODELS, device: str = "cuda") -> Path:
    """Record every model's revision and library versions before the first embedding."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_sha256": sample_sha256,
        "code_sha": code_sha,
        "device": device,
        "libraries": library_versions(),
        "models": {model.key: {"repo_id": model.repo_id, "revision": model.revision,
                               "dim": model.dim} for model in models},
    }
    path = out_dir / "pins.json"
    write_text_atomically(path, json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    for model in models:
        print(f"model {model.key}: {model.repo_id} @ {model.revision} (dim {model.dim})",
              flush=True)
    libraries = ", ".join(f"{name} {value}" for name, value in payload["libraries"].items())
    print(f"libraries: {libraries}", flush=True)
    return path


def embed_sample(clip_ids: list[str], waveforms, model: speaker_embed.Model, out_dir: Path,
                 *, encoder) -> np.ndarray:
    """Embed one model's pass; refuse to run until that model's revision is pinned."""
    from .speaker_cluster import normalise

    out_dir = Path(out_dir)
    pins = json.loads((out_dir / "pins.json").read_text(encoding="utf-8"))
    recorded = pins["models"].get(model.key)
    if recorded is None or recorded["revision"] != model.revision:
        raise ValueError(f"model {model.key} is not pinned in {out_dir / 'pins.json'}")
    matrix = normalise(encoder(list(waveforms)))
    if matrix.shape != (len(clip_ids), model.dim):
        raise ValueError(f"encoder returned {matrix.shape}, expected "
                         f"{(len(clip_ids), model.dim)}")
    np.savez(out_dir / f"embeddings.{model.key}.npz",
             clip_ids=np.array([str(clip) for clip in clip_ids]), embeddings=matrix)
    return matrix


def stream_waveforms(rows: list[dict]):
    """Yield ``(sample row, first-20-s waveform)`` one Tadabur shard at a time, deleting it."""
    from .audio import decode_to_mono_16k
    from .shard_reader import iter_selected_rows

    by_shard: dict[int, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        by_shard[row["shard"]][row["row"]] = row
    for shard in sorted(by_shard):
        selected = by_shard[shard]
        for position, streamed in enumerate(iter_selected_rows(shard, set(selected))):
            if streamed is None:
                continue
            sample_row = selected[position]
            waveform = decode_to_mono_16k(streamed["audio"]["bytes"])[:SPAN_SAMPLES]
            yield sample_row, waveform


def embed_all(rows: list[dict], out_dir: Path, *, models=speaker_embed.MODELS,
              device: str = "cuda") -> dict[str, np.ndarray]:
    """One decode pass over the sample, then each model's embedding pass over the waveforms."""
    encoders = {model.key: speaker_embed.encoder(model, device=device) for model in models}
    clip_ids, waveforms = [], []
    for row, waveform in stream_waveforms(rows):
        clip_ids.append(row["audio_filename"])
        waveforms.append(waveform)
        if len(clip_ids) % 200 == 0:
            print(f"decoded {len(clip_ids)}/{len(rows)} clips", flush=True)
    matrices = {}
    for model in models:
        matrices[model.key] = embed_sample(clip_ids, waveforms, model, out_dir,
                                           encoder=encoders[model.key])
        print(f"embedded model {model.key}: {matrices[model.key].shape}", flush=True)
    return matrices


# --- the report ------------------------------------------------------------------------------


def analyse(sample_rows: list[dict], embeddings: dict) -> dict:
    """Every grid τ's metrics for every model, with Wilson intervals on the co-cluster rates."""
    from training.counterfactual_eval import wilson_interval

    ids = [row["audio_filename"] for row in sample_rows]
    reciters = {row["audio_filename"]: row["reciter_id"] for row in sample_rows}
    ayahs = {row["audio_filename"]: row["surah_ayah"] for row in sample_rows}
    results = {}
    for key, (clip_ids, matrix) in embeddings.items():
        position = {clip: index for index, clip in enumerate(clip_ids)}
        missing = set(ids) - set(position)
        if missing or len(clip_ids) != len(ids):
            raise ValueError(f"model {key}'s embeddings do not cover the sample: "
                             f"{len(missing)} missing of {len(ids)}")
        ordered = np.asarray(matrix)[[position[clip] for clip in ids]]
        per_tau = {}
        for tau in GRID:
            entry = metrics(cluster(ids, ordered, tau), reciters, ayahs)
            for block in ("pre_cap", "post_cap"):
                for stratum in ("same_reciter", "same_ayah", "different_ayah"):
                    cell = entry[block][stratum]
                    cell["wilson"] = list(wilson_interval(cell["clustered"], cell["pairs"]))
            per_tau[tau] = entry
        results[key] = per_tau
    return results


def build_report(*, sample_path: Path = SAMPLE_PATH, embeddings_dir: Path,
                 selection_path: Path = SELECTION_PATH) -> dict:
    """Compute the grid and the choice, and assemble the committed report."""
    rows = read_sample(sample_path)
    census = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    pins = json.loads((Path(embeddings_dir) / "pins.json").read_text(encoding="utf-8"))
    if "libraries" not in pins:
        raise ValueError(f"{embeddings_dir}/pins.json records no library versions; rerun "
                         f"`python -m tadabur.tlog_s2 embed` to write them")
    embeddings = {}
    for model in speaker_embed.MODELS:
        stored = np.load(Path(embeddings_dir) / f"embeddings.{model.key}.npz")
        embeddings[model.key] = (list(stored["clip_ids"]), stored["embeddings"])
    results = analyse(rows, embeddings)
    outcomes = {key: model_outcome(per_tau) for key, per_tau in results.items()}
    choice = decide(results)
    if "model" in choice:
        choice["revision"] = pins["models"][choice["model"]]["revision"]
    cuts, longest, clips = tarteel_look()
    return {
        "issue": 186,
        "preregistration": {"path": "docs/tlog-holdout-preregistration.md",
                            "sha256": sha256_file(PREREGISTRATION_PATH)},
        "tarteel_look": {"path": "tools/tadabur/nonpro_eval/tarteel_v1/decodes.base.jsonl.gz",
                         "sha256": sha256_file(DECODES_PATH), "clips": clips,
                         "edges": list(cuts), "min_seconds": cuts[0],
                         "max_seconds": longest},
        "sample": {"path": "tools/tadabur/tlog/s2/sample.jsonl",
                   "sha256": sha256_file(sample_path), "salt": census["salt"],
                   "shards": census["shards"], "clips": len(rows),
                   "reciters": census["reciters"],
                   "ayat_with_five_reciters": census["ayat_with_five_reciters"],
                   "reciter_cap": census["reciter_cap"], "bins": census["bins"]},
        "embedding_run": {"code_sha": pins["code_sha"], "device": pins["device"],
                          "sample_sha256": pins["sample_sha256"],
                          "libraries": pins["libraries"]},
        "models": {model.key: {"repo_id": model.repo_id, "revision": model.revision,
                               "dim": model.dim} for model in speaker_embed.MODELS},
        "grid": list(GRID),
        "results": {key: {f"{tau:.2f}": entry for tau, entry in per_tau.items()}
                    for key, per_tau in results.items()},
        "per_model": outcomes,
        "choice": choice,
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _interval(cell: dict) -> str:
    low, high = cell["wilson"]
    return f"{_pct(cell['rate'])} [{_pct(low)}, {_pct(high)}]"


def render(report: dict) -> str:
    """The generated page; every number is read from the report."""
    lines = [
        "# Speaker clustering on Tadabur (#186)",
        "",
        "Generated by `python -m tadabur.tlog_s2 report` (from `tools/`); do not edit by "
        "hand. The machine-readable report is "
        "[`tools/tadabur/tlog/s2/report.json`](../tools/tadabur/tlog/s2/report.json).",
        "",
        "> **Tadabur only.** No tlog audio was read. The threshold and the model below are the "
        "ones the frozen rule selects; nothing else was tried.",
        "",
        "## Reading of the pre-registration",
        "",
        "`docs/tlog-holdout-preregistration.md` (\"Groups and roles\", items 2-3) is applied "
        "as written, with three readings. The kNN graph is the undirected union of each row's "
        "20 nearest by cosine: exactly 20 per row, ordered by cosine and, on an exact tie, by "
        "ascending clip id. A cutoff tie widens only the candidate set, never the graph, and "
        "the blocked search is not authoritative — every candidate's float64 cosine is "
        "recomputed once on the CPU before the 20 are chosen and before an edge is tested "
        "against τ, so block shape and backend cannot move an edge. The 1% cap "
        "re-clusters an oversized component inside itself at τ + 0.02, then +0.02 a round, "
        "until no component exceeds the cap; the threshold stops at 1.0, above which no cosine "
        "edge survives. The τ pick, the chaining and text-independence tests and the recall "
        "bar are read from the components **before** the cap (architect decision 2026-10-24); "
        "the post-cap numbers are reported beside them for information.",
        "",
        "## Inputs",
        "",
        "| | |",
        "|---|---|",
        f"| pre-registration | `{report['preregistration']['path']}` "
        f"sha256 `{report['preregistration']['sha256'][:12]}` |",
        f"| Tarteel v1 look decodes | `{report['tarteel_look']['path']}` "
        f"sha256 `{report['tarteel_look']['sha256'][:12]}`, "
        f"{report['tarteel_look']['clips']:,} clips |",
        f"| sample | `{report['sample']['path']}` "
        f"sha256 `{report['sample']['sha256'][:12]}`, salt `{report['sample']['salt']}` |",
        f"| frame | shards {', '.join(str(shard) for shard in report['sample']['shards'])} |",
        f"| embedding run | code `{report['embedding_run']['code_sha'][:12]}`, "
        f"{report['embedding_run']['device']} |",
        f"| models | "
        + "; ".join(f"{key} `{record['repo_id']}` @ `{record['revision']}`"
                    for key, record in report["models"].items()) + " |",
        "",
        "## The sample",
        "",
        f"{report['sample']['clips']:,} clips from "
        f"{report['sample']['reciters']} reciters, at most "
        f"{report['sample']['reciter_cap']} each, with "
        f"{report['sample']['ayat_with_five_reciters']} ayat recited by at least "
        "five sampled reciters. Clips per Tarteel look decile: "
        + ", ".join(str(count) for count in report["sample"]["bins"]) + ".",
        "",
        "The look decile cut points (seconds): "
        + ", ".join(f"{edge:.3f}" for edge in report["tarteel_look"]["edges"]) + ".",
        "",
    ]
    for key, record in report["models"].items():
        lines.extend([
            f"## Model {key.upper()}: `{record['repo_id']}` @ `{record['revision']}`",
            "",
            "| τ | largest cluster (pre-cap) | recall (pre-cap) | same ayah, different reciter "
            "| different ayah, different reciter | largest cluster (post-cap) | re-splits |",
            "|---|---|---|---|---|---|---|",
        ])
        chosen = report["choice"].get("tau") if report["choice"].get("model") == key else None
        for tau in report["grid"]:
            entry = report["results"][key][f"{tau:.2f}"]
            mark = " *" if chosen == tau else ""
            lines.append(
                f"| {tau:.2f}{mark} | {entry['pre_cap']['largest_cluster']} "
                f"({_pct(entry['pre_cap']['largest_cluster_share'])}) "
                f"| {_pct(entry['pre_cap']['recall'])} "
                f"| {_interval(entry['pre_cap']['same_ayah'])} "
                f"| {_interval(entry['pre_cap']['different_ayah'])} "
                f"| {_pct(entry['post_cap']['largest_cluster_share'])} "
                f"| {entry['resplits']} |")
        lines.append("")
        outcome = report["per_model"][key]
        if outcome["tau"] is None:
            lines.append(f"No τ on the grid meets both chaining and text-independence for "
                         f"model {key.upper()}.")
        elif not outcome["pass"]:
            lines.append(f"τ = {outcome['tau']:.2f} is model {key.upper()}'s lowest admissible "
                         f"value; same-reciter recall {_pct(outcome['recall'])} is below the "
                         f"{_pct(RECALL_FLOOR)} bar, so the model fails.")
        else:
            lines.append(f"`*` marks τ = {outcome['tau']:.2f}, the value the rule selects for "
                         f"model {key.upper()} (recall {_pct(outcome['recall'])}).")
        lines.append("")
    choice = report["choice"]
    lines.extend(["## Choice", ""])
    if "fallback" in choice:
        lines.extend([
            "**Fallback.** Neither candidate passed the rule, so the pre-registered fallback "
            "applies: the group is the id group alone, and every tlog verdict states "
            "\"clip-level split, same users possible on both sides\".",
        ])
    else:
        record = report["models"][choice["model"]]
        lines.extend([
            f"Model {choice['model'].upper()} (`{record['repo_id']}` @ "
            f"`{record['revision']}`) at τ = {choice['tau']:.2f}.",
            "",
            "That is the first candidate on the pre-registered grid, with no chaining, "
            "text-independent co-clustering, and same-reciter recall at or above the bar.",
        ])
    lines.append("")
    return "\n".join(lines)


# --- the commands ----------------------------------------------------------------------------


def _sample_command(args) -> None:
    rows, frame = build_sample()
    write_sample(rows, args.out)
    census = selection(rows, frame)
    write_text_atomically(Path(args.selection),
                          json.dumps(census, indent=1, ensure_ascii=False) + "\n")
    print(f"{census['clips']} clips, {census['reciters']} reciters, "
          f"{census['ayat_with_five_reciters']} ayat with five reciters, bins {census['bins']}",
          flush=True)


def _embed_command(args) -> None:
    rows = read_sample(args.sample)
    if args.limit:
        rows = rows[:args.limit]
    assert_tadabur_sample(rows)
    out_dir = Path(args.out_dir)
    write_pins(out_dir, sample_sha256=sha256_file(Path(args.sample)), code_sha=head_sha(),
               device=args.device)
    embed_all(rows, out_dir, device=args.device)


def _report_command(args) -> None:
    report = build_report(sample_path=Path(args.sample), embeddings_dir=Path(args.embeddings_dir))
    write_text_atomically(Path(args.report),
                          json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    write_text_atomically(Path(args.doc), render(report))
    print(json.dumps(report["choice"], ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="draw and write the committed sample")
    sample.add_argument("--out", type=Path, default=SAMPLE_PATH)
    sample.add_argument("--selection", type=Path, default=SELECTION_PATH)
    sample.set_defaults(run=_sample_command)

    embed = commands.add_parser("embed", help="pin the models and embed the sample (GPU)")
    embed.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    embed.add_argument("--out-dir", type=Path, required=True,
                       help="scratch directory for pins.json and the embedding npz files")
    embed.add_argument("--device", default="cuda")
    embed.add_argument("--limit", type=int, default=0,
                       help="embed only the first N sample rows (the one-clip smoke)")
    embed.set_defaults(run=_embed_command)

    report = commands.add_parser("report", help="run the grid and write the report and page")
    report.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    report.add_argument("--embeddings-dir", type=Path, required=True)
    report.add_argument("--report", type=Path, default=REPORT_PATH)
    report.add_argument("--doc", type=Path, default=DOC_PATH)
    report.set_defaults(run=_report_command)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

"""Synthetic edits of real recitations, each paired with an unedited decoy (#88).

ADR-0011 §3 admits synthetic edits as training signal only when every edit is paired 1:1
with an **unedited decoy** labelled unchanged, and only after a blind listen confirms the
edits sound real. This module makes both, from audio of reciters that appear in no
evaluation item, and writes the blind-check worklist for that listen.

**The edit frame** (:func:`edit_frame`). Acceptance rules §6 require reciter-level
disjointness: no edit source or donor may be a reciter of any evaluation item. Every
evaluation item this work has (the truth sites, the frozen ``decode_evalset``, the mining
pool #87 listens to) and every one it plans (the sealed panel, #89) comes from the shards
``h448`` never trained on: the held-out block 0-20 and the strided reserve. So the frame is
the rows of the **other** shards whose reciter has no row at all in those shards. Those
reciters cannot reach an evaluation item, now or later, without a new shard range being
opened for evaluation. The frame is a census: every eligible clip is staged.

**Operations** (:data:`OPERATIONS`), timed by the base teacher's CTC frames
(:class:`tadabur.synthetic_edit_plan.TimedClip`, from a whole-clip
:meth:`training.decoding.Decoder.span_class_ids` row):

* ``shaddah_removed`` crops a doubled consonant from the centre of its first emission to
  the centre of its second (the held part; for a stop, its closure), so one is left;
* ``shaddah_added`` stretches a single voiceless fricative's hold by the median held span;
* ``consonant_swap`` splices the carrier's cell, through its following haraka, from a
  same-reciter donor of the other letter of a fricative pair (``س↔ص``, ``ذ↔ز``, ``ض↔ظ``,
  ``ذ↔ظ``) with the same haraka.

**Decoys.** Each edit's decoy is the same source clip with the same timing change at a
neutral place, so the edited and the unedited version differ only in where the change
falls. A crop or a stretch is made with the same length inside a long madd (a run of at
least :data:`MIN_MADD_RUN` madd letters, far from the carrier). A swap's decoy splices the
carrier's cell from a same-reciter donor of the **same** letter, so it carries the same
splice at the same place and the same length change (none). The decoy is labelled with
the source's state at the carrier, the edit with the changed one.

Usage (from ``tools/``; ``stage`` downloads one 2.4 GB shard at a time, ``decode`` needs
the GPU)::

  python -m tadabur.synthetic_edits frame --unseen-index stage/shard_index.jsonl \\
      --train-index stage/train_shard_index.jsonl \\
      --evalset-manifest tadabur/gate_eval/manifest.json
  python -m tadabur.synthetic_edits stage --train-index stage/train_shard_index.jsonl \\
      --audio-dir stage/clips --shard-cache stage/hf_cache
  python -m tadabur.synthetic_edits decode --audio-dir stage/clips
  python -m tadabur.synthetic_edits generate --audio-dir stage/clips --out-dir stage/edits
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .staged_audio import IndexRow, StagedClip, read_shard_index
from .synthetic_edit_plan import (
    CROP,
    DECOY,
    EDIT,
    OPERATIONS,
    STRETCH,
    EditPair,
    TimedClip,
    anchored_positions,
    carrier_positions,
    labelled_reference,
    plan_pairs,
    rank,
    select_pairs,
)
from .truth_sites import MARKS, PENDING, SYNTHETIC_EDIT, TruthSite
from .waveform_edits import FADE, DoesNotFit, Rendered, crop, splice, stretch

SYNTHETIC_EDITS_DIR = Path(__file__).parent / "synthetic_edits"
FRAME_PATH = SYNTHETIC_EDITS_DIR / "frame.json"
BASE_FRAMES_PATH = SYNTHETIC_EDITS_DIR / "base_frames.json"
EDITS_PATH = SYNTHETIC_EDITS_DIR / "edits.jsonl"
SUMMARY_PATH = SYNTHETIC_EDITS_DIR / "summary.json"
WORKLIST_PATH = SYNTHETIC_EDITS_DIR / "blind_check.jsonl"
TEACHER_CHECK_PATH = SYNTHETIC_EDITS_DIR / "teacher_check.json"

SALT = "issue-88-synthetic-edits-v1"


# --- the edit frame --------------------------------------------------------------------


def h448_unseen_shards() -> list[int]:
    """The shards ``h448`` never trained on: the held-out block and the strided reserve.
    Every evaluation item is drawn from these."""
    from training.decode_evalset import gate_eval_shards

    from .mining_pool import HELD_OUT_BLOCK

    return sorted(set(HELD_OUT_BLOCK) | set(gate_eval_shards()))


def edit_frame(
    train_rows: Iterable[IndexRow], unseen_rows: Iterable[IndexRow]
) -> tuple[list[IndexRow], Counter, list[int]]:
    """The clips edits may come from, the training-shard rows left out by reason, and the
    reciters every evaluation item is drawn from (sorted).

    A row is eligible when it is in a shard ``h448`` trained on, its reciter has no row in
    any ``h448``-unseen shard, it is 1.5-50 s long (``decode_evalset``'s bounds) and its
    ayah can be phonetized. ``unseen_rows`` must index every unseen shard completely, or
    a reciter of a future evaluation item could pass for an unseen one.
    """
    import generate_phonemes
    from training.decode_evalset import MAX_CLIP_SECONDS, MIN_CLIP_SECONDS

    unseen = set(h448_unseen_shards())
    unseen_rows = list(unseen_rows)
    if {row.shard for row in unseen_rows} != unseen:
        raise ValueError("the unseen index must cover every h448-unseen shard")
    evaluation_reciters = sorted({row.reciter_id for row in unseen_rows})
    blocked = set(evaluation_reciters)
    frame: list[IndexRow] = []
    excluded: Counter = Counter()
    for row in train_rows:
        if row.shard in unseen:
            raise ValueError(f"{row.audio_filename} is in unseen shard {row.shard}")
        if row.reciter_id in blocked:
            excluded["reciter_in_unseen_shards"] += 1
        elif not MIN_CLIP_SECONDS <= row.duration_s <= MAX_CLIP_SECONDS:
            excluded["duration"] += 1
        elif row.surah_ayah in generate_phonemes.FALLBACK_PHONEMES:
            excluded["phonetizer_unsupported"] += 1
        else:
            frame.append(row)
    frame.sort(key=lambda r: r.audio_filename)
    return frame, excluded, evaluation_reciters


def evalset_reciters(manifest_path: Path) -> list[int]:
    """The reciters of the frozen ``decode_evalset``, refusing a clip outside the unseen
    shards (the frame's reasoning assumes every evaluation item is drawn from them)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unseen = set(h448_unseen_shards())
    outside = sorted({c["shard"] for c in manifest["clips"]} - unseen)
    if outside:
        raise ValueError(f"{manifest_path} has clips in trained shards {outside}")
    return sorted({c["reciter_id"] for c in manifest["clips"]})


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(args) -> None:
    unseen_rows = read_shard_index(args.unseen_index).values()
    train_rows = read_shard_index(args.train_index).values()
    frame, excluded, evaluation = edit_frame(train_rows, unseen_rows)
    evalset = evalset_reciters(args.evalset_manifest)
    if not set(evalset) <= set(evaluation):
        raise ValueError("a decode_evalset reciter is missing from the unseen index")
    per_reciter = Counter(row.reciter_id for row in frame)
    record = {
        "rule": "rows of h448-trained shards whose reciter has no row in an h448-unseen "
                "shard; 1.5-50 s; phonetizable ayah",
        "salt": SALT,
        "unseen_shards": h448_unseen_shards(),
        "unseen_index_sha256": _file_sha256(args.unseen_index),
        "train_index_sha256": _file_sha256(args.train_index),
        "train_index_rows": len(train_rows),
        "decode_evalset_manifest_sha256": _file_sha256(args.evalset_manifest),
        "decode_evalset_reciters": evalset,
        "evaluation_reciters": evaluation,
        "excluded": dict(sorted(excluded.items())),
        "frame_clips": len(frame),
        "frame_reciters": len(per_reciter),
        "clips_per_reciter": {str(k): v for k, v in sorted(per_reciter.items())},
        "clips": [row.audio_filename for row in frame],
    }
    FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRAME_PATH.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("clips", "clips_per_reciter", "evaluation_reciters",
                                   "decode_evalset_reciters")}, indent=2))


def read_frame(path: Path = FRAME_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_in_parallel(
    groups: list[dict[str, frozenset[str]]],
    known: dict[str, StagedClip],
    run,
    write,
    workers: int,
) -> tuple[dict[str, StagedClip], list[str]]:
    """Stage request ``groups`` on ``workers`` threads without losing provenance.

    ``run(requests, checkpoint)`` stages one group (``staged_audio.stage_clips``) and calls
    ``checkpoint`` with what it holds after each shard; ``write(entries)`` persists the
    registry's synthetic-edit entries. Every checkpoint writes **every** entry ``known``
    held when the run began, overlaid with whatever any worker has staged since, so an
    interruption never drops a recorded checksum that a resume must re-verify. Entries
    are pruned to exactly the staged clips only after every group has finished.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    entries = dict(known)
    lock = threading.Lock()

    def checkpoint(staged: dict[str, StagedClip]) -> None:
        with lock:
            entries.update(staged)
            write(dict(entries))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda group: run(group, checkpoint), groups))
    staged = {name: clip for found, _ in results for name, clip in found.items()}
    write(staged)
    return staged, [name for _, missing in results for name in missing]


def _stage(args) -> None:
    """Stage every frame clip into ``--audio-dir`` and register it (use ``synthetic_edit``).

    The frame's clips are spread one or two per training shard, so the shards are read by
    ``--workers`` threads at once, each over its own shards; the registry is rewritten
    after every shard with the clips staged so far, so a resumed run re-verifies rather
    than re-downloads them.
    """
    from .staged_audio import (
        REGISTRY_PATH,
        SYNTHETIC_EDIT,
        load_staged_clips,
        stage_clips,
        write_staged_clips,
    )

    frame = read_frame()
    if _file_sha256(args.train_index) != frame["train_index_sha256"]:
        raise SystemExit(f"{args.train_index} is not the index the frame was made from")
    index = read_shard_index(args.train_index)
    registry = load_staged_clips(REGISTRY_PATH)
    others = {n: c for n, c in registry.items() if SYNTHETIC_EDIT not in c.uses}
    known = {n: c for n, c in registry.items() if SYNTHETIC_EDIT in c.uses}
    if set(frame["clips"]) & set(others):
        raise SystemExit("a frame clip is already registered for another use")
    groups: dict[int, dict[str, frozenset[str]]] = {}
    for name in frame["clips"]:
        groups.setdefault(index[name].shard % args.workers, {})[name] = frozenset({SYNTHETIC_EDIT})

    wanted: dict[int, set[int]] = {}
    for name in frame["clips"]:
        wanted.setdefault(index[name].shard, set()).add(index[name].row_index)

    def shard_rows(shard: int):
        """The shard's rows in order, materialized only where a frame clip sits.

        A shard holds one or two frame clips among ~1,000 rows, and turning every row's
        audio into Python bytes costs more than downloading it. ``stage_clips`` reads a
        row only at a wanted index, so every other position is a ``None`` placeholder.
        """
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        from .dataset_source import DATASET_ID
        from .shard_reader import SHARD_TEMPLATE, _remove_shard_blob

        path = hf_hub_download(DATASET_ID, SHARD_TEMPLATE.format(index=shard),
                               repo_type="dataset", cache_dir=str(args.shard_cache))
        try:
            position = 0
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=64, columns=["audio", "reciter_id"]):
                for offset in range(batch.num_rows):
                    keep = position + offset in wanted[shard]
                    yield batch.slice(offset, 1).to_pylist()[0] if keep else None
                position += batch.num_rows
        finally:
            _remove_shard_blob(path)

    def write(entries: dict[str, StagedClip]) -> None:
        write_staged_clips([*others.values(), *entries.values()], REGISTRY_PATH)
        print(f"  {len(entries)} edit clip entries registered", flush=True)

    def run(requests, checkpoint):
        return stage_clips(requests, index, args.audio_dir, shard_rows, known,
                           on_shard_done=checkpoint)

    staged, unlocatable = stage_in_parallel(
        [groups[k] for k in sorted(groups)], known, run, write, args.workers)
    print(f"Staged {len(staged)} edit clips into {args.audio_dir}")
    if unlocatable:
        raise SystemExit(f"{len(unlocatable)} frame clips are in no indexed shard")


# --- the base teacher's frame times ----------------------------------------------------


def _decode(args) -> None:
    """Decode every staged frame clip whole with the base teacher and commit its CTC
    segments and realized reference (``base_frames.json``), so edits can be planned and
    re-rendered anywhere without a GPU."""
    import soundfile as sf

    import hafs_phonetizer
    from training.decoding import SPANS, Decoder, scan_ctc

    from .mining_pool import whole_ayah_references
    from .resegment import BASE_TEACHER, DECODE_BATCH_SIZE, WEIGHTS_DTYPE
    from .staged_audio import verify_staged

    clips = edit_clips()
    if sorted(clips) != sorted(read_frame()["clips"]):
        raise SystemExit("the registry's synthetic-edit clips are not the frame's clips")
    decoder = Decoder.load(
        BASE_TEACHER, args.device, weights_dtype=WEIGHTS_DTYPE, batch_size=DECODE_BATCH_SIZE
    )
    references = whole_ayah_references(c.surah_ayah for c in clips.values())
    frames = {}
    for name, clip in sorted(clips.items()):
        verify_staged(clip, args.audio_dir)
        samples, _ = sf.read(args.audio_dir / name, dtype="float32")
        (row,) = decoder.span_class_ids([samples])
        frames[name] = {
            "reference": references[clip.surah_ayah],
            "steps": [[s.token_id, s.start_step, s.end_step] for s in scan_ctc(row)],
        }
    record = {
        "decode_fingerprint": decoder.fingerprint(SPANS).as_dict(),
        "phonetizer_revision": hafs_phonetizer.REVISION,
        "clips": frames,
    }
    BASE_FRAMES_PATH.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Decoded {len(frames)} clips into {BASE_FRAMES_PATH}")


def edit_clips() -> dict[str, StagedClip]:
    """The registry's clips staged as synthetic-edit sources and donors."""
    from .staged_audio import load_staged_clips

    return {n: c for n, c in load_staged_clips().items() if SYNTHETIC_EDIT in c.uses}


def timed_clips(path: Path = BASE_FRAMES_PATH) -> list[TimedClip]:
    """Every frame clip with a realized reference, timed by its committed base decode."""
    record = json.loads(path.read_text(encoding="utf-8"))
    clips = edit_clips()
    return [
        TimedClip.from_steps(name, clips[name].reciter_id, clips[name].num_samples,
                             frame["reference"], [tuple(s) for s in frame["steps"]])
        for name, frame in sorted(record["clips"].items())
        if frame["reference"] is not None
    ]


# --- exposure ----------------------------------------------------------------------------


def truth_site_files() -> list[Path]:
    """Every committed truth-site file (a fixture's re-location log is not one)."""
    from .truth_sites import TRUTH_SITES_DIR

    return sorted(p for p in TRUTH_SITES_DIR.glob("*.jsonl")
                  if not p.name.endswith(".relocation.jsonl"))


def evaluation_reciters() -> set[int]:
    """Every reciter an evaluation item has or may have, from the committed artifacts.

    The frame's record of the ``h448``-unseen shards' reciters (where the sealed panel and
    every future evaluation item are drawn) and of the frozen ``decode_evalset``'s, plus
    the reciter of every registered clip staged for another use (the truth sites and the
    mining pool) and of every truth site. A truth site on an unregistered clip fails: its
    reciter would be unknown.
    """
    from .staged_audio import load_staged_clips
    from .truth_sites import load_truth_sites

    frame = read_frame()
    registry = load_staged_clips()
    reciters = set(frame["evaluation_reciters"]) | set(frame["decode_evalset_reciters"])
    reciters |= {c.reciter_id for c in registry.values() if c.uses != (SYNTHETIC_EDIT,)}
    for path in truth_site_files():
        for site in load_truth_sites(path):
            if site.audio_filename not in registry:
                raise ValueError(f"{path}: {site.audio_filename} is not in the staging registry")
            reciters.add(registry[site.audio_filename].reciter_id)
    return reciters


def edit_reciters(items: list[dict]) -> set[int]:
    """The reciters of every edit source and donor in a manifest."""
    used = {item["source"]["reciter_id"] for item in items}
    return used | {item["change"]["donor"]["reciter_id"]
                   for item in items if item["change"]["donor"] is not None}


#: The use names the exposure registry (#89) reserves for this work.
EXPOSURE_SOURCE = "synthetic_edit.source"
EXPOSURE_DONOR = "synthetic_edit.donor"


def exposure_rows(items: list[dict]) -> dict[str, list[dict]]:
    """This work's rows for the exposure registry (#89), per use, sorted and de-duplicated:
    every source clip whole, and every donor span with its clip's checksum. Each row is
    ``{audio_filename, shard, row_index, reciter_id, start_sample, end_sample,
    audio_sha256}``, the registry's ``Exposure`` shape."""
    from .staged_audio import load_staged_clips

    registry = load_staged_clips()

    def row(name: str, start: int | None, end: int | None) -> tuple:
        clip = registry[name]
        return (name, clip.shard, clip.row_index, clip.reciter_id, start, end,
                clip.audio_sha256)

    sources = {row(i["source"]["audio_filename"], None, None) for i in items}
    donors = {row(d["audio_filename"], d["start_sample"], d["end_sample"])
              for d in (i["change"]["donor"] for i in items) if d is not None}
    keys = ("audio_filename", "shard", "row_index", "reciter_id", "start_sample",
            "end_sample", "audio_sha256")
    return {use: [dict(zip(keys, r)) for r in sorted(rows, key=str)]
            for use, rows in ((EXPOSURE_SOURCE, sources), (EXPOSURE_DONOR, donors))}


def check_disjoint(items: list[dict]) -> None:
    """Fail if any edit source or donor is a reciter of an evaluation item."""
    overlap = sorted(edit_reciters(items) & evaluation_reciters())
    if overlap:
        raise ValueError(f"edit reciters {overlap} also appear in evaluation items")


# --- rendering and the manifest ------------------------------------------------------------


def opaque_id(item_id: str) -> str:
    """A name for an item that says nothing about it: what the blind check sees."""
    return rank(item_id, SALT)[:20]


def output_filename(item_id: str) -> str:
    return f"se_{opaque_id(item_id)}.wav"


#: A splice's donor is scaled to the RMS of the span it replaces. A donor more than this
#: factor louder or quieter than that span is incompatible: its pair is rejected rather
#: than clamped, which would leave a level jump.
MAX_DONOR_LEVEL_RATIO = 2.0
#: The largest absolute sample a rendered item may hold; PCM_16 clips anything beyond.
PEAK_LIMIT = 1.0


class Unusable(ValueError):
    """A pair the renderer refuses; ``reason`` names the check that failed."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class RenderedItem:
    """One rendered item and the donor gain it applied (``None`` without a donor)."""

    rendered: Rendered
    gain: float | None


def item_seed(item_id: str) -> int:
    return int(rank(item_id, SALT)[:8], 16)


def render(pair: EditPair, role: str, audio) -> RenderedItem:
    """One item of ``pair``, through the one chain both roles share.

    ``audio(name)`` returns a staged clip's float32 samples. A crop or a stretch goes
    through :mod:`tadabur.waveform_edits`, which picks its method from the signal alone. A
    splice's donor is scaled to the RMS of the span it replaces (a reciter's clips are
    recorded at different levels, and a level jump is a cue no listener or model should
    get); a donor beyond :data:`MAX_DONOR_LEVEL_RATIO` raises :class:`Unusable`.
    """
    change = pair.edit if role == EDIT else pair.decoy
    x = audio(pair.audio_filename)
    if change.kind == CROP:
        return RenderedItem(crop(x, change.start_sample, change.end_sample - change.start_sample), None)
    if change.kind == STRETCH:
        seed = item_seed(f"{pair.pair_id}:{role}")
        return RenderedItem(stretch(x, change.start_sample, change.inserted_samples,
                                    change.fill_region, seed), None)
    donor = change.donor
    d = audio(donor.audio_filename)
    ratio = _rms(x[change.start_sample:change.end_sample]) / max(
        _rms(d[donor.start_sample:donor.end_sample]), 1e-9)
    if not 1 / MAX_DONOR_LEVEL_RATIO <= ratio <= MAX_DONOR_LEVEL_RATIO:
        raise Unusable("donor_level", f"{pair.pair_id}:{role} needs gain {ratio:.2f}")
    gain = round(ratio, 6)
    return RenderedItem(splice(x, change.start_sample, change.end_sample, d,
                               donor.start_sample, gain), gain)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def render_pair(pair: EditPair, audio) -> dict[str, RenderedItem]:
    """Both items of ``pair``, or :class:`Unusable` when they cannot be told apart by
    identity alone: they took different methods (one periodic, one not), or either
    would clip; or when a change does not fit its region (too short for a whole-period
    unit once phase-aligned, say)."""
    try:
        items = {role: render(pair, role, audio) for role in (EDIT, DECOY)}
    except DoesNotFit as error:
        raise Unusable("does_not_fit", f"{pair.pair_id}: {error}") from error
    paths = {role: item.rendered.path for role, item in items.items()}
    if paths[EDIT] != paths[DECOY]:
        raise Unusable("render_path", f"{pair.pair_id}: {paths}")
    for role, item in items.items():
        peak = float(np.max(np.abs(item.rendered.samples)))
        if peak > PEAK_LIMIT:
            raise Unusable("peak", f"{pair.pair_id}:{role} peaks at {peak:.3f}")
    return items


def item_record(
    pair: EditPair, role: str, clips: dict[str, StagedClip], item: RenderedItem,
    output: dict,
) -> dict:
    """One manifest row: the item's full provenance, how it was rendered, and its label."""
    from dataclasses import asdict

    change = pair.edit if role == EDIT else pair.decoy
    record = asdict(change)
    if change.donor is not None:
        donor = clips[change.donor.audio_filename]
        record["donor"].update(reciter_id=donor.reciter_id, audio_sha256=donor.audio_sha256,
                               gain=item.gain)
    rendered = item.rendered
    source = clips[pair.audio_filename]
    item_id = f"{pair.pair_id}:{role}"
    return {
        "item_id": item_id,
        "pair_id": pair.pair_id,
        "role": role,
        "operation": pair.operation,
        "source": {k: getattr(source, k) for k in (
            "audio_filename", "shard", "row_index", "reciter_id", "surah_ayah",
            "num_samples", "audio_sha256")},
        "reference": pair.reference,
        "reference_index": pair.reference_index,
        "mark": pair.mark,
        "prescribed": pair.prescribed,
        "label": pair.edited if role == EDIT else pair.prescribed,
        "labelled_reference": labelled_reference(pair, role),
        "change": record,
        "length_change": change.length_change,
        "seed": item_seed(item_id),
        "render": {"path": rendered.path, "period_samples": rendered.period,
                   "changed_start": rendered.changed[0], "changed_end": rendered.changed[1]},
        "output": output,
    }


def check_manifest(items: list[dict]) -> None:
    """The pairing invariants every manifest must hold, or ``ValueError``.

    Each pair is one edit and one decoy on the same carrier of the same source, with the
    same kind of change, rendered the same way, and the same length change; the decoy is
    labelled with the prescribed state and the edit with another, and their labelled
    references follow; the output is as long as the change says, is named opaquely and
    does not clip; a donor is the source's reciter.
    """
    pairs: dict[str, dict[str, dict]] = {}
    for item in items:
        if item["item_id"] != f"{item['pair_id']}:{item['role']}":
            raise ValueError(f"{item['item_id']}: id does not name its pair and role")
        if item["role"] in pairs.setdefault(item["pair_id"], {}):
            raise ValueError(f"{item['item_id']}: duplicate role in its pair")
        pairs[item["pair_id"]][item["role"]] = item
        donor = item["change"]["donor"]
        if donor is not None and donor["reciter_id"] != item["source"]["reciter_id"]:
            raise ValueError(f"{item['item_id']}: the donor is another reciter")
        if item["output"]["num_samples"] != item["source"]["num_samples"] + item["length_change"]:
            raise ValueError(f"{item['item_id']}: output length does not follow the change")
        if item["output"]["audio_filename"] != output_filename(item["item_id"]):
            raise ValueError(f"{item['item_id']}: output is not opaquely named")
        if item["output"]["peak"] > PEAK_LIMIT:
            raise ValueError(f"{item['item_id']}: output clips")
    shared = ("operation", "source", "reference", "reference_index", "mark", "prescribed",
              "length_change")
    for pair_id, roles in sorted(pairs.items()):
        if set(roles) != {EDIT, DECOY}:
            raise ValueError(f"{pair_id}: has {sorted(roles)}, not one edit and one decoy")
        edit, decoy = roles[EDIT], roles[DECOY]
        if any(edit[k] != decoy[k] for k in shared) or \
                edit["change"]["kind"] != decoy["change"]["kind"] or \
                edit["render"]["path"] != decoy["render"]["path"]:
            raise ValueError(f"{pair_id}: the edit and its decoy differ beyond the change")
        if decoy["label"] != decoy["prescribed"] or edit["label"] == edit["prescribed"]:
            raise ValueError(f"{pair_id}: labels do not follow the edit")
        if decoy["labelled_reference"] != decoy["reference"] or \
                edit["labelled_reference"] == edit["reference"]:
            raise ValueError(f"{pair_id}: labelled references do not follow the edit")


def read_items(path: Path = EDITS_PATH) -> list[dict]:
    """The committed edit manifest, checked."""
    with open(path, encoding="utf-8") as f:
        items = [json.loads(raw) for raw in f if raw.strip()]
    check_manifest(items)
    return items


# --- the blind check ---------------------------------------------------------------------

BLIND_CHECK_STRATUM = "synthetic_edit:blind_check"
#: Items per operation in the blind check: about 30 in all.
BLIND_CHECK_PER_OPERATION = 10


def blind_check(
    items: list[dict], per_operation: int = BLIND_CHECK_PER_OPERATION
) -> list[TruthSite]:
    """The blind-check worklist: truth-site skeletons of edits and decoys, mixed.

    Per operation, pairs in hash order; each contributes one item, edits and decoys
    alternating, and no two items share a source clip, so the listener never hears both
    versions of one recitation. Only marks the truth-site schema accepts are drawn. A row
    names the opaque output file and the source's reference and carrier, so an edit and
    its decoy would read identically; ``heard`` is ``pending`` until the listener answers.
    Rows are in hash order of their ids.
    """
    eligible = [item for item in items if item["mark"] in MARKS]
    by_pair: dict[str, dict[str, dict]] = {}
    for item in eligible:
        by_pair.setdefault(item["pair_id"], {})[item["role"]] = item
    chosen: list[dict] = []
    sources: set[str] = set()
    for operation in OPERATIONS:
        pair_ids = sorted((p for p, roles in by_pair.items()
                           if roles[EDIT]["operation"] == operation),
                          key=lambda p: rank(p, f"{SALT}:blind_check"))
        taken = 0
        for pair_id in pair_ids:
            if taken == per_operation:
                break
            item = by_pair[pair_id][EDIT if taken % 2 == 0 else DECOY]
            if item["source"]["audio_filename"] in sources:
                continue
            sources.add(item["source"]["audio_filename"])
            chosen.append(item)
            taken += 1
    sites = [
        TruthSite(
            site_id=f"synthetic_edit:{opaque_id(item['item_id'])}",
            source=SYNTHETIC_EDIT,
            assumes_competent_reciter=False,
            audio_filename=item["output"]["audio_filename"],
            shard=item["source"]["shard"],
            start_sample=0,
            end_sample=item["output"]["num_samples"],
            audio_sha256=item["output"]["audio_sha256"],
            surah_ayah=item["source"]["surah_ayah"],
            reference=item["reference"],
            reference_index=item["reference_index"],
            mark=item["mark"],
            prescribed=item["prescribed"],
            heard=PENDING,
            stratum=BLIND_CHECK_STRATUM,
            stratum_population=len(eligible),
        )
        for item in chosen
    ]
    return sorted(sites, key=lambda s: rank(s.site_id, SALT))


# --- generation ------------------------------------------------------------------------------

#: Pairs per operation and mark, and per reciter within one.
QUOTA_PER_MARK = 60
PER_RECITER = 3


def write_item(samples: np.ndarray, path: Path) -> dict:
    """Write one rendered item as 16 kHz PCM_16 and return its ``output`` record. Fails on
    any sample beyond :data:`PEAK_LIMIT` rather than let the encoder clip it silently."""
    import soundfile as sf

    from .truth_sites import audio_sha256

    peak = float(np.max(np.abs(samples)))
    if peak > PEAK_LIMIT:
        raise ValueError(f"{path.name} peaks at {peak:.3f}; PCM_16 would clip it")
    sf.write(path, samples, 16000, subtype="PCM_16")
    return {"audio_filename": path.name, "num_samples": sf.info(path).frames,
            "audio_sha256": audio_sha256(path), "peak": round(peak, 6)}


def _generate(args) -> None:
    """Plan, render and check every pair; write the manifest, the summary and the
    blind-check worklist. The audio goes to ``--out-dir/audio`` and stays there."""
    import soundfile as sf

    from .staged_audio import verify_staged
    from .truth_sites import write_truth_sites

    clips = edit_clips()
    timed = timed_clips()
    candidates, stretch_samples = plan_pairs(timed, SALT)
    cache: dict[str, np.ndarray] = {}

    def audio(name: str) -> np.ndarray:
        if name not in cache:
            verify_staged(clips[name], args.audio_dir)
            cache[name] = sf.read(args.audio_dir / name, dtype="float32")[0]
        return cache[name]

    rendered: dict[str, dict[str, RenderedItem]] = {}
    rejected: Counter = Counter()

    def accept(pair: EditPair) -> bool:
        try:
            rendered[pair.pair_id] = render_pair(pair, audio)
        except Unusable as refusal:
            rejected[f"{pair.operation} {pair.mark} {refusal.reason}"] += 1
            return False
        return True

    pairs = select_pairs(candidates, QUOTA_PER_MARK, PER_RECITER, SALT, accept)
    out_dir = args.out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for pair in pairs:
        for role, item in rendered[pair.pair_id].items():
            item_id = f"{pair.pair_id}:{role}"
            output = write_item(item.rendered.samples, out_dir / output_filename(item_id))
            items.append(item_record(pair, role, clips, item, output))
    check_manifest(items)
    check_disjoint(items)
    worklist = blind_check(items)
    with open(EDITS_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    write_truth_sites(worklist, WORKLIST_PATH)

    record = json.loads(BASE_FRAMES_PATH.read_text(encoding="utf-8"))
    by_output = {i["output"]["audio_filename"]: i for i in items}
    summary = {
        "salt": SALT,
        "decode_fingerprint": record["decode_fingerprint"],
        "frame_clips_timed": len(timed),
        "frame_clips_without_reference": len(record["clips"]) - len(timed),
        "stretch_samples": stretch_samples,
        "fade_samples": FADE,
        "quota_per_mark": QUOTA_PER_MARK,
        "per_reciter": PER_RECITER,
        "max_donor_level_ratio": MAX_DONOR_LEVEL_RATIO,
        "candidates": dict(sorted(Counter(f"{p.operation} {p.mark}" for p in candidates).items())),
        "rejected": dict(sorted(rejected.items())),
        "pairs": dict(sorted(Counter(f"{p.operation} {p.mark}" for p in pairs).items())),
        "render_paths": dict(sorted(Counter(
            f"{i['operation']} {i['role']} {i['render']['path']}" for i in items).items())),
        "pair_reciters": len({p.reciter_id for p in pairs}),
        "items": len(items),
        "max_peak": max(i["output"]["peak"] for i in items),
        "output_seconds": round(sum(i["output"]["num_samples"] for i in items) / 16000, 1),
        "blind_check": dict(sorted(Counter(
            f"{by_output[s.audio_filename]['operation']} {by_output[s.audio_filename]['role']}"
            for s in worklist).items())),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
                            + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# --- what the teacher hears ------------------------------------------------------------------


def teacher_agrees(item: dict, decode: str) -> bool:
    """Whether the base teacher's decode of an item spells its label at the carrier: every
    position of the label is anchored against the item's labelled reference."""
    anchors = anchored_positions(item["labelled_reference"], decode)
    return all(k in anchors for k in carrier_positions(item["reference_index"], item["label"]))


def _audit(args) -> None:
    """Decode every rendered item with the base teacher and record whether it hears the
    label (``teacher_check.json``). A pre-screen beside the blind check, never a
    substitute for it: the teacher is not truth, and the blind check is drawn
    independently of this."""
    import soundfile as sf

    from training.decoding import SPANS, Decoder

    from .resegment import BASE_TEACHER, DECODE_BATCH_SIZE, WEIGHTS_DTYPE
    from .truth_sites import audio_sha256

    items = read_items()
    decoder = Decoder.load(
        BASE_TEACHER, args.device, weights_dtype=WEIGHTS_DTYPE, batch_size=DECODE_BATCH_SIZE
    )
    agrees: dict[str, bool] = {}
    for item in items:
        path = args.out_dir / "audio" / item["output"]["audio_filename"]
        if audio_sha256(path) != item["output"]["audio_sha256"]:
            raise SystemExit(f"{path} does not match the manifest's checksum")
        (decode,) = decoder.decode_spans([sf.read(path, dtype="float32")[0]])
        agrees[item["item_id"]] = teacher_agrees(item, decode)
    cells: dict[str, list[bool]] = {}
    for item in items:
        cells.setdefault(f"{item['operation']} {item['mark']} {item['role']}", []).append(
            agrees[item["item_id"]])
    record = {
        "decode_fingerprint": decoder.fingerprint(SPANS).as_dict(),
        "agrees_by_cell": {k: {"items": len(v), "agree": sum(v)} for k, v in sorted(cells.items())},
        "agrees": dict(sorted(agrees.items())),
    }
    TEACHER_CHECK_PATH.write_text(json.dumps(record, indent=1, ensure_ascii=False, sort_keys=True)
                                  + "\n", encoding="utf-8")
    print(json.dumps(record["agrees_by_cell"], indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    frame = commands.add_parser("frame", help="write the edit frame (frame.json)")
    frame.add_argument("--unseen-index", type=Path, required=True,
                       help="shard index of every h448-unseen shard (0-20 and the reserve)")
    frame.add_argument("--train-index", type=Path, required=True,
                       help="shard index of every h448-trained shard")
    frame.add_argument("--evalset-manifest", type=Path, required=True)
    stage = commands.add_parser("stage", help="stage the frame's clips and register them")
    stage.add_argument("--train-index", type=Path, required=True)
    stage.add_argument("--audio-dir", type=Path, required=True)
    stage.add_argument("--shard-cache", type=Path, required=True)
    stage.add_argument("--workers", type=int, default=4,
                       help="shards read at once (each holds one 2.4 GB shard on disk)")
    decode = commands.add_parser("decode", help="time the staged clips with the base teacher")
    decode.add_argument("--audio-dir", type=Path, required=True)
    decode.add_argument("--device", default="cuda")
    generate = commands.add_parser("generate", help="render the edits, manifest and worklist")
    generate.add_argument("--audio-dir", type=Path, required=True)
    generate.add_argument("--out-dir", type=Path, required=True)
    audit = commands.add_parser("audit", help="record whether the teacher hears each label")
    audit.add_argument("--out-dir", type=Path, required=True)
    audit.add_argument("--device", default="cuda")
    args = parser.parse_args()
    commands_by_name = {"frame": _frame, "stage": _stage, "decode": _decode,
                        "generate": _generate, "audit": _audit}
    commands_by_name[args.command](args)


if __name__ == "__main__":
    main()

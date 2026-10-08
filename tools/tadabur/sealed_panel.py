"""The sealed held-out panel: fresh reciters from shards ``h448`` never trained on (#89).

The old ``decode_evalset`` test half is spent (ADR-0010), and rebuilding it gives the same
reciters, because ``reciter_split`` is a stable hash of the reciter id. The final one-time
claim (#97, acceptance rules §3) needs a panel nothing has touched. This module fixes it,
writes its committed manifest, and **seals** it.

**The frame.** Every row of the shards ``h448`` never trained on (:func:`unseen_shards`: the
held-out block 0-20 and the strided reserve 39, 58, ..., 381) whose canonical reciter has
**no use in the exposure registry** (:mod:`tadabur.exposure`) besides ``h448``'s own
streaming training: no truth site, no human label, no mining-pool clip, no
``decode_evalset`` record (dev, test or the legacy stratified sample), no Muraja corpus clip,
no synthetic-edit source or donor. "A new salt" over used reciters would not make them
fresh (§6), so used reciters leave whole. ``h448.training`` is the one use allowed: it
touched 667 of Tadabur's 671 reciters, so no panel without it could exist, and the panel
takes no recording from its shards (``check_disjoint`` by source holds). The rows left are
then held to the mining pool's bounds: 1.5-50 s, and an ayah the phonetizer can realize.

**The panel is the whole frame**, a census of the fresh reciters' eligible clips. No cap
and no draw: the listening budget is sized later (#105), and a site worklist mined from it
then records its own inclusion probabilities against ``frame.json``.

**What is committed** (``sealed_panel/``): the staging registry of its clips
(``staged_clips.jsonl``, :class:`tadabur.staged_audio.StagedClip` rows with use
``sealed_panel``, kept apart from ``staged_audio/clips.jsonl`` so no tool that walks the
shared registry ever reaches a panel clip), the segmented manifest (``clips.jsonl``, the
mining pool's shape: segments, sample spans, realized references, word times), the base
teacher's decode of every kept segment with its fingerprint (``teacher_decodes.json``),
the frame, and a model-independent capacity count (``summary.json``). Its exposure rows
are ``exposure/sealed_panel.jsonl``.

**The seal.** Nothing scores the panel until #97, with the owner's authorization, and then
once. :func:`open_for_scoring` is the only way to the panel's audio for scoring, and it
raises :class:`SealedPanelError` unless it is passed :data:`SHIP_CRITERION_AUTHORIZATION`;
a test fails if any module but this one names that constant or this module. The only decode
of the panel that exists is the base teacher's, made by :mod:`tadabur.resegment` for
segmentation, the same cache the mining pool carries. No count here compares it with
anything: the capacity is read from the references alone.

Usage (from ``tools/``; ``stage`` downloads each needed shard once, ~2.4 GB, then deletes it)::

  python -m tadabur.sealed_panel select --index stage/full_index.jsonl --out stage/panel.jsonl
  python -m tadabur.sealed_panel stage --index stage/full_index.jsonl --selection stage/panel.jsonl \\
      --audio-dir stage/panel_clips --shard-cache stage/hf_cache
  python -m tadabur.resegment --registry tadabur/sealed_panel/staged_clips.jsonl \\
      --use sealed_panel --audio-dir stage/panel_clips --out-dir stage/seg_panel
  python -m tadabur.sealed_panel build --selection stage/panel.jsonl --seg-dir stage/seg_panel
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .exposure import (
    H448_TRAINING,
    Exposure,
    ExposureRegistry,
    load_registry,
    write_use,
)
from .mining_pool import (
    HELD_OUT_BLOCK,
    TARGET_PAIRS,
    PoolSegment,
    read_selection,
    reference_counts,
    segment_key,
)
from .staged_audio import (
    SEALED_PANEL,
    IndexRow,
    StagedClip,
    load_staged_clips,
    read_shard_index,
    verify_staged,
    write_staged_clips,
)

PANEL_DIR = Path(__file__).parent / "sealed_panel"
STAGED_PATH = PANEL_DIR / "staged_clips.jsonl"
FRAME_PATH = PANEL_DIR / "frame.json"
CLIPS_PATH = PANEL_DIR / "clips.jsonl"
DECODES_PATH = PANEL_DIR / "teacher_decodes.json"
SUMMARY_PATH = PANEL_DIR / "summary.json"

#: The one registry use a panel reciter may also have (module docstring).
ALLOWED_RECITER_OVERLAP = frozenset({H448_TRAINING})

#: The explicit flag that unseals the panel. Only the ship criterion (#97) may pass it,
#: once, with the owner's authorization (acceptance rules §3); a test pins every module
#: that names it.
SHIP_CRITERION_AUTHORIZATION = (
    "issue-97: the owner authorized the one-time score of the sealed panel"
)


class SealedPanelError(RuntimeError):
    """Something tried to score the sealed panel without the ship criterion's flag."""


# --- the frame -------------------------------------------------------------------------


def unseen_shards() -> list[int]:
    """The shards ``h448`` never trained on: the held-out block and the strided reserve."""
    from training.decode_evalset import gate_eval_shards

    return sorted(set(HELD_OUT_BLOCK) | set(gate_eval_shards()))


def exposed_reciters(registry: ExposureRegistry) -> dict[int, list[str]]:
    """Every reciter with a use that bars it from the panel, and those uses."""
    barred: dict[int, list[str]] = {}
    for use in registry.uses():
        if use in ALLOWED_RECITER_OVERLAP or use == SEALED_PANEL:
            continue
        for reciter in registry.reciters(use):
            barred.setdefault(reciter, []).append(use)
    return barred


def panel_frame(
    rows: Iterable[IndexRow], barred: Mapping[int, list[str]]
) -> tuple[list[IndexRow], list[tuple[IndexRow, str]]]:
    """The panel, and every other row of the unseen shards with why it was left out
    (``reciter_exposed`` before the row-level ``duration`` and ``phonetizer_unsupported``)."""
    import generate_phonemes
    from training.decode_evalset import MAX_CLIP_SECONDS, MIN_CLIP_SECONDS

    shards = set(unseen_shards())
    kept: list[IndexRow] = []
    excluded: list[tuple[IndexRow, str]] = []
    for row in sorted(rows, key=lambda r: r.audio_filename):
        if row.shard not in shards:
            continue
        if row.reciter_id in barred:
            excluded.append((row, "reciter_exposed"))
        elif not MIN_CLIP_SECONDS <= row.duration_s <= MAX_CLIP_SECONDS:
            excluded.append((row, "duration"))
        elif row.surah_ayah in generate_phonemes.FALLBACK_PHONEMES:
            excluded.append((row, "phonetizer_unsupported"))
        else:
            kept.append(row)
    return kept, excluded


def frame_record(
    panel: list[IndexRow],
    excluded: list[tuple[IndexRow, str]],
    barred: Mapping[int, list[str]],
    registry: ExposureRegistry,
) -> dict:
    """What the panel was drawn from: rows per shard by outcome, the unseen-shard reciters
    each use barred, and per panel reciter its clips and its rows in ``h448``'s training
    shards (the overlap the panel allows)."""
    shards: dict[int, Counter] = {shard: Counter() for shard in unseen_shards()}
    for row in panel:
        shards[row.shard]["panel"] += 1
    for row, reason in excluded:
        shards[row.shard][f"excluded_{reason}"] += 1
    unseen_reciters = {row.reciter_id for row in panel} | {row.reciter_id for row, _ in excluded}
    by_use = Counter(use for r in unseen_reciters & set(barred) for use in barred[r])
    training = registry.shard_uses[H448_TRAINING].rows_per_reciter
    clips = Counter(row.reciter_id for row in panel)
    return {
        "shards": unseen_shards(),
        "per_shard": {str(k): dict(sorted(v.items())) for k, v in sorted(shards.items())},
        "unseen_reciters": len(unseen_reciters),
        "unseen_reciters_barred_by_use": dict(sorted(by_use.items())),
        "unseen_reciters_barred": len(unseen_reciters & set(barred)),
        "panel_reciters": len(clips),
        "per_reciter": {
            str(r): {"clips": n, "h448_training_rows": training.get(r, 0)}
            for r, n in sorted(clips.items())
        },
        "registry_sha256": registry_fingerprint(),
    }


def registry_fingerprint(directory: Path | None = None) -> str:
    """SHA-256 over the exposure registry's files (name and bytes), the panel's own
    excepted: what the frame's exclusions are a function of."""
    from .exposure import EXPOSURE_DIR

    digest = hashlib.sha256()
    for path in sorted((directory or EXPOSURE_DIR).iterdir()):
        if path.name.startswith(f"{SEALED_PANEL}.") or path.name == "README.md":
            continue
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


# --- the committed manifest ------------------------------------------------------------


@dataclass(frozen=True)
class PanelClip:
    """One panel clip as segmented: the :class:`tadabur.clip_status.ClipStatus` fields
    (word times included) and every segment, in the mining pool's segment shape.
    Provenance (shard, row, length, checksum) is its row in ``staged_clips.jsonl``."""

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    n_words: int
    skip_reason: str | None
    re_reads: int
    recited_words: int | None
    recitation_start_s: float
    recitation_end_s: float
    word_times: tuple[float, ...]
    segments: tuple[PoolSegment, ...]


def build_manifest(
    statuses: list[dict], segmentation: list[dict], staged: Mapping[str, StagedClip]
) -> list[PanelClip]:
    """The panel manifest from :mod:`tadabur.resegment`'s ``clip_status`` and
    ``segmentation`` rows. Every staged panel clip must have been segmented; spans are
    converted to samples by the rule that sliced them for decoding."""
    from .segment_score import segment_sample_bounds

    by_name = {status["audio_filename"]: status for status in statuses}
    if set(by_name) != set(staged):
        raise ValueError("the segmentation and the panel's staged clips name different clips")
    segments_of = {row["audio_filename"]: row["segments"] for row in segmentation}
    clips = []
    for name in sorted(by_name):
        status, clip = by_name[name], staged[name]
        segments = []
        for seg in segments_of.get(name, []):
            start, end = segment_sample_bounds(clip.num_samples, seg["start_s"], seg["end_s"])
            segments.append(PoolSegment(
                segment_index=seg["segment_index"], word_start=seg["word_start"],
                word_end=seg["word_end"], start_sample=start, end_sample=end,
                reference=seg["reference"], raw_word_offsets=tuple(seg["raw_word_offsets"]),
                kept=seg["kept"],
            ))
        clips.append(PanelClip(
            audio_filename=name, surah_ayah=status["surah_ayah"],
            reciter_id=status["reciter_id"], n_words=status["n_words"],
            skip_reason=status["skip_reason"], re_reads=status["re_reads"],
            recited_words=status["recited_words"],
            recitation_start_s=status["recitation_start_s"],
            recitation_end_s=status["recitation_end_s"],
            word_times=tuple(status["word_times"]), segments=tuple(segments),
        ))
    return clips


def write_manifest(clips: list[PanelClip], path: Path = CLIPS_PATH) -> None:
    """One clip per line, sorted by filename, keys sorted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for clip in sorted(clips, key=lambda c: c.audio_filename):
            f.write(json.dumps(asdict(clip), ensure_ascii=False, sort_keys=True) + "\n")


def load_manifest(
    path: Path = CLIPS_PATH, staged: Mapping[str, StagedClip] | None = None
) -> list[PanelClip]:
    """The committed panel, checked against its staging registry: the same clips, each with
    the registry's reciter and ayah, and every segment span inside the clip.

    For the registry, mining a site worklist and listening. **Not for scoring**: a model
    decode of the panel goes through :func:`open_for_scoring`.
    """
    staged = load_staged_clips(STAGED_PATH) if staged is None else staged
    clips = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            data["word_times"] = tuple(data["word_times"])
            data["segments"] = tuple(
                PoolSegment(**{**seg, "raw_word_offsets": tuple(seg["raw_word_offsets"])})
                for seg in data["segments"]
            )
            clip = PanelClip(**data)
            entry, where = staged.get(clip.audio_filename), f"{path}:{lineno}"
            if entry is None or SEALED_PANEL not in entry.uses:
                raise ValueError(f"{where}: {clip.audio_filename} is not staged for the panel")
            if (entry.reciter_id, entry.surah_ayah) != (clip.reciter_id, clip.surah_ayah):
                raise ValueError(f"{where}: reciter or ayah disagrees with the staging registry")
            for seg in clip.segments:
                if not 0 <= seg.start_sample <= seg.end_sample <= entry.num_samples:
                    raise ValueError(f"{where}: segment {seg.segment_index} span is outside")
            clips.append(clip)
    if {c.audio_filename for c in clips} != set(staged):
        raise ValueError(f"{path} and the panel's staging registry name different clips")
    return clips


def load_teacher_decodes(path: Path = DECODES_PATH) -> tuple[dict, dict[str, str]]:
    """The base teacher's decode of every kept panel segment, keyed by
    :func:`tadabur.mining_pool.segment_key`, and its decode fingerprint (as a dict)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["decode_fingerprint"], data["decodes"]


def panel_exposures(staged: Mapping[str, StagedClip]) -> list[Exposure]:
    """The panel's rows in the exposure registry: each whole clip, with its checksum."""
    return [
        Exposure(c.audio_filename, c.shard, c.row_index, c.reciter_id, None, None, c.audio_sha256)
        for c in staged.values()
    ]


def reference_capacity(clips: list[PanelClip]) -> dict:
    """The candidate sites the kept segments' realized references hold, per listening
    stratum (:func:`tadabur.mining_pool.reference_counts`), read from the references
    alone. No decode enters it, so it sizes mining without scoring anything."""
    total: Counter = Counter()
    for clip in clips:
        for seg in clip.segments:
            if seg.kept:
                total += reference_counts(seg.reference)
    return {
        "haraka_carriers": {name: total[name] for name in ("damma", "fatha", "kasra")},
        "reference_geminates": total["reference_geminates"],
        "sukun_mid_word_carriers": total["sukun_mid_word_carriers"],
        "pair_carriers": {pair: total[pair] for pair in TARGET_PAIRS},
    }


def summarize(clips: list[PanelClip], staged: Mapping[str, StagedClip], run: dict) -> dict:
    """Counts per shard and reciter, hours, segmentation outcomes and the capacity."""
    entries = [staged[c.audio_filename] for c in clips]
    segments = [seg for c in clips for seg in c.segments]
    per_reciter = Counter(c.reciter_id for c in clips)
    kept = [c for c in clips if any(seg.kept for seg in c.segments)]
    return {
        "clips": len(clips),
        "reciters": len(per_reciter),
        "clips_with_a_kept_segment": len(kept),
        "reciters_with_a_kept_segment": len({c.reciter_id for c in kept}),
        "audio_hours": round(sum(e.num_samples for e in entries) / 16000 / 3600, 2),
        "kept_segment_hours": round(
            sum(s.end_sample - s.start_sample for s in segments if s.kept) / 16000 / 3600, 2
        ),
        "clips_per_shard": {
            str(k): v for k, v in sorted(Counter(e.shard for e in entries).items())
        },
        "clips_per_reciter_histogram": {
            str(k): v for k, v in sorted(Counter(per_reciter.values()).items())
        },
        "clip_skip_reasons": dict(sorted(Counter(c.skip_reason or "none" for c in clips).items())),
        "segments": len(segments),
        "segments_kept": sum(seg.kept for seg in segments),
        "decode_fingerprint": run["decode_fingerprint"],
        "phonetizer_revision": run["phonetizer_revision"],
        "vad": run["vad"],
        "reference_capacity": reference_capacity(clips),
    }


# --- the seal --------------------------------------------------------------------------


def open_for_scoring(audio_dir: Path, *, authorization: str | None = None) -> list[PanelClip]:
    """The panel, with every staged clip under ``audio_dir`` verified, **for scoring**.

    The only sanctioned way to score the panel, and it is sealed: unless ``authorization``
    is :data:`SHIP_CRITERION_AUTHORIZATION`, which only the ship criterion (#97) passes,
    once and with the owner's authorization, it raises :class:`SealedPanelError` before
    reading anything.
    """
    if authorization != SHIP_CRITERION_AUTHORIZATION:
        raise SealedPanelError(
            "the sealed held-out panel (#89) is scored only by the ship criterion (#97), "
            "once, with the owner's authorization (acceptance rules §3). Nothing else may "
            "decode or score it; select and tune on the mining pool or decode_evalset dev."
        )
    staged = load_staged_clips(STAGED_PATH)
    for clip in staged.values():
        verify_staged(clip, audio_dir)
    return load_manifest(staged=staged)


# --- the CLI ---------------------------------------------------------------------------


def _select(index_path: Path, out: Path) -> None:
    registry = load_registry()
    barred = exposed_reciters(registry)
    panel, excluded = panel_frame(read_shard_index(index_path).values(), barred)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in panel:
            f.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    frame = frame_record(panel, excluded, barred, registry)
    FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRAME_PATH.write_text(json.dumps(frame, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in frame.items() if k != "per_reciter"}, indent=1))


def _stage(index_path: Path, selection: Path, audio_dir: Path, shard_cache: Path) -> None:
    from .shard_reader import iter_shard_rows
    from .staged_audio import stage_clips

    requests = {row.audio_filename: frozenset({SEALED_PANEL}) for row in read_selection(selection)}
    existing = load_staged_clips(STAGED_PATH) if STAGED_PATH.exists() else {}

    def shard_rows(shard: int) -> Iterator[dict]:
        return iter_shard_rows([shard], cache_dir=shard_cache, delete_after=True,
                               columns=["audio", "reciter_id"])

    def checkpoint(staged: dict[str, StagedClip]) -> None:
        write_staged_clips(staged.values(), STAGED_PATH)
        print(f"  {len(staged)} clips staged so far", flush=True)

    staged, unlocatable = stage_clips(
        requests, read_shard_index(index_path), audio_dir, shard_rows, existing,
        on_shard_done=checkpoint,
    )
    if unlocatable:
        raise SystemExit(f"{len(unlocatable)} panel clips are in no indexed shard")
    write_staged_clips(staged.values(), STAGED_PATH)
    print(f"Staged {len(staged)} panel clips into {audio_dir}; registry {STAGED_PATH}")


def _build(selection: Path, seg_dir: Path) -> None:
    def rows(path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(raw) for raw in f if raw.strip()]

    staged = load_staged_clips(STAGED_PATH)
    if set(staged) != {row.audio_filename for row in read_selection(selection)}:
        raise SystemExit("the staged panel is not the selection; re-run stage")
    run = json.loads((seg_dir / "run.json").read_text(encoding="utf-8"))
    if run["use"] != SEALED_PANEL:
        raise SystemExit(f"{seg_dir} segmented use {run['use']!r}, not the panel")
    clips = build_manifest(
        rows(seg_dir / "clip_status.jsonl"), rows(seg_dir / "segmentation.jsonl"), staged
    )
    decodes = {
        segment_key(row["clip_audio_filename"], row["segment_index"]): row["predicted_phonemes"]
        for row in rows(seg_dir / "segment_manifest.jsonl")
    }
    write_manifest(clips)
    DECODES_PATH.write_text(
        json.dumps({"decode_fingerprint": run["decode_fingerprint"], "decodes": decodes},
                   ensure_ascii=False, indent=0, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    load_manifest(staged=staged)  # what was written must load against the registry
    write_use(SEALED_PANEL, panel_exposures(staged))
    summary = summarize(clips, staged, run)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="fix the panel's frame from a full shard index")
    select.add_argument("--index", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    stage = commands.add_parser("stage", help="stage the selected clips with provenance")
    stage.add_argument("--index", type=Path, required=True)
    stage.add_argument("--selection", type=Path, required=True)
    stage.add_argument("--audio-dir", type=Path, required=True)
    stage.add_argument("--shard-cache", type=Path, required=True,
                       help="a cache directory of this run's own; shards are deleted once read")
    build = commands.add_parser("build", help="write the committed manifest and its exposure")
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--seg-dir", type=Path, required=True,
                       help="output of `tadabur.resegment --use sealed_panel`")
    args = parser.parse_args()

    if args.command == "select":
        _select(args.index, args.out)
    elif args.command == "stage":
        _stage(args.index, args.selection, args.audio_dir, args.shard_cache)
    else:
        _build(args.selection, args.seg_dir)


if __name__ == "__main__":
    main()

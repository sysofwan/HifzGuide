"""The sealed held-out panel: fresh reciters from shards ``h448`` never trained on (#89).

The old ``decode_evalset`` test half is spent (ADR-0010), and rebuilding it gives the same
reciters, because ``reciter_split`` is a stable hash of the reciter id. The final one-time
claim (#97, acceptance rules §3) needs a panel nothing in this PRD has touched. This module
fixes it, writes its committed manifest, and **seals** it.

**What it certifies** (acceptance rules §6, owner amendment 2026-10-08): the **paired**
#97 ship criterion. The panel is reciter-disjoint from everything this PRD tunes, trains on
or selects with. Toward ``h448``'s *original* training and initialisation it is held out by
recording only: those exposures (the registry's shared-baseline uses) are shared by the
baseline and by every candidate warm-started from ``h448``. It does **not** certify
absolute accuracy on unseen reciters.

**The frame.** Every row of the shards ``h448`` never trained on (:func:`unseen_shards`: the
held-out block 0-20 and the strided reserve 39, 58, ..., 381) whose canonical reciter has
no use in the exposure registry (:mod:`tadabur.exposure`) other than a shared-baseline one:
no truth site, human label, mining-pool clip, ``decode_evalset`` record (dev, test or the
legacy stratified sample), Muraja corpus clip, synthetic edit, probe, bias or shaddah-probe
use. "A new salt" over used reciters would not make them fresh (§6), so used reciters leave
whole. Then the mining pool's row bounds: 1.5-50 s, and an ayah the phonetizer can realize.
Then the recording itself (#117): a row whose **probable copy** (the registry's screen,
:func:`tadabur.exposure.copy_groups`) is a recording of a use the panel must be
source-disjoint from, an ``h448.training`` shard row included, leaves; and of rows that
are one recording (a probable copy, or one checksum among staged panel clips) only the
first by file name stays. **No model's decode decides eligibility**, and the panel is the
whole frame: no cap, no draw, so a site worklist mined later records its inclusion
probabilities against ``frame.json``.

**Preparation, not scoring.** :func:`segment_panel` runs the recitation VAD and today's
pause-to-word placement (:func:`tadabur.segment_score.segment_clips`, whose whole-clip
decode by the base teacher places pauses on words) and decodes **every** segment with a
reference once with the base teacher, for the decode cache. No gate, no ``match_ratio``,
no contrast attribution and no decode-dependent drop: every VAD segment with a reference
is kept. It runs inside :func:`tadabur.panel_seal.unsealed` with the preparation
authorization, which only this module names.

**What is committed** (``sealed_panel/``): ``staged_clips.jsonl`` (provenance, the
:class:`tadabur.staged_audio.StagedClip` schema with use ``sealed_panel``, never in the
shared registry), ``frame.json``, ``clips.jsonl`` (segments, sample spans, realized
references, word times), ``teacher_decodes.json`` (the base teacher's decode of every
segment, with its fingerprint) and ``summary.json`` (a reference-only capacity). Its
exposure rows are ``exposure/sealed_panel.jsonl``.

**The seal** is :mod:`tadabur.panel_seal`, enforced where audio enters the tools.
:func:`open_for_scoring` is the one way to score the panel, for #97 only.

Usage (from ``tools/``; ``stage`` downloads each needed shard once, ~2.4 GB, then deletes it)::

  python -m tadabur.sealed_panel select --index stage/full_index.jsonl --out stage/panel.jsonl
  python -m tadabur.sealed_panel stage --index stage/full_index.jsonl --selection stage/panel.jsonl \\
      --audio-dir stage/panel_clips --shard-cache stage/hf_cache
  python -m tadabur.sealed_panel segment --audio-dir stage/panel_clips --out-dir stage/seg_panel
  python -m tadabur.sealed_panel build --selection stage/panel.jsonl --seg-dir stage/seg_panel
  python -m tadabur.sealed_panel prune --selection stage/panel.jsonl  # drop clips, no audio
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .exposure import (
    EXPOSURE_DIR,
    PROBABLE_COPIES_NAME,
    SOURCES_DIRNAME,
    Exposure,
    ExposureRegistry,
    load_registry,
    read_probable_copies,
    require_complete,
    write_use,
)
from .mining_pool import (
    HELD_OUT_BLOCK,
    TARGET_PAIRS,
    read_selection,
    reference_counts,
    segment_key,
)
from .panel_seal import (
    PANEL_REGISTRY_PATH,
    PREPARATION_AUTHORIZATION,
    SealedPanelError,
    unsealed,
)
from .staged_audio import (
    SEALED_PANEL,
    IndexRow,
    StagedClip,
    read_shard_index,
    read_staged_clips,
    verify_staged,
    write_staged_clips,
)

PANEL_DIR = PANEL_REGISTRY_PATH.parent
STAGED_PATH = PANEL_REGISTRY_PATH
FRAME_PATH = PANEL_DIR / "frame.json"
CLIPS_PATH = PANEL_DIR / "clips.jsonl"
DECODES_PATH = PANEL_DIR / "teacher_decodes.json"
SUMMARY_PATH = PANEL_DIR / "summary.json"


# --- the frame -------------------------------------------------------------------------


def unseen_shards() -> list[int]:
    """The shards ``h448`` never trained on: the held-out block and the strided reserve."""
    from training.decode_evalset import gate_eval_shards

    return sorted(set(HELD_OUT_BLOCK) | set(gate_eval_shards()))


def reciter_overlap_allowed(registry: ExposureRegistry) -> list[str]:
    """The uses a panel reciter may also have: the shared-baseline exposures only."""
    return registry.shared_baseline_uses()


def source_overlap_allowed(registry: ExposureRegistry) -> list[str]:
    """The uses a panel recording may also be in: shared-baseline exposures whose
    membership is uncertain (every row of their shards counts as possibly exposed)."""
    return [u for u in registry.shared_baseline_uses()
            if registry.shard_uses[u].membership == "uncertain"]


def exposed_reciters(registry: ExposureRegistry) -> dict[int, list[str]]:
    """Every reciter with a use that bars it from the panel, and those uses."""
    allowed = set(reciter_overlap_allowed(registry)) | {SEALED_PANEL}
    barred: dict[int, list[str]] = {}
    for use in registry.uses():
        if use not in allowed:
            for reciter in registry.reciters(use):
                barred.setdefault(reciter, []).append(use)
    return barred


def copy_exposed(registry: ExposureRegistry) -> frozenset[tuple[int, int]]:
    """The rows (``(shard, row_index)``) with a probable copy the panel must be
    source-disjoint from: a recording another use lists, or a row of a shard use whose
    membership is exact (``h448.training``; an uncertain one may overlap, §6)."""
    allowed = set(source_overlap_allowed(registry))
    shards = {shard for use, e in registry.shard_uses.items() if use not in allowed
              for shard in e.shards}
    listed = {r.source for use, rows in registry.recordings.items() if use != SEALED_PANEL
              for r in rows}
    return frozenset(source for source, copies in registry.copies.items()
                     if any(c in listed or c[0] in shards for c in copies))


def panel_frame(
    rows: Iterable[IndexRow],
    barred: Mapping[int, list[str]],
    exposed: frozenset[tuple[int, int]] = frozenset(),
    recording: Callable[[IndexRow], set] = lambda row: {(row.shard, row.row_index)},
) -> tuple[list[IndexRow], list[tuple[IndexRow, str]]]:
    """The panel, and every other row of the unseen shards with why it was left out:
    ``reciter_exposed``, then the row-level ``duration`` and ``phonetizer_unsupported``,
    then the recording-level ``probable_copy_exposed`` (its source in ``exposed``) and
    ``duplicate_recording`` (``recording(row)``, the keys that identify its recording,
    meets a row kept before it by file name)."""
    import generate_phonemes
    from training.decode_evalset import MAX_CLIP_SECONDS, MIN_CLIP_SECONDS

    shards = set(unseen_shards())
    kept: list[IndexRow] = []
    excluded: list[tuple[IndexRow, str]] = []
    seen: set = set()
    for row in sorted(rows, key=lambda r: r.audio_filename):
        if row.shard not in shards:
            continue
        if row.reciter_id in barred:
            excluded.append((row, "reciter_exposed"))
        elif not MIN_CLIP_SECONDS <= row.duration_s <= MAX_CLIP_SECONDS:
            excluded.append((row, "duration"))
        elif row.surah_ayah in generate_phonemes.FALLBACK_PHONEMES:
            excluded.append((row, "phonetizer_unsupported"))
        elif (row.shard, row.row_index) in exposed:
            excluded.append((row, "probable_copy_exposed"))
        elif recording(row) & seen:
            excluded.append((row, "duplicate_recording"))
        else:
            seen |= recording(row)
            kept.append(row)
    return kept, excluded


def frame_record(
    panel: list[IndexRow],
    excluded: list[tuple[IndexRow, str]],
    barred: Mapping[int, list[str]],
    registry: ExposureRegistry,
) -> dict:
    """What the panel was drawn from: rows per shard by outcome, the unseen-shard reciters
    each use barred, and per panel reciter its clips and its rows in each shared-baseline
    use's shards (the overlap the paired claim allows)."""
    shards: dict[int, Counter] = {shard: Counter() for shard in unseen_shards()}
    for row in panel:
        shards[row.shard]["panel"] += 1
    for row, reason in excluded:
        shards[row.shard][f"excluded_{reason}"] += 1
    unseen_reciters = {row.reciter_id for row in panel} | {row.reciter_id for row, _ in excluded}
    by_use = Counter(use for r in unseen_reciters & set(barred) for use in barred[r])
    baselines = {
        use: registry.shard_uses[use].rows_per_reciter for use in reciter_overlap_allowed(registry)
    }
    clips = Counter(row.reciter_id for row in panel)
    return {
        "shards": unseen_shards(),
        "per_shard": {str(k): dict(sorted(v.items())) for k, v in sorted(shards.items())},
        "unseen_reciters": len(unseen_reciters),
        "unseen_reciters_barred_by_use": dict(sorted(by_use.items())),
        "unseen_reciters_barred": len(unseen_reciters & set(barred)),
        "panel_reciters": len(clips),
        "per_reciter": {
            str(r): {"clips": n, **{f"{u}_rows": rows.get(r, 0) for u, rows in baselines.items()}}
            for r, n in sorted(clips.items())
        },
        "registry_sha256": registry_fingerprint(),
    }


def registry_fingerprint(directory: Path = EXPOSURE_DIR) -> str:
    """SHA-256 over the exposure registry's use files (name and bytes), the panel's own
    excepted, and the probable-copy groups' rows (not the uses they are annotated with,
    which name the panel): what the frame's exclusions are a function of."""
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        if path.name.startswith(f"{SEALED_PANEL}.") or path.name in ("README.md", SOURCES_DIRNAME):
            continue
        data = path.read_bytes()
        if path.name == PROBABLE_COPIES_NAME:
            data = json.dumps([[g["screen"], g["rows"]] for g in read_probable_copies(path)],
                              sort_keys=True).encode("utf-8")
        digest.update(path.name.encode("utf-8") + b"\0" + data + b"\0")
    return digest.hexdigest()


# --- the committed manifest ------------------------------------------------------------


@dataclass(frozen=True)
class PanelSegment:
    """One waqf segment of a panel clip: its word range, its exact sample span in the
    staged clip, and its realized reference with per-word offsets into it. Every
    segment with a reference is kept; nothing here depends on a decode's outcome."""

    segment_index: int
    word_start: int
    word_end: int
    start_sample: int
    end_sample: int
    reference: str
    raw_word_offsets: tuple[int, ...]


@dataclass(frozen=True)
class PanelClip:
    """One panel clip as segmented: the :class:`tadabur.clip_status.ClipStatus` fields
    (word times included) and every segment. Provenance (shard, row, length, checksum)
    is its row in ``staged_clips.jsonl``."""

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
    segments: tuple[PanelSegment, ...]


def load_panel_registry(path: Path = STAGED_PATH) -> dict[str, StagedClip]:
    """The panel's staging registry, every row staged for the panel alone. The shared
    loader (:func:`tadabur.staged_audio.load_staged_clips`) refuses it by design."""
    clips = read_staged_clips(path)
    if any(c.uses != (SEALED_PANEL,) for c in clips.values()):
        raise ValueError(f"{path}: every row must be staged for {SEALED_PANEL} alone")
    return clips


def build_manifest(
    statuses: list[dict], segmentation: list[dict], staged: Mapping[str, StagedClip]
) -> list[PanelClip]:
    """The panel manifest from :func:`segment_panel`'s ``clip_status`` and
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
            segments.append(PanelSegment(
                segment_index=seg["segment_index"], word_start=seg["word_start"],
                word_end=seg["word_end"], start_sample=start, end_sample=end,
                reference=seg["reference"], raw_word_offsets=tuple(seg["raw_word_offsets"]),
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
    """The committed panel (metadata only, no audio), checked against its staging
    registry: the same clips, each with the registry's reciter and ayah, and every
    segment span inside the clip. For the registry, mining a site worklist and listening;
    a model decode of the panel goes through :func:`open_for_scoring`."""
    staged = load_panel_registry() if staged is None else staged
    clips = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            data["word_times"] = tuple(data["word_times"])
            data["segments"] = tuple(
                PanelSegment(**{**seg, "raw_word_offsets": tuple(seg["raw_word_offsets"])})
                for seg in data["segments"]
            )
            clip = PanelClip(**data)
            entry, where = staged.get(clip.audio_filename), f"{path}:{lineno}"
            if entry is None:
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


def decodable_segment_keys(clips: list[PanelClip]) -> set[str]:
    """The segments long enough to featurize, the ones the decode cache must hold."""
    from .segment_score import MIN_DECODE_SAMPLES

    return {
        segment_key(c.audio_filename, s.segment_index)
        for c in clips for s in c.segments if s.end_sample - s.start_sample >= MIN_DECODE_SAMPLES
    }


def load_teacher_decodes(path: Path = DECODES_PATH) -> tuple[dict, dict[str, str]]:
    """The base teacher's decode of every decodable panel segment, keyed by
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
    """The candidate sites the segments' realized references hold, per listening stratum
    (:func:`tadabur.mining_pool.reference_counts`), read from the references alone. No
    decode enters it, so it sizes mining without scoring anything."""
    total: Counter = Counter()
    for clip in clips:
        for seg in clip.segments:
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
    segmented = [c for c in clips if c.segments]
    return {
        "clips": len(clips),
        "reciters": len(per_reciter),
        "clips_with_segments": len(segmented),
        "reciters_with_segments": len({c.reciter_id for c in segmented}),
        "audio_hours": round(sum(e.num_samples for e in entries) / 16000 / 3600, 2),
        "segment_hours": round(
            sum(s.end_sample - s.start_sample for s in segments) / 16000 / 3600, 2
        ),
        "clips_per_shard": {
            str(k): v for k, v in sorted(Counter(e.shard for e in entries).items())
        },
        "clips_per_reciter_histogram": {
            str(k): v for k, v in sorted(Counter(per_reciter.values()).items())
        },
        "clip_skip_reasons": dict(sorted(Counter(c.skip_reason or "none" for c in clips).items())),
        "segments": len(segments),
        "preparation": run,
        "reference_capacity": reference_capacity(clips),
    }


# --- preparation: segmentation and the teacher decode cache, never a score ------------


def segment_panel(audio_dir: Path, device: str, vad_dtype: str) -> tuple[list, list, dict, dict]:
    """Segment every panel clip and decode every segment once with the base teacher.

    Returns ``clip_status`` rows, ``segmentation`` rows (every segment with a reference),
    the decode cache keyed by :func:`tadabur.mining_pool.segment_key`, and the run record.
    The teacher's whole-clip decode only places pauses on words, and the decode cache is
    compared with nothing. Runs unsealed for preparation only.
    """
    import hafs_phonetizer
    import torch

    from training.decoding import SPANS, Decoder

    from . import vad
    from .resegment import BASE_TEACHER, DECODE_BATCH_SIZE, WEIGHTS_DTYPE, DecoderSegmentationModel
    from .segment_score import MIN_DECODE_SAMPLES, _load_clip, segment_clips, slice_segment
    from .waqf_segments import hafs_segment_reference, hafs_word_reference

    with unsealed(PREPARATION_AUTHORIZATION):
        clips = sorted(load_panel_registry().values(), key=lambda c: c.audio_filename)
        for clip in clips:
            verify_staged(clip, audio_dir)
        pauses = vad.compute_clip_pauses(
            clips, audio_dir, device=torch.device(device), dtype=getattr(torch, vad_dtype)
        )
        if len(pauses) != len(clips):
            raise SystemExit(f"VAD saw {len(pauses)} of {len(clips)} clips: audio is missing")
        decoder = Decoder.load(
            BASE_TEACHER, device, weights_dtype=WEIGHTS_DTYPE, batch_size=DECODE_BATCH_SIZE
        )
        segments, skips, statuses, _ = segment_clips(
            clips, audio_dir, DecoderSegmentationModel(decoder),
            hafs_segment_reference(), hafs_word_reference(), pauses,
        )
        by_clip: dict[str, list] = {}
        for seg in segments:
            by_clip.setdefault(seg.audio_filename, []).append(seg)
        decodes: dict[str, str] = {}
        undecodable = 0
        for name, clip_segments in sorted(by_clip.items()):
            waveform = _load_clip(audio_dir, name)
            spans = {s.segment_index: slice_segment(waveform, s.start_s, s.end_s)
                     for s in clip_segments}
            # Too short to featurize (a structural fact of the span, not a decode outcome):
            # kept in the manifest, absent from the cache.
            decodable = {i: w for i, w in spans.items() if len(w) >= MIN_DECODE_SAMPLES}
            undecodable += len(spans) - len(decodable)
            for index, text in zip(decodable, decoder.decode_spans(decodable.values()),
                                   strict=True):
                decodes[segment_key(name, index)] = text
    segmentation = [
        {"audio_filename": name, "segments": [
            {"segment_index": s.segment_index, "word_start": s.word_start,
             "word_end": s.word_end, "start_s": s.start_s, "end_s": s.end_s,
             "reference": s.realized_reference_phonemes,
             "raw_word_offsets": list(s.word_offsets)}
            for s in sorted(segs, key=lambda s: s.segment_index)
        ]}
        for name, segs in sorted(by_clip.items())
    ]
    run = {
        "clips": len(clips),
        "decode_fingerprint": decoder.fingerprint(SPANS).as_dict(),
        "vad": {"model": vad.VAD_MODEL_ID, "dtype": vad_dtype,
                "min_silence_ms": vad.DEFAULT_MIN_SILENCE_MS,
                "min_speech_ms": vad.DEFAULT_MIN_SPEECH_MS, "pad_ms": vad.DEFAULT_PAD_MS},
        "segmentation_skips": dict(sorted(skips.items())),
        "segments_too_short_to_decode": undecodable,
        "phonetizer_revision": hafs_phonetizer.REVISION,
        "scoring": "none: no gate, match_ratio, contrast or decode-dependent drop",
    }
    return [asdict(s) for s in statuses], segmentation, decodes, run


# --- the seal --------------------------------------------------------------------------


@contextlib.contextmanager
def open_for_scoring(
    audio_dir: Path, *, authorization: str | None = None
) -> Iterator[list[PanelClip]]:
    """The panel for scoring, its audio verified and the seal lifted inside the block.

    Only the ship criterion (#97) may enter it, once, with the owner's authorization
    (acceptance rules §3), passing ``tadabur.panel_seal``'s ship-criterion flag. Any other
    ``authorization``, the preparation one included, raises :class:`SealedPanelError`
    before anything is read.
    """
    if authorization == PREPARATION_AUTHORIZATION:
        raise SealedPanelError("the preparation authorization cannot score the panel")
    with unsealed(authorization):
        staged = load_panel_registry()
        for clip in staged.values():
            verify_staged(clip, audio_dir)
        yield load_manifest(staged=staged)


# --- the CLI ---------------------------------------------------------------------------


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(raw) for raw in f if raw.strip()]


def _select(index_path: Path, out: Path) -> None:
    registry = load_registry()
    require_complete(registry)
    barred = exposed_reciters(registry)
    # Checksums are known for clips already staged for the panel (a rebuild), so a copy the
    # name screen misses (filed under another ayah) is still one recording.
    checksums = ({n: c.audio_sha256 for n, c in load_panel_registry().items()}
                 if STAGED_PATH.exists() else {})

    def recording(row: IndexRow) -> set:
        keys: set = set(registry.same_recording((row.shard, row.row_index)))
        if row.audio_filename in checksums:
            keys.add(checksums[row.audio_filename])
        return keys

    panel, excluded = panel_frame(read_shard_index(index_path).values(), barred,
                                  copy_exposed(registry), recording)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in panel:
            f.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    frame = frame_record(panel, excluded, barred, registry)
    FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRAME_PATH.write_text(json.dumps(frame, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in frame.items() if k not in ("per_reciter", "per_shard")},
                     indent=1))


def _stage(index_path: Path, selection: Path, audio_dir: Path, shard_cache: Path) -> None:
    from .shard_reader import iter_shard_rows
    from .staged_audio import stage_clips

    requests = {row.audio_filename: frozenset({SEALED_PANEL}) for row in read_selection(selection)}

    def shard_rows(shard: int) -> Iterator[dict]:
        return iter_shard_rows([shard], cache_dir=shard_cache, delete_after=True,
                               columns=["audio", "reciter_id"])

    def checkpoint(staged: dict[str, StagedClip]) -> None:
        write_staged_clips(staged.values(), STAGED_PATH)
        print(f"  {len(staged)} clips staged so far", flush=True)

    with unsealed(PREPARATION_AUTHORIZATION):
        existing = load_panel_registry() if STAGED_PATH.exists() else {}
        staged, unlocatable = stage_clips(
            requests, read_shard_index(index_path), audio_dir, shard_rows, existing,
            on_shard_done=checkpoint,
        )
    if unlocatable:
        raise SystemExit(f"{len(unlocatable)} panel clips are in no indexed shard")
    write_staged_clips(staged.values(), STAGED_PATH)
    print(f"Staged {len(staged)} panel clips into {audio_dir}; registry {STAGED_PATH}")


def _segment(audio_dir: Path, out_dir: Path, device: str, vad_dtype: str) -> None:
    statuses, segmentation, decodes, run = segment_panel(audio_dir, device, vad_dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("clip_status.jsonl", statuses), ("segmentation.jsonl", segmentation)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "decodes.json").write_text(
        json.dumps(decodes, ensure_ascii=False, indent=0, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run, indent=2, sort_keys=True))


def _build(selection: Path, seg_dir: Path) -> None:
    staged = load_panel_registry()
    if set(staged) != {row.audio_filename for row in read_selection(selection)}:
        raise SystemExit("the staged panel is not the selection; re-run stage")
    run = json.loads((seg_dir / "run.json").read_text(encoding="utf-8"))
    clips = build_manifest(
        _rows(seg_dir / "clip_status.jsonl"), _rows(seg_dir / "segmentation.jsonl"), staged
    )
    decodes = json.loads((seg_dir / "decodes.json").read_text(encoding="utf-8"))
    if set(decodes) != decodable_segment_keys(clips):
        raise SystemExit("the decode cache does not cover exactly the panel's decodable segments")
    _write_panel(staged, clips, decodes, run)


def _write_panel(
    staged: Mapping[str, StagedClip], clips: list[PanelClip], decodes: dict[str, str], run: dict
) -> None:
    """The committed manifest, decode cache, exposure rows and summary of ``staged``."""
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


def _prune(selection: Path) -> None:
    """Drop the panel clips a new ``select`` left out, and their segments and decodes,
    from every committed panel file. Metadata only: no audio is read or staged, and the
    preparation of the clips kept (segmentation, teacher decodes) is unchanged."""
    keep = {row.audio_filename for row in read_selection(selection)}
    staged = load_panel_registry()
    if not keep <= set(staged):
        raise SystemExit("the selection holds clips the panel never staged; run stage")
    clips = [c for c in load_manifest(staged=staged) if c.audio_filename in keep]
    fingerprint, decodes = load_teacher_decodes()
    run = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["preparation"]
    if run["decode_fingerprint"] != fingerprint:
        raise SystemExit("the decode cache and the preparation record disagree")
    staged = {name: clip for name, clip in staged.items() if name in keep}
    write_staged_clips(staged.values(), STAGED_PATH)
    kept = decodable_segment_keys(clips)
    _write_panel(staged, clips, {k: v for k, v in decodes.items() if k in kept}, run)


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
    segment = commands.add_parser("segment", help="segment and teacher-decode (no scoring)")
    segment.add_argument("--audio-dir", type=Path, required=True)
    segment.add_argument("--out-dir", type=Path, required=True)
    segment.add_argument("--device", default="cuda")
    segment.add_argument("--vad-dtype", default="bfloat16")
    build = commands.add_parser("build", help="write the committed manifest and its exposure")
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--seg-dir", type=Path, required=True)
    prune = commands.add_parser(
        "prune", help="drop the clips a new select left out from the committed panel")
    prune.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "select":
        _select(args.index, args.out)
    elif args.command == "stage":
        _stage(args.index, args.selection, args.audio_dir, args.shard_cache)
    elif args.command == "segment":
        _segment(args.audio_dir, args.out_dir, args.device, args.vad_dtype)
    elif args.command == "prune":
        _prune(args.selection)
    else:
        _build(args.selection, args.seg_dir)


if __name__ == "__main__":
    main()

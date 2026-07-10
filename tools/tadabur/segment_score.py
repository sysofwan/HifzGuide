"""Per-segment decode + score pass over waqf segments, for the P3.5 audit (#6).

The waqf stage (:mod:`tadabur.waqf_segments`) splits each admitted Tadabur clip
into **waqf segments** — contiguous word ranges bounded by the reciter's intra-ayah
pauses — and labels each with its *realized* reference (terminal word in waqf form,
interior words in wasl). Those segments, not the whole ayah, are the units that go
to the fine-tune (#8), so the human poison audit (#6) must grade *them*.

But a :class:`~tadabur.waqf_segments.SegmentRecord` is a training label only: it
carries the realized reference and an audio ``(start_s, end_s)`` offset, **not** a
model decode, a ``match_ratio``, or the contrasts the audit samples on. This module
supplies them. For each segment it slices the whole-clip 16 kHz waveform to the
segment span, decodes it with the Muaalem phoneme head, and scores that decode
against the segment's *realized* reference with the ported ``.balanced`` gate — the
same normalization / Smith-Waterman / contrast attribution the full-ayah filter
uses, only per segment.

Unlike the filter it does **not** gate-reject: every segment of an admitted clip is
already in the training set, so all segments are emitted. ``match_ratio`` and
``contrasts`` are observational here — they only drive the audit's per-contrast and
marginal-band sampling.

The output is a single segment manifest (JSONL). Its :class:`ManifestRecord` fields
(``audio_filename`` = the per-segment id, ``surah_ayah``, ``match_ratio``,
``ayah_duration_s`` = the segment's own duration, ``reciter_id``, ``contrasts``,
``predicted_phonemes``) let :mod:`tadabur.audit_sampler` draw the worklist directly
via :func:`~tadabur.manifest.read_records`, which ignores the extra per-segment
display fields (``uthmani``, ``raw_reference_phonemes``, ``reference_phonemes``,
``start_s`` / ``end_s`` / ``segment_index``) that :mod:`tadabur.audit_ui` reads to
show the reviewer the segment's realized reference rather than the full ayah's.
Each segment's sliced audio is written to ``--audio-out`` under the same
:func:`~tadabur.audit_sampler.local_audio_path` name the sampler/UI expect, so no
separate HF export step is needed.

Usage:
  python -m tadabur.segment_score \
      --segments segments.jsonl --clips-dir clips/ \
      --out-manifest segment_manifest.jsonl --audio-out segment_audio/ \
      [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import TARGET_SAMPLE_RATE
from .audit_sampler import local_audio_path
from .contrast_attribution import all_contrasts
from .manifest import ManifestRecord
from .normalization import normalize_phonemes
from .scorer import BALANCED_SCORER
from .waqf_segments import (
    SegmentRecord,
    _uthmani_words,
    read_segment_manifest,
)

DEFAULT_BATCH_SIZE = 16


def segment_id(record: SegmentRecord) -> str:
    """A stable, unique per-segment audio id derived from the clip + index.

    ``audio_filename`` alone repeats across a clip's segments; suffixing the
    ``segment_index`` makes the id unique so the sampler can key, and the UI can
    play, each segment independently. Kept ``.wav``-suffixed and human-legible;
    :func:`~tadabur.audit_sampler.local_audio_path` hashes it for the on-disk name.
    """
    stem = Path(record.audio_filename).stem
    return f"{stem}__seg{record.segment_index}.wav"


def slice_segment(
    clip_waveform: np.ndarray, start_s: float, end_s: float
) -> np.ndarray:
    """The ``[start_s, end_s)`` span of a 16 kHz mono clip waveform.

    Sample bounds are clamped to the waveform so a segment whose alignment end
    slightly overshoots the decoded audio still yields a valid (non-empty) slice.
    """
    n = len(clip_waveform)
    start = max(0, min(n, int(round(start_s * TARGET_SAMPLE_RATE))))
    end = max(start, min(n, int(round(end_s * TARGET_SAMPLE_RATE))))
    return np.ascontiguousarray(clip_waveform[start:end], dtype=np.float32)


def _load_clip(clips_dir: Path, audio_filename: str) -> np.ndarray:
    """Read a whole 16 kHz mono clip WAV (written by ``waqf_segments``) as float32."""
    waveform, sample_rate = sf.read(clips_dir / audio_filename, dtype="float32")
    if waveform.ndim > 1:  # defensive: collapse any stray channel dim to mono
        waveform = waveform.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"{audio_filename} is {sample_rate} Hz, expected {TARGET_SAMPLE_RATE} Hz; "
            "clips must be the 16 kHz mono WAVs written by tadabur.waqf_segments."
        )
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _uthmani_segment_text(surah_ayah: str, word_start: int, word_end: int) -> str:
    """The space-joined Uthmani words a segment covers (for the UI display)."""
    words = _uthmani_words(surah_ayah)
    return " ".join(words[word_start:word_end])


def _decode_all(
    model, waveforms: list[np.ndarray], batch_size: int
) -> list[str]:
    """Greedy-CTC-decode every segment waveform, in fixed-size batches, in order."""
    phonemes: list[str] = []
    for start in range(0, len(waveforms), batch_size):
        batch = waveforms[start : start + batch_size]
        for decode in model.decode_batch(batch, TARGET_SAMPLE_RATE):
            phonemes.append(decode.phonemes)
    return phonemes


def score_segments(
    segments: list[SegmentRecord],
    clips_dir: Path,
    model,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Decode and score every segment, returning one manifest row (dict) per segment.

    Segments are processed in a stable ``(audio_filename, segment_index)`` order so
    the manifest is deterministic. Each row carries the :class:`ManifestRecord`
    fields the sampler needs plus the per-segment display fields the UI reads; the
    realized reference is normalized **once** here (the gate/attribution require a
    pre-normalized reference — normalization is not idempotent).
    """
    ordered = sorted(segments, key=lambda s: (s.audio_filename, s.segment_index))

    clip_cache: dict[str, np.ndarray] = {}
    waveforms: list[np.ndarray] = []
    for seg in ordered:
        if seg.audio_filename not in clip_cache:
            clip_cache[seg.audio_filename] = _load_clip(clips_dir, seg.audio_filename)
        waveforms.append(
            slice_segment(clip_cache[seg.audio_filename], seg.start_s, seg.end_s)
        )

    predicted = _decode_all(model, waveforms, batch_size)

    rows: list[dict] = []
    for seg, decode in zip(ordered, predicted):
        reference = normalize_phonemes(seg.realized_reference_phonemes).normalized
        result = BALANCED_SCORER.gate(decode, reference)
        record = ManifestRecord(
            audio_filename=segment_id(seg),
            surah_ayah=seg.surah_ayah,
            match_ratio=result.match_ratio,
            ayah_duration_s=round(seg.end_s - seg.start_s, 3),
            reciter_id=seg.reciter_id,
            contrasts=BALANCED_SCORER.attribute(decode, reference),
            predicted_phonemes=decode,
        )
        row = asdict(record)
        row["contrasts"] = list(record.contrasts)
        row["uthmani"] = _uthmani_segment_text(
            seg.surah_ayah, seg.word_start, seg.word_end
        )
        row["raw_reference_phonemes"] = seg.realized_reference_phonemes
        row["reference_phonemes"] = reference
        row["start_s"] = seg.start_s
        row["end_s"] = seg.end_s
        row["segment_index"] = seg.segment_index
        rows.append(row)
    return rows


def write_segment_manifest(path: Path, rows: list[dict]) -> None:
    """Write scored segment rows as a deterministic, key-sorted JSONL manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stage_segment_audio(
    segments: list[SegmentRecord],
    clips_dir: Path,
    audio_out: Path,
) -> None:
    """Slice each segment's audio and write it under its ``local_audio_path`` name.

    Pre-stages the exact files the sampler's worklist and the UI's ``/audio``
    route reference (keyed on the per-segment id), so the audit needs no separate
    HF audio-export step — the whole clips are already on local disk.
    """
    audio_out.mkdir(parents=True, exist_ok=True)
    clip_cache: dict[str, np.ndarray] = {}
    for seg in segments:
        if seg.audio_filename not in clip_cache:
            clip_cache[seg.audio_filename] = _load_clip(clips_dir, seg.audio_filename)
        waveform = slice_segment(
            clip_cache[seg.audio_filename], seg.start_s, seg.end_s
        )
        out_name = local_audio_path(segment_id(seg))
        sf.write(audio_out / out_name, waveform, TARGET_SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--segments", type=Path, required=True,
        help="Waqf offsets manifest from tadabur.waqf_segments (JSONL).",
    )
    parser.add_argument(
        "--clips-dir", type=Path, required=True,
        help="Directory of whole 16 kHz mono clip WAVs (waqf_segments --audio-dir).",
    )
    parser.add_argument(
        "--out-manifest", type=Path, required=True,
        help="Output scored segment manifest (JSONL) for the sampler + UI.",
    )
    parser.add_argument(
        "--audio-out", type=Path, required=True,
        help="Directory to write each segment's sliced audio for listening.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Segments per decode batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device", default="cuda", help="Torch device for the model (default: cuda).",
    )
    args = parser.parse_args()

    segments = read_segment_manifest(args.segments)

    from .inference import MuaalemPhonemeModel

    model = MuaalemPhonemeModel.load(device=args.device)
    rows = score_segments(segments, args.clips_dir, model, args.batch_size)
    write_segment_manifest(args.out_manifest, rows)
    stage_segment_audio(segments, args.clips_dir, args.audio_out)

    contrasted = sum(1 for r in rows if r["contrasts"])
    buckets = {c: sum(1 for r in rows if c in r["contrasts"]) for c in all_contrasts()}
    print(
        f"Scored {len(rows)} segments from {len(segments)} waqf segments "
        f"({contrasted} with contrasts). Wrote manifest to {args.out_manifest} and "
        f"segment audio to {args.audio_out}."
    )
    print("Per-contrast segment counts: " + ", ".join(
        f"{c}={n}" for c, n in buckets.items() if n
    ))


if __name__ == "__main__":
    main()

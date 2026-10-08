"""Re-derive the waqf segments of re-staged clips and decode them with the base teacher (#83).

The labels' segment boundaries were lost with ``audit_run/``, so the clips they sit on are
segmented again by **today's** segmentation (:mod:`tadabur.segment_score`): the recitation
VAD's pauses (:mod:`tadabur.vad`), placed on words by
:func:`tadabur.waqf_detect.segment_clip` over a whole-clip decode, then each segment
decoded whole and passed through the same drop rules. Two things differ from
``segment_score.main``, and both are deliberate:

* **Every decode goes through** :class:`training.decoding.Decoder` (the base teacher, bf16
  weights, batch size 1), and its :class:`~training.decoding.DecodeFingerprint` is written
  beside the outputs, so the decodes these artifacts carry are comparable with every later
  one. ``segment_score`` drives a ``decode`` / ``decode_batch`` duck type;
  :class:`DecoderSegmentationModel` serves it from the decoder.
* **References are built after #100.** Both references a segment needs (the whole-ayah
  alignment reference and each segment's realized reference) come from
  :mod:`tadabur.waqf_segments`, which phonetizes through ``hafs_phonetizer.phonetize``, so a
  segment that ends on ``ةً`` is realized ``ه``, not quran-transcript 0.5.2's ``تَاا``.
  Segment manifests built on the GPU box before that fix must not be reused.

Audio is read back from the staged PCM_16 WAV, and every clip's checksum and length are
verified against the registry first, so every decode is of the file the registry records.

Outputs in ``--out-dir``, the first three in ``segment_score``'s own formats so the tools
downstream of it (``training.windowed_labels``, ``training.tashkeel_worklist``) run on
them unchanged:

* ``segment_manifest.jsonl`` -- one row per **kept** segment, carrying its base decode
  (``predicted_phonemes``) and realized reference;
* ``clip_status.jsonl`` and ``pause_attrib.jsonl`` -- per-clip status with word times, and
  each VAD pause's word;
* ``segmentation.jsonl`` -- per clip, **every** segment the segmenter produced (kept or
  not) with its span, realized reference and word offsets, and the clip's drop tally, so a
  label on a segment that is now dropped is reported with the reason;
* ``run.json`` -- the decode fingerprint, VAD settings and the clips processed.

Usage (Linux + CUDA, from ``tools/``)::

  python -m tadabur.resegment --registry stage/staged_clips.jsonl --use p35_fixture \\
      --audio-dir stage/clips --out-dir stage/seg_p35
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.distill_student import TEACHER_MODEL_ID

from .audio import TARGET_SAMPLE_RATE
from .clip_status import write_clip_status
from .segment_score import (
    DEFAULT_BATCH_SIZE,
    score_segments,
    segment_clips,
    write_pause_attributions,
    write_segment_manifest,
)
from .staged_audio import USES, StagedClip, load_staged_clips, verify_staged
from .waqf_segments import SegmentRecord, hafs_segment_reference, hafs_word_reference

#: The model every segmentation decode is made with: the frozen base teacher, in the bf16
#: weights it has always been decoded at, one span per forward pass so no padding or batch
#: composition enters a decode.
BASE_TEACHER = TEACHER_MODEL_ID
WEIGHTS_DTYPE = "bf16"
DECODE_BATCH_SIZE = 1


@dataclass(frozen=True)
class _Decode:
    phonemes: str
    class_ids: tuple[int, ...]


class DecoderSegmentationModel:
    """The ``decode`` / ``decode_batch`` interface :mod:`tadabur.segment_score` drives,
    served by a :class:`training.decoding.Decoder` so its numerics are fingerprinted."""

    def __init__(self, decoder) -> None:
        self.decoder = decoder

    def decode(self, waveform: np.ndarray, sample_rate: int) -> _Decode:
        """One whole-clip decode with its per-frame class ids (for word placement)."""
        from training.decoding import scan_ctc, tokens_to_phonemes

        _require_rate(sample_rate)
        (row,) = self.decoder.span_class_ids([waveform])
        return _Decode(
            tokens_to_phonemes(seg.token_id for seg in scan_ctc(row)),
            tuple(int(i) for i in row),
        )

    def decode_batch(self, waveforms: list[np.ndarray], sample_rate: int) -> list[_Decode]:
        """Each segment decoded whole (:meth:`~training.decoding.Decoder.decode_spans`)."""
        _require_rate(sample_rate)
        return [_Decode(text, ()) for text in self.decoder.decode_spans(waveforms)]


def _require_rate(sample_rate: int) -> None:
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"expected {TARGET_SAMPLE_RATE} Hz audio, got {sample_rate} Hz")


def segmentation_rows(
    segments: list[SegmentRecord], kept_rows: list[dict], drops_by_clip: dict[str, Counter]
) -> list[dict]:
    """One ``segmentation.jsonl`` row per clip: every segment, whether it was kept, and
    the clip's drop tally (the reasons its dropped segments fell to)."""
    kept = {(r["clip_audio_filename"], r["segment_index"]) for r in kept_rows}
    by_clip: dict[str, list[SegmentRecord]] = {}
    for seg in segments:
        by_clip.setdefault(seg.audio_filename, []).append(seg)
    return [
        {
            "audio_filename": clip,
            "drops": dict(sorted(drops_by_clip.get(clip, Counter()).items())),
            "segments": [
                {
                    "segment_index": seg.segment_index,
                    "word_start": seg.word_start,
                    "word_end": seg.word_end,
                    "start_s": seg.start_s,
                    "end_s": seg.end_s,
                    "reference": seg.realized_reference_phonemes,
                    "raw_word_offsets": list(seg.word_offsets),
                    "kept": (clip, seg.segment_index) in kept,
                }
                for seg in sorted(by_clip[clip], key=lambda s: s.segment_index)
            ],
        }
        for clip in sorted(by_clip)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--registry", type=Path, required=True,
                        help="staged-clip registry (tadabur.staged_audio)")
    parser.add_argument("--use", choices=sorted(USES), required=True,
                        help="segment the clips staged for this use")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vad-dtype", default="bfloat16")
    parser.add_argument("--limit", type=int, default=0, help="first N clips only (smoke)")
    args = parser.parse_args()

    import hafs_phonetizer
    import torch

    from training.decoding import SPANS, Decoder

    from . import vad

    clips: list[StagedClip] = sorted(
        (c for c in load_staged_clips(args.registry).values() if args.use in c.uses),
        key=lambda c: c.audio_filename,
    )
    if args.limit:
        clips = clips[: args.limit]
    for clip in clips:  # every decode below must be of the bytes the registry records
        verify_staged(clip, args.audio_dir)
    print(f"{len(clips)} clips staged for {args.use}, checksums verified", flush=True)

    pauses = vad.compute_clip_pauses(
        clips, args.audio_dir, device=torch.device(args.device),
        dtype=getattr(torch, args.vad_dtype),
    )
    if len(pauses) != len(clips):
        raise SystemExit(f"VAD saw {len(pauses)} of {len(clips)} clips: audio is missing")
    print(f"VAD: {sum(map(len, pauses.values()))} pauses", flush=True)

    decoder = Decoder.load(
        BASE_TEACHER, args.device, weights_dtype=WEIGHTS_DTYPE, batch_size=DECODE_BATCH_SIZE
    )
    model = DecoderSegmentationModel(decoder)
    segments, skips, statuses, attributions = segment_clips(
        clips, args.audio_dir, model,
        hafs_segment_reference(),
        hafs_word_reference(),
        pauses,
    )
    print(f"Segmented into {len(segments)} segments; skips {dict(skips)}", flush=True)

    rows: list[dict] = []
    drops_by_clip: dict[str, Counter] = {}
    by_clip: dict[str, list[SegmentRecord]] = {}
    for seg in segments:
        by_clip.setdefault(seg.audio_filename, []).append(seg)
    for clip in sorted(by_clip):
        clip_rows, _, drops = score_segments(
            by_clip[clip], args.audio_dir, model, DEFAULT_BATCH_SIZE
        )
        rows.extend(clip_rows)
        drops_by_clip[clip] = drops
    print(f"Kept {len(rows)} segments; drops {dict(sum(drops_by_clip.values(), Counter()))}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_segment_manifest(out / "segment_manifest.jsonl", rows)
    write_clip_status(out / "clip_status.jsonl", statuses)
    write_pause_attributions(out / "pause_attrib.jsonl", attributions)
    with open(out / "segmentation.jsonl", "w", encoding="utf-8") as f:
        for row in segmentation_rows(segments, rows, drops_by_clip):
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    run = {
        "use": args.use,
        "clips": len(clips),
        "decode_fingerprint": decoder.fingerprint(SPANS).as_dict(),
        "vad": {
            "model": vad.VAD_MODEL_ID,
            "dtype": args.vad_dtype,
            "min_silence_ms": vad.DEFAULT_MIN_SILENCE_MS,
            "min_speech_ms": vad.DEFAULT_MIN_SPEECH_MS,
            "pad_ms": vad.DEFAULT_PAD_MS,
        },
        "segmentation_skips": dict(sorted(skips.items())),
        "phonetizer_revision": hafs_phonetizer.REVISION,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""Waqf-head distillation soft labels: pool the 20 ms VAD teacher to Muaalem's 40 ms.

The waqf head (ADR-0004) is a per-frame silence classifier riding the Muaalem adapter
+ CTC output — the **40 ms** post-downsample lattice — distilled from the Recitation
VAD (``obadx/recitation-segmenter-v2``), whose frame classifier runs at **20 ms**.
Distillation is therefore **2:1**: the teacher's 20 ms silence posteriors are pooled to
the 40 ms grid before the KL. This module owns that pooling and the persisted soft-label
artifact; the torch VAD forward pass lives in :mod:`tadabur.vad`.

The pooling and frame-alignment are **torch-free and deterministic** so they can be
unit-tested (golden fixtures) without a GPU. The pinned rule is:

* **Student frame ``i`` owns teacher frames ``2i`` and ``2i+1``** — a non-overlapping
  pair, left-anchored (frame 0 of both lattices starts at sample 0). Because a 1–2 frame
  drift between the two feature extractors moves a boundary snap across a word edge,
  anchoring the pairing at index 0 keeps every interior boundary on its true timestamp
  and confines the drift to the clip tail, where :func:`pool_silence_2to1` reconciles it.
* **A student frame is silent iff both its teacher frames are** — so the pooled *silence*
  posterior is the **min** of the pair (equivalently, max-pool the speech posterior),
  matching ADR-0004's "a window is silent iff its two teacher frames are".

The artifact is keyed to the passing-subset manifest (``audio_filename``) and is
deterministic and idempotent: a resumed run skips clips already written.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tadabur.manifest import ManifestRecord, read_records

# Muaalem's single stride-2, kernel-3 adapter conv maps the 20 ms encoder lattice to
# the 40 ms CTC lattice (``ml-model-transformation.md``; config ``add_adapter``,
# ``num_adapter_layers=1``, ``adapter_kernel_size=3``, ``adapter_stride=2``). Pinned
# here so the soft labels land on the exact student frames the phoneme head — and the
# CTC target length — use, without loading the model to generate them.
ADAPTER_KERNEL = 3
ADAPTER_STRIDE = 2
ADAPTER_PADDING = ADAPTER_KERNEL // 2

# One 40 ms student frame consumes two 20 ms teacher frames.
TEACHER_FRAMES_PER_STUDENT = 2


def muaalem_lattice_length(feature_frames: int) -> int:
    """40 ms student-lattice length for a 20 ms encoder length ``feature_frames``.

    Mirrors ``Wav2Vec2BertModel._get_feat_extract_output_lengths`` for the single
    stride-2 adapter conv (kernel 3, pad 1): ``floor((T-1)/2) + 1 == ceil(T/2)``. For
    the fixed 5 s export window (T≈250) this is 125, per ADR-0004.
    """
    return (feature_frames + 2 * ADAPTER_PADDING - ADAPTER_KERNEL) // ADAPTER_STRIDE + 1


def _reconcile_teacher_length(silence_20ms: np.ndarray, needed_frames: int) -> np.ndarray:
    """Left-anchor ``silence_20ms`` to exactly ``needed_frames`` teacher frames.

    Extra tail frames are dropped and a short tail is edge-held (repeat the last
    frame), so the ±few-frame feature-extractor drift is absorbed at the clip end —
    never by shifting an interior frame boundary. An empty teacher (no frames) with a
    non-zero requirement is a data-integrity failure, not something to pad from nothing.
    """
    have = len(silence_20ms)
    if have == needed_frames:
        return silence_20ms
    if have > needed_frames:
        return silence_20ms[:needed_frames]
    if have == 0:
        raise ValueError(
            f"cannot reconcile 0 teacher frames up to {needed_frames}: no silence signal"
        )
    return np.concatenate(
        [silence_20ms, np.full(needed_frames - have, silence_20ms[-1], dtype=silence_20ms.dtype)]
    )


def pool_silence_2to1(silence_20ms: np.ndarray, num_student_frames: int) -> np.ndarray:
    """Pool 20 ms teacher silence posteriors to a 40 ms student lattice, 2:1.

    Returns one ``P(silence)`` per 40 ms student frame under the pinned rule (see the
    module docstring): student ``i`` = ``min(teacher[2i], teacher[2i+1])``. The teacher
    is first reconciled to exactly ``2*num_student_frames`` frames
    (:func:`_reconcile_teacher_length`), so a drifted teacher still yields exactly
    ``num_student_frames`` targets aligned to the Muaalem lattice.
    """
    if num_student_frames < 0:
        raise ValueError(f"num_student_frames must be non-negative, got {num_student_frames}")
    needed = TEACHER_FRAMES_PER_STUDENT * num_student_frames
    aligned = _reconcile_teacher_length(np.asarray(silence_20ms, dtype=np.float32), needed)
    pairs = aligned.reshape(num_student_frames, TEACHER_FRAMES_PER_STUDENT)
    return pairs.min(axis=1)


def clip_silence_soft_labels(silence_20ms: np.ndarray) -> np.ndarray:
    """40 ms silence soft targets for a whole clip's 20 ms teacher posteriors.

    The clip's student-lattice length is derived from its own teacher frame count via
    :func:`muaalem_lattice_length` (both lattices share the 20 ms encoder rate), then
    pooled 2:1. A windowed collator (#8) re-slices these 40 ms targets per training
    window, reconciling to the window's exact Muaalem logit length with the same rule.
    """
    return pool_silence_2to1(silence_20ms, muaalem_lattice_length(len(silence_20ms)))


class SoftLabelStore:
    """Deterministic, idempotent on-disk store of per-clip 40 ms silence soft labels.

    Each clip's targets are a ``.npy`` under ``root/silence_40ms/`` and an index line in
    ``root/soft_labels.jsonl`` keyed by ``audio_filename`` (the manifest key). Open with
    :meth:`open`; :meth:`has` reports clips already written so a resumed run skips them,
    and :meth:`write` appends one clip atomically (array then fsynced index line), so an
    interrupted run replays only the last clip and never double-writes an index entry.
    """

    ARRAYS_SUBDIR = "silence_40ms"
    INDEX_NAME = "soft_labels.jsonl"

    def __init__(self, root: Path, index_file, seen: set[str]) -> None:
        self.root = root
        self.arrays_dir = root / self.ARRAYS_SUBDIR
        self.index_path = root / self.INDEX_NAME
        self._index_file = index_file
        self._seen = seen

    @classmethod
    def open(cls, root: Path) -> "SoftLabelStore":
        root = Path(root)
        (root / cls.ARRAYS_SUBDIR).mkdir(parents=True, exist_ok=True)
        index_path = root / cls.INDEX_NAME
        seen = _read_index_keys(index_path)
        index_file = open(index_path, "a", encoding="utf-8")
        return cls(root, index_file, seen)

    def has(self, audio_filename: str) -> bool:
        return audio_filename in self._seen

    def write(self, audio_filename: str, silence_40ms: np.ndarray, num_teacher_frames: int) -> None:
        """Persist one clip's 40 ms silence targets and record it in the index.

        A no-op if ``audio_filename`` is already stored, so replaying an interrupted
        batch adds no duplicate array or index line.
        """
        if audio_filename in self._seen:
            return
        array_name = f"{audio_filename}.npy"
        np.save(self.arrays_dir / array_name, np.asarray(silence_40ms, dtype=np.float32))
        self._seen.add(audio_filename)
        self._index_file.write(
            json.dumps(
                {
                    "audio_filename": audio_filename,
                    "array_path": f"{self.ARRAYS_SUBDIR}/{array_name}",
                    "num_student_frames": int(len(silence_40ms)),
                    "num_teacher_frames": int(num_teacher_frames),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        self._index_file.flush()
        os.fsync(self._index_file.fileno())

    def close(self) -> None:
        self._index_file.close()

    def __enter__(self) -> "SoftLabelStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _read_index_keys(index_path: Path) -> set[str]:
    """Recover the ``audio_filename`` keys already written to ``index_path``."""
    if not index_path.exists():
        return set()
    seen: set[str] = set()
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["audio_filename"])
    return seen


def generate_soft_labels(
    records: list[ManifestRecord],
    clips_dir: Path,
    out_dir: Path,
    *,
    device: str = "cuda",
    dtype_str: str = "bfloat16",
    batch_size: int = 8,
) -> int:
    """Build the waqf soft-label artifact for ``records`` under ``out_dir``.

    Runs the VAD teacher over every clip present under ``clips_dir`` (deterministic
    ``audio_filename`` order), pools each clip's 20 ms silence posteriors to the 40 ms
    Muaalem lattice, and writes them to a :class:`SoftLabelStore`. Idempotent: clips
    already in the store are skipped, so a re-run over the same manifest reproduces the
    identical artifact and a resumed run does not re-infer. Returns the number of clips
    newly written.
    """
    import torch

    from tadabur.vad import compute_clip_silence_posteriors

    ordered = sorted(records, key=lambda r: r.audio_filename)
    with SoftLabelStore.open(out_dir) as store:
        pending = [r for r in ordered if not store.has(r.audio_filename)]
        posteriors = compute_clip_silence_posteriors(
            pending,
            Path(clips_dir),
            device=torch.device(device),
            dtype=getattr(torch, dtype_str),
            batch_size=batch_size,
        )
        written = 0
        for audio_filename in sorted(posteriors):
            silence_20ms = posteriors[audio_filename]
            store.write(
                audio_filename,
                clip_silence_soft_labels(silence_20ms),
                num_teacher_frames=len(silence_20ms),
            )
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="passing-subset JSONL manifest")
    parser.add_argument("--clips-dir", type=Path, required=True, help="staged 16 kHz clips directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="soft-label artifact directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    records = read_records(args.manifest)
    written = generate_soft_labels(
        records,
        args.clips_dir,
        args.out_dir,
        device=args.device,
        dtype_str=args.dtype,
        batch_size=args.batch_size,
    )
    print(f"Wrote {written} new soft-label clips to {args.out_dir} ({len(records)} in manifest).")


if __name__ == "__main__":
    main()

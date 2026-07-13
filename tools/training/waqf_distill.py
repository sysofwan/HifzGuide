"""Waqf-head distillation soft labels: per-window VAD teacher pooled to Muaalem's 40 ms.

The waqf head (ADR-0004) is a per-frame silence classifier riding the Muaalem adapter
+ CTC output — the **40 ms** post-downsample lattice — distilled from the Recitation
VAD (``obadx/recitation-segmenter-v2``), whose frame classifier runs at **20 ms**.
Distillation is therefore **2:1**: the teacher's 20 ms silence posteriors are pooled to
the 40 ms grid before the KL. This module owns that pooling, the per-training-window
frame alignment, and the persisted soft-label artifact; the torch VAD forward pass
lives in :mod:`tadabur.vad`.

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

The deployed model runs **fixed 5 s windows** (250 feature frames → 125 student frames),
not whole clips, so the soft-label artifact is emitted **per training window** and keyed
to the training manifest by ``(audio_filename, window_index)``. A window starting at an
even teacher frame ``start`` begins at student frame ``start // 2``, so its student frame
``j`` owns clip teacher frames ``start + 2j`` / ``start + 2j + 1`` — the pinned mapping is
therefore **independent of how windows are spaced**, which is what lets #23 pin the
drift-critical alignment before the window contract is frozen.

The window *length* (5 s / 250 feature frames) is the already-deployed inference window
(``convert_to_coreml.py``, ``ml-model-transformation.md``). The window **spacing**
(overlap / edge-ownership / stitch) is the frozen inference contract owned by #24 (HITL,
still open); :class:`WindowContract` takes it as a parameter and defaults to a
**provisional non-overlapping tiling** so this artifact can be built now. Because the
store is deterministic and idempotent, it is regenerated cheaply once #24 fixes the
spacing — and the per-window frame alignment above does not change when it does.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
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

# The deployed fixed inference window: 250 feature frames ≈ 5 s at 20 ms
# (``convert_to_coreml.py`` ``FIXED_SEQ_LEN``; ADR-0004). Its 40 ms length is 125.
DEPLOYED_WINDOW_FEATURE_FRAMES = 250


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
    frame), so the ±few-frame feature-extractor drift is absorbed at the window end —
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


@dataclass(frozen=True)
class WindowContract:
    """How the un-waqf-segmented recitation is cut into fixed training windows.

    ``feature_frames`` is the window length on the 20 ms teacher/encoder grid — the
    deployed 5 s inference window (250). ``hop_feature_frames`` is the step between
    consecutive window starts on that grid; the default equals ``feature_frames``, a
    **non-overlapping tiling**. The final overlap / edge-ownership / stitch policy is
    the frozen inference contract owned by #24 (HITL); until it lands this default is
    **provisional**, and because the artifact is deterministic it is regenerated cheaply
    when #24 fixes the spacing.

    Both are required to be **even** so every window starts on an even teacher frame and
    its student frames line up exactly with the clip's 40 ms lattice (``start // 2``);
    an odd start would split a teacher pair across two windows and reintroduce the
    ±1-frame boundary drift the alignment pins down.
    """

    feature_frames: int = DEPLOYED_WINDOW_FEATURE_FRAMES
    hop_feature_frames: int = DEPLOYED_WINDOW_FEATURE_FRAMES

    def __post_init__(self) -> None:
        if self.feature_frames <= 0 or self.feature_frames % 2 != 0:
            raise ValueError(f"feature_frames must be a positive even int, got {self.feature_frames}")
        if self.hop_feature_frames <= 0 or self.hop_feature_frames % 2 != 0:
            raise ValueError(
                f"hop_feature_frames must be a positive even int, got {self.hop_feature_frames}"
            )

    @property
    def student_frames(self) -> int:
        """The full-window 40 ms lattice length (125 for the deployed 5 s window)."""
        return muaalem_lattice_length(self.feature_frames)


@dataclass(frozen=True)
class Window:
    """One fixed training window over a clip's 20 ms teacher lattice.

    ``start_feature_frame`` is the (even) 20 ms start; ``start_student_frame`` is the
    matching 40 ms start (``start_feature_frame // 2``). ``num_feature_frames`` is
    clamped to the teacher frames the clip actually has (a tail window covers fewer than
    a full window), and ``num_student_frames`` is that slice's exact Muaalem 40 ms length
    — so the soft target is reconciled to each window's own logit length.
    """

    index: int
    start_feature_frame: int
    start_student_frame: int
    num_feature_frames: int
    num_student_frames: int


def enumerate_windows(num_teacher_frames: int, contract: WindowContract) -> list[Window]:
    """Fixed training windows tiling ``num_teacher_frames`` under ``contract``.

    Windows start at ``0, hop, 2*hop, …`` while the start is inside the clip; each covers
    up to ``contract.feature_frames`` teacher frames, clamped at the clip end. A clip
    with no teacher frames yields no windows. Every start is even (the contract enforces
    an even hop), so ``start_student_frame == start // 2`` places each window exactly on
    the clip's 40 ms lattice.
    """
    windows: list[Window] = []
    start = 0
    while start < num_teacher_frames:
        num_feat = min(contract.feature_frames, num_teacher_frames - start)
        windows.append(
            Window(
                index=len(windows),
                start_feature_frame=start,
                start_student_frame=start // TEACHER_FRAMES_PER_STUDENT,
                num_feature_frames=num_feat,
                num_student_frames=muaalem_lattice_length(num_feat),
            )
        )
        start += contract.hop_feature_frames
    return windows


def window_silence_soft_labels(silence_20ms: np.ndarray, window: Window) -> np.ndarray:
    """40 ms silence soft targets for one training ``window`` of a clip's teacher posteriors.

    Slices the window's teacher frames and pools them 2:1 to the window's exact Muaalem
    40 ms length (:func:`pool_silence_2to1`). Because the window starts on an even teacher
    frame, student frame ``j`` of the result owns clip teacher frames
    ``window.start_feature_frame + 2j`` / ``+ 2j + 1`` — the pinned, drift-checked mapping.
    """
    stop = window.start_feature_frame + window.num_feature_frames
    teacher_slice = np.asarray(silence_20ms, dtype=np.float32)[window.start_feature_frame : stop]
    return pool_silence_2to1(teacher_slice, window.num_student_frames)


def clip_window_soft_labels(
    silence_20ms: np.ndarray, contract: WindowContract
) -> list[tuple[Window, np.ndarray]]:
    """Every training window of a clip paired with its 40 ms silence soft target.

    Enumerates the clip's windows under ``contract`` and pools each
    (:func:`window_silence_soft_labels`), so a clip's whole per-window artifact is
    produced from a single VAD pass over its 20 ms posteriors.
    """
    windows = enumerate_windows(len(silence_20ms), contract)
    return [(w, window_silence_soft_labels(silence_20ms, w)) for w in windows]


class SoftLabelStore:
    """Deterministic, idempotent on-disk store of per-window 40 ms silence soft labels.

    Each training window's targets are a ``.npy`` under ``root/silence_40ms/`` named
    ``{audio_filename}#w{index}.npy``; each clip contributes exactly one line to
    ``root/soft_labels.jsonl`` listing all of its windows (their 40 ms/20 ms frame
    counts and start frames). Open with :meth:`open`; :meth:`has` reports clips already
    written so a resumed run skips them, and :meth:`write_clip` persists one clip's
    windows **atomically** — every window array first, then the single index line
    fsynced — so an interrupted multi-hour VAD run replays only the in-flight clip and
    never double-writes an index entry. Resume granularity is the clip because a clip's
    VAD forward pass is the expensive, non-resumable unit.
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

    def write_clip(
        self,
        audio_filename: str,
        windows: list[tuple[Window, np.ndarray]],
        num_teacher_frames: int,
    ) -> None:
        """Persist all of one clip's per-window 40 ms targets and index the clip once.

        A no-op if ``audio_filename`` is already stored, so replaying an interrupted
        clip adds no duplicate array or index line. Window arrays are written before the
        index line and the line is fsynced, so a crash before the line leaves the clip
        un-indexed (re-done on resume), never half-indexed.
        """
        if audio_filename in self._seen:
            return
        entries = []
        for window, labels in windows:
            array_name = f"{audio_filename}#w{window.index}.npy"
            np.save(self.arrays_dir / array_name, np.asarray(labels, dtype=np.float32))
            entries.append(
                {
                    "window_index": window.index,
                    "start_feature_frame": window.start_feature_frame,
                    "start_student_frame": window.start_student_frame,
                    "num_feature_frames": window.num_feature_frames,
                    "num_student_frames": int(len(labels)),
                    "array_path": f"{self.ARRAYS_SUBDIR}/{array_name}",
                }
            )
        self._seen.add(audio_filename)
        self._index_file.write(
            json.dumps(
                {
                    "audio_filename": audio_filename,
                    "num_teacher_frames": int(num_teacher_frames),
                    "windows": entries,
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
    contract: WindowContract | None = None,
    device: str = "cuda",
    dtype_str: str = "bfloat16",
    batch_size: int = 8,
) -> int:
    """Build the per-window waqf soft-label artifact for ``records`` under ``out_dir``.

    Streams the VAD teacher over the clips present under ``clips_dir`` one clip at a time
    (:func:`tadabur.vad.iter_clip_silence_posteriors`, deterministic ``audio_filename``
    order), pools each clip's 20 ms silence posteriors into its per-window 40 ms targets
    under ``contract``, and writes + fsyncs each clip to a :class:`SoftLabelStore` before
    the next runs. Clips already in the store are skipped, so the store's ``has`` resume
    contract protects a multi-hour run: a crash keeps every clip already written, and the
    full manifest is never materialized in memory. Returns the number of clips newly
    written.
    """
    import torch

    from tadabur.vad import iter_clip_silence_posteriors

    contract = contract or WindowContract()
    with SoftLabelStore.open(out_dir) as store:
        pending = [r for r in records if not store.has(r.audio_filename)]
        written = 0
        for audio_filename, silence_20ms in iter_clip_silence_posteriors(
            pending,
            Path(clips_dir),
            device=torch.device(device),
            dtype=getattr(torch, dtype_str),
            batch_size=batch_size,
        ):
            store.write_clip(
                audio_filename,
                clip_window_soft_labels(silence_20ms, contract),
                num_teacher_frames=len(silence_20ms),
            )
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="passing-subset JSONL manifest")
    parser.add_argument("--clips-dir", type=Path, required=True, help="staged 16 kHz clips directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="soft-label artifact directory")
    parser.add_argument(
        "--window-feature-frames",
        type=int,
        default=DEPLOYED_WINDOW_FEATURE_FRAMES,
        help="window length on the 20 ms grid (deployed 5 s = 250)",
    )
    parser.add_argument(
        "--hop-feature-frames",
        type=int,
        default=None,
        help="window step on the 20 ms grid; default = window length (non-overlapping, "
        "provisional pending the #24 inference-contract freeze)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    contract = WindowContract(
        feature_frames=args.window_feature_frames,
        hop_feature_frames=args.hop_feature_frames or args.window_feature_frames,
    )
    records = read_records(args.manifest)
    written = generate_soft_labels(
        records,
        args.clips_dir,
        args.out_dir,
        contract=contract,
        device=args.device,
        dtype_str=args.dtype,
        batch_size=args.batch_size,
    )
    print(f"Wrote {written} new soft-label clips to {args.out_dir} ({len(records)} in manifest).")


if __name__ == "__main__":
    main()

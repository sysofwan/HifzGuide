"""Per-clip segmentation status — the eligibility sidecar the windowed labels need.

The segment manifest (:mod:`tadabur.segment_score`) records only the **kept** waqf
segments plus aggregate drop counters. That is not enough for the whole-clip
eligibility ADR-0004 (P7.C, #25) requires: a clip the segmenter could not split safely
is **kept whole** (``repeated_recitation`` / ``low_alignment``) and looks identical to a
legitimate no-interior-pause clip, and a clip whose reference cannot be phonetized
(``phonetizer_unsupported``) or whose audio is missing (``clip_missing``) leaves **no**
manifest rows at all. Either way the manifest alone cannot tell the label builder that
the clip is ineligible.

This module owns the per-clip status record that closes that gap: one
:class:`ClipStatus` per passing clip, written by ``segment_score`` alongside the segment
manifest and read by :mod:`training.windowed_labels`. It is deliberately torch-free (a
plain JSONL of dataclasses, like :mod:`tadabur.manifest`) so the label builder never
pulls in the GPU segmentation stack.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# The two whole-clip skip reasons ``tadabur.waqf_detect.segment_clip`` can return, plus
# the two the ``segment_clips`` driver records for a clip it never scores. All four make
# the whole clip ineligible for windowed training (its audio would carry no usable target
# or a mislabelled one), so they are enumerated here as the canonical vocabulary the
# eligibility report keys on.
SKIP_REPEATED_RECITATION = "repeated_recitation"
SKIP_LOW_ALIGNMENT = "low_alignment"
SKIP_PHONETIZER_UNSUPPORTED = "phonetizer_unsupported"
SKIP_CLIP_MISSING = "clip_missing"
CLIP_SKIP_REASONS = (
    SKIP_REPEATED_RECITATION,
    SKIP_LOW_ALIGNMENT,
    SKIP_PHONETIZER_UNSUPPORTED,
    SKIP_CLIP_MISSING,
)


@dataclass(frozen=True)
class ClipStatus:
    """Whole-clip segmentation outcome for one passing clip.

    ``audio_filename`` is the whole clip's stable key (the same key
    :class:`tadabur.manifest.ManifestRecord` uses and that every segment row's
    ``clip_audio_filename`` points back to). ``n_words`` is the ayah's Uthmani word
    count — the full word range ``[0, n_words)`` the clip's kept segments must cover
    contiguously to be eligible. ``duration_s`` is the whole 16 kHz clip's duration.
    ``skip_reason`` is one of :data:`CLIP_SKIP_REASONS` when the segmenter could not
    produce trustworthy per-segment labels for the clip, else ``None``.
    """

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    n_words: int
    duration_s: float
    skip_reason: str | None = None


def write_clip_status(path: Path, statuses: list[ClipStatus]) -> None:
    """Write per-clip statuses as a deterministic, key-sorted JSONL sidecar.

    Ordered by ``audio_filename`` so the sidecar is byte-for-byte reproducible across
    runs (the segment manifest and soft-label store share this idempotency contract).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for status in sorted(statuses, key=lambda s: s.audio_filename):
            f.write(json.dumps(asdict(status), ensure_ascii=False, sort_keys=True) + "\n")


def read_clip_status(path: Path) -> list[ClipStatus]:
    """Load every :class:`ClipStatus` from ``path`` in file order."""
    statuses: list[ClipStatus] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            statuses.append(
                ClipStatus(
                    audio_filename=data["audio_filename"],
                    surah_ayah=data["surah_ayah"],
                    reciter_id=data["reciter_id"],
                    n_words=data["n_words"],
                    duration_s=data["duration_s"],
                    skip_reason=data.get("skip_reason"),
                )
            )
    return statuses

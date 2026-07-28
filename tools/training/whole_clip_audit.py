"""Whole-clip audit view models — the training data path C (#25) feeds, for the #6 audit.

The poison audit (:mod:`tadabur.audit_ui`, #6) grades **isolated waqf segments**. But the
whole-clip fine-tune (ADR-0004, P7.C / #25) does *not* train on those segments: it trains on
**fixed windows over the un-waqf-segmented recitation**, whose phoneme CTC target is the
*concatenation* of the constituent segments' realized references, and it **excludes** whole
clips a segment could not be safely windowed for. So a human who signs off on the segments has
still not seen the exact data the model learns from.

This module reconstructs that data path for audit, as plain (torch-free) view models the UI
renders read-only. It is a thin projection over the **canonical** builder
(:func:`training.windowed_labels.build_windowed_labels`) — it does *not* re-derive the
eligibility or window rules, so the audit shows byte-for-byte what the training-label build
produces, never a drifting second opinion. Per clip it surfaces:

* the **whole-clip concatenated label** — the per-segment realized references joined in
  segment order, the clip-level reconstruction ADR-0004 windows are sliced from;
* the **per-segment breakdown** — each kept segment's word range, Uthmani words, and realized
  reference, so the auditor can read what each piece contributes; and
* for an **eligible** clip, the **training windows** — the exact per-window CTC targets
  (:class:`training.windowed_labels.WindowLabel`) the collator feeds, or for an **excluded**
  clip its exclusion **reason** (segment-dropped / repeated_recitation / low_alignment /
  over-long / unsupported / …).

Inputs are the scored segment manifest and the per-clip status sidecar
(:mod:`tadabur.segment_score` / :mod:`tadabur.clip_status`) — the same two artifacts #25
consumes — so the audit view and the training labels are built from one source of truth.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tadabur.clip_status import ClipStatus, read_clip_status
from training.waqf_distill import WindowContract
from training.windowed_labels import (
    Segment,
    WindowLabel,
    build_windowed_labels,
    read_segments,
)


@dataclass(frozen=True)
class SegmentBreakdown:
    """One kept waqf segment's contribution to the whole-clip label, for display.

    ``reference`` is the **normalized** realized reference (the exact CTC label piece the
    window target concatenates); ``uthmani`` is the segment's Uthmani words so the auditor
    can read what it covers.
    """

    segment_index: int
    word_start: int
    word_end: int
    uthmani: str
    reference: str


@dataclass(frozen=True)
class TrainingWindow:
    """One fixed training window's exact phoneme CTC target over the clip's recitation.

    A direct projection of :class:`training.windowed_labels.WindowLabel` — the concatenated
    ``phoneme_label`` the collator feeds, which segments it owns, and its 20 ms/40 ms frame
    lengths — so the auditor confirms the realized training unit, not just the segments.
    """

    window_index: int
    word_start: int
    word_end: int
    feature_frames: int
    logit_frames: int
    phoneme_label: str
    segment_indices: tuple[int, ...]


@dataclass(frozen=True)
class WholeClipView:
    """The whole-clip training data path for one passing clip, eligible or excluded.

    ``whole_clip_label`` is the segment references concatenated in order (the clip-level
    reconstruction). ``included`` clips carry their ``windows`` (the exact CTC targets) and
    ``exclusion_reason is None``; excluded clips carry no windows and the reason they were
    dropped from training, both surfaced from the canonical
    :func:`training.windowed_labels.build_windowed_labels`.
    """

    clip_id: str
    surah_ayah: str
    reciter_id: int
    n_words: int
    included: bool
    exclusion_reason: str | None
    whole_clip_label: str
    segments: tuple[SegmentBreakdown, ...]
    windows: tuple[TrainingWindow, ...]


@dataclass(frozen=True)
class WholeClipAudit:
    """Every passing clip's whole-clip view plus the training-eligibility summary."""

    views: tuple[WholeClipView, ...]

    @property
    def clips_included(self) -> int:
        return sum(1 for v in self.views if v.included)

    @property
    def clips_excluded(self) -> int:
        return sum(1 for v in self.views if not v.included)

    @property
    def exclusions_by_reason(self) -> dict[str, int]:
        """Excluded-clip counts keyed by reason, in sorted (deterministic) order."""
        counts = Counter(v.exclusion_reason for v in self.views if not v.included)
        return dict(sorted(counts.items()))


def _segment_uthmani(segment_manifest_path: Path) -> dict[tuple[str, int], str]:
    """Map ``(clip_audio_filename, segment_index)`` -> the segment's Uthmani words.

    The Uthmani text is a *display* field :func:`training.windowed_labels.read_segments`
    (a training-label parse) deliberately omits, so it is read here for the audit view.
    """
    uthmani: dict[tuple[str, int], str] = {}
    with open(segment_manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uthmani[(row["clip_audio_filename"], row["segment_index"])] = row.get("uthmani", "")
    return uthmani


def _breakdown(
    segments: list[Segment], uthmani: dict[tuple[str, int], str]
) -> tuple[SegmentBreakdown, ...]:
    """Ordered per-segment breakdown for one clip (segment order = word order)."""
    return tuple(
        SegmentBreakdown(
            segment_index=seg.segment_index,
            word_start=seg.word_start,
            word_end=seg.word_end,
            uthmani=uthmani.get((seg.clip_audio_filename, seg.segment_index), ""),
            reference=seg.label_phonemes,
        )
        for seg in sorted(segments, key=lambda s: s.segment_index)
    )


def _windows(labels: list[WindowLabel]) -> tuple[TrainingWindow, ...]:
    """Ordered training windows for one clip, projected from its window labels."""
    return tuple(
        TrainingWindow(
            window_index=label.window_index,
            word_start=label.word_start,
            word_end=label.word_end,
            feature_frames=label.feature_frames,
            logit_frames=label.logit_frames,
            phoneme_label=label.phoneme_label,
            segment_indices=label.segment_indices,
        )
        for label in sorted(labels, key=lambda x: x.window_index)
    )


def build_whole_clip_audit(
    segment_manifest_path: Path,
    clip_status_path: Path,
    contract: WindowContract | None = None,
) -> WholeClipAudit:
    """Reconstruct every passing clip's whole-clip training data path for the #6 audit.

    Runs the canonical training-label build over the same segment manifest + status sidecar
    #25 consumes, then joins its per-window labels and exclusion reasons back to each clip's
    segments. Clips are ordered by ``clip_id`` (the status sidecar's stable order) so the
    audit view is deterministic and idempotent, matching the label build.
    """
    contract = contract or WindowContract()
    segments = read_segments(segment_manifest_path)
    statuses = read_clip_status(clip_status_path)
    uthmani = _segment_uthmani(segment_manifest_path)

    built = build_windowed_labels(segments, statuses, contract)
    exclusion_reason = dict(built.exclusions)
    segments_by_clip: dict[str, list[Segment]] = {}
    for seg in segments:
        segments_by_clip.setdefault(seg.clip_audio_filename, []).append(seg)
    labels_by_clip: dict[str, list[WindowLabel]] = {}
    for label in built.labels:
        labels_by_clip.setdefault(label.clip_audio_filename, []).append(label)

    views: list[WholeClipView] = []
    for status in sorted(statuses, key=lambda s: s.audio_filename):
        clip = status.audio_filename
        breakdown = _breakdown(segments_by_clip.get(clip, []), uthmani)
        reason = exclusion_reason.get(clip)
        views.append(
            WholeClipView(
                clip_id=clip,
                surah_ayah=status.surah_ayah,
                reciter_id=status.reciter_id,
                n_words=status.n_words,
                included=reason is None,
                exclusion_reason=reason,
                whole_clip_label="".join(seg.reference for seg in breakdown),
                segments=breakdown,
                windows=_windows(labels_by_clip.get(clip, [])),
            )
        )
    return WholeClipAudit(views=tuple(views))

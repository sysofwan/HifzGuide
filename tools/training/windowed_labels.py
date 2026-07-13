"""Whole-clip windowed phoneme CTC labels + preflight + reciter split (ADR-0004 P7.C).

The fine-tune runs on **fixed windows over the un-waqf-segmented recitation** (the A2
frozen contract, :mod:`training.waqf_distill`), *not* on the individual waqf segments.
This module turns the scored **segment manifest** (:mod:`tadabur.segment_score`) into the
per-window phoneme CTC labels the collator (#8) and the training runs (#28/#29/#31)
consume, and it owns the two data-integrity gates ADR-0004 requires around them:

* **Clip-level eligibility (not just the two named exclusions).** A window's audio is a
  slice of the *whole* recitation, so it contains the audio of every segment it overlaps
  — including any the segment scorer dropped. Concatenating only the surviving segments'
  labels would therefore leave spoken words with no CTC target. Eligibility is thus
  **clip-level**: a clip is excluded if the segmenter skipped it whole
  (``repeated_recitation`` / ``low_alignment`` / ``phonetizer_unsupported`` /
  ``clip_missing`` — from the :mod:`tadabur.clip_status` sidecar), if any of its segments
  was **dropped** (its surviving segments then no longer cover the ayah's words
  contiguously → ``dropped_segment``), if the recitation is **over-long** (beyond the ~40 s
  provisional cap → ``over_long``, flagged for review not truncated), or if the windowing
  itself cannot cleanly cover the audio (``empty_window`` / ``target_too_long`` below).

* **Per-window labels reconstructed from segment word ranges.** The clip's phoneme label
  is the concatenation of its per-segment realized references (waqf form at each interior
  stop, wasl inside each run — ADR-0002). Each fixed window owns the segments whose audio
  falls in its **center-trusted band** (the frozen #24 ownership rule, reused here at
  segment granularity), and its label is those segments' references concatenated in order.
  The build **asserts** the resulting per-window word ranges have no dropped or duplicated
  word and tile the recitation contiguously; a violation is a bug, not bad data, so it
  raises. ``target_len < logit_frames`` is checked against the **post-adapter 40 ms**
  length of each window (:func:`training.waqf_distill.muaalem_lattice_length`), the CTC
  feasibility bound.

* **Whole-clip reciter split.** The train/val partition is drawn at the **reciter** level,
  so no reciter — and therefore no clip, and therefore none of a clip's windows — can
  straddle the split. :func:`assert_no_reciter_leakage` proves it.

**Conservative windowing (documented limitation).** Ownership is at *segment* granularity
because that is the finest audio↔phoneme correspondence the persisted manifest carries
(per-word audio timestamps are not stored). A single waqf segment whose audio spans more
than one window's center band — a long single-breath run with no interior pause (in
practice a recitation beyond ~4.5 s with no waqf) — cannot be split across the windows its
audio covers, so a later window would be left with audio but no owned segment. Rather than
mislabel it, such a clip is excluded as ``empty_window`` (ADR-0004: "flagged for review,
not silently truncated"). Recovering these clips would require per-word timestamps threaded
from the alignment; that is out of scope here.

Usage:
  python -m training.windowed_labels --segments segment_manifest.jsonl \\
      --clip-status segment_manifest.jsonl.clip_status.jsonl \\
      --out-labels windowed_labels.jsonl --out-report windowed_report.json \\
      [--val-fraction 0.1] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from tadabur.clip_status import ClipStatus, read_clip_status
from training.waqf_distill import (
    SAMPLES_PER_TEACHER_FRAME,
    TARGET_SAMPLE_RATE,
    WindowContract,
    enumerate_windows,
    muaalem_lattice_length,
)

# The provisional per-clip cap from the frozen A2 contract (ADR-0004 "Frozen windowing
# contract"): ~40 s ≈ 2000 feature frames ≈ 8 windows, the ~99th percentile of whole-clip
# durations. A recitation beyond it is excluded as ``over_long`` (flagged for review), not
# silently truncated, so one runaway clip cannot balloon the longest CTC target.
PROVISIONAL_CAP_FEATURE_FRAMES = 2000

# Exclusion reasons this module adds on top of the four whole-clip skip reasons carried in
# from :data:`tadabur.clip_status.CLIP_SKIP_REASONS`. Enumerated so the report keys and the
# tests share one vocabulary.
EXCLUDE_DROPPED_SEGMENT = "dropped_segment"
EXCLUDE_OVER_LONG = "over_long"
EXCLUDE_EMPTY_WINDOW = "empty_window"
EXCLUDE_TARGET_TOO_LONG = "target_too_long"


@dataclass(frozen=True)
class Segment:
    """One kept waqf segment, as read back from the scored segment manifest.

    Only the fields the windowing needs are parsed: the whole-clip key
    (``clip_audio_filename``), the segment's ``segment_index`` order, its half-open
    Uthmani ``word_start`` / ``word_end`` range, its ``(start_s, end_s)`` span within the
    clip, and its normalized realized-reference phoneme string (the CTC label piece).
    """

    clip_audio_filename: str
    surah_ayah: str
    reciter_id: int
    segment_index: int
    word_start: int
    word_end: int
    start_s: float
    end_s: float
    reference_phonemes: str


@dataclass(frozen=True)
class WindowLabel:
    """One fixed training window's phoneme CTC label over a recitation.

    ``start_sample`` / ``num_samples`` are relative to the **recitation** start (the first
    segment's onset, after the neighbour-ayah lead-in was trimmed), the offsets the
    collator (#8) slices the window audio at. ``feature_frames`` is the window's 20 ms
    length and ``logit_frames`` its post-adapter 40 ms length — the CTC lattice the
    ``phoneme_label`` must be shorter than. ``word_start`` / ``word_end`` and
    ``segment_indices`` record which segments' realized references were concatenated.
    """

    clip_audio_filename: str
    surah_ayah: str
    reciter_id: int
    window_index: int
    start_sample: int
    num_samples: int
    feature_frames: int
    logit_frames: int
    phoneme_label: str
    word_start: int
    word_end: int
    segment_indices: tuple[int, ...]


@dataclass(frozen=True)
class WindowedLabels:
    """The full build result: the per-window labels and why every excluded clip was."""

    labels: list[WindowLabel]
    exclusions: list[tuple[str, str]]  # (clip_audio_filename, reason)

    @property
    def reasons(self) -> Counter:
        """Exclusion counts keyed by reason, for the preflight report."""
        return Counter(reason for _clip, reason in self.exclusions)

    @property
    def clips_kept(self) -> int:
        """Distinct clips that contributed at least one window."""
        return len({label.clip_audio_filename for label in self.labels})


def read_segments(path: Path) -> list[Segment]:
    """Load the scored segment manifest (JSONL) into :class:`Segment` rows.

    Requires the whole-clip windowing fields ``segment_score`` now emits
    (``clip_audio_filename`` / ``word_start`` / ``word_end``); a manifest written before
    they existed raises ``KeyError`` rather than silently mis-grouping segments.
    """
    segments: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            segments.append(
                Segment(
                    clip_audio_filename=row["clip_audio_filename"],
                    surah_ayah=row["surah_ayah"],
                    reciter_id=row["reciter_id"],
                    segment_index=row["segment_index"],
                    word_start=row["word_start"],
                    word_end=row["word_end"],
                    start_s=row["start_s"],
                    end_s=row["end_s"],
                    reference_phonemes=row["reference_phonemes"],
                )
            )
    return segments


def _covers_all_words(segments: list[Segment], n_words: int) -> bool:
    """True iff ``segments`` tile ``[0, n_words)`` contiguously with no gap or overlap.

    The scored manifest holds only *kept* segments, so a dropped segment shows up as a
    gap in this word coverage — the whole-clip signal that a constituent segment was
    invalid and the clip is ineligible.
    """
    expected = 0
    for seg in segments:
        if seg.word_start != expected:
            return False
        expected = seg.word_end
    return expected == n_words


def _owner_window(midpoint_sample: int, contract: WindowContract, n_windows: int) -> int:
    """The window index whose center-trusted band owns a segment at ``midpoint_sample``.

    The frozen #24 contract makes each window authoritative over its central band (the
    outer 0.5 s is the overlap owned by the nearer-center neighbour). With a fixed hop the
    bands tile the recitation timeline, so a segment belongs to the single window whose
    band contains its temporal midpoint — ``floor((mid - edge) / hop)``, clamped so the
    leading edge maps to window 0 and the trailing edge to the last window.
    """
    edge = (contract.window_samples - contract.hop_samples) // 2
    index = (midpoint_sample - edge) // contract.hop_samples
    return max(0, min(n_windows - 1, index))


def _needed_windows(recitation_samples: int, contract: WindowContract, n_windows: int) -> int:
    """How many of the tiled windows carry a distinct center band over the recitation.

    :func:`training.waqf_distill.enumerate_windows` steps a window every hop while its
    *start* is inside the clip, so a recitation that overruns the last full window by less
    than the overlap gets a trailing window whose center band lies entirely at/after the
    recitation end — pure overlap the previous window already center-owns. Such redundant
    tails carry no new coverage and own no segment, so they are excluded from the label
    windows; only windows whose band starts before the recitation end are "needed".
    """
    edge = (contract.window_samples - contract.hop_samples) // 2
    needed = 1
    for w in range(1, n_windows):
        if w * contract.hop_samples + edge >= recitation_samples:
            break
        needed = w + 1
    return needed


def build_clip_windows(
    segments: list[Segment],
    status: ClipStatus,
    contract: WindowContract,
    cap_feature_frames: int = PROVISIONAL_CAP_FEATURE_FRAMES,
) -> tuple[list[WindowLabel], str | None]:
    """Windowed labels for one clip, or ``(_, reason)`` when the clip is ineligible.

    Applies the clip-level eligibility gates in order (skip reason → dropped-segment
    coverage → over-long → per-window coverage/target length) and, for an eligible clip,
    assigns each segment to its owning window and concatenates the owned segments'
    realized references into that window's CTC label. Raises ``AssertionError`` if the
    per-window word ranges ever drop, duplicate, or fail to tile the recitation — those
    are invariants of an eligible clip, not data conditions, so a violation is a bug.
    """
    if status.skip_reason is not None:
        return [], status.skip_reason

    ordered = sorted(segments, key=lambda s: s.segment_index)
    if not _covers_all_words(ordered, status.n_words):
        return [], EXCLUDE_DROPPED_SEGMENT

    recitation_start_s = ordered[0].start_s
    recitation_samples = round((ordered[-1].end_s - recitation_start_s) * TARGET_SAMPLE_RATE)
    if recitation_samples // SAMPLES_PER_TEACHER_FRAME > cap_feature_frames:
        return [], EXCLUDE_OVER_LONG

    all_windows = enumerate_windows(recitation_samples, contract)
    needed = _needed_windows(recitation_samples, contract, len(all_windows))
    windows = all_windows[:needed]
    owned: list[list[Segment]] = [[] for _ in windows]
    for seg in ordered:
        start = round((seg.start_s - recitation_start_s) * TARGET_SAMPLE_RATE)
        end = round((seg.end_s - recitation_start_s) * TARGET_SAMPLE_RATE)
        owner = _owner_window((start + end) // 2, contract, needed)
        owned[owner].append(seg)

    labels: list[WindowLabel] = []
    next_word = 0
    for window, window_segments in zip(windows, owned):
        if not window_segments:
            # A needed window (its band adds new coverage) with no owned segment: a waqf
            # segment longer than the hop spanned it (see the module docstring). Flag it.
            return [], EXCLUDE_EMPTY_WINDOW
        assert window_segments[0].word_start == next_word, "window word range not contiguous"
        for prev, cur in zip(window_segments, window_segments[1:]):
            assert cur.word_start == prev.word_end, "duplicated/dropped word within window"
        phoneme_label = "".join(seg.reference_phonemes for seg in window_segments)
        feature_frames = window.num_samples // SAMPLES_PER_TEACHER_FRAME
        logit_frames = muaalem_lattice_length(feature_frames)
        if len(phoneme_label) >= logit_frames:
            return [], EXCLUDE_TARGET_TOO_LONG
        labels.append(
            WindowLabel(
                clip_audio_filename=status.audio_filename,
                surah_ayah=status.surah_ayah,
                reciter_id=status.reciter_id,
                window_index=window.index,
                start_sample=window.start_sample,
                num_samples=window.num_samples,
                feature_frames=feature_frames,
                logit_frames=logit_frames,
                phoneme_label=phoneme_label,
                word_start=window_segments[0].word_start,
                word_end=window_segments[-1].word_end,
                segment_indices=tuple(seg.segment_index for seg in window_segments),
            )
        )
        next_word = window_segments[-1].word_end
    assert next_word == status.n_words, "windows do not tile the recitation's words"
    return labels, None


def build_windowed_labels(
    segments: list[Segment],
    statuses: list[ClipStatus],
    contract: WindowContract | None = None,
    cap_feature_frames: int = PROVISIONAL_CAP_FEATURE_FRAMES,
) -> WindowedLabels:
    """Build every eligible clip's windowed labels and the exclusion-by-reason report.

    Clips are processed in a stable ``audio_filename`` order (the status sidecar's order)
    so the output is deterministic and idempotent. Every scored clip must have a status
    record; a segment whose clip is absent from ``statuses`` is a data-integrity failure
    (a stale/mismatched sidecar) and raises rather than being silently dropped.
    """
    contract = contract or WindowContract()
    by_clip: dict[str, list[Segment]] = {}
    for seg in segments:
        by_clip.setdefault(seg.clip_audio_filename, []).append(seg)

    known = {status.audio_filename for status in statuses}
    orphan = set(by_clip) - known
    if orphan:
        raise ValueError(
            f"{len(orphan)} clip(s) in the segment manifest have no clip-status record "
            f"(stale or mismatched sidecar): {sorted(orphan)[:5]}"
        )

    labels: list[WindowLabel] = []
    exclusions: list[tuple[str, str]] = []
    for status in sorted(statuses, key=lambda s: s.audio_filename):
        clip_labels, reason = build_clip_windows(
            by_clip.get(status.audio_filename, []), status, contract, cap_feature_frames
        )
        if reason is not None:
            exclusions.append((status.audio_filename, reason))
        else:
            labels.extend(clip_labels)
    return WindowedLabels(labels=labels, exclusions=exclusions)


def split_by_reciter(
    labels: list[WindowLabel], val_fraction: float, seed: int
) -> tuple[list[WindowLabel], list[WindowLabel]]:
    """Partition windows into (train, val) at the **reciter** level — no clip leakage.

    Reciters are shuffled with a seeded RNG (deterministic) and greedily assigned to val
    until val holds at least ``val_fraction`` of the windows; because whole reciters move
    together, every clip — and every one of its windows — lands in exactly one partition.
    ``val_fraction`` of 0 yields an empty val set (all training).
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    windows_by_reciter: dict[int, int] = Counter(label.reciter_id for label in labels)
    reciters = sorted(windows_by_reciter)
    random.Random(seed).shuffle(reciters)

    target = val_fraction * len(labels)
    val_reciters: set[int] = set()
    val_count = 0
    for reciter in reciters:
        if val_count >= target:
            break
        val_reciters.add(reciter)
        val_count += windows_by_reciter[reciter]

    train = [label for label in labels if label.reciter_id not in val_reciters]
    val = [label for label in labels if label.reciter_id in val_reciters]
    return train, val


def assert_no_reciter_leakage(
    train: list[WindowLabel], val: list[WindowLabel]
) -> dict:
    """Prove the split has no reciter/clip/window leakage; raise if it does.

    Verifies the two partitions share no reciter and no clip, and that every clip's
    windows resolve to a single partition. Returns a small proof summary (reciter and clip
    counts per side) for the report.
    """
    train_reciters = {label.reciter_id for label in train}
    val_reciters = {label.reciter_id for label in val}
    assert train_reciters.isdisjoint(val_reciters), "reciter appears in both train and val"

    train_clips = {label.clip_audio_filename for label in train}
    val_clips = {label.clip_audio_filename for label in val}
    assert train_clips.isdisjoint(val_clips), "clip appears in both train and val"

    return {
        "train_reciters": len(train_reciters),
        "val_reciters": len(val_reciters),
        "train_clips": len(train_clips),
        "val_clips": len(val_clips),
        "train_windows": len(train),
        "val_windows": len(val),
    }


def write_labels(path: Path, labels: list[WindowLabel], split: str) -> None:
    """Append windowed labels as deterministic, key-sorted JSONL tagged with ``split``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for label in sorted(
            labels, key=lambda x: (x.clip_audio_filename, x.window_index)
        ):
            row = asdict(label)
            row["segment_indices"] = list(label.segment_indices)
            row["split"] = split
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--segments", type=Path, required=True,
        help="Scored segment manifest (JSONL) from tadabur.segment_score.",
    )
    parser.add_argument(
        "--clip-status", type=Path, required=True,
        help="Per-clip status sidecar (JSONL) from tadabur.segment_score.",
    )
    parser.add_argument(
        "--out-labels", type=Path, required=True,
        help="Output windowed CTC labels (JSONL), one row per training window.",
    )
    parser.add_argument(
        "--out-report", type=Path, required=True,
        help="Output preflight report (JSON): exclusion-by-reason + split proof.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of windows to hold out for validation, split by reciter.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the deterministic reciter split.",
    )
    parser.add_argument(
        "--hop-feature-frames", type=int, default=None,
        help="Override the frozen 200-frame (4 s) hop; the window length is fixed at 5 s.",
    )
    parser.add_argument(
        "--cap-feature-frames", type=int, default=PROVISIONAL_CAP_FEATURE_FRAMES,
        help=f"Over-long recitation cap in 20 ms frames (default: {PROVISIONAL_CAP_FEATURE_FRAMES}).",
    )
    args = parser.parse_args()

    contract = WindowContract(
        **({"hop_feature_frames": args.hop_feature_frames} if args.hop_feature_frames else {})
    )
    segments = read_segments(args.segments)
    statuses = read_clip_status(args.clip_status)
    print(f"Loaded {len(segments)} segments across {len(statuses)} clips.")

    built = build_windowed_labels(segments, statuses, contract, args.cap_feature_frames)
    train, val = split_by_reciter(built.labels, args.val_fraction, args.seed)
    proof = assert_no_reciter_leakage(train, val)

    if args.out_labels.exists():
        args.out_labels.unlink()  # append-writer: start clean so a re-run is idempotent
    write_labels(args.out_labels, train, "train")
    write_labels(args.out_labels, val, "val")

    report = {
        "clips_total": len(statuses),
        "clips_kept": built.clips_kept,
        "windows_total": len(built.labels),
        "exclusions_by_reason": dict(sorted(built.reasons.items())),
        "split": proof,
        "contract": {
            "window_feature_frames": contract.feature_frames,
            "hop_feature_frames": contract.hop_feature_frames,
            "cap_feature_frames": args.cap_feature_frames,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Kept {built.clips_kept}/{len(statuses)} clips → {len(built.labels)} windows "
        f"({len(train)} train / {len(val)} val). Exclusions: {dict(built.reasons)}. "
        f"Wrote labels to {args.out_labels} and report to {args.out_report}."
    )


if __name__ == "__main__":
    main()

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
  stop, wasl inside each run — ADR-0002). Each fixed window's label is the references of
  the segments whose audio falls inside that window's audio span, concatenated in order.
  Because the persisted manifest carries only *segment*-level audio timing (no per-word
  timestamps), a window can only be labelled when **every segment overlapping its audio is
  fully contained in it**: then we can prove every spoken word in the window audio is a
  full CTC target. A segment that crosses a window edge (its audio reaches past the
  boundary) leaves a partially-spoken word we cannot represent, so the whole clip is
  excluded (``segment_crosses_window``) rather than mislabelled — the same
  exclude-don't-corrupt rule as a dropped segment. Interior waqf is still learned: a clip
  whose whole recitation fits one 5 s window keeps all its interior segments. The build
  **asserts** each window's owned segments are word-contiguous and that the windows cover
  ``[0, n_words)``. ``target_len < logit_frames`` is checked against the **post-adapter
  40 ms** length of each window
  (:func:`training.waqf_distill.muaalem_lattice_length`), the CTC feasibility bound.

* **One shared clip-relative window grid with the waqf soft labels.** Windows are
  enumerated over the **recitation span** (``[recitation_start_s, recitation_end_s]`` from
  the status sidecar — the un-waqf-segmented recitation, trimming the neighbour-ayah
  lead-in / trailing bleed the staged clip keeps), on the **identical grid**
  :func:`training.waqf_distill.generate_soft_labels` uses
  (:func:`training.waqf_distill.enumerate_recitation_windows`). ``start_sample`` is
  **clip-relative** (the recitation offset is folded in and persisted as
  ``recitation_start_sample``), so a window's phoneme CTC target and its waqf soft target
  share the same ``(window_index, start_sample, num_samples)`` and the joint fine-tune
  (#28/#29/#31) pairs them without misalignment (ADR-0004 "same window contract").

* **Whole-clip reciter split.** The train/val partition is drawn at the **reciter** level,
  so no reciter — and therefore no clip, and therefore none of a clip's windows — can
  straddle the split. :func:`assert_no_reciter_leakage` proves it.

**Conservative windowing (documented limitation).** Ownership is at *segment* granularity
because that is the finest audio↔phoneme correspondence the persisted manifest carries
(per-word audio timestamps are not stored). A recitation longer than one 5 s window whose
segment (waqf) boundaries do not happen to fall on the window grid has at least one segment
crossing a window edge, so it is excluded (``segment_crosses_window``): with only
segment-level timing we cannot prove that window's edge word is a full target, and a
mislabelled window is worse than a dropped clip (ADR-0004 "flagged for review, not silently
truncated"). Also excluded: a **dropped** segment (``dropped_segment`` word-coverage gap),
an **over-long** recitation (``over_long``), a window overlapping **no** segment
(``empty_window`` defensive guard), or a target longer than its lattice
(``target_too_long``). Recovering the longer clips would require per-word timestamps
threaded from the alignment; that is out of scope here.

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
    enumerate_recitation_windows,
    muaalem_lattice_length,
    recitation_window_span,
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
EXCLUDE_RE_READ = "re_read"
EXCLUDE_OVER_LONG = "over_long"
EXCLUDE_EMPTY_WINDOW = "empty_window"
EXCLUDE_TARGET_TOO_LONG = "target_too_long"
EXCLUDE_SEGMENT_CROSSES_WINDOW = "segment_crosses_window"


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

    ``start_sample`` / ``num_samples`` are **clip-relative** (the recitation offset is
    folded in), so they name the exact whole-clip sample span the collator (#8) slices the
    window audio at and the key the waqf soft label
    (:class:`training.waqf_distill.SoftLabelStore`) is joined on. ``recitation_start_sample``
    records that clip-relative offset explicitly (window 0's ``start_sample``).
    ``feature_frames`` is the window's 20 ms length and ``logit_frames`` its post-adapter
    40 ms length — the CTC lattice the ``phoneme_label`` must be shorter than.
    ``word_start`` / ``word_end`` and ``segment_indices`` record which segments' realized
    references were concatenated (segments in a window overlap are labelled in both
    neighbouring windows).
    """

    clip_audio_filename: str
    surah_ayah: str
    reciter_id: int
    window_index: int
    start_sample: int
    num_samples: int
    recitation_start_sample: int
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


def _window_segments(
    ordered: list[Segment], clip_start_sample: int, clip_end_sample: int
) -> tuple[list[Segment], bool]:
    """The segments inside a window's clip-relative span, and whether any one crosses it.

    A segment's audio overlaps ``[clip_start_sample, clip_end_sample)`` when it starts
    before the window ends and ends after the window starts (segment times are converted
    to the same clip-relative sample grid the window spans use). It is **fully contained**
    only when it also starts at/after ``clip_start_sample`` and ends at/before
    ``clip_end_sample``. Returns the overlapping segments in order and a flag that is true
    if any overlapping segment is *not* fully contained — i.e. it crosses the window edge,
    so with segment-level timing alone we cannot prove that window's edge word is a full
    CTC target and the clip must be excluded.
    """
    overlapping: list[Segment] = []
    crosses = False
    for seg in ordered:
        seg_start = round(seg.start_s * TARGET_SAMPLE_RATE)
        seg_end = round(seg.end_s * TARGET_SAMPLE_RATE)
        if seg_end > clip_start_sample and seg_start < clip_end_sample:
            overlapping.append(seg)
            if seg_start < clip_start_sample or seg_end > clip_end_sample:
                crosses = True
    return overlapping, crosses


def build_clip_windows(
    segments: list[Segment],
    status: ClipStatus,
    contract: WindowContract,
    cap_feature_frames: int = PROVISIONAL_CAP_FEATURE_FRAMES,
) -> tuple[list[WindowLabel], str | None]:
    """Windowed labels for one clip, or ``(_, reason)`` when the clip is ineligible.

    Applies the clip-level eligibility gates in order (skip reason → re-read → dropped-
    segment coverage → over-long → per-window containment/target length) and, for an
    eligible clip, labels each window with the concatenated realized references of the
    segments **fully contained in that window's clip-relative span**. If a segment's audio
    crosses a window edge, segment-level timing cannot prove that window's edge word is a
    full CTC target, so the clip is excluded (``segment_crosses_window``) rather than
    mislabelled. Windows are enumerated on the shared clip-relative grid the waqf soft
    labels use (:func:`training.waqf_distill.enumerate_recitation_windows`), so the two
    artifacts' ``(window_index, start_sample, num_samples)`` match. Raises
    ``AssertionError`` if a kept window's segments are not word-contiguous or the windows
    fail to cover the recitation's words — invariants of an eligible clip, so a violation is
    a bug, not a data condition.
    """
    if status.skip_reason is not None:
        return [], status.skip_reason

    # A re-read clip's segments overlap in words (two passes over a phrase), so they cannot
    # tile the recitation into one contiguous whole-clip target. Exclude it under its own
    # reason rather than the misleading ``dropped_segment`` (nothing was dropped).
    if status.re_reads:
        return [], EXCLUDE_RE_READ

    ordered = sorted(segments, key=lambda s: s.segment_index)
    if not _covers_all_words(ordered, status.n_words):
        return [], EXCLUDE_DROPPED_SEGMENT

    recitation_start_sample, recitation_num_samples = recitation_window_span(
        status.recitation_start_s, status.recitation_end_s
    )
    if recitation_num_samples // SAMPLES_PER_TEACHER_FRAME > cap_feature_frames:
        return [], EXCLUDE_OVER_LONG

    windows = enumerate_recitation_windows(
        recitation_start_sample, recitation_num_samples, contract
    )
    labels: list[WindowLabel] = []
    covered_end = 0
    for window in windows:
        clip_end_sample = window.start_sample + window.num_samples
        window_segments, crosses = _window_segments(
            ordered, window.start_sample, clip_end_sample
        )
        if crosses:
            # A segment reaches past this window's edge: its edge word is only partly in
            # the window audio, so with segment-level timing we cannot label the window
            # without corrupting the target. Exclude the whole clip (ADR-0004).
            return [], EXCLUDE_SEGMENT_CROSSES_WINDOW
        if not window_segments:
            # A window overlapping no segment: its audio carries no labellable word. A 5 s
            # window never fits inside a sub-second waqf pause, so this is a defensive
            # guard against a degenerate span, not a normal outcome.
            return [], EXCLUDE_EMPTY_WINDOW
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
                recitation_start_sample=recitation_start_sample,
                feature_frames=feature_frames,
                logit_frames=logit_frames,
                phoneme_label=phoneme_label,
                word_start=window_segments[0].word_start,
                word_end=window_segments[-1].word_end,
                segment_indices=tuple(seg.segment_index for seg in window_segments),
            )
        )
        # Consecutive windows overlap, so a later window's first word may repeat the
        # previous window's last; coverage must still advance to the ayah's final word
        # with no gap between what the windows cover.
        assert window_segments[0].word_start <= covered_end, "window coverage gap"
        covered_end = max(covered_end, window_segments[-1].word_end)
    assert covered_end == status.n_words, "windows do not cover the recitation's words"
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


def read_labels(path: Path) -> dict[str, list[WindowLabel]]:
    """Inverse of :func:`write_labels`: JSONL rows back into ``WindowLabel`` lists by split.

    Returns a mapping ``split -> labels`` (``"train"`` / ``"val"``) so the training run
    (#29/#31) consumes exactly the partitions the label build wrote, without re-deriving
    the reciter split. Rows are read in file order (already key-sorted by the writer), so
    the reconstruction is deterministic. A row missing the split tag or any window field
    raises ``KeyError`` rather than silently dropping a window.
    """
    by_split: dict[str, list[WindowLabel]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = row["split"]
            by_split.setdefault(split, []).append(
                WindowLabel(
                    clip_audio_filename=row["clip_audio_filename"],
                    surah_ayah=row["surah_ayah"],
                    reciter_id=row["reciter_id"],
                    window_index=row["window_index"],
                    start_sample=row["start_sample"],
                    num_samples=row["num_samples"],
                    recitation_start_sample=row["recitation_start_sample"],
                    feature_frames=row["feature_frames"],
                    logit_frames=row["logit_frames"],
                    phoneme_label=row["phoneme_label"],
                    word_start=row["word_start"],
                    word_end=row["word_end"],
                    segment_indices=tuple(row["segment_indices"]),
                )
            )
    return by_split


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

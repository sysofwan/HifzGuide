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

* **Per-window labels cut at word edges.** The clip's phoneme label is the concatenation
  of its per-segment realized references (waqf form at each interior stop, wasl inside each
  run — ADR-0002). A fixed window edge almost never falls on a word boundary, so each
  window is first **snapped inward to the whole words its audio contains**
  (:func:`training.waqf_distill.snap_window_to_words`, using the per-word onset times the
  segment scorer persists on the clip-status sidecar). Its label is then the concatenation
  of each overlapping segment's own phonetization *sliced* at those words
  (:meth:`Segment.slice_words`) — sliced, never re-phonetized, because re-phonetizing a
  window's word range would apply the phonetizer's CleanEnd and invent a waqf at what is
  only a window edge. Because the snapped span is a sub-span of the fixed window and both
  artifacts snap identically, the shared grid survives. This is what makes recitations
  **longer than one window** trainable — and with them the interior waqf ADR-0004 needs the
  waqf head to see in context. ``target_len < logit_frames`` is checked against the
  **post-adapter 40 ms** length of each window
  (:func:`training.waqf_distill.muaalem_lattice_length`), the CTC feasibility bound.
  A manifest without per-word offsets falls back to the older segment-granularity rule
  below, which excludes any clip whose segment crosses a window edge.

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

**What is still excluded.** A clip the segmenter skipped whole (its ``skip_reason``), a
**re-read** clip (``re_read`` — its segments overlap in words, so they cannot tile one
recitation), a **dropped** segment (``dropped_segment`` word-coverage gap), an **over-long**
recitation (``over_long``, flagged for review not truncated), a window with no whole word
(``empty_window``), or a target longer than its lattice (``target_too_long``). A word longer
than the grid's 1 s window overlap (a long madd) fits in no window; it is simply **not
trained on**, and no window holds its audio either, so no target is corrupted. On a manifest
predating per-word offsets the legacy segment-granularity path also excludes any clip with a
segment crossing a window edge (``segment_crosses_window``) or an uncovered word
(``word_uncovered``).

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
    SAMPLES_PER_STUDENT_FRAME,
    TARGET_SAMPLE_RATE,
    WindowContract,
    clip_recitation_windows,
    feature_frames_for_samples,
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
EXCLUDE_NO_WORD_TIMES = "no_word_times"
EXCLUDE_WORD_UNCOVERED = "word_uncovered"
EXCLUDE_HELD_OUT_EVAL_CLIP = "held_out_eval_clip"


def read_held_out_clips(path: Path) -> frozenset[str]:
    """Clip ids reserved for evaluation, from ``tadabur.waqf_freeze``'s partition report.

    The #34 waqf event eval scores the calibration and test clips named in that report. They
    must not also be training examples, or the eval measures memorization of those exact
    clips rather than the waqf head's behaviour. The freeze emits the clip lists (and a
    stricter ``must_exclude_reciters``); this reads the clip lists, which is the leak that
    makes the reported number meaningless rather than merely optimistic.
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = {"calibration_clips", "test_clips"} - set(report)
    if missing:
        raise KeyError(f"{path} is not a waqf_freeze partition report (no {sorted(missing)})")
    return frozenset(report["calibration_clips"]) | frozenset(report["test_clips"])


@dataclass(frozen=True)
class Segment:
    """One kept waqf segment, as read back from the scored segment manifest.

    Only the fields the windowing needs are parsed: the whole-clip key
    (``clip_audio_filename``), the segment's ``segment_index`` order, its half-open
    Uthmani ``word_start`` / ``word_end`` range, its ``(start_s, end_s)`` span within the
    clip, and its realized-reference phoneme string (the CTC label piece).

    That label piece is the **tashkeel-bearing** ``raw_reference_phonemes``, per ADR-0003:
    the fine-tune's whole point is to raise the model's short-vowel reliability, and the
    model emits fatha/damma/kasra as real output classes (ids 32-34). The manifest's other
    reference, ``reference_phonemes``, is ``.balanced``-normalized — it exists to mirror
    the gate's vowel-blind tolerance, and training on it would leave classes 32-34 with no
    positive target anywhere in the corpus. Since the phoneme head trains in full
    (``modules_to_save``), that does not merely fail to teach tashkeel: it actively
    suppresses the classes, destroying a capability the base checkpoint already has.
    """

    clip_audio_filename: str
    surah_ayah: str
    reciter_id: int
    segment_index: int
    word_start: int
    word_end: int
    start_s: float
    end_s: float
    label_phonemes: str
    label_word_offsets: tuple[int, ...] = ()

    def slice_words(self, word_start: int, word_end: int) -> str:
        """This segment's realized reference restricted to ``[word_start, word_end)``.

        Slicing the segment's *own* phonetization (rather than re-phonetizing the word
        range) is what keeps a mid-segment window edge free of a phantom waqf: the edge
        word keeps the wasl form the segment gave it. Requires ``word_offsets``; the
        range must lie inside this segment's word span.
        """
        assert self.label_word_offsets, "segment has no per-word phoneme offsets"
        assert self.word_start <= word_start <= word_end <= self.word_end
        return self.label_phonemes[
            self.label_word_offsets[word_start - self.word_start] :
            self.label_word_offsets[word_end - self.word_start]
        ]


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

    The CTC label is read from the **tashkeel-bearing** ``raw_reference_phonemes`` and
    sliced with the offsets that index *it* (``raw_word_offsets``) — see :class:`Segment`
    for why the vowel-stripped ``reference_phonemes`` must not be used. A manifest written
    before ``raw_word_offsets`` existed raises with a regeneration instruction rather than
    falling back to the normalized offsets, which index a *different string* and would
    silently slice the label at the wrong characters.
    """
    segments: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "raw_word_offsets" not in row:
                raise KeyError(
                    f"{path} predates 'raw_word_offsets' — its 'word_offsets' index the "
                    "vowel-stripped reference, so training on it would drop every "
                    "fatha/damma/kasra from the CTC target (ADR-0003). Regenerate the "
                    "manifest with tadabur.segment_score, or backfill it with "
                    "tadabur.backfill_raw_word_offsets."
                )
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
                    label_phonemes=row["raw_reference_phonemes"],
                    label_word_offsets=tuple(row["raw_word_offsets"]),
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


def _contained_word_range(
    word_times: tuple[float, ...],
    word_start: int,
    word_end: int,
    start_sample: int,
    end_sample: int,
) -> tuple[int, int]:
    """The half-open word range fully inside ``[start_sample, end_sample)`` audio.

    Word ``j`` spans ``[word_times[j], word_times[j + 1])`` (clip-relative seconds from
    the whole-clip alignment, :func:`tadabur.waqf_detect.word_onset_times`), so it is a
    full CTC target for this window only when both edges lie inside the window's audio.
    The window was already snapped to whole words
    (:func:`training.waqf_distill.snap_window_to_words`) but its edges were then rounded
    **in** to the 40 ms student lattice, so an edge word can start/end up to one student
    frame outside the snapped span; the comparison allows exactly that slack. No other
    word can sneak in through it — every recited word is far longer than 40 ms.
    The onsets are non-decreasing, so the qualifying words form one contiguous run;
    returns an empty range (``a == b``) when no word fits. Search is clamped to the
    recited range ``[word_start, word_end)`` so a never-recited tail is never labelled.
    """
    slack = SAMPLES_PER_STUDENT_FRAME
    first = None
    last = word_start
    for word in range(word_start, word_end):
        onset = round(word_times[word] * TARGET_SAMPLE_RATE)
        offset = round(word_times[word + 1] * TARGET_SAMPLE_RATE)
        if onset >= start_sample - slack and offset <= end_sample + slack:
            if first is None:
                first = word
            last = word + 1
        elif first is not None:
            break
    return (word_start, word_start) if first is None else (first, last)


def _label_word_range(
    ordered: list[Segment], word_start: int, word_end: int
) -> tuple[str, tuple[int, ...]]:
    """Concatenated realized reference for ``[word_start, word_end)`` across segments.

    A window's word range may start or end inside a segment (that edge is a window edge,
    not a waqf), so each overlapping segment contributes only its own slice of the range
    — taken from its persisted phonetization via :meth:`Segment.slice_words`, never
    re-phonetized. Returns the label and the contributing segment indices.
    """
    pieces: list[str] = []
    indices: list[int] = []
    for seg in ordered:
        lo = max(seg.word_start, word_start)
        hi = min(seg.word_end, word_end)
        if lo < hi:
            pieces.append(seg.slice_words(lo, hi))
            indices.append(seg.segment_index)
    return "".join(pieces), tuple(indices)


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
    if feature_frames_for_samples(recitation_num_samples) > cap_feature_frames:
        return [], EXCLUDE_OVER_LONG

    windows = clip_recitation_windows(
        recitation_start_sample, recitation_num_samples, contract, status.word_times
    )
    word_level = bool(status.word_times) and all(seg.label_word_offsets for seg in ordered)
    if word_level and len(status.word_times) != status.n_words + 1:
        return [], EXCLUDE_NO_WORD_TIMES

    labels: list[WindowLabel] = []
    covered_end = 0
    for window in windows:
        clip_end_sample = window.start_sample + window.num_samples
        if word_level:
            # Per-word timing: the window owns exactly the words whose audio is fully
            # inside it, so a recitation longer than one window is cut at a *word* edge
            # instead of being discarded for crossing a segment (see module docstring).
            word_start, word_end = _contained_word_range(
                status.word_times, 0, status.n_words, window.start_sample, clip_end_sample
            )
            if word_start == word_end:
                return [], EXCLUDE_EMPTY_WINDOW
            phoneme_label, segment_indices = _label_word_range(
                ordered, word_start, word_end
            )
        else:
            window_segments, crosses = _window_segments(
                ordered, window.start_sample, clip_end_sample
            )
            if crosses:
                # No per-word timing: a segment reaching past this window's edge leaves an
                # edge word we cannot prove is a full CTC target. Exclude the whole clip.
                return [], EXCLUDE_SEGMENT_CROSSES_WINDOW
            if not window_segments:
                return [], EXCLUDE_EMPTY_WINDOW
            for prev, cur in zip(window_segments, window_segments[1:]):
                assert cur.word_start == prev.word_end, "duplicated/dropped word in window"
            phoneme_label = "".join(seg.label_phonemes for seg in window_segments)
            segment_indices = tuple(seg.segment_index for seg in window_segments)
            word_start = window_segments[0].word_start
            word_end = window_segments[-1].word_end
        feature_frames = feature_frames_for_samples(window.num_samples)
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
                word_start=word_start,
                word_end=word_end,
                segment_indices=segment_indices,
            )
        )
        # Consecutive windows overlap, so a later window's first word may repeat the
        # previous window's last; coverage must still advance to the ayah's final word
        # with no gap between what the windows cover.
        covered_end = max(covered_end, word_end)
    if not word_level and covered_end != status.n_words:
        # Segment-granularity windows tile the whole recitation, so a coverage shortfall
        # there is a bug. Word-snapped windows deliberately leave gaps: a word longer than
        # the grid's 1 s window overlap fits in no window, so it is simply not trained on
        # (its audio is in no window either — see ``snap_window_to_words``).
        return [], EXCLUDE_WORD_UNCOVERED
    return labels, None


def build_windowed_labels(
    segments: list[Segment],
    statuses: list[ClipStatus],
    contract: WindowContract | None = None,
    cap_feature_frames: int = PROVISIONAL_CAP_FEATURE_FRAMES,
    held_out_clips: frozenset[str] = frozenset(),
) -> WindowedLabels:
    """Build every eligible clip's windowed labels and the exclusion-by-reason report.

    Clips are processed in a stable ``audio_filename`` order (the status sidecar's order)
    so the output is deterministic and idempotent. Every scored clip must have a status
    record; a segment whose clip is absent from ``statuses`` is a data-integrity failure
    (a stale/mismatched sidecar) and raises rather than being silently dropped.

    ``held_out_clips`` (see :func:`read_held_out_clips`) are dropped before any other test so
    an eval clip can never become a training example, whatever its data condition.
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
        if status.audio_filename in held_out_clips:
            exclusions.append((status.audio_filename, EXCLUDE_HELD_OUT_EVAL_CLIP))
            continue
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
    parser.add_argument(
        "--held-out-clips", type=Path, default=None,
        help="waqf_freeze partition report (JSON); its calibration+test clips are excluded "
             "from training so the #34 event eval is not scored on its own training data.",
    )
    args = parser.parse_args()

    contract = WindowContract(
        **({"hop_feature_frames": args.hop_feature_frames} if args.hop_feature_frames else {})
    )
    segments = read_segments(args.segments)
    statuses = read_clip_status(args.clip_status)
    print(f"Loaded {len(segments)} segments across {len(statuses)} clips.")

    held_out = read_held_out_clips(args.held_out_clips) if args.held_out_clips else frozenset()
    if held_out:
        print(f"Holding out {len(held_out)} eval clips from training.")

    built = build_windowed_labels(
        segments, statuses, contract, args.cap_feature_frames, held_out
    )
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

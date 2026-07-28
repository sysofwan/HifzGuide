"""Waqf-segment-scoped phoneme CTC labels — the rung-(1) control for the #33 ladder.

The #33 ablation ladder (:mod:`training.ablation_ladder`) reads three #7 eval reports and
asks two questions, one per transition:

* **(1) → (2)** — does moving the training **unit** from the individual waqf segment to
  the whole clip erode should-reject discrimination? ADR-0004 calls this the dangerous
  transition, because a whole clip exposes the backbone to far more audio to drift on.
* **(2) → (3)** — does *adding the waqf head* erode phoneme quality? Pinned near zero by
  the gradient-isolation check the joint run asserts.

Each transition is only interpretable if it changes **exactly one** variable. Rung (1) is
therefore not the stock checkpoint: it is a fine-tune on the *same corpus, same frozen
window grid, same LoRA config* as rung (2), differing only in the unit. Scoring the stock
model in that slot conflates "segmented vs whole-clip" with "fine-tuned vs not", and the
stock model biases the answer in a known direction — it wrongly rejects most genuinely-good
recitations, and a trigger-happy rejecter scores high on should-reject discrimination for
free, manufacturing a (1) → (2) "regression" out of the baseline rather than the move.

This module builds that control. It reuses :mod:`training.windowed_labels` wholesale — the
same clip-level eligibility gates, the same frozen grid
(:func:`training.waqf_distill.clip_recitation_windows`), the same inward word snapping, and
the same slice-never-re-phonetize labelling — and changes one thing: the grid is enumerated
over each **waqf segment's** span (``[start_s, end_s)``) instead of the whole recitation
span. So rung (1) and rung (2) see the same clips, the same reciter split, and window audio
cut on the same lattice; only which spans become training examples differs.

Because both artifacts index windows by ``(clip_audio_filename, window_index)`` and the
segment-scoped grids restart per segment, window indices are renumbered **running over the
clip** so the key stays unique. The output is otherwise a byte-compatible
:func:`training.windowed_labels.write_labels` file, consumed unchanged by
``training.whole_clip_phoneme train --labels ... --audio-dir <clip audio>``.

Rung (1) is phoneme-only, so these labels are never joined to waqf soft targets; the
``recitation_start_sample`` field records the segment's own start for traceability.

Usage:
  python -m training.segmented_labels \\
      --manifest tadabur/audit_run/seg_v21/manifest.jsonl \\
      --out tadabur/audit_run/seg_v21/segmented_labels.jsonl \\
      [--val-fraction 0.1] [--seed 0]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tadabur.clip_status import ClipStatus, read_clip_status
from training.waqf_distill import (
    TARGET_SAMPLE_RATE,
    WindowContract,
    clip_recitation_windows,
    feature_frames_for_samples,
    muaalem_lattice_length,
    recitation_window_span,
)
from training.windowed_labels import (
    EXCLUDE_DROPPED_SEGMENT,
    EXCLUDE_NO_WORD_TIMES,
    EXCLUDE_OVER_LONG,
    EXCLUDE_RE_READ,
    PROVISIONAL_CAP_FEATURE_FRAMES,
    Segment,
    WindowedLabels,
    WindowLabel,
    _contained_word_range,
    _covers_all_words,
    assert_no_reciter_leakage,
    EXCLUDE_HELD_OUT_EVAL_CLIP,
    ctc_target_slots,
    read_held_out_clips,
    resolve_held_out_clips,
    read_segments,
    split_by_reciter,
    write_labels,
)

#: No segment of this clip produced a labellable window (every one is shorter than a
#: whole word on the lattice, or its target does not fit its CTC lattice).
EXCLUDE_NO_SEGMENT_WINDOW = "no_segment_window"


def build_clip_segment_windows(
    segments: list[Segment],
    status: ClipStatus,
    contract: WindowContract,
    cap_feature_frames: int = PROVISIONAL_CAP_FEATURE_FRAMES,
) -> tuple[list[WindowLabel], str | None]:
    """Segment-scoped windowed labels for one clip, or ``(_, reason)`` if ineligible.

    Applies the *same* clip-level eligibility gates as
    :func:`training.windowed_labels.build_clip_windows` — skip reason, re-read, dropped-
    segment word coverage, over-long recitation, per-word timing present — so rung (1) and
    rung (2) train on the same clip population and the ladder's (1) → (2) delta isolates
    the training unit alone. Each surviving segment then gets its own frozen-grid windows
    over ``[start_s, end_s)``, word-snapped inward exactly as the whole-clip path snaps.

    A segment that yields no labellable window (too short to hold a whole word on the
    40 ms lattice, or a target that does not fit its CTC lattice) is dropped on its own —
    unlike the whole-clip path, an unusable *segment* leaves no un-targeted audio in any
    other segment's window, so it cannot corrupt a label and need not cost the clip. The
    clip is excluded only when no segment survives.
    """
    if status.skip_reason is not None:
        return [], status.skip_reason
    if status.re_reads:
        return [], EXCLUDE_RE_READ

    ordered = sorted(segments, key=lambda s: s.segment_index)
    if not _covers_all_words(ordered, status.n_words):
        return [], EXCLUDE_DROPPED_SEGMENT
    if not status.word_times or not all(seg.label_word_offsets for seg in ordered):
        return [], EXCLUDE_NO_WORD_TIMES
    if len(status.word_times) != status.n_words + 1:
        return [], EXCLUDE_NO_WORD_TIMES

    recitation_frames = feature_frames_for_samples(
        round((status.recitation_end_s - status.recitation_start_s) * TARGET_SAMPLE_RATE)
    )
    if recitation_frames > cap_feature_frames:
        return [], EXCLUDE_OVER_LONG

    labels: list[WindowLabel] = []
    for seg in ordered:
        # Floored to the 40 ms student lattice exactly as the whole-clip path floors the
        # recitation origin, so both rungs' windows start on the same lattice.
        start_sample, num_samples = recitation_window_span(seg.start_s, seg.end_s)
        if num_samples <= 0:
            continue
        for window in clip_recitation_windows(
            start_sample, num_samples, contract, status.word_times
        ):
            end_sample = window.start_sample + window.num_samples
            word_start, word_end = _contained_word_range(
                status.word_times,
                seg.word_start,
                seg.word_end,
                window.start_sample,
                end_sample,
            )
            if word_start == word_end:
                continue
            phoneme_label = seg.slice_words(word_start, word_end)
            feature_frames = feature_frames_for_samples(window.num_samples)
            logit_frames = muaalem_lattice_length(feature_frames)
            if ctc_target_slots(phoneme_label) >= logit_frames:
                continue
            labels.append(
                WindowLabel(
                    clip_audio_filename=status.audio_filename,
                    surah_ayah=status.surah_ayah,
                    reciter_id=status.reciter_id,
                    window_index=len(labels),
                    start_sample=window.start_sample,
                    num_samples=window.num_samples,
                    recitation_start_sample=start_sample,
                    feature_frames=feature_frames,
                    logit_frames=logit_frames,
                    phoneme_label=phoneme_label,
                    word_start=word_start,
                    word_end=word_end,
                    segment_indices=(seg.segment_index,),
                )
            )
    if not labels:
        return [], EXCLUDE_NO_SEGMENT_WINDOW
    return labels, None


def build_segmented_labels(
    segments: list[Segment],
    statuses: list[ClipStatus],
    contract: WindowContract | None = None,
    cap_feature_frames: int = PROVISIONAL_CAP_FEATURE_FRAMES,
    held_out_clips: frozenset[str] = frozenset(),
) -> WindowedLabels:
    """Every eligible clip's segment-scoped windows plus the exclusion-by-reason report.

    Mirrors :func:`training.windowed_labels.build_windowed_labels` (stable
    ``audio_filename`` order, orphan segments raise, same ``held_out_clips`` eval-clip
    exclusion) so the two builds are directly comparable clip for clip — which is the whole
    point of this rung-(1) control.
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
        clip_labels, reason = build_clip_segment_windows(
            by_clip.get(status.audio_filename, []), status, contract, cap_feature_frames
        )
        if reason is not None:
            exclusions.append((status.audio_filename, reason))
        else:
            labels.extend(clip_labels)
    return WindowedLabels(labels=labels, exclusions=exclusions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="scored segment manifest (tadabur.segment_score); its "
                             "'.clip_status.jsonl' sidecar is read alongside.")
    parser.add_argument("--out", type=Path, required=True,
                        help="output JSONL (train + val splits, windowed_labels format).")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out-clips", type=Path, default=None,
                        help="waqf_freeze partition report (JSON); its calibration+test "
                             "clips are excluded from training.")
    parser.add_argument("--allow-eval-clips-in-training", action="store_true",
                        help="build without the partition, knowingly training on the "
                             "#34 eval clips.")
    args = parser.parse_args()

    segments = read_segments(args.manifest)
    statuses = read_clip_status(Path(f"{args.manifest}.clip_status.jsonl"))
    held_out = resolve_held_out_clips(args.held_out_clips, args.allow_eval_clips_in_training)
    built = build_segmented_labels(segments, statuses, held_out_clips=held_out)
    train, val = split_by_reciter(built.labels, args.val_fraction, args.seed)
    assert_no_reciter_leakage(train, val)

    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; the build is append-only.")
    write_labels(args.out, train, "train")
    write_labels(args.out, val, "val")

    print(f"clips kept:  {built.clips_kept}")
    print(f"windows:     {len(train)} train / {len(val)} val")
    for reason, count in sorted(Counter(built.reasons).items()):
        print(f"  excluded {reason}: {count}")


if __name__ == "__main__":
    main()

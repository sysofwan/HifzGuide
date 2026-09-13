"""Clipping detected neighbour-ayah bleed off a rejected clip, and proving the cut helped.

:mod:`tadabur.bleed_detect` says *which decoded phonemes* belong to a neighbour ayah.
This module turns that phoneme-space boundary into a **time** — so the staged audio can
be recut before it reaches Muraja ADR-0016's replay — and then refuses to trust itself:
every recut clip is re-decoded and re-gated, and the cut is kept only if the gate's
verdict actually improved. A bad detection therefore costs **yield, never correctness**,
which is the same safety property ADR-0016 decision 4 rests the excision machinery on.

Three things decide where the cut lands.

* **The timed decode.** :func:`tadabur.waqf_detect.collapse_with_times` gives every
  decoded phoneme an onset at the CTC frame rate, which is a forced alignment for free.
  The detector works in *normalized* phoneme space, so the onsets are mapped through
  ``PhonemeNormalization.offset_map`` — a translation :func:`normalized_onsets` owns.
* **A VAD pause, when one exists.** An ayah boundary is the likeliest place in a
  recitation for a genuine waqf, so a pause usually sits in the window between the
  bleed and the recitation; cutting in the middle of that silence beats cutting in the
  middle of a signal. It is a preference, not a requirement — ADR-0016 already records
  that pause coverage of a *seam* is unmeasured, and this cut does not depend on one.
* **A conservative bias.** Leaving a little bleed is recoverable; cutting into the
  target ayah's first phoneme is not. So every cut is clamped to the window between the
  bleed and this ayah's own recitation, and the fallback (no pause) is a pad short of
  the target rather than a pad into it.

The mirror case — a clip that never finishes its ayah (``uncovered_tail``) — is
**reported, not recut**: there is no bleed to remove, only a covered word range that
should be shortened by whoever asserts over it.

Usage:
  python -m tadabur.bleed_recut --decodes bleed_run/decodes.jsonl
    --clips bleed_run/clips/ --out bleed_run/recuts.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .bleed_detect import BleedDetector, BleedVerdict
from .normalization import cluster_offsets, normalize_phonemes
from .scorer import BALANCED_SCORER, GateResult
from .vad import pauses_from_intervals
from .waqf_detect import EDGE_RECUT_PAD_S, collapse_with_times

# Floor on what a re-cut may leave behind. Purely defensive: the detector needs matched
# phonemes at an edge and a non-empty covered span in the middle, so a real clip cannot
# collapse this far. If one ever does, the honest answer is that no boundary was found —
# not a fraction of a second of audio handed to the model as if it were a recitation.
MIN_RETAINED_S = 0.5


@dataclass(frozen=True)
class RecutSpan:
    """Where a clip's own recitation starts and ends, in clip-relative seconds.

    Shaped to match :class:`~tadabur.clip_status.ClipStatus`'s
    ``recitation_start_s`` / ``recitation_end_s``, which is what the ADR-0016 stager and
    Muraja's transcriber read — a rejected clip simply has no ``ClipStatus``, so the
    fields travel on this record instead. ``pause_anchored`` says whether the cut landed
    inside a VAD silence or fell back to the padded phoneme onset, which is the number
    that tells us how often a real ayah seam carries a detectable waqf.
    """

    start_s: float = 0.0
    end_s: float = 0.0
    leading_clipped_s: float = 0.0
    trailing_clipped_s: float = 0.0
    pause_anchored_leading: bool = False
    pause_anchored_trailing: bool = False

    @property
    def clipped(self) -> bool:
        """Whether the span removes anything at all."""
        return self.leading_clipped_s > 0.0 or self.trailing_clipped_s > 0.0


def normalized_onsets(decode: str, decode_times: list[float]) -> list[float]:
    """Clip-relative onset of each *normalized* phoneme of ``decode``.

    ``decode_times`` is indexed by raw decoded character (one per emitted CTC phoneme,
    diacritics included); normalization groups those characters into clusters and
    clusters into normalized phonemes. Each normalized phoneme takes the onset of the
    first raw character it was built from, so the result is non-decreasing and indexable
    by the query positions :class:`~tadabur.bleed_detect.BleedVerdict` reports.
    """
    starts = cluster_offsets(decode)
    onsets: list[float] = []
    for cluster_start, _ in normalize_phonemes(decode).offset_map:
        if cluster_start < len(starts) and starts[cluster_start] < len(decode_times):
            onsets.append(decode_times[starts[cluster_start]])
        elif onsets:
            onsets.append(onsets[-1])
        else:
            onsets.append(0.0)
    return onsets


def _pause_within(pauses: list[tuple[float, float]], lo: float, hi: float) -> float | None:
    """Midpoint of the first VAD silence lying inside ``(lo, hi)``, or ``None``."""
    for start, end in pauses:
        if start >= lo and end <= hi:
            return (start + end) / 2.0
    return None


def recut_span(
    verdict: BleedVerdict,
    onsets: list[float],
    duration_s: float,
    pauses: list[tuple[float, float]],
    *,
    pad_s: float = EDGE_RECUT_PAD_S,
) -> RecutSpan:
    """The clip-relative recitation span implied by ``verdict``'s detected bleed.

    Returns the whole clip when neither edge fired. The leading cut is clamped never to
    pass this ayah's first decoded phoneme and the trailing cut never to fall before its
    last, so a mis-detection can only leave bleed behind — it cannot eat recitation.
    """
    start, end = 0.0, duration_s
    anchored_leading = anchored_trailing = False
    if not onsets:
        return RecutSpan(start_s=start, end_s=end)

    def onset(index: int) -> float:
        return onsets[min(max(index, 0), len(onsets) - 1)]

    if verdict.leading.detected:
        bleed_end = onset(verdict.leading.query_end - 1)
        target_start = onset(verdict.query_covered_start)
        pause = _pause_within(pauses, bleed_end, target_start)
        anchored_leading = pause is not None
        start = pause if pause is not None else target_start - pad_s
        start = min(max(start, 0.0), target_start)

    if verdict.trailing.detected:
        target_end = onset(verdict.query_covered_end - 1)
        bleed_start = onset(verdict.trailing.query_start)
        pause = _pause_within(pauses, target_end, bleed_start)
        anchored_trailing = pause is not None
        end = pause if pause is not None else bleed_start + pad_s
        end = max(min(end, duration_s), target_end)

    if end - start < MIN_RETAINED_S:  # the two edges met; trust neither
        return RecutSpan(start_s=0.0, end_s=duration_s)
    return RecutSpan(
        start_s=start,
        end_s=end,
        leading_clipped_s=start,
        trailing_clipped_s=duration_s - end,
        pause_anchored_leading=anchored_leading,
        pause_anchored_trailing=anchored_trailing,
    )


@dataclass(frozen=True)
class RecutValidation:
    """The re-gate verdict on a recut clip — the reason a cut is kept or thrown away.

    A recut is accepted only when the gate's ``match_ratio`` **rises** and neither edge
    trim **grows**. The ratio must rise because that is the whole claim being made: the
    bleed was inflating ``query_phoneme_count`` and dragging the score down. The trims
    must not grow because a cut that landed inside the recitation would show up there
    first — the aligner would start trimming real phonemes off the new edge.
    """

    accepted: bool
    reason: str
    match_ratio_before: float
    match_ratio_after: float
    leading_trim_before: int
    leading_trim_after: int
    trailing_trim_before: int
    trailing_trim_after: int
    max_insertion_run_after: int


# Reasons a recut was not kept. ``no_bleed`` is not a failure — it is the detector
# declining to act, which is the common case and is counted separately from a cut that
# was made and then refused.
REASON_ACCEPTED = "accepted"
REASON_NO_BLEED = "no_bleed"
REASON_RATIO_NOT_IMPROVED = "ratio_not_improved"
REASON_TRIM_GREW = "trim_grew"


def validate_recut(before: GateResult, after: GateResult) -> RecutValidation:
    """Judge a recut by re-gating the clipped audio against the same reference."""
    if after.match_ratio <= before.match_ratio:
        reason = REASON_RATIO_NOT_IMPROVED
    elif (
        after.leading_trim > before.leading_trim
        or after.trailing_trim > before.trailing_trim
    ):
        reason = REASON_TRIM_GREW
    else:
        reason = REASON_ACCEPTED
    return RecutValidation(
        accepted=reason == REASON_ACCEPTED,
        reason=reason,
        match_ratio_before=before.match_ratio,
        match_ratio_after=after.match_ratio,
        leading_trim_before=before.leading_trim,
        leading_trim_after=after.leading_trim,
        trailing_trim_before=before.trailing_trim,
        trailing_trim_after=after.trailing_trim,
        max_insertion_run_after=after.max_insertion_run,
    )


@dataclass(frozen=True)
class RecutRecord:
    """One clip's recut outcome, written whether or not the cut was kept.

    ``recitation_start_s`` / ``recitation_end_s`` are the span a consumer should read —
    the recut when it was accepted, and the **whole clip** otherwise, so a reader never
    has to branch on ``accepted`` to get a usable span. ``uncovered_tail`` carries the
    truncation signal alongside, since a clip that never finishes its ayah needs its
    asserted word range shortened rather than its audio cut.
    """

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    duration_s: float
    recitation_start_s: float
    recitation_end_s: float
    leading_bleed: bool
    trailing_bleed: bool
    accepted: bool
    reason: str
    uncovered_head: int
    uncovered_tail: int
    pause_anchored_leading: bool = False
    pause_anchored_trailing: bool = False
    validation: dict | None = None


def write_recut_records(path: Path, records: list[RecutRecord]) -> None:
    """Write recut records as a deterministic, key-sorted JSONL sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in sorted(records, key=lambda r: r.audio_filename):
            f.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def read_recut_records(path: Path) -> list[RecutRecord]:
    """Load every :class:`RecutRecord` from ``path`` in file order."""
    records: list[RecutRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(RecutRecord(**data))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decodes",
        type=Path,
        required=True,
        help="Staged decodes JSONL (class_ids + speech intervals per clip).",
    )
    parser.add_argument(
        "--clips", type=Path, required=True, help="Directory of staged 16 kHz WAVs."
    )
    parser.add_argument("--out", type=Path, required=True, help="Recut records JSONL.")
    parser.add_argument(
        "--recut-clips",
        type=Path,
        help="Write the accepted recut audio here (default: alongside --out).",
    )
    args = parser.parse_args()

    import soundfile as sf
    import torch

    from .audio import TARGET_SAMPLE_RATE
    from .inference import MuaalemPhonemeModel
    from .reference_phonemes import load_reference_phonemes

    references = load_reference_phonemes()
    detector = BleedDetector.of(references)
    clips = [json.loads(line) for line in open(args.decodes, encoding="utf-8")]
    recut_dir = args.recut_clips or args.out.parent / "recut_clips"

    # Pass 1 — decide every cut from the staged decode alone (no model needed).
    planned: list[tuple[dict, BleedVerdict, RecutSpan]] = []
    for clip in clips:
        seconds_per_frame = clip["duration_s"] / max(len(clip["class_ids"]), 1)
        decode, decode_times = collapse_with_times(clip["class_ids"], seconds_per_frame)
        verdict = detector.detect(decode, clip["surah_ayah"])
        span = recut_span(
            verdict,
            normalized_onsets(decode, decode_times),
            clip["duration_s"],
            pauses_from_intervals(
                [(float(a), float(b)) for a, b in clip.get("speech_intervals", [])]
            ),
        )
        planned.append((clip, verdict, span))

    cut = [(clip, verdict, span) for clip, verdict, span in planned if span.clipped]
    print(f"{len(cut)} of {len(planned)} clips carry detected bleed", flush=True)

    # Pass 2 — re-decode the cut audio and re-gate it. One model load for the batch.
    validations: dict[str, RecutValidation] = {}
    if cut:
        recut_dir.mkdir(parents=True, exist_ok=True)
        model = MuaalemPhonemeModel.load(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        for clip, verdict, span in cut:
            waveform, rate = sf.read(args.clips / clip["audio_filename"], dtype="float32")
            lo = int(span.start_s * rate)
            hi = int(span.end_s * rate)
            clipped = waveform[lo:hi]
            reference = references[clip["surah_ayah"]]
            before = BALANCED_SCORER.gate(clip["decode"], reference)
            after = BALANCED_SCORER.gate(
                model.decode(clipped, TARGET_SAMPLE_RATE).phonemes, reference
            )
            validation = validate_recut(before, after)
            validations[clip["audio_filename"]] = validation
            if validation.accepted:
                sf.write(
                    recut_dir / clip["audio_filename"], clipped, rate, subtype="PCM_16"
                )

    records = []
    for clip, verdict, span in planned:
        validation = validations.get(clip["audio_filename"])
        accepted = bool(validation and validation.accepted)
        records.append(
            RecutRecord(
                audio_filename=clip["audio_filename"],
                surah_ayah=clip["surah_ayah"],
                reciter_id=clip["reciter_id"],
                duration_s=clip["duration_s"],
                recitation_start_s=span.start_s if accepted else 0.0,
                recitation_end_s=span.end_s if accepted else clip["duration_s"],
                leading_bleed=verdict.leading.detected,
                trailing_bleed=verdict.trailing.detected,
                accepted=accepted,
                reason=validation.reason if validation else REASON_NO_BLEED,
                uncovered_head=verdict.uncovered_head,
                uncovered_tail=verdict.uncovered_tail,
                pause_anchored_leading=span.pause_anchored_leading,
                pause_anchored_trailing=span.pause_anchored_trailing,
                validation=asdict(validation) if validation else None,
            )
        )
    write_recut_records(args.out, records)

    kept = [r for r in records if r.accepted]
    print(f"kept {len(kept)} recuts, discarded {len(cut) - len(kept)}")
    for record in sorted(records, key=lambda r: r.surah_ayah):
        if record.validation is None:
            continue
        v = record.validation
        print(
            f"  {record.surah_ayah:9s} {v['match_ratio_before']:.3f} -> "
            f"{v['match_ratio_after']:.3f}  trims "
            f"({v['leading_trim_before']},{v['trailing_trim_before']}) -> "
            f"({v['leading_trim_after']},{v['trailing_trim_after']})  {record.reason}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

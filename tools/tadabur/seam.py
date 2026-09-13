"""The re-read seam — where a repeat sits in a decode, and whether a pause marks it.

Muraja ADR-0016 decision 4 builds its second oracle by **excising** the repeated span
from a re-read clip and asserting that the clip and its excised twin reach the same
terminal state. The cut boundaries come from the CTC-timed decode, "snapped to a VAD
pause where one exists" — and *whether one exists* is the thing that decision
deliberately declined to assume.

It declined for a stated reason. The 0.7% spurious / 21% missed figures ADR-0016
constraint 4 quotes are measured on ``waqf_event_fixtures/`` — **waqf boundaries, which
are pauses by definition**. HifzGuide #67 measured the same thing at an *ayah* edge and
found 80% coverage (33 of 41 re-cuts landed inside a VAD silence), again a boundary class
where stopping is what the reciter is doing. A **re-read seam is neither**. The VAD
surfaces only silences >= 300 ms bounded by speech spans >= 700 ms (:mod:`tadabur.vad`),
and a reciter doubling back on a single word may leave none at all. Nothing in the reject
path had ever run a VAD at a seam; the seam was known only in *phoneme* space, as
:func:`~tadabur.smith_waterman.longest_insertion_run`.

So this module is the measurement, and it lands before the machinery that would rest on
it. It holds two things and no policy:

* **Seam geometry.** :func:`insertion_runs` recovers, from an alignment's columns, every
  run of query-only phonemes together with the query *and* reference offsets it sits at —
  which :func:`~tadabur.smith_waterman.longest_insertion_run` does not, because the gate
  only ever needed the length. :func:`repeat_runs` keeps the runs long enough to be the
  re-read the corpus was mined for (:data:`~tadabur.scorer.MAX_INSERTION_RUN`, the gate's
  own bar — a clip clearing it would not be in the reject pile).
* **Pause coverage.** :func:`cut_boundaries` turns a run into the two edges an excision
  would cut at, each carrying the **window** it may move inside without eating a phoneme
  on either side; :func:`boundary_pause` asks whether a VAD silence lies in that window,
  and how far the nearest silence is when none does.

The window is the honest question, not "is the onset inside a silence". A cut can only be
snapped to a pause it can reach — :func:`tadabur.bleed_recut._pause_within` is the same
predicate at an ayah edge — and a silence of the VAD's own minimum length fits between two
consecutive phoneme onsets exactly when the reciter paused between those two phonemes.
Containment is therefore the operational number; the nearest-silence gap is reported
beside it as the descriptive one.

Usage:
  python -m tadabur.seam --decodes bleed_run/decodes.jsonl \
      --rejects reject_run/rejects.jsonl --recuts bleed_run/recuts.jsonl \
      --json bleed_run/seam_pauses.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .scorer import MAX_INSERTION_RUN
from .smith_waterman import AlignmentResult


@dataclass(frozen=True)
class InsertionRun:
    """A run of consecutive query-only phonemes — recitation the reference does not hold.

    ``query_start``/``query_end`` are the half-open range of *normalized decode* indices
    the run spans, and ``ref_position`` the reference offset it is wedged at: the next
    reference phoneme the alignment would have consumed had the reciter not doubled back.
    Those coordinates are what separates this from
    :func:`~tadabur.smith_waterman.longest_insertion_run`, which answers only "how long is
    the longest one" because the gate never needed to know *where*.

    When a reciter says a phrase twice the two utterances score identically, so which one
    is "the repeat" is a tie — and the aligner breaks it by matching the **later** copy,
    leaving the **earlier** one as the run. The oracle does not care (excising either
    leaves one clean pass), but the cut times do: the excised span is the one that *ends*
    at the reciter's hesitation rather than the one that begins after it, so the pause —
    where there is one — falls at the closing edge.
    """

    query_start: int
    query_end: int
    ref_position: int

    @property
    def length(self) -> int:
        """Decoded phonemes the run covers."""
        return self.query_end - self.query_start


def insertion_runs(alignment: AlignmentResult) -> list[InsertionRun]:
    """Every run of query-only columns in ``alignment``, in decode order.

    Walks the complete column sequence — the only view that sees an insertion at all,
    since ``ref_matches``/``ref_to_query`` are indexed by reference position and so omit
    query-only columns — carrying the query and reference cursors alongside it. A space
    breaks a run exactly as it does in
    :func:`~tadabur.smith_waterman.longest_insertion_run` (a word boundary is not an
    inserted phoneme) and still advances the cursor it belongs to.
    """
    runs: list[InsertionRun] = []
    query = alignment.query_start
    reference = alignment.ref_start
    open_start = -1
    for column in alignment.columns:
        if column.ref_char is None and column.query_char != " ":
            if open_start < 0:
                open_start = query
            query += 1
            continue
        if open_start >= 0:
            runs.append(InsertionRun(open_start, query, reference))
            open_start = -1
        if column.ref_char is None:
            query += 1
        elif column.query_char is None:
            reference += 1
        else:
            query += 1
            reference += 1
    if open_start >= 0:
        runs.append(InsertionRun(open_start, query, reference))
    return runs


def repeat_runs(
    alignment: AlignmentResult, min_length: int = MAX_INSERTION_RUN
) -> list[InsertionRun]:
    """The insertion runs long enough to be a re-read, in decode order.

    The default bar is the gate's own :data:`~tadabur.scorer.MAX_INSERTION_RUN`: a clip
    whose longest run falls below it would have passed the gate and would not be in the
    reject pile at all, so the predicate that mined the clip and the predicate that finds
    its seams are the same number. A clip may carry more than one — a reciter can double
    back twice — and every one of them is a repeat to be excised, not just the longest.
    """
    return [run for run in insertion_runs(alignment) if run.length >= min_length]


@dataclass(frozen=True)
class CutBoundary:
    """One edge of an excised span: where the cut is read off, and how far it may move.

    ``onset_s`` is the CTC onset of the phoneme the cut falls before. ``window_start_s``
    / ``window_end_s`` bound the interval the cut may travel in without eating a phoneme
    on either side — from the previous decoded phoneme's onset to this one's. A pause the
    excision could actually snap to is a silence lying wholly inside that window.
    """

    onset_s: float
    window_start_s: float
    window_end_s: float


def cut_boundaries(
    run: InsertionRun, onsets: list[float], duration_s: float
) -> tuple[CutBoundary, CutBoundary]:
    """The two edges an excision of ``run`` would cut at, with their movable windows.

    ``onsets`` is the clip-relative onset of each *normalized* decode phoneme
    (:func:`tadabur.bleed_recut.normalized_onsets`). The closing edge of a run that
    reaches the end of the decode has no following phoneme, so it is pinned to the clip's
    end — excising it removes the clip's tail, which is what the reciter actually left.
    """

    def onset(index: int) -> float:
        if index <= 0:
            return 0.0
        if index >= len(onsets):
            return duration_s
        return onsets[index]

    start = CutBoundary(onset(run.query_start), onset(run.query_start - 1), onset(run.query_start))
    end = CutBoundary(onset(run.query_end), onset(run.query_end - 1), onset(run.query_end))
    return start, end


@dataclass(frozen=True)
class BoundaryPause:
    """Whether a VAD silence sits where a cut boundary could use it, and how near one is.

    ``pause_start_s`` / ``pause_end_s`` describe the silence lying **wholly inside** the
    boundary's window — the one an excision would snap to — and are ``None`` when there
    is none. ``nearest_gap_s`` is the distance from ``onset_s`` to the closest silence
    anywhere in the clip (0.0 when the onset falls inside one), or ``None`` when the clip
    has no interior silence at all; it is descriptive only — reported to 0.1 ms, which is
    already finer than the 40 ms CTC lattice the onset came off — and says how far off
    being anchored an unanchored boundary was.
    """

    pause_start_s: float | None = None
    pause_end_s: float | None = None
    nearest_gap_s: float | None = None

    @property
    def anchored(self) -> bool:
        """Whether a usable silence lies inside the boundary's window."""
        return self.pause_start_s is not None


def boundary_pause(
    pauses: list[tuple[float, float]], boundary: CutBoundary
) -> BoundaryPause:
    """Score one cut boundary against the clip's VAD silences."""
    inside = [
        (start, end)
        for start, end in pauses
        if start >= boundary.window_start_s and end <= boundary.window_end_s
    ]
    nearest = _nearest_gap(pauses, boundary.onset_s)
    if not inside:
        return BoundaryPause(nearest_gap_s=nearest)
    start, end = inside[0]
    return BoundaryPause(pause_start_s=start, pause_end_s=end, nearest_gap_s=nearest)


def _nearest_gap(pauses: list[tuple[float, float]], time_s: float) -> float | None:
    """Distance from ``time_s`` to the nearest silence, 0.0 when it falls inside one."""
    if not pauses:
        return None
    return round(
        min(
            0.0 if start <= time_s <= end else min(abs(time_s - start), abs(time_s - end))
            for start, end in pauses
        ),
        4,
    )


@dataclass(frozen=True)
class SeamCoverage:
    """One re-read seam, and the pause evidence at both edges an excision would cut at.

    One record per **run**, not per clip: a clip with two repeats has two seams, and
    collapsing them would report a clip as covered when only one of its seams was.
    ``covered`` is the pair-level verdict — both edges anchored — because an excision
    splices the two together and a pair is only as clean as its worse edge.
    """

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    run_index: int
    repeat_phonemes: int
    query_start: int
    query_end: int
    ref_position: int
    span_start_s: float
    span_end_s: float
    start: BoundaryPause
    end: BoundaryPause

    @property
    def repeat_seconds(self) -> float:
        """Clip time the repeat occupies, between the two cut onsets."""
        return max(0.0, self.span_end_s - self.span_start_s)

    @property
    def covered(self) -> bool:
        """Whether both cut edges have a silence they could snap to."""
        return self.start.anchored and self.end.anchored


def seam_coverage(
    audio_filename: str,
    surah_ayah: str,
    reciter_id: int,
    alignment: AlignmentResult,
    onsets: list[float],
    duration_s: float,
    pauses: list[tuple[float, float]],
    *,
    min_length: int = MAX_INSERTION_RUN,
) -> list[SeamCoverage]:
    """Measure every re-read seam in one clip against its VAD silences."""
    measured: list[SeamCoverage] = []
    for index, run in enumerate(repeat_runs(alignment, min_length)):
        start, end = cut_boundaries(run, onsets, duration_s)
        measured.append(
            SeamCoverage(
                audio_filename=audio_filename,
                surah_ayah=surah_ayah,
                reciter_id=reciter_id,
                run_index=index,
                repeat_phonemes=run.length,
                query_start=run.query_start,
                query_end=run.query_end,
                ref_position=run.ref_position,
                span_start_s=start.onset_s,
                span_end_s=end.onset_s,
                start=boundary_pause(pauses, start),
                end=boundary_pause(pauses, end),
            )
        )
    return measured


@dataclass(frozen=True)
class SeamPauseYield:
    """Deliverable 0: how often a real re-read seam carries a pause an excision can use.

    ``seams_covered`` is the number that decides how much of ADR-0016 decision 4 rests on
    pause anchoring: a pair is spliced at *both* of its cut edges, so a seam counts as
    covered only when both are anchored. The per-edge counts are reported beside it
    because they are not interchangeable — the edge where the reciter doubled back is a
    hesitation and the edge where he carried on is fluent speech, so a split between them
    is the expected shape, not an anomaly.

    ``unanchored_gap_s`` summarises how far the unanchored edges were from the nearest
    silence anywhere in their clip. A large gap means there was no pause to miss; a small
    one means the VAD's 300/700 ms definition, not the recitation, is what withheld it.
    """

    clips: int
    clips_with_a_seam: int
    clips_with_multiple_seams: int
    clips_without_interior_pause: int
    seams: int
    seams_start_anchored: int
    seams_end_anchored: int
    seams_covered: int
    start_anchored_rate: float
    end_anchored_rate: float
    covered_rate: float
    repeat_phonemes: dict[str, float]
    repeat_seconds: dict[str, float]
    unanchored_gap_s: dict[str, float]


def summarize_seams(seams: list[SeamCoverage], clips: int) -> SeamPauseYield:
    """Reduce per-seam measurements to the report's numbers."""
    by_clip: dict[str, int] = {}
    for seam in seams:
        by_clip[seam.audio_filename] = by_clip.get(seam.audio_filename, 0) + 1
    boundaries = [seam.start for seam in seams] + [seam.end for seam in seams]
    gaps = [
        boundary.nearest_gap_s
        for boundary in boundaries
        if not boundary.anchored and boundary.nearest_gap_s is not None
    ]
    return SeamPauseYield(
        clips=clips,
        clips_with_a_seam=len(by_clip),
        clips_with_multiple_seams=sum(1 for count in by_clip.values() if count > 1),
        clips_without_interior_pause=len(
            {
                seam.audio_filename
                for seam in seams
                if seam.start.nearest_gap_s is None and seam.end.nearest_gap_s is None
            }
        ),
        seams=len(seams),
        seams_start_anchored=sum(1 for seam in seams if seam.start.anchored),
        seams_end_anchored=sum(1 for seam in seams if seam.end.anchored),
        seams_covered=sum(1 for seam in seams if seam.covered),
        start_anchored_rate=_share(sum(1 for s in seams if s.start.anchored), len(seams)),
        end_anchored_rate=_share(sum(1 for s in seams if s.end.anchored), len(seams)),
        covered_rate=_share(sum(1 for s in seams if s.covered), len(seams)),
        repeat_phonemes=_spread([float(seam.repeat_phonemes) for seam in seams]),
        repeat_seconds=_spread([seam.repeat_seconds for seam in seams]),
        unanchored_gap_s=_spread(gaps),
    )


def _share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _spread(values: list[float]) -> dict[str, float]:
    """Min / median / max of ``values``, or an empty dict when there are none."""
    if not values:
        return {}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "min": round(ordered[0], 3),
        "median": round(median, 3),
        "max": round(ordered[-1], 3),
        "n": len(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decodes",
        type=Path,
        required=True,
        help="Staged decodes JSONL (class_ids + speech intervals per clip).",
    )
    parser.add_argument(
        "--rejects", type=Path, required=True, help="Reject sink JSONL (the predicate)."
    )
    parser.add_argument(
        "--recuts",
        type=Path,
        help="Recut records JSONL (#68). Restricts each clip to its recitation span.",
    )
    parser.add_argument("--json", type=Path, help="Write the summary + per-seam rows here.")
    args = parser.parse_args()

    from .bleed_recut import normalized_onsets, read_recut_records
    from .normalization import normalize_phonemes
    from .reference_phonemes import load_reference_phonemes
    from .rejects import read_reject_records
    from .smith_waterman import smith_waterman
    from .vad import pauses_from_intervals
    from .waqf_detect import collapse_with_times

    references = load_reference_phonemes()
    wanted = {
        record.audio_filename
        for record in read_reject_records(args.rejects)
        if record.is_clean_re_read
    }
    recuts = (
        {record.audio_filename: record for record in read_recut_records(args.recuts)}
        if args.recuts
        else {}
    )

    clips = [json.loads(line) for line in open(args.decodes, encoding="utf-8")]
    clips = sorted(
        (clip for clip in clips if clip["audio_filename"] in wanted),
        key=lambda clip: clip["audio_filename"],
    )

    seams: list[SeamCoverage] = []
    outside_span = 0
    for clip in clips:
        duration_s = clip["duration_s"]
        seconds_per_frame = duration_s / max(len(clip["class_ids"]), 1)
        decode, decode_times = collapse_with_times(clip["class_ids"], seconds_per_frame)
        alignment = smith_waterman(
            query=normalize_phonemes(decode).normalized,
            reference=references[clip["surah_ayah"]],
        )
        recut = recuts.get(clip["audio_filename"])
        lo = recut.recitation_start_s if recut else 0.0
        hi = recut.recitation_end_s if recut else duration_s
        pauses = [
            pause
            for pause in pauses_from_intervals(
                [(float(a), float(b)) for a, b in clip.get("speech_intervals", [])]
            )
            if pause[0] >= lo and pause[1] <= hi
        ]
        measured = seam_coverage(
            clip["audio_filename"],
            clip["surah_ayah"],
            clip["reciter_id"],
            alignment,
            normalized_onsets(decode, decode_times),
            duration_s,
            pauses,
        )
        outside_span += sum(
            1 for seam in measured if seam.span_start_s < lo or seam.span_end_s > hi
        )
        seams.extend(measured)

    result = summarize_seams(seams, len(clips))
    print(
        f"{result.seams} seams in {result.clips_with_a_seam} of {result.clips} clips; "
        f"{result.seams_covered} covered at both edges "
        f"({result.covered_rate:.0%}), {result.seams_start_anchored} at the opening "
        f"edge, {result.seams_end_anchored} at the closing edge"
    )
    print(f"{outside_span} seams fall outside their clip's recitation span")
    for seam in seams:
        print(
            f"  {seam.surah_ayah:9s} {seam.audio_filename[:28]:28s} "
            f"run {seam.repeat_phonemes:3d} ph / {seam.repeat_seconds:5.2f} s  "
            f"[{seam.span_start_s:6.2f},{seam.span_end_s:6.2f}]  "
            f"open {'pause' if seam.start.anchored else f'{seam.start.nearest_gap_s}'}  "
            f"close {'pause' if seam.end.anchored else f'{seam.end.nearest_gap_s}'}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": asdict(result),
                    "seams_outside_recitation_span": outside_span,
                    "seams": [asdict(seam) for seam in seams],
                },
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

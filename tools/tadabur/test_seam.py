"""Tests for re-read seam geometry and its pause coverage.

Two halves, deliberately different in kind. The run-finding tests go through the **real**
Smith-Waterman on invented "ayat" of distinct bare consonants (so normalization leaves
them alone and the alignment is the only thing under test) — the coordinates only mean
something if they come out of the same aligner the gate uses. The pause tests are built
by hand from onsets and intervals, because a real clip would not let the anchored and
unanchored branches be exercised separately.
"""

from __future__ import annotations

from tadabur.normalization import normalize_phonemes
from tadabur.scorer import MAX_INSERTION_RUN
from tadabur.seam import (
    CutBoundary,
    InsertionRun,
    SeamCoverage,
    boundary_pause,
    cut_boundaries,
    insertion_runs,
    repeat_runs,
    seam_coverage,
    summarize_seams,
)
from tadabur.smith_waterman import longest_insertion_run, smith_waterman

# Twenty distinct bare consonants. Nothing here collapses, so a decode's phoneme indices
# are its character indices and a repeat is exactly as long as it reads.
REFERENCE = "بتثجحخدذرزسشصضطظعغفق"


def align(decode: str):
    return smith_waterman(query=normalize_phonemes(decode).normalized, reference=REFERENCE)


# --- seam geometry -----------------------------------------------------------------


def test_a_repeated_phrase_is_one_run_at_the_reference_position_it_doubled_back_to():
    # The reciter reaches reference position 10, goes back and says 5..10 again, then
    # carries on. Which of the two identical utterances is "the repeat" is a tie, and the
    # aligner breaks it by matching the LATER copy — so the run covers query 5..10, the
    # first utterance, and sits at reference position 5. Pinned because the cut times
    # depend on it: the excised span is the one that ends at the reciter's hesitation,
    # not the one that begins after it.
    alignment = align(REFERENCE[:10] + REFERENCE[5:10] + REFERENCE[10:])

    assert insertion_runs(alignment) == [InsertionRun(5, 10, 5)]


def test_run_lengths_agree_with_the_gate_s_own_measure():
    # The gate reports only a length; this module adds coordinates. If the two ever
    # disagreed, the clips mined by one would not be the clips excised by the other.
    for decode in (
        REFERENCE[:10] + REFERENCE[5:10] + REFERENCE[10:],
        REFERENCE[:8] + REFERENCE[2:8] + REFERENCE[8:],
        REFERENCE,
    ):
        alignment = align(decode)
        runs = insertion_runs(alignment)
        longest = max((run.length for run in runs), default=0)
        assert longest == longest_insertion_run(alignment.columns)


def test_a_clip_with_two_re_reads_yields_two_runs_in_decode_order():
    # Both repeats interior: a repeat at either clip edge is trimmed by the local
    # aligner and shows up as leading/trailing_trim, never as an insertion run.
    decode = (
        REFERENCE[:8] + REFERENCE[2:8] + REFERENCE[8:16] + REFERENCE[10:16] + REFERENCE[16:]
    )
    runs = repeat_runs(align(decode))

    assert [run.length for run in runs] == [6, 6]
    assert runs[0].query_start < runs[1].query_start
    assert runs[0].ref_position < runs[1].ref_position


def test_a_clean_recitation_has_no_runs_at_all():
    assert insertion_runs(align(REFERENCE)) == []


def test_repeat_runs_keeps_only_runs_the_gate_would_have_rejected_on():
    # One phoneme below the gate's bar is a stumble, not a re-read: such a clip would
    # have passed the gate and could not be in the reject pile being mined.
    short = REFERENCE[:10] + REFERENCE[10 - (MAX_INSERTION_RUN - 1) : 10] + REFERENCE[10:]

    assert insertion_runs(align(short)) != []
    assert repeat_runs(align(short)) == []


# --- cut boundaries ----------------------------------------------------------------

# Ten decoded phonemes, one per second, in a twelve-second clip.
ONSETS = [float(i) for i in range(10)]
DURATION = 12.0


def test_a_cut_lands_on_the_onsets_bounding_the_repeat():
    start, end = cut_boundaries(InsertionRun(3, 7, 3), ONSETS, DURATION)

    assert (start.onset_s, end.onset_s) == (3.0, 7.0)
    # Each edge may travel back only as far as the previous phoneme's onset.
    assert (start.window_start_s, start.window_end_s) == (2.0, 3.0)
    assert (end.window_start_s, end.window_end_s) == (6.0, 7.0)


def test_a_repeat_running_to_the_end_of_the_decode_cuts_to_the_clip_end():
    # Nothing follows the last decoded phoneme, so the closing edge is the clip's end —
    # excising it removes the tail the reciter actually left there.
    _, end = cut_boundaries(InsertionRun(7, 10, 20), ONSETS, DURATION)

    assert end.onset_s == DURATION
    assert (end.window_start_s, end.window_end_s) == (9.0, DURATION)


def test_a_repeat_opening_the_decode_cuts_from_the_clip_start():
    start, _ = cut_boundaries(InsertionRun(0, 5, 0), ONSETS, DURATION)

    assert start.onset_s == 0.0
    assert (start.window_start_s, start.window_end_s) == (0.0, 0.0)


# --- pause coverage ----------------------------------------------------------------

BOUNDARY = CutBoundary(onset_s=3.0, window_start_s=2.0, window_end_s=3.0)


def test_a_silence_inside_the_window_anchors_the_boundary():
    measured = boundary_pause([(2.2, 2.8)], BOUNDARY)

    assert measured.anchored
    assert (measured.pause_start_s, measured.pause_end_s) == (2.2, 2.8)
    assert measured.nearest_gap_s == 0.2


def test_a_silence_overhanging_the_window_is_not_one_the_cut_could_reach():
    # It is near — 0.1 s away — but a cut snapped to it would eat the phoneme before the
    # repeat, so it does not anchor. The gap is still reported, which is the point: this
    # is the case where the VAD's definition, not the recitation, withheld the anchor.
    measured = boundary_pause([(1.5, 2.9)], BOUNDARY)

    assert not measured.anchored
    assert measured.nearest_gap_s == 0.1


def test_a_clip_with_no_interior_silence_reports_no_gap_rather_than_a_large_one():
    # None and "very far away" are different claims, and only one of them is true here.
    assert boundary_pause([], BOUNDARY).nearest_gap_s is None


def test_an_onset_inside_a_silence_has_a_zero_gap():
    assert boundary_pause([(2.5, 3.5)], BOUNDARY).nearest_gap_s == 0.0


def test_a_seam_is_covered_only_when_both_of_its_edges_are_anchored():
    # An excision splices the two edges together, so the pair is only as clean as its
    # worse edge. Here the opening edge has a pause and the closing edge does not.
    seams = seam_coverage(
        "a.wav", "2:2", 7,
        align(REFERENCE[:10] + REFERENCE[5:10] + REFERENCE[10:]),
        ONSETS + [10.0, 11.0, 12.0, 13.0, 14.0],
        DURATION,
        [(4.3, 4.8)],
    )

    assert len(seams) == 1
    assert seams[0].start.anchored
    assert not seams[0].end.anchored
    assert not seams[0].covered


# --- the summary -------------------------------------------------------------------


def _seam(name: str, start: bool, end: bool, phonemes: int = 8) -> SeamCoverage:
    from tadabur.seam import BoundaryPause

    anchored = BoundaryPause(pause_start_s=1.0, pause_end_s=1.2, nearest_gap_s=0.0)
    missing = BoundaryPause(nearest_gap_s=0.9)
    return SeamCoverage(
        audio_filename=name,
        surah_ayah="2:2",
        reciter_id=1,
        run_index=0,
        repeat_phonemes=phonemes,
        query_start=10,
        query_end=10 + phonemes,
        ref_position=10,
        span_start_s=1.0,
        span_end_s=1.0 + phonemes / 10.0,
        start=anchored if start else missing,
        end=anchored if end else missing,
    )


def test_coverage_is_counted_per_seam_and_per_edge():
    result = summarize_seams(
        [_seam("a.wav", True, True), _seam("b.wav", True, False)], clips=3
    )

    assert (result.clips, result.clips_with_a_seam, result.seams) == (3, 2, 2)
    assert result.seams_covered == 1
    assert (result.seams_start_anchored, result.seams_end_anchored) == (2, 1)
    assert result.covered_rate == 0.5


def test_a_clip_with_two_seams_is_counted_once_and_its_seams_twice():
    result = summarize_seams(
        [_seam("a.wav", True, True), _seam("a.wav", False, False)], clips=1
    )

    assert (result.clips_with_a_seam, result.clips_with_multiple_seams) == (1, 1)
    assert result.seams == 2


def test_only_unanchored_edges_enter_the_gap_spread():
    # An anchored edge's gap is 0.0 by construction; counting it would report a coverage
    # figure twice, once dressed as a distance.
    result = summarize_seams([_seam("a.wav", True, False)], clips=1)

    assert result.unanchored_gap_s == {"min": 0.9, "median": 0.9, "max": 0.9, "n": 1}


def test_an_empty_measurement_reports_zero_rather_than_dividing_by_zero():
    result = summarize_seams([], clips=4)

    assert result.seams == 0
    assert result.covered_rate == 0.0
    assert result.repeat_phonemes == {}

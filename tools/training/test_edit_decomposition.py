"""Tests for the edit decomposition.

The decomposition exists to decide which objective arm is worth 4 GPU-hours, so its failure
mode is not a crash -- it is a plausible-looking region breakdown that points at the wrong
arm. These pin the two things that could produce one: the alignment must charge exactly the
edits the metric charges, and the attribution must not smuggle in an answer the tie-break
chose.
"""

from __future__ import annotations

import pytest

from training.distill_eval import CONFIRM_TIMESTEPS, Emission, levenshtein
from training import edit_decomposition as ed


def emission(token: int, start: int = 0, end: int = 0, window: int = 0, final: bool = False):
    return Emission(
        token_id=token,
        window=window,
        start_step=start,
        end_step=end,
        is_final_window=final,
    )


def stream(tokens, **kwargs):
    return [emission(t, **kwargs) for t in tokens]


# --- alignment ----------------------------------------------------------------------


@pytest.mark.parametrize("tie_break", ed.TIE_BREAKS)
@pytest.mark.parametrize(
    "teacher,student",
    [
        ([], []),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], []),
        ([], [1, 2]),
        ([1, 2, 3], [1, 9, 3]),
        ([1, 2, 3], [1, 2, 2, 3]),
        ([1, 2, 2, 3], [1, 2, 3]),
        ([5, 5, 5], [5]),
        ([1, 2, 3, 4, 5], [2, 3, 9, 5, 7]),
    ],
)
def test_alignment_charges_exactly_the_metric_s_edits(teacher, student, tie_break):
    """The decomposition must not disagree with the number the headline metric reports."""
    ops = ed.align(teacher, student, tie_break)
    charged = sum(1 for kind, _, _ in ops if kind != "match")
    assert charged == levenshtein(teacher, student)


@pytest.mark.parametrize("tie_break", ed.TIE_BREAKS)
def test_alignment_consumes_both_streams_exactly_once(tie_break):
    teacher, student = [1, 2, 3, 4], [1, 9, 4, 4]
    ops = ed.align(teacher, student, tie_break)
    assert [i for _, i, _ in ops if i is not None] == list(range(len(teacher)))
    assert [j for _, _, j in ops if j is not None] == list(range(len(student)))


def test_identical_streams_are_all_matches():
    ops = ed.align([7, 7, 3], [7, 7, 3])
    assert {kind for kind, _, _ in ops} == {"match"}


def test_an_unknown_tie_break_is_refused():
    with pytest.raises(ValueError, match="tie_break"):
        ed.align([1], [1], "whatever")


# --- attribution --------------------------------------------------------------------


def test_a_substitution_is_charged_to_its_own_emission():
    emissions = stream([1, 2, 3])
    edits = ed.decompose(emissions, [1, 9, 3])
    assert [e.kind for e in edits] == ["substitution"]
    assert edits[0].emission is emissions[1]
    assert (edits[0].teacher_token, edits[0].student_token) == (2, 9)


def test_a_deletion_is_charged_to_the_emission_that_went_missing():
    emissions = stream([1, 2, 3])
    edits = ed.decompose(emissions, [1, 3])
    assert [e.kind for e in edits] == ["deletion"]
    assert edits[0].emission is emissions[1]
    assert edits[0].student_token is None


def test_an_insertion_is_charged_to_the_emission_it_sits_against():
    emissions = stream([1, 2, 3])
    edits = ed.decompose(emissions, [1, 2, 9, 3], tie_break="diagonal_first")
    assert [e.kind for e in edits] == ["insertion"]
    # The inserted 9 sits before teacher token 3, which is emissions[2].
    assert edits[0].emission is emissions[2]
    assert edits[0].student_token == 9


def test_a_trailing_insertion_falls_back_to_the_last_emission():
    """There is no emission after the end of the stream to attribute it to."""
    emissions = stream([1, 2])
    edits = ed.decompose(emissions, [1, 2, 9])
    assert [e.kind for e in edits] == ["insertion"]
    assert edits[0].emission is emissions[-1]


def test_no_emissions_charges_nothing():
    assert ed.decompose([], [1, 2, 3]) == []


def test_a_split_run_reads_as_an_adjacent_duplicate():
    """The student breaking one teacher run into two is the split/merge shape."""
    emissions = stream([4, 7, 4])
    edits = ed.decompose(emissions, [4, 7, 7, 4])
    assert len(edits) == 1
    assert edits[0].kind == "insertion"
    assert edits[0].is_adjacent_duplicate


def test_a_plain_confusion_is_not_an_adjacent_duplicate():
    edits = ed.decompose(stream([4, 7, 4]), [4, 9, 4])
    assert [e.is_adjacent_duplicate for e in edits] == [False]


# --- provenance ---------------------------------------------------------------------


def test_only_the_final_window_past_the_confirm_split_counts_as_flush():
    assert not emission(1, 40, 44, final=False).is_flush  # never committed in deployment
    assert not emission(1, 0, 4, final=True).is_flush     # the final window's own first second
    assert emission(1, 40, 44, final=True).is_flush


def test_a_run_crossing_the_confirm_boundary_is_a_seam():
    assert emission(1, CONFIRM_TIMESTEPS - 1, CONFIRM_TIMESTEPS).straddles_seam
    assert not emission(1, 0, CONFIRM_TIMESTEPS - 1).straddles_seam
    assert not emission(1, CONFIRM_TIMESTEPS, CONFIRM_TIMESTEPS + 3).straddles_seam


# --- summary ------------------------------------------------------------------------


def test_region_rates_use_their_own_denominators():
    """The decision is a rate comparison, so a region's share must not stand in for it."""
    committed = stream([1, 2, 3, 4], start=0, end=4)
    flush = stream([5, 6], start=40, end=44, final=True)
    emissions = committed + flush
    # One edit in each region, but the flush region has half the characters.
    report = ed.summarise([(emissions, [1, 9, 3, 4, 5, 9])])
    assert report["region"]["committed"]["reference_characters"] == 4
    assert report["region"]["flush"]["reference_characters"] == 2
    assert report["region"]["committed"]["error_rate"] == pytest.approx(0.25)
    assert report["region"]["flush"]["error_rate"] == pytest.approx(0.5)


def test_the_summary_edit_count_matches_the_metric():
    emissions = stream([1, 2, 3, 4, 5])
    student = [1, 3, 3, 9, 5, 5]
    report = ed.summarise([(emissions, student)])
    assert report["edits"] == levenshtein([e.token_id for e in emissions], student)
    assert report["reference_characters"] == 5


def test_the_summary_pools_over_clips():
    a = (stream([1, 2]), [1, 9])
    b = (stream([3, 4, 5]), [3, 4, 5])
    report = ed.summarise([a, b])
    assert report["reference_characters"] == 5
    assert report["edits"] == 1


def test_kind_counts_sum_to_the_edit_total():
    emissions = stream([1, 2, 3, 4])
    report = ed.summarise([(emissions, [1, 9, 9, 4, 7])])
    assert sum(report["kinds"].values()) == report["edits"]


def test_an_empty_region_reports_a_zero_rate_not_a_division_error():
    report = ed.summarise([(stream([1, 2]), [1, 2])])
    assert report["region"]["flush"]["reference_characters"] == 0
    assert report["region"]["flush"]["error_rate"] == 0.0


# --- tie-break sensitivity ----------------------------------------------------------


def test_both_tie_breaks_are_reported_and_agree_on_the_edit_count():
    """The count is a property of the streams; only its attribution is a choice."""
    per_clip = [(stream([1, 2, 2, 3]), [1, 2, 3, 3])]
    report = ed.tie_break_spread(per_clip)
    counts = {tb: report["by_tie_break"][tb]["edits"] for tb in ed.TIE_BREAKS}
    assert len(set(counts.values())) == 1


def test_the_spread_exposes_an_attribution_that_depends_on_the_tie_break():
    """A duplicate spanning the region boundary is attributed differently by each rule.

    This is the case the spread exists to catch: the same edit lands in `committed` under
    one optimal alignment and in `flush` under the other, so a region conclusion drawn from
    one backtrace alone would be an artefact.
    """
    emissions = [
        emission(7, start=0, end=4),
        emission(7, start=40, end=44, final=True),
    ]
    per_clip = [(emissions, [7])]
    report = ed.tie_break_spread(per_clip)
    attributions = {
        tb: report["by_tie_break"][tb]["region"]["flush"]["edits"] for tb in ed.TIE_BREAKS
    }
    assert set(attributions.values()) == {0, 1}
    assert report["max_abs_difference"]["flush_error_rate"] > 0

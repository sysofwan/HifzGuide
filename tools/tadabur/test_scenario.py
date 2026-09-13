"""Tests for the scenario manifest — the cross-repo interface Muraja consumes.

The word-range tests use an invented three-word ayah whose phonemes are distinct bare
consonants, so the only thing under test is the offset arithmetic. The round-trip test is
the one that matters most to the other repo: a record that does not survive its own
reader is a broken interface however well the arithmetic works.
"""

from __future__ import annotations

import json

from tadabur.excision import cut_time
from tadabur.normalization import ALGORITHM_VERSION
from tadabur.scenario import (
    AUDIO_DIR,
    EARLY_START_TRIM,
    EXCISED_DIR,
    SCHEMA_VERSION,
    ScenarioRecord,
    Seam,
    build_seams,
    covered_word_range,
    read_scenario_records,
    unusable_reason,
    verify_bundle,
    warrants_early_start,
    write_scenario_records,
)
from tadabur.seam import BoundaryPause, SeamCoverage

# Three words of three phonemes each, space-separated, exactly as a normalized reference
# is shaped. Word j owns offsets [OFFSETS[j], OFFSETS[j + 1]) — which includes the
# separating space, because that is what the phonetizer's char mappings give.
REFERENCE = "بتث جحخ دذر"
OFFSETS = [0, 4, 8, 11]


# --- the covered word range --------------------------------------------------------


def test_a_decode_covering_the_whole_ayah_covers_every_word():
    assert covered_word_range(REFERENCE, OFFSETS, 0, len(REFERENCE)) == (0, 3)


def test_a_word_is_covered_when_its_phonemes_are_even_though_its_space_is_not():
    # The span stops at position 7, the space after the second word. Requiring the space
    # to be covered would drop that word — and the query has no spaces to cover it with.
    assert covered_word_range(REFERENCE, OFFSETS, 0, 7) == (0, 2)


def test_a_word_the_decode_only_reached_halfway_into_is_not_covered():
    # A word the oracle cannot grade must not be in the range it asserts over; admitting
    # it would manufacture the false-skip class the corpus exists to measure.
    assert covered_word_range(REFERENCE, OFFSETS, 0, 6) == (0, 1)


def test_a_decode_that_began_late_drops_the_word_it_began_inside():
    assert covered_word_range(REFERENCE, OFFSETS, 1, len(REFERENCE)) == (1, 3)


def test_a_span_covering_no_whole_word_is_empty_rather_than_a_guess():
    assert covered_word_range(REFERENCE, OFFSETS, 1, 3) == (0, 0)


def test_an_empty_span_covers_nothing():
    assert covered_word_range(REFERENCE, OFFSETS, 0, 0) == (0, 0)


# --- the early-start flag ----------------------------------------------------------


def test_early_start_fires_only_at_the_trim_bar():
    assert not warrants_early_start(EARLY_START_TRIM - 1)
    assert warrants_early_start(EARLY_START_TRIM)


# --- seams on the record -----------------------------------------------------------


def _coverage(*, anchored: bool) -> SeamCoverage:
    pause = BoundaryPause(pause_start_s=2.0, pause_end_s=2.4, nearest_gap_s=0.0)
    return SeamCoverage(
        audio_filename="a.wav",
        surah_ayah="2:2",
        reciter_id=3,
        run_index=0,
        repeat_phonemes=9,
        query_start=11,
        query_end=20,
        ref_position=11,
        span_start_s=1.5,
        span_end_s=4.0,
        start=pause if anchored else BoundaryPause(nearest_gap_s=0.8),
        end=BoundaryPause(nearest_gap_s=2.5),
    )


def test_a_seam_carries_its_phoneme_coordinates_and_the_time_a_cut_would_land():
    (seam,) = build_seams([_coverage(anchored=True)])

    assert (seam.query_start, seam.query_end, seam.phonemes) == (11, 20, 9)
    assert seam.ref_position == 11
    assert seam.start_s == cut_time(_coverage(anchored=True).start, 1.5) == 2.2
    assert seam.end_s == 4.0
    assert seam.pause_anchored_start and not seam.pause_anchored_end


def test_seams_are_reported_even_where_no_cut_was_anchored():
    # A record that listed seams only for anchored or surviving cuts would make the
    # corpus look like it held fewer re-reads than it does.
    (seam,) = build_seams([_coverage(anchored=False)])

    assert seam.start_s == 1.5
    assert not seam.pause_anchored_start


# --- the manifest ------------------------------------------------------------------


def _record(clip_id: str, **overrides) -> ScenarioRecord:
    fields = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "audio": f"{AUDIO_DIR}/{clip_id}.wav",
        "surah_ayah": "2:2",
        "reciter_id": 3,
        "duration_s": 12.5,
        "match_ratio": 0.71,
        "max_insertion_run": 9,
        "leading_trim": 0,
        "trailing_trim": 0,
        "added_shadda": False,
        "predicted_phonemes": "بتثجحخ",
        "word_start": 0,
        "word_end": 3,
        "ref_covered_start": 0,
        "ref_covered_end": 11,
        "ref_length": 11,
        "uncovered_head": 0,
        "uncovered_tail": 0,
        "early_start": False,
        "recut_applied": True,
        "recitation_start_s": 0.0,
        "recitation_end_s": 12.5,
        "seams": (Seam(11, 20, 11, 9, 1.5, 4.0, False, False),),
    }
    return ScenarioRecord(**{**fields, **overrides})


def test_the_schema_version_is_tied_to_the_normalization_it_indexes_into():
    # Every phoneme offset in a record is an index into a normalized string, so a
    # normalization change moves them even when this module does not.
    assert SCHEMA_VERSION == f"v1+norm{ALGORITHM_VERSION}"


def test_a_manifest_survives_its_own_reader(tmp_path):
    records = [
        _record("b_clip"),
        _record(
            "a_clip",
            excised_audio=f"{EXCISED_DIR}/a_clip.wav",
            excised_duration_s=9.8,
            excised_phonemes="بتث",
            excision={"accepted": True, "reason": "accepted"},
        ),
    ]
    path = tmp_path / "scenario.jsonl"
    write_scenario_records(path, records)

    loaded = read_scenario_records(path)

    assert [record.clip_id for record in loaded] == ["a_clip", "b_clip"]
    assert loaded[0].has_pair and not loaded[1].has_pair
    assert loaded[0].seams[0].phonemes == 9
    assert loaded[1].words_covered == 3


def test_the_manifest_is_byte_identical_however_the_records_arrive(tmp_path):
    # The acceptance criterion behind a reproducible --limit run: order in, order out.
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    write_scenario_records(first, [_record("a"), _record("b")])
    write_scenario_records(second, [_record("b"), _record("a")])

    assert first.read_bytes() == second.read_bytes()


def test_every_record_names_its_own_audio_relative_to_the_manifest(tmp_path):
    # The bundle moves to the Mac as one directory, so a path that resolved against the
    # run's own layout would not survive the trip.
    path = tmp_path / "scenario.jsonl"
    write_scenario_records(path, [_record("a_clip")])

    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert line["audio"] == "audio/a_clip.wav"
    assert not line["audio"].startswith("/")


# --- verifying a shipped bundle ----------------------------------------------------


def _bundle(tmp_path, *records: ScenarioRecord):
    """Write a bundle with every record's audio actually present."""
    write_scenario_records(tmp_path / "scenario.jsonl", list(records))
    for record in records:
        for relative in (record.audio, record.excised_audio):
            if relative and not relative.startswith("/") and ".." not in relative:
                path = tmp_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
    return tmp_path


def test_an_intact_bundle_reports_nothing(tmp_path):
    bundle = _bundle(
        tmp_path,
        _record("a", excised_audio=f"{EXCISED_DIR}/a.wav", excised_duration_s=9.0),
        _record("b"),
    )

    assert verify_bundle(bundle) == []


def test_a_bundle_that_never_arrived_says_so_rather_than_reporting_nothing_wrong(tmp_path):
    assert verify_bundle(tmp_path) == [f"missing {tmp_path / 'scenario.jsonl'}"]


def test_a_truncated_transfer_names_the_clip_whose_audio_is_missing(tmp_path):
    write_scenario_records(tmp_path / "scenario.jsonl", [_record("a")])

    assert verify_bundle(tmp_path) == ["a: audio is missing (audio/a.wav)"]


def test_a_kept_pair_missing_its_control_clip_is_a_problem_too(tmp_path):
    bundle = _bundle(tmp_path, _record("a"))
    write_scenario_records(
        bundle / "scenario.jsonl",
        [_record("a", excised_audio=f"{EXCISED_DIR}/a.wav", excised_duration_s=9.0)],
    )

    assert verify_bundle(bundle) == ["a: excised_audio is missing (excised/a.wav)"]


def test_a_path_pointing_outside_the_bundle_is_refused(tmp_path):
    write_scenario_records(tmp_path / "scenario.jsonl", [_record("a", audio="../elsewhere/a.wav")])

    assert verify_bundle(tmp_path) == ["a: audio escapes the bundle (../elsewhere/a.wav)"]


def test_a_record_from_an_older_normalization_is_flagged(tmp_path):
    # Every phoneme offset in the record indexes into a normalized string, so a bundle
    # built under a different algorithm describes different positions.
    bundle = _bundle(tmp_path, _record("a", schema_version="v1+norm1"))

    assert verify_bundle(bundle) == [f"a: schema v1+norm1, expected {SCHEMA_VERSION}"]


def test_a_record_the_oracle_could_not_use_is_caught_here_not_in_the_scoreboard(tmp_path):
    bundle = _bundle(tmp_path, _record("a", word_start=3, word_end=3, seams=()))

    assert verify_bundle(bundle) == ["a: covers no words", "a: carries no re-read seam"]


# --- what a re-decode can cost a selected clip ---------------------------------------
#
# The stager applies these before writing a row; verify_bundle applies the same two
# checks on the Mac after the transfer. The test above
# (`..._caught_here_not_in_the_scoreboard`) pins the second half.


def test_a_clip_that_lost_its_repeat_on_the_re_decode_is_dropped_not_staged():
    # Selection reads the sink's decode of the *source* clip; the staged decode is of
    # re-cut audio and can disagree. With no seam there is no re-read left to resolve.
    assert unusable_reason([], 0, 5) == "no_seam"


def test_a_clip_covering_no_whole_word_is_dropped_even_though_it_has_a_seam():
    # Decision 1 asserts only over words wholly inside the alignment span. An empty
    # range is an oracle with nothing to say, which scores as a silent zero if shipped.
    assert unusable_reason([_coverage(anchored=True)], 7, 7) == "no_words"
    assert unusable_reason([_coverage(anchored=True)], 7, 6) == "no_words"


def test_a_clip_with_a_seam_and_covered_words_is_staged():
    assert unusable_reason([_coverage(anchored=False)], 0, 5) is None

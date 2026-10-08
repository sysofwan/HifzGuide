"""Tests for converting the adjudicated waqf boundaries into truth sites (#79).

The conversion logic runs on synthetic boundaries with a fake realizer whose "Uthmani
words" are already phoneme strings, so no phonetizer is needed. The Hafs realizer and the
committed file's regeneration are checked against quran-transcript where it is installed.
"""

from __future__ import annotations

import json

import pytest

from tadabur.truth_sites import CONSONANTS, load_truth_sites
from tadabur.waqf_truth_sites import (
    EXCLUDED_CLOSURE_AMBIGUOUS,
    EXCLUDED_CLOSURE_MERGED,
    EXCLUDED_FINAL_LETTER_ASSIMILATED,
    EXCLUDED_INCONSISTENT_CLIP,
    EXCLUDED_MID_WORD_CLOSURE,
    EXCLUDED_PHONETIZER_UNSUPPORTED,
    FIXTURE_PATHS,
    MID_WORD_CLOSURE,
    SITES_PATH,
    SUMMARY_PATH,
    WAQF,
    WASL,
    Boundary,
    ClipExcluded,
    Edge,
    RealizedRun,
    RecitedRun,
    clip_edges,
    convert,
    final_mark,
    pausal_taa_marbuta,
    read_boundaries,
    recited_runs,
    summary_table,
)

_HARAKAT = "َُِ"
#: Fake words: a word the fake realizer treats as assimilated into the next one, and a
#: word it cannot realize in waqf form.
_ASSIMILATED = "مِن"
_UNREALIZABLE = "شَيء"


def _fake_realize(words: list[str]) -> RealizedRun:
    """Words are phoneme strings already; waqf drops the last word's final haraka."""
    if words[-1] == _UNREALIZABLE:
        raise ClipExcluded(EXCLUDED_PHONETIZER_UNSUPPORTED)
    realized = words[:-1] + [words[-1].rstrip(_HARAKAT)]
    ends: list[int | None] = []
    offset = 0
    for word in realized:
        last = max(i for i, c in enumerate(word) if c in CONSONANTS)
        ends.append(None if word == _ASSIMILATED else offset + last)
        offset += len(word) + 1
    return RealizedRun(" ".join(realized), tuple(ends))


def _boundary(clip: str, index: int, word: int, verdict: str, predicted: str | None = None,
              surah_ayah: str = "1:1") -> Boundary:
    return Boundary(clip, surah_ayah, index, word, predicted or verdict, verdict)


# --- word edges ----------------------------------------------------------------------


def test_regular_rows_are_edges_in_time_order():
    rows = [_boundary("c", 1, 1, WAQF), _boundary("c", 0, 0, WASL)]
    edges, excluded = clip_edges(rows)
    assert edges == [Edge(0, 0, False), Edge(1, 1, True)]
    assert excluded == {}


def test_a_mid_word_closure_verdict_is_excluded():
    rows = [
        _boundary("c", 0, 0, WASL),
        _boundary("c", 1, 1, MID_WORD_CLOSURE, predicted=MID_WORD_CLOSURE),
        _boundary("c", 2, 1, WASL),
    ]
    edges, excluded = clip_edges(rows)
    assert edges == [Edge(0, 0, False), Edge(2, 1, False)]
    assert excluded == {1: EXCLUDED_MID_WORD_CLOSURE}


def test_a_closure_called_wasl_merges_into_the_edge_beside_it():
    rows = [
        _boundary("c", 0, 0, WASL),
        _boundary("c", 1, 0, WASL, predicted=MID_WORD_CLOSURE),
        _boundary("c", 2, 1, WASL),
    ]
    edges, excluded = clip_edges(rows)
    assert edges == [Edge(0, 0, False), Edge(2, 1, False)]
    assert excluded == {1: EXCLUDED_CLOSURE_MERGED}


def test_a_closure_called_waqf_turns_the_edge_beside_it_into_a_pause():
    rows = [
        _boundary("c", 0, 0, WASL),
        _boundary("c", 1, 1, WAQF, predicted=MID_WORD_CLOSURE),
        _boundary("c", 2, 1, WASL),
    ]
    edges, excluded = clip_edges(rows)
    assert edges == [Edge(0, 0, False), Edge(2, 1, True)]
    assert excluded == {1: EXCLUDED_CLOSURE_MERGED}


def test_a_closure_merges_into_the_edge_after_its_word_wherever_that_sits_in_time():
    # Regression (spk0202_S1_A154 #6): the closure on word 2 sits in time before the
    # interpolated edge after word 1, two rows from the edge after word 2. Its waqf must
    # still land on word 2, or the reference continues through a known pause.
    rows = [
        _boundary("c", 0, 0, WASL),
        _boundary("c", 1, 2, WAQF, predicted=MID_WORD_CLOSURE),
        _boundary("c", 2, 1, WASL),
        _boundary("c", 3, 2, WASL),
        _boundary("c", 4, 3, WASL),
    ]
    edges, excluded = clip_edges(rows)
    assert edges == [Edge(0, 0, False), Edge(2, 1, False), Edge(3, 2, True),
                     Edge(4, 3, False)]
    assert excluded == {1: EXCLUDED_CLOSURE_MERGED}


def test_a_closure_with_no_edge_at_its_word_is_an_edge_of_its_own():
    # A stop judged after the ayah's last word, which has no regular edge.
    rows = [_boundary("c", 0, 0, WASL), _boundary("c", 1, 1, WAQF, predicted=MID_WORD_CLOSURE)]
    edges, excluded = clip_edges(rows)
    assert edges == [Edge(0, 0, False), Edge(1, 1, True)]
    assert excluded == {}


def test_a_waqf_closure_on_a_re_read_word_excludes_the_clip():
    rows = [
        _boundary("c", 0, 0, WASL),
        _boundary("c", 1, 1, WAQF),
        _boundary("c", 2, 0, WASL),  # re-read from word 0
        _boundary("c", 3, 0, WAQF, predicted=MID_WORD_CLOSURE),  # which pass?
        _boundary("c", 4, 1, WASL),
    ]
    with pytest.raises(ClipExcluded) as excinfo:
        clip_edges(rows)
    assert excinfo.value.reason == EXCLUDED_CLOSURE_AMBIGUOUS


def test_a_wasl_closure_on_a_re_read_word_changes_nothing():
    rows = [
        _boundary("c", 0, 0, WASL),
        _boundary("c", 1, 1, WAQF),
        _boundary("c", 2, 0, WASL),
        _boundary("c", 3, 0, WASL, predicted=MID_WORD_CLOSURE),
    ]
    edges, excluded = clip_edges(rows)
    assert [e.waqf for e in edges] == [False, True, False]
    assert excluded == {3: EXCLUDED_CLOSURE_MERGED}


# --- recited runs --------------------------------------------------------------------


def test_a_clip_with_no_pause_is_one_run():
    edges = [Edge(0, 0, False), Edge(1, 1, False)]
    assert recited_runs(edges, 4) == [RecitedRun(0, 3, tuple(edges))]


def test_a_pause_splits_the_runs():
    edges = [Edge(0, 0, False), Edge(1, 1, True), Edge(2, 2, False)]
    assert recited_runs(edges, 5) == [
        RecitedRun(0, 2, tuple(edges[:2])),
        RecitedRun(2, 4, (edges[2],)),
    ]


def test_a_re_read_restarts_at_an_earlier_word():
    edges = [Edge(0, 0, False), Edge(1, 1, True), Edge(2, 0, False), Edge(3, 1, False)]
    assert recited_runs(edges, 3) == [
        RecitedRun(0, 2, tuple(edges[:2])),
        RecitedRun(0, 3, tuple(edges[2:])),
    ]


def test_a_missing_edge_inside_a_run_is_continuation():
    edges = [Edge(0, 0, False), Edge(1, 2, True)]
    assert recited_runs(edges, 4) == [RecitedRun(0, 3, tuple(edges)), RecitedRun(3, 4, ())]


def test_a_pause_on_the_last_word_ends_the_clip():
    edges = [Edge(0, 0, False), Edge(1, 1, True)]
    assert recited_runs(edges, 2) == [RecitedRun(0, 2, tuple(edges))]


@pytest.mark.parametrize(
    "edges, n_words",
    [
        ([Edge(0, 0, False), Edge(1, 1, False), Edge(2, 0, False)], 4),  # back, no pause
        ([Edge(0, 0, False), Edge(1, 4, True)], 4),  # past the ayah's end
        ([Edge(0, 0, False), Edge(1, 1, False)], 2),  # continues past the last word
    ],
)
def test_edges_that_are_not_one_recitation_exclude_the_clip(edges, n_words):
    with pytest.raises(ClipExcluded) as excinfo:
        recited_runs(edges, n_words)
    assert excinfo.value.reason == EXCLUDED_INCONSISTENT_CLIP


def test_final_mark_reads_the_haraka_after_the_carrier():
    assert final_mark("كَتَبَ", 4) == "fatha"
    assert final_mark("كَتَبِ ", 4) == "kasra"
    assert final_mark("كَتَب", 4) == "sukun"
    assert final_mark("قَدڇ", 2) == "sukun"


# --- conversion ----------------------------------------------------------------------

_WORDS = {
    "1:1": ["كَتَبَ", "قَلَمُ", _ASSIMILATED, "دَارِ", "فَتَحَ"],
    "1:2": ["كَتَبَ", _UNREALIZABLE, "دَارِ"],
}


def _convert(boundaries):
    return convert(boundaries, _WORDS.__getitem__, _fake_realize)


def test_convert_places_each_site_on_the_clip_reference():
    boundaries = [
        _boundary("a.wav", 0, 0, WASL),
        _boundary("a.wav", 1, 1, WAQF),
        _boundary("a.wav", 2, 2, WASL),
        _boundary("a.wav", 3, 3, WASL),
    ]
    sites, summary = _convert(boundaries)

    reference = "كَتَبَ قَلَم مِن دَارِ فَتَح"
    assert {s.reference for s in sites} == {reference}
    by_id = {s.site_id: s for s in sites}
    assert set(by_id) == {"waqf_boundary:a.wav#0", "waqf_boundary:a.wav#1",
                          "waqf_boundary:a.wav#3"}

    fatha = by_id["waqf_boundary:a.wav#0"]
    assert (fatha.reference_index, fatha.mark, fatha.prescribed, fatha.heard) == (
        4, "fatha", "fatha", "fatha")
    assert fatha.assumes_competent_reciter and fatha.stratum == "waqf_boundary:wasl"

    pause = by_id["waqf_boundary:a.wav#1"]
    assert reference[pause.reference_index] == "م" and pause.mark == "sukun"
    assert not pause.assumes_competent_reciter and pause.stratum == "waqf_boundary:waqf"

    kasra = by_id["waqf_boundary:a.wav#3"]
    assert reference[kasra.reference_index] == "ر" and kasra.mark == "kasra"

    for site in sites:
        assert (site.source, site.audio_filename, site.start_sample) == ("waqf_boundary",
                                                                          "a.wav", 0)
        assert (site.shard, site.end_sample, site.audio_sha256) == (None, None, None)
    assert {s.stratum: s.stratum_population for s in sites} == {
        "waqf_boundary:wasl": 2, "waqf_boundary:waqf": 1}
    assert summary["rows_by_verdict_and_outcome"] == {
        "waqf": {"waqf_boundary:waqf": 1},
        "wasl": {EXCLUDED_FINAL_LETTER_ASSIMILATED: 1, "waqf_boundary:wasl": 2},
    }


def test_a_pause_on_a_long_vowel_is_a_haraka_site_that_assumes_a_competent_reciter():
    words = {"1:3": ["مَاا", "قَاالَ"]}
    sites, _ = convert([_boundary("b.wav", 0, 0, WAQF, surah_ayah="1:3")],
                       words.__getitem__, _fake_realize)
    (site,) = sites
    assert site.mark == "fatha" and site.assumes_competent_reciter
    assert site.stratum == "waqf_boundary:waqf"


def test_convert_accounts_for_every_row():
    boundaries = [
        _boundary("a.wav", 0, 0, WASL),
        _boundary("a.wav", 1, 0, MID_WORD_CLOSURE, predicted=MID_WORD_CLOSURE),
        _boundary("a.wav", 2, 1, WASL),
        _boundary("c.wav", 0, 0, WASL, surah_ayah="1:2"),
        _boundary("c.wav", 1, 1, WAQF, surah_ayah="1:2"),  # the run ends on _UNREALIZABLE
        _boundary("d.wav", 0, 0, WASL),
        _boundary("d.wav", 1, 1, WASL),
        _boundary("d.wav", 2, 0, WASL),  # back to word 0 with no pause
        _boundary("e.wav", 0, 0, WASL),
        _boundary("e.wav", 1, 1, WAQF),
        _boundary("e.wav", 2, 0, WASL),
        _boundary("e.wav", 3, 0, WAQF, predicted=MID_WORD_CLOSURE),  # re-read: which pass?
        _boundary("e.wav", 4, 1, MID_WORD_CLOSURE, predicted=MID_WORD_CLOSURE),
    ]
    sites, summary = _convert(boundaries)
    assert summary["fixture_rows"] == 13
    assert summary["sites"] == len(sites) == 2
    assert summary["rows_by_verdict_and_outcome"] == {
        MID_WORD_CLOSURE: {EXCLUDED_MID_WORD_CLOSURE: 2},
        WAQF: {EXCLUDED_CLOSURE_AMBIGUOUS: 2, EXCLUDED_PHONETIZER_UNSUPPORTED: 1},
        WASL: {EXCLUDED_CLOSURE_AMBIGUOUS: 2, EXCLUDED_INCONSISTENT_CLIP: 3,
               EXCLUDED_PHONETIZER_UNSUPPORTED: 1, "waqf_boundary:wasl": 2},
    }
    assert summary["clips"] == 4 and summary["clips_with_sites"] == 1


def test_convert_is_deterministic_under_row_order():
    boundaries = [_boundary("a.wav", i, i, WASL) for i in range(4)] + [
        _boundary("b.wav", 0, 0, WAQF)]
    assert _convert(boundaries) == _convert(list(reversed(boundaries)))


def test_duplicate_boundary_indices_fail_loudly():
    with pytest.raises(ValueError, match="duplicate boundary_index"):
        _convert([_boundary("a.wav", 0, 0, WASL), _boundary("a.wav", 0, 1, WASL)])


def test_read_boundaries_rejects_a_row_that_is_not_a_whole_clip(tmp_path):
    path = tmp_path / "events.jsonl"
    row = {"audio_ref": "a.wav__seg1.wav", "clip_id": "a.wav", "surah_ayah": "1:1",
           "boundary_index": 0, "word_index": 0, "predicted": WASL, "verdict": WASL}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="whole clip"):
        read_boundaries([path])


# --- the committed file --------------------------------------------------------------


def test_committed_sites_load_and_match_their_summary():
    sites = load_truth_sites(SITES_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["fixture_rows"] == len(read_boundaries(FIXTURE_PATHS)) == 2050
    assert summary["sites"] == len(sites)
    assert summary["sites_assuming_competent_reciter"] == sum(
        s.assumes_competent_reciter for s in sites)
    counts: dict[str, dict[str, int]] = {}
    for site in sites:
        counts.setdefault(site.stratum, {}).setdefault(site.mark, 0)
        counts[site.stratum][site.mark] += 1
    assert counts == summary["sites_by_stratum_and_mark"]
    assert all(s.source == "waqf_boundary" for s in sites)
    assert all(s.audio_sha256 is None for s in sites)  # filled when #83 re-stages


def test_readme_carries_the_committed_counts():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    readme = (SITES_PATH.parent / "README.md").read_text(encoding="utf-8")
    for table in summary_table(summary).split("\n\n"):
        assert table in readme


def test_the_committed_file_regenerates_identically():
    pytest.importorskip("quran_transcript")
    waqf_segments = pytest.importorskip("tadabur.waqf_segments")
    from tadabur.waqf_truth_sites import hafs_realizer

    sites, summary = convert(
        read_boundaries(), waqf_segments._uthmani_words, hafs_realizer()
    )
    assert sites == load_truth_sites(SITES_PATH)
    assert summary == json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


# --- the Hafs realizer ---------------------------------------------------------------


def _realize_ayah_prefix(surah_ayah: str, n_words: int) -> tuple[list[str], RealizedRun]:
    pytest.importorskip("quran_transcript")
    from quran_transcript import Aya

    from tadabur.waqf_truth_sites import hafs_realizer

    surah, ayah = map(int, surah_ayah.split(":"))
    words = list(Aya(surah, ayah).get().uthmani_words)[:n_words]
    return words, hafs_realizer()(words)


def _marks(run: RealizedRun) -> list[tuple[str, str] | None]:
    return [None if end is None else (run.phonemes[end], final_mark(run.phonemes, end))
            for end in run.word_ends]


def test_hafs_realizer_finds_each_word_final_mark():
    # وَنُرِيدُ أَن نَّمُنَّ عَلَى ٱلَّذِينَ, stopping on the last word.
    _, run = _realize_ayah_prefix("28:5", 5)
    assert _marks(run) == [
        ("د", "damma"),
        None,  # أَن: its noon merges into نَّ (idgham)
        ("ن", "fatha"),  # the second of the doubled noon carries the fatha
        ("ل", "fatha"),  # عَلَى shortened before hamzat wasl; the mark is on ل
        ("ن", "sukun"),  # waqf
    ]


def test_hafs_realizer_handles_tanween_ikhfa_and_hamza_marks():
    # وَٱتَّقُوا۟ يَوْمًۭا لَّا تَجْزِى نَفْسٌ عَن نَّفْسٍۢ شَيْـًۭٔا, stopping on the last word.
    _, run = _realize_ayah_prefix("2:48", 8)
    assert _marks(run) == [
        ("ق", "damma"),  # the madd و and silent alif pass the mark back to ق
        ("م", "fatha"),  # tanween fatha merged into لّ keeps its fatha
        ("ل", "fatha"),
        ("ز", "kasra"),
        ("س", "damma"),
        None,  # عَن: idgham into نّ
        ("س", "kasra"),  # tanween kasra in ikhfa keeps its kasra
        ("ء", "fatha"),  # hamza written as a mark; tanween fatha becomes اا at waqf
    ]


def test_hafs_realizer_reports_an_unsupported_waqf():
    pytest.importorskip("quran_transcript")
    from quran_transcript import Aya

    from tadabur.waqf_truth_sites import hafs_realizer

    words = list(Aya(27, 88).get().uthmani_words)[8:14]  # ends in waqf on شَىْءٍ
    with pytest.raises(ClipExcluded) as excinfo:
        hafs_realizer()(words)
    assert excinfo.value.reason == EXCLUDED_PHONETIZER_UNSUPPORTED


# --- pausal forms ----------------------------------------------------------------------


def test_pausal_taa_marbuta_rewrites_only_a_final_tanween_fatha_on_taa_marbuta():
    assert pausal_taa_marbuta("رَحْمَةًۭ") == "رَحْمَةَ"
    for word in ("رَحْمَةٌۭ", "رَحْمَةٍۢ", "عَلِيمًا", "بِنَآءًۭ", "قَالَ"):
        assert pausal_taa_marbuta(word) == word


@pytest.mark.parametrize(
    "surah_ayah, n_words, carrier, mark",
    [
        ("25:32", 9, "ه", "sukun"),  # وَٰحِدَةًۭ at waqf: ه with sukun, not تَاا
        ("2:7", 9, "ه", "sukun"),  # غِشَـٰوَةٌۭ
        ("2:23", 10, "ه", "sukun"),  # بِسُورَةٍۢ
        ("2:17", 15, "ت", "sukun"),  # ظُلُمَـٰتٍۢ: tanween kasra drops
        ("25:32", 15, "ل", "fatha"),  # تَرْتِيلًۭا: madd al-iwad, لَاا
        ("2:22", 7, "ء", "fatha"),  # بِنَآءًۭ: madd al-iwad after a final hamza
        ("2:48", 8, "ء", "fatha"),  # شَيْـًۭٔا: hamza written as a mark
        ("28:5", 4, "ل", "fatha"),  # عَلَى: a madd ending keeps the fatha before it
    ],
)
def test_hafs_realizer_gives_the_pausal_form_of_the_terminal_word(
    surah_ayah, n_words, carrier, mark
):
    _, run = _realize_ayah_prefix(surah_ayah, n_words)
    end = run.word_ends[-1]
    assert (run.phonemes[end], final_mark(run.phonemes, end)) == (carrier, mark)


def test_taa_marbuta_with_tanween_fatha_keeps_its_wasl_form_inside_a_run():
    # جُمْلَةًۭ وَٰحِدَةًۭ: only the terminal word is pausal.
    words, run = _realize_ayah_prefix("25:32", 9)
    end = run.word_ends[-2]
    assert words[-2].endswith("ةًۭ")
    assert (run.phonemes[end], final_mark(run.phonemes, end)) == ("ت", "fatha")
    assert run.phonemes.endswith("وَااحِدَه")

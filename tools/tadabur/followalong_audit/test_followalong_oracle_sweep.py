"""The oracle sweeps over a stub engine (#308): clean and planted, seeded and reproducible.

The engine is a table, not Muraja: it grades each request's words as the test says, so what is
pinned is the sweep's own work — the requests it builds, the rows it keeps, and the rules that
turn a grade into a finding, a catch, a miss or a ``not_graded``. A real run goes through
``Harness.run``, which the stub shares its ``run(requests) -> (build, results)`` shape with.
"""

from __future__ import annotations

import random

import pytest

from tadabur.followalong_audit.plant import KINDS, plant, plantable_words, planting_seed
from tadabur.followalong_audit.sweep import (
    AyahReferences,
    SweepClip,
    _observed_exemptions,
    _planted_row,
    catch_rates,
    clean_sweep,
    false_flags_per_1000,
    median_speech_pace,
    oracle_summary,
    paced_word_times,
    planted_sweep,
    splits_window,
    tlog_clips,
    word_position,
)
from training.decoding import SAMPLE_RATE, clusters, committed_text
from training.muraja_policy import AUDIT_SCORINGS, CORRECT

STRICT, BALANCED = (scoring.name for scoring in AUDIT_SCORINGS)
WORDS = ("ءَلحَمدُ", "لِللَااهِ", "رَببِ", "لعَاالَمِۦۦن")
SUBSTITUTION = [{"group_index": 0, "kind": "substitution"}]
NOT_GRADED = None


class StubEngine:
    """Grades each word of each request from ``grades[(item, scoring name, word)]``.

    ``(quality, errors)`` is the word's kept grade; ``NOT_GRADED`` leaves it out of ``final``;
    an absent entry is a correct word. It records the requests it was asked to grade.
    """

    def __init__(self, grades: dict | None = None):
        self.grades = grades or {}
        self.requests: list[dict] = []

    def run(self, requests, *args, **kwargs):
        results = []
        for request in requests:
            self.requests.append(request)
            for scoring in AUDIT_SCORINGS:
                final = []
                for word in request["report_words"]:
                    grade = self.grades.get((request["item"], scoring.name, word), (CORRECT, []))
                    if grade is NOT_GRADED:
                        continue
                    quality, errors = grade
                    final.append({"word": word, "quality": quality, "score": 1.0, "errors": errors})
                results.append({
                    "item": request["item"], "scoring": scoring.engine_name, "final": final,
                    "letters": [], "checks": [{"statuses": []} for _ in request["checks"]],
                })
        return {}, results


def clip(item: str = "c1", words=WORDS, *, surah: int = 1, ayah: int = 2, times=None,
         n_samples: int = 8 * SAMPLE_RATE, waqf=frozenset(), engine=None, word_of=None,
         cluster_groups=None) -> SweepClip:
    """A sweep clip; by default Muraja recites one engine word per canonical word, and each
    canonical cluster lands in its own word's group of the same index."""
    words = tuple(words)
    engine = words if engine is None else tuple(engine)
    word_of = (tuple(range(1, len(words) + 1)) if word_of is None else tuple(word_of))
    if cluster_groups is None:
        cluster_groups = tuple(
            (word + 1, group)
            for word, characters in enumerate(words)
            for group in range(len(clusters(characters))))
    return SweepClip(item=item, surah=surah, ayah=ayah, words=words, engine_words=engine,
                     word_of=word_of, word_times=times, n_samples=n_samples,
                     waqf=frozenset(waqf), cluster_groups=tuple(cluster_groups))


def trace(word: int, *, mode: str = STRICT, quality: str = "minor", groups=(), marked=()) -> dict:
    """One ``word_traces`` row for a stub engine's grade of ``word``."""
    return {"scoring": mode, "word": word, "quality": quality, "score": 1.0,
            "errors": [{"group_index": group} for group in groups],
            "marked": list(marked), "checks": [{"status": None}], "setting_checks": []}


def reference(words) -> str:
    return "".join(words)


# --- the pure helpers -------------------------------------------------------------------------
def test_a_words_position_is_last_first_waqf_or_middle():
    assert word_position(0, 4, waqf=False) == "first"
    assert word_position(3, 4, waqf=False) == "last"
    assert word_position(1, 4, waqf=True) == "waqf"
    assert word_position(2, 4, waqf=False) == "middle"
    assert word_position(0, 1, waqf=False) == "last"  # an ayah's end is a stop, first or not
    assert word_position(3, 4, waqf=True) == "last"


def test_a_window_boundary_splits_a_word_it_falls_strictly_inside():
    eight = 8 * SAMPLE_RATE  # windows (0,5) (1,6) (2,7) (3,8): an edge at every whole second
    assert splits_window(2.5, 3.5, eight)
    assert splits_window(5.5, 6.5, eight)
    assert not splits_window(3.1, 3.9, eight)
    assert not splits_window(0.0, 0.9, eight)  # the clip's own start is no split
    assert not splits_window(1.0, 2.0, eight)  # edges on a word's own ends do not split it


def test_a_clip_too_short_for_a_window_has_no_boundary_to_split():
    assert not splits_window(0.5, 2.5, 3 * SAMPLE_RATE)


def test_the_median_speech_pace_is_over_aligned_clips_first_start_to_last_end():
    clips = [
        (10, [(0.0, 1.0), (1.0, 2.0)]),   # 10 Characters in 2 s: 5
        (30, [(1.0, 4.0), (4.0, 6.0)]),   # 30 in 5 s: 6
        (7, None),                        # no alignment: left out
        (40, [(0.0, 5.0)]),               # 8
    ]
    assert median_speech_pace(clips) == pytest.approx(6.0)


def test_there_is_no_pace_without_an_aligned_clip():
    with pytest.raises(ValueError):
        median_speech_pace([(7, None)])


def test_paced_word_times_lay_the_words_end_to_end_at_the_pace():
    times, n_samples = paced_word_times([3, 6], pace=3.0)
    assert times == [(0.0, 1.0), (1.0, 3.0)]
    assert n_samples == 3 * SAMPLE_RATE
    _, rounded_up = paced_word_times([1], pace=3.0)
    assert rounded_up == 5334  # ceil(16000 / 3): the clip never ends inside its last Character


# --- the clean sweep --------------------------------------------------------------------------
def flagged_clean_sweep():
    first = clip("a", times=None)
    second = clip("b", surah=2, ayah=7, waqf={1})
    grades = {
        ("a/oracle", STRICT, 2): ("wrong", SUBSTITUTION),
        ("b/oracle", STRICT, 4): ("skipped", []),
        ("b/oracle", BALANCED, 4): ("skipped", []),
        ("b/oracle", STRICT, 1): NOT_GRADED,
        ("b/oracle", BALANCED, 1): NOT_GRADED,
    }
    engine = StubEngine(grades)
    return clean_sweep([first, second], engine), engine


def test_the_clean_sweep_grades_each_clip_as_one_oracle_request_with_the_whole_reference():
    result, engine = flagged_clean_sweep()
    assert [request["item"] for request in engine.requests] == ["a/oracle", "b/oracle"]
    for request, source in zip(engine.requests, (clip("a"), clip("b", surah=2, ayah=7))):
        assert (request["surah"], request["ayah"], request["start_word"]) == (source.surah, source.ayah, 1)
        assert request["report_words"] == [1, 2, 3, 4]
        assert committed_text(request["checks"]) == reference(WORDS)


def test_every_word_that_is_not_correct_is_a_finding_with_its_evidence():
    result, _ = flagged_clean_sweep()
    by_key = {(f["item"], f["mode"], f["word"]): f for f in result.findings}
    assert set(by_key) == {("a", STRICT, 2), ("b", STRICT, 4), ("b", BALANCED, 4)}
    wrong = by_key[("a", STRICT, 2)]
    assert (wrong["surah"], wrong["ayah"], wrong["quality"]) == (1, 2, "wrong")
    assert wrong["error_kind"] == "substitution"  # the first error's kind
    assert wrong["form"] == WORDS[1]
    assert wrong["position"] == "middle"
    assert isinstance(wrong["boundary"], bool) or wrong["boundary"] is None
    assert isinstance(wrong["statuses"], list) and wrong["statuses"]  # one status slot per check
    skipped = by_key[("b", STRICT, 4)]
    assert (skipped["error_kind"], skipped["position"]) == ("skipped", "last")  # no error: the quality


def test_a_word_the_engine_never_graded_is_counted_apart_not_as_a_finding():
    result, _ = flagged_clean_sweep()
    assert result.not_graded == {STRICT: 1, BALANCED: 1}
    assert result.graded_words == {STRICT: 7, BALANCED: 7}
    assert result.words_by_position[STRICT] == {"first": 1, "middle": 3, "waqf": 1, "last": 2}


def test_false_flags_are_reported_per_thousand_graded_words_by_mode_cause_and_position():
    result, _ = flagged_clean_sweep()
    rates = false_flags_per_1000(result)
    assert rates[STRICT]["all"] == pytest.approx(2 / 7 * 1000)
    assert rates[BALANCED]["all"] == pytest.approx(1 / 7 * 1000)
    assert rates[STRICT]["error_kind"] == {
        "substitution": pytest.approx(1 / 7 * 1000), "skipped": pytest.approx(1 / 7 * 1000)}
    assert rates[STRICT]["position"]["last"] == pytest.approx(1 / 2 * 1000)
    assert rates[STRICT]["position"]["first"] == 0.0


def test_a_clean_clip_without_a_finding_leaves_no_row_behind():
    result = clean_sweep([clip()], StubEngine())
    assert result.findings == []
    assert result.graded_words == {STRICT: 4, BALANCED: 4}


# --- the planted sweep ------------------------------------------------------------------------
def expected_plant(seed: int, source: SweepClip, kind: str):
    return plant(kind, source.words, random.Random(planting_seed(seed, source.item, kind)))


def planted_rows(result, kind: str, mode: str) -> list[dict]:
    return [row for row in result.rows if row["kind"] == kind and row["mode"] == mode]


def test_each_clip_is_planted_once_per_kind_in_each_mode_and_the_rest_are_counted_unplantable():
    source = clip()
    engine = StubEngine()
    result = planted_sweep([source], engine, seed=7)
    plantable = [kind for kind in KINDS if plantable_words(kind, source.words)]
    assert sorted((row["kind"], row["mode"]) for row in result.rows) == sorted(
        (kind, mode) for kind in plantable for mode in (STRICT, BALANCED))
    assert result.unplantable == {kind: 1 for kind in KINDS if kind not in plantable}
    assert sorted(request["item"] for request in engine.requests) == sorted(
        f"c1/{kind}" for kind in plantable)


def test_the_engine_is_handed_the_planted_reference_over_the_engines_own_word_numbers():
    source = clip()
    engine = StubEngine()
    planted_sweep([source], engine, seed=7)
    for request in engine.requests:
        kind = request["item"].split("/", 1)[1]
        planted = expected_plant(7, source, kind)
        assert committed_text(request["checks"]) == reference(planted.words)
        assert request["report_words"] == [1, 2, 3, 4]  # Muraja's words are the ayah's, planted or not
        assert request["start_word"] == 1


def test_the_planted_word_is_the_seeded_draw_and_rows_name_it_by_its_engine_word_number():
    source = clip()
    result = planted_sweep([source], StubEngine(), seed=7)
    for row in result.rows:
        assert row["word"] == expected_plant(7, source, row["kind"]).word + 1
        assert row["item"] == "c1"


def test_a_planted_word_graded_below_correct_is_caught_in_that_mode_only():
    source = clip()
    word = expected_plant(7, source, "haraka_swap").word + 1
    engine = StubEngine({("c1/haraka_swap", STRICT, word): ("minor", SUBSTITUTION)})
    result = planted_sweep([source], engine, seed=7)
    (strict,) = planted_rows(result, "haraka_swap", STRICT)
    (balanced,) = planted_rows(result, "haraka_swap", BALANCED)
    assert (strict["caught"], strict["quality"], strict["not_graded"]) == (True, "minor", False)
    assert (balanced["caught"], balanced["quality"], balanced["not_graded"]) == (False, CORRECT, False)
    assert strict["marked_is_planted"] in (True, False)


def test_a_miss_carries_its_replay_evidence_position_and_boundary():
    times = [(0.1, 0.9), (1.1, 1.9), (2.7, 3.4), (3.6, 3.9)]  # word 3 spans the 3 s window edge
    source = clip(times=times, waqf={1})
    planted = plant("haraka_swap", source.words, random.Random(planting_seed(7, "c1", "haraka_swap")))
    result = planted_sweep([source], StubEngine(), seed=7)
    (miss,) = planted_rows(result, "haraka_swap", STRICT)
    assert miss["caught"] is False
    assert isinstance(miss["statuses"], list) and miss["statuses"]
    assert miss["position"] == word_position(planted.word, 4, waqf=planted.word == 1)
    assert miss["boundary"] is (planted.word == 2)
    assert miss["length"] == len(WORDS[planted.word]) - sum(  # Characters, not scalars
        1 for char in WORDS[planted.word] if "ً" <= char <= "ٟ")


def test_a_skipped_words_boundary_comes_from_the_word_the_skip_removes():
    times = ((0.1, 0.9), (1.1, 1.9), (2.7, 3.4), (3.6, 3.9))  # word 3 spans the 3 s edge
    source = clip(times=times)
    planted = plant("skip_word", WORDS, random.Random(0), word=2)
    row = _planted_row(source, "skip_word", planted, STRICT, {(STRICT, 3): trace(3)},
                       {"checks": [{"flush": False}]})
    assert (row["word"], row["boundary"]) == (3, True)


def test_a_planted_word_the_engine_leaves_out_is_not_graded_and_neither_caught_nor_missed():
    source = clip()
    word = expected_plant(7, source, "skip_word").word + 1
    engine = StubEngine({("c1/skip_word", STRICT, word): NOT_GRADED})
    result = planted_sweep([source], engine, seed=7)
    (strict,) = planted_rows(result, "skip_word", STRICT)
    assert (strict["not_graded"], strict["caught"], strict["quality"]) == (True, None, None)
    assert planted_rows(result, "skip_word", BALANCED)[0]["caught"] is False


def test_an_added_word_is_caught_on_the_copied_word_or_the_one_after_and_reports_the_split():
    source = clip()
    planted = expected_plant(7, source, "add_word")
    copied = planted.word + 1                     # the ayah word the copy doubles, 1-based
    following = copied + 1                        # the word the copy's phonemes displace
    assert planted.inserted == planted.word + 1   # the copy is reference-only: no engine word
    on_copied = StubEngine({("c1/add_word", STRICT, copied): ("wrong", SUBSTITUTION)})
    (row,) = planted_rows(planted_sweep([source], on_copied, seed=7), "add_word", STRICT)
    assert (row["caught"], row["caught_on_copied"], row["caught_on_following"]) == (True, True, False)
    assert (row["word"], row["quality"]) == (copied, "wrong")
    on_following = StubEngine({("c1/add_word", STRICT, following): ("wrong", SUBSTITUTION)})
    (row,) = planted_rows(planted_sweep([source], on_following, seed=7), "add_word", STRICT)
    assert (row["caught"], row["caught_on_copied"], row["caught_on_following"]) == (True, False, True)
    before = StubEngine({("c1/add_word", STRICT, copied - 1): ("wrong", SUBSTITUTION)})
    (row,) = planted_rows(planted_sweep([source], before, seed=7), "add_word", STRICT)
    assert (row["caught"], row["caught_on_copied"], row["caught_on_following"]) == (False, False, False)
    (quiet,) = planted_rows(planted_sweep([source], StubEngine(), seed=7), "add_word", STRICT)
    assert (quiet["caught"], quiet["caught_on_copied"], quiet["caught_on_following"]) == (False, False, False)


def test_a_haraka_drop_row_says_whether_the_dropped_letter_was_protected():
    source = clip()
    planted = expected_plant(7, source, "haraka_drop")
    result = planted_sweep([source], StubEngine(), seed=7)
    for row in planted_rows(result, "haraka_drop", STRICT):
        assert row["protected"] is planted.protected
    assert all(row["protected"] is None for row in result.rows if row["kind"] != "haraka_drop")


def test_an_observed_exemption_names_the_letter_the_drop_was_planted_on():
    rows = [{"kind": "haraka_drop", "protected": False, "mode": mode, "form": "بِاسمِ",
             "label": "-م", "caught": False} for mode in (STRICT, BALANCED)]
    assert _observed_exemptions(rows) == [{"letter": "م", "planted": 2, "caught": 0}]


def test_a_letter_the_engine_flags_a_drop_on_is_no_exemption():
    rows = [{"kind": "haraka_drop", "protected": False, "mode": STRICT, "form": "بِاسمِ",
             "label": "-ب", "caught": True}]
    assert _observed_exemptions(rows) == []


def test_a_pair_swap_row_carries_the_direction_it_planted():
    result = planted_sweep([clip()], StubEngine(), seed=7)
    (row,) = planted_rows(result, "consonant_pair:ح↔ه", STRICT)
    assert row["label"] in ("ح→ه", "ه→ح")


def test_the_same_seed_reproduces_the_sweep_and_another_seed_plants_elsewhere():
    clips = [clip(f"c{n}", words=("ءَلحَمدُ", "لِللَااهِ", "رَببِ", "لعَاالَمِۦۦن", "صِرَااطَ",
                                  "ءَنعَمتَ", "عَلَيهِم")) for n in range(6)]
    first = planted_sweep(clips, StubEngine(), seed=11)
    again = planted_sweep(clips, StubEngine(), seed=11)
    assert first.rows == again.rows and first.unplantable == again.unplantable
    other = planted_sweep(clips, StubEngine(), seed=12)
    assert [row["word"] for row in other.rows] != [row["word"] for row in first.rows]


def test_the_rows_do_not_depend_on_the_order_the_clips_arrive_in():
    clips = [clip(f"c{n}") for n in range(4)]
    forward = planted_sweep(clips, StubEngine(), seed=3)
    backward = planted_sweep(list(reversed(clips)), StubEngine(), seed=3)
    assert sorted(forward.rows, key=lambda r: (r["item"], r["kind"], r["mode"])) == \
        sorted(backward.rows, key=lambda r: (r["item"], r["kind"], r["mode"]))


def test_a_planted_sweep_over_word_times_places_the_planted_reference_on_the_clock():
    times = [(0.1, 0.9), (1.1, 1.9), (2.1, 2.9), (3.1, 3.9)]
    source = clip(times=times)
    engine = StubEngine()
    planted_sweep([source], engine, seed=5, kinds=("haraka_swap", "skip_word", "add_word"))
    assert sorted(request["item"] for request in engine.requests) == [
        "c1/add_word", "c1/haraka_swap", "c1/skip_word"]
    for request in engine.requests:
        planted = expected_plant(5, source, request["item"].split("/", 1)[1])
        assert committed_text(request["checks"]) == reference(planted.words)


# --- the catch rate ---------------------------------------------------------------------------
def row(kind="haraka_swap", mode=STRICT, position="first", length=4, *, caught=True) -> dict:
    return {"kind": kind, "mode": mode, "position": position, "length": length, "caught": caught,
            "not_graded": caught is None}


def test_the_catch_rate_is_caught_over_graded_per_kind_mode_length_and_position():
    rows = [row(caught=True), row(caught=True), row(caught=False), row(caught=None),
            row(position="last", caught=False), row(mode=BALANCED, caught=True)]
    cells = {(c["kind"], c["mode"], c["position"]): c for c in catch_rates(rows)}
    assert len(cells) == 3
    cell = cells[("haraka_swap", STRICT, "first")]
    assert (cell["n"], cell["caught"], cell["not_graded"]) == (3, 2, 1)
    assert cell["rate"] == pytest.approx(2 / 3)
    assert cells[("haraka_swap", STRICT, "last")]["rate"] == 0.0
    assert cells[("haraka_swap", BALANCED, "first")]["rate"] == 1.0


def test_words_of_very_different_length_fall_in_different_cells():
    cells = catch_rates([row(length=2), row(length=14)])
    assert len(cells) == 2
    assert len({cell["length_bin"] for cell in cells}) == 2


def test_a_cell_with_nothing_graded_has_no_rate():
    (cell,) = catch_rates([row(caught=None)])
    assert (cell["n"], cell["not_graded"], cell["rate"]) == (0, 1, None)


# --- both coordinate systems --------------------------------------------------------------------
#: Two canonical words Muraja recites as one, as 4:109 merges ``هَٰٓأَنتُمۡ`` and
#: ``هَٰٓؤُلَآءِ``; the engine words are its own.
MERGED_WORDS = ("ءَ", "بِ", "تُ", "ثَ")
MERGED_ENGINE = ("ءَبِ", "تُ", "ثَ")
MERGED_WORD_OF = (1, 1, 2, 3)
MERGED_GROUPS = ((1, 0), (1, 1), (2, 0), (3, 0))


def merged_clip(**fields) -> SweepClip:
    return clip(words=MERGED_WORDS, engine=MERGED_ENGINE, word_of=MERGED_WORD_OF,
                cluster_groups=MERGED_GROUPS, **fields)


def test_a_merged_word_is_graded_and_counted_as_one_engine_word_not_two_canonical_ones():
    source = merged_clip()
    engine = StubEngine({("c1/oracle", STRICT, 1): ("minor", SUBSTITUTION)})
    result = clean_sweep([source], engine)
    (request,) = engine.requests
    assert request["report_words"] == [1, 2, 3]  # Muraja's words, not the four canonical ones
    assert result.graded_words == {STRICT: 3, BALANCED: 3}
    assert [(f["word"], f["form"]) for f in result.findings if f["mode"] == STRICT] == [
        (1, MERGED_ENGINE[0])]
    assert [row["word"] for row in result.word_rows] == [1, 2, 3]


def test_a_planted_word_is_read_at_the_engine_word_it_is_recited_in():
    source = merged_clip(times=((0.1, 0.9), (0.9, 1.5), (1.5, 2.0), (2.0, 2.6)))
    planted = plant("haraka_swap", MERGED_WORDS, random.Random(0), word=1)
    request = {"checks": [{"flush": False}]}
    on_engine = {(STRICT, 1): trace(1, groups=[0]),
                 (STRICT, 2): trace(2, quality=CORRECT),
                 (STRICT, 3): trace(3, quality=CORRECT)}
    row = _planted_row(source, "haraka_swap", planted, STRICT, on_engine, request)
    assert (row["word"], row["reference_word"]) == (1, 2)
    assert (row["caught"], row["quality"], row["form"]) == (True, "minor", planted.words[1])
    # The canonical ordinal would have read engine word 2, which is correct here.
    ordinal = {(STRICT, 1): trace(1, quality=CORRECT),
               (STRICT, 2): trace(2, groups=[0]),
               (STRICT, 3): trace(3, quality=CORRECT)}
    assert _planted_row(source, "haraka_swap", planted, STRICT, ordinal,
                        request)["caught"] is False


def test_the_marked_letter_is_compared_in_murajas_group_coordinates():
    # لِللَااهِ: the madd doubles ا across one group, and the geminate's two ل share another.
    source = clip(words=("لِللَااهِ",), cluster_groups=((1, 0), (1, 1), (1, 1), (1, 2),
                                                       (1, 2), (1, 3)))
    planted = plant("haraka_swap", source.words, random.Random(0), word=0)
    assert planted.character == 2  # لَ: group 1, not cluster index 2
    request = {"checks": [{"flush": False}]}
    on_group = {(STRICT, 1): trace(1, groups=[1])}
    assert _planted_row(source, "haraka_swap", planted, STRICT, on_group,
                        request)["marked_is_planted"] is True
    elsewhere = {(STRICT, 1): trace(1, groups=[3])}
    assert _planted_row(source, "haraka_swap", planted, STRICT, elsewhere,
                        request)["marked_is_planted"] is False


class FakeMurajaText:
    """One ayah's Muraja reference, as :class:`AyahReferences` reads it."""

    def __init__(self, phonemes: str, printed: dict[int, str], groups: dict[int, tuple]):
        self._phonemes = phonemes
        self._printed = printed
        self._groups = groups

    def phonemes(self, surah: int, ayah: int) -> str:
        return self._phonemes

    def printed(self, surah: int, ayah: int, word: int) -> str:
        return self._printed[word]

    def word_groups(self, surah: int, ayah: int, word: int) -> tuple:
        return self._groups[word]


def test_the_loader_marks_an_internal_printed_pause_sign_and_keeps_both_word_identities():
    # Two canonical words recited as one engine word, and an internal pause sign on engine
    # word 2's printed text (U+06D6), as 2:5's رَّبِّهِمۡۖ carries it.
    text = FakeMurajaText("ءَبَ تُ ثَ", {1: "ءَبَ", 2: "تُۖ", 3: "ثَ"},
                          {1: ((0, 2), (2, 4)), 2: ((5, 7),), 3: ((8, 10),)})
    references = AyahReferences(text, lambda label: (
        ("ءَ", "بَ", "تُ", "ثَ"), "ءَبَتُثَ", (0, 2, 4, 6, 8)))
    record = references("1:1")
    assert record.engine_words == ("ءَبَ", "تُ", "ثَ")
    assert record.word_of == (1, 1, 2, 3)
    assert record.cluster_groups == ((1, 0), (1, 1), (2, 0), (3, 0))
    assert record.waqf == frozenset({1})
    (loaded,) = tlog_clips(
        [{"audio_path": "c.flac", "filename_label": "1:1"}],
        {"c.flac#0": {"aligned": True,
                      "word_times": [[0.1, 0.5], [0.5, 0.9], [0.9, 1.3], [1.3, 1.7]]}},
        {"c.flac#0": 8 * SAMPLE_RATE}, references)
    assert loaded.waqf == frozenset({1})
    assert word_position(1, len(loaded.engine_words), waqf=1 in loaded.waqf) == "waqf"


# --- the App arm's oracle section -------------------------------------------------------------
def cell(flagged: bool) -> dict:
    return {"graded": True, "flagged": flagged, "causes": []}


def test_the_oracle_summary_counts_the_oracles_own_flags_and_the_App_flags_it_shares():
    words = [{"item": "a", "word": 1, "strict": cell(False)},
             {"item": "a", "word": 2, "strict": cell(True)},
             {"item": "b", "word": 1, "strict": cell(True)}]
    app = [{"item": "a", "word": 2, "strict": cell(True)},
           {"item": "b", "word": 1, "strict": cell(False)}]
    summary = oracle_summary(words, app)
    assert (summary["flagged"], summary["shared"], summary["app_flagged"]) == (2, 1, 1)
    assert summary["share"] == 1.0

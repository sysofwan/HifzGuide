"""The scorer's weighted rates against a hand-computed toy, and the edge words (#311 PR2).

Committed before the PR2 code exists and **locked**: later turns may add tests but never weaken or
rewrite these. They pin prereg §7 and amendment §10 and the edge erratum.

Contract this file pins (``tadabur.followalong_audit.s5_score``)
-----------------------------------------------------------------
A *record* is one first-verdict listened word: ``{"item", "stratum", "cell", "pi", "verdict",
"app_flagged", "ayah_edge"}`` (``pi`` is the key's ``inclusion_probability``; ``verdict`` the five
top-level verdicts; ``app_flagged`` is the App's kept outcome ``flagged``, so a passed *or* ungraded
word is ``False``; ``ayah_edge`` marks the clip's word 1 and highest word).

* ``false_flags_per_1000(records, *, without_edges=False) -> float | None``:
  ``1000 · Σ π⁻¹·F / Σ π⁻¹`` over records whose verdict is ``correct`` (all strata, S4 included),
  ``F`` = ``app_flagged``; ``None`` when no such record.
* ``misses_per_100(records, *, scope="headline", without_edges=False) -> float | None``:
  ``100 · Σ π⁻¹ [mistake, not flagged] / Σ π⁻¹ [mistake]`` over the scope's records: ``headline`` is
  H = S1 ∪ S3 ∪ S5-interior (stratum ``S5`` with cell ``interior``), ``s4`` S4 only, ``s5_edge``
  S5 cells ``leading``/``trailing``, ``full`` every stratum. ``None`` when no mistake in scope.
* ``rate_interval(records, statistic, *, seed, iterations=4000) -> [low, high]``: the 95% percentile
  bootstrap resampling whole clips (``item``) with
  :func:`training.decode_evalset.cluster_bootstrap`; ``statistic`` maps a list of records to a
  float. Reproducible from ``seed``.
* ``ayah_edge_words(app_grades) -> {(item, word)}``: per clip, the words in the App word table that
  are word 1 and the clip's highest word number (the erratum); a one-word ayah's single word.
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.s5_score import (
    ayah_edge_words,
    false_flags_per_1000,
    misses_per_100,
    rate_interval,
)


def rec(item, stratum, pi, verdict, flagged, *, cell="all", edge=False) -> dict:
    return {"item": item, "stratum": stratum, "cell": cell, "pi": pi, "verdict": verdict,
            "app_flagged": flagged, "ayah_edge": edge}


#: weights: S3 4, S4 100, S1 2, S5-interior 20, S5-leading 10
FALSE_FLAG_TOY = [
    rec("a", "S3", 0.25, "correct", True),                  # a false flag, weight 4
    rec("a", "S3", 0.25, "correct", False),
    rec("b", "S3", 0.25, "correct", False, edge=True),      # an ayah edge word
    rec("c", "S4", 0.01, "correct", False),                 # weight 100, never flagged
    rec("d", "S1", 0.5, "correct", True),                   # two false flags, weight 2 each
    rec("d", "S1", 0.5, "correct", True),
    rec("d", "S1", 0.5, "mistake", True),                   # a catch: not in this rate
    rec("e", "S3", 0.25, "not sure", True),                 # excluded
    rec("e", "S3", 0.25, "unusable audio", False),
]

MISS_TOY = [
    rec("a", "S3", 0.25, "mistake", False),                              # a miss, weight 4
    rec("a", "S3", 0.25, "mistake", True),                               # a correct catch
    rec("b", "S1", 0.5, "mistake", True, cell="tashkeel:no"),            # a catch, weight 2
    rec("c", "S5", 0.05, "mistake", False, cell="interior"),             # a miss, weight 20
    rec("d", "S4", 0.01, "mistake", False),                              # a miss, weight 100
    rec("e", "S5", 0.1, "mistake", False, cell="leading"),               # a miss, weight 10
    rec("e", "S3", 0.25, "correct", False),                              # not a mistake
]


def test_the_false_flag_rate_is_the_inverse_probability_weighted_ratio_per_thousand():
    # numerator 4*1 + 2*2 = 8; denominator 4*3 + 100 + 2*2 = 116
    assert false_flags_per_1000(FALSE_FLAG_TOY) == pytest.approx(1000 * 8 / 116)


def test_the_unweighted_ratio_is_not_the_rate():
    assert false_flags_per_1000(FALSE_FLAG_TOY) != pytest.approx(1000 * 3 / 6)


def test_without_the_ayah_edge_words_the_rate_drops_them_from_both_sides():
    # the edge word leaves the denominator only: 8 / 112
    assert false_flags_per_1000(FALSE_FLAG_TOY, without_edges=True) == pytest.approx(1000 * 8 / 112)


def test_a_stratum_that_never_flags_still_enters_the_denominator():
    only_s4 = [rec("a", "S4", 0.01, "correct", False), rec("b", "S3", 0.25, "correct", True)]
    assert false_flags_per_1000(only_s4) == pytest.approx(1000 * 4 / 104)


def test_no_correct_word_is_no_false_flag_rate():
    assert false_flags_per_1000([]) is None
    assert false_flags_per_1000([rec("a", "S3", 0.25, "mistake", True)]) is None


def test_the_headline_miss_rate_is_on_s1_s3_and_the_s5_interior_only():
    # numerator 4 + 20 = 24; denominator 4*2 + 2 + 20 = 30
    assert misses_per_100(MISS_TOY) == pytest.approx(80.0)
    assert misses_per_100(MISS_TOY, scope="headline") == pytest.approx(80.0)


def test_s4_and_the_s5_edge_are_separate_labelled_bounds():
    assert misses_per_100(MISS_TOY, scope="s4") == pytest.approx(100.0)
    assert misses_per_100(MISS_TOY, scope="s5_edge") == pytest.approx(100.0)


def test_the_full_frame_sensitivity_includes_every_stratum():
    # numerator 24 + 100 + 10 = 134; denominator 30 + 100 + 10 = 140
    assert misses_per_100(MISS_TOY, scope="full") == pytest.approx(100 * 134 / 140)


def test_a_catch_is_in_the_denominator_and_never_the_numerator():
    only_catch = [rec("a", "S3", 0.25, "mistake", True)]
    assert misses_per_100(only_catch) == pytest.approx(0.0)


def test_the_miss_rate_without_edges_drops_edge_mistakes_from_both_sides():
    records = [*MISS_TOY, rec("f", "S3", 0.25, "mistake", False, edge=True)]
    assert misses_per_100(records) == pytest.approx(100 * 28 / 34)
    assert misses_per_100(records, without_edges=True) == pytest.approx(80.0)


def test_no_mistake_in_scope_is_no_miss_rate():
    assert misses_per_100([], scope="s4") is None
    assert misses_per_100([rec("a", "S3", 0.25, "mistake", False)], scope="s4") is None


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError):
        misses_per_100(MISS_TOY, scope="everything")


# --- the interval -------------------------------------------------------------------------------
def test_the_interval_resamples_whole_clips_not_words():
    # two clips, one never flagged, one always: a clip resample is 0, 500 or 1000 per thousand,
    # so the interval spans 0..1000, which a word-level resample of 10 words would not reach
    records = ([rec("quiet", "S3", 0.25, "correct", False) for _ in range(5)]
               + [rec("loud", "S3", 0.25, "correct", True) for _ in range(5)])
    low, high = rate_interval(records, false_flags_per_1000, seed=7, iterations=400)
    assert (low, high) == (pytest.approx(0.0), pytest.approx(1000.0))


def test_clips_that_agree_give_a_collapsed_interval():
    records = [rec(f"clip{n}", "S3", 0.25, "correct", True) for n in range(6)]
    low, high = rate_interval(records, false_flags_per_1000, seed=7, iterations=200)
    assert low == pytest.approx(1000.0) == pytest.approx(high)


def test_the_interval_is_reproducible_and_depends_on_the_seed():
    records = [rec(f"clip{n}", "S3", 0.25, "correct", n % 3 == 0) for n in range(30)]
    first = rate_interval(records, false_flags_per_1000, seed=11, iterations=300)
    assert rate_interval(records, false_flags_per_1000, seed=11, iterations=300) == first
    assert rate_interval(records, false_flags_per_1000, seed=12, iterations=300) != first


# --- the edge words (erratum) --------------------------------------------------------------------
def grades(item, count) -> list[dict]:
    return [{"item": item, "word": word, "strict": {}, "shipped_default": {}}
            for word in range(1, count + 1)]


def test_the_edge_words_are_each_clips_word_one_and_its_highest_word():
    table = grades("a#0", 5) + grades("b#0", 3)
    assert ayah_edge_words(table) == {("a#0", 1), ("a#0", 5), ("b#0", 1), ("b#0", 3)}


def test_a_one_word_ayah_loses_its_only_word():
    assert ayah_edge_words(grades("a#0", 1)) == {("a#0", 1)}


def test_the_edge_words_do_not_depend_on_what_the_app_graded():
    graded = [{"item": "a#0", "word": word, "strict": {"graded": word == 2},
               "shipped_default": {}} for word in range(1, 5)]
    assert ayah_edge_words(graded) == {("a#0", 1), ("a#0", 4)}

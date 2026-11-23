"""The candidate strata S4 draws from (#309): total, single-valued, and the pools they size.

A word is ``{"key", "position", "app", "oracle", "detectors", "causes"}``:

* ``app`` is the App arm's strict outcome: ``flagged`` / ``passed`` / ``ungraded``;
* ``oracle`` is S2's outcome for the word (``flagged`` / ``passed``), or ``None`` while S2
  (#308) has not run;
* ``detectors`` is ``{"base", "c95", "r2_candidate_ema", "r3_candidate_ema": bool}`` — whether
  each detector flags the word, re-graded at 028bc51 in strict;
* ``causes`` are the App flag's causes (empty unless the App flagged).

Strata: S5 the App left ungraded; S1 the App flags and the oracle passes (``S1_PENDING`` until
S2 has run; ``ORACLE_EXPLAINED`` when the oracle flags too); S3 the App passes and the base
detector, or at least two detectors, flag; S4 nothing flags; ``RESIDUAL`` the one remaining case
(the App passes and exactly one non-base detector flags).

``s1_by_cause_and_base`` counts the S1 pool, which until S2 has run is ``S1_PENDING``: every
word the App flags and the oracle does not, once per cause, keyed ``(cause, base_flags)``.
"""

from __future__ import annotations

from itertools import product

from tadabur.followalong_audit.strata import (
    ORACLE_EXPLAINED,
    RESIDUAL,
    S1,
    S1_PENDING,
    S3,
    S4,
    S5,
    STRATA,
    assign_stratum,
    pool_sizes,
    s1_by_cause_and_base,
    stratify,
)

DETECTORS = ("base", "c95", "r2_candidate_ema", "r3_candidate_ema")


def detectors(*flagging: str) -> dict[str, bool]:
    return {name: name in flagging for name in DETECTORS}


def word(app: str, *flagging: str, oracle: str | None = None, causes=(), key="a#0",
         position=0) -> dict:
    return {"key": key, "position": position, "app": app, "oracle": oracle,
            "detectors": detectors(*flagging), "causes": list(causes)}


def test_every_combination_of_outcomes_gets_exactly_one_known_stratum():
    for app, oracle, flags in product(("flagged", "passed", "ungraded"), (None, "flagged", "passed"),
                                      product((False, True), repeat=4)):
        flagging = [name for name, on in zip(DETECTORS, flags) if on]
        assert assign_stratum(word(app, *flagging, oracle=oracle)) in STRATA


def test_an_ungraded_word_is_s5_whatever_else_says():
    assert assign_stratum(word("ungraded", "base", oracle="passed")) == S5


def test_an_app_flag_is_s1_when_the_oracle_passes_and_pending_until_s2_has_run():
    assert assign_stratum(word("flagged", oracle="passed")) == S1
    assert assign_stratum(word("flagged", "base", oracle="passed")) == S1
    assert assign_stratum(word("flagged", oracle=None)) == S1_PENDING


def test_an_app_flag_the_oracle_shares_is_explained_not_a_candidate():
    assert assign_stratum(word("flagged", oracle="flagged")) == ORACLE_EXPLAINED


def test_a_pass_the_base_detector_flags_is_s3():
    assert assign_stratum(word("passed", "base")) == S3


def test_a_pass_two_detectors_flag_is_s3_even_without_the_base_detector():
    assert assign_stratum(word("passed", "c95", "r3_candidate_ema")) == S3


def test_a_pass_nothing_flags_is_s4():
    assert assign_stratum(word("passed")) == S4


def test_a_pass_one_non_base_detector_flags_is_residual():
    assert assign_stratum(word("passed", "r2_candidate_ema")) == RESIDUAL


def test_the_strata_are_the_documented_seven():
    assert set(STRATA) == {S1, S1_PENDING, ORACLE_EXPLAINED, S3, S4, S5, RESIDUAL}


# --- the pools ---------------------------------------------------------------------------------

WORDS = [
    word("flagged", causes=["tashkeel"], position=0),
    word("flagged", "base", causes=["tashkeel", "soft_pair"], position=1),
    word("passed", "base", position=2),
    word("passed", position=3),
    word("passed", position=4),
    word("ungraded", position=5),
    word("passed", "c95", position=6),
]


def test_stratify_gives_each_word_one_stratum_and_marks_the_catch_loss_words():
    rows = stratify(WORDS, catch_loss={("a#0", 2), ("a#0", 3)})
    assert [row["stratum"] for row in rows] == [
        S1_PENDING, S1_PENDING, S3, S4, S4, S5, RESIDUAL]
    assert [row["catch_loss"] for row in rows] == [False, False, True, True, False, False, False]
    assert [(row["key"], row["position"]) for row in rows] == [("a#0", p) for p in range(7)]


def test_pool_sizes_lists_every_stratum_and_sums_to_the_words():
    sizes = pool_sizes(stratify(WORDS, catch_loss={("a#0", 2), ("a#0", 3)}))
    assert set(sizes["strata"]) == set(STRATA)
    assert sum(sizes["strata"].values()) == len(WORDS)
    assert sizes["strata"][S4] == 2 and sizes["strata"][S3] == 1 and sizes["strata"][S5] == 1
    assert sizes["catch_loss"][S3] == 1 and sizes["catch_loss"][S4] == 1
    assert sum(sizes["catch_loss"].values()) == 2


def test_s1_candidates_are_counted_by_cause_and_whether_the_base_detector_flags():
    counts = s1_by_cause_and_base(stratify(WORDS, catch_loss=set()))
    assert counts == {("tashkeel", False): 1, ("tashkeel", True): 1, ("soft_pair", True): 1}


def test_s1_candidates_exclude_words_the_oracle_explains():
    explained = word("flagged", oracle="flagged", causes=["tashkeel"])
    assert s1_by_cause_and_base(stratify([explained], catch_loss=set())) == {}

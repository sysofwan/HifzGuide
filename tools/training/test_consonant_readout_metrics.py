"""The consonant round's new metrics over synthetic grades (#175, bars from #174).

Torch-free and harness-free: items, decodes and word grades are synthetic. The metrics are the
cause filter on false flags, agreement on the teacher-departure subset, the per-substitution
paired metric at 99.5 %, and the catch arm's paired count.
"""

from __future__ import annotations

import numpy as np
import pytest

from tadabur.qud_text import ItemReference
from training import aug_readout_metrics as am
from training import consonant_readout as cr
from training.acceptance_stats import NO_BOUND
from training.external_corpus import WordGrade
from training.muraja_policy import SHIPPED_DEFAULT, SiteGrade
from training.nonpro_eval import ClipItem
from training.word_flags import Flag

SHIPPED = SHIPPED_DEFAULT.name
REFERENCE = "قَالَ سَططَ"
CLEAN = REFERENCE.replace(" ", "")
SWAPPED = "قَالَشَططَ"  # the second word's س heard as ش


def _item(n: int) -> ClipItem:
    name = f"c{n:03d}.wav"
    return ClipItem(f"{name}#0", "qdat", name, 5, 109, 0, 16000, "e" * 64,
                    ItemReference(("قَالَ", "سَططَ"), (7, 8), REFERENCE, (0, 6)),
                    f"r{n % 12}", n % 12, {"final_label": "in_correct"})


def _grade(model: str, item: ClipItem, position: int, flag: Flag | None) -> WordGrade:
    outcome = "flagged" if flag is not None else "not_flagged"
    return WordGrade(model, SHIPPED, item, position, 2, position + 1, 7 + position,
                     SiteGrade(outcome), flag)


def _flag(cause: str, substitution: str) -> Flag:
    return Flag((cause,), (), "minor", (), consonants=((cause, substitution),))


#: A flagged word carrying one consonant substitution.
FLAG = Flag(("other_consonant",), (), "minor", (), consonants=(("other_consonant", "س→ش"),))


# --- the cause filter on false flags -----------------------------------------------------------


def _cause_grades() -> list[WordGrade]:
    """Four items, two words each: one flag per cause, and the madd alif alone."""
    flags = [(("other_consonant", "س→ش")), (("other_consonant", "ا→∅")),
             None, (("target_pair", "ط→ق")),
             (("soft_pair", "ن→ل")), None, None, None]
    items = [_item(i) for i in range(4)]
    out = []
    for model in ("candidate", "c95"):
        for i, item in enumerate(items):
            for position in (0, 1):
                flag = flags[2 * i + position]
                out.append(_grade(model, item, position,
                                  None if flag is None else _flag(*flag)))
    return out


def _totals(terms: am.Terms, model: str = "candidate") -> tuple[int, int]:
    num, den = terms.by_model[model]
    return int(num.sum()), int(den.sum())


def test_the_consonant_cause_counts_every_consonant_substitution_but_the_madd_alif():
    grades = _cause_grades()
    models = ("candidate", "c95")
    all_cause = am.false_flag_terms(grades, models, SHIPPED)
    consonant = am.false_flag_terms(grades, models, SHIPPED, cause="consonant")
    tashkeel = am.false_flag_terms(grades, models, SHIPPED, cause="tashkeel")
    assert _totals(all_cause) == (4, 8)
    # س→ش, ط→ق and ن→ل count; ا→∅ alone does not.
    assert _totals(consonant) == (3, 8)
    assert _totals(tashkeel) == (0, 8)
    # A filter never changes the denominator.
    assert all(int(t.by_model[m][1].sum()) == 8
               for t in (consonant, tashkeel) for m in models)


def test_the_tashkeel_cause_counts_only_the_flags_that_carry_it():
    item = _item(0)
    tashkeel = Flag(("tashkeel",), ("fatha dropped",), "tashkeel", ())
    grades = [_grade("candidate", item, 0, tashkeel), _grade("candidate", item, 1, None)]
    terms = am.false_flag_terms(grades, ("candidate",), SHIPPED, cause="tashkeel")
    assert _totals(terms) == (1, 2)
    assert _totals(am.false_flag_terms(grades, ("candidate",), SHIPPED,
                                       cause="consonant")) == (0, 2)


# --- teacher-departure agreement ----------------------------------------------------------------


def test_the_departure_and_equals_subsets_partition_the_pooled_agreement():
    items = [_item(0), _item(1), _item(2)]
    decodes = {"base": {items[0].key: SWAPPED, items[1].key: CLEAN, items[2].key: SWAPPED},
               "candidate": {items[0].key: SWAPPED, items[1].key: SWAPPED,
                             items[2].key: CLEAN}}
    models = ("candidate",)
    pooled = am.word_agreement_terms(items, decodes, models)
    departs = am.word_agreement_terms(items, decodes, models, subset="departs")
    equals = am.word_agreement_terms(items, decodes, models, subset="equals")
    # The teacher departs from the reference word on the second word of items 0 and 2.
    assert departs.by_model["candidate"][1].tolist() == [1, 0, 1]
    assert departs.by_model["candidate"][0].tolist() == [1, 0, 0]
    assert equals.by_model["candidate"][1].tolist() == [1, 2, 1]
    assert equals.by_model["candidate"][0].tolist() == [1, 1, 1]
    for model in models:
        assert np.array_equal(departs.by_model[model][0] + equals.by_model[model][0],
                              pooled.by_model[model][0])
        assert np.array_equal(departs.by_model[model][1] + equals.by_model[model][1],
                              pooled.by_model[model][1])
    with pytest.raises(ValueError, match="subset"):
        am.word_agreement_terms(items, decodes, models, subset="all")


# --- the per-substitution metric at 99.5 % -------------------------------------------------------


def test_the_substitution_metric_counts_only_the_words_that_carry_it():
    item = _item(0)
    grades = [_grade("candidate", item, 0, _flag("other_consonant", "س→ش")),
              _grade("candidate", item, 1, _flag("target_pair", "ط→ق")),
              _grade("c95", item, 0, None),
              _grade("c95", item, 1, _flag("target_pair", "ط→ق"))]
    terms = am.substitution_terms(grades, ("candidate", "c95"), SHIPPED, ("س→ش", "ط→ق"))
    assert set(terms) == {"س→ش", "ط→ق"}
    assert _totals(terms["س→ش"]) == (1, 2)
    assert _totals(terms["س→ش"], "c95") == (0, 2)
    assert _totals(terms["ط→ق"]) == (1, 2)
    assert _totals(terms["ط→ق"], "c95") == (1, 2)


def test_the_veto_interval_is_the_99_5_percent_bound_from_the_same_draws():
    assert am.VETO_QUANTILES == (0.0025, 0.9975)
    items = [_item(i) for i in range(24)]
    grades = [_grade(model, item, position, FLAG if flagged(i, position) else None)
              for model, flagged in (("candidate", lambda i, p: p == 1 and i % 6 == 0),
                                     ("c95", lambda i, p: p == 1 and i % 2 == 0))
              for i, item in enumerate(items) for position in (0, 1)]
    terms = am.false_flag_terms(grades, ("candidate", "c95"), SHIPPED)
    tight = am.compare(terms, am.PER_1000)["comparisons"]["candidate−c95"]
    wide = am.compare(terms, am.PER_1000,
                      quantiles=am.VETO_QUANTILES)["comparisons"]["candidate−c95"]
    assert tight["interval"]["method"] == wide["interval"]["method"] == "bootstrap"
    assert wide["point"] == tight["point"]
    assert wide["interval"]["lower"] <= tight["interval"]["lower"]
    assert wide["interval"]["upper"] >= tight["interval"]["upper"]


def test_a_substitution_view_carries_its_95_and_99_5_percent_paired_intervals():
    item = _item(0)
    grades = [_grade("candidate", item, 0, _flag("other_consonant", "س→ش")),
              _grade("candidate", item, 1, None),
              _grade("c95", item, 0, None),
              _grade("c95", item, 1, _flag("other_consonant", "س→ش"))]
    views = am.substitution_views(
        am.substitution_terms(grades, ("candidate", "c95"), SHIPPED, ("س→ش",)), am.PER_1000)
    difference = views["س→ش"]["comparisons"]["candidate−c95"]
    assert difference["interval"]["method"] in ("bootstrap", NO_BOUND)
    assert set(difference["veto_interval"]) == {"lower", "upper", "method", "reason"}
    assert views["س→ش"]["models"]["candidate"]["num"] == 1.0


# --- the catch arm's paired count ---------------------------------------------------------------


def test_the_catch_difference_is_the_paired_count_of_flagged_sites():
    assert cr.catch_difference(3, 5) == -2
    assert cr.catch_difference(5, 3) == 2
    assert cr.catch_difference(4, 4) == 0
    assert cr.catch_difference(0, 2) == -2

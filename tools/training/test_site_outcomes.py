"""The per-site outcomes C, A and F frozen in #84 (acceptance rules §1, §9), on synthetic sites.

Every mark is pinned for commit, empty and wrong, on both recitation sides, with the
substitution / insertion / unaligned / wrong-carrier rules §9 asks to freeze.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from tadabur.truth_sites import TruthSite, parse_site
from training.site_outcomes import (
    CORRECT_SIDE,
    DHAL_ZAH,
    MISTAKE_SIDE,
    MULTIPLE,
    TARGET_PAIRS,
    SiteOutcome,
    item_outcomes,
    side,
)
from training.tashkeel_eval import SUKUN_MARK

FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
KAF, TAA, BAA, LAM, QAF, HAA, MEEM, DAL, RAA, DHAL, ZAI, ZAH, NOON = (
    "ك", "ت", "ب", "ل", "ق", "ه", "م", "د", "ر",
    "ذ", "ز", "ظ", "ن",
)

KATABA = KAF + FATHA + TAA + FATHA + BAA + FATHA + LAM + FATHA  # كَتَبَلَ: fatha on ت at 2
KUTIBA = KAF + DAMMA + TAA + KASRA + BAA + DAMMA + LAM + FATHA  # damma at 0, kasra at 2
QULHU = QAF + DAMMA + LAM + HAA + DAMMA + MEEM + FATHA  # sukun on ل at 2
MADDA = MEEM + FATHA + DAL + DAL + FATHA + RAA + FATHA  # held د at 2
MADA = MEEM + FATHA + DAL + FATHA + RAA + FATHA  # not_held د at 2
DHAKARA = RAA + FATHA + DHAL + FATHA + KAF + FATHA + RAA + FATHA  # ذ at 2


def make(reference: str, index: int, mark: str, prescribed: str, heard: str, **extra) -> TruthSite:
    site = TruthSite(
        site_id=extra.pop("site_id", f"s{index}{mark}{heard}"),
        source="new_audit",
        assumes_competent_reciter=False,
        audio_filename="clip.wav",
        shard=None,
        start_sample=0,
        end_sample=None,
        audio_sha256=None,
        surah_ayah="1:1",
        reference=reference,
        reference_index=index,
        mark=mark,
        prescribed=prescribed,
        heard=heard,
        stratum="s",
        stratum_population=1,
    )
    site = replace(site, **extra)
    return parse_site(asdict(site), "test")  # every synthetic site is a valid truth site


def outcome(site: TruthSite, decode: str) -> SiteOutcome:
    return item_outcomes([site], decode)[site.site_id]


def cad(site: TruthSite, decode: str) -> tuple[bool, bool, bool]:
    o = outcome(site, decode)
    return o.commits, o.correct(site), o.flagged(site)


# --- sides -----------------------------------------------------------------------------


def test_sides_and_no_verdict():
    assert side(make(KATABA, 2, "fatha", "fatha", "fatha")) == CORRECT_SIDE
    assert side(make(KATABA, 2, "fatha", "fatha", "kasra")) == MISTAKE_SIDE
    assert side(make(KATABA, 2, "fatha", "fatha", "unclear")) is None
    assert side(make(KATABA, 2, "fatha", "fatha", "pending")) is None


def test_target_pairs_are_the_six_soft_pairs_and_dhal_zah():
    assert len(TARGET_PAIRS) == 7 and DHAL_ZAH in TARGET_PAIRS


# --- harakat: commit, empty, wrong, both sides -----------------------------------------


@pytest.mark.parametrize(
    "reference, index, mark",
    [(KATABA, 2, "fatha"), (KUTIBA, 0, "damma"), (KUTIBA, 2, "kasra")],
)
def test_each_haraka_commit_empty_and_wrong_on_the_correct_side(reference, index, mark):
    site = make(reference, index, mark, mark, mark)
    assert cad(site, reference) == (True, True, False)
    empty = reference[: index + 1] + reference[index + 2 :]
    assert cad(site, empty) == (False, False, False)  # an empty slot is never flagged
    other = FATHA if mark != "fatha" else KASRA
    wrong = reference[: index + 1] + other + reference[index + 2 :]
    assert cad(site, wrong) == (True, False, True)


@pytest.mark.parametrize(
    "reference, index, mark, heard, heard_char",
    [
        (KATABA, 2, "fatha", "kasra", KASRA),
        (KUTIBA, 0, "damma", "fatha", FATHA),
        (KUTIBA, 2, "kasra", "damma", DAMMA),
    ],
)
def test_each_haraka_on_the_mistake_side(reference, index, mark, heard, heard_char):
    site = make(reference, index, mark, mark, heard)
    said = reference[: index + 1] + heard_char + reference[index + 2 :]
    assert cad(site, said) == (True, True, True)  # followed the audio: flagged and correct
    assert cad(site, reference) == (True, False, False)  # the mushaf's mark: silent correction
    empty = reference[: index + 1] + reference[index + 2 :]
    assert cad(site, empty) == (False, False, False)


def test_two_distinct_marks_commit_multiple():
    site = make(KATABA, 2, "fatha", "fatha", "fatha")
    o = outcome(site, KAF + FATHA + TAA + FATHA + DAMMA + BAA + FATHA + LAM + FATHA)
    assert o.committed == MULTIPLE and (o.commits, o.correct(site), o.flagged(site)) == (
        True, False, True,
    )
    assert o.marks == ("fatha", "damma")


def test_a_wrong_or_missing_carrier_is_no_commit():
    site = make(KATABA, 2, "fatha", "fatha", "fatha")
    wrong_carrier = outcome(site, KAF + FATHA + LAM + FATHA + BAA + FATHA + LAM + FATHA)
    assert wrong_carrier == SiteOutcome(False, None)
    missing = outcome(site, KAF + FATHA + BAA + FATHA + LAM + FATHA)
    assert not missing.carrier_aligned and not missing.commits


# --- sukun -----------------------------------------------------------------------------


def test_sukun_without_a_sukun_class_never_commits():
    site = make(QULHU, 2, "sukun", "sukun", "sukun")
    assert cad(site, QULHU) == (False, False, False)


def test_sukun_with_a_spurious_haraka_and_with_the_explicit_mark():
    site = make(QULHU, 2, "sukun", "sukun", "sukun")
    spurious = QAF + DAMMA + LAM + KASRA + HAA + DAMMA + MEEM + FATHA
    assert cad(site, spurious) == (True, False, True)
    assert outcome(site, spurious).marks == ("kasra",)
    explicit = QAF + DAMMA + LAM + SUKUN_MARK + HAA + DAMMA + MEEM + FATHA
    assert cad(site, explicit) == (True, True, False)


def test_a_haraka_said_where_the_mushaf_has_sukun():
    site = make(QULHU, 2, "sukun", "sukun", "fatha")
    said = QAF + DAMMA + LAM + FATHA + HAA + DAMMA + MEEM + FATHA
    assert cad(site, said) == (True, True, True)
    assert cad(site, QULHU) == (False, False, False)


def test_sukun_said_by_mistake():
    # The mushaf has fatha on ت; the reciter said sukun.
    site = make(KATABA, 2, "fatha", "fatha", "sukun")
    assert cad(site, KATABA) == (True, False, False)
    assert cad(site, KAF + FATHA + TAA + BAA + FATHA + LAM + FATHA) == (False, False, False)


# --- shaddah (provisional) -------------------------------------------------------------


def test_held_shaddah_commit_and_drop():
    site = make(MADDA, 2, "shaddah", "held", "held")
    assert cad(site, MADDA) == (True, True, False)
    dropped = MEEM + FATHA + DAL + FATHA + RAA + FATHA
    assert outcome(site, dropped).committed == "not_held"
    assert cad(site, dropped) == (True, False, True)


def test_held_shaddah_not_held_by_the_reciter():
    site = make(MADDA, 2, "shaddah", "held", "not_held")
    assert cad(site, MEEM + FATHA + DAL + FATHA + RAA + FATHA) == (True, True, True)
    assert cad(site, MADDA) == (True, False, False)


def test_not_held_shaddah_commit_and_add_on_both_sides():
    correct = make(MADA, 2, "shaddah", "not_held", "not_held")
    assert cad(correct, MADA) == (True, True, False)
    assert cad(correct, MADDA) == (True, False, True)
    mistake = make(MADA, 2, "shaddah", "not_held", "held")
    assert cad(mistake, MADDA) == (True, True, True)
    assert cad(mistake, MADA) == (True, False, False)


def test_a_held_site_is_aligned_through_either_half_of_its_run():
    site = make(MADDA, 2, "shaddah", "held", "held")
    assert outcome(site, MEEM + FATHA + DAL + FATHA + RAA + FATHA).carrier_aligned


def test_shaddah_with_its_carrier_gone_is_no_commit():
    site = make(MADDA, 2, "shaddah", "held", "held")
    assert not outcome(site, MEEM + FATHA + RAA + FATHA + RAA + FATHA).commits


# --- consonant pairs -------------------------------------------------------------------


def test_pair_commit_empty_and_wrong_on_both_sides():
    correct = make(DHAKARA, 2, "ذ↔ز", DHAL, DHAL)
    assert cad(correct, DHAKARA) == (True, True, False)
    swapped = DHAKARA.replace(DHAL, ZAI)
    assert cad(correct, swapped) == (True, False, True)
    gap = RAA + FATHA + KAF + FATHA + RAA + FATHA
    assert cad(correct, gap) == (False, False, False)
    other_letter = DHAKARA.replace(DHAL, NOON)
    assert cad(correct, other_letter) == (True, False, True)  # outside the pair still commits
    mistake = make(DHAKARA, 2, "ذ↔ز", DHAL, ZAI)
    assert cad(mistake, swapped) == (True, True, True)
    assert cad(mistake, DHAKARA) == (True, False, False)


def test_the_dhal_zah_pair_is_scored_like_any_pair():
    site = replace(make(DHAKARA, 2, "ذ↔ز", DHAL, DHAL), mark=DHAL_ZAH)
    assert cad(site, DHAKARA.replace(DHAL, ZAH)) == (True, False, True)


def test_flagged_implies_commit_everywhere():
    sites = [
        make(KATABA, 2, "fatha", "fatha", "fatha"),
        make(QULHU, 2, "sukun", "sukun", "sukun"),
        make(DHAKARA, 2, "ذ↔ز", DHAL, DHAL),
    ]
    for site in sites:
        for decode in ("", KAF, site.reference[:3], site.reference):
            o = outcome(site, decode)
            assert not o.flagged(site) or o.commits


def test_item_outcomes_refuses_sites_of_two_items():
    with pytest.raises(ValueError, match="one item"):
        item_outcomes(
            [make(KATABA, 2, "fatha", "fatha", "fatha"), make(QULHU, 2, "sukun", "sukun", "sukun")],
            KATABA,
        )

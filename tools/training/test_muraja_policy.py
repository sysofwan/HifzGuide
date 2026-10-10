"""Muraja's grading replayed at truth sites (``muraja_policy``): every rule per mode, every
allowance on and off, and the ratchet over several cycles. Torch-free, synthetic sites."""

from __future__ import annotations

from dataclasses import replace

import pytest

from training.muraja_policy import (
    ALLOWANCES,
    BALANCED_SCORES,
    CORRECT,
    EVERY_WORD_ENDS,
    GAP,
    LENIENT_SCORES,
    MURAJA_REVISION,
    NOT_GRADED,
    PRESETS,
    RUN_ENDS,
    STRICT_SCORES,
    TODAY,
    WRONG,
    Cycle,
    CycleGrade,
    Quality,
    Swap,
    allowances_off,
    decode_span,
    evaluate,
    grade,
    grade_decode_sequence,
    grade_item,
    locate,
    single_decode_cycles,
    upgrades,
)
from training.site_outcomes import DHAL_ZAH, SiteOutcome, item_outcomes
from training.test_site_outcomes import (
    DAL,
    DAMMA,
    DHAKARA,
    DHAL,
    FATHA,
    KASRA,
    KATABA,
    MADA,
    MADDA,
    QULHU,
    ZAH,
    ZAI,
    make,
)

STRICT, BALANCED, LENIENT = PRESETS["strict"], PRESETS["balanced"], PRESETS["lenient"]
MODES = (STRICT, BALANCED, LENIENT)
WAW, YA, HAMZA, HAA, HA = "و", "ي", "ء", "ح", "ه"
SUKUN_MARK = "sukun"
SOFT = "ذ↔ز"
#: Two words: كَتَبَلَ then مَدَرَ; the first ends no recited run.
TWO_WORDS = KATABA + " " + MADA


def committed(mark: str | None, *marks: str) -> SiteOutcome:
    return SiteOutcome(True, mark, marks or ((mark,) if mark else ()))


def letter(char: str | None) -> SiteOutcome:
    return SiteOutcome(char is not None, char)


def allowance(name: str):
    return next(a for a in ALLOWANCES if a.name == name)


def by_mode(site, outcome, **extra) -> tuple[str, str, str]:
    return tuple(grade(site, outcome, mode, **extra) for mode in MODES)


# --- configuration ------------------------------------------------------------------------


def test_today_is_muraja_balanced_pinned_to_its_commit():
    assert TODAY is BALANCED and TODAY.revision == MURAJA_REVISION
    assert TODAY.tashkeel_errors and not TODAY.mask_minor and not TODAY.empty_slot_not_graded
    assert TODAY.soft_pairs == frozenset(
        {"ذ↔ز", "ت↔ط", "ض↔ظ", "ق↔ك", "س↔ص", "ح↔ه"}
    ) and DHAL_ZAH not in TODAY.soft_pairs
    assert TODAY.as_dict()["revision"] == MURAJA_REVISION


def test_presets_compose_the_scoring_parameters_of_each_mode():
    # FollowAlongTypes.swift:111-139
    assert (STRICT_SCORES.correct_threshold, STRICT_SCORES.minor_threshold) == (0.75, 0.50)
    assert (BALANCED_SCORES.correct_threshold, BALANCED_SCORES.minor_threshold) == (0.65, 0.40)
    assert (LENIENT_SCORES.correct_threshold, LENIENT_SCORES.minor_threshold) == (0.55, 0.30)
    assert LENIENT_SCORES.mismatch_best_credit == 0.2 and LENIENT_SCORES.realign_threshold == 0.55
    assert LENIENT_SCORES.lenient_sifat_boost and not LENIENT_SCORES.phoneme_gate_enabled
    assert STRICT.scores == STRICT_SCORES and not STRICT.soft_pairs and not STRICT.shaddah_suppression
    assert BALANCED.soft_pairs and BALANCED.shaddah_suppression and not BALANCED.suppress_haraka_drop
    assert LENIENT.suppress_haraka_drop and LENIENT.mask_minor and LENIENT.soft_pairs
    for mode in MODES:
        assert mode.haraka_drop_letters == frozenset("واءي")
        assert mode.final_at_waqf_tashkeel and mode.final_at_waqf_consonant
        assert mode.group_has_gap and mode.leading_assimilation


def test_every_allowance_switches_one_toggle_off_and_all_of_them_compose():
    names = [a.name for a in ALLOWANCES]
    assert len(names) == len(set(names)) == 13
    assert sum(name.startswith("soft_pair") for name in names) == 6
    for entry in ALLOWANCES:
        assert entry.switch_off(TODAY) != TODAY or entry.name == "suppress_haraka_drop"
    off = allowances_off(TODAY)
    assert not off.soft_pairs and not off.shaddah_suppression and not off.suppress_haraka_drop
    assert not off.haraka_drop_letters and not off.final_at_waqf_tashkeel
    assert not off.final_at_waqf_consonant and not off.group_has_gap and not off.leading_assimilation
    assert off.scores == TODAY.scores and off.tashkeel_errors


# --- tashkeel -------------------------------------------------------------------------------


def test_haraka_states_per_mode():
    site = make(KATABA, 2, "fatha", "fatha", "fatha")  # ت, not the final consonant
    assert by_mode(site, committed("fatha")) == (CORRECT,) * 3
    assert by_mode(site, committed("kasra")) == (WRONG,) * 3  # a wrong haraka, every mode
    assert by_mode(site, committed("multiple", "kasra", "fatha")) == (CORRECT,) * 3  # trailing
    assert by_mode(site, committed(None)) == (WRONG, WRONG, NOT_GRADED)  # missing haraka
    assert by_mode(site, SiteOutcome(False, None)) == (NOT_GRADED,) * 3  # no slot


@pytest.mark.parametrize("carrier", [WAW, YA, HAMZA])
def test_missing_haraka_on_a_long_vowel_carrier_is_never_flagged(carrier):
    site = make(carrier + FATHA + KATABA, 0, "fatha", "fatha", "fatha")
    exemption = allowance("haraka_drop_letters")
    assert exemption.affects(site) and not allowance("suppress_haraka_drop").affects(site)
    assert by_mode(site, committed(None)) == (NOT_GRADED,) * 3
    assert grade(site, committed(None), exemption.switch_off(BALANCED)) == WRONG
    assert grade(site, committed(None), exemption.switch_off(LENIENT)) == NOT_GRADED  # suppressHarakaDrop
    assert by_mode(site, committed("kasra")) == (WRONG,) * 3


def test_suppress_haraka_drop_is_lenients_alone():
    site = make(KATABA, 2, "fatha", "fatha", "fatha")
    assert allowance("suppress_haraka_drop").affects(site)
    assert grade(site, committed(None), replace(BALANCED, suppress_haraka_drop=True)) == NOT_GRADED
    assert grade(site, committed(None), allowance("suppress_haraka_drop").switch_off(LENIENT)) == WRONG


def test_sukun_is_heard_nil():
    site = make(QULHU, 2, "sukun", "sukun", "sukun")
    assert by_mode(site, committed(None)) == (CORRECT,) * 3
    assert by_mode(site, committed(SUKUN_MARK)) == (CORRECT,) * 3  # a residual, not a haraka
    assert by_mode(site, committed("damma")) == (WRONG,) * 3  # a haraka added at sukun
    haraka = make(KATABA, 2, "fatha", "fatha", "sukun")
    assert by_mode(haraka, committed(SUKUN_MARK)) == by_mode(haraka, committed(None))


def test_the_final_consonant_of_a_waqf_word_is_exempt_from_tashkeel():
    final = make(KATABA, 6, "fatha", "fatha", "fatha")  # ل of the item's last word
    waqf = allowance("final_at_waqf_tashkeel")
    assert waqf.affects(final)
    assert by_mode(final, committed("kasra")) == (NOT_GRADED,) * 3
    assert by_mode(final, committed(None)) == (NOT_GRADED,) * 3
    assert grade(final, committed("kasra"), waqf.switch_off(TODAY)) == WRONG
    assert grade(final, committed("fatha")) == NOT_GRADED  # the position is not checked
    pause = make(TWO_WORDS, 6, "fatha", "fatha", "fatha", stratum="waqf_boundary:waqf",
                 source="waqf_boundary")  # the first word, a heard pause after it
    assert waqf.affects(pause) and grade(pause, committed("kasra")) == NOT_GRADED


def test_inside_a_run_the_final_consonant_is_exempt_only_under_every_word_ends():
    inner = make(TWO_WORDS, 6, "fatha", "fatha", "fatha")  # ل, but the run goes on
    assert not allowance("final_at_waqf_tashkeel").affects(inner)
    assert grade(inner, committed("kasra")) == WRONG
    assert grade(inner, committed("kasra"), approximation=EVERY_WORD_ENDS) == NOT_GRADED


def test_a_trailing_connection_waw_is_never_scored():
    law = "ل" + FATHA + WAW  # لَو: Muraja trims a trailing bare و (WS:219-252)
    site = make(law, 2, "sukun", "sukun", "sukun")
    assert by_mode(site, committed("fatha")) == (NOT_GRADED,) * 3


def test_tashkeel_toggle_and_the_candidate_rule():
    site = make(KATABA, 2, "fatha", "fatha", "fatha")
    assert grade(site, committed("kasra"), replace(TODAY, tashkeel_errors=False)) == NOT_GRADED
    candidate = replace(TODAY, empty_slot_not_graded=True)
    assert grade(site, committed(None), candidate) == NOT_GRADED
    assert grade(site, committed("fatha"), candidate) == CORRECT
    assert grade(site, committed("kasra"), candidate) == WRONG
    assert grade(site, committed(SUKUN_MARK), candidate) == WRONG
    sukun = make(QULHU, 2, "sukun", "sukun", "sukun")
    assert grade(sukun, committed(SUKUN_MARK), candidate) == CORRECT
    assert grade(sukun, committed("damma"), candidate) == WRONG
    assert grade(sukun, committed(None), candidate) == NOT_GRADED


# --- consonants -----------------------------------------------------------------------------


def test_a_soft_pair_per_mode_and_per_pair():
    site = make(DHAKARA, 2, SOFT, DHAL, DHAL)
    soft = allowance(f"soft_pair {SOFT}")
    assert soft.affects(site)
    assert by_mode(site, letter(DHAL)) == (CORRECT,) * 3
    assert by_mode(site, letter(ZAI)) == (WRONG, NOT_GRADED, NOT_GRADED)
    assert grade(site, letter(ZAI), soft.switch_off(TODAY)) == WRONG
    other = next(a for a in ALLOWANCES if a.name.startswith("soft_pair") and a is not soft)
    assert not other.affects(site) and grade(site, letter(ZAI), other.switch_off(TODAY)) == NOT_GRADED


def test_a_soft_pair_is_forgiven_only_while_the_word_scores_at_least_065():
    # حَفِۦۦظ: three scored letters; ح→ه costs the word 1.075/3 (0.642 < 0.65) and the
    # local re-alignment does no better through the madd (WS:672-694).
    short = make("حَفِۦۦظ", 0, "ح↔ه", HAA, HAA)
    word = locate(short).word
    assert evaluate(word, {0: Swap(HA)}, False, BALANCED).score == pytest.approx((2 - 0.075) / 3)
    assert by_mode(short, letter(HA)) == (WRONG, WRONG, NOT_GRADED)
    longer = make("حَفِظَهُ", 0, "ح↔ه", HAA, HAA)
    assert by_mode(longer, letter(HA)) == (WRONG, NOT_GRADED, NOT_GRADED)


def test_dhal_zah_is_no_soft_pair():
    site = replace(make(DHAKARA, 2, SOFT, DHAL, DHAL), mark=DHAL_ZAH)
    assert by_mode(site, letter(ZAH)) == (WRONG, WRONG, NOT_GRADED)  # lenient: no phoneme gate


def test_a_deletion_is_a_gap_the_phoneme_gate_fires_on():
    site = make(DHAKARA, 2, SOFT, DHAL, DHAL)
    assert by_mode(site, letter(None)) == (WRONG, WRONG, NOT_GRADED)


def test_the_final_consonant_of_a_waqf_word_gets_full_credit():
    final = make(KATABA[:6] + DHAL + FATHA, 6, SOFT, DHAL, DHAL)  # كَتَبَذَ, the item's end
    credit = allowance("final_at_waqf_consonant")
    assert credit.affects(final) and not allowance("final_at_waqf_tashkeel").affects(final)
    replaced = replace(final, mark=DHAL_ZAH)
    assert by_mode(replaced, letter(ZAH)) == (NOT_GRADED,) * 3
    assert grade(replaced, letter(ZAH), credit.switch_off(TODAY)) == WRONG
    assert grade(final, letter(None)) == WRONG  # a gap gets no waqf credit


# --- shaddah --------------------------------------------------------------------------------


def test_an_in_word_geminate_held_as_one_consonant_per_mode():
    site = make(MADDA, 2, "shaddah", "held", "held")  # مَددَرَ
    suppression = allowance("shaddah_suppression")
    assert suppression.affects(site)
    assert by_mode(site, committed("held")) == (CORRECT,) * 3
    assert by_mode(site, committed("not_held")) == (WRONG, NOT_GRADED, NOT_GRADED)
    assert grade(site, committed("not_held"), suppression.switch_off(TODAY)) == WRONG


def test_a_collapsed_geminate_discards_its_groups_tashkeel_in_every_mode():
    site = make(MADDA, 3, "fatha", "fatha", "fatha")  # the fatha on مَددَرَ's second د
    gap_rule = allowance("group_has_gap")
    assert gap_rule.affects(site)
    decode = MADA.replace(DAL + FATHA, DAL + KASRA, 1)  # مَدِرَ: one د, a wrong haraka
    outcome = item_outcomes([site], decode)[site.site_id]
    assert outcome.geminate_collapsed
    assert by_mode(site, outcome) == (NOT_GRADED,) * 3
    assert grade(site, outcome, gap_rule.switch_off(TODAY)) == WRONG
    assert by_mode(site, committed("kasra")) == (WRONG,) * 3  # the geminate held: flagged


def test_a_word_initial_assimilated_geminate_is_unchecked_but_its_haraka_is():
    reference = "لل" + FATHA + "ذ" + KASRA + YA + "ن" + FATHA  # للَذِينَ
    held = make(reference, 0, "shaddah", "held", "held")
    rule = allowance("leading_assimilation")
    assert rule.affects(held) and allowance("shaddah_suppression").affects(held)
    assert by_mode(held, committed("not_held")) == (NOT_GRADED,) * 3
    assert grade(held, committed("not_held"), rule.switch_off(STRICT)) == WRONG
    haraka = make(reference, 1, "fatha", "fatha", "fatha")
    assert by_mode(haraka, committed("kasra")) == (WRONG,) * 3


def test_a_single_consonant_decoded_double_is_invisible():
    site = make(MADA, 2, "shaddah", "not_held", "not_held")
    assert by_mode(site, committed("held")) == (NOT_GRADED,) * 3
    assert by_mode(site, committed("not_held")) == (CORRECT,) * 3


# --- the ratchet over cycles -----------------------------------------------------------------


def cycles(site, *outcomes, end_word=None):
    return [Cycle({site.site_id: o}, end_word) for o in outcomes]


def test_one_clean_cycle_locks_the_word_correct():
    site = make(TWO_WORDS, 2, "fatha", "fatha", "fatha")
    wrong, right = committed("kasra"), committed("fatha")
    assert grade_item([site], cycles(site, wrong, wrong, right)).sites[site.site_id] == CORRECT
    assert grade_item([site], cycles(site, right, wrong, wrong)).sites[site.site_id] == CORRECT
    result = grade_item([site], cycles(site, wrong, wrong))
    assert result.sites[site.site_id] == WRONG
    assert result.words[0].quality == Quality.TASHKEEL_ERROR


def test_a_higher_rank_replaces_a_lower_one_and_never_the_reverse():
    tashkeel = make(MADDA, 3, "fatha", "fatha", "fatha")
    collapsed = item_outcomes([tashkeel], MADA)[tashkeel.site_id]  # مَدَرَ: minor in strict
    wrong = committed("kasra")  # tashkeelError
    for order in ((wrong, collapsed), (collapsed, wrong)):
        result = grade_item([tashkeel], cycles(tashkeel, *order), STRICT)
        assert result.words[0].quality == Quality.MINOR


def test_a_same_quality_grade_replaces_only_when_it_scores_more_than_001_higher():
    site = make(MADDA, 2, "shaddah", "held", "held")
    single, double = committed("not_held"), committed("held")
    # balanced: one consonant scores 0.75 (correct, suppressed); two score 1.0
    assert grade_item([site], cycles(site, single, double)).sites[site.site_id] == CORRECT
    assert grade_item([site], cycles(site, double, single)).sites[site.site_id] == CORRECT
    assert grade_item([site], cycles(site, single, single)).sites[site.site_id] == NOT_GRADED

    def made(quality, score):
        return CycleGrade(0, quality, score, False, None, {})

    kept = made(Quality.CORRECT, 0.80)
    assert not upgrades(kept, made(Quality.CORRECT, 0.81))
    assert upgrades(kept, made(Quality.CORRECT, 0.8101))
    assert not upgrades(made(Quality.WRONG, 0.1), made(Quality.UNCERTAIN, 0.9))  # same rank
    assert not upgrades(None, made(Quality.PENDING, 1.0))
    assert upgrades(None, made(Quality.SKIPPED, 0.0))


def test_the_end_word_holds_back_anything_but_correct():
    site = make(TWO_WORDS, 2, "fatha", "fatha", "fatha")
    wrong = committed("kasra")
    held = grade_item([site], cycles(site, wrong, wrong, end_word=0))
    assert held.words[0].quality == Quality.PENDING and held.sites[site.site_id] == NOT_GRADED
    later = grade_item([site], [*cycles(site, wrong, end_word=0), *cycles(site, wrong)])
    assert later.sites[site.site_id] == WRONG


def test_a_cycle_grades_only_the_words_it_reaches():
    first = make(TWO_WORDS, 2, "fatha", "fatha", "fatha")
    second = make(TWO_WORDS, 11, "fatha", "fatha", "fatha")  # مَدَرَ's د
    outcomes = {first.site_id: committed("kasra"), second.site_id: committed("kasra")}
    result = grade_item([first, second], [Cycle(outcomes, end_word=0)])
    assert set(result.words) == {0, 1} and result.words[1].kept is None
    assert result.sites[second.site_id] == NOT_GRADED


def test_single_decode_cycles_end_on_each_run_end():
    first = make(TWO_WORDS, 2, "fatha", "fatha", "fatha")
    last = make(TWO_WORDS, 11, "fatha", "fatha", "fatha")
    outcomes = {s.site_id: committed("fatha") for s in (first, last)}
    assert [c.end_word for c in single_decode_cycles([first, last], outcomes, RUN_ENDS)] == [1, None]
    every = single_decode_cycles([first, last], outcomes, EVERY_WORD_ENDS)
    assert [c.end_word for c in every] == [0, 1, None]
    with pytest.raises(ValueError):
        single_decode_cycles([first], outcomes, "nope")


def test_a_real_decode_sequence_is_placed_and_ratcheted():
    site = make(TWO_WORDS, 2, "fatha", "fatha", "fatha")
    wrong = KATABA.replace(FATHA, KASRA)[:4] + KATABA[4:]  # كِتِبَلَ: ت with a kasra
    partial = wrong[:4]  # the check has heard half the first word
    assert decode_span(TWO_WORDS, wrong) == (0, 0)
    assert decode_span(TWO_WORDS, wrong + " " + MADA) == (0, 1)
    result = grade_decode_sequence([site], [partial, wrong, wrong + " " + MADA[:2], wrong + " " + MADA])
    assert result.sites[site.site_id] == WRONG
    fixed = grade_decode_sequence([site], [wrong, TWO_WORDS])
    assert fixed.sites[site.site_id] == CORRECT
    assert grade_decode_sequence([site], [wrong], reciter_moved_on=False).sites[site.site_id] == NOT_GRADED


# --- several sites in one word ---------------------------------------------------------------


def test_a_site_is_wrong_only_when_its_own_deviation_shows():
    reference = "رَذَكَرَ"  # ذ at 2, ك at 4
    soft = make(reference, 2, SOFT, DHAL, DHAL)
    hard = make(reference, 4, "ق↔ك", "ك", "ك", site_id="hard")
    tashkeel = make(reference, 0, "fatha", "fatha", "fatha", site_id="ra")
    outcomes = {soft.site_id: letter(ZAI), hard.site_id: letter("ق"), tashkeel.site_id: committed("fatha")}
    strict_pair = replace(TODAY, soft_pairs=TODAY.soft_pairs - {"ق↔ك"})
    sites = [soft, hard, tashkeel]
    result = grade_item(sites, single_decode_cycles(sites, outcomes), strict_pair)
    assert result.words[0].quality == Quality.MINOR
    assert result.sites == {soft.site_id: NOT_GRADED, hard.site_id: WRONG, tashkeel.site_id: CORRECT}


# --- populations -----------------------------------------------------------------------------


def test_affected_populations_are_decided_by_the_truth_record():
    haraka = make(KATABA, 2, "fatha", "fatha", "fatha")
    sukun = make(QULHU, 2, "sukun", "sukun", "sukun")
    damma_on_waw = make(WAW + DAMMA + KATABA, 0, "damma", "damma", "fatha")
    assert {a.name for a in ALLOWANCES if a.affects(haraka)} == {"suppress_haraka_drop"}
    assert not any(a.affects(sukun) for a in ALLOWANCES)
    assert {a.name for a in ALLOWANCES if a.affects(damma_on_waw)} == {"haraka_drop_letters"}
    shaddah = make(MADDA, 2, "shaddah", "held", "held")
    assert {a.name for a in ALLOWANCES if a.affects(shaddah)} == {"shaddah_suppression"}
    assert {a.name for a in ALLOWANCES if a.affects(make(MADDA, 3, "fatha", "fatha", "fatha"))} == {
        "suppress_haraka_drop", "group_has_gap"
    }


def test_a_geminate_is_two_groups_and_its_gap_a_shaddah_gap():
    word = locate(make(MADDA, 2, "shaddah", "held", "held")).word
    assert word.groups == ((0, 1), (1, 2), (2, 3), (3, 4))  # مَ د دَ رَ: the geminate is two groups
    assert evaluate(word, {1: GAP}, False, BALANCED).quality == Quality.CORRECT
    assert evaluate(word, {1: GAP}, False, STRICT).quality == Quality.MINOR
    assert evaluate(word, {}, False, STRICT).quality == Quality.CORRECT

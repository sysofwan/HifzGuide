"""The fixture-side, decode-level report (#55) on synthetic P3.5 sites and decodes.

Torch-free. Pins the per-side split and its sign convention, the directional rows, the
refusal to pool, the pending reject side filling in once verdicts land, the support marks,
and the per-site records and fingerprints the paired diff (#57) keys on.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tadabur.eval_report import (
    AS_HEARD,
    AS_PARTNER,
    FAMILIES,
    INSUFFICIENT_EVIDENCE,
    NO_COMMIT,
    NO_SITE,
    OTHER_CONSONANT,
    SHOULD_ACCEPT,
    SHOULD_REJECT,
    SIGN_CONVENTIONS,
    SUFFICIENT,
    SUPPORTED,
    TOO_SMALL,
    FixtureSite,
    confusion_row,
    fixture_report,
    item_fingerprint,
    members,
)
from tadabur.listening_session import Verdict, adjudicated
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE, item_outcomes
from training.test_site_outcomes import DAL, DHAKARA, DHAL, FATHA, KAF, MADA, MADDA, MEEM, RAA, ZAI, make
from training.truth_scorer import ALL, HEADLINE, SAFEGUARD, Cell, item_key, score

PAIR = "ذ↔ز"
PAIR_STRATUM = f"p35_fixture:{PAIR}"
#: Decodes of DHAKARA (ذ at index 2): faithful, the partner, a third letter, and nothing there.
SAID_DHAL, SAID_ZAI, SAID_DAL, DROPPED = (
    DHAKARA,
    DHAKARA.replace(DHAL, ZAI),
    DHAKARA.replace(DHAL, DAL),
    RAA + FATHA + KAF + FATHA + RAA + FATHA,
)


def p35(clip: str, heard: str, reference: str = DHAKARA, mark: str = PAIR, prescribed: str = DHAL,
        stratum: str = PAIR_STRATUM, population: int = 4):
    return make(
        reference, 2, mark, prescribed, heard,
        site_id=f"p35_fixture:{clip}:{mark}#2", source="p35_fixture", audio_filename=clip,
        stratum=stratum, stratum_population=population,
    )


@pytest.fixture
def cohort():
    """Two accepts, one reject heard as the partner, one reject still pending; one shaddah accept."""
    sites = [
        p35("a.wav", DHAL),
        p35("b.wav", DHAL),
        p35("c.wav", ZAI),
        p35("d.wav", "pending"),
        p35("e.wav", "held", MADDA, "shaddah", "held", "p35_fixture:shadda", 1),
    ]
    sides = {s.site_id: SHOULD_ACCEPT for s in sites}
    sides[sites[2].site_id] = sides[sites[3].site_id] = SHOULD_REJECT
    reciters = {"a.wav": 1, "b.wav": 2, "c.wav": 3, "d.wav": 4, "e.wav": 5}
    return sites, sides, reciters


def decodes(sites, by_clip: dict[str, str]) -> dict[str, str]:
    return {item_key(s): by_clip.get(s.audio_filename, s.reference) for s in sites}


def models(arms) -> dict:
    """Each arm's identity and decode fingerprint, shaped as #84's caches record them."""
    decode = {"mode": "whole-spans", "weights_dtype": "bf16", "batch_size": 1}
    return {arm: {"model": arm, "identity": {"model_ref": arm, "hub_revision": "r"}, "decode": decode}
            for arm in arms}


def report_for(cohort, arms: dict[str, dict[str, str]]):
    sites, sides, reciters = cohort
    decoded = {arm: decodes(sites, by_clip) for arm, by_clip in arms.items()}
    return fixture_report(sites, sides, reciters, decoded, models(decoded))


def row(report, side, family, heard):
    return next(
        r for r in report["rows"] if (r["side"], r["family"], r["heard"]) == (side, family, heard)
    )


def test_every_direction_of_every_family_has_a_row_on_both_sides(cohort):
    report = report_for(cohort, {"m": {}})
    keys = [(r["side"], r["family"], r["heard"]) for r in report["rows"]]
    expected = [(s, f, h) for s in (CORRECT_SIDE, MISTAKE_SIDE) for f in FAMILIES for h in members(f)]
    assert keys == expected
    assert "ذ↔ظ" in FAMILIES  # in scope by owner decision (§7), with no P3.5 site
    assert row(report, CORRECT_SIDE, "ذ↔ظ", "ظ")["support"] == NO_SITE


def test_sides_follow_what_was_heard_and_record_the_fixture_they_came_from(cohort):
    report = report_for(cohort, {"m": {}})
    correct = row(report, CORRECT_SIDE, PAIR, DHAL)
    assert (correct["prescribed"], correct["sites"]) == (DHAL, 2)
    assert correct["fixture_sides"] == {SHOULD_ACCEPT: 2, SHOULD_REJECT: 0}
    mistake = row(report, MISTAKE_SIDE, PAIR, ZAI)  # the mushaf's ذ, said as ز
    assert (mistake["prescribed"], mistake["sites"]) == (DHAL, 1)
    assert mistake["fixture_sides"] == {SHOULD_ACCEPT: 0, SHOULD_REJECT: 1}
    pair = next(f for f in report["families"] if f["family"] == PAIR)
    assert pair["without_verdict"] == 1
    assert report["pending"] == [
        {"family": PAIR, "prescribed": DHAL, "heard": "pending", "fixture_side": SHOULD_REJECT,
         "sites": 1}
    ]


def test_one_committed_letter_reads_opposite_ways_on_the_two_sides(cohort):
    """The partner on a correct site is a mishearing; the mushaf's letter on a mistake site is
    a collapse. Both are AS_PARTNER, and each side's convention names what it means."""
    report = report_for(cohort, {"m": {"a.wav": SAID_ZAI, "b.wav": SAID_DHAL, "c.wav": SAID_DHAL}})
    correct = row(report, CORRECT_SIDE, PAIR, DHAL)["arms"]["m"]
    assert (correct[AS_HEARD]["sites"], correct[AS_PARTNER]["sites"]) == (1, 1)
    assert correct[AS_PARTNER]["point"] == 0.5
    mistake = row(report, MISTAKE_SIDE, PAIR, ZAI)["arms"]["m"]
    assert (mistake[AS_PARTNER]["sites"], mistake[AS_HEARD]["sites"]) == (1, 0)
    assert "misheard" in SIGN_CONVENTIONS[CORRECT_SIDE][AS_PARTNER]
    assert "collapsed onto the reference" in SIGN_CONVENTIONS[MISTAKE_SIDE][AS_PARTNER]


def test_every_role_is_read_off_the_84_outcome_and_a_row_sums_to_one(cohort):
    sites, sides, reciters = cohort
    extra = [p35(f"x{i}.wav", DHAL) for i in range(4)]
    sites = sites + extra
    sides = {**sides, **{s.site_id: SHOULD_ACCEPT for s in extra}}
    reciters = {**reciters, **{s.audio_filename: 10 + i for i, s in enumerate(extra)}}
    clips = {"b.wav": SAID_ZAI, "x0.wav": SAID_DAL, "x1.wav": DROPPED}
    report = fixture_report(sites, sides, reciters, {"m": decodes(sites, clips)}, models(["m"]))
    rates = row(report, CORRECT_SIDE, PAIR, DHAL)["arms"]["m"]
    assert {r: rates[r]["sites"] for r in rates} == {
        AS_HEARD: 3, AS_PARTNER: 1, OTHER_CONSONANT: 1, NO_COMMIT: 1,  # a, x2, x3 decode faithfully
    }
    assert sum(rates[r]["point"] for r in rates) == pytest.approx(1.0)
    assert rates[AS_HEARD]["den"] == pytest.approx(6 * 4 / 8)  # stratum population 4, 8 sampled


def test_shaddah_rows_have_no_third_letter_and_are_provisional(cohort):
    gone = MEEM + FATHA + RAA + FATHA
    report = report_for(cohort, {"held": {}, "dropped": {"e.wav": MADA}, "gone": {"e.wav": gone}})
    held = row(report, CORRECT_SIDE, "shaddah", "held")
    assert held["provisional"] and held["support"] == TOO_SMALL
    assert set(held["arms"]["held"]) == {AS_HEARD, AS_PARTNER, NO_COMMIT}
    assert held["arms"]["held"][AS_HEARD]["sites"] == 1
    assert held["arms"]["dropped"][AS_PARTNER]["sites"] == 1  # a dropped shaddah: misheard
    assert held["arms"]["gone"][NO_COMMIT]["sites"] == 1
    assert held["arms"]["held"][AS_HEARD]["method"] == "wilson"  # one site: independent, unit weight


def test_a_row_refuses_any_site_it_does_not_hold(cohort):
    """Pooling is impossible through the public API: a row is built only from its own cell."""
    sites, sides, reciters = cohort
    fixture = {s.site_id: FixtureSite(s, sides[s.site_id], reciters[s.audio_filename], 1.0)
               for s in sites}
    outcomes = {"m": {s.site_id: item_outcomes([s], s.reference)[s.site_id] for s in sites}}
    correct = Cell(SAFEGUARD, CORRECT_SIDE, PAIR, DHAL)
    accepts = [fixture[s.site_id] for s in sites[:2]]
    assert confusion_row(correct, accepts, [], outcomes)["sites"] == 2
    for stray in (sites[2], sites[3]):  # a real mistake; a pending site with no side
        with pytest.raises(ValueError, match="never pooled"):
            confusion_row(correct, accepts + [fixture[stray.site_id]], [], outcomes)
    with pytest.raises(ValueError, match="never pooled"):  # nor the other direction
        confusion_row(Cell(SAFEGUARD, MISTAKE_SIDE, PAIR, DHAL), [fixture[sites[2].site_id]], [],
                      outcomes)
    with pytest.raises(ValueError, match="never pooled"):  # an exclusion must lack a verdict
        confusion_row(correct, accepts, [fixture[sites[2].site_id]], outcomes)


@pytest.mark.parametrize("cell", [
    Cell(SAFEGUARD, CORRECT_SIDE, ALL, ALL),  # every family and direction of a side
    Cell(SAFEGUARD, MISTAKE_SIDE, ALL, ALL),
    Cell(SAFEGUARD, CORRECT_SIDE, PAIR, ALL),
    Cell(SAFEGUARD, CORRECT_SIDE, PAIR, "held"),  # not a value of the family
    Cell(SAFEGUARD, CORRECT_SIDE, "fatha", "fatha"),  # not a pair or shaddah family
    Cell(SAFEGUARD, "both", PAIR, DHAL),  # not a side
    Cell(HEADLINE, CORRECT_SIDE, PAIR, DHAL),  # not the P3.5 population
])
def test_a_row_is_only_ever_one_direction_of_one_family_on_one_side(cohort, cell):
    """A pooled ``all`` cell would hold both directions and every family: refused outright."""
    sites, sides, reciters = cohort
    accepts = [FixtureSite(s, sides[s.site_id], reciters[s.audio_filename], 1.0) for s in sites[:2]]
    outcomes = {"m": {s.site_id: item_outcomes([s], s.reference)[s.site_id] for s in sites}}
    with pytest.raises(ValueError, match="never pooled"):
        confusion_row(cell, accepts if cell.family == ALL else [], [], outcomes)


def test_sites_without_a_verdict_give_each_rate_its_section_1_range():
    """One faithful site and one compatible ``unclear`` one: 100% as measured, 50–100% once
    the unclear site is counted in, the same range #84's scorer gives the cell."""
    sites = [p35("a.wav", DHAL, population=2), p35("u.wav", "unclear", population=2)]
    sides = {s.site_id: SHOULD_ACCEPT for s in sites[:1]} | {sites[1].site_id: SHOULD_REJECT}
    reciters = {"a.wav": 1, "u.wav": 2}
    arms = {"m": decodes(sites, {})}
    report = fixture_report(sites, sides, reciters, arms, models(arms))
    correct = row(report, CORRECT_SIDE, PAIR, DHAL)
    assert correct["excluded"] == [sites[1].site_id]
    faithful = correct["arms"]["m"][AS_HEARD]
    assert faithful["point"] == 1.0
    assert faithful["sensitivity"] == {"worst": 0.5, "best": 1.0}
    assert correct["arms"]["m"][AS_PARTNER]["sensitivity"] == {"worst": 0.0, "best": 0.5}
    canonical = next(
        c for c in score(sites, reciters, arms)["cells"]
        if (c["population"], c["side"], c["label"]) == (SAFEGUARD, CORRECT_SIDE, f"{PAIR}:{DHAL}")
    )
    assert canonical["arms"]["m"]["sensitivity"]["commit_rate"] == faithful["sensitivity"]
    # The unclear site could also be the mistake ز said for ذ; it is listed there too.
    assert row(report, MISTAKE_SIDE, PAIR, ZAI)["excluded"] == [sites[1].site_id]
    assert row(report, CORRECT_SIDE, PAIR, ZAI)["excluded"] == []


def test_a_mistake_side_decode_never_moves_a_correct_side_number(cohort):
    before = report_for(cohort, {"m": {"c.wav": SAID_ZAI}})
    after = report_for(cohort, {"m": {"c.wav": SAID_DHAL}})

    def correct_rows(report):
        return [r for r in report["rows"] if r["side"] == CORRECT_SIDE]

    assert correct_rows(before) == correct_rows(after)
    assert row(before, MISTAKE_SIDE, PAIR, ZAI) != row(after, MISTAKE_SIDE, PAIR, ZAI)
    assert not {"pooled", "total", "discrimination", "recall"} & set(before)


def test_a_collapse_hidden_behind_an_unchanged_pooled_matrix_shows_on_both_sides():
    """ADR-0001's failure mode. Model A mishears both correct sites and collapses both
    mistakes; model B is right everywhere. A matrix pooled over the two sides (the old
    report) is identical for the two; the per-side rows move in opposite directions."""
    sites = [p35("a.wav", DHAL), p35("b.wav", DHAL), p35("c.wav", ZAI), p35("d.wav", ZAI)]
    sides = {s.site_id: SHOULD_ACCEPT if s.heard == DHAL else SHOULD_REJECT for s in sites}
    reciters = {"a.wav": 1, "b.wav": 2, "c.wav": 3, "d.wav": 4}
    arms = {
        "A": decodes(sites, {"a.wav": SAID_ZAI, "b.wav": SAID_ZAI, "c.wav": SAID_DHAL, "d.wav": SAID_DHAL}),
        "B": decodes(sites, {"a.wav": SAID_DHAL, "b.wav": SAID_DHAL, "c.wav": SAID_ZAI, "d.wav": SAID_ZAI}),
    }
    report = fixture_report(sites, sides, reciters, arms, models(arms))

    def pooled(arm):  # the mushaf's ذ in every site, both sides summed
        return sorted(r["arms"][arm]["committed"] for r in report["sites"])

    assert pooled("A") == pooled("B")
    correct = row(report, CORRECT_SIDE, PAIR, DHAL)["arms"]
    mistake = row(report, MISTAKE_SIDE, PAIR, ZAI)["arms"]
    assert (correct["A"][AS_HEARD]["point"], correct["B"][AS_HEARD]["point"]) == (0.0, 1.0)
    assert (mistake["A"][AS_PARTNER]["point"], mistake["B"][AS_PARTNER]["point"]) == (1.0, 0.0)


def test_the_reject_side_fills_in_when_site_level_verdicts_land(cohort):
    sites, sides, reciters = cohort
    before = report_for(cohort, {"m": {}})
    assert row(before, MISTAKE_SIDE, PAIR, ZAI)["sites"] == 1
    heard_wrong = adjudicated(sites, {sites[3].site_id: Verdict(sites[3].site_id, ZAI)})
    after = report_for((heard_wrong, sides, reciters), {"m": {}})
    assert row(after, MISTAKE_SIDE, PAIR, ZAI)["sites"] == 2
    assert row(after, MISTAKE_SIDE, PAIR, ZAI)["fixture_sides"][SHOULD_REJECT] == 2
    assert after["pending"] == []
    assert after["fingerprints"]["fixtures"] != before["fingerprints"]["fixtures"]
    # A reject heard as the mushaf's letter is correct recitation, whatever its fixture said.
    heard_right = adjudicated(sites, {sites[3].site_id: Verdict(sites[3].site_id, DHAL)})
    right = report_for((heard_right, sides, reciters), {"m": {}})
    assert row(right, CORRECT_SIDE, PAIR, DHAL)["fixture_sides"] == {SHOULD_ACCEPT: 2, SHOULD_REJECT: 1}


def test_small_rows_are_marked_and_a_family_needs_support_on_both_sides():
    sites, sides, reciters = [], {}, {}
    for heard, fixture in ((DHAL, SHOULD_ACCEPT), (ZAI, SHOULD_REJECT)):
        for i in range(20):
            site = p35(f"{heard}{i}.wav", heard, population=40)
            sites.append(site)
            sides[site.site_id] = fixture
            reciters[site.audio_filename] = i % 10

    def report(subset):
        return fixture_report(subset, sides, reciters, {"m": decodes(subset, {})}, models(["m"]))

    status = {f["family"]: f for f in report(sites)["families"]}
    assert (status[PAIR][CORRECT_SIDE], status[PAIR][MISTAKE_SIDE]) == (SUFFICIENT, SUFFICIENT)
    assert status[PAIR]["status"] == SUPPORTED
    assert status["ت↔ط"]["status"] == INSUFFICIENT_EVIDENCE
    assert status[PAIR]["without_verdict"] == 0
    one_sided = [s for s in sites if s.heard == DHAL]
    pair = next(f for f in report(one_sided)["families"] if f["family"] == PAIR)
    assert (pair[MISTAKE_SIDE], pair["status"]) == (NO_SITE, INSUFFICIENT_EVIDENCE)
    assert row(report(one_sided[:19]), CORRECT_SIDE, PAIR, DHAL)["support"] == TOO_SMALL


def test_site_records_and_fingerprints_for_the_paired_diff(cohort):
    sites, sides, reciters = cohort
    report = report_for(cohort, {"base": {}, "cand": {"a.wav": SAID_ZAI}})
    records = report["sites"]
    assert [r["site_id"] for r in records] == sorted(s.site_id for s in sites)
    a = next(r for r in records if r["item"] == item_key(sites[0]))
    assert a["item_fingerprint"] == item_fingerprint(sites[0])
    assert a["arms"]["cand"] == {"committed": ZAI, "role": AS_PARTNER}
    pending = next(r for r in records if r["heard"] == "pending")
    assert pending["side"] is None and pending["arms"]["base"]["role"] is None
    arms = report["fingerprints"]["arms"]
    assert arms["base"]["decodes_sha256"] != arms["cand"]["decodes_sha256"]
    assert arms["base"]["identity"] == {"model_ref": "base", "hub_revision": "r"}
    assert report_for(cohort, {"base": {}, "cand": {"a.wav": SAID_ZAI}}) == report  # deterministic
    moved = replace(sites[0], reference=DHAKARA + RAA)
    assert item_fingerprint(moved) != item_fingerprint(sites[0])


def test_inputs_are_checked(cohort):
    sites, sides, reciters = cohort
    arms = {"m": decodes(sites, {})}
    with pytest.raises(ValueError, match="no fixture side"):
        fixture_report(sites, {}, reciters, arms, models(arms))
    with pytest.raises(ValueError, match="heard as prescribed"):
        fixture_report(sites, {**sides, sites[2].site_id: SHOULD_ACCEPT}, reciters, arms, models(arms))
    with pytest.raises(ValueError, match="not a P3.5"):
        fixture_report([replace(sites[0], source="new_audit")], sides, reciters, arms, models(arms))
    with pytest.raises(ValueError, match="model fingerprint"):
        fixture_report(sites, sides, reciters, arms, models(["m", "other"]))
    with pytest.raises(ValueError, match="no decode"):  # #84: a site never drops out of an arm
        fixture_report(sites, sides, reciters, {"m": {}}, models(["m"]))

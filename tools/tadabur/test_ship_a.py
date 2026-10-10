"""Ship A's sites and reciter split (#126): the mining rules and the committed files."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from tadabur.listening_session import POOL_CLIPS_PATH
from tadabur.mining_pool import PoolClip, PoolSegment, load_manifest
from tadabur.power_frame import FIELDS, FRAME_PATH, score_pool
from tadabur.ship_a import (
    ALTERNATIVE_HELD_OUT_SHARE,
    HELD_OUT,
    SITES_SUMMARY_PATH,
    SPLIT_PATH,
    SPLIT_SALT,
    TRAIN,
    load_sites,
    mine,
    reciter_half,
    split_record,
)
from tadabur.staged_audio import StagedClip
from tadabur.test_power_frame import FATHA, H448, KASRA, REFERENCE, scored_pool
from training.acceptance_stats import reciter_hash

FIELD = {name: i for i, name in enumerate(FIELDS)}


# --- the split ---------------------------------------------------------------------------


def test_the_split_is_the_section_9_hash_under_its_own_salt():
    assert SPLIT_SALT == "ship-a-verified-split-2026-10"
    halves = [reciter_half(i) for i in range(1000)]
    assert halves == [TRAIN if reciter_hash(i, SPLIT_SALT) < 2**63 else HELD_OUT
                      for i in range(1000)]
    assert 450 < halves.count(TRAIN) < 550


def test_the_alternative_holds_out_a_superset():
    alternative = [reciter_half(i, ALTERNATIVE_HELD_OUT_SHARE) for i in range(1000)]
    assert 280 < alternative.count(TRAIN) < 380
    assert all(a == HELD_OUT for i, a in enumerate(alternative) if reciter_half(i) == HELD_OUT)


@pytest.mark.parametrize("bad", [-1, True, "7"])
def test_the_split_refuses_a_non_canonical_id(bad):
    with pytest.raises(ValueError):
        reciter_half(bad)


# --- mining ------------------------------------------------------------------------------


def _by_kind(sites):
    return {s.kind: s for s in sites}


def test_a_two_pair_carrier_is_one_site_asked_as_the_pair_whose_partner_was_committed():
    sites = mine(scored_pool())
    pairs = [s for s in sites if s.kind == "pair"]
    assert len(pairs) == 1
    (pair,) = pairs
    assert (pair.site.mark, pair.site.stratum, pair.site.stratum_population) == (
        "ذ↔ز", "ship_a:ذ→ز:h448_partner", 1)
    assert pair.partner_swap and pair.base_already_right and pair.listened
    assert (pair.h448_committed, pair.base_committed) == ("ز", "ذ")


def test_another_letter_is_asked_as_the_carriers_first_pair():
    sites = mine(scored_pool(h448=H448.replace("ز", "د")))
    (pair,) = [s for s in sites if s.kind == "pair"]
    assert (pair.site.mark, pair.site.stratum) == ("ذ↔ز", "ship_a:ذ:h448_other")
    assert not pair.partner_swap


def test_a_geminate_the_teacher_holds_needs_no_listening():
    geminate = _by_kind(mine(scored_pool()))["geminate"]
    assert geminate.site.stratum == "ship_a:shaddah:held:h448_single:base_held"
    assert geminate.base_already_right and not geminate.listened
    single = REFERENCE.replace("بب", "ب")
    geminate = _by_kind(mine(scored_pool(base=single)))["geminate"]
    assert geminate.site.stratum == "ship_a:shaddah:held:h448_single:base_not_held"
    assert geminate.listened


def test_nothing_is_mined_where_h448_gets_the_mushaf_value():
    assert mine(scored_pool(h448=REFERENCE)) == []


def test_a_cross_word_geminate_is_never_mined():
    reference = f"ك{FATHA}بب{KASRA}ت"  # كَب|بِت: words split inside the run بب
    segment = PoolSegment(0, 0, 2, 16000, 64000, reference, (0, 3, len(reference)), True)
    clip = PoolClip("c.wav", ("uniform",), 0.5, "2:2", 3, 2, None, 0, 2, 0.0, 4.0,
                    (1.0, 2.0, 3.0), (segment,))
    registry = {"c.wav": StagedClip("c.wav", 77, 5, 3, "2:2", 80000, "b" * 64,
                                    ("mining_pool",))}
    scored = score_pool([clip], {"c.wav#0": reference},
                        {"c.wav#0": reference.replace("بب", "ب")}, registry,
                        lambda _: [False, False])
    assert any(one.stratum == "shaddah:held:cross_word" for one in scored)
    assert mine(scored) == []


# --- the committed files -----------------------------------------------------------------


def test_the_committed_sites_agree_with_the_frame():
    """Every flagged pair carrier once, and every collapsed in-word geminate, as the frame
    counts them: pair directions sharing a carrier report the same flags."""
    sites = load_sites()
    frame = json.loads(FRAME_PATH.read_text(encoding="utf-8"))["strata"]

    def total(stratum: str, field: str) -> int:
        return sum(v[FIELD[field]] for v in frame[stratum]["per_reciter"].values())

    letters = Counter(s.site.prescribed for s in sites if s.kind == "pair")
    for name in frame:
        if name.startswith("pair:"):
            letter = name.split(":")[1][0]
            assert letters[letter] == total(name, "h448_commit") - total(name, "h448_mushaf")
    geminates = [s for s in sites if s.kind == "geminate"]
    held = "shaddah:held:in_word"
    assert len(geminates) == total(held, "h448_single_on_held")
    assert sum(s.base_already_right for s in geminates) == total(
        held, "base_mushaf_where_h448_flagged")


def test_the_committed_summary_and_split_agree_with_the_sites():
    sites = load_sites()
    summary = json.loads(SITES_SUMMARY_PATH.read_text(encoding="utf-8"))
    frame = json.loads(FRAME_PATH.read_text(encoding="utf-8"))
    assert summary["inputs"] == frame["inputs"]
    assert {name: entry["sites"] for name, entry in summary["strata"].items()} == dict(
        Counter(s.site.stratum for s in sites))
    assert summary["families"]["shaddah"]["listened"] == sum(
        s.listened for s in sites if s.kind == "geminate")
    assert json.loads(SPLIT_PATH.read_text(encoding="utf-8")) == json.loads(json.dumps(
        split_record(clip.reciter_id for clip in load_manifest(POOL_CLIPS_PATH))))

"""Ship A's sites and reciter split (#126): the mining rules and the committed files."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from tadabur.listening_session import POOL_CLIPS_PATH
from tadabur.mining_pool import PoolClip, PoolSegment, load_manifest
from tadabur.power_frame import FIELDS, FRAME_PATH, score_pool
from tadabur.ship_a import (
    HELD_OUT,
    SITES_SUMMARY_PATH,
    SPLIT_PATH,
    SPLIT_SALT,
    TRAIN,
    inputs,
    load_sites,
    mine_all,
    reciter_half,
    split_record,
)
from tadabur.ship_a_mining import FILES
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


def test_the_extension_contributes_only_its_partner_swaps():
    pool = scored_pool(base=REFERENCE.replace("بب", "ب"))
    extension = scored_pool(h448=H448.replace("ز", "د"))
    mined = mine_all(pool, scored_pool())
    assert {s.site.stratum.split(":")[0] for s in mined} == {"ship_a", "ship_a_mining"}
    assert all(s.partner_swap for s in mined if s.site.stratum.startswith("ship_a_mining"))
    assert not any(s.site.stratum.startswith("ship_a_mining") for s in mine_all(pool, extension))


@pytest.mark.parametrize("bad", [-1, True, "7"])
def test_the_split_refuses_a_non_canonical_id(bad):
    with pytest.raises(ValueError):
        reciter_half(bad)


# --- mining ------------------------------------------------------------------------------


def _by_kind(sites):
    return {s.kind: s for s in sites}


def test_a_two_pair_carrier_is_one_site_asked_as_the_pair_whose_partner_was_committed():
    sites = mine_all(scored_pool())
    pairs = [s for s in sites if s.kind == "pair"]
    assert len(pairs) == 1
    (pair,) = pairs
    assert (pair.site.mark, pair.site.stratum, pair.site.stratum_population) == (
        "ذ↔ز", "ship_a:ذ→ز:h448_partner:base_right", 1)
    assert pair.partner_swap and pair.base_already_right
    assert not pair.listened  # a train-half swap the teacher gets right takes its decode
    assert (pair.h448_committed, pair.base_committed) == ("ز", "ذ")


def test_another_letter_is_mined_as_the_carriers_first_pair_and_not_listened():
    sites = mine_all(scored_pool(h448=H448.replace("ز", "د")))
    (pair,) = [s for s in sites if s.kind == "pair"]
    assert (pair.site.mark, pair.site.stratum) == ("ذ↔ز", "ship_a:ذ:h448_other")
    assert not pair.partner_swap and not pair.listened


def test_an_added_pair_partner_is_a_partner_swap():
    """``ت`` carriers are in ت↔ط only; a ط→ق commit (D9) is a ط↔ق partner swap."""
    reference = REFERENCE.replace("ت", "ط")
    sites = mine_all(scored_pool(h448=reference.replace("ط", "ق"),
                                 base=reference.replace("ط", "ق"), reference=reference))
    (pair,) = [s for s in sites if s.kind == "pair"]
    assert (pair.site.mark, pair.site.stratum) == ("ط↔ق", "ship_a:ط→ق:h448_partner:base_wrong")
    assert pair.listen_probability == 1.0


def test_a_geminate_the_teacher_holds_needs_no_listening():
    geminate = _by_kind(mine_all(scored_pool()))["geminate"]
    assert geminate.site.stratum == "ship_a:shaddah:held:h448_single:base_held"
    assert geminate.base_already_right and not geminate.listened
    single = REFERENCE.replace("بب", "ب")
    geminate = _by_kind(mine_all(scored_pool(base=single)))["geminate"]
    assert geminate.site.stratum == "ship_a:shaddah:held:h448_single:base_not_held"
    assert geminate.listened


def test_nothing_is_mined_where_h448_gets_the_mushaf_value():
    assert mine_all(scored_pool(h448=REFERENCE)) == []


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
    assert mine_all(scored) == []


# --- the committed files -----------------------------------------------------------------


def test_the_committed_sites_agree_with_the_frame():
    """Every flagged pair carrier once, and every collapsed in-word geminate, as the frame
    counts them: pair directions sharing a carrier report the same flags."""
    sites = [s for s in load_sites() if s.site.stratum.startswith("ship_a:")]
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
    assert summary["inputs"] == inputs(FILES)
    assert frame["inputs"] == {k: v for k, v in summary["inputs"].items()
                               if k.startswith("mining_pool/")}
    assert {name: entry["sites"] for name, entry in summary["strata"].items()} == dict(
        Counter(s.site.stratum for s in sites))
    assert summary["families"]["shaddah"]["listened"] == sum(
        s.listened for s in sites if s.kind == "geminate")
    clips = [*load_manifest(POOL_CLIPS_PATH),
             *load_manifest(FILES.clips, None, FILES.use, FILES.strata)]
    assert json.loads(SPLIT_PATH.read_text(encoding="utf-8")) == json.loads(json.dumps(
        split_record(clip.reciter_id for clip in clips)))
    assert {s.reciter_id for s in sites} <= {clip.reciter_id for clip in clips}


# --- what is listened to (owner decisions D10, 2026-10-10 A and B) ------------------------


def _clone(site, number: int, reciter: int, base: str, stratum: str | None = None):
    """``site`` moved to another reciter and id, with another base commit and stratum."""
    from dataclasses import replace

    row = replace(site.row, site=replace(site.site, site_id=f"{site.site.site_id}#{number}",
                                         stratum=stratum or site.site.stratum))
    return replace(site, row=row, reciter_id=reciter, half=reciter_half(reciter),
                   base_committed=base)


def test_teacher_wrong_swaps_are_all_heard_and_teacher_right_ones_once_per_held_out_reciter():
    from tadabur.ship_a import select_listening

    (swap,) = [s for s in mine_all(scored_pool()) if s.kind == "pair"]
    held = [r for r in range(200) if reciter_half(r) == HELD_OUT][:2]
    trained = next(r for r in range(200) if reciter_half(r) == TRAIN)
    sites = [_clone(swap, 0, trained, "ز"), _clone(swap, 1, trained, "ذ"),
             *(_clone(swap, 2 + k, held[0], "ذ") for k in range(3)), _clone(swap, 9, held[1], "ذ")]
    chances = select_listening(sites)
    assert chances[sites[0].site.site_id] == 1.0  # the teacher got it wrong
    assert sites[1].site.site_id not in chances  # train half, teacher right: its decode
    first = [s.site.site_id for s in sites[2:5] if s.site.site_id in chances]
    assert len(first) == 1 and chances[first[0]] == 1 / 3
    assert chances[sites[5].site.site_id] == 1.0
    assert select_listening(list(reversed(sites))) == chances


def test_the_kasb_sample_spreads_over_reciters_and_records_each_share():
    from tadabur.ship_a import KASB_SAMPLE, KASB_STRATUM, select_listening

    (swap,) = [s for s in mine_all(scored_pool()) if s.kind == "pair"]
    sites = [_clone(swap, k, reciter, "ش", f"ship_a:{KASB_STRATUM}")
             for k, reciter in enumerate([1] * 30 + list(range(2, 12)))]
    chances = select_listening(sites)
    assert len(chances) == KASB_SAMPLE
    per_reciter = Counter(s.reciter_id for s in sites if s.site.site_id in chances)
    assert per_reciter[1] == KASB_SAMPLE - 10 and all(per_reciter[r] == 1 for r in range(2, 12))
    assert {chances[s.site.site_id] for s in sites if s.reciter_id == 1
            and s.site.site_id in chances} == {(KASB_SAMPLE - 10) / 30}


def test_a_kasb_carrier_is_the_sin_of_kasaba():
    from tadabur.ship_a import kasb_form

    reference = "كَسَبَت لَهُم"
    (swap,) = [s for s in mine_all(scored_pool()) if s.kind == "pair"]
    from dataclasses import replace

    row = replace(swap.row, site=replace(swap.site, reference=reference, reference_index=2),
                  word_offsets=(0, 7, len(reference)))
    assert kasb_form(row)
    assert not kasb_form(replace(row, site=replace(row.site, reference="بَسَبَت لَهُم")))


def test_every_delivered_population_is_h448_unseen():
    """Sites, worklist, word sample and mined clips hold no recording tied, even by alias or
    probable copy, to what h448 trained on, decode_evalset or the sealed panel, and no
    panel reciter. The P3.5 re-listens are §8's targeted safeguards, so they are exempt."""
    from tadabur.exposure import SHIP_A_MINING as EXPOSURE_USE
    from tadabur.exposure import excluded_reciters, unseen_violations
    from tadabur.listening_session import load_worklist
    from tadabur.ship_a_mining import CLIPS_PATH
    from tadabur.ship_a_worklist import WORKLIST_PATH
    from tadabur.staged_audio import load_staged_clips
    from tadabur.word_sample import load_words

    registry = load_staged_clips()
    linked, panel = unseen_violations(registry), excluded_reciters(EXPOSURE_USE)
    populations = {
        "sites": [s.site.audio_filename for s in load_sites()],
        "worklist": [r.site.audio_filename for r in load_worklist(WORKLIST_PATH)
                     if r.site.source != "p35_fixture"],
        "word sample": [w.site.audio_filename for w in load_words()],
        "mined clips": [c.audio_filename for c in load_manifest(
            CLIPS_PATH, registry, FILES.use, FILES.strata)],
    }
    for name, clips in populations.items():
        assert clips and not set(clips) & linked, name
        assert not {registry[n].reciter_id for n in clips} & panel, name


def test_every_scanned_recording_is_a_mining_exposure():
    from tadabur.exposure import SHIP_A_MINING as EXPOSURE_USE
    from tadabur.exposure import load_registry
    from tadabur.ship_a_mining import SCAN_PATH

    with open(SCAN_PATH, encoding="utf-8") as f:
        scanned = {json.loads(raw)["audio_filename"] for raw in f if raw.strip()}
    assert {r.audio_filename for r in load_registry().recordings[EXPOSURE_USE]} == scanned

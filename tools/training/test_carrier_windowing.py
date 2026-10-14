"""Tests for the unselected windowing-vs-weights measurement (#146).

Torch-free and model-free: the references, decodes, carriers and verdicts are synthetic.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tadabur.mining_pool import PoolSegment
from tadabur.truth_sites import HELD, NOT_HELD
from training.carrier_windowing import (
    ALL_DIRECTIONS,
    CACHE,
    FAMILIES,
    GEMINATE,
    LETTER_DIRECTIONS,
    MUSHAF,
    OTHER,
    ROWS,
    SOFT_PAIRS_FAMILY,
    SWAP,
    UNSEEN,
    WEIGHTS,
    WINDOWING,
    Carrier,
    Clip,
    Position,
    RowKey,
    app_check_report,
    cached_plans,
    clip_commits,
    geminate_positions,
    joined_reference,
    outcome_indicators,
    owner_verdicts,
    pair_positions,
    pool_report,
    row_members,
    row_report,
)
from training.carrier_windowing_doc import gap_cell, rate_cell, render

# Two waqf segments of one clip: ``ك`` ``ت`` ``ه`` are pair letters in the first, ``ط``
# (in two pairs) and ``ه`` in the second.
SEGMENTS = ("كَتَبَ لَهُم", "طَابَ ءَمرُهُم")


def _segment(reference: str, offsets: tuple[int, ...], index: int = 0) -> PoolSegment:
    return PoolSegment(index, 0, len(offsets) - 1, 0, 16_000, reference, offsets, True)


# --- carriers -------------------------------------------------------------------------------


def test_a_pair_carrier_is_the_first_of_its_run_and_only_target_pair_letters_count():
    assert pair_positions(0, SEGMENTS[0]) == [Position(0, 0, False), Position(0, 2, False),
                                              Position(0, 9, False)]
    # A geminate ``طط`` is one carrier, on its first half; ``ب`` and ``ل`` are no pair letter.
    assert pair_positions(3, "بَططَلَ") == [Position(3, 2, False)]


def test_an_in_word_geminate_is_a_carrier_and_one_across_a_word_boundary_is_not():
    in_word = _segment("رَببِ", (0, 5))
    assert geminate_positions(1, in_word) == [Position(1, 2, True)]
    # ``مِن مَا`` read as ``مِممَا``: the run joins two words (idgham), so it is no carrier.
    across = _segment("مِممَا", (0, 3, 6))
    assert geminate_positions(0, across) == []


def test_every_letter_of_a_target_pair_has_its_directions():
    assert LETTER_DIRECTIONS["ط"] == ("ط→ت", "ط→ق")
    assert LETTER_DIRECTIONS["ت"] == ("ت→ط",)
    assert set(LETTER_DIRECTIONS["س"]) == {"س→ش", "س→ص"}
    directions = [d for family, ds in FAMILIES.items() if family != GEMINATE for d in ds]
    assert sorted(directions) == sorted(d for ds in LETTER_DIRECTIONS.values() for d in ds)
    assert len(directions) == 18


def test_rows_pool_each_family_after_its_directions():
    assert ROWS[-1] == RowKey(GEMINATE, GEMINATE)
    assert RowKey(SOFT_PAIRS_FAMILY, ALL_DIRECTIONS) in ROWS
    assert ROWS.index(RowKey("ط↔ق", "ط→ق")) < ROWS.index(RowKey("ط↔ق", ALL_DIRECTIONS))


# --- reading the decodes ------------------------------------------------------------------


def test_the_stream_reference_joins_segments_with_a_word_space():
    assert joined_reference(["ab", "cd", "e"]) == ("ab cd e", [0, 3, 6])


def test_whole_reads_each_segment_and_the_stream_reads_the_joined_reference():
    positions = tuple(p for n, ref in enumerate(SEGMENTS) for p in pair_positions(n, ref))
    clip = Clip("c", 1, 1.0, SEGMENTS, positions,
                by_segment={"m/whole": ("كَتَبَلَهُم", "طَابَءَمرُهُم")},
                by_span={
                    "m/stream_b0": "كَتَبَلَهُمقَابَءَمرُهُم",  # the second segment's ط heard as ق
                    "m/stream_b1": "كَتَبَلَهُم",  # the stream never reached the second segment
                })
    commits = clip_commits(clip)
    assert commits["m/whole"] == ["ك", "ت", "ه", "ط", "ه"]
    assert commits["m/stream_b0"] == ["ك", "ت", "ه", "ق", "ه"]
    assert commits["m/stream_b1"][:3] == ["ك", "ت", "ه"]
    assert commits["m/stream_b1"][3:] == [None, None]


def test_a_geminate_commits_held_single_or_nothing():
    segment = _segment("رَببِ", (0, 5))
    positions = tuple(geminate_positions(0, segment))
    clip = Clip("c", 1, 1.0, (segment.reference,), positions, by_segment={"m/whole": ("رَببِ",)},
                by_span={"m/stream_b0": "رَبِ", "m/stream_b1": "سَقَ"})
    commits = clip_commits(clip)
    assert commits == {"m/whole": [HELD], "m/stream_b0": [NOT_HELD], "m/stream_b1": [None]}


# --- rows and statistics ------------------------------------------------------------------


def _carrier(letter: str, values: dict, reciter: int = 1, weight: float = 1.0,
             geminate: bool = False, clip: str = "c") -> Carrier:
    return Carrier(clip, reciter, weight, letter, geminate, values)


def test_a_carrier_in_two_pairs_counts_in_both_and_once_per_family_row():
    taa = _carrier("ط", {"a": "ق"})
    assert [d for d, _ in row_members([taa], RowKey("ط↔ق", "ط→ق"))] == ["ط→ق"]
    assert [d for d, _ in row_members([taa], RowKey(SOFT_PAIRS_FAMILY, ALL_DIRECTIONS))] == ["ط→ت"]
    assert [d for d, _ in row_members([taa], RowKey("ط↔ق", ALL_DIRECTIONS))] == ["ط→ق"]
    assert row_members([taa], RowKey("س↔ش", ALL_DIRECTIONS)) == []


def test_outcomes_are_read_against_each_carriers_own_direction():
    members = [("ط→ق", _carrier("ط", {"a": "ق"})), ("ط→ق", _carrier("ط", {"a": "ط"})),
               ("ق→ط", _carrier("ق", {"a": "ك"})), ("ق→ط", _carrier("ق", {"a": None})),
               (GEMINATE, _carrier("ب", {"a": NOT_HELD}, geminate=True))]
    indicators = outcome_indicators(members, "a")
    assert indicators[SWAP].tolist() == [1, 0, 0, 0, 1]
    assert indicators[MUSHAF].tolist() == [0, 1, 0, 0, 0]
    assert indicators[OTHER].tolist() == [0, 0, 1, 1, 0]


def _population(n_reciters: int = 30) -> list[Carrier]:
    """Three ``ط`` carriers per reciter, decoded by every pool arm: the b=0 stream swaps
    more than the whole decode, b=1 as the whole decode does, and both models alike."""
    rng = np.random.default_rng(0)
    carriers = []
    for reciter in range(n_reciters):
        for k in range(3):
            whole = "ق" if rng.random() < 0.1 else "ط"
            stream = "ق" if rng.random() < 0.4 else ("ط" if k else None)
            by_protocol = {"whole": whole, "stream_b0": stream, "stream_b1": whole}
            carriers.append(_carrier(
                "ط", {f"{m}/{p}": v for m in ("base", "h448") for p, v in by_protocol.items()},
                reciter=reciter, weight=1 + reciter % 3))
    return carriers


def test_a_paired_gap_is_the_difference_of_the_two_rates_on_one_draw():
    members = row_members(_population(), RowKey("ط↔ق", "ط→ق"))
    report = row_report(members, ["base/whole", "base/stream_b0"],
                        [("base/stream_b0", "base/whole")])
    whole, stream = report["rates"]["base/whole"], report["rates"]["base/stream_b0"]
    gap = report["gaps"]["base/stream_b0 − base/whole"]
    assert math.isclose(gap["point"], stream[SWAP]["point"] - whole[SWAP]["point"])
    assert gap["method"] == "bootstrap" and gap["lower"] < gap["point"] < gap["upper"]
    for rates in (whole, stream):
        assert math.isclose(sum(rates[r]["point"] for r in (SWAP, MUSHAF, OTHER)), 1.0)
    assert report["carriers"] == 90 and report["reciters"] == 30
    assert stream[SWAP]["count"] == sum(c.committed["base/stream_b0"] == "ق"
                                        for _, c in members)


def test_rates_are_weighted_by_inverse_inclusion_probability():
    members = [("ط→ق", _carrier("ط", {"a": "ق"}, reciter=1, weight=3.0)),
               ("ط→ق", _carrier("ط", {"a": "ط"}, reciter=2, weight=1.0))]
    report = row_report(members, ["a"], [])
    assert report["rates"]["a"][SWAP]["point"] == pytest.approx(0.75)
    assert report["rates"]["a"][SWAP]["method"] == "none"  # sparse and unequally weighted


def test_an_empty_row_reports_its_size_only():
    assert row_report([], ["a"], []) == {"carriers": 0, "reciters": 0}


def test_cached_plans_must_be_exactly_the_streamed_clips():
    plans = [SimpleNamespace(eligible=True, clip=SimpleNamespace(audio_filename="a")),
             SimpleNamespace(eligible=False, clip=SimpleNamespace(audio_filename="b"))]
    assert cached_plans(plans, {"m": {"stream_rows_clips": ["a"]}}) == plans[:1]
    with pytest.raises(ValueError):
        cached_plans(plans, {"m": {"stream_rows_clips": ["a", "b"]}})


# --- the owner's verdicts --------------------------------------------------------------


def _ship_a(site_id: str, mark: str, prescribed: str, swap: bool = True, base_right: bool = True):
    site = SimpleNamespace(site_id=site_id, mark=mark, prescribed=prescribed)
    return SimpleNamespace(site=site, partner_swap=swap,
                           base_already_right=base_right, family=mark)


def test_owner_verdicts_count_only_answered_partner_swaps_the_base_got_right():
    sites = [_ship_a("1", "ط↔ق", "ط"), _ship_a("2", "ط↔ق", "ق"), _ship_a("3", "ط↔ق", "ط"),
             _ship_a("4", "ط↔ق", "ط", base_right=False), _ship_a("5", "ق↔ك", "ك"),
             _ship_a("6", "ط↔ق", "ط")]
    verdicts = {"1": SimpleNamespace(heard="ط"), "2": SimpleNamespace(heard="ط"),
                "3": SimpleNamespace(heard="unclear"), "4": SimpleNamespace(heard="ط"),
                "5": SimpleNamespace(heard="ق")}
    counts = owner_verdicts(sites, verdicts)
    assert counts["ط↔ق"] == {MUSHAF: 1, SWAP: 1, "unclear": 1}
    assert counts["ق↔ك"] == {SWAP: 1}
    assert counts[SOFT_PAIRS_FAMILY] == {SWAP: 1}


# --- the page ---------------------------------------------------------------------------


def _report() -> dict:
    app = [_carrier("س", {f"{m}/{p}": "ش" if r % 4 == 0 else "س"
                          for m in ("base", "h448") for p in ("spans", "stream_b0", "checks")},
                    reciter=r) for r in range(24)]
    population = {"pool": {"clips": 9, "reciters": 3}, "not_in_cache": {"re_read": 1},
                  CACHE: {"clips": 8, "reciters": 3, "segments": 10}, "h448_linked_clips": 1,
                  UNSEEN: {"clips": 7, "reciters": 3, "segments": 9,
                           "inclusion_probability_1": 2}}
    model = {"ref": "x", "hub_revision": "0" * 40}
    return {
        "inputs": {"haraka_gap_cache": {"base.json": "a"}, "sources": {"s": "b"},
                   "models": {"base": model, "h448": {"ref": "y", "checkpoint_sha256": "1" * 64}},
                   "pool_audio": "c", "other": {"o": "d"}},
        "decode_settings": {"weights_dtype": "bf16", "batch_size": 1, "device_type": "cuda",
                            "policy": "p"},
        "bootstrap": {"replicates": 10_000, "seed": 1},
        "segment_separator": " ",
        "population": population,
        "pool": pool_report(_population(), {"c"}),
        "owner_verdicts": {"ط↔ق": {MUSHAF: 2}, SOFT_PAIRS_FAMILY: {}},
        "app_checks": app_check_report(app, {"items": 5, "h448_linked_items": 1,
                                             "kept_items": 4, "reciters": 4}),
    }


def test_the_page_is_rendered_from_the_report():
    report = _report()
    assert set(report["pool"]["gaps"]) == {WINDOWING, WEIGHTS}
    page = render(report)
    assert page.startswith("# Windowing vs weights at target-pair carriers")
    assert "Generated by `python -m training.carrier_windowing`" in page
    assert "| ط→ق | 90 / 30 |" in page
    assert "h448 b0 − h448 whole" in page and "h448 b1 − base b1" in page
    assert "base checks − base spans" in page
    assert "-0.0" not in page


def test_cells_mark_the_interval_method_and_carry_the_count():
    entry = {"point": 0.125, "lower": 0.05, "upper": 0.2, "method": "bootstrap", "count": 3}
    assert rate_cell(entry) == "12.5 [5.0, 20.0] (3)"
    assert rate_cell({**entry, "method": "wilson"}) == "12.5 [5.0, 20.0] W (3)"
    assert rate_cell({**entry, "method": "none"}) == "12.5 (n/c) (3)"
    assert gap_cell({"point": -0.0001, "lower": "-inf", "upper": 0.0, "method": "bootstrap"}) \
        == "+0.0 [−∞, 0.0]"

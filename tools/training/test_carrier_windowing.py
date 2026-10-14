"""Tests for the unselected windowing-vs-weights measurement (#146).

Torch-free and model-free: the references, decodes, carriers and verdicts are synthetic.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tadabur.mining_pool import PoolSegment
from tadabur.truth_sites import HELD, NOT_HELD
from training import truth_baseline
from training.carrier_windowing import (
    APP_PROTOCOLS,
    CACHE,
    DIAGNOSTIC,
    LETTER_PAIRS,
    ROWS,
    SPANS,
    UNSEEN,
    WEIGHTS,
    WINDOWING,
    WORD_SAMPLE,
    Clip,
    Position,
    RowKey,
    app_decodes,
    cached_plans,
    carrier_sites,
    clip_commits,
    gap_report,
    geminate_positions,
    joined_reference,
    owner_verdicts,
    pair_positions,
    pool_report,
    row_members,
    row_report,
    uncommitted,
)
from training.carrier_windowing_doc import gap_cell, rate_cell, render
from training.direction_table import (
    ALL_DIRECTIONS,
    GEMINATE,
    MUSHAF,
    OTHER,
    SOFT_PAIRS_FAMILY,
    SWAP,
    direction,
)
from training.site_outcomes import SiteOutcome
from training.truth_scorer import SiteRow

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


def test_rows_pool_each_family_after_its_directions():
    assert ROWS[-1] == RowKey(GEMINATE, GEMINATE)
    assert RowKey(SOFT_PAIRS_FAMILY, ALL_DIRECTIONS) in ROWS
    assert ROWS.index(RowKey("ط↔ق", "ط→ق")) < ROWS.index(RowKey("ط↔ق", ALL_DIRECTIONS))


# --- reading the decodes ------------------------------------------------------------------


def test_the_stream_reference_joins_segments_with_a_word_space():
    assert joined_reference(["ab", "cd", "e"]) == ("ab cd e", [0, 3, 6])


def test_whole_reads_each_segment_and_the_stream_reads_the_joined_reference():
    positions = tuple(p for n, ref in enumerate(SEGMENTS) for p in pair_positions(n, ref))
    clip = Clip("c", "1:1", 1, 1.0, SEGMENTS, (0, 16_000), positions,
                by_segment={"m/whole": ("كَتَبَلَهُم", "طَابَءَمرُهُم")},
                by_span={
                    "m/stream_b0": "كَتَبَلَهُمقَابَءَمرُهُم",  # the second segment's ط heard as ق
                    "m/stream_b1": "كَتَبَلَهُم",  # the stream never reached the second segment
                })
    commits = clip_commits(clip)
    letters = {a: [o.committed for o in outcomes] for a, outcomes in commits.items()}
    assert letters["m/whole"] == ["ك", "ت", "ه", "ط", "ه"]
    assert letters["m/stream_b0"] == ["ك", "ت", "ه", "ق", "ه"]
    assert letters["m/stream_b1"] == ["ك", "ت", "ه", None, None]


def test_a_geminate_commits_held_single_or_nothing():
    segment = _segment("رَببِ", (0, 5))
    positions = tuple(geminate_positions(0, segment))
    clip = Clip("c", "1:1", 1, 1.0, (segment.reference,), (0,), positions,
                by_segment={"m/whole": ("رَببِ",)},
                by_span={"m/stream_b0": "رَبِ", "m/stream_b1": "سَقَ"})
    commits = clip_commits(clip)
    assert {a: [o.committed for o in v] for a, v in commits.items()} == {
        "m/whole": [HELD], "m/stream_b0": [NOT_HELD], "m/stream_b1": [None]}


# --- rows and statistics ------------------------------------------------------------------


def _clip(reference: str, geminate: bool = False, name: str = "c", reciter: int = 1,
          weight: float = 1.0) -> tuple[Clip, Position]:
    clip = Clip(name, "1:1", reciter, weight, (reference,), (0,),
                (Position(0, 0, geminate),), {}, {})
    return clip, clip.positions[0]


def _rows(letter: str, committed: dict, geminate: bool = False, **kwargs) -> list[SiteRow]:
    """One carrier of ``letter`` as its site rows, each arm committing ``committed[arm]``."""
    clip, position = _clip(f"{letter}{letter if geminate else ''}َ", geminate, **kwargs)
    outcomes = {a: SiteOutcome(v is not None, v) for a, v in committed.items()}
    return [SiteRow(site, clip.weight, clip.reciter_id, outcomes, {})
            for site in carrier_sites(clip, position)]


def test_a_carrier_is_one_site_per_direction_it_sits_in():
    clip, position = _clip("طَ")
    assert [direction(s) for s in carrier_sites(clip, position)] == ["ط→ت", "ط→ق"]
    clip, position = _clip("ببَ", geminate=True)
    assert [direction(s) for s in carrier_sites(clip, position)] == [GEMINATE]
    assert LETTER_PAIRS["ط"] == ("ت↔ط", "ط↔ق") and LETTER_PAIRS["ت"] == ("ت↔ط",)


def test_a_carrier_in_two_pairs_counts_in_both_and_once_per_family_row():
    taa = _rows("ط", {"a": "ق"})
    assert [direction(r.site) for r in row_members(taa, RowKey("ط↔ق", "ط→ق"))] == ["ط→ق"]
    assert [direction(r.site) for r in row_members(taa, RowKey(SOFT_PAIRS_FAMILY, ALL_DIRECTIONS))
            ] == ["ط→ت"]
    assert len(row_members(taa, RowKey("ط↔ق", ALL_DIRECTIONS))) == 1
    assert row_members(taa, RowKey("س↔ش", ALL_DIRECTIONS)) == []
    # Counted once as a carrier, whatever its directions.
    assert uncommitted(taa, ["a"])["a"]["pair"] == {"carriers": 1, "uncommitted": 0}


def test_outcomes_are_read_against_each_sites_own_direction():
    rows = [*_rows("ط", {"a": "ق"}, reciter=1), *_rows("ط", {"a": "ط"}, reciter=2),
            *_rows("ق", {"a": "ك"}, reciter=3), *_rows("ق", {"a": None}, reciter=4),
            *_rows("ب", {"a": NOT_HELD}, geminate=True, reciter=5)]
    tq = row_report(row_members(rows, RowKey("ط↔ق", ALL_DIRECTIONS)), ["a"], [])
    assert tq["counts"]["a"] == {SWAP: 1, MUSHAF: 1, OTHER: 2}
    gem = row_report(row_members(rows, RowKey(GEMINATE, GEMINATE)), ["a"], [])
    assert gem["counts"]["a"] == {SWAP: 1, MUSHAF: 0, OTHER: 0}


def _population(n_reciters: int = 30) -> list[SiteRow]:
    """Three ``ط`` carriers per reciter, decoded by every pool arm: the b=0 stream swaps
    more than the whole decode, b=1 as the whole decode does, and both models alike."""
    rng = np.random.default_rng(0)
    rows = []
    for reciter in range(n_reciters):
        for k in range(3):
            whole = "ق" if rng.random() < 0.1 else "ط"
            stream = "ق" if rng.random() < 0.4 else ("ط" if k else None)
            by_protocol = {"whole": whole, "stream_b0": stream, "stream_b1": whole}
            rows += _rows("ط", {f"{m}/{p}": v for m in ("base", "h448")
                                for p, v in by_protocol.items()},
                          name=f"c{reciter}-{k}", reciter=reciter, weight=1 + reciter % 3)
    return rows


def test_a_paired_gap_is_the_difference_of_the_two_rates_on_one_draw():
    members = row_members(_population(), RowKey("ط↔ق", "ط→ق"))
    report = row_report(members, ["base/whole", "base/stream_b0"],
                        [("base/stream_b0", "base/whole")])
    whole, stream = report["arms"]["base/whole"], report["arms"]["base/stream_b0"]
    gap = report["gaps"]["base/stream_b0 − base/whole"]
    assert math.isclose(gap["point"], stream[SWAP]["point"] - whole[SWAP]["point"])
    assert gap["method"] == "bootstrap" and gap["lower"] < gap["point"] < gap["upper"]
    for rates in (whole, stream):
        assert math.isclose(sum(rates[r]["point"] for r in (SWAP, MUSHAF, OTHER)), 1.0)
    assert report["sites"] == 90 and report["reciters"] == 30
    assert report["counts"]["base/stream_b0"][SWAP] == sum(
        r.outcomes["base/stream_b0"].committed == "ق" for r in members)


def test_rates_are_weighted_by_inverse_inclusion_probability():
    rows = [*_rows("ط", {"a": "ق"}, reciter=1, weight=3.0),
            *_rows("ط", {"a": "ط"}, reciter=2, weight=1.0)]
    report = row_report(row_members(rows, RowKey("ط↔ق", "ط→ق")), ["a"], [])
    assert report["arms"]["a"][SWAP]["point"] == pytest.approx(0.75)
    assert report["arms"]["a"][SWAP]["method"] == "none"  # sparse and unequally weighted


def test_an_empty_row_reports_its_size_only():
    assert row_report([], ["a"], []) == {"sites": 0, "reciters": 0,
                                         "insufficient_evidence": True}


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
    app = [row for r in range(24) for row in _rows(
        "س", {f"{m}/{p}": "ش" if r % 4 == 0 else "س" for m in ("base", "h448")
             for p in APP_PROTOCOLS}, name=f"i{r}", reciter=r)]
    checks = {"items": 5, "h448_linked_items": 1, "kept_items": 4, "reciters": 4,
              **gap_report(app, APP_PROTOCOLS, SPANS)}
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
        "pool": pool_report(_population(), {f"c{r}-{k}" for r in range(20) for k in range(3)}),
        "owner_verdicts": {"ط↔ق": {MUSHAF: 2}, SOFT_PAIRS_FAMILY: {}},
        "app_checks": {DIAGNOSTIC: checks, WORD_SAMPLE: checks},
    }


def test_the_page_is_rendered_from_the_report():
    report = _report()
    assert set(report["pool"][UNSEEN]["gaps"]) == {WINDOWING, WEIGHTS}
    assert report["pool"][UNSEEN]["rows"][ROWS.index(RowKey("ط↔ق", "ط→ق"))]["sites"] == 60
    page = render(report)
    assert page.startswith("# Windowing vs weights at target-pair carriers")
    assert "Generated by `python -m training.carrier_windowing`" in page
    assert "| ط→ق | 60 / 20 |" in page and "| ط→ق | 90 / 30 |" in page
    assert "h448 b0 − h448 whole" in page and "h448 b1 − base b1" in page
    assert "base checks − base spans" in page and "### The word sample's segments" in page
    assert "-0.0" not in page


def test_cells_mark_the_interval_method_and_carry_the_count():
    entry = {"point": 0.125, "lower": 0.05, "upper": 0.2, "method": "bootstrap"}
    assert rate_cell(entry, 3) == "12.5 [5.0, 20.0] (3)"
    assert rate_cell({**entry, "method": "wilson"}, 3) == "12.5 [5.0, 20.0] W (3)"
    assert rate_cell({**entry, "method": "none"}, 3) == "12.5 (n/c) (3)"
    assert gap_cell({"point": -0.0001, "lower": "-inf", "upper": 0.0, "method": "bootstrap"}) \
        == "+0.0 [−∞, 0.0]"


# --- the cross-checks' caches -----------------------------------------------------------------


def _write_caches(tmp_path, monkeypatch, checks_weights: str = "same") -> dict:
    """Both models' truth-baseline caches for one item, in temporary stores."""
    decode_store = replace(truth_baseline.DECODE_STORE, directory=tmp_path / "decodes")
    check_store = replace(truth_baseline.CHECK_STORE, directory=tmp_path / "checks")
    monkeypatch.setattr(truth_baseline, "DECODE_STORE", decode_store)
    monkeypatch.setattr(truth_baseline, "CHECK_STORE", check_store)
    pool_models = {}
    for model, ref in truth_baseline.DEFAULT_MODELS.items():
        identity = {"model_ref": ref, "checkpoint_sha256": f"{model}-weights"}
        pool_models[model] = {"ref": ref, "checkpoint_sha256": f"{model}-weights"}
        for store, item, weights in (
            (decode_store, {"spans": "طَ", "stream_b0": "قَ"}, identity),
            (check_store, {"checks": [{"hop": "طَ", "overlap": "", "flush": True}]},
             identity if checks_weights == "same" else {**identity, "checkpoint_sha256": "x"}),
        ):
            store.directory.mkdir(exist_ok=True)
            store.path(model).write_text(json.dumps({
                "model": model, "identity": weights,
                "fingerprints": {p: truth_baseline.expected_fingerprint(ref, mode)
                                 for p, mode in store.protocols.items()},
                "items": {"clip.wav@0:16000": {"audio_sha256": "audio", **item}},
            }), encoding="utf-8")
    return pool_models


ITEMS = {"clip.wav@0:16000": SimpleNamespace(audio_sha256="audio")}


def test_cross_check_decodes_are_read_through_the_truth_baselines_checks(tmp_path, monkeypatch):
    pool_models = _write_caches(tmp_path, monkeypatch)
    decodes = app_decodes(ITEMS, {}, pool_models)
    assert decodes["h448/stream_b0"] == {"clip.wav@0:16000": "قَ"}
    assert decodes["base/checks"] == {"clip.wav@0:16000": "طَ"}


def test_cross_checks_refuse_checks_from_other_weights(tmp_path, monkeypatch):
    pool_models = _write_caches(tmp_path, monkeypatch, checks_weights="other")
    with pytest.raises(ValueError, match="different weights"):
        app_decodes(ITEMS, {}, pool_models)


def test_cross_checks_refuse_weights_other_than_the_pool_caches(tmp_path, monkeypatch):
    pool_models = _write_caches(tmp_path, monkeypatch)
    pool_models["h448"] = {**pool_models["h448"], "checkpoint_sha256": "retrained"}
    with pytest.raises(ValueError, match="pool cache"):
        app_decodes(ITEMS, {}, pool_models)


def test_cross_checks_refuse_items_decoded_from_other_audio(tmp_path, monkeypatch):
    pool_models = _write_caches(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="other audio"):
        app_decodes({"clip.wav@0:16000": SimpleNamespace(audio_sha256="re-staged")}, {},
                    pool_models)


def test_owner_verdicts_serialise_identically_under_any_hash_seed():
    script = (
        "import json; from types import SimpleNamespace as N\n"
        "from training.carrier_windowing import owner_verdicts\n"
        "pairs = [('ذ↔ز', 'ذ', 'ز'), ('ض↔ظ', 'ظ', 'unclear'), ('ق↔ك', 'ك', 'ب'),\n"
        "         ('س↔ص', 'س', 'س')]\n"
        "sites = [N(site=N(site_id=str(i), mark=m, prescribed=p), partner_swap=True,\n"
        "           base_already_right=True, family=m) for i, (m, p, _) in enumerate(pairs)]\n"
        "verdicts = {str(i): N(heard=h) for i, (_, _, h) in enumerate(pairs)}\n"
        "print(json.dumps(owner_verdicts(sites, verdicts), ensure_ascii=False))\n"
    )
    outputs = {
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                       cwd=Path(__file__).parent.parent,
                       env={**os.environ, "PYTHONHASHSEED": seed}).stdout
        for seed in map(str, range(8))
    }
    assert len(outputs) == 1

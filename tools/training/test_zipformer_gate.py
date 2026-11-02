"""Tests for the card-reproduction gate's pure parts (#245, step 3); torch-free, no download.

The gate scores zipformer with the card's own unit-level PER against the card's own table, so
its inputs and rules are pinned before any decode (``training.zipformer_gate``):

* **Unit rule.** The card's tokenizer is unpublished; greedy longest-match over
  ``phoneme_units.json`` stands in, justified by a round trip: re-tokenising what the model
  emitted must reproduce the emitted units on at least 99% of clips, or -- architect r3's D2
  sort, written before the failures were inspected -- every clip it does not must be a model
  quirk (the divergent window emitted the stand-in's way somewhere in the decode set), else
  the gate stops (:class:`GateStop`) and no other rule is tuned against it.
* **Scale rule.** The fbank's sample scale ([-1, 1] or x32768) is pinned by the lower PER on all
  three benchmark sources; sources that disagree, or scales within 2 points on any source, stop.
* **Table misses** (a clip whose reference text is not in the card's table) are dropped from the
  PER and counted, as the card's script drops them.
"""

from __future__ import annotations

import json

import pytest

from training.zipformer_decode import ZIPFORMER_DIR, read_tokens
from training.zipformer_gate import (
    GateStop,
    edit_counts,
    encode_units,
    first_divergence,
    pin_scale,
    require_stand_in,
    roundtrip_identical,
    stand_in_record,
    source_per,
)

UNITS = ["ب", "بَ", "ا", "اا", "ءَ", "ء", "ت"]


# --- the unit rule: greedy longest match, and its round trip ----------------------------


def test_the_longest_unit_that_matches_wins():
    assert encode_units("بَاا", UNITS) == ["بَ", "اا"]
    assert encode_units("ءَب", UNITS) == ["ءَ", "ب"]


def test_whitespace_is_not_a_unit():
    assert encode_units("بَ اا ت", UNITS) == ["بَ", "اا", "ت"]


def test_a_character_no_unit_holds_is_refused_not_dropped():
    with pytest.raises(ValueError):
        encode_units("بخ", UNITS)


def test_emitted_units_that_re_tokenise_to_themselves_round_trip():
    assert roundtrip_identical(["بَ", "اا", "ت"], UNITS)


def test_two_units_greedy_would_merge_do_not_round_trip():
    # the model emitted alif, alif; the tokenizer would have produced the merged "اا"
    assert not roundtrip_identical(["بَ", "ا", "ا"], UNITS)


def _clips(failures: int, total: int = 100) -> dict[str, list[str]]:
    return {f"c{i}": (["ا", "ا"] if i < failures else ["اا"]) for i in range(total)}


def test_one_failure_in_a_hundred_clips_is_exactly_the_floor_and_passes():
    record = stand_in_record(_clips(1), UNITS)
    assert record["share"] == pytest.approx(0.99)
    assert record["accepted"]
    require_stand_in(record)


def test_every_failure_being_a_model_quirk_accepts_the_stand_in_below_the_floor():
    # the other clips emit "اا", so the divergence is an uncertain split, not a rule
    record = stand_in_record(_clips(2), UNITS, minimum=0.99)
    assert not record["share"] >= 0.99
    assert [failure["classification"] for failure in record["failures"]] == ["quirk"] * 2
    assert record["accepted"]
    require_stand_in(record)


def test_one_failure_the_model_never_emits_elsewhere_stops_the_gate():
    units = ["ا", "ب", "اب", "بَ"]
    clips = {"split": ["ا", "ب"], "other": ["بَ"]}
    record = stand_in_record(clips, units)
    assert [failure["classification"] for failure in record["failures"]] == ["rule"]
    assert not record["accepted"]
    with pytest.raises(GateStop, match="not the card's tokenizer"):
        require_stand_in(record)


def test_no_clips_is_an_error_not_a_pass():
    with pytest.raises(ValueError):
        stand_in_record({}, UNITS)


def test_the_first_divergence_names_the_window_and_both_segmentations():
    start, end, mine, theirs = first_divergence(["بَ", "ا", "ا", "ت"], ["بَ", "اا", "ت"])
    assert (start, end) == (2, 4)
    assert (mine, theirs) == (["ا", "ا"], ["اا"])


def test_identical_segmentations_have_no_divergence():
    assert first_divergence(["اا", "ت"], ["اا", "ت"]) is None


# --- the committed units ----------------------------------------------------------------


def test_the_gates_units_are_tokens_txts_250_strings():
    units = json.loads((ZIPFORMER_DIR / "phoneme_units.json").read_text(encoding="utf-8"))
    tokens = read_tokens(ZIPFORMER_DIR / "tokens.txt")
    assert set(units) - {"<blank>"} == {t for i, t in tokens.items() if i != 250}
    assert len(units) == 251


def test_the_real_units_tokenise_a_merged_pair_as_one_unit():
    units = [u for u in json.loads(
        (ZIPFORMER_DIR / "phoneme_units.json").read_text(encoding="utf-8")) if u != "<blank>"]
    assert encode_units("بَاا", units) == ["بَ", "اا"]


# --- the scale rule ---------------------------------------------------------------------


def _per(unit: float, int16: float) -> dict[float, dict[str, float]]:
    return {1.0: {"studio": unit, "phone": unit + 1, "unseen": unit + 6},
            32_768.0: {"studio": int16, "phone": int16 + 1, "unseen": int16 + 6}}


def test_the_scale_with_the_lower_per_on_every_source_is_pinned():
    assert pin_scale(_per(1.5, 40.0)) == 1.0
    assert pin_scale(_per(40.0, 1.5)) == 32_768.0


def test_sources_that_disagree_stop_the_gate():
    per = _per(1.5, 40.0)
    per[32_768.0]["phone"] = 0.1  # better at x32768 on one source only, by over 2 points
    with pytest.raises(GateStop):
        pin_scale(per)


def test_scales_within_two_points_on_any_source_stop_the_gate():
    per = _per(1.5, 10.0)
    per[32_768.0]["unseen"] = per[1.0]["unseen"] + 2.0  # within 2 points: stop
    with pytest.raises(GateStop):
        pin_scale(per)


def test_a_gap_just_over_two_points_on_every_source_is_decisive():
    assert pin_scale(_per(1.5, 3.6)) == 1.0


# --- PER with table misses dropped and counted ------------------------------------------


def test_edit_counts_are_edits_and_reference_length_for_units_and_characters():
    assert edit_counts(["a", "b", "c"], ["a", "x", "c"]) == (1, 3)
    assert edit_counts("abc", "abd") == (1, 3)
    assert edit_counts(["a", "b"], []) == (2, 2)
    assert edit_counts(["a"], ["a", "b", "c"]) == (2, 1)


def test_per_is_pooled_over_clips_and_misses_are_dropped_and_counted():
    rows = [
        {"source": "studio", "gold": ["a", "b", "c", "d"], "pred": ["a", "b", "c", "d"]},
        {"source": "studio", "gold": ["a", "b"], "pred": ["a", "x"]},
        {"source": "studio", "gold": None, "pred": ["q"]},  # reference text not in the table
        {"source": "phone", "gold": ["a", "b"], "pred": []},
    ]
    assert source_per(rows) == {
        "studio": {"per": pytest.approx(100 * 1 / 6), "clips": 2, "misses": 1},
        "phone": {"per": pytest.approx(100.0), "clips": 1, "misses": 0},
    }

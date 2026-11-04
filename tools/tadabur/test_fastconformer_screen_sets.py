"""Tests for #267's sets: G6 removal, the round-3 refusal, and the erratum before any decode.

* Ship A's 8 held-out G6 sites (``consonant_readout/g6_sites.json``) are removed from every set,
  and so are the decode items that hold only them;
* only ship-a, edits and qdat may be decoded: tlog look, every confirm tranche, qdat_bench, QUD
  other half and Tadabur dev are round-3 inputs, refused before any file is opened;
* the dated erratum (architect r4: every column is scored letter-level, so the teacher's bars
  move) is the run's first commit, and the decode refuses without it, bars included;
* nothing is added to ``tools/requirements*.txt``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tadabur.fastconformer_screen.gates import ScreenStop
from tadabur.fastconformer_screen.decode import preflight
from tadabur.fastconformer_screen.sets import (
    DECODABLE,
    ERRATUM_PATH,
    decode_plan,
    edit_items,
    require_decodable,
    require_erratum,
    ship_a_sites,
    split_sides,
    teacher_bars,
)
from tadabur.fastconformer_screen.verdict import Rates
from tadabur.ship_a_verified import load_verified
from training.consonant_readout import load_g6_sites
from training.truth_scorer import item_key

TOOLS = Path(__file__).resolve().parent.parent
BARS = Rates(mistakes_flagged=83.6, correct_flagged=63.4, edits_heard=22.7, decoys_unchanged=58.7)


# --- G6 ---------------------------------------------------------------------------------------


def test_the_g6_sites_are_verified_mistakes():
    # the arithmetic below (75 - 8 mistakes) rests on this
    mistakes, _ = split_sides(load_verified()[0])
    assert set(load_g6_sites()) <= {site.site_id for site in mistakes}
    assert len(load_g6_sites()) == 8


def test_no_g6_site_is_in_ship_a():
    sites = ship_a_sites()
    assert not set(load_g6_sites()) & {site.site_id for site in sites}


def test_ship_a_is_67_mistakes_and_410_correct_sites():
    mistakes, correct = split_sides(ship_a_sites())
    assert (len(mistakes), len(correct)) == (75 - 8, 410)


def test_sites_without_a_side_are_in_neither_denominator():
    sites = ship_a_sites()
    mistakes, correct = split_sides(sites)
    assert len(mistakes) + len(correct) < len(sites)      # the 45 unclear/pending sites


def test_the_ship_a_decode_plan_leaves_out_every_g6_item():
    g6_items = {item_key(site) for site in load_verified()[0] if site.site_id in set(load_g6_sites())}
    kept_items = {item_key(site) for site in ship_a_sites()}
    plan = decode_plan("ship-a")
    assert len(g6_items) == 8 and not g6_items & kept_items       # one site per G6 item
    assert set(plan) == kept_items and len(plan) == len(kept_items) == 504
    assert plan == sorted(plan)                                    # a deterministic order


def test_the_edits_are_all_300_items_and_their_plan_is_their_item_ids():
    items = edit_items()
    assert len(items) == 300
    assert decode_plan("edits") == sorted(item["item_id"] for item in items)


# --- the round-3 refusal ----------------------------------------------------------------------


def test_only_ship_a_edits_and_qdat_are_decodable():
    assert sorted(DECODABLE) == ["edits", "qdat", "ship-a"]
    for name in DECODABLE:
        require_decodable(name)


@pytest.mark.parametrize("name", [
    "tlog-look", "look", "confirm.1", "confirm.2", "confirm.3", "confirm.4", "tlog",
    "qdat_bench", "qud-other", "qud_other", "tadabur-dev", "tadabur_dev", "g6", "", "unknown",
])
def test_every_round_3_input_is_refused_with_its_name(name):
    with pytest.raises(ScreenStop, match=name or "set"):
        require_decodable(name)


def test_decode_plan_refuses_a_round_3_input_too():
    with pytest.raises(ScreenStop, match="confirm.1"):
        decode_plan("confirm.1")


# --- the teacher's letter-level bars and the erratum ------------------------------------------


def test_the_teachers_edit_bars_are_the_244_letter_level_ones():
    # edits are already letter-level (teacher_agrees) and carry no G6 site: 34/150 and 88/150
    bars = teacher_bars()
    assert bars.edits_heard == pytest.approx(100 * 34 / 150)
    assert bars.decoys_unchanged == pytest.approx(100 * 88 / 150)
    assert 0.0 <= bars.mistakes_flagged <= 100.0 and 0.0 <= bars.correct_flagged <= 100.0


def test_the_erratum_lives_in_docs():
    assert ERRATUM_PATH == TOOLS.parent / "docs" / "fastconformer-screen-erratum.md"


def write_erratum(path: Path, bars: Rates) -> Path:
    path.write_text(
        "# Erratum, 2026-11-04\n\nThe teacher's letter-level bars, minus G6:\n"
        f"- mistakes flagged {bars.mistakes_flagged:.1f}%\n- correct flagged {bars.correct_flagged:.1f}%\n"
        f"- edits heard {bars.edits_heard:.1f}%\n- decoys unchanged {bars.decoys_unchanged:.1f}%\n",
        encoding="utf-8")
    return path


def test_an_erratum_with_the_bars_passes(tmp_path):
    require_erratum(BARS, write_erratum(tmp_path / "erratum.md", BARS))


def test_a_missing_erratum_is_refused(tmp_path):
    with pytest.raises(ScreenStop, match="erratum"):
        require_erratum(BARS, tmp_path / "erratum.md")


def test_an_erratum_that_lacks_a_bar_is_refused(tmp_path):
    path = write_erratum(tmp_path / "erratum.md", BARS)
    path.write_text(path.read_text(encoding="utf-8").replace("63.4", "63.5"), encoding="utf-8")
    with pytest.raises(ScreenStop, match="63.4"):
        require_erratum(BARS, path)


def test_the_decode_refuses_without_the_erratum(tmp_path):
    with pytest.raises(ScreenStop, match="erratum"):
        preflight("ship-a", BARS, erratum_path=tmp_path / "erratum.md")


def test_the_decode_preflight_returns_the_plan_once_the_erratum_is_in(tmp_path):
    erratum = write_erratum(tmp_path / "erratum.md", BARS)
    assert preflight("edits", BARS, erratum_path=erratum) == decode_plan("edits")


def test_the_preflight_refuses_a_round_3_input_even_with_the_erratum(tmp_path):
    erratum = write_erratum(tmp_path / "erratum.md", BARS)
    with pytest.raises(ScreenStop, match="qdat_bench"):
        preflight("qdat_bench", BARS, erratum_path=erratum)


# --- the owner's environment constraint ----------------------------------------------------------


def test_nothing_is_added_to_the_requirements_files():
    for path in TOOLS.glob("requirements*.txt"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(name in text for name in ("nemo", "onnxruntime", "onnx-asr")), path.name

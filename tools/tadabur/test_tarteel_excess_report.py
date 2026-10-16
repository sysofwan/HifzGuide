"""Tests for tadabur.tarteel_excess_report on synthetic verdicts (#160)."""

from __future__ import annotations

import pytest

from tadabur import tarteel_excess as te
from tadabur.listening_session import Verdict
from tadabur.tarteel_excess_report import POOLED, report, table


def _row(k: int, stratum: str, population: int, drawn: int, clip_words: int = 1,
         surah: int | None = None) -> te.ListeningWord:
    draw = drawn / population
    return te.ListeningWord(
        site_id=f"tarteel_v1:{k}_1_{k}.wav#0:w1", audio_filename=f"{k}_1_{k}.wav",
        audio_sha256="e" * 64, surah_ayah="1:1", cluster=k if surah is None else surah,
        start_sample=0, end_sample=1600, reference="قَالَ", word_offsets=(0, 5), word=1,
        position_from=0, position_to=1, h448_flag={}, stratum=stratum,
        stratum_words=population, stratum_population=population, clip_words=clip_words,
        representative_probability=1 / clip_words, draw_probability=draw,
        inclusion_probability=draw / clip_words)


def _verdicts(rows, heard) -> dict[str, Verdict]:
    return {r.site_id: Verdict(r.site_id, h) for r, h in zip(rows, heard) if h is not None}


def test_shares_are_weighted_by_the_design_per_stratum_and_pooled():
    # ل→ن: 4 of 100 representatives drawn (weight 25); ج→ش: 4 of 8 (weight 2).
    rows = ([_row(k, "ل→ن", 100, 4) for k in range(4)]
            + [_row(k, "ج→ش", 8, 4) for k in range(4, 8)])
    heard = ["correct", "correct", "correct", "mistake",  # ل→ن: 3 of 4 model errors
             "mistake", "mistake", "unclear", None]       # ج→ش: one unanswered
    result = report(rows, _verdicts(rows, heard))
    nun = result["strata"]["ل→ن"]
    assert (nun["drawn"], nun["answered"], nun["estimated_frame_words"]) == (4, 4, 100.0)
    assert nun["correct"]["share"] == 0.75 and nun["correct"]["interval"]["method"] == "wilson"
    jim = result["strata"]["ج→ش"]
    assert jim["answered"] == 3 and jim["mistake"]["share"] == pytest.approx(0.6667, abs=1e-4)
    assert jim["unclear"]["share"] == pytest.approx(0.3333, abs=1e-4)
    assert jim["model_error_of_clear"] == {**jim["model_error_of_clear"], "words": 0, "of": 2,
                                           "share": 0.0}
    # Pooled: (3 * 25) / (4 * 25 + 3 * 2) correct, not the unweighted 3 / 7.
    pooled = result[POOLED]
    assert pooled["correct"]["share"] == pytest.approx(75 / 106, abs=1e-4)
    assert pooled["model_error_of_clear"]["share"] == pytest.approx(75 / 104, abs=1e-4)
    assert pooled["estimated_frame_words"] == 108.0
    assert result["strata"]["ز→س"]["correct"]["share"] is None  # nothing drawn there
    assert result["verdicts"] == {"correct": 3, "mistake": 3, "unclear": 1}
    assert "| ل→ن | 4 | 4 | 75%" in table(result)


def test_a_clip_s_other_excess_words_raise_its_weight():
    rows = [_row(0, "other", 10, 2, clip_words=3), _row(1, "other", 10, 2)]
    result = report(rows, _verdicts(rows, ["correct", "mistake"]))
    assert result["strata"]["other"]["correct"]["share"] == 0.75  # weights 15 and 5
    # Unequal weights in a sparse cell: no admissible bound.
    assert result["strata"]["other"]["correct"]["interval"]["method"] == "none"


def test_a_verdict_no_drawn_word_can_take_is_refused():
    rows = [_row(0, "other", 10, 2)]
    with pytest.raises(ValueError):
        report(rows, {"elsewhere": Verdict("elsewhere", "correct")})
    with pytest.raises(ValueError):
        report(rows, _verdicts(rows, ["fatha"]))


def test_controls_give_the_baseline_and_each_excess_cell_is_netted_against_it():
    rows = ([_row(k, "ل→ن", 100, 4) for k in range(4)]
            + [_row(k, "ج→ش", 8, 4) for k in range(4, 8)]
            + [_row(k, te.CONTROL, 40, 4) for k in range(8, 12)])
    heard = ["correct", "correct", "correct", "mistake",
             "mistake", "mistake", "unclear", None,
             "correct", "correct", "correct", "mistake"]  # controls: one mistake in four
    result = report(rows, _verdicts(rows, heard))
    control = result[te.CONTROL]
    assert (control["drawn"], control["mistake"]["share"]) == (4, 0.25)
    assert result[POOLED]["drawn"] == 8  # the pooled excess cell leaves the controls out
    net = result["net_of_control"]
    # Pooled excess mistakes (25 + 2 + 2) / 106, less the controls' 1 / 4.
    assert net[POOLED]["mistake"]["difference"] == pytest.approx(29 / 106 - 0.25, abs=1e-4)
    assert net["ل→ن"]["correct"]["difference"] == 0.0  # 3 / 4 against 3 / 4
    # Twelve words: sparse, and two disjoint groups have no paired score bound.
    assert net[POOLED]["mistake"]["interval"]["method"] == "none"
    assert net["ز→س"]["mistake"]["difference"] is None  # nothing drawn there
    assert "Net of the controls" in table(result)


def test_a_net_share_on_enough_surahs_has_a_bootstrap_interval():
    rows = ([_row(k, "other", 300, 30) for k in range(30)]
            + [_row(k, te.CONTROL, 1000, 20) for k in range(30, 50)])
    heard = ["mistake"] * 15 + ["correct"] * 15 + ["mistake"] * 2 + ["correct"] * 18
    net = report(rows, _verdicts(rows, heard))["net_of_control"][POOLED]["mistake"]
    assert net["difference"] == pytest.approx(0.5 - 0.1)
    interval = net["interval"]
    assert interval["method"] == "bootstrap" and interval["lower"] < 0.4 < interval["upper"]
    assert interval["lower"] > 0  # half the flags against a tenth of the controls


def test_six_words_against_twenty_controls_have_no_bound_for_their_difference():
    """Pooled support must not stand in for a six-word stratum's (Sol, #162)."""
    rows = ([_row(k, "ا→∅", 26, 6) for k in range(6)]
            + [_row(k, te.CONTROL, 4379, 20) for k in range(6, 26)])
    heard = ["mistake"] * 3 + ["correct"] * 3 + ["mistake"] * 5 + ["correct"] * 15
    net = report(rows, _verdicts(rows, heard))["net_of_control"]["ا→∅"]["mistake"]
    assert net["difference"] == pytest.approx(0.5 - 0.25)
    assert net["interval"]["method"] == "none" and net["interval"]["lower"] is None


def test_all_correct_verdicts_on_the_committed_worklist_certify_nothing():
    """Sol's reproduction (#162): every answer ``correct`` gave the six-word strata a
    bootstrap interval of [0, 0]. A degenerate group has no bound, however many words."""
    rows = te.load_worklist()
    result = report(rows, _verdicts(rows, ["correct"] * len(rows)))
    for name, net in result["net_of_control"].items():
        for measure, cell in net.items():
            assert cell["interval"]["method"] == "none", (name, measure)
    assert result["net_of_control"][POOLED]["mistake"]["difference"] == 0.0

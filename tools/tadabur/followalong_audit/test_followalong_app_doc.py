"""``docs/followalong-app-arm.md`` is generated from the committed summary, never hand-edited (#309).

``render(summary)`` takes the summary the run commits (``SUMMARY_PATH``):

    {"population": {"clips": int, "failed": [{"item", "reason"}]},
     "rates": {"strict": flag_rates(...), "shipped_default": flag_rates(...)},
     "oracle": None | {...},            # S2 (#308) fills this in
     "p2": {"bar": 0.97, "verdict": "met" | "below", "source": <decode path>,
            "subset": agreement(...), "all": agreement(...)},
     "p3": {"device_ratios": {log: float | None}, "latencies_ms": [0, 130], "clips": int,
            "cadence": {"strict": cadence_report(...), "shipped_default": cadence_report(...)}},
     "pools": pool_sizes(...), "s1_by_cause_and_base": [{"cause", "base_flags", "words"}]}
"""

from __future__ import annotations

import copy
import json

from tadabur.followalong_audit.app_doc import DOC_PATH, SUMMARY_PATH, render
from tadabur.followalong_audit.bridge import agreement, bridge_verdict
from tadabur.followalong_audit.cadence import cadence_report
from tadabur.followalong_audit.rates import flag_rates
from tadabur.followalong_audit.strata import pool_sizes, stratify

SOURCE = "tadabur/tlog_mining/decodes/tlog.gating.r3_candidate_ema.jsonl.gz"


def cell(flagged: bool = False, graded: bool = True, *causes: str) -> dict:
    return {"graded": graded, "flagged": flagged, "causes": list(causes)}


def rows_of(item: str, cells: list[dict]) -> list[dict]:
    return [{"item": item, "word": index + 1, "strict": c, "shipped_default": cell()}
            for index, c in enumerate(cells)]


APP = (rows_of("a#0", [cell(True, True, "tashkeel")] + [cell()] * 9)
       + rows_of("b#0", [cell()] * 10))
TORCH = (rows_of("a#0", [cell()] * 10) + rows_of("b#0", [cell()] * 10))
SHIFTED = [{**row, "strict": cell(True, True, "tashkeel") if (row["item"], row["word"]) == ("b#0", 4)
            else row["strict"]} for row in APP]


def summary(p2_subset=None, cadence_other=None) -> dict:
    subset = agreement(APP, p2_subset or TORCH)
    return {
        "population": {"clips": 2, "failed": []},
        "rates": {name: flag_rates(APP, name, seed=1, iterations=100)
                  for name in ("strict", "shipped_default")},
        "oracle": None,
        "p2": {"bar": 0.97, "verdict": bridge_verdict(subset["rate"]), "source": SOURCE,
               "subset": subset, "all": subset},
        "p3": {"device_ratios": {"surah3_page51": 3.25, "owner 69#1": None},
               "latencies_ms": [0, 130], "clips": 500,
               "cadence": {name: cadence_report(APP, cadence_other or APP, name)
                           for name in ("strict", "shipped_default")}},
        "pools": pool_sizes(stratify([
            {"key": "a#0", "position": 0, "app": "passed", "oracle": None, "causes": [],
             "detectors": {"base": False, "c95": False, "r2_candidate_ema": False,
                           "r3_candidate_ema": False}}], catch_loss=set())),
        "s1_by_cause_and_base": [{"cause": "tashkeel", "base_flags": False, "words": 1}],
    }


def test_the_doc_is_headlined_placeable_phone_audio():
    assert "placeable phone audio" in render(summary()).splitlines()[0].lower()


def test_the_doc_names_both_scorings_and_gives_each_cause_rate_with_its_interval():
    text = render(summary())
    strict = summary()["rates"]["strict"]
    assert "strict" in text and "balanced" in text
    assert f"{strict['per_1000']:.1f}" in text
    low, high = strict["interval"]
    assert f"{low:.1f}" in text and f"{high:.1f}" in text
    assert "tashkeel" in text


def test_the_oracle_parts_say_they_are_pending_s2_while_there_is_no_oracle_result():
    assert render(summary()).count("pending S2 (#308)") >= 2


def test_p2_below_the_bar_is_reported_with_its_number_and_every_disagreeing_word():
    text = render(summary(p2_subset=SHIFTED))
    assert "below" in text and "97%" in text
    assert "95.0%" in text  # 19 of 20 words agree
    assert "b#0" in text


def test_p2_at_the_bar_does_not_claim_a_shortfall():
    met, short = render(summary()), render(summary(p2_subset=SHIFTED))
    assert "100.0%" in met
    assert met.lower().count("below") < short.lower().count("below")


def test_p2_states_it_reuses_the_committed_pytorch_decode_and_names_it():
    text = render(summary())
    assert SOURCE.rsplit("/", 1)[-1] in text
    assert "reuse" in text.lower()


def test_p3_reports_the_ratios_and_the_share_of_words_that_changed():
    text = render(summary())
    assert "3.25" in text and "130" in text and "500" in text


def test_p3_over_two_percent_is_a_follow_along_finding_and_otherwise_is_not():
    quiet = render(summary()).lower().count("follow-along finding")
    moved = [{**row, "strict": cell(True, True, "tashkeel") if row["word"] <= 2 else row["strict"]}
             for row in rows_of("a#0", [cell()] * 10) + rows_of("b#0", [cell()] * 10)]
    assert render(summary(cadence_other=moved)).lower().count("follow-along finding") > quiet


def test_rendering_is_deterministic_and_ends_with_one_newline():
    first = render(summary())
    assert render(copy.deepcopy(summary())) == first
    assert first.endswith("\n") and not first.endswith("\n\n")


def test_the_committed_doc_is_the_committed_summary_rendered():
    committed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert DOC_PATH.read_text(encoding="utf-8") == render(committed)

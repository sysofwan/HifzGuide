"""The summary carries its run identity, and the doc renders it (#309, S1.5, S1.6, S1.14).

The committed numbers only mean something beside the weights, engine, latency, seeds and the
tables' digests that produced them; the App latency must come from that block, not from prose.
The catch-loss count must show ``marked`` beside ``matched`` so a zero is visible, and the P2
disagreement counters must account for every row of the exhaustive table.
"""

from __future__ import annotations

from tadabur.followalong_audit.app_doc import render
from tadabur.followalong_audit.audit_report import build_summary
from tadabur.followalong_audit.bridge import agreement, bridge_verdict
from tadabur.followalong_audit.cadence import cadence_report
from tadabur.followalong_audit.rates import flag_rates
from tadabur.followalong_audit.strata import pool_sizes, stratify


def cell(flagged: bool = False, graded: bool = True) -> dict:
    return {"graded": graded, "flagged": flagged, "causes": []}


def rows_of(item: str, cells: list[dict]) -> list[dict]:
    return [{"item": item, "word": index + 1, "strict": cell_of, "shipped_default": cell()}
            for index, cell_of in enumerate(cells)]


APP = rows_of("a#0", [cell()] * 4)
TORCH = rows_of("a#0", [cell()] * 4)

RUN = {
    "muraja_commit": "028bc513eeb233caa7240a5c37c423da59f3d9d7",
    "harness": {"binary_sha256": "a" * 64},
    "release": "v1.0.7",
    "latency_ms": 17,
    "seeds": {"p2": 319, "p3": 309},
    "driver_sha256": "b" * 64,
    "audio_manifest_sha256": "c" * 64,
    "word_grades_sha256": "d" * 64,
}

CATCH_LOSS = {"marked": 150, "matched": 0,
              "worklist": "consonant2_readout/catch_loss/worklist.jsonl"}


def summary(*, run=None, catch_loss=None, p2=None, p3=None) -> dict:
    subset = agreement(APP, TORCH)
    word = {"key": "a#0", "position": 0, "app": "passed", "oracle": None, "causes": [],
            "detectors": {"base": False, "c95": False, "r2_candidate_ema": False,
                          "r3_candidate_ema": False}}
    return {
        "population": {"clips": 1, "failed": []},
        "run": run,
        "rates": {name: flag_rates(APP, name, seed=1, iterations=50)
                  for name in ("strict", "shipped_default")},
        "oracle": None,
        "p2": p2 or {"bar": 0.97, "verdict": bridge_verdict(subset["rate"]),
                     "source": "tadabur/tlog_mining/decodes/tlog.gating.r3_candidate_ema.jsonl.gz",
                     "subset": subset, "all": subset},
        "p3": p3 or {"device_ratios": {"log": 1.5}, "latencies_ms": [0, 130], "clips": 4,
                     "seed": 309,
                     "cadence": {name: cadence_report(APP, APP, name)
                                 for name in ("strict", "shipped_default")}},
        "pools": pool_sizes(stratify([word], catch_loss=set())),
        "s1_by_cause_and_base": [],
        "catch_loss": catch_loss,
    }


def test_the_run_block_is_rendered_with_weights_engine_latency_and_digests():
    text = render(summary(run=RUN))
    assert RUN["muraja_commit"] in text and "v1.0.7" in text
    assert "17 ms per inference" in text
    assert "319" in text and "309" in text
    assert "c" * 64 in text and "d" * 64 in text


def test_without_a_run_block_the_doc_claims_no_app_latency():
    text = render(summary())
    assert "CoreML decoder; " in text
    assert "CoreML decoder at" not in text


def test_the_catch_loss_count_shows_marked_beside_matched():
    text = render(summary(catch_loss=CATCH_LOSS))
    assert "marks 150 words; 0 of them are in this population" in text


def test_a_nonzero_catch_loss_match_is_reported_as_such():
    text = render(summary(catch_loss={**CATCH_LOSS, "matched": 3}))
    assert "3 of them are in this population" in text


def test_a_grade_presence_disagreement_has_its_own_counter():
    shifted = rows_of("a#0", [cell(graded=False)] + [cell()] * 3)
    subset = agreement(APP, shifted)
    assert subset["disagreements"]
    p2 = {"bar": 0.97, "verdict": bridge_verdict(subset["rate"]),
          "source": "x", "subset": subset, "all": subset}
    assert "graded on one side only (ungraded against a grade): 1" in render(summary(p2=p2))


def test_a_skipped_p2_clip_is_named():
    p2 = {"bar": 0.97, "verdict": "met", "source": "x", "subset": agreement(APP, TORCH),
          "all": None, "skipped": ["b#0"]}
    assert "Skipped (no word table): 1 clip(s): b#0." in render(summary(p2=p2))


def test_the_summary_keeps_the_run_block_and_the_catch_loss_counts():
    built = build_summary(APP, clips=1, failed=[], p2=summary()["p2"], p3=summary()["p3"],
                          pools={"pools": summary()["pools"], "s1_by_cause_and_base": [],
                                 "catch_loss": CATCH_LOSS},
                          run=RUN)
    assert built["run"] == RUN and built["catch_loss"] == CATCH_LOSS

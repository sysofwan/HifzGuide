"""The fixture-side report's runner (#55): its inputs and the rendered report. Torch-free."""

from __future__ import annotations

from tadabur.eval_harness import build, fixture_sides, render
from tadabur.eval_report import SHOULD_ACCEPT, SHOULD_REJECT
from tadabur.listening_session import Verdict, adjudicated
from tadabur.p35_truth_sites import SITES_PATH
from tadabur.test_eval_report import PAIR, ZAI, cohort, report_for  # noqa: F401  (a fixture)
from tadabur.truth_sites import load_truth_sites
from training.truth_baseline import DEFAULT_MODELS


def test_every_p35_site_has_the_fixture_side_it_was_relocated_from():
    sites = load_truth_sites(SITES_PATH)
    sides = fixture_sides()
    assert set(sides) == {s.site_id for s in sites}
    assert set(sides.values()) == {SHOULD_ACCEPT, SHOULD_REJECT}
    for site in sites:  # an accept says the mushaf's value was said there
        if sides[site.site_id] == SHOULD_ACCEPT:
            assert site.heard == site.prescribed


def test_the_report_builds_from_the_committed_decode_caches_without_decoding():
    report = build(DEFAULT_MODELS, audio_dir=None)
    assert report["arms"] == ["base/spans", "base/stream_b0", "h448/spans", "h448/stream_b0"]
    assert len(report["sites"]) == len(load_truth_sites(SITES_PATH))
    assert report["p35_base_reproduction"]["identical"] == report["p35_base_reproduction"]["items"]
    for arm in report["arms"]:
        assert report["fingerprints"]["arms"][arm]["identity"]["model_ref"] in DEFAULT_MODELS.values()


def test_a_side_with_no_verdict_renders_as_pending_adjudication(cohort):
    sites, sides, reciters = cohort
    pending_only = [s for s in sites if s.heard != ZAI]
    text = render(report_for((pending_only, sides, reciters), {"m": {}}))
    assert "**Pending adjudication.**" in text
    assert "mushaf's value (collapsed)" not in text.split("## Sites without a verdict")[0].split(
        "**Pending adjudication.**")[1]
    assert "| shaddah (provisional) | held → held | 1 / 1 (too small) | 1 / 0 |" in text


def test_a_side_with_verdicts_renders_its_own_tables(cohort):
    sites, sides, reciters = cohort
    heard = adjudicated(sites, {sites[3].site_id: Verdict(sites[3].site_id, ZAI)})
    text = render(report_for((heard, sides, reciters), {"m": {}}))
    mistakes = text.split("## Real mistakes")[1].split("## Sites without a verdict")[0]
    assert "Pending adjudication" not in mistakes
    assert "mushaf's value (collapsed)" in mistakes
    # Both decoded the mushaf's ذ: no mistake heard, both collapsed onto the reference.
    line = next(x for x in mistakes.splitlines() if x.startswith(f"| {PAIR} | ذ → ز |"))
    assert line.startswith(f"| {PAIR} | ذ → ز | 2 / 2 (too small) | 0 / 2 | 0 | 0 · 0.0 ")
    assert line.split(" | ")[6].startswith("2 · 100.0")


def test_a_rate_with_sites_awaiting_a_verdict_shows_its_range(cohort):
    text = render(report_for(cohort, {"m": {}}))
    correct = text.split("## Correct recitation")[1].split("## Real mistakes")[0]
    line = next(x for x in correct.splitlines() if x.startswith(f"| {PAIR} | ذ → ذ |"))
    # Two faithful accepts; the pending reject (mushaf ذ) could be a third correct site.
    assert "| 2 / 0 | 1 | 2 · 100.0" in line and "⟨66.7–100.0⟩" in line


def test_no_site_without_a_verdict_renders_as_none(cohort):
    sites, sides, reciters = cohort
    decided = [s for s in sites if s.heard != "pending"]
    text = render(report_for((decided, sides, reciters), {"m": {}}))
    assert text.split("## Sites without a verdict")[1].split("## Fingerprints")[0].strip().endswith("None.")

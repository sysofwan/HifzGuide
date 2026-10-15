"""Tests for training.nonpro_gap: the tally, the pooled and paired intervals, the committed grid
and its sample (#159).

Torch-free; the decodes are synthetic, the grid and the exposure registry the committed ones.
"""

from __future__ import annotations

import pytest

from training import nonpro_gap as ng
from training.perturbation import CONDITIONS

# Two words, ``قَالَ لَننَ``: a ق, an ل and the geminate نن (the focus group), each with a haraka.
REFERENCE = "قَالَ لَننَ"
WIDTH = 2 * len(ng.CLASSES)


def _rates(vector) -> dict:
    return {name: vector[2 * k] / vector[2 * k + 1] for k, name in enumerate(ng.CLASSES)}


def test_a_decode_identical_to_its_reference_agrees_everywhere():
    counts = ng.tally(REFERENCE, REFERENCE)
    assert counts == (len(REFERENCE), len(REFERENCE), 4, 4, 3, 3, 4, 4, 1, 1)


def test_a_focus_consonant_misheard_counts_in_its_group_and_in_every_consonant():
    rates = _rates(ng.tally("قَالَ نَننَ", REFERENCE))
    assert rates["consonant"] == 3 / 4 and rates["focus"] == 2 / 3
    assert rates["phoneme"] == 1 - 1 / len(REFERENCE)


def test_an_empty_slot_is_a_disagreement():
    rates = _rates(ng.tally("قَالَ", REFERENCE))
    assert rates["consonant"] == 2 / 4 and rates["shaddah"] == 0.0


def test_rates_pool_over_clusters_and_leave_a_class_with_no_slot_out():
    rows = [("r1", ng.tally(REFERENCE, REFERENCE)), ("r1", ng.tally("قَالَ", REFERENCE)),
            ("r2", ng.tally("قَالَ نَننَ", REFERENCE))]
    out = ng.rates(rows)
    assert out["consonant"]["point"] == pytest.approx((4 + 2 + 3) / 12, abs=1e-5)
    assert out["consonant"]["slots"] == 12 and out["consonant"]["clusters"] == 2
    assert out["consonant"]["low"] <= out["consonant"]["point"] <= out["consonant"]["high"]
    no_geminate = ng.rates([("r1", ng.tally("قَالَ", "قَالَ"))])
    assert no_geminate["shaddah"] is None and no_geminate["consonant"]["point"] == 1.0


def test_a_paired_difference_of_identical_tallies_is_zero():
    rows = [(f"r{i}", ng.tally(d, REFERENCE) * 2)
            for i, d in enumerate([REFERENCE, "قَالَ", "قَالَ نَننَ", "لَ لَننَ"])]
    out = ng.paired(rows, lambda r: r(WIDTH) - r(0))
    assert all(v == {"point": 0.0, "low": 0.0, "high": 0.0, "clusters": 4} for v in out.values())
    worse = [(c, v[:WIDTH] + ng.tally("قَالَ", REFERENCE)) for c, v in rows]
    assert ng.paired(worse, lambda r: r(0) - r(WIDTH))["consonant"]["point"] > 0


def test_the_committed_grid_is_the_codes_and_its_sample_reproduces():
    committed = ng.committed_grid()
    assert committed == ng.grid_file()
    sample = committed["sample"]
    assert len(sample["clips"]) == ng.D2_SAMPLE_SIZE == len(set(sample["clips"]))
    assert not set(sample["clips"]) & set(sample["h448_seen"])


def test_each_condition_decodes_the_dev_half_or_the_sample():
    dev = ng.dev_clips()
    assert len(dev) == 970 and all(c.cached_teacher for c in dev)
    clean = ng.decode_plan(CONDITIONS["clean"])
    noisy = ng.decode_plan(CONDITIONS["noise_snr10"])
    assert clean == dev
    assert [c.filename for c in noisy] == ng.committed_grid()["sample"]["clips"]


def _entry(point: float) -> dict:
    return {"point": point, "low": point - 0.01, "high": point + 0.01, "clusters": 3,
            "slots": 100}


def _report() -> dict:
    from training.perturbation import GRID, grid_record

    rates = {c: _entry(0.9) for c in ng.CLASSES}
    block = {"items": 10, "rates": rates}
    gap = {c: _entry(0.02) for c in ng.CLASSES}
    return {
        "grid": grid_record(),
        "sample": {"eligible": 671, "h448_seen": ["a"] * 299, "dev_clips": 970,
                   "reciters": 107, "hours": 1.676},
        "noise": {"recordings": 445, "hours": 4.08, "left_out": {"ineligible": 84},
                  "recordings_drawn": 300, "licences_drawn": {"CC0-1.0": 200}},
        "numerics": {f"{m}_vs_cached_teacher": rates for m in ng.MODELS},
        "d1": {"tadabur_dev": {"stream_b0": block}, "tadabur_dev_unseen": {"stream_b0": block},
               **{k: {"stream_b0": block, "checks": block}
                  for k in ("qud", "tarteel_v1", "qdat")}},
        "d2": {"clean": {"teacher": {"h448": rates}},
               **{c.name: {"own": {m: rates for m in ng.MODELS},
                           "teacher": {m: rates for m in ng.MODELS},
                           "excess_drift": gap, "excess_clone_drop": gap} for c in GRID[1:]}},
        "compute": {"gpu_hours": 0.5, "parallel_runs": 3, "gpu": "GPU", "audio_hours": 17.0,
                    "peak_scratch_gb": 2.1, "disk_sampled_from": "t", "aborted_attempt": None},
    }


def test_the_doc_renders_every_corpus_condition_and_class():
    from training.nonpro_gap_doc import CLASS_NAMES, CORPORA, render

    doc = render(_report())
    for name in [*CORPORA.values(), *CLASS_NAMES.values(), "opus_12k_nb", "reverb_rt1"]:
        assert name in doc
    assert "90.0% [89.0, 91.0]" in doc and "+2.0 [+1.0, +3.0]" in doc
    assert "`h448` further (interval above zero)" in doc

"""Tests for the fixed-window duration/memory envelope (training.window_envelope)."""

from __future__ import annotations

import json

import pytest

from training import window_envelope as we
from training.waqf_distill import muaalem_lattice_length


# ---------------------------------------------------------------------------
# Frame ↔ time geometry.
# ---------------------------------------------------------------------------


def test_deployed_window_is_250_feature_frames_125_logit_frames():
    frames = we.seconds_to_feature_frames(we.DEPLOYED_WINDOW_SECONDS)
    assert frames == 250
    assert muaalem_lattice_length(frames) == 125


def test_seconds_frames_round_trip():
    assert we.feature_frames_to_seconds(we.seconds_to_feature_frames(8.0)) == 8.0


# ---------------------------------------------------------------------------
# Duration histogram.
# ---------------------------------------------------------------------------


def test_histogram_bins_including_open_ended_tail():
    edges = (0.0, 5.0, 10.0)
    # 2 bins: [0,5) and the open-ended >=5 tail.
    hist = we.duration_histogram("t", [1.0, 4.9, 5.0, 9.9, 12.0, 100.0], edges)
    assert hist.counts == (2, 4)
    assert hist.count == 6


def test_histogram_boundary_value_lands_in_upper_bin():
    hist = we.duration_histogram("t", [5.0], (0.0, 5.0, 10.0))
    assert hist.counts == (0, 1)


def test_histogram_percentiles_and_exceedance():
    durations = [float(i) for i in range(1, 101)]  # 1..100
    hist = we.duration_histogram("t", durations, (0.0, 50.0, 200.0))
    assert hist.minimum == 1.0
    assert hist.maximum == 100.0
    assert hist.percentiles[50] == pytest.approx(50.0, abs=1.0)
    # exactly 50 of 100 values (51..100) are > 50
    assert hist.fraction_exceeding(50.0) == pytest.approx(0.5)
    assert hist.fraction_at_most(50.0) == pytest.approx(0.5)


def test_histogram_rejects_empty_population():
    with pytest.raises(ValueError, match="no durations"):
        we.duration_histogram("t", [], (0.0, 1.0))


def test_histogram_requires_two_edges():
    with pytest.raises(ValueError, match="two bin edges"):
        we.duration_histogram("t", [1.0], (0.0,))


def test_histogram_is_order_independent():
    edges = (0.0, 5.0, 10.0)
    a = we.duration_histogram("t", [1.0, 6.0, 3.0, 9.0], edges)
    b = we.duration_histogram("t", [9.0, 3.0, 6.0, 1.0], edges)
    assert a == b


# ---------------------------------------------------------------------------
# Manifest loading.
# ---------------------------------------------------------------------------


def test_load_whole_clip_durations(tmp_path):
    manifest = tmp_path / "passing.jsonl"
    rows = [
        {"audio_filename": "a.wav", "surah_ayah": "1:1", "match_ratio": 1.0,
         "ayah_duration_s": 3.5, "reciter_id": 0},
        {"audio_filename": "b.wav", "surah_ayah": "1:2", "match_ratio": 0.9,
         "ayah_duration_s": 12.0, "reciter_id": 1},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert we.load_whole_clip_durations(manifest) == [3.5, 12.0]


def test_load_segment_durations_uses_span(tmp_path):
    manifest = tmp_path / "segments.jsonl"
    rows = [
        {"start_s": 0.0, "end_s": 2.5},
        {"start_s": 2.5, "end_s": 7.0},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    assert we.load_segment_durations(manifest) == [2.5, 4.5]


# ---------------------------------------------------------------------------
# Memory estimate.
# ---------------------------------------------------------------------------


def test_memory_estimate_deployed_window_fits():
    est = we.memory_estimate(250, batch=1, checkpointed=True)
    assert est.fits_usable
    assert est.total_gb < we.USABLE_VRAM_GB
    # weights ≈ 606M * 2 bytes
    assert est.weights_gb == pytest.approx(606_000_000 * 2 / (1024 ** 3), rel=1e-6)


def test_activation_memory_grows_with_batch_and_window():
    base = we.memory_estimate(250, batch=1).activation_gb
    assert we.memory_estimate(250, batch=4).activation_gb > base
    assert we.memory_estimate(500, batch=1).activation_gb > base


def test_activation_superlinear_in_window_length():
    # The attention term makes activations grow faster than linearly with seq length.
    a = we.memory_estimate(250, batch=1).activation_gb
    b = we.memory_estimate(500, batch=1).activation_gb
    assert b > 2 * a


def test_checkpointing_reduces_activation_memory():
    ckpt = we.memory_estimate(250, batch=8, checkpointed=True).activation_gb
    full = we.memory_estimate(250, batch=8, checkpointed=False).activation_gb
    assert ckpt < full


def test_memory_estimate_rejects_nonpositive():
    with pytest.raises(ValueError):
        we.memory_estimate(0, batch=1)
    with pytest.raises(ValueError):
        we.memory_estimate(250, batch=0)


def test_max_micro_batch_monotonic_and_bounded():
    small = we.max_micro_batch(250)
    large = we.max_micro_batch(2000)
    assert small >= 1
    assert small > large  # a longer window fits fewer per batch
    # every reported batch actually fits, the next one does not
    assert we.memory_estimate(250, small).fits_usable
    assert not we.memory_estimate(250, small + 1).fits_usable


def test_whole_clip_window_ooms_without_checkpointing():
    # A ~80 s whole clip (4000 frames) must not fit even one example without checkpointing;
    # checkpointing is what makes long windows fit, so the window is pinned by the ANE, not
    # by training memory.
    assert not we.memory_estimate(we.seconds_to_feature_frames(80.0), 1, checkpointed=False).fits_usable


# ---------------------------------------------------------------------------
# Window candidates.
# ---------------------------------------------------------------------------


def _demo_histograms():
    clips = we.duration_histogram("clips", [x / 2 for x in range(1, 200)], we._CLIP_EDGES)
    segs = we.duration_histogram("segs", [x / 4 for x in range(1, 200)], we._SEGMENT_EDGES)
    return clips, segs


def test_window_candidates_flag_deployed_and_measure_logit_length():
    clips, segs = _demo_histograms()
    candidates = we.window_candidates(clips, segs)
    deployed = [c for c in candidates if c.deployed]
    assert len(deployed) == 1
    assert deployed[0].seconds == we.DEPLOYED_WINDOW_SECONDS
    assert deployed[0].student_frames == 125
    # coverage is a fraction
    for c in candidates:
        assert 0.0 <= c.clip_coverage <= 1.0
        assert 0.0 <= c.segment_coverage <= 1.0


def test_window_candidate_coverage_increases_with_window():
    clips, segs = _demo_histograms()
    candidates = we.window_candidates(clips, segs)
    coverages = [c.clip_coverage for c in candidates]
    assert coverages == sorted(coverages)  # longer windows cover at least as much


def test_window_candidates_accept_arbitrary_lengths():
    # Coverage is computed from the stored population, so any candidate length works.
    clips, segs = _demo_histograms()
    candidates = we.window_candidates(clips, segs, candidate_seconds=(7.5,))
    assert candidates[0].seconds == 7.5
    assert candidates[0].feature_frames == 375


# ---------------------------------------------------------------------------
# Policy options + recommendation.
# ---------------------------------------------------------------------------


def test_policy_options_hop_and_overlap_consistent():
    options = we.policy_options()
    assert len(options) == 2
    tiling = options[0]
    assert tiling.overlap_seconds == 0.0
    assert tiling.hop_seconds == tiling.window_seconds
    assert tiling.windows_per_multiple == 1.0
    overlap = options[1]
    assert overlap.overlap_seconds > 0.0
    assert overlap.hop_seconds < overlap.window_seconds
    assert overlap.windows_per_multiple > 1.0


def test_recommendation_mentions_deployed_window_and_overlap():
    clips, segs = _demo_histograms()
    candidates = we.window_candidates(clips, segs)
    text = we.recommendation(candidates)
    assert "5 s" in text
    assert "overlap" in text.lower()


# ---------------------------------------------------------------------------
# End-to-end report.
# ---------------------------------------------------------------------------


def test_render_report_is_nonempty_markdown():
    clips, segs = _demo_histograms()
    candidates = we.window_candidates(clips, segs)
    report = we.render_report(clips, segs, candidates, we.policy_options())
    assert report.startswith("# Fixed-window duration/memory envelope")
    assert "## 1. Duration histograms" in report
    assert "## 2. Per-window logit length" in report
    assert "## 3. Candidate windowing policies" in report
    assert "## 4. Recommendation for A2" in report


def test_build_report_end_to_end(tmp_path):
    whole = tmp_path / "passing.jsonl"
    whole.write_text(
        "\n".join(
            json.dumps({"audio_filename": f"c{i}.wav", "surah_ayah": "1:1",
                        "match_ratio": 1.0, "ayah_duration_s": float(i), "reciter_id": 0})
            for i in range(1, 30)
        ),
        encoding="utf-8",
    )
    segments = tmp_path / "segments.jsonl"
    segments.write_text(
        "\n".join(json.dumps({"start_s": 0.0, "end_s": float(i)}) for i in range(1, 30)),
        encoding="utf-8",
    )
    report = we.build_report(whole, segments)
    assert "Recommendation for A2" in report

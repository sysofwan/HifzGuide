"""Unit tests for the waqf post-processing reference (``tadabur.waqf_postprocess``).

Covers each stage in isolation — the A2 center-trusted window stitch, the 300/700 ms
frame-level silence cleaning, and the phoneme-alignment boundary snap with mid-word-closure
rejection (including that a madd elongation is never a stop) — plus a **fixture-driven** end
-to-end check that lays every frozen F0 clip's real verdict sequence onto a controlled 40 ms
timeline and asserts the reference recovers exactly the true-waqf set and rejects every
mid-word closure. All torch-free, no audio or model.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pytest

from .waqf_event_fixtures import _FIXTURE_DIR
from .waqf_postprocess import (
    DEFAULT_INTERIOR_TOL_FRAMES,
    STUDENT_FRAME_MS,
    SilenceRun,
    WindowFrames,
    WordSpan,
    detect_pauses,
    seconds_to_frame,
    snap_pauses,
    stitch_silence,
    waqf_events,
    waqf_events_from_windows,
)

# 40 ms frames: min silence 300 ms ≈ 8 frames, min speech 700 ms ≈ 18 frames.
_MIN_SILENCE_FRAMES = 8
_MIN_SPEECH_FRAMES = 18


def _silence(*spans: tuple[int, int], length: int) -> np.ndarray:
    """A speech (0.0) track of ``length`` frames with 1.0 silence over each ``[a, b)`` span."""
    track = np.zeros(length, dtype=np.float32)
    for a, b in spans:
        track[a:b] = 1.0
    return track


# ---------------------------------------------------------------------------
# Stitch (A2 center-trusted overlap).
# ---------------------------------------------------------------------------


def test_stitch_single_window_is_identity():
    frames = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    np.testing.assert_array_equal(stitch_silence([WindowFrames(0, frames)]), frames)


def test_stitch_empty_is_empty():
    assert stitch_silence([]).size == 0


def test_stitch_center_trusted_splits_overlap_at_midpoint():
    # Two 10-frame windows, start 0 and start 6 → overlap frames [6, 10). Each window's value
    # is its index, so the owner is visible per frame. Centers: w0 at 5, w1 at 11.
    w0 = WindowFrames(0, np.zeros(10, dtype=np.float32))
    w1 = WindowFrames(6, np.ones(10, dtype=np.float32))
    track = stitch_silence([w1, w0])  # unsorted input → must sort by start
    assert track.shape == (16,)
    # Overlap [6,10): frame 6,7 nearer w0's center (5); frame 8 is equidistant (d=3) → tie to the
    # earlier window w0; frame 9 nearer w1's center (11). So w0 owns [0,9), w1 owns [9,16).
    np.testing.assert_array_equal(track[:9], np.zeros(9, dtype=np.float32))
    np.testing.assert_array_equal(track[9:], np.ones(7, dtype=np.float32))


def test_stitch_tail_window_owns_its_center():
    # A short tail window past the previous window's end still owns its own frames.
    w0 = WindowFrames(0, np.zeros(10, dtype=np.float32))
    w1 = WindowFrames(8, np.ones(6, dtype=np.float32))  # covers [8, 14), extends past 10
    track = stitch_silence([w0, w1])
    assert track.shape == (14,)
    assert track[13] == 1.0  # tail owned by w1 (only it covers it)
    assert track[0] == 0.0


def test_stitch_uncovered_frame_fails_loudly():
    # A hole between windows (frames [5, 8) covered by neither) is a broken tiling.
    with pytest.raises(ValueError, match="covered by no window"):
        stitch_silence(
            [WindowFrames(0, np.zeros(5, dtype=np.float32)), WindowFrames(8, np.zeros(5, dtype=np.float32))]
        )


# ---------------------------------------------------------------------------
# Silence runs + 300/700 ms cleaning.
# ---------------------------------------------------------------------------


def test_detect_pause_interior_gap():
    # speech(20) | silence(10) | speech(20): one interior pause on the gap.
    track = _silence((20, 30), length=50)
    assert detect_pauses(track) == [SilenceRun(20, 30)]


def test_detect_pause_leading_and_trailing_silence_excluded():
    # Silence before the first speech / after the last is a clip edge, not an interior stop.
    track = _silence((0, 10), (40, 50), length=50)  # lead + trail silence, no interior gap
    assert detect_pauses(track) == []


def test_detect_short_silence_is_merged_not_a_stop():
    # A sub-300 ms silence (5 frames = 200 ms) between speech is not a waqf.
    track = _silence((20, 25), length=50)
    assert detect_pauses(track) == []


def test_detect_short_speech_island_is_dropped():
    # speech(20) | sil(10) | speech(4=160ms<700ms) | sil(10) | speech(20): the tiny island is
    # not speech, so the two silences fuse into one pause spanning the whole [20, 44) gap.
    track = _silence((20, 30), (34, 44), length=64)
    assert detect_pauses(track) == [SilenceRun(20, 44)]


def test_detect_madd_dip_is_never_a_stop():
    # A madd (elongation) is speech; even a shallow silence dip during it stays under 300 ms,
    # so no pause is emitted — the elongation is not mistaken for a stop.
    track = _silence((25, 31), length=60)  # 6-frame = 240 ms dip < 300 ms
    assert detect_pauses(track) == []


def test_detect_threshold_binarises_at_argmax():
    # Posteriors straddling 0.5: >=0.5 is silence (the VAD argmax boundary).
    track = np.array([0.1] * 20 + [0.6] * 10 + [0.1] * 20, dtype=np.float32)
    assert detect_pauses(track) == [SilenceRun(20, 30)]
    # Below threshold everywhere → all speech → no pause.
    assert detect_pauses(np.full(50, 0.49, dtype=np.float32)) == []


# ---------------------------------------------------------------------------
# Boundary snap + mid-word-closure rejection.
# ---------------------------------------------------------------------------


def _two_words_with_gap() -> list[WordSpan]:
    # word 0 = [0, 20), gap [20, 30), word 1 = [30, 50).
    return [WordSpan(0, 0, 20), WordSpan(1, 30, 50)]


def test_snap_gap_pause_is_waqf_after_preceding_word():
    result = snap_pauses([SilenceRun(20, 30)], _two_words_with_gap())
    assert result.rejected_closures == []
    assert [(e.word_index, e.pause) for e in result.waqf] == [(0, SilenceRun(20, 30))]


def test_snap_interior_pause_is_rejected_closure():
    # A silence deep inside word 0's span (a qalqala/hamza closure) is not a waqf.
    word = [WordSpan(0, 0, 40), WordSpan(1, 40, 60)]
    result = snap_pauses([SilenceRun(15, 27)], word)
    assert result.waqf == []
    assert result.rejected_closures == [SilenceRun(15, 27)]


def test_snap_tolerates_small_boundary_jitter():
    # A genuine stop whose word alignment bleeds `tol` frames into the silence is still a waqf.
    tol = DEFAULT_INTERIOR_TOL_FRAMES
    words = [WordSpan(0, 0, 20 + tol), WordSpan(1, 30, 50)]  # word 0 end overlaps pause by tol
    result = snap_pauses([SilenceRun(20, 30)], words)
    assert [e.word_index for e in result.waqf] == [0]
    assert result.rejected_closures == []
    # One frame more overlap tips it over into a mid-word closure.
    words_over = [WordSpan(0, 0, 20 + tol + 1), WordSpan(1, 30, 50)]
    over = snap_pauses([SilenceRun(20, 30)], words_over)
    assert over.waqf == []
    assert over.rejected_closures == [SilenceRun(20, 30)]


def test_snap_leading_silence_before_first_word_is_neither():
    # A pause before any word has no word to follow: not a stop, not a closure.
    result = snap_pauses([SilenceRun(0, 10)], [WordSpan(0, 15, 35), WordSpan(1, 35, 55)])
    assert result.waqf == []
    assert result.rejected_closures == []


# ---------------------------------------------------------------------------
# End-to-end reference + determinism.
# ---------------------------------------------------------------------------


def test_waqf_events_end_to_end_from_windows():
    # Two windows stitch into: speech(0-20) gap(20-30) speech(30-...) with a mid-word closure
    # inside word 1, then a real stop after word 1.
    length = 80
    track = _silence((20, 30), (48, 58), length=length)  # [20,30) gap after w0; [48,58) inside w1
    # word 0 = [0,20); word 1 = [30, 70) (contains closure [48,58)); gap [30... actually stop after 70)
    words = [WordSpan(0, 0, 20), WordSpan(1, 30, 70)]
    # split the track across two overlapping windows
    win = [WindowFrames(0, track[:50]), WindowFrames(40, track[40:])]
    result = waqf_events_from_windows(win, words)
    assert [e.word_index for e in result.waqf] == [0]
    assert result.rejected_closures == [SilenceRun(48, 58)]


def test_waqf_events_deterministic():
    track = _silence((20, 30), length=60)
    words = _two_words_with_gap()
    first = waqf_events(track, words)
    second = waqf_events(track, words)
    assert first == second


# ---------------------------------------------------------------------------
# Fixture-driven: real F0 verdict structure on a controlled 40 ms timeline.
# ---------------------------------------------------------------------------

# Synthetic frame budget, chosen so every constructed span clears the 300/700 ms rules
# regardless of a clip's real spacing: a plain word is 1 s of speech, a waqf gap 600 ms of
# silence, and a closure word carries 800 ms of speech on EACH side of its 400 ms silence hole
# so the closure is always a valid (flanked) pause the snap must reject on interior overlap.
_WORD_FRAMES = 25
_WAQF_GAP_FRAMES = 15
_CLOSURE_PAD_FRAMES = 20
_CLOSURE_FRAMES = 10


def _load_fixture_clips(path: Path) -> dict[str, list[dict]]:
    by_clip: dict[str, list[dict]] = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entry = json.loads(line)
            by_clip[entry["clip_id"]].append(entry)
    return by_clip


def _build_timeline(boundaries: list[dict]) -> tuple[np.ndarray, list[WordSpan], set[int], int]:
    """A clip's real verdict sequence laid on a controlled timeline; returns the reference input.

    Preserves the fixture's per-word verdicts and which word each mid-word closure sits inside,
    while giving every span generous, deterministic spacing so the 300/700 ms rules are never
    the thing under test — only the snap's waqf-vs-closure decision is.
    """
    closures_by_word = collections.defaultdict(int)
    for b in boundaries:
        if b["verdict"] == "mid_word_closure":
            closures_by_word[b["word_index"]] += 1
    edges = sorted(
        (b for b in boundaries if b["verdict"] in ("waqf", "wasl")),
        key=lambda b: b["word_index"],
    )
    silence: list[float] = [0.0] * _WORD_FRAMES  # leading speech so the first stop is flanked
    words: list[WordSpan] = []
    expected_waqf: set[int] = set()
    for b in edges:
        start = len(silence)
        if closures_by_word.get(b["word_index"]):
            silence.extend([0.0] * _CLOSURE_PAD_FRAMES + [1.0] * _CLOSURE_FRAMES + [0.0] * _CLOSURE_PAD_FRAMES)
        else:
            silence.extend([0.0] * _WORD_FRAMES)
        words.append(WordSpan(b["word_index"], start, len(silence)))
        if b["verdict"] == "waqf":
            silence.extend([1.0] * _WAQF_GAP_FRAMES)
            expected_waqf.add(b["word_index"])
    silence.extend([0.0] * _WORD_FRAMES)  # trailing speech
    n_closures = sum(closures_by_word.values())
    return np.array(silence, dtype=np.float32), words, expected_waqf, n_closures


@pytest.mark.parametrize("partition", ["calibration", "test"])
def test_reference_recovers_waqf_and_rejects_closures_on_fixtures(partition):
    path = _FIXTURE_DIR / f"waqf_events.{partition}.jsonl"
    by_clip = _load_fixture_clips(path)
    assert by_clip, f"no clips in {path}"
    total_closures = 0
    total_rejected = 0
    for clip_id, boundaries in by_clip.items():
        silence, words, expected_waqf, n_closures = _build_timeline(boundaries)
        result = waqf_events(silence, words)
        got = {e.word_index for e in result.waqf}
        assert got == expected_waqf, f"{clip_id}: waqf set {sorted(got)} != {sorted(expected_waqf)}"
        # Every mid-word closure must be rejected — never leak into the waqf events.
        assert len(result.rejected_closures) == n_closures, clip_id
        total_closures += n_closures
        total_rejected += len(result.rejected_closures)
    assert total_closures > 0  # the partitions do contain the qalqala/hamza rejection set
    assert total_rejected == total_closures

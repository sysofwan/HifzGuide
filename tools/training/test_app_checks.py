"""Muraja's check sequence (``training.decoding``, #125): the transcriber's schedule, the split,
the text rules and the flush gate, on synthetic durations and class-id rows. Torch-free.

The schedule's oracle below is ``RealtimeTranscriber.processCurrentBuffer``'s sliding-window
branch (``Models/RealtimeTranscriber+iOS.swift:568-667`` at Muraja ``99c326f``), transcribed
branch by branch and polled until idle each time audio arrives, with the speech detector
reporting speech throughout (the bypass the module docstring states).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from training.decoding import (
    APP_CHECKS,
    FLUSH,
    HOP_SAMPLES,
    PREVIEW,
    PREVIEW_STEP_SAMPLES,
    SAMPLE_RATE,
    WINDOW,
    WINDOW_SAMPLES,
    CheckSpan,
    app_check,
    app_checks,
    check_schedule,
    committed_text,
    device_output_steps,
    likely_hallucination,
    strip_padding_artifacts,
    tokens_to_phonemes,
)

BA, TA, MIM, NUN, MADD_YA = 2, 3, 24, 25, 30  # PhonemeVocabulary ids: ب ت م ن ۦ
FATHA = 32


def transcriber_oracle(num_samples: int, arrival: int = PREVIEW_STEP_SAMPLES) -> list[tuple]:
    """The inferences ``processCurrentBuffer`` runs as audio arrives ``arrival`` samples at a
    time (the rest last), polled until idle after each arrival, then the closing flush."""
    window_samples, hop_samples, min_new = 80_000, 16_000, 3_200
    state = {"buffer": 0, "window_start": 0, "last_preview": 0}
    runs: list[tuple] = []

    def process() -> bool:
        pending = state["buffer"] - state["window_start"]
        if pending / SAMPLE_RATE < 2.0:  # :569
            return False
        if pending >= window_samples:  # :575
            runs.append((WINDOW, state["window_start"], state["window_start"] + window_samples))
            state["window_start"] += hop_samples  # advanceWindow, :616
            state["last_preview"] = 0  # :618
            return True
        if pending - state["last_preview"] < min_new:  # :626-627
            return False
        runs.append((PREVIEW, state["window_start"], state["buffer"]))  # getPendingSamples, :634
        state["last_preview"] = pending  # :662
        return True

    for arrived in [*range(arrival, num_samples, arrival), num_samples]:
        state["buffer"] = arrived
        while process():
            pass
    runs.append((FLUSH, state["window_start"], num_samples))  # silenceFlush, :549, :676-700
    return runs


DURATIONS = [0.5, 1.91, 2.0, 2.3095, 4.91, 5.0, 5.1, 6.0, 9.99, 17.37, 56.099]


@pytest.mark.parametrize("seconds", DURATIONS)
def test_the_schedule_is_the_transcribers(seconds):
    num_samples = round(seconds * SAMPLE_RATE)
    schedule = [(s.kind, s.start, s.end) for s in check_schedule(num_samples)]
    assert schedule == transcriber_oracle(num_samples)


@pytest.mark.parametrize("seconds", DURATIONS)
def test_audio_arriving_in_smaller_steps_changes_nothing_but_the_preview_grid(seconds):
    """Windows and the flush do not depend on how finely audio arrives; previews only move
    within their 200 ms throttle."""
    num_samples = round(seconds * SAMPLE_RATE)
    coarse = transcriber_oracle(num_samples)
    fine = transcriber_oracle(num_samples, arrival=160)
    assert [r for r in fine if r[0] != PREVIEW] == [r for r in coarse if r[0] != PREVIEW]


@pytest.mark.parametrize(
    ("seconds", "previews", "windows"),
    [(1.91, 0, 0), (2.0, 1, 0), (4.91, 15, 0), (5.0, 16, 1), (6.0, 21, 2), (8.0, 31, 4)],
)
def test_window_and_preview_counts(seconds, previews, windows):
    """A preview every 200 ms from 2 s on; a window at 5 s and every second after; nothing is
    rounded to 200 ms (1.91 s has no preview, 4.91 s no window)."""
    kinds = [s.kind for s in check_schedule(round(seconds * SAMPLE_RATE))]
    assert (kinds.count(PREVIEW), kinds.count(WINDOW), kinds.count(FLUSH)) == (previews, windows, 1)
    assert kinds[-1] == FLUSH


@pytest.mark.parametrize("seconds", DURATIONS)
def test_every_span_is_audio_the_app_decodes(seconds):
    num_samples = round(seconds * SAMPLE_RATE)
    spans = check_schedule(num_samples)
    for previous, span in zip([None, *spans], spans):
        assert 0 <= span.start < span.end <= num_samples
        assert span.end - span.start <= WINDOW_SAMPLES
        if span.kind == WINDOW:
            assert span.start % HOP_SAMPLES == 0 and span.end - span.start == WINDOW_SAMPLES
        if span.kind == PREVIEW:
            assert span.end % PREVIEW_STEP_SAMPLES == 0 and span.end - span.start >= 2 * SAMPLE_RATE
        if previous is not None and previous.kind == WINDOW:
            # the throttle is reset, so a preview of what is left follows at once
            assert (span.kind, span.start, span.end) == (PREVIEW, previous.start + HOP_SAMPLES, previous.end)
    windows = [s for s in spans if s.kind == WINDOW]
    assert spans[-1].start == (windows[-1].start + HOP_SAMPLES if windows else 0)


# --- what the app decodes -------------------------------------------------------------------


def frames(num_samples: int) -> int:
    return (num_samples - 400) // 160 + 1


@pytest.mark.parametrize(
    ("num_samples", "steps"),
    [(WINDOW_SAMPLES, 125), (2 * SAMPLE_RATE, 50), (400, 1), (4 * SAMPLE_RATE, 100)],
)
def test_only_the_timesteps_real_audio_reaches_are_decoded(num_samples, steps):
    pairs = -(-frames(num_samples) // 2)
    assert device_output_steps(pairs) == steps
    assert device_output_steps(0) == 1 and device_output_steps(400) == 125


def row(*runs: tuple[int, int, int], length: int = 125) -> np.ndarray:
    """Class ids from ``(token, start, end_inclusive)`` runs; blank elsewhere."""
    ids = np.zeros(length, dtype=np.int64)
    for token, start, end in runs:
        ids[start : end + 1] = token
    return ids


def test_a_window_splits_at_the_midpoint_before_timestep_25():
    ids = row((BA, 24, 25), (TA, 25, 25), (MIM, 100, 101))  # midpoints 24.5, 25, 100.5
    check = app_check(CheckSpan(WINDOW, 0, WINDOW_SAMPLES), ids)
    assert check == {"hop": "ب", "overlap": "تم", "flush": False}


def test_a_short_row_splits_at_its_own_end():
    """``splitPoint = min(confirmTimeSteps, realOutputSteps)``."""
    ids = row((BA, 18, 19), length=20)
    assert app_check(CheckSpan(WINDOW, 0, WINDOW_SAMPLES), ids)["hop"] == "ب"


def test_hop_and_overlap_are_stripped_each_on_its_own():
    ids = row((BA, 2, 3), (MADD_YA, 10, 11), (MADD_YA, 40, 41), (TA, 60, 61))
    check = app_check(CheckSpan(WINDOW, 0, WINDOW_SAMPLES), ids)
    assert check == {"hop": "ب", "overlap": "ت", "flush": False}


def test_an_empty_window_or_preview_is_not_sent_but_the_flush_is():
    blank = row()
    assert app_check(CheckSpan(WINDOW, 0, WINDOW_SAMPLES), blank) is None
    assert app_check(CheckSpan(PREVIEW, 0, 2 * SAMPLE_RATE), blank[:50]) is None
    assert app_check(CheckSpan(FLUSH, 0, 2 * SAMPLE_RATE), blank[:50]) == {
        "hop": "", "overlap": "", "flush": True,
    }


def test_a_preview_is_all_of_its_audio_as_overlap():
    ids = row((BA, 2, 3), (TA, 40, 41), length=50)
    assert app_check(CheckSpan(PREVIEW, 0, 2 * SAMPLE_RATE), ids) == {
        "hop": "", "overlap": "بت", "flush": False,
    }


def test_the_flush_confirms_its_audio_unless_it_looks_hallucinated():
    speech = row((BA, 0, 3), (TA, 5, 8), (MIM, 10, 14), (NUN, 16, 19), length=25)
    assert app_check(CheckSpan(FLUSH, 0, 1), speech) == {"hop": "بتمن", "overlap": "", "flush": True}
    # 3 non-blank steps of 40: blank ratio 0.925 >= 0.90
    sparse = row((BA, 0, 0), (TA, 10, 10), (MIM, 20, 20), length=40)
    assert app_check(CheckSpan(FLUSH, 0, 1), sparse) == {"hop": "", "overlap": "", "flush": True}


@pytest.mark.parametrize(
    ("runs", "length", "expected"),
    [
        ((), 25, True),  # no non-blank step
        (((BA, 0, 0), (TA, 9, 9)), 100, False),  # fewer than 3 non-blank steps
        (((BA, 0, 2),), 30, True),  # blank ratio exactly 0.90, compared in float32
        (((BA, 0, 3),), 30, True),  # blank 0.867 >= 0.75 and one token throughout
        (((BA, 0, 3), (TA, 10, 13)), 32, True),  # blank exactly 0.75, dominant 0.5 >= 0.35
        (((BA, 0, 3), (TA, 10, 13), (MIM, 20, 23)), 40, False),  # blank 0.70 < 0.75
        (tuple((token, 4 * token, 4 * token) for token in range(1, 9)), 40, False),  # dominant 0.125
    ],
)
def test_the_hallucination_gate_is_swifts(runs, length, expected):
    """``CTCDecoder.Stats.isLikelyHallucination`` (``Models/CTCDecoder+iOS.swift:44-49``)."""
    assert likely_hallucination(row(*runs, length=length)) is expected


@pytest.mark.parametrize(
    ("text", "stripped"),
    [
        ("ۦبَ", "بَ"),  # a leading madd ya
        ("مۦبَ", "بَ"),  # a mim followed by a madd ya, then the madd ya
        ("نۦۦبَ", "بَ"),
        ("م", ""),  # a lone mim
        ("مبَ", "مبَ"),  # a mim followed by a letter stays
        ("مَبَ", "مَبَ"),  # a mim with a haraka is another Character
        ("ۦَبَ", "ۦَبَ"),  # so is a madd ya with one
        ("بَۦ", "بَ"),  # a trailing madd ya
        ("بَۦم", "بَ"),  # a trailing mim preceded by a madd ya, then the madd ya
        ("بَم", "بَم"),  # a trailing mim after a letter stays
        ("بَۦن۪", "بَۦن۪"),  # a nun with a sukun is another Character
        ("", ""),
    ],
)
def test_padding_artifacts_are_stripped_as_swift_compares_characters(text, stripped):
    assert strip_padding_artifacts(text) == stripped


def test_the_committed_transcript_is_every_hop_the_flushs_included():
    checks = [
        {"hop": "", "overlap": "بَ", "flush": False},
        {"hop": "بَ", "overlap": "تَ", "flush": False},
        {"hop": "تَ", "overlap": "", "flush": True},
    ]
    assert committed_text(checks) == "بَتَ"


def test_rows_must_match_the_schedule():
    with pytest.raises(ValueError, match="scheduled"):
        app_checks(check_schedule(3 * SAMPLE_RATE), [row()])


def test_a_whole_item_from_synthetic_rows():
    """A 6 s item: previews from 2 s, windows at 5 s and 6 s. Every row carries two tokens, so
    the 4 s flush (100 timesteps, 96% blank) is gated as a hallucination and confirms nothing."""
    spans = check_schedule(6 * SAMPLE_RATE)
    rows = []
    for span in spans:
        steps = device_output_steps(-(-frames(span.end - span.start) // 2))
        rows.append(row((BA, 2, 3), (TA, steps - 3, steps - 2), length=steps))
    checks = app_checks(spans, rows)
    assert len(checks) == len(spans) and checks[-1]["flush"]
    windows = [c for c in checks if c["hop"] and not c["flush"]]
    assert windows == [{"hop": "ب", "overlap": "ت", "flush": False}] * 2
    assert checks[-1] == {"hop": "", "overlap": "", "flush": True}
    assert committed_text(checks) == tokens_to_phonemes([BA, BA])
    assert APP_CHECKS == "muraja-app-checks-v1"


def test_the_check_sequence_needs_no_torch():
    code = (
        "import sys; import training.decoding as d; d.check_schedule(160000); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)

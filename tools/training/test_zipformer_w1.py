"""Tests for the zipformer's warm-started per-check arm ``checks-w1`` and the per-metric arm map
(#245, architect r2 + principal r1); torch-free, no ONNX file, no download, no GPU.

Muraja runs one inference per check (every window, every preview, the stop), so a word gets
several independent looks. Zipformer's single stream (``checks``, the locked tests in
``test_zipformer_decode``) gives one look. ``checks-w1`` gives it one fresh inference per check
of :func:`training.decoding.check_schedule`:

* the inference is over ``[max(0, start - W), end)`` of the item, ``W = 1 s`` (the transcriber's
  hop, fixed before any verdict set is decoded), followed by the card's 1.05 s of silence, from a
  **zero cache** that is never carried from one check to the next;
* the warm-up second is not scored: an emission whose midpoint is before the check's ``start``
  is dropped, and **everything else the inference emitted is kept, the flush included**
  (principal r1: a ``[start, end)`` cut would drop the delayed last word the flush forces out,
  which ours never drops). That cut stays the rule of the single-stream ``checks`` arm only;
* the kept emissions are split into hop, overlap, preview and stop as the transcriber does
  (a window's hop is its first second; a preview or the stop sends everything kept);
* ``checks-w1-windows`` (B') runs the fresh inferences for the windows only and takes previews
  and the stop from the one continuous stream. It is report-only.

The verdict reads each metric from one arm per model (:func:`metric_arm`): zipformer's app-outcome
metrics from ``checks-w1``, its whole-item metrics from ``spans``; ours from ``checks`` and
``stream_b0`` as today. The single-stream ``checks`` and ``checks-w1-windows`` can never be a
zipformer verdict arm. The two new arms live in their own stores, since they are decoded only on
the sets the Muraja harness grades; a zipformer line from ``ZIPFORMER_STORES`` is unchanged.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from training import decoding as dc
from training.external_decode import Recording, decode_recordings
from training.zipformer_decode import (
    BLANK_ID,
    CHECKS_W1,
    CHECKS_W1_WINDOWS,
    METRICS,
    REPORT_ONLY_ARMS,
    WARMUP_SAMPLES,
    ZIPFORMER_PROTOCOLS,
    ZIPFORMER_STORES,
    ZIPFORMER_W1_STORES,
    Emission,
    ZipformerDecoder,
    arms_decoded_samples,
    decoded_audio_ratio,
    metric_arm,
    scoring_arm,
    w1_check,
    warmup_start,
    zipformer_fingerprint,
)

STEP = 640  # one 40 ms output step at 16 kHz
BA, TA, THA = "ب", "ت", "ث"


def _at(text: str, midpoint: float) -> Emission:
    """An emission whose first and last frame average to ``midpoint`` (a whole or half step)."""
    low = int(midpoint)
    return Emission(text, low, low + 1) if midpoint % 1 else Emission(text, low, low)


def _span(kind: str, start_step: int, end_step: int) -> dc.CheckSpan:
    return dc.CheckSpan(kind, start_step * STEP, end_step * STEP)


# --- the warm-up ------------------------------------------------------------------------


def test_the_warm_up_is_one_hop():
    assert WARMUP_SAMPLES == dc.HOP_SAMPLES == 16_000


@pytest.mark.parametrize(("start", "begin"), [(0, 0), (16_000, 0), (32_000, 16_000),
                                              (48_000, 32_000), (160_000, 144_000)])
def test_the_inference_starts_one_second_before_the_check_clipped_at_zero(start, begin):
    assert warmup_start(dc.CheckSpan(dc.PREVIEW, start, start + 64_000)) == begin


# --- what a check keeps of its inference ------------------------------------------------


def test_emissions_before_the_checks_start_are_the_warm_up_and_are_dropped():
    window = _span(dc.WINDOW, 50, 175)  # a window keeps from step 50; its hop is steps [50, 75)
    emissions = [_at(BA, 30), _at(TA, 49.5), _at("ء", 50.0), _at(THA, 60), _at("ن", 75.0)]
    # 30 and 49.5 are warm-up; 50.0 is the start itself and stays; 75.0 starts the overlap
    assert w1_check(window, emissions) == {"hop": "ء" + THA, "overlap": "ن", "flush": False}


def test_an_emission_after_the_checks_end_stays_with_that_check():
    # the contrast with the single-stream rule: a fresh inference's delayed last word is
    # exactly what its flush forces out, so a [start, end) cut would truncate every check
    window = _span(dc.WINDOW, 50, 175)
    flush = [_at(BA, 160), _at(TA, 175.0), _at(THA, 180)]  # 175.0 is the end; 180 is flush
    assert w1_check(window, flush) == {"hop": "", "overlap": BA + TA + THA, "flush": False}
    preview = _span(dc.PREVIEW, 0, 50)
    emitted = [_at(BA, 20), _at(TA, 49), _at(THA, 50.0), _at("ن", 53)]
    assert w1_check(preview, emitted) == {"hop": "", "overlap": BA + TA + THA + "ن",
                                          "flush": False}


def test_nothing_is_warm_up_at_the_first_checks_because_the_inference_starts_at_the_start():
    first = _span(dc.WINDOW, 0, 125)
    assert w1_check(first, [_at(BA, 0.0), _at(TA, 24.5), _at(THA, 25.0)]) == {
        "hop": BA + TA, "overlap": THA, "flush": False}


def test_a_preview_sends_everything_it_kept_as_overlap():
    preview = _span(dc.PREVIEW, 50, 150)
    assert w1_check(preview, [_at(BA, 10), _at(TA, 50.0), _at(THA, 90)]) == {
        "hop": "", "overlap": TA + THA, "flush": False}


def test_the_stop_sends_everything_it_kept_as_its_hop():
    stop = _span(dc.STOP, 50, 150)
    assert w1_check(stop, [_at(BA, 10), _at(TA, 50.0), _at(THA, 155)]) == {
        "hop": TA + THA, "overlap": "", "flush": False, "stop": True}


def test_a_stop_with_only_warm_up_is_still_sent_empty():
    assert w1_check(_span(dc.STOP, 50, 75), [_at(BA, 30)]) == {
        "hop": "", "overlap": "", "flush": False, "stop": True}


@pytest.mark.parametrize("kind", [dc.WINDOW, dc.PREVIEW])
def test_a_window_or_preview_with_nothing_kept_sends_nothing(kind):
    assert w1_check(_span(kind, 50, 175), [_at(BA, 30)]) is None


def test_neither_of_our_window_guards_is_applied():
    stop = _span(dc.STOP, 0, 75)
    sparse = [_at(BA, step) for step in (10, 30, 50, 70)]  # likely_hallucination's shape
    assert w1_check(stop, sparse)["hop"] == BA * 4
    assert w1_check(stop, [_at("م", 4), _at("ۦ", 6), _at(BA, 30)])["hop"] == "مۦ" + BA


# --- the decoder: one fresh inference per check -----------------------------------------

#: Absolute step -> the token the fake model emits where it hears that step: ba, ta, tha, ba.
MARKS = {10: 3, 30: 4, 60: 5, 140: 3}
TOKENS = {3: BA, 4: TA, 5: THA, BLANK_ID: "<blank>"}
#: 6 s of audio, 150 steps; sample i holds ``i // 640 + 1``, so a frame's value names its step.
AUDIO = np.repeat(np.arange(1, 151, dtype=np.float32), STEP)


class _StepFbank:
    """One frame per 10 ms whose value is the mean of its 160 samples, so a frame carries the
    absolute step it came from (zero in the appended silence)."""

    def __init__(self) -> None:
        self.waves: list[np.ndarray] = []

    def __call__(self, waveform, **options):
        wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
        self.waves.append(wave)
        frames = len(wave) // 160
        values = wave[: frames * 160].reshape(frames, 160).mean(axis=1)
        return np.repeat(values[:, None], 80, axis=1).astype(np.float32)


class _MarkSession:
    """A model that emits ``MARKS``' token at the output frame where it hears that step (only
    audio that reaches it), plus a cold-start artifact (ta at the first frame of a call that
    starts from a zero cache), the kind of emission a warm-up second exists to hide."""

    def __init__(self) -> None:
        self.caches: list[int] = []

    def get_inputs(self):
        return [SimpleNamespace(name="x", shape=[1, 61, 80], type="tensor(float)"),
                SimpleNamespace(name="cached_len_0", shape=[2, 1], type="tensor(int64)"),
                SimpleNamespace(name="cached_val_0", shape=[1, 4], type="tensor(float)")]

    def run(self, output_names, feed):
        cache = int(feed["cached_len_0"].flat[0])
        self.caches.append(cache)
        values = feed["x"][0, :, 0]
        log_probs = np.full((1, 12, 251), -10.0, dtype=np.float32)
        log_probs[0, :, BLANK_ID] = 0.0
        for frame in range(12):
            step = int(round(float(values[4 * frame]))) - 1
            if step in MARKS:
                log_probs[0, frame, MARKS[step]] = 1.0
        if cache == 0:
            log_probs[0, 0, 4] = 1.0
        return [log_probs, feed["cached_len_0"] + 1, feed["cached_val_0"] + 1]

    def inferences(self) -> list[list[int]]:
        """The cache values each fresh inference saw, call by call."""
        runs: list[list[int]] = []
        for cache in self.caches:
            if cache == 0:
                runs.append([])
            runs[-1].append(cache)
        return runs


def _decoder() -> tuple[ZipformerDecoder, _MarkSession, _StepFbank]:
    session, fbank = _MarkSession(), _StepFbank()
    return ZipformerDecoder(session, TOKENS, fbank, scale=1.0), session, fbank


def _spans() -> list[dc.CheckSpan]:
    return dc.check_schedule(len(AUDIO))


def test_every_check_of_the_schedule_is_one_fresh_inference_and_checks_are_what_they_kept():
    decoder, session, fbank = _decoder()
    checks = decoder.checks_w1(AUDIO)
    spans = _spans()
    assert len(spans) == 24
    # a fresh zero cache per check, carried only within it
    runs = session.inferences()
    assert len(runs) == len(spans)
    assert all(run == list(range(len(run))) for run in runs)
    # the cold-start artifact (ta at an inference's first step) is warm-up for every check
    # that starts after it; it is kept only where the inference starts at the check's start
    assert checks == (
        [{"hop": "", "overlap": TA + BA + TA, "flush": False}] * 3         # previews to 60
        + [{"hop": "", "overlap": TA + BA + TA + THA, "flush": False}] * 12  # previews 65..120
        + [{"hop": TA + BA, "overlap": TA + THA, "flush": False}]          # window 0..125
        + [{"hop": "", "overlap": TA + THA, "flush": False}] * 4           # previews 25..
        + [{"hop": "", "overlap": TA + THA + BA, "flush": False}]          # ..145
        + [{"hop": TA, "overlap": THA + BA, "flush": False}]               # window 25..150
        + [{"hop": "", "overlap": THA + BA, "flush": False}]               # preview 50..150
        + [{"hop": THA + BA, "overlap": "", "flush": False, "stop": True}]  # the stop
    )


def test_each_inference_covers_one_second_before_the_check_then_the_flush():
    decoder, _, fbank = _decoder()
    decoder.checks_w1(AUDIO)
    assert len(fbank.waves) == len(_spans())  # one front end pass per inference, none shared
    for span, wave in zip(_spans(), fbank.waves, strict=True):
        begin = max(0, span.start - 16_000)
        assert len(wave) == span.end - begin + 16_800
        np.testing.assert_array_equal(wave[: span.end - begin], AUDIO[begin: span.end])
        assert not wave[span.end - begin:].any()


def test_the_decoder_resets_between_items_as_well():
    decoder, session, _ = _decoder()
    decoder.checks_w1(AUDIO)
    decoder.checks_w1(AUDIO)
    assert all(run[0] == 0 for run in session.inferences())
    assert len(session.inferences()) == 2 * len(_spans())


# --- B': fresh inferences for the windows, the continuous stream for the rest -------------


def test_windows_only_runs_the_windows_fresh_and_the_rest_from_the_one_stream():
    decoder, session, _ = _decoder()
    windows_only = decoder.checks_w1_windows(AUDIO)
    windows = sum(span.kind == dc.WINDOW for span in _spans())
    assert len(session.inferences()) == windows + 1  # each window fresh, plus one stream
    full, single = _decoder()[0].checks_w1(AUDIO), _decoder()[0].checks(AUDIO)
    assert len(windows_only) == len(full) == len(single) == len(_spans())
    for span, mine, fresh, stream in zip(_spans(), windows_only, full, single, strict=True):
        assert mine == (fresh if span.kind == dc.WINDOW else stream)


# --- the stores: new arms, own files, no stream_b0 -------------------------------------


def test_the_new_arms_have_their_own_stores_and_the_old_ones_are_unchanged():
    assert set().union(*(s.protocols for s in ZIPFORMER_STORES.values())) == {"spans", "checks"}
    assert set().union(*(s.protocols for s in ZIPFORMER_W1_STORES.values())) == {
        CHECKS_W1, CHECKS_W1_WINDOWS} == {"checks-w1", "checks-w1-windows"}


def test_a_w1_line_carries_the_two_new_arms_and_no_stream_b0(tmp_path):
    w1, windows = [{"hop": BA, "overlap": "", "flush": False, "stop": True}], [{"hop": TA}]
    decoder = SimpleNamespace(checks_w1=lambda span: w1, checks_w1_windows=lambda span: windows)
    record = {"key": "a#0", "start_sample": 0, "end_sample": len(AUDIO)}
    path = tmp_path / "w1.jsonl"
    decode_recordings([Recording(("a#0",), lambda: (AUDIO, [record]))], {"zipformer": decoder},
                      {"zipformer": path}, "test", ZIPFORMER_W1_STORES)
    (line,) = [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines()]
    assert line["checks-w1"] == w1 and line["checks-w1-windows"] == windows
    assert not {"stream_b0", "spans", "checks"} & set(line)


def test_the_w1_modes_name_the_warm_up_and_the_flush_and_are_all_different():
    modes = [ZIPFORMER_PROTOCOLS[arm] for arm in ("spans", "checks", CHECKS_W1, CHECKS_W1_WINDOWS)]
    assert len(set(modes)) == 4
    for mode in modes[2:]:
        assert "w1000ms" in mode and "flush1050ms" in mode
    fingerprint = zipformer_fingerprint("z", CHECKS_W1)
    assert fingerprint.mode == ZIPFORMER_PROTOCOLS[CHECKS_W1]
    assert (fingerprint.weights_dtype, fingerprint.device_type, fingerprint.autocast) == (
        "fp32", "cpu", False)


# --- the per-metric arm map -------------------------------------------------------------

APP_METRICS = ("qdat_false_flags", "ship_a_catches")
WHOLE_ITEM_METRICS = ("qdat_per_class", "qdat_bench", "edits_heard", "ship_a_whole_item",
                      "benchmark_per_class")
STUDENTS = ("base", "c95", "r2_candidate_ema")


def test_the_metrics_are_the_app_outcome_ones_and_the_whole_item_ones():
    assert set(METRICS) == set(APP_METRICS) | set(WHOLE_ITEM_METRICS)


@pytest.mark.parametrize("metric", APP_METRICS)
def test_zipformers_app_outcome_metrics_read_the_warm_started_arm(metric):
    assert metric_arm("zipformer", metric) == "checks-w1"


@pytest.mark.parametrize("metric", WHOLE_ITEM_METRICS)
def test_zipformers_whole_item_metrics_read_spans(metric):
    assert metric_arm("zipformer", metric) == "spans"


@pytest.mark.parametrize("model", STUDENTS)
def test_ours_read_checks_for_app_outcomes_and_stream_b0_for_whole_items(model):
    assert {metric: metric_arm(model, metric) for metric in METRICS} == (
        {m: "checks" for m in APP_METRICS} | {m: "stream_b0" for m in WHOLE_ITEM_METRICS})


@pytest.mark.parametrize("model", [*STUDENTS, "zipformer"])
def test_the_whole_item_metrics_agree_with_the_whole_item_scoring_arm(model):
    assert {metric_arm(model, metric) for metric in WHOLE_ITEM_METRICS} == {scoring_arm(model)}


@pytest.mark.parametrize("metric", METRICS)
def test_a_report_only_arm_is_never_zipformers_verdict_arm(metric):
    assert REPORT_ONLY_ARMS == ("checks", CHECKS_W1_WINDOWS)
    assert metric_arm("zipformer", metric) not in REPORT_ONLY_ARMS


def test_zipformer_is_never_read_from_stream_b0_and_ours_never_from_zipformers_arms():
    arms = {(model, metric): metric_arm(model, metric)
            for model in (*STUDENTS, "zipformer") for metric in METRICS}
    assert all(arm != "stream_b0" for (model, _), arm in arms.items() if model == "zipformer")
    assert all(arm in {"checks", "stream_b0"} for (model, _), arm in arms.items()
               if model != "zipformer")


def test_an_unknown_metric_or_model_is_refused():
    with pytest.raises(ValueError):
        metric_arm("zipformer", "ship_b_catches")
    with pytest.raises(ValueError):
        metric_arm("h448_unknown", "qdat_false_flags")


# --- the compute table ------------------------------------------------------------------


def test_decoded_audio_seconds_per_audio_second_ours_and_checks_w1():
    # a 60 s item through check_schedule: ours 25.8 s of audio per second, checks-w1 37.4
    assert decoded_audio_ratio("ours", 960_000) == pytest.approx(25.8, abs=0.05)
    assert decoded_audio_ratio(CHECKS_W1, 960_000) == pytest.approx(37.4, abs=0.05)


def test_an_arm_with_no_decoded_audio_figure_is_refused():
    with pytest.raises(ValueError):
        decoded_audio_ratio("stream_b0", 960_000)


def test_one_item_runs_one_stream_whichever_single_stream_arm_is_asked_for():
    assert arms_decoded_samples(("spans", "checks"), 960_000) == 960_000
    assert arms_decoded_samples(("checks",), 960_000) == 960_000


def test_checks_w1_re_decodes_every_check_from_its_own_zero_cache():
    # a 60 s item's schedule, every check decoded from its own zero cache: 37.44 s per second
    assert arms_decoded_samples((CHECKS_W1,), 960_000) == 35_942_400


def test_the_windows_only_arm_adds_one_stream_and_reuses_checks_w1_s_windows():
    both = arms_decoded_samples((CHECKS_W1, CHECKS_W1_WINDOWS), 960_000)
    alone = arms_decoded_samples((CHECKS_W1,), 960_000)
    windows_alone = arms_decoded_samples((CHECKS_W1_WINDOWS,), 960_000)
    fresh_windows = decoded_audio_ratio(CHECKS_W1_WINDOWS, 960_000) * 960_000
    assert both == alone + 960_000  # B' only adds the one stream to checks-w1
    assert windows_alone == pytest.approx(960_000 + fresh_windows)
    assert windows_alone < alone  # its previews and stop are cut from the stream


def test_the_batch_arithmetic_refuses_an_arm_it_cannot_count_and_an_empty_item():
    with pytest.raises(ValueError):
        arms_decoded_samples(("stream_b0",), 960_000)
    with pytest.raises(ValueError):
        arms_decoded_samples((CHECKS_W1,), 0)

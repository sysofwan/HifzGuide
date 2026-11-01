"""Tests for the zipformer decoder (#245): detokenisation, the ``checks`` arm, the front end, the
stream, the arm map and the fingerprints. Torch-free, no ONNX file, no download, no GPU.

Contract these pin (``training.zipformer_decode``):

* **Detokenisation** is by ``tokens.txt`` strings: blank is id 250, and the ids are NOT ours
  (our blank is 0 and the characters are in another order), so no id ever goes through
  :func:`training.decoding.tokens_to_phonemes`.
* **The stream** is one continuous run per item: the fbank is computed once over the item plus
  1.05 s of silence, the model's cache state is carried from call to call and starts from zero
  per *item* (never per 5 s window), and the log-prob frames of all calls form one 40 ms timeline.
* **The ``checks`` arm** slices that one stream with the app's check schedule
  (:func:`training.decoding.check_schedule`): a check takes the emissions whose midpoint (the
  mean of the unit's first and last frame, :attr:`Emission.midpoint`) lies in ``[start, end)`` of
  its window, as :func:`training.decoding.app_check` does with its ``low <= midpoint < high``.
  Neither of our window guards (``likely_hallucination``, ``strip_padding_artifacts``) applies.
* **Arms**: a zipformer record carries ``spans`` and ``checks``, never a ``stream_b0`` (#245's
  per-model arm map: the students are scored from ``stream_b0``, zipformer from ``spans``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR
from training import decoding as dc
from training.external_decode import Recording, decode_recordings
from training.zipformer_decode import (
    BLANK_ID,
    FBANK_OPTIONS,
    TAIL_SAMPLES,
    ZIPFORMER_DIR,
    ZIPFORMER_STORES,
    Emission,
    ZipformerDecoder,
    detokenise,
    featurize,
    read_tokens,
    scan_units,
    scoring_arm,
    stream_checks,
    whole_item_text,
    zipformer_fingerprint,
)

TOOLS = Path(__file__).parent.parent
STEP_SAMPLES = 640  # one 40 ms output step at 16 kHz
BA, TA, THA, JA, HAMZA = "بَ", "تُ", "ثِ", "جَ", "ءِ"
#: A small table in tokens.txt's shape: ids are not ours, 42/45 are merged units, 250 is blank.
TOKENS = {0: "ؙ", 1: "ء", 2: "ا", 3: "ب", 42: "ءَ", 45: "اا", 46: "بَ", 250: "<blank>"}


def _checks(emissions: list[Emission], num_samples: int) -> list[dict]:
    return stream_checks(emissions, num_samples)


def _windows(checks: list[dict]) -> list[dict]:
    return [c for c in checks if c["hop"] and not c["flush"] and "stop" not in c]


def _previews(checks: list[dict]) -> list[str]:
    return [c["overlap"] for c in checks if not c["hop"] and not c["flush"] and "stop" not in c]


# --- detokenisation: tokens.txt strings, blank 250 --------------------------------------


def test_blank_is_250():
    assert BLANK_ID == 250


def test_a_run_of_one_id_is_one_emission_and_a_blank_separates_repeats():
    frames = [250, 3, 3, 250, 250, 3, 45, 45, 250]
    assert scan_units(frames, TOKENS) == [
        Emission("ب", 1, 2), Emission("ب", 5, 5), Emission("اا", 6, 7)]
    assert detokenise(frames, TOKENS) == "ب" + "ب" + "اا"


def test_a_merged_unit_is_one_emission_of_its_whole_string():
    assert detokenise([42, 45, 46], TOKENS) == "ءَ" + "اا" + "بَ"


def test_blank_never_reaches_the_text():
    assert detokenise([250, 250, 250], TOKENS) == ""
    assert scan_units([], TOKENS) == []


def test_an_id_the_table_does_not_hold_is_refused_not_dropped():
    with pytest.raises(ValueError):
        detokenise([3, 99], TOKENS)


def test_an_emissions_midpoint_is_the_mean_of_its_first_and_last_frame():
    assert Emission("ب", 24, 25).midpoint == 24.5
    assert Emission("ب", 125, 125).midpoint == 125.0


def test_ids_are_read_from_tokens_txt_not_from_our_vocabulary(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("<blank> 250\nؙ 0\nء 1\nا 2\n", encoding="utf-8")
    assert read_tokens(path) == {250: "<blank>", 0: "ؙ", 1: "ء", 2: "ا"}


def test_tokens_without_blank_at_250_are_refused(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("<blank> 0\nء 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_tokens(path)


def test_the_committed_tokens_cover_our_alphabet_in_another_order():
    tokens = read_tokens(ZIPFORMER_DIR / "tokens.txt")
    assert len(tokens) == 251 and tokens[250] == "<blank>"
    ours = PHONEME_ID_TO_CHAR[1:]  # ours has [PAD] = blank at 0
    assert {tokens[i] for i in range(42)} == set(ours)
    assert [tokens[i] for i in range(42)] != list(ours)
    # id 0 is a real unit here and the blank of ours; id 2 is alif here and ba' of ours
    assert detokenise([0, 2], tokens) == "ؙا"
    assert dc.tokens_to_phonemes([0, 2]) == "ب"
    assert detokenise([2], tokens) != dc.tokens_to_phonemes([2])


def test_every_merged_unit_is_a_string_over_the_42_characters():
    tokens = read_tokens(ZIPFORMER_DIR / "tokens.txt")
    alphabet = {tokens[i] for i in range(42)}
    assert all(set(tokens[i]) <= alphabet for i in range(42, 250))
    assert all(len(tokens[i]) > 1 for i in range(42, 250))


def test_the_license_is_committed_beside_the_decodes():
    text = (ZIPFORMER_DIR / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("Quran-Lab No-Profit License") and "NPL-1.2" in text


# --- the checks arm: slices of one stream -----------------------------------------------


def test_no_emissions_still_send_the_stop_and_nothing_else():
    assert _checks([], 48_000) == [{"hop": "", "overlap": "", "flush": False, "stop": True}]


def test_previews_before_the_first_window_carry_everything_emitted_so_far():
    # 3 s item: previews end at steps 50, 55, ... 75; b (midpoint 50) is not before step 50.
    emissions = [Emission(BA, 10, 10), Emission(TA, 50, 50), Emission(THA, 70, 70),
                 Emission(JA, 80, 80)]  # step 80 is in the 1.05 s flush, after the audio
    checks = _checks(emissions, 48_000)
    assert _previews(checks) == [BA, BA + TA, BA + TA, BA + TA, BA + TA, BA + TA + THA]
    assert checks[-1] == {"hop": BA + TA + THA + JA, "overlap": "", "flush": False, "stop": True}


#: A 6 s item (windows end at steps 125 and 150, the audio at 150), emissions placed on the
#: half-open boundaries: 24.5 and 25.0 around the hop/overlap split, 124.5 and 125.0 around the
#: first window's end, and one in the flush after the audio.
BOUNDARY = [Emission(BA, 24, 25), Emission(TA, 25, 25), Emission(THA, 124, 125),
            Emission(JA, 125, 125), Emission(HAMZA, 155, 155)]


def test_the_hop_is_the_windows_first_second_and_the_overlap_the_rest():
    first, second = _windows(_checks(BOUNDARY, 96_000))
    assert first == {"hop": BA, "overlap": TA + THA, "flush": False}  # 25.0 is overlap, not hop
    assert second == {"hop": TA, "overlap": THA + JA, "flush": False}  # abs [25, 50), [50, 150)


def test_an_emission_at_a_windows_end_goes_to_the_later_check():
    checks = _checks(BOUNDARY, 96_000)
    first_window = _windows(checks)[0]
    assert JA not in first_window["hop"] + first_window["overlap"]  # midpoint 125.0 == its end
    assert JA in _windows(checks)[1]["overlap"]
    previews = _previews(checks)
    # the preview over steps [25, 125) holds 124.5 but not 125.0; the next one holds both
    assert previews[15:17] == [TA + THA, TA + THA + JA]


def test_previews_over_the_whole_item_are_cut_by_their_own_window():
    previews = _previews(_checks(BOUNDARY, 96_000))
    assert previews == [BA + TA] * 15 + [TA + THA] + [TA + THA + JA] * 4 + [THA + JA]


def test_the_stop_takes_everything_from_its_start_including_the_flush():
    assert _checks(BOUNDARY, 96_000)[-1] == {
        "hop": THA + JA + HAMZA, "overlap": "", "flush": False, "stop": True}


@pytest.mark.parametrize("num_samples", [48_000, 96_000, 112_000, 200_000])
def test_every_emission_enters_the_committed_transcript_once(num_samples):
    texts = (BA, TA, THA)
    end = num_samples // STEP_SAMPLES + 20  # into the flush
    emissions = [Emission(texts[i % 3], step, step)
                 for i, step in enumerate(range(3, end, 7))]
    assert dc.committed_text(_checks(emissions, num_samples)) == "".join(
        e.text for e in emissions)


def test_the_schedule_is_the_transcribers_so_each_check_is_one_of_its_inferences():
    spans = dc.check_schedule(96_000)
    emissions = [Emission(BA, step, step) for step in range(2, 150, 3)]
    checks = _checks(emissions, 96_000)
    # nothing is left empty here, so every scheduled inference sends exactly one check
    assert len(checks) == len(spans)
    assert [c.get("stop", False) for c in checks] == [s.kind == dc.STOP for s in spans]


def test_neither_of_our_window_guards_is_applied():
    # strip_padding_artifacts would cut a leading mim + madd ya; likely_hallucination would blank
    # a stop that is four sparse repeats of one unit (blank ratio 0.95, dominant ratio 1.0).
    leading_mim = _checks(
        [Emission("م", 4, 4), Emission("ۦ", 6, 6), Emission(BA, 30, 30)], 48_000)
    assert leading_mim[-1]["hop"] == "مۦ" + BA
    assert _previews(leading_mim)[0] == "مۦ" + BA
    sparse = [Emission(BA, step, step) for step in (10, 30, 50, 70)]
    assert _checks(sparse, 48_000)[-1] == {
        "hop": BA * 4, "overlap": "", "flush": False, "stop": True}


# --- the front end: kaldi fbank, injected -----------------------------------------------


class _RecordingFbank:
    """Stands in for ``torchaudio.compliance.kaldi.fbank``: one 80-bin frame per 160 samples."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, dict]] = []

    def __call__(self, waveform, **options):
        wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
        self.calls.append((wave, options))
        return np.zeros((round(len(wave) / 160), 80), dtype=np.float32)


def _samples(seconds: float = 1.0) -> np.ndarray:
    return np.random.default_rng(3).uniform(-0.5, 0.5, int(seconds * 16_000)).astype(np.float32)


def test_the_front_end_settings_are_the_cards():
    assert {"num_mel_bins": 80, "sample_frequency": 16_000, "window_type": "povey",
            "dither": 0.0, "snip_edges": False, "high_freq": -400.0}.items() <= (
                FBANK_OPTIONS.items())


def test_featurize_passes_the_cards_options_to_the_injected_fbank():
    fbank = _RecordingFbank()
    featurize(_samples(), fbank, scale=1.0)
    (_, options), = fbank.calls
    assert FBANK_OPTIONS.items() <= options.items()


def test_the_item_gets_1_05_s_of_silence_after_it():
    assert TAIL_SAMPLES == 16_800
    fbank = _RecordingFbank()
    samples = _samples(2.0)
    features = featurize(samples, fbank, scale=1.0)
    (wave, _), = fbank.calls
    assert len(wave) == len(samples) + 16_800
    assert not wave[len(samples):].any()
    assert features.shape == (round(len(wave) / 160), 80) and features.dtype == np.float32


@pytest.mark.parametrize("scale", [1.0, 32_768.0])
def test_the_samples_are_scaled_before_the_fbank_sees_them(scale):
    fbank = _RecordingFbank()
    samples = _samples()
    featurize(samples, fbank, scale=scale)
    (wave, _), = fbank.calls
    np.testing.assert_allclose(wave[: len(samples)], samples * scale, rtol=1e-6)


def test_the_scale_has_no_default():
    with pytest.raises(TypeError):
        featurize(_samples(), _RecordingFbank())


def test_the_decoder_needs_neither_torch_nor_our_feature_extractor():
    code = ("import sys; import training.zipformer_decode; "
            "assert not {'torch', 'torchaudio', 'transformers'} & set(sys.modules), "
            "sorted({'torch', 'torchaudio', 'transformers'} & set(sys.modules))")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=TOOLS)


# --- the stream: one continuous run per item --------------------------------------------


class _FakeSession:
    """An icefall streaming export in miniature: ``x`` plus two cache inputs; the outputs are
    ``log_probs`` (1, 12, 251) and each cache's next value, in input order. Every call emits
    unit 3 at its first frame and counts in its cache, so the log shows what state each call saw."""

    def __init__(self) -> None:
        self.seen: list[tuple[tuple, int]] = []

    def get_inputs(self):
        return [SimpleNamespace(name="x", shape=[1, 61, 80], type="tensor(float)"),
                SimpleNamespace(name="cached_len_0", shape=[2, 1], type="tensor(int64)"),
                SimpleNamespace(name="cached_val_0", shape=[1, 4], type="tensor(float)")]

    def run(self, output_names, feed):
        assert set(feed) == {"x", "cached_len_0", "cached_val_0"}
        assert feed["x"].dtype == np.float32
        assert feed["cached_len_0"].dtype == np.int64 and feed["cached_val_0"].dtype == np.float32
        self.seen.append((feed["x"].shape, int(feed["cached_len_0"].flat[0])))
        log_probs = np.full((1, 12, 251), -10.0, dtype=np.float32)
        log_probs[0, :, BLANK_ID] = 0.0
        log_probs[0, 0, 3] = 1.0
        return [log_probs, feed["cached_len_0"] + 1, feed["cached_val_0"] + 1]


def _decoder(session: _FakeSession, fbank=None) -> ZipformerDecoder:
    return ZipformerDecoder(session, {3: "ب", BLANK_ID: "<blank>"},
                            fbank or _RecordingFbank(), scale=1.0)


def test_the_cache_is_carried_across_calls_so_no_window_restarts_from_zero():
    session, fbank = _FakeSession(), _RecordingFbank()
    _decoder(session, fbank).stream(_samples(3.0))
    calls = len(session.seen)
    assert calls >= 6  # 3 s is 75 output steps, 12 per call
    assert [shape for shape, _ in session.seen] == [(1, 61, 80)] * calls
    assert [state for _, state in session.seen] == list(range(calls))  # zero once, then carried
    assert len(fbank.calls) == 1  # one fbank over the item plus its tail, not one per call


def test_the_calls_log_probs_form_one_40_ms_timeline():
    session = _FakeSession()
    emissions = _decoder(session).stream(_samples(3.0))
    assert [e.start_step for e in emissions] == [12 * k for k in range(len(session.seen))]


def test_each_item_starts_from_a_zero_cache():
    session = _FakeSession()
    decoder = _decoder(session)
    decoder.stream(_samples(1.0))
    first_item_calls = len(session.seen)
    decoder.stream(_samples(1.0))
    assert session.seen[first_item_calls][1] == 0


def test_spans_is_the_whole_stream_and_checks_are_cut_from_it():
    session = _FakeSession()
    decoder = _decoder(session)
    samples = _samples(3.0)
    emissions = decoder.stream(samples)
    assert decoder.spans(samples) == "".join(e.text for e in emissions)
    assert decoder.checks(samples) == stream_checks(emissions, len(samples))


# --- the arm map: zipformer has spans and checks, never stream_b0 ------------------------


def test_zipformer_lines_carry_spans_and_checks_and_no_stream_b0(tmp_path):
    checks = [{"hop": BA, "overlap": "", "flush": False, "stop": True}]
    decoder = SimpleNamespace(spans=lambda span: BA, checks=lambda span: checks)
    samples = _samples(1.0)
    record = {"key": "a#0", "start_sample": 0, "end_sample": len(samples)}
    path = tmp_path / "zipformer.jsonl"
    decode_recordings([Recording(("a#0",), lambda: (samples, [record]))],
                      {"zipformer": decoder}, {"zipformer": path}, "test", ZIPFORMER_STORES)
    (line,) = [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines()]
    assert line["spans"] == BA and line["checks"] == checks
    assert "stream_b0" not in line
    assert set().union(*(store.protocols for store in ZIPFORMER_STORES.values())) == {
        "spans", "checks"}


def test_the_students_are_scored_from_stream_b0_and_zipformer_from_spans():
    assert {m: scoring_arm(m) for m in ("base", "c95", "r2_candidate_ema", "zipformer")} == {
        "base": "stream_b0", "c95": "stream_b0", "r2_candidate_ema": "stream_b0",
        "zipformer": "spans"}
    with pytest.raises(ValueError):
        scoring_arm("h448_unknown")


def test_a_lines_scored_text_is_its_models_arm():
    assert whole_item_text({"spans": "x", "stream_b0": "y"}, "r2_candidate_ema") == "y"
    assert whole_item_text({"spans": "x"}, "zipformer") == "x"


def test_a_zipformer_line_holding_a_stream_b0_is_refused():
    with pytest.raises(ValueError):
        whole_item_text({"spans": "x", "stream_b0": "y"}, "zipformer")


# --- identity: a fingerprint that cannot be compared with ours ---------------------------


def test_zipformer_fingerprints_are_cpu_fp32_and_one_per_arm():
    spans = zipformer_fingerprint("Quran-Lab/zipformer_p-arabic-v3@abc", "spans")
    checks = zipformer_fingerprint("Quran-Lab/zipformer_p-arabic-v3@abc", "checks")
    for fingerprint in (spans, checks):
        assert (fingerprint.weights_dtype, fingerprint.batch_size, fingerprint.device_type,
                fingerprint.autocast) == ("fp32", 1, "cpu", False)
        assert fingerprint.policy != dc.INFERENCE_POLICY
    assert spans.mode != checks.mode
    ours = {dc.SPANS, dc.APP_CHECKS, dc.stream_protocol()}
    assert not ({spans.mode, checks.mode} & ours)
    assert dc.DecodeFingerprint.from_dict(spans.as_dict(), "zipformer") == spans


def test_a_zipformer_decode_is_not_comparable_with_one_of_ours():
    student = dc.DecodeFingerprint(model="m", mode=dc.stream_protocol(), weights_dtype="bf16",
                                   batch_size=1, device_type="cuda", autocast=True)
    with pytest.raises(ValueError):
        student.check_comparable(zipformer_fingerprint("z", "spans"), "student vs zipformer")


def test_there_is_no_stream_b0_fingerprint_for_zipformer():
    with pytest.raises(ValueError):
        zipformer_fingerprint("z", "stream_b0")

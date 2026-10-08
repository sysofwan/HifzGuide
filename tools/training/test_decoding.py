"""Tests for the one decode interface and the streaming protocol it replays.

The protocol's job is to replay what Muraja does exactly. If ``scan_ctc`` or the commit rule
drift from ``MuaalemInference.predictSplit``, every number built on the stream stops
predicting what the user sees. The protocol half is pinned on synthetic argmax rows with no
model at all; the interface half runs a tiny randomly-initialised student on CPU -- no GPU,
no download.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from training import decoding as dc
from training.distill_data import SAMPLE_RATE, WINDOW_SAMPLES
from training.decoding import CONFIRM_TIMESTEPS
from training.distill_loss import BLANK_ID
from training.distill_student import DEPLOYED_LOGIT_FRAMES, NUM_PHONEME_CLASSES
from training.window_position import block_tokens

CPU = torch.device("cpu")


def _ids(*tokens: int) -> np.ndarray:
    return np.array(tokens, dtype=np.int64)


def row(*spans) -> np.ndarray:
    """A 125-step argmax row from ``(token, start, end_inclusive)`` spans; rest blank."""
    ids = np.zeros(DEPLOYED_LOGIT_FRAMES, dtype=np.int64)
    for token, start, end in spans:
        ids[start : end + 1] = token
    return ids


def random_rows(rng: random.Random, windows: int, frames: int = DEPLOYED_LOGIT_FRAMES):
    """Blank-heavy argmax rows with runs of varying length, so segments straddle blocks."""
    rows = []
    for _ in range(windows):
        ids: list[int] = []
        while len(ids) < frames:
            token = BLANK_ID if rng.random() < 0.55 else rng.randrange(1, 12)
            ids.extend([token] * rng.randint(1, 9))
        rows.append(np.array(ids[:frames], dtype=np.int64))
    return rows


def tokens(emissions) -> list[int]:
    return [e.token_id for e in emissions]


# --- CTC collapse, mirroring Swift's scanCTC ----------------------------------------


def test_scan_ctc_merges_a_run_into_one_segment():
    segments = dc.scan_ctc(_ids(0, 7, 7, 7, 0))
    assert len(segments) == 1
    assert segments[0].token_id == 7
    assert (segments[0].start_step, segments[0].end_step) == (1, 3)


def test_blank_separates_repeated_tokens():
    """The entire point of the CTC blank: 7 blank 7 is two tokens, not one."""
    assert [s.token_id for s in dc.scan_ctc(_ids(7, 0, 7))] == [7, 7]


def test_adjacent_identical_tokens_collapse_to_one():
    assert [s.token_id for s in dc.scan_ctc(_ids(7, 7, 7))] == [7]


def test_blank_runs_are_never_emitted():
    assert dc.scan_ctc(_ids(0, 0, 0)) == []


def test_a_segment_running_to_the_end_is_closed():
    """A token still open at the last timestep must still be emitted."""
    segments = dc.scan_ctc(_ids(0, 9, 9))
    assert [s.token_id for s in segments] == [9]
    assert segments[0].end_step == 2


def test_midpoint_is_the_centre_of_the_run():
    assert dc.scan_ctc(_ids(0, 5, 5, 5, 0))[0].midpoint == pytest.approx(2.0)


# --- The deployed commit split --------------------------------------------------------


def test_only_segments_before_the_split_are_confirmed():
    """`predictSplit` commits on `seg.midpoint < 25` -- the OLDEST second of the buffer."""
    assert dc.confirmed_tokens(row((7, 5, 7), (9, 60, 62))) == [7]


def test_a_segment_straddling_the_split_goes_by_its_midpoint():
    """Not by its start or end -- the midpoint is what Swift compares."""
    assert dc.confirmed_tokens(row((7, 20, 31))) == []  # midpoint 25.5
    assert dc.confirmed_tokens(row((7, 18, 29))) == [7]  # midpoint 23.5


def test_confirmation_boundary_is_exclusive():
    assert dc.confirmed_tokens(row((7, CONFIRM_TIMESTEPS, CONFIRM_TIMESTEPS))) == []
    assert dc.confirmed_tokens(row((7, CONFIRM_TIMESTEPS - 1, CONFIRM_TIMESTEPS - 1))) == [7]


def test_blank_is_never_confirmed():
    assert dc.confirmed_tokens(row()) == []
    assert BLANK_ID == 0


def test_flushing_adds_the_tail_of_its_own_window():
    """The flush adds this window's steps [25, 125), which no later window exists to decode."""
    ids = np.array([5] * 10 + [0] * 40 + [7] * 20 + [0] * 55)
    assert dc.confirmed_tokens(ids, CONFIRM_TIMESTEPS) == [5]
    assert dc.confirmed_tokens(ids, DEPLOYED_LOGIT_FRAMES) == [5, 7]


def test_a_segment_straddling_the_boundary_is_emitted_by_both_windows():
    """Documents a real double-emission, so nobody re-derives it as a surprise.

    A run at steps 18-29 has midpoint 23.5 and commits from this window. One second later the
    same audio sits at steps 0-4 of the next window and commits again. Faithful to
    ``predictSplit``; pinned so the behaviour is a recorded property, not an assumed absence.
    """
    window_k = np.array([0] * 18 + [9] * 12 + [0] * 95)
    window_k1 = np.array([9] * 5 + [0] * 120)
    assert tokens(dc.stream_emissions([window_k, window_k1], flush_tail=False)) == [9, 9]


# --- Window scheduling ----------------------------------------------------------------


def test_windows_advance_by_one_second():
    starts = dc.clip_windows(WINDOW_SAMPLES + 3 * dc.HOP_SAMPLES)
    assert starts == [0, dc.HOP_SAMPLES, 2 * dc.HOP_SAMPLES, 3 * dc.HOP_SAMPLES]


def test_a_clip_shorter_than_one_window_yields_a_single_padded_pass():
    assert dc.clip_windows(WINDOW_SAMPLES // 2) == [0]
    (window,) = dc.window_audio(np.ones(WINDOW_SAMPLES // 2, dtype=np.float32))
    assert len(window) == WINDOW_SAMPLES
    assert window[WINDOW_SAMPLES // 2 :].sum() == 0.0


def test_no_window_runs_off_the_end():
    starts = dc.clip_windows(WINDOW_SAMPLES + 12345)
    assert all(s + WINDOW_SAMPLES <= WINDOW_SAMPLES + 12345 for s in starts)


# --- Which timesteps each window commits ----------------------------------------------


def test_the_deployed_block_commits_the_oldest_second_and_flushes_the_last_window():
    assert dc.commit_bounds(0, 4) == (0, CONFIRM_TIMESTEPS)
    assert dc.commit_bounds(3, 4) == (0, CONFIRM_TIMESTEPS)
    assert dc.commit_bounds(4, 4) == (0, DEPLOYED_LOGIT_FRAMES)
    assert dc.commit_bounds(0, 0) == (0, DEPLOYED_LOGIT_FRAMES)


def test_the_flush_can_be_turned_off_to_reproduce_the_old_protocol():
    assert dc.commit_bounds(4, 4, flush_tail=False) == (0, CONFIRM_TIMESTEPS)
    assert dc.commit_bounds(0, 0, flush_tail=False) == (0, CONFIRM_TIMESTEPS)


def test_a_later_block_commits_its_own_second_with_startup_and_flush():
    assert dc.commit_bounds(2, 5, block=1) == (25, 50)
    assert dc.commit_bounds(0, 5, block=1) == (0, 50)  # startup: no window -1
    assert dc.commit_bounds(5, 5, block=1) == (25, DEPLOYED_LOGIT_FRAMES)  # flush
    assert dc.commit_bounds(0, 0, block=2) == (0, DEPLOYED_LOGIT_FRAMES)
    assert dc.commit_bounds(0, 0, block=2, flush_tail=False) == (0, 75)


@pytest.mark.parametrize("bad", [-1, dc.NUM_BLOCKS])
def test_an_out_of_range_block_is_refused(bad):
    with pytest.raises(ValueError, match="block"):
        dc.commit_bounds(0, 3, block=bad)


@pytest.mark.parametrize("block", range(dc.NUM_BLOCKS))
@pytest.mark.parametrize("num_windows", [1, 2, 6])
@pytest.mark.parametrize("flush_tail", [True, False])
def test_a_stream_at_block_b_is_window_positions_block_b_plus_startup_and_flush(
    block, num_windows, flush_tail
):
    """``training.window_position`` defines block b per window; the stream is that, in order.

    The only additions are the stated startup rule (the first window also commits the blocks
    before b) and the flush (the last window also commits the blocks after b).
    """
    rows = random_rows(random.Random(block * 100 + num_windows), num_windows)
    last = num_windows - 1
    expected: list[int] = []
    for window, ids in enumerate(rows):
        first_block = 0 if window == 0 else block
        last_block = dc.NUM_BLOCKS - 1 if flush_tail and window == last else block
        for b in range(first_block, last_block + 1):
            expected.extend(block_tokens(ids, b))
    assert tokens(dc.stream_emissions(rows, block, flush_tail)) == expected


@pytest.mark.parametrize("block", range(dc.NUM_BLOCKS))
def test_every_replayed_second_is_committed_exactly_once_in_order(block):
    """With a model that hears each second the same way from every window, the stream at any
    block is each second once -- the startup rule leaves no hole at the start, and the flush
    none at the end."""
    num_windows = 6
    seconds = num_windows - 1 + dc.NUM_BLOCKS

    def heard(second: int) -> int:
        return 1 + second % (NUM_PHONEME_CLASSES - 1)

    rows = [
        row(*((heard(w + b), 25 * b + 10, 25 * b + 12) for b in range(dc.NUM_BLOCKS)))
        for w in range(num_windows)
    ]
    assert tokens(dc.stream_emissions(rows, block)) == [heard(s) for s in range(seconds)]
    assert tokens(dc.stream_emissions(rows, block, flush_tail=False)) == [
        heard(s) for s in range(num_windows + block)
    ]


def test_provenance_follows_the_committed_block():
    rows = [row(), row((3, 30, 32), (4, 47, 52), (5, 60, 62))]
    by_token = {e.token_id: e for e in dc.stream_emissions(rows, block=1)}
    assert not by_token[3].is_flush  # block 1 is committed by every window
    assert by_token[4].straddles_seam  # crosses step 50, the end of block 1
    assert by_token[5].is_flush  # block 2 of the last window: only the flush commits it
    assert {e.block for e in by_token.values()} == {1}


# --- Identity of a stored decode -------------------------------------------------------


def test_the_deployed_stream_keeps_its_protocol_version_and_other_blocks_get_their_own():
    """Frozen eval sets carry PROTOCOL_VERSION; it must not move, and b>0 must not reuse it."""
    assert dc.PROTOCOL_VERSION == "confirmed-stream-v2-flush" == dc.stream_protocol()
    names = {dc.stream_protocol(b, flush) for b in range(dc.NUM_BLOCKS) for flush in (True, False)}
    assert len(names) == 2 * dc.NUM_BLOCKS
    assert dc.stream_protocol(0, flush_tail=False) == "confirmed-stream-v1"


def _decoder(model_ref: str, device: str = "cpu", batch_size: int = 16, dtype=torch.float32):
    """A decoder around a one-parameter module: enough to read its effective settings."""
    return dc.Decoder(model_ref, torch.nn.Linear(1, 1).to(dtype), None, device, batch_size)


def test_a_fingerprint_reads_the_decoder_as_it_actually_runs():
    on_cpu = _decoder("h448.pt").fingerprint(dc.SPANS)
    assert (on_cpu.device_type, on_cpu.autocast, on_cpu.weights_dtype) == ("cpu", False, "fp32")
    on_cuda = _decoder("h448.pt", device="cuda", dtype=torch.bfloat16).fingerprint(dc.SPANS)
    assert (on_cuda.device_type, on_cuda.autocast, on_cuda.weights_dtype) == ("cuda", True, "bf16")
    assert on_cpu.policy == dc.INFERENCE_POLICY


def test_fp32_on_cpu_and_fp32_on_cuda_are_not_comparable():
    """Same --weights-dtype, different numerics: CUDA runs under bf16 autocast, CPU does not."""
    on_cpu = _decoder("teacher").fingerprint(dc.SPANS)
    on_cuda = _decoder("h448.pt", device="cuda").fingerprint(dc.SPANS)
    with pytest.raises(ValueError, match="autocast"):
        on_cpu.check_comparable(on_cuda, "two models")


def test_fingerprints_compare_models_but_not_settings():
    base = _decoder("teacher").fingerprint(dc.PROTOCOL_VERSION)
    base.check_comparable(_decoder("h448.pt").fingerprint(dc.PROTOCOL_VERSION), "two models")
    for other in (
        _decoder("h448.pt").fingerprint(dc.stream_protocol(block=1)),
        _decoder("h448.pt").fingerprint(dc.SPANS),
        _decoder("h448.pt", dtype=torch.bfloat16).fingerprint(dc.PROTOCOL_VERSION),
        _decoder("h448.pt", batch_size=8).fingerprint(dc.PROTOCOL_VERSION),
    ):
        with pytest.raises(ValueError, match="Regenerate the baseline"):
            base.check_comparable(other, "two models")


def test_a_fingerprint_round_trips_and_a_malformed_one_is_refused():
    fingerprint = _decoder("teacher").fingerprint(dc.SPANS)
    assert dc.DecodeFingerprint.from_dict(fingerprint.as_dict(), "x") == fingerprint
    with pytest.raises(ValueError, match="malformed"):
        dc.DecodeFingerprint.from_dict({"model": "teacher"}, "x")
    with pytest.raises(ValueError, match="no decode fingerprint"):
        dc.DecodeFingerprint.from_dict(None, "x")


def test_importing_the_decoding_module_does_not_import_torch():
    """The protocol and the fingerprint are read by torch-free tools (the audit UI)."""
    code = (
        "import sys; import training.decoding, tadabur.tashkeel_acceptance; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr


# --- b=0 reproduces the pre-refactor distill_eval protocol exactly --------------------
#
# ``_reference_*`` below are training.distill_eval's scan_ctc, confirm_split_for_window,
# clip_windows, window_emissions and confirmed_emissions as of bdb1c5c, verbatim but for
# names and the provenance tuple. They are the oracle: the refactor must change nothing.


def _reference_scan_ctc(class_ids):
    segments = []
    current_token = -1
    current_start = 0
    for step, token in enumerate(int(t) for t in class_ids):
        if token == current_token:
            continue
        if current_token != BLANK_ID and current_token != -1:
            segments.append((current_token, current_start, step - 1))
        current_token = token
        current_start = step
    if current_token != BLANK_ID and current_token != -1:
        segments.append((current_token, current_start, len(class_ids) - 1))
    return segments


def _reference_confirm_split_for_window(position, last_index, flush_tail=True):
    if flush_tail and position == last_index:
        return DEPLOYED_LOGIT_FRAMES
    return CONFIRM_TIMESTEPS


def _reference_clip_windows(num_samples, hop_samples=SAMPLE_RATE):
    if num_samples < WINDOW_SAMPLES:
        return [0]
    return list(range(0, num_samples - WINDOW_SAMPLES + 1, hop_samples))


def _reference_window_emissions(class_ids, confirm_timesteps, window, is_final_window):
    return [
        (token, window, start, end, is_final_window)
        for token, start, end in _reference_scan_ctc(class_ids)
        if (start + end) / 2.0 < float(confirm_timesteps)
    ]


@torch.no_grad()
def _reference_confirmed_emissions(
    model, extractor, samples, device, batch_size=16, flush_tail=True
):
    starts = _reference_clip_windows(len(samples))
    windows = []
    for start in starts:
        chunk = samples[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk)
    last_index = len(windows) - 1
    stream = []
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset : offset + batch_size]
        extracted = extractor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        features = extracted.input_features.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(features, return_dict=True)["logits"]["phonemes"]
        ids = logits.float().argmax(dim=-1).cpu().numpy()
        for position, ids_row in enumerate(ids, start=offset):
            stream.extend(
                _reference_window_emissions(
                    ids_row[:DEPLOYED_LOGIT_FRAMES],
                    _reference_confirm_split_for_window(position, last_index, flush_tail),
                    window=position,
                    is_final_window=position == last_index,
                )
            )
    return stream


class _Features:
    def __init__(self, input_features, attention_mask):
        self.input_features = input_features
        self.attention_mask = attention_mask


class _RawExtractor:
    """Passes each waveform through as a one-channel "feature", right-padded with ``pad``."""

    def __init__(self, pad: float = 0.0):
        self.pad = pad
        self.calls = 0

    def __call__(self, waveforms, sampling_rate, return_tensors, padding):
        assert sampling_rate == SAMPLE_RATE and return_tensors == "pt" and padding
        self.calls += 1
        longest = max(len(w) for w in waveforms)
        features = np.full((len(waveforms), longest, 1), self.pad, dtype=np.float32)
        mask = np.zeros((len(waveforms), longest), dtype=np.int64)
        for i, waveform in enumerate(waveforms):
            features[i, : len(waveform), 0] = waveform
            mask[i, : len(waveform)] = 1
        return _Features(torch.from_numpy(features), torch.from_numpy(mask))


def _one_hot(class_ids: np.ndarray) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.from_numpy(class_ids), NUM_PHONEME_CLASSES).float()


class _ScriptedWindowModel:
    """Emits a scripted argmax row per window, keyed by the window's first sample.

    The clip is built so that every sample of second ``s`` holds the value ``s``, so a
    window's first sample is its index. Rows are 126 steps, one more than the deployed
    lattice, so the slice to 125 is exercised too.
    """

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, features, attention_mask=None, return_dict=True):
        assert attention_mask is None  # the fixed-shape window forward passes no mask
        windows = features[:, 0, 0].round().long().tolist()
        logits = torch.stack([_one_hot(self.rows[w]) for w in windows])
        return {"logits": {"phonemes": logits}}


@pytest.mark.parametrize("seconds", [3.0, 5.0, 5.5, 9.0, 12.3])
@pytest.mark.parametrize("batch_size", [1, 3, 16])
@pytest.mark.parametrize("flush_tail", [True, False])
def test_the_deployed_stream_is_the_pre_refactor_distill_eval_stream(
    seconds, batch_size, flush_tail
):
    num_samples = int(seconds * SAMPLE_RATE)
    samples = (np.arange(num_samples) // SAMPLE_RATE).astype(np.float32)
    num_windows = len(dc.clip_windows(num_samples))
    rows = random_rows(random.Random(num_samples + batch_size), num_windows, frames=126)
    for ids in rows:
        ids[3:6] = 11  # every window commits at least one token
    model = _ScriptedWindowModel(rows)
    extractor = _RawExtractor()

    expected = _reference_confirmed_emissions(
        model, extractor, samples, CPU, batch_size, flush_tail
    )
    decoder = dc.Decoder("scripted", model, extractor, CPU, batch_size)
    emissions = decoder.emissions(samples, flush_tail=flush_tail)

    assert len(expected) >= num_windows
    assert [
        (e.token_id, e.window, e.start_step, e.end_step, e.is_final_window) for e in emissions
    ] == expected
    assert decoder.decode_stream(samples, flush_tail=flush_tail) == dc.tokens_to_phonemes(
        token for token, *_ in expected
    )


def test_a_bad_block_is_refused_before_any_forward_pass():
    extractor = _RawExtractor()
    decoder = dc.Decoder("scripted", _ScriptedWindowModel([]), extractor, CPU)
    with pytest.raises(ValueError, match="block"):
        decoder.emissions(np.zeros(WINDOW_SAMPLES, dtype=np.float32), block=dc.NUM_BLOCKS)
    assert extractor.calls == 0


# --- Whole spans -----------------------------------------------------------------------


class _FrameModel:
    """Reads each feature frame as a class id and halves the frame rate, like the adapter."""

    def __call__(self, features, attention_mask=None, return_dict=True):
        assert attention_mask is not None  # padded spans need a real mask
        class_ids = features[:, ::2, 0].round().long().numpy()
        return {"logits": {"phonemes": torch.stack([_one_hot(ids) for ids in class_ids])}}

    @staticmethod
    def _get_feat_extract_output_lengths(lengths):
        return (lengths + 1) // 2


def test_spans_are_decoded_whole_and_cut_to_their_own_length():
    """Padding is a non-blank class here, so any padding frame that leaked would show."""
    spans = [
        np.array([3, 3, 0, 0, 5, 5], dtype=np.float32),
        np.array([7, 7], dtype=np.float32),
        np.array([2, 2, 0, 0, 2, 2, 4, 4, 4, 4], dtype=np.float32),
    ]
    extractor = _RawExtractor(pad=9.0)
    decoder = dc.Decoder("frames", _FrameModel(), extractor, CPU, batch_size=2)

    decodes = decoder.decode_spans(iter(spans))

    assert decodes == [
        dc.tokens_to_phonemes([3, 5]),
        dc.tokens_to_phonemes([7]),
        dc.tokens_to_phonemes([2, 2, 4]),
    ]
    assert extractor.calls == 2  # consumed lazily, one batch at a time
    assert decoder.decode_spans([]) == []


def test_span_class_ids_are_the_uncollapsed_rows_cut_to_each_span():
    spans = [np.array([3, 3, 0, 0, 5, 5], dtype=np.float32), np.array([7, 7], dtype=np.float32)]
    decoder = dc.Decoder("frames", _FrameModel(), _RawExtractor(pad=9.0), CPU, batch_size=2)

    rows = decoder.span_class_ids(spans)

    assert [row.tolist() for row in rows] == [[3, 0, 5], [7]]


def test_span_log_posteriors_are_normalized_rows_of_the_same_pass():
    spans = [np.array([3, 3, 0, 0, 5, 5], dtype=np.float32), np.array([7, 7], dtype=np.float32)]
    decoder = dc.Decoder("frames", _FrameModel(), _RawExtractor(pad=9.0), CPU, batch_size=2)

    rows = decoder.span_log_posteriors(spans)

    assert [row.shape for row in rows] == [(3, NUM_PHONEME_CLASSES), (1, NUM_PHONEME_CLASSES)]
    assert all(row.dtype == np.float32 for row in rows)
    for posteriors in rows:
        np.testing.assert_allclose(np.exp(posteriors).sum(axis=1), 1.0, rtol=1e-6)
    assert [row.argmax(axis=1).tolist() for row in rows] == [
        row.tolist() for row in decoder.span_class_ids(spans)
    ]


# --- Loading a model reference ---------------------------------------------------------


TINY = "tiny-test"


@pytest.fixture
def tiny_student(tmp_path, monkeypatch):
    """One set of random student weights, saved both ways a model reference can name them.

    Returns ``(checkpoint_file, hf_dir)``. The checkpoint loader needs the preset to exist
    and the teacher's feature extractor, which is pointed at the saved directory so nothing
    is downloaded.
    """
    from transformers import SeamlessM4TFeatureExtractor

    from training.distill_student import PRESETS, StudentSpec, build_student

    monkeypatch.setitem(
        PRESETS,
        TINY,
        StudentSpec(
            TINY, hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=2
        ),
    )
    torch.manual_seed(0)
    student = build_student(PRESETS[TINY]).eval()

    hf_dir = tmp_path / "hf"
    student.save_pretrained(hf_dir)
    SeamlessM4TFeatureExtractor().save_pretrained(hf_dir)
    monkeypatch.setattr(dc, "STUDENT_FEATURE_EXTRACTOR", str(hf_dir))

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": {"preset": TINY}, "student": student.state_dict(), "step": 7}, checkpoint)
    return checkpoint, hf_dir


def test_the_same_weights_decode_identically_through_either_loader(tiny_student):
    checkpoint, hf_dir = tiny_student
    rng = np.random.default_rng(0)
    clip = rng.normal(scale=0.1, size=int(7.4 * SAMPLE_RATE)).astype(np.float32)
    spans = [clip[: int(1.3 * SAMPLE_RATE)], clip[SAMPLE_RATE : 4 * SAMPLE_RATE]]

    decoders = [
        dc.Decoder.load(ref, CPU, weights_dtype="fp32", batch_size=2)
        for ref in (checkpoint, hf_dir)
    ]
    from_checkpoint, from_hf = decoders

    windows = dc.window_audio(clip)
    rows = [d.window_rows(windows) for d in decoders]
    assert all(np.array_equal(a, b) for a, b in zip(*rows))
    assert any(len(set(r.tolist())) > 1 for r in rows[0])  # not a constant output

    assert from_checkpoint.decode_spans(spans) == from_hf.decode_spans(spans)
    for block in (0, 1):
        assert from_checkpoint.decode_stream(clip, block) == from_hf.decode_stream(clip, block)


def test_bf16_weights_are_refused_without_cuda_autocast(tiny_student):
    checkpoint, _ = tiny_student
    with pytest.raises(ValueError, match="autocast"):
        dc.Decoder.load(checkpoint, CPU, weights_dtype="bf16")


def test_averaged_weights_cannot_be_asked_of_a_hugging_face_reference(tiny_student):
    _, hf_dir = tiny_student
    with pytest.raises(ValueError, match="averaged"):
        dc.Decoder.load(hf_dir, CPU, weights_dtype="fp32", use_ema=True)


def test_a_head_with_another_vocabulary_is_refused(tmp_path):
    from tadabur.muaalem import Wav2Vec2BertForMultilevelCTC

    from training.distill_student import PRESETS, build_student_config

    config = build_student_config(PRESETS["h256"])
    config.num_hidden_layers = 1
    config.level_to_vocab_size = {"phonemes": NUM_PHONEME_CLASSES + 1}
    Wav2Vec2BertForMultilevelCTC(config).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="classes"):
        dc.load_hf_model(str(tmp_path))

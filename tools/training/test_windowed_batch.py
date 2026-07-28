"""Tests for the phoneme-only windowed CTC collator (ADR-0004 P7.D2, issue #29).

Cover the data-integrity guarantees the fine-tune rests on: the CTC label strips
word-separator spaces and rejects out-of-vocabulary characters (no silent label
corruption), the length bucketing honours the frame/window budget deterministically
(the 16 GB knob), and the collate pads labels with the ``-100`` ignore index.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tadabur.phoneme_vocab import PHONEME_CHAR_TO_ID
from training.windowed_batch import (
    WindowedCtcCollator,
    WindowedCtcExample,
    encode_phoneme_label,
    length_bucketed_batches,
    pad_labels,
)
from training.windowed_labels import WindowLabel, read_labels, write_labels


# --- encode_phoneme_label ----------------------------------------------------


def test_encode_strips_word_separator_spaces():
    # "م لق" — a space word-separator between phonemes must not become a CTC class.
    ids = encode_phoneme_label("\u0645 \u0644\u0642")
    assert ids == [
        PHONEME_CHAR_TO_ID["\u0645"],
        PHONEME_CHAR_TO_ID["\u0644"],
        PHONEME_CHAR_TO_ID["\u0642"],
    ]


def test_encode_rejects_out_of_vocab_character():
    with pytest.raises(ValueError, match="not a model phoneme class"):
        encode_phoneme_label("\u0645X")  # 'X' has no phoneme class


def test_encode_empty_and_all_spaces():
    assert encode_phoneme_label("") == []
    assert encode_phoneme_label("   ") == []


# --- WindowedCtcExample feasibility ------------------------------------------


def _example(feature_frames, logit_frames, label_len, key=("clip", 0)):
    return WindowedCtcExample(
        key=key,
        audio=np.zeros(feature_frames * 320, dtype=np.float32),
        label_ids=tuple(range(1, label_len + 1)),
        start_sample=0,
        num_samples=feature_frames * 320,
        feature_frames=feature_frames,
        logit_frames=logit_frames,
    )


def test_example_rejects_infeasible_ctc_target():
    with pytest.raises(ValueError, match="infeasible CTC target"):
        _example(feature_frames=20, logit_frames=10, label_len=11)


def test_example_accepts_feasible_target():
    ex = _example(feature_frames=20, logit_frames=10, label_len=10)
    assert len(ex.label_ids) == 10


# --- length bucketing --------------------------------------------------------


def test_bucketing_respects_window_cap():
    examples = [_example(50, 25, 5, key=("c", i)) for i in range(7)]
    batches = length_bucketed_batches(
        examples, max_frames_per_batch=10_000, max_windows_per_batch=3, seed=0
    )
    assert all(len(b) <= 3 for b in batches)
    assert sum(len(b) for b in batches) == 7


def test_bucketing_respects_frame_budget():
    # Each window is 100 feature frames; a 250-frame budget fits 2 (padded 200) not 3 (300).
    examples = [_example(100, 50, 5, key=("c", i)) for i in range(6)]
    batches = length_bucketed_batches(
        examples, max_frames_per_batch=250, max_windows_per_batch=99, seed=0
    )
    assert all(len(b) <= 2 for b in batches)
    assert sum(len(b) for b in batches) == 6


def test_bucketing_groups_similar_lengths():
    short = [_example(20, 10, 3, key=("c", i)) for i in range(2)]
    long = [_example(200, 100, 3, key=("c", i + 10)) for i in range(2)]
    batches = length_bucketed_batches(
        short + long, max_frames_per_batch=10_000, max_windows_per_batch=2, seed=0
    )
    lengths_per_batch = {tuple(sorted(e.feature_frames for e in b)) for b in batches}
    # Similar lengths land together: no batch mixes a 20-frame and a 200-frame window.
    assert (20, 20) in lengths_per_batch and (200, 200) in lengths_per_batch


def test_bucketing_is_deterministic():
    examples = [_example(50, 25, 5, key=("c", i)) for i in range(10)]
    a = length_bucketed_batches(examples, 10_000, 3, seed=7)
    b = length_bucketed_batches(examples, 10_000, 3, seed=7)
    assert [[e.key for e in batch] for batch in a] == [[e.key for e in batch] for batch in b]


def test_bucketing_rejects_over_budget_window():
    with pytest.raises(ValueError, match="over the"):
        length_bucketed_batches([_example(300, 150, 5)], max_frames_per_batch=250,
                                max_windows_per_batch=8, seed=0)


# --- pad_labels --------------------------------------------------------------


def test_pad_labels_uses_minus_100_ignore_index():
    padded = pad_labels([(1, 2, 3), (4,)])
    assert padded.shape == (2, 3)
    assert padded[0].tolist() == [1, 2, 3]
    assert padded[1].tolist() == [4, -100, -100]


# --- collator ----------------------------------------------------------------


class _StubFeatureExtractor:
    """Minimal stand-in: pads waveforms to a shared frame count with a validity mask."""

    sampling_rate = 16000

    def __call__(self, waveforms, sampling_rate, return_tensors, padding):
        frames = [len(w) // 160 for w in waveforms]
        max_frames = max(frames)
        features = torch.zeros(len(waveforms), max_frames, 4)
        mask = torch.zeros(len(waveforms), max_frames, dtype=torch.long)
        for i, n in enumerate(frames):
            mask[i, :n] = 1
        return type("F", (), {"input_features": features, "attention_mask": mask})()


def test_collator_pads_features_labels_and_mask():
    examples = [
        WindowedCtcExample(("c", 0), np.zeros(160 * 30, np.float32), (1, 2), 0, 160 * 30, 30, 15),
        WindowedCtcExample(("c", 1), np.zeros(160 * 20, np.float32), (3,), 0, 160 * 20, 20, 10),
    ]
    batch = WindowedCtcCollator(_StubFeatureExtractor())(examples)
    assert batch.input_features.shape == (2, 30, 4)
    assert batch.attention_mask[1].sum() == 20  # shorter window's real frames
    assert batch.labels[1].tolist() == [3, -100]
    assert batch.keys == [("c", 0), ("c", 1)]


def test_collator_rejects_wrong_sample_rate():
    class Wrong(_StubFeatureExtractor):
        sampling_rate = 8000

    with pytest.raises(ValueError, match="16000 Hz"):
        WindowedCtcCollator(Wrong())


def test_collator_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        WindowedCtcCollator(_StubFeatureExtractor())([])


# --- read_labels round-trips write_labels ------------------------------------


def _label(clip, window_index, reciter_id):
    return WindowLabel(
        clip_audio_filename=clip,
        surah_ayah="101:2",
        reciter_id=reciter_id,
        window_index=window_index,
        start_sample=window_index * 64000,
        num_samples=80000,
        recitation_start_sample=0,
        feature_frames=250,
        logit_frames=125,
        phoneme_label="\u0645\u0644\u0642",
        word_start=0,
        word_end=2,
        segment_indices=(0, 1),
    )


def test_read_labels_round_trips_by_split(tmp_path):
    path = tmp_path / "labels.jsonl"
    train = [_label("clipA.wav", 0, 1), _label("clipA.wav", 1, 1)]
    val = [_label("clipB.wav", 0, 2)]
    write_labels(path, train, "train")
    write_labels(path, val, "val")

    by_split = read_labels(path)
    assert {w.window_index for w in by_split["train"]} == {0, 1}
    assert by_split["val"][0].clip_audio_filename == "clipB.wav"
    assert by_split["train"][0].segment_indices == (0, 1)  # tuple restored, not list


# --- joint batch: phoneme + pooled VAD silence teacher (ADR-0004 D3, #31) -----

from training.windowed_batch import (  # noqa: E402
    JointWindowedCollator,
    JointWindowedExample,
    load_joint_examples,
    pad_target_silence,
)


def _joint_example(feature_frames, logit_frames, label_len, silence_len, key=("c", 0)):
    return JointWindowedExample(
        ctc=_example(feature_frames, logit_frames, label_len, key=key),
        target_silence=np.zeros(silence_len, dtype=np.float32),
    )


def test_joint_example_rejects_silence_length_mismatch():
    # The silence teacher must be exactly logit_frames long (same 40 ms lattice as the head).
    with pytest.raises(ValueError, match="different window grids"):
        _joint_example(feature_frames=20, logit_frames=10, label_len=5, silence_len=9)


def test_joint_example_delegates_key_and_feature_frames():
    ex = _joint_example(feature_frames=40, logit_frames=20, label_len=5, silence_len=20,
                        key=("clip", 3))
    assert ex.key == ("clip", 3)
    assert ex.feature_frames == 40


def test_pad_target_silence_pads_with_speech_zero():
    padded = pad_target_silence(
        [np.array([0.2, 0.9, 0.1], np.float32), np.array([0.5], np.float32)]
    )
    assert padded.shape == (2, 3)
    assert padded[0].tolist() == pytest.approx([0.2, 0.9, 0.1])
    assert padded[1].tolist() == pytest.approx([0.5, 0.0, 0.0])  # speech-padded tail


def test_joint_collator_pads_phoneme_and_silence():
    examples = [
        JointWindowedExample(
            WindowedCtcExample(("c", 0), np.zeros(160 * 30, np.float32), (1, 2), 0, 160 * 30, 30, 15),
            np.full(15, 0.3, np.float32),
        ),
        JointWindowedExample(
            WindowedCtcExample(("c", 1), np.zeros(160 * 20, np.float32), (3,), 0, 160 * 20, 20, 10),
            np.full(10, 0.7, np.float32),
        ),
    ]
    batch = JointWindowedCollator(_StubFeatureExtractor())(examples)
    assert batch.phoneme.input_features.shape == (2, 30, 4)
    assert batch.target_silence.shape == (2, 15)  # padded to the longest window's lattice
    assert batch.target_silence[1, :10].tolist() == pytest.approx([0.7] * 10)
    assert batch.target_silence[1, 10:].tolist() == pytest.approx([0.0] * 5)
    assert batch.keys == [("c", 0), ("c", 1)]


def test_joint_collator_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        JointWindowedCollator(_StubFeatureExtractor())([])


# --- load_joint_examples: the phoneme↔soft-label join integrity ---------------


def _write_joint_fixture(tmp_path, *, soft_num_samples=80000, soft_start_shift=0):
    """Write windowed phoneme labels, a 16 kHz clip, and a matching recitation soft store.

    ``soft_num_samples`` / ``soft_start_shift`` deliberately drift the soft store's window
    span (length / origin) from the phoneme labels so the join's fail-fast can be exercised.
    """
    import soundfile as sf

    from tadabur.audit_sampler import local_audio_path
    from training.waqf_distill import SoftLabelStore, WindowContract, Window
    from training.waqf_distill import WINDOW_ORIGIN_RECITATION

    labels_path = tmp_path / "labels.jsonl"
    write_labels(labels_path, [_label("clipA.wav", 0, 1), _label("clipA.wav", 1, 1)], "train")

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / local_audio_path("clipA.wav"),
             np.zeros(160000, dtype=np.float32), 16000)

    soft_root = tmp_path / "soft"
    windows = [
        (Window(index=0, start_sample=0 + soft_start_shift, num_samples=soft_num_samples),
         np.full(125, 0.2, np.float32)),
        (Window(index=1, start_sample=64000 + soft_start_shift, num_samples=soft_num_samples),
         np.full(125, 0.8, np.float32)),
    ]
    with SoftLabelStore.open(
        soft_root, WindowContract(), window_origin=WINDOW_ORIGIN_RECITATION
    ) as store:
        store.write_clip("clipA.wav", windows, num_samples=160000,
                         recitation_num_samples=160000)
    return labels_path, audio_dir, soft_root


def test_load_joint_examples_pairs_phoneme_and_silence(tmp_path):
    labels_path, audio_dir, soft_root = _write_joint_fixture(tmp_path)
    examples = load_joint_examples(labels_path, audio_dir, soft_root, "train")
    assert [e.key for e in examples] == [("clipA.wav", 0), ("clipA.wav", 1)]
    assert examples[0].target_silence.shape == (125,)
    np.testing.assert_allclose(examples[1].target_silence, 0.8, atol=1e-6)


def test_load_joint_examples_rejects_span_drift(tmp_path):
    # Soft target on a different-length audio span than the phoneme label → fail fast.
    labels_path, audio_dir, soft_root = _write_joint_fixture(tmp_path, soft_num_samples=79000)
    with pytest.raises(ValueError, match="different window grids"):
        load_joint_examples(labels_path, audio_dir, soft_root, "train")


def test_load_joint_examples_rejects_same_length_shifted_start(tmp_path):
    # Same window length but a shifted hop/origin (same window_index) must be rejected — a
    # length-only check would silently pair a misaligned silence teacher (ADR-0004).
    labels_path, audio_dir, soft_root = _write_joint_fixture(tmp_path, soft_start_shift=640)
    with pytest.raises(ValueError, match="different window grids"):
        load_joint_examples(labels_path, audio_dir, soft_root, "train")


# --- real SeamlessM4TFeatureExtractor parity (ADR-0004 A2 frozen 5 s window, #8) ---
#
# Every other collator test above uses a stub extractor. Acceptance criterion #1 of #8 is
# *train/inference parity*: features must come from the model's own SeamlessM4TFeatureExtractor
# at the frozen 5 s window, and the window must land on the 40 ms lattice length the CTC
# feasibility check and the waqf silence grid assume. That contract is only real against the
# actual extractor, so it is pinned here — guarded to skip cleanly offline (the extractor's
# preprocessor config is a small download, not the model weights).


def _real_feature_extractor():
    pytest.importorskip("transformers")
    from transformers import SeamlessM4TFeatureExtractor

    from tadabur.inference import MODEL_ID

    try:
        return SeamlessM4TFeatureExtractor.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001 — offline / config not cached → skip, don't fail
        pytest.skip(f"SeamlessM4TFeatureExtractor for {MODEL_ID} unavailable: {exc}")


def test_real_extractor_5s_window_lands_on_frozen_40ms_lattice():
    from training.waqf_distill import (
        DEPLOYED_WINDOW_FEATURE_FRAMES,
        SAMPLES_PER_TEACHER_FRAME,
        WindowContract,
        muaalem_lattice_length,
    )

    fe = _real_feature_extractor()
    window_samples = DEPLOYED_WINDOW_FEATURE_FRAMES * SAMPLES_PER_TEACHER_FRAME  # 80 000 = 5 s
    student_frames = WindowContract().student_frames  # 125 (the frozen 40 ms lattice)
    example = WindowedCtcExample(
        key=("clip", 0),
        audio=np.zeros(window_samples, dtype=np.float32),
        label_ids=(1, 2, 3),
        start_sample=0,
        num_samples=window_samples,
        feature_frames=DEPLOYED_WINDOW_FEATURE_FRAMES,
        logit_frames=student_frames,
    )

    batch = WindowedCtcCollator(fe)([example])

    # The extractor emits the model's 160-d SeamlessM4T features — the exact input inference
    # feeds the backbone (tadabur.inference), so training preprocessing is identical.
    assert batch.input_features.shape[0] == 1
    assert batch.input_features.shape[-1] == 160

    # The real extractor emits 249 frames for a 5 s window, not the analytical 250 — but both
    # map through the model's adapter-lattice rule to the same frozen 125-frame 40 ms lattice
    # the label's logit_frames was built on. That ≤1-frame tail drift must never move the
    # lattice length the CTC feasibility check and the waqf silence teacher are asserted against.
    real_feature_frames = int(batch.attention_mask[0].sum())
    assert abs(real_feature_frames - DEPLOYED_WINDOW_FEATURE_FRAMES) <= 1
    assert muaalem_lattice_length(real_feature_frames) == student_frames == 125


def test_clip_audio_cache_retains_only_the_current_clip(tmp_path):
    """The corpus-scale RAM guard: caching every clip costs ~10 GB and thrashes the box."""
    import numpy as np
    import soundfile as sf
    from training.windowed_batch import ClipAudioCache

    for name in ("a.wav", "b.wav"):
        sf.write(tmp_path / name, np.zeros(1600, dtype=np.float32), 16000, subtype="PCM_16")
    cache = ClipAudioCache(tmp_path)

    first = cache.waveform("a.wav")
    cache.waveform("b.wav")

    assert len(cache._cache) == 1
    assert len(first) == 1600  # the handed-out array stays valid after eviction

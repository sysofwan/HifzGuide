"""Phase A's word times (#263): the teacher's CTC forced alignment of the gating reference.

Torch-free: the posteriors are planted, so a word's frames are known. The interface these tests
fix, in ``tadabur.tlog_mining.word_times``:

* ``align_words(log_posteriors, labels, offsets, *, duration_s) -> Alignment``: runs
  :func:`training.ctc_paths.ctc_best_path` on ``labels`` (the reference's class ids) and maps the
  labels to words through ``offsets`` (``len(words) + 1`` label offsets, word ``j`` spanning
  ``offsets[j]:offsets[j + 1]``). ``Alignment`` has ``word_times`` (one ``(start_s, end_s)`` per
  word: its first label's first frame, and the end of its last label's last frame),
  ``seconds_per_frame`` (``duration_s / frames``) and ``ll_per_frame``
  (``ctc_log_likelihood / frames``). A transcript the frames cannot hold raises ``ValueError``,
  as does an ``offsets`` that does not partition the labels.
* ``build_record(row, *, reference_for, log_posteriors, duration_s) -> dict``: the committed
  row of one gating clip. ``reference_for(filename_label)`` returns the raw spaceless phoneme
  reference and its word offsets (``waqf_segments.hafs_word_reference`` over the ayah's words).
  The reference is the **filename** ayah, never ``assigned_label``. A clip that cannot be aligned
  is a record with ``aligned: False``, ``word_times: None`` and a ``reason`` (an unphonetizable
  ayah, no ayah, a character outside the vocabulary, too few frames), never an exception or a
  missing row.
"""

from __future__ import annotations

import numpy as np
import pytest

from tadabur.phoneme_vocab import NUM_PHONEME_CLASSES, PHONEME_CHAR_TO_ID
from tadabur.tlog_mining.word_times import align_words, build_record
from training.ctc_paths import ctc_log_likelihood

#: Six phonemes in three words, the middle word starting on a repeat of the first word's last
#: label: a CTC repeat that needs the blank the plan puts between them.
LABELS = [PHONEME_CHAR_TO_ID[c] for c in "بتتثجح"]
OFFSETS = [0, 2, 5, 6]
LEAD, ON, OFF, TAIL = 6, 2, 2, 4
FRAMES = LEAD + len(LABELS) * (ON + OFF) + TAIL
#: Each label's planted ``(first, last)`` frames.
PLANTED = [(LEAD + k * (ON + OFF), LEAD + k * (ON + OFF) + ON - 1) for k in range(len(LABELS))]


def _posteriors(labels=LABELS, planted=PLANTED, frames=FRAMES) -> np.ndarray:
    probabilities = np.full((frames, NUM_PHONEME_CLASSES), 0.1 / (NUM_PHONEME_CLASSES - 1))
    probabilities[:, 0] = 0.9
    for label, (first, last) in zip(labels, planted):
        probabilities[first:last + 1] = 0.1 / (NUM_PHONEME_CLASSES - 1)
        probabilities[first:last + 1, label] = 0.9
    return np.log(probabilities / probabilities.sum(axis=1, keepdims=True))


def _expected(per_frame: float) -> list[tuple[float, float]]:
    return [(PLANTED[first][0] * per_frame, (PLANTED[last - 1][1] + 1) * per_frame)
            for first, last in zip(OFFSETS, OFFSETS[1:])]


def _assert_times(times, per_frame: float) -> None:
    """Every edge within one frame of the planted one (the end convention is open by a frame)."""
    assert len(times) == len(OFFSETS) - 1
    for (start, end), (planted_start, planted_end) in zip(times, _expected(per_frame)):
        assert start == pytest.approx(planted_start, abs=per_frame * 1.01)
        assert end == pytest.approx(planted_end, abs=per_frame * 1.01)
        assert end > start


def test_word_times_land_on_the_planted_word_boundaries():
    alignment = align_words(_posteriors(), LABELS, OFFSETS, duration_s=FRAMES * 0.04)
    _assert_times(alignment.word_times, 0.04)


def test_the_words_are_in_order_and_the_lead_in_is_not_part_of_the_first_word():
    alignment = align_words(_posteriors(), LABELS, OFFSETS, duration_s=FRAMES * 0.04)
    starts = [start for start, _ in alignment.word_times]
    assert starts == sorted(starts)
    assert starts[0] >= (LEAD - 1) * 0.04
    assert alignment.word_times[-1][1] <= FRAMES * 0.04


def test_the_frame_length_is_the_clip_duration_over_the_frames():
    duration = 3.7 * FRAMES / 100
    alignment = align_words(_posteriors(), LABELS, OFFSETS, duration_s=duration)
    assert alignment.seconds_per_frame == pytest.approx(duration / FRAMES)
    _assert_times(alignment.word_times, duration / FRAMES)


def test_the_log_likelihood_is_the_forward_probability_per_frame():
    posteriors = _posteriors()
    alignment = align_words(posteriors, LABELS, OFFSETS, duration_s=FRAMES * 0.04)
    assert alignment.ll_per_frame == pytest.approx(
        ctc_log_likelihood(posteriors, LABELS) / FRAMES)


@pytest.mark.parametrize("offsets", [[0, 2, 5], [1, 2, 5, 6], [0, 2, 2, 6], [0, 5, 2, 6], [0]])
def test_offsets_that_do_not_partition_the_labels_are_refused(offsets):
    with pytest.raises(ValueError):
        align_words(_posteriors(), LABELS, offsets, duration_s=FRAMES * 0.04)


def test_too_few_frames_for_the_transcript_is_a_value_error():
    with pytest.raises(ValueError, match="cannot hold"):
        align_words(_posteriors()[:4], LABELS, OFFSETS, duration_s=0.16)


# --- the committed record ---------------------------------------------------------------------


ROW = {"split": "clean", "shard_file": "clean-00001-of-00411", "row": 7,
       "audio_path": "2_5_123.flac", "audio_sha256": "ab" * 32, "filename_label": "2:5",
       "assigned_label": "2:6", "stratum": "gating"}
REFERENCE = "بتتثجح"


def _record(row=ROW, *, reference_for=None, posteriors=None, duration_s=FRAMES * 0.04):
    seen = []

    def default(label):
        seen.append(label)
        return REFERENCE, OFFSETS

    record = build_record(row, reference_for=reference_for or default,
                          log_posteriors=_posteriors() if posteriors is None else posteriors,
                          duration_s=duration_s)
    return record, seen


def test_an_aligned_clip_is_a_record_with_its_provenance_and_word_times():
    record, _ = _record()
    assert record["aligned"] is True and record["reason"] is None
    assert record["key"] == "2_5_123.flac#0"
    for field in ("split", "shard_file", "row", "audio_path", "audio_sha256"):
        assert record[field] == ROW[field]
    assert record["surah_ayah"] == "2:5"
    _assert_times(record["word_times"], 0.04)
    assert record["seconds_per_frame"] == pytest.approx(0.04)
    assert record["ll_per_frame"] == pytest.approx(
        ctc_log_likelihood(_posteriors(), LABELS) / FRAMES)


def test_the_reference_is_the_filename_ayah_not_the_assigned_label():
    _, seen = _record()
    assert seen == ["2:5"]


@pytest.mark.parametrize("error", [KeyError("2:5"), IndexError("no mapping")])
def test_an_ayah_the_phonetizer_cannot_handle_is_listed_unaligned(error):
    def unsupported(label):
        raise error

    record, _ = _record(reference_for=unsupported)
    assert record["aligned"] is False and record["word_times"] is None
    assert isinstance(record["reason"], str) and record["reason"]
    assert record["key"] == "2_5_123.flac#0" and record["audio_sha256"] == ROW["audio_sha256"]


def test_a_clip_with_no_filename_ayah_is_listed_unaligned_without_a_reference_call():
    record, seen = _record({**ROW, "filename_label": None})
    assert record["aligned"] is False and record["word_times"] is None and record["reason"]
    assert seen == []


def test_a_reference_character_outside_the_vocabulary_is_listed_unaligned():
    record, _ = _record(reference_for=lambda label: ("بتتثجh", OFFSETS))
    assert record["aligned"] is False and record["word_times"] is None and record["reason"]


def test_too_few_frames_is_listed_unaligned_not_raised():
    record, _ = _record(posteriors=_posteriors()[:4], duration_s=0.16)
    assert record["aligned"] is False and record["word_times"] is None and record["reason"]


def test_the_record_is_a_function_of_the_reference_and_posteriors_alone():
    first, _ = _record()
    second, _ = _record({**ROW, "assigned_label": "114:1", "stratum": "gating"})
    assert first == second

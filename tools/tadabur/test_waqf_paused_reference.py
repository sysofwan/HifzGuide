"""The paused-reference helper (#130): one segment realized with a waqf at a heard pause.

The runs are phonetized separately, so the pausal taa-marbuta rule stays in
``tools/hafs_phonetizer.py`` and a mid-segment pause gets the same waqf form as a segment end.
"""

from tadabur.waqf_segments import (
    _uthmani_words,
    hafs_paused_reference,
    hafs_segment_reference,
)

#: After شَفَاعَةٌ in 2:254: the word the guard heard paused (#134, #130).
CUT_254 = 18


def _realizations():
    words = _uthmani_words("2:254")
    return words, hafs_paused_reference(words, {CUT_254}), hafs_segment_reference()(words)


def test_254_splits_the_pause_into_a_pausal_taa_marbuta_and_a_new_run():
    _, (reference, _), _ = _realizations()
    assert "شَفَااعَتُوو" not in reference
    assert "شَفَااعَه وَلكَاافِرُۥۥن" in reference


def test_the_offsets_partition_the_reference_into_its_words():
    words, (reference, offsets), _ = _realizations()
    assert len(offsets) == len(words) + 1
    assert (offsets[0], offsets[-1]) == (0, len(reference))
    assert all(a < b for a, b in zip(offsets, offsets[1:]))
    assert reference[offsets[17]:offsets[18]].strip() == "شَفَااعَه"
    assert reference[offsets[18]:offsets[19]].strip().startswith("وَلكَاافِرُۥۥن")


def test_no_cuts_is_the_unpaused_segment_reference():
    words, _, (wasl, wasl_offsets) = _realizations()
    reference, offsets = hafs_paused_reference(words, ())
    assert (reference, tuple(offsets)) == (wasl, tuple(wasl_offsets))


def test_the_realization_is_idempotent_and_ignores_duplicate_cuts():
    words = _uthmani_words("2:254")
    once = hafs_paused_reference(words, (CUT_254,))
    assert hafs_paused_reference(words, (CUT_254,)) == once
    assert hafs_paused_reference(words, (CUT_254, CUT_254)) == once

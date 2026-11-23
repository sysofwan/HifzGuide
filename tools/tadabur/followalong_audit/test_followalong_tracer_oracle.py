"""The oracle arm (#307): the reference phonemes placed on a clip's phase A teacher word times.

``oracle_checks`` generalises :func:`training.muraja_policy.single_decode_checks` from an even
spread to timed characters: each word's Swift Characters (a consonant with its marks) are spread
over that word's ``(start_s, end_s)``, and a check holds the Characters whose time falls in the
audio the check decodes. A clip whose alignment failed (``word_times`` is ``None``, the committed
``aligned: false`` record) falls back to the even spread and says so.
"""

from __future__ import annotations

import unicodedata

import pytest

from tadabur.followalong_audit.oracle import oracle_checks
from training.decoding import SAMPLE_RATE, clusters, committed_text
from training.muraja_policy import single_decode_checks

#: Two phonetizer words, spaceless as ``hafs_word_reference`` returns them: ``offsets`` partition
#: the reference, word j spanning ``[offsets[j], offsets[j + 1])``. Every consonant carries a
#: haraka, and the last two carry a shadda too (two combining marks).
WORD_1 = "وَلعَاادِيَاتِ"
WORD_2 = "ضَبحَاا"
WORD_3 = "فَالمُورِيَاتِقَدحَا"
REFERENCE = WORD_1 + WORD_2 + WORD_3
OFFSETS = [0, len(WORD_1), len(WORD_1) + len(WORD_2), len(REFERENCE)]
CLIP_SAMPLES = 8 * SAMPLE_RATE


def is_mark(char: str) -> bool:
    return unicodedata.combining(char) != 0


def timed(first: tuple[float, float], second: tuple[float, float],
          third: tuple[float, float]) -> list[tuple[float, float]]:
    return [first, second, third]


def first_window_index(checks: list[dict]) -> int:
    """The first full window: a check that has a hop and is neither a flush nor the stop."""
    return next(i for i, c in enumerate(checks) if c["hop"] and not c["flush"] and not c.get("stop"))


def test_a_word_is_heard_only_once_its_audio_has_arrived():
    # Word 1 ends at 0.8 s, word 2 starts at 2.4 s: the 2.0 s preview holds word 1 and no more.
    checks = oracle_checks(REFERENCE, OFFSETS, timed((0.2, 0.8), (2.4, 3.4), (5.0, 7.0)),
                           CLIP_SAMPLES).checks
    assert checks[0] == {"hop": "", "overlap": WORD_1, "flush": False}


def test_previews_grow_as_prefixes_of_the_reference_ending_on_whole_characters():
    checks = oracle_checks(REFERENCE, OFFSETS, timed((0.2, 0.8), (2.4, 3.4), (5.0, 7.0)),
                           CLIP_SAMPLES).checks
    previews = [c["overlap"] for c in checks[:first_window_index(checks)]]
    assert len(previews) >= 10
    for earlier, later in zip(previews, previews[1:]):
        assert later.startswith(earlier)
    for preview in previews:
        assert REFERENCE.startswith(preview)
        assert len(preview) == len(REFERENCE) or not is_mark(REFERENCE[len(preview)])
    assert previews[-1] != previews[0]


def test_the_first_window_splits_its_text_at_the_one_second_hop():
    checks = oracle_checks(REFERENCE, OFFSETS, timed((0.2, 0.8), (2.4, 3.4), (5.0, 7.0)),
                           CLIP_SAMPLES).checks
    window = checks[first_window_index(checks)]
    # The window is [0, 5 s): its hop is [0, 1 s) and holds word 1; word 2 is in its overlap, and
    # word 3 (from 5.0 s) is not in the window at all.
    assert (window["hop"], window["overlap"]) == (WORD_1, WORD_2)


@pytest.mark.parametrize("start", [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
def test_no_haraka_leaves_its_consonant_wherever_a_word_straddles_a_cut(start):
    # Word 1 straddles the 1 s hop cut at every offset; the previews cut at every 200 ms.
    checks = oracle_checks(REFERENCE, OFFSETS, timed((start, start + 1.1), (2.4, 3.4), (5.0, 7.0)),
                           CLIP_SAMPLES).checks
    for check in checks:
        for piece in (check["hop"], check["overlap"]):
            assert not piece or not is_mark(piece[0]), f"{piece!r} starts on a mark"
    for check in checks[:first_window_index(checks)]:
        text = check["overlap"]
        assert REFERENCE.startswith(text)
        assert len(text) == len(REFERENCE) or not is_mark(REFERENCE[len(text)])
    window = checks[first_window_index(checks)]
    cut = len(window["hop"])
    assert REFERENCE.startswith(window["hop"] + window["overlap"])
    assert cut == len(REFERENCE) or not is_mark(REFERENCE[cut])


def test_every_character_is_committed_once_in_order():
    for times in (timed((0.2, 0.8), (2.4, 3.4), (5.0, 7.0)),
                  timed((0.7, 1.9), (2.0, 2.9), (4.5, 7.9))):
        checks = oracle_checks(REFERENCE, OFFSETS, times, CLIP_SAMPLES).checks
        assert committed_text(checks) == REFERENCE


def test_the_reference_is_cut_in_whole_swift_characters():
    checks = oracle_checks(REFERENCE, OFFSETS, timed((0.2, 0.8), (2.4, 3.4), (5.0, 7.0)),
                           CLIP_SAMPLES).checks
    assert all(clusters(c["overlap"]) == clusters(REFERENCE)[: len(clusters(c["overlap"]))]
               for c in checks[: first_window_index(checks)])


def test_a_timed_clip_is_marked_aligned():
    arm = oracle_checks(REFERENCE, OFFSETS, timed((0.2, 0.8), (2.4, 3.4), (5.0, 7.0)), CLIP_SAMPLES)
    assert arm.aligned is True


def test_a_clip_with_no_word_times_falls_back_to_the_even_spread_and_is_marked_unaligned():
    arm = oracle_checks(REFERENCE, OFFSETS, None, CLIP_SAMPLES)
    assert arm.aligned is False
    assert arm.checks == single_decode_checks(REFERENCE, CLIP_SAMPLES)


def test_word_times_that_do_not_match_the_words_are_refused():
    with pytest.raises(ValueError, match="word"):
        oracle_checks(REFERENCE, OFFSETS, [(0.2, 0.8), (2.4, 3.4)], CLIP_SAMPLES)


def test_the_same_inputs_give_identical_checks():
    times = timed((0.2, 0.8), (2.4, 3.4), (5.0, 7.0))
    assert oracle_checks(REFERENCE, OFFSETS, times, CLIP_SAMPLES) == \
        oracle_checks(REFERENCE, OFFSETS, times, CLIP_SAMPLES)

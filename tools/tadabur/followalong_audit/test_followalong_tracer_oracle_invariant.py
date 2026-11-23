"""The Oracle arm refuses a placement that cannot commit the whole reference (#307).

Word times and WAV come from the same decode, so the invariant should hold; but a truncated or
re-encoded WAV would make the schedule stop before the last Character's time, and the Oracle
table would then show a silently truncated reference as a perfect recitation. The arm raises
instead.
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.oracle import oracle_checks
from training.decoding import SAMPLE_RATE

REFERENCE = "بَتِ" * 20  # 40 Characters, 80 scalars
OFFSETS = [0, len(REFERENCE) // 2, len(REFERENCE)]
CLIP_SAMPLES = 8 * SAMPLE_RATE


def test_word_times_past_the_clip_are_refused_not_silently_truncated():
    with pytest.raises(ValueError, match="past the clip"):
        oracle_checks(REFERENCE, OFFSETS, [(0.2, 4.0), (5.0, 9.0)], CLIP_SAMPLES)


def test_word_times_inside_the_clip_commit_the_whole_reference():
    arm = oracle_checks(REFERENCE, OFFSETS, [(0.2, 4.0), (5.0, 7.9)], CLIP_SAMPLES)
    assert "".join(check["hop"] for check in arm.checks) == REFERENCE

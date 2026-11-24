"""One clip repeated measures the decode's variation, which a seeded pair only samples (#309).

The committed two-run comparison came back identical while one clip had already differed in an
earlier pass, so the audit keeps a probe beside it: the same clip run many times, counting the
decode digests and the word tables. Each repetition must run for real — a shared cache directory
would answer every later repetition from the first and find perfect stability by construction.
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.app_arm import payload_variation
from tadabur.followalong_audit.app_run import ClipOutput

ITEM = "41_29_x.flac#0"
DIGEST_E, DIGEST_F = "e" * 64, "f" * 64


def clip() -> dict:
    return {"audio_path": "41_29_x.flac", "audio_sha256": "a" * 64}


def row(word: int, flagged: bool = False) -> dict:
    cell = {"graded": True, "flagged": flagged, "causes": []}
    return {"item": ITEM, "word": word, "strict": dict(cell), "shipped_default": dict(cell)}


def test_the_probe_counts_each_distinct_payload_and_runs_every_repetition(tmp_path):
    calls: list[int] = []

    def arm(clip_row: dict) -> ClipOutput:
        calls.append(1)
        return ClipOutput([row(1)], DIGEST_E if len(calls) <= 3 else DIGEST_F)

    arm.latency_ms = 40
    probe = payload_variation(clip(), arm, tmp_path, runs=4)

    assert len(calls) == 4, "a repetition answered from another's cache would not call the arm"
    assert probe == {"item": ITEM, "latency_ms": 40, "runs": 4, "failed_runs": [],
                     "payloads": {DIGEST_E: 3, DIGEST_F: 1}, "changed_words": []}


def test_a_repetition_that_moved_a_word_is_named(tmp_path):
    outputs = iter([[row(1), row(2)], [row(1), row(2, flagged=True)]])

    def arm(clip_row: dict) -> ClipOutput:
        return ClipOutput(next(outputs), DIGEST_E)

    arm.latency_ms = 40
    probe = payload_variation(clip(), arm, tmp_path, runs=2)

    assert probe["changed_words"] == [{"item": ITEM, "word": 2}]


def test_a_failed_repetition_is_recorded_and_left_out_of_the_tally(tmp_path):
    calls: list[int] = []

    def arm(clip_row: dict) -> ClipOutput:
        calls.append(1)
        if len(calls) <= 3:  # the first repetition's three attempts
            raise RuntimeError("exited 0 with no driver payload")
        return ClipOutput([row(1)], DIGEST_E)

    arm.latency_ms = 40
    probe = payload_variation(clip(), arm, tmp_path, runs=2)

    assert probe["failed_runs"] == [0]
    assert probe["payloads"] == {DIGEST_E: 1} and probe["changed_words"] == []


def test_a_probe_needs_at_least_two_repetitions(tmp_path):
    with pytest.raises(ValueError):
        payload_variation(clip(), lambda clip_row: ClipOutput([row(1)], DIGEST_E), tmp_path,
                          runs=1)

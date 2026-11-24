"""The determinism check compares the driver's decode, not only the flags it reduced to (#309).

``determinism_report`` runs the seeded subset twice, and each run's result carries the digest of
the driver's own stdout (``ClipOutput.payload_sha256``) alongside the word table. Two runs that
decode differently but ratchet to the same flags are not byte-identical runs, and two runs that
both failed are not a check. Its directories are resumable caches, so a second invocation must
clear them: otherwise both sides are answered from the first invocation's results, a determinism
rerun after a failure proves nothing, and the check silently becomes a no-op on exactly the rerun
an operator makes.
"""

from __future__ import annotations

from tadabur.followalong_audit.app_arm import determinism_report
from tadabur.followalong_audit.app_run import ATTEMPTS, ClipOutput


def clip(index: int) -> dict:
    return {"audio_path": f"{index}_1_{index}.flac", "audio_sha256": f"{index:064x}"}


CLIPS = [clip(index) for index in range(6)]


def test_the_second_invocation_runs_the_subset_again(tmp_path):
    ran: list[str] = []

    def run_one(clip_row: dict) -> list[dict]:
        ran.append(clip_row["audio_path"])
        return [{"word": 1}]

    first = determinism_report(CLIPS, run_one, tmp_path, size=3, seed=1)
    assert len(ran) == 6  # the two runs of the three-clip subset
    again = determinism_report(CLIPS, run_one, tmp_path, size=3, seed=1)
    assert len(ran) == 12, "the second invocation re-ran nothing: both sides came from its cache"
    assert again["clips"] == first["clips"]
    assert again["identical"] is True and first["identical"] is True


def test_a_result_that_changes_between_the_two_runs_is_a_difference(tmp_path):
    calls = {"n": 0}

    def run_one(clip_row: dict) -> list[dict]:
        calls["n"] += 1
        return [{"call": calls["n"]}]

    report = determinism_report(CLIPS, run_one, tmp_path, size=2, seed=1)
    assert report["identical"] is False
    assert len(report["differences"]) == 2


def test_a_different_decode_with_the_same_grades_is_a_difference(tmp_path):
    """Two check schedules that reduce to the same word table are still not identical runs."""
    seen: dict[str, int] = {}

    def run_one(clip_row: dict) -> ClipOutput:
        path = clip_row["audio_path"]
        seen[path] = seen.get(path, 0) + 1
        return ClipOutput([{"word": 1, "strict": {"graded": True}}],
                          "a" * 64 if seen[path] == 1 else "b" * 64)

    report = determinism_report(CLIPS, run_one, tmp_path, size=2, seed=1)
    assert report["identical"] is False
    assert len(report["differences"]) == 2
    assert all(pair["run1"] != pair["run2"] for pair in report["payloads"].values())


def test_two_identical_failures_are_not_a_determinism_pass(tmp_path):
    def run_one(clip_row: dict) -> list[dict]:
        raise RuntimeError("no driver payload")

    report = determinism_report(CLIPS, run_one, tmp_path, size=2, seed=1)
    assert report["identical"] is False
    assert len(report["failed"]) == 2 and report["differences"] == []


def test_a_clip_that_fails_in_one_run_only_is_a_failure(tmp_path):
    calls = {"n": 0}

    def run_one(clip_row: dict) -> list[dict]:
        calls["n"] += 1
        if calls["n"] <= ATTEMPTS:  # every attempt of the first run
            raise RuntimeError("no driver payload")
        return [{"word": 1}]

    report = determinism_report(CLIPS, run_one, tmp_path, size=1, seed=1)
    assert report["identical"] is False
    assert report["failed"] == [report["clips"][0]]
    assert report["differences"] == []


def test_every_compared_clip_reports_both_runs_payload_digests(tmp_path):
    seen = {"n": 0}

    def run_one(clip_row: dict) -> ClipOutput:
        seen["n"] += 1
        return ClipOutput([{"word": 1}], f"{seen['n']:064x}")

    report = determinism_report(CLIPS, run_one, tmp_path, size=2, seed=1)
    digests = report["payloads"]
    assert len(digests) == 2
    assert all(set(pair) == {"run1", "run2"} for pair in digests.values())
    assert all(pair["run1"] != pair["run2"] for pair in digests.values())

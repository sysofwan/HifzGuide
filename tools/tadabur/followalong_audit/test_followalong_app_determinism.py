"""The determinism check re-runs its subset on every invocation (#309, S1.1).

``determinism_report`` compares two runs of the seeded subset. Its directories are resumable
caches, so a second invocation must clear them: otherwise both sides are answered from the first
invocation's results, a determinism rerun after a failure proves nothing, and the check silently
becomes a no-op on exactly the rerun an operator makes.
"""

from __future__ import annotations

from tadabur.followalong_audit.app_arm import determinism_report


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

"""The App-arm run (#309): complete per-word rows, a resumable process-pool driver, accounting.

* ``word_grade_rows`` turns an item's ``word_traces`` rows into one row per word of the ayah, the
  committed ``word_grades.jsonl.gz`` line: ``{"item", "word", "strict": {...}, "shipped_default":
  {...}}``, each scoring's cell ``{"graded", "flagged", "causes"}`` (#300's per-scoring shape). A
  word the engine's store never kept is ``graded: false`` — it is a stratum (S5), not a gap.
* ``run_clips`` runs ``run_one(clip) -> rows`` over the clips, persisting each clip's result under
  ``out_dir`` so a killed run resumes; ``workers=1`` runs in this process, more run a process pool.
"""

from __future__ import annotations

import os

import pytest

from tadabur.followalong_audit.app_run import (
    determinism_keys,
    require_accounted,
    run_clips,
    word_grade_rows,
)
from training.nonpro_decode import item_key

ITEM = "1_2_3.flac#0"


def trace(word: int, quality: str, scoring: str = "strict") -> dict:
    return {"item": ITEM, "scoring": scoring, "word": word, "quality": quality, "errors": []}


def causes_of(row: dict) -> tuple[str, ...]:
    return ("tashkeel",) if row["quality"] == "tashkeelError" else ("other_consonant",)


def rows_of(traces: list[dict], word_count: int = 4) -> list[dict]:
    return word_grade_rows(ITEM, word_count, traces, causes_of)


def test_there_is_one_row_per_word_of_the_ayah_in_order():
    rows = rows_of([trace(2, "correct")], word_count=4)
    assert [(row["item"], row["word"]) for row in rows] == [(ITEM, 1), (ITEM, 2), (ITEM, 3), (ITEM, 4)]
    assert all(set(row) == {"item", "word", "strict", "shipped_default"} for row in rows)


def test_a_word_the_store_never_kept_is_ungraded_not_missing():
    cell = rows_of([trace(2, "correct")])[0]["strict"]
    assert cell == {"graded": False, "flagged": False, "causes": []}


def test_a_pending_word_is_ungraded():
    assert rows_of([trace(1, "pending")])[0]["strict"]["graded"] is False


def test_a_correct_word_is_graded_and_not_flagged():
    assert rows_of([trace(1, "correct")])[0]["strict"] == {
        "graded": True, "flagged": False, "causes": []}


def test_a_flagged_word_carries_the_causes_of_its_trace():
    rows = rows_of([trace(1, "tashkeelError"), trace(2, "minor")])
    assert rows[0]["strict"] == {"graded": True, "flagged": True, "causes": ["tashkeel"]}
    assert rows[1]["strict"] == {"graded": True, "flagged": True, "causes": ["other_consonant"]}


def test_the_two_scorings_are_graded_independently():
    rows = rows_of([trace(1, "minor", "strict"), trace(1, "correct", "shipped_default")])
    assert rows[0]["strict"]["flagged"] is True
    assert rows[0]["shipped_default"] == {"graded": True, "flagged": False, "causes": []}


def test_causes_are_read_only_for_flagged_words():
    seen = []

    def spy(row: dict) -> tuple[str, ...]:
        seen.append(row["word"])
        return ("other_consonant",)

    word_grade_rows(ITEM, 3, [trace(1, "correct"), trace(2, "minor")], spy)
    assert seen == [2]


def test_a_trace_for_a_word_the_ayah_does_not_have_is_refused():
    with pytest.raises(ValueError):
        rows_of([trace(5, "correct")], word_count=4)


# --- the resumable run ------------------------------------------------------------------------


def clip(index: int) -> dict:
    return {"audio_path": f"{index}_1_{index}.flac", "audio_sha256": f"{index:064x}"}


CLIPS = [clip(index) for index in range(4)]


def ok(clip: dict) -> list[dict]:
    return [{"item": item_key(clip["audio_path"]), "word": 1}]


def test_every_clip_gets_a_result_with_its_rows_in_clip_order(tmp_path):
    results = run_clips(CLIPS, ok, tmp_path, workers=1)
    assert [result["item"] for result in results] == [item_key(c["audio_path"]) for c in CLIPS]
    assert all(result["failure"] is None and result["rows"] for result in results)


def test_a_clip_that_raises_is_recorded_with_its_reason_and_the_run_goes_on(tmp_path):
    def flaky(clip: dict) -> list[dict]:
        if clip is CLIPS[1]:
            raise RuntimeError("driver: unpublishable clip")
        return ok(clip)

    results = run_clips(CLIPS, flaky, tmp_path, workers=1)
    assert "unpublishable clip" in results[1]["failure"]
    assert results[1]["rows"] == []
    assert [result["failure"] for result in results[:1] + results[2:]] == [None] * 3


def test_a_rerun_does_not_run_a_finished_clip_again(tmp_path):
    first = run_clips(CLIPS, ok, tmp_path, workers=1)
    again = run_clips(CLIPS, lambda clip: pytest.fail(f"{clip} ran twice"), tmp_path, workers=1)
    assert again == first


def test_a_rerun_retries_only_the_clips_that_failed(tmp_path):
    def flaky(clip: dict) -> list[dict]:
        if clip is CLIPS[2]:
            raise RuntimeError("transient")
        return ok(clip)

    run_clips(CLIPS, flaky, tmp_path, workers=1)
    ran = []
    results = run_clips(CLIPS, lambda clip: ran.append(clip) or ok(clip), tmp_path, workers=1)
    assert ran == [CLIPS[2]]
    assert all(result["failure"] is None for result in results)


def in_a_worker(clip: dict) -> list[dict]:
    return [{"pid": os.getpid()}]


def test_more_than_one_worker_runs_the_clips_in_other_processes(tmp_path):
    results = run_clips(CLIPS, in_a_worker, tmp_path, workers=2)
    assert {row["pid"] for result in results for row in result["rows"]}.isdisjoint({os.getpid()})


# --- accounting --------------------------------------------------------------------------------


def test_a_run_where_every_clip_has_rows_or_a_failure_reason_is_accounted_for():
    results = [{"item": item_key(CLIPS[0]["audio_path"]), "failure": None, "rows": [{"word": 1}]},
               {"item": item_key(CLIPS[1]["audio_path"]), "failure": "driver: x", "rows": []}]
    require_accounted(CLIPS[:2], results)


def test_a_clip_with_no_result_is_refused_by_name():
    with pytest.raises(ValueError, match=CLIPS[1]["audio_path"]):
        require_accounted(CLIPS[:2], [
            {"item": item_key(CLIPS[0]["audio_path"]), "failure": None, "rows": [{"word": 1}]}])


def test_a_clip_with_neither_rows_nor_a_reason_is_refused():
    with pytest.raises(ValueError, match=CLIPS[0]["audio_path"]):
        require_accounted(CLIPS[:1], [
            {"item": item_key(CLIPS[0]["audio_path"]), "failure": None, "rows": []}])


# --- the determinism subset --------------------------------------------------------------------


def test_the_determinism_subset_is_seeded_sized_and_inside_the_clips():
    wide = [{**clip(index), "split": "clean", "shard_file": f"s{index // 10}", "row": index}
            for index in range(60)]
    keys = determinism_keys(wide, 10, seed=3)
    assert len(keys) == 10 and keys <= {item_key(c["audio_path"]) for c in wide}
    assert determinism_keys(list(reversed(wide)), 10, seed=3) == keys
    assert determinism_keys(wide, 10, seed=4) != keys

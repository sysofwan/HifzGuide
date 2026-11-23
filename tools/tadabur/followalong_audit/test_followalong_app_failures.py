"""A failed clip is a reported exclusion, never an abort or an S5 word (#309, S1.3).

The run records per-clip failures because some of 5,742 clips will fail. P2 and P3 must compare
the clips that produced rows and list the rest; the strata must exclude a failed clip's words
rather than fold them into S5, which means "the App graded the clip and left that word ungraded".
"""

from __future__ import annotations

from types import SimpleNamespace

from tadabur.followalong_audit import bridge
from tadabur.followalong_audit.cadence import p3_section
from tadabur.followalong_audit.strata_run import words_from_grades
from training.nonpro_decode import item_key

DETECTORS = ("base", "c95", "r2_candidate_ema", "r3_candidate_ema")


def clip(index: int) -> dict:
    return {"audio_path": f"{index}_1_{index}.flac", "audio_sha256": f"{index:064x}",
            "split": "clean", "shard_file": f"clean-{index:05d}-of-00411", "row": index,
            "filename_label": "1:2", "stratum": "gating", "group_key": f"g{index}"}


def cell(flagged: bool = False, graded: bool = True) -> dict:
    return {"graded": graded, "flagged": flagged, "causes": []}


def word_rows(item: str, flags: list[bool]) -> list[dict]:
    return [{"item": item, "word": index + 1, "strict": cell(flag),
             "shipped_default": cell()} for index, flag in enumerate(flags)]


# --- P3 -----------------------------------------------------------------------------------------


def test_a_clip_missing_from_one_latency_run_is_skipped_and_named():
    zero = word_rows("a#0", [False, False]) + word_rows("b#0", [False])
    other = word_rows("a#0", [True, False])
    report = p3_section(zero, other, device_ratios={"log": 1.5}, latencies=(0, 130), seed=7)
    assert (report["clips"], report["skipped"], report["seed"]) == (1, ["b#0"], 7)
    assert report["cadence"]["strict"]["words"] == 2
    assert report["cadence"]["strict"]["changed"] == 1


# --- P2 -----------------------------------------------------------------------------------------


def _grade_checked(clips, checks, harness):
    return [{"item": item_key(row["audio_path"]), "word": 1, "strict": cell(),
             "shipped_default": cell()} for row in clips]


def test_p2_draws_only_graded_clips_and_lists_the_failures(monkeypatch):
    population = [clip(index) for index in range(6)]
    failed = {item_key(population[5]["audio_path"])}
    grades = [{"item": item_key(row["audio_path"]), "word": 1, "strict": cell(),
               "shipped_default": cell()} for row in population[:5]]
    chosen = []

    def fake_grade(clips, checks, harness):
        chosen.extend(clips)
        return _grade_checked(clips, checks, harness)

    monkeypatch.setattr(bridge, "grade_checked_clips", fake_grade)

    def load_decodes():
        return {"gating": {"r3_candidate_ema": {}}}

    monkeypatch.setattr("tadabur.tlog_mining.candidate_inventory.load_decodes", load_decodes)
    report = bridge.p2_section(population, grades, harness=object(), subset_size=2, seed=1,
                               failed=failed)
    assert report["subset_clips"] == 2 and report["graded_clips"] == 5
    assert report["skipped"] == [item_key(population[5]["audio_path"])]
    assert all(item_key(row["audio_path"]) != item_key(population[5]["audio_path"])
               for row in chosen)
    assert report["subset"]["rate"] == 1.0


# --- the strata ---------------------------------------------------------------------------------


def grade(key: str, position: int, model: str, word: int, word_index: int, *,
          graded: bool = True, flagged: bool = False):
    return SimpleNamespace(item=SimpleNamespace(key=key), position=position, model=model,
                           word=word, word_index=word_index, graded=graded, flagged=flagged)


def four(key: str, position: int = 0, word: int = 1, word_index: int = 1) -> list:
    return [grade(key, position, model, word, word_index) for model in DETECTORS]


APP = [{"item": "a#0", "word": 1, "strict": cell(), "shipped_default": cell()}]


def test_a_failed_clips_words_are_excluded_and_counted_not_s5():
    words, meta = words_from_grades(four("a#0") + four("b#0"), APP, failed={"b#0"})
    assert [row["key"] for row in words] == ["a#0"]
    assert meta["excluded_failed_clips"] == 1
    assert meta["excluded_failed_words"] == 1


def test_an_oracle_table_fills_each_words_oracle_outcome():
    oracle = [{"item": "a#0", "word": 1, "strict": cell(flagged=True),
               "shipped_default": cell()}]
    words, _ = words_from_grades(four("a#0"), APP, oracle=oracle)
    assert words[0]["oracle"] == "flagged"
    assert words[0]["word_index"] == 1

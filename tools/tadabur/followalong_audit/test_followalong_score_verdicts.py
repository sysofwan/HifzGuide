"""The verdicts file the scorer reads and the PR3 UI writes (#311 PR2; prereg §4).

Committed before the PR2 code exists and **locked**: later turns may add tests but never weaken or
rewrite these.

Contract this file pins (``tadabur.followalong_audit.s5_score``)
-----------------------------------------------------------------
* ``VERDICTS = ("correct", "mistake", "not sure", "different text", "unusable audio")``,
  ``MISTAKE_KINDS = ("letter", "haraka", "shaddah", "skipped", "added")``,
  ``SUB_REASONS = ("not_in_clip",)``.
* ``s5_draw/verdicts.jsonl`` (``VERDICTS_PATH``): one line per answered served item (repeats have
  their own id)::

      {"id": str, "verdict": VERDICTS, "kinds": [MISTAKE_KINDS...], "sub_reason": null | "not_in_clip",
       "whole_played": bool, "note": str}

  ``kinds`` is non-empty, duplicate-free and from ``MISTAKE_KINDS`` exactly when the verdict is
  ``mistake`` (else ``[]``); ``sub_reason`` is only ever set on ``unusable audio``; the fields are
  exactly these. A file that breaks any rule, or answers one id twice, is refused (``ValueError``).
* ``read_verdicts(path, key) -> {id: row}``: a missing file is no verdicts (``{}``); an id the
  ``key`` rows do not hold is refused.
* ``agreement(key, verdicts) -> {"pairs", "agree"}``: over repeats whose copy and original both
  have a verdict, how many give the same top-level verdict (kinds and sub-reasons are ignored).
"""

from __future__ import annotations

import json

import pytest

from tadabur.followalong_audit.s5_score import (
    MISTAKE_KINDS,
    SUB_REASONS,
    VERDICTS,
    VERDICTS_PATH,
    agreement,
    read_verdicts,
)

KEY = [{"id": "aaaaaaaaaaaaaaa1", "repeat_of": None}, {"id": "aaaaaaaaaaaaaaa2", "repeat_of": None},
       {"id": "bbbbbbbbbbbbbbb1", "repeat_of": "aaaaaaaaaaaaaaa1"},
       {"id": "bbbbbbbbbbbbbbb2", "repeat_of": "aaaaaaaaaaaaaaa2"}]


def row(ident, verdict="correct", kinds=(), sub_reason=None, whole=False, note="") -> dict:
    return {"id": ident, "verdict": verdict, "kinds": list(kinds), "sub_reason": sub_reason,
            "whole_played": whole, "note": note}


def write(tmp_path, rows) -> object:
    path = tmp_path / "verdicts.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def test_the_vocabulary_is_the_preregistrations():
    assert VERDICTS == ("correct", "mistake", "not sure", "different text", "unusable audio")
    assert MISTAKE_KINDS == ("letter", "haraka", "shaddah", "skipped", "added")
    assert SUB_REASONS == ("not_in_clip",)
    assert VERDICTS_PATH.name == "verdicts.jsonl" and VERDICTS_PATH.parent.name == "s5_draw"


def test_every_valid_shape_is_read(tmp_path):
    rows = [row("aaaaaaaaaaaaaaa1", "mistake", ["letter", "added"], whole=True, note="x"),
            row("aaaaaaaaaaaaaaa2", "unusable audio", sub_reason="not_in_clip"),
            row("bbbbbbbbbbbbbbb1", "not sure"), row("bbbbbbbbbbbbbbb2", "different text")]
    assert read_verdicts(write(tmp_path, rows), KEY) == {r["id"]: r for r in rows}


def test_a_missing_file_is_no_verdicts(tmp_path):
    assert read_verdicts(tmp_path / "none.jsonl", KEY) == {}


@pytest.mark.parametrize("bad", [
    row("aaaaaaaaaaaaaaa1", "maybe"),
    row("aaaaaaaaaaaaaaa1", "mistake"),
    row("aaaaaaaaaaaaaaa1", "mistake", ["typo"]),
    row("aaaaaaaaaaaaaaa1", "mistake", ["letter", "letter"]),
    row("aaaaaaaaaaaaaaa1", "correct", ["letter"]),
    row("aaaaaaaaaaaaaaa1", "correct", sub_reason="not_in_clip"),
    row("aaaaaaaaaaaaaaa1", "unusable audio", sub_reason="noisy"),
    row("zzzzzzzzzzzzzzzz", "correct"),
    {**row("aaaaaaaaaaaaaaa1"), "extra": 1},
    {k: v for k, v in row("aaaaaaaaaaaaaaa1").items() if k != "whole_played"},
    {**row("aaaaaaaaaaaaaaa1"), "whole_played": "yes"},
])
def test_a_row_that_breaks_the_schema_is_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        read_verdicts(write(tmp_path, [bad]), KEY)


def test_one_id_answered_twice_is_refused(tmp_path):
    with pytest.raises(ValueError):
        read_verdicts(write(tmp_path, [row("aaaaaaaaaaaaaaa1"), row("aaaaaaaaaaaaaaa1", "mistake",
                                                                    ["letter"])]), KEY)


def test_agreement_counts_repeats_whose_two_answers_exist(tmp_path):
    verdicts = {r["id"]: r for r in [
        row("aaaaaaaaaaaaaaa1", "mistake", ["letter"]), row("bbbbbbbbbbbbbbb1", "mistake", ["haraka"]),
        row("aaaaaaaaaaaaaaa2", "correct"), row("bbbbbbbbbbbbbbb2", "not sure")]}
    assert agreement(KEY, verdicts) == {"pairs": 2, "agree": 1}


def test_a_repeat_with_no_answer_or_whose_original_is_unanswered_is_not_a_pair():
    verdicts = {"aaaaaaaaaaaaaaa1": row("aaaaaaaaaaaaaaa1"), "bbbbbbbbbbbbbbb2": row("bbbbbbbbbbbbbbb2")}
    assert agreement(KEY, verdicts) == {"pairs": 0, "agree": 0}
    assert agreement(KEY, {}) == {"pairs": 0, "agree": 0}

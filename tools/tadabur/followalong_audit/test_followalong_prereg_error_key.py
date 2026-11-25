"""``error_key`` sorts the engine's real error shapes without comparing ``None`` to ``str`` (#310).

The locked tests pin the documented tuple ``(group_index, kind.type, kind.expected, heard or
None)``; this pins the sort on shapes the committed traces already carry — ``expected: None`` on
tashkeel errors and ``heard: None`` — so a same-group pair with one ``None`` cannot abort F-ref.
"""

from __future__ import annotations

from tadabur.followalong_audit.attribution import error_key


def error(group_index: int, *, expected: str | None = None,
          heard: str | None = None) -> dict:
    return {"group_index": group_index,
            "kind": {"type": "tashkeel", "expected": expected, "heard": heard}}


def test_an_error_with_no_expected_value_sorts_before_one_with_a_value():
    listed = error(0, expected="ِ", heard=None)
    bare = error(0, expected=None, heard="ُ")
    assert error_key([listed, bare]) == error_key([bare, listed]) == (
        (0, "tashkeel", None, "ُ"), (0, "tashkeel", "ِ", None))


def test_the_key_keeps_none_and_distinguishes_different_expected_values():
    key = error_key([error(0, expected=None), error(1, expected="ِ", heard="ُ")])
    assert key == ((0, "tashkeel", None, None), (1, "tashkeel", "ِ", "ُ"))
    assert key != error_key([error(0, expected="ُ"), error(1, expected="ِ", heard="ُ")])

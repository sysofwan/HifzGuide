"""D5's pending+withheld pin against the canonical ``setting_checks`` (#310).

The audit's traces are produced by ``training.muraja_policy.word_traces``. D5 says a kept
``pending`` that carries a ``withheld`` value is ungraded, and that ``setting_checks`` returns
``()`` for such a word — it compares the kept quality with each status's ``withheld or quality``.
The attribution fixtures hand-write ``setting_checks`` empty, so this feeds the real function the
shape the harness returns, so nobody can "fix" one side only.
"""

from __future__ import annotations

from training.muraja_policy import setting_checks


def _check(quality: str, withheld: str | None = None, word: int = 3) -> dict:
    status = {"word": word, "quality": quality, "score": 0.9, "errors": []}
    return {"statuses": [{**status, **({"withheld": withheld} if withheld else {})}]}


def test_a_kept_pending_that_carries_a_withheld_grade_sets_no_check():
    kept = {"quality": "pending", "score": 0.9, "errors": [], "withheld": "minor"}
    assert setting_checks([_check("pending", withheld="minor")], 3, kept) == ()


def test_a_kept_grade_sets_every_check_whose_effective_quality_it_is():
    kept = {"quality": "correct", "score": 0.9, "errors": []}
    checks = [_check("pending", withheld="correct"), _check("correct")]
    assert setting_checks(checks, 3, kept) == (0, 1)

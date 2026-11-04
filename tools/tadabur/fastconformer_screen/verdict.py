"""#267's pre-registered screening rule: GO / PRIOR / STOP, thresholds frozen.

A screen, never a ship gate. The correct side and the catch side are never pooled; a great
catch side cannot buy back a failing correct side, nor the reverse.

* **STOP** when the correct side fails: Ship A correct sites flagged is more than the teacher's
  rate + 5 points, or decoys left unchanged are fewer than the teacher's - 5 points. (Each is a
  strict inequality: exactly five points passes.)
* **GO** when the correct side passes, Ship A mistakes flagged is at least the teacher's - 5,
  edits heard are at least the teacher's - 5, and there are at least three unique catches.
* **PRIOR** otherwise: the correct side passes but the catch side or the unique catches fail.
"""

from __future__ import annotations

from dataclasses import dataclass

GO, PRIOR, STOP = "GO", "PRIOR", "STOP"

#: The frozen margins, in percentage points, and the unique-catch bar.
CORRECT_MARGIN = 5.0
CATCH_MARGIN = 5.0
UNIQUE_MINIMUM = 3


@dataclass(frozen=True)
class Rates:
    """The four measures, in percent; the teacher's and FastConformer's are read alike."""

    mistakes_flagged: float
    correct_flagged: float
    edits_heard: float
    decoys_unchanged: float


def verdict(teacher: Rates, fastconformer: Rates, unique_catches: int) -> str:
    """The frozen rule's answer on the report's own rates."""
    if unique_catches < 0:
        raise ValueError(f"unique catches cannot be negative, not {unique_catches}")
    if (fastconformer.correct_flagged > teacher.correct_flagged + CORRECT_MARGIN
            or fastconformer.decoys_unchanged < teacher.decoys_unchanged - CORRECT_MARGIN):
        return STOP
    if (fastconformer.mistakes_flagged >= teacher.mistakes_flagged - CATCH_MARGIN
            and fastconformer.edits_heard >= teacher.edits_heard - CATCH_MARGIN
            and unique_catches >= UNIQUE_MINIMUM):
        return GO
    return PRIOR

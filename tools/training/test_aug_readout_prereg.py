"""The consonant round's pre-registration (#174) is frozen by its SHA-256.

The bars in ``docs/consonant-round-preregistration.md`` were owner-approved before any training of
the round. A later edit is a changed pre-registration, which needs a new dated document, never a
re-run; this pin makes such an edit fail CI until it is made deliberately.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PREREGISTRATION_PATH = (
    Path(__file__).parent.parent.parent / "docs" / "consonant-round-preregistration.md"
)
#: SHA-256 of the committed text, the owner-approved bars of 2026-10-22.
PREREGISTRATION_SHA256 = "daa611ab401f1350ef920b6801c9c439262579d4aced3f7be062a240cf7ed9d7"


def test_the_consonant_round_preregistration_is_the_pinned_document():
    digest = hashlib.sha256(PREREGISTRATION_PATH.read_bytes()).hexdigest()
    assert digest == PREREGISTRATION_SHA256, (
        "docs/consonant-round-preregistration.md changed: the bars are frozen, so a deliberate "
        "change needs a new dated pre-registration and its pin updated here"
    )

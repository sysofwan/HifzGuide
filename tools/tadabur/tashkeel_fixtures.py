"""Human adjudications of mined tashkeel sites (#60).

:mod:`training.tashkeel_worklist` mines positions where the base and fine-tuned checkpoints
disagree about a short vowel. This module stores what a listener decided the **reciter
actually said** there — the ground truth Tadabur does not ship and the reference cannot
supply, because the reference records the vowel the mushaf prescribes, not the one the
reciter produced.

The verdict vocabulary is deliberately about the *audio*, never about the models:

* ``fatha`` / ``damma`` / ``kasra`` — the colour the listener heard.
* ``none`` — the carrier was voiced but no short vowel was (an elision, a swallowed ending).
* ``unclear`` — the listener could not tell. Kept rather than discarded, so the share of
  sites nobody can adjudicate is visible instead of silently shrinking the denominator.

A site is scoreable only when the listener heard the *reference* colour: that is the case
where declining to mark the vowel is unambiguously the model's error rather than the
reciter's. :func:`tadabur.tashkeel_acceptance.compare` restricts to those.

There is no database. Adjudications are appended to a single JSONL keyed by ``site_id``
(:func:`training.tashkeel_worklist.site_id`), which is derived from clip/window/position
rather than from sampling order — so re-mining with a different seed or a different
candidate checkpoint resumes an audit already done.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

from training.tashkeel_eval import DAMMA, FATHA, KASRA

#: What the listener heard, keyed by the name the UI sends.
HEARD_VOWELS = {"fatha": FATHA, "damma": DAMMA, "kasra": KASRA}

#: The carrier was voiced but carried no short vowel.
NONE = "none"
#: The listener could not tell. Excluded from the comparison, counted in the report.
UNCLEAR = "unclear"
VERDICTS = frozenset(HEARD_VOWELS) | {NONE, UNCLEAR}


@dataclass(frozen=True)
class Adjudication:
    """One listener verdict on one mined site.

    ``clip_audio_filename`` and ``reference_index`` are stored alongside the opaque
    ``site_id`` purely so the file stays readable and greppable by a human; ``site_id`` is
    the key everything joins on.
    """

    site_id: str
    verdict: str
    clip_audio_filename: str
    reference_index: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"{self.verdict!r} is not a tashkeel verdict (expected one of "
                f"{sorted(VERDICTS)})."
            )

    @property
    def heard_vowel(self) -> str | None:
        """The short-vowel character heard, or ``None`` for ``none``/``unclear``."""
        return HEARD_VOWELS.get(self.verdict)


SCHEMA_FIELDS = tuple(f.name for f in fields(Adjudication))
REQUIRED_FIELDS = frozenset(
    f.name for f in fields(Adjudication) if f.default is MISSING
)


def read_adjudications(path: Path) -> dict[str, Adjudication]:
    """Every adjudication in ``path``, keyed by ``site_id``; empty when the file is absent.

    A repeated ``site_id`` keeps the **last** entry, so the file behaves as an append log
    and a listener correcting a verdict just re-submits it. Unknown or missing fields are a
    hard error: a verdict silently dropped for a schema drift would shrink one arm of a
    paired comparison without any visible failure.
    """
    if not path.is_file():
        return {}
    adjudications: dict[str, Adjudication] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        unknown = record.keys() - set(SCHEMA_FIELDS)
        missing = REQUIRED_FIELDS - record.keys()
        if unknown or missing:
            raise ValueError(
                f"{path}:{number} does not match the adjudication schema "
                f"(missing: {sorted(missing)}, unknown: {sorted(unknown)})."
            )
        entry = Adjudication(**record)
        adjudications[entry.site_id] = entry
    return adjudications


def write_adjudications(path: Path, adjudications: dict[str, Adjudication]) -> None:
    """Rewrite ``path`` with one adjudication per line, in stable ``site_id`` order.

    A full rewrite rather than an append keeps the file free of superseded verdicts, so its
    line count is the number of sites judged — the progress figure the UI shows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(asdict(adjudications[key]), ensure_ascii=False)
        for key in sorted(adjudications)
    ]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

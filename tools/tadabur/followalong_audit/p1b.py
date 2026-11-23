"""Gate P1b: the 30-Character overlap clip of Muraja's ``.log`` against the full overlap (#307).

Muraja's device log renders ``String(update.overlap.prefix(30))``
(``TranscribeLog.swift:43-46``), so a replay of that log sees 30 Characters where the engine
saw the whole overlap. The gate replays the same clips both ways and asks whether any reported
word's kept quality differs; the audit's answer is the full-overlap run, and this one exists
only to show what the ``.log`` loses.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from training.decoding import clusters

#: The Characters ``.log`` keeps of each overlap.
LOG_OVERLAP_CHARACTERS = 30


def clip_overlaps(checks: Iterable[Mapping], limit: int = LOG_OVERLAP_CHARACTERS) -> list[dict]:
    """``checks`` with each overlap cut to ``limit`` Swift Characters, and nothing else changed.

    The cut is on Characters (a consonant and its marks), as Swift's ``String.prefix`` counts
    them: ``clusters`` groups exactly that way, so a mark is never left on the wrong side. Hops
    are not clipped — the device log renders them in full.
    """
    return [
        {**check, "overlap": "".join(clusters(check["overlap"])[:limit])}
        for check in checks
    ]


def changed_decisions(full: Iterable[Mapping], clipped: Iterable[Mapping]) -> list[dict]:
    """The words whose kept quality differs between the two runs, in order.

    Rows are :func:`training.muraja_policy.word_traces` rows, one per graded word per scoring.
    A word the engine graded in one run and not the other is a decision difference, not an
    error: the missing side reports ``None``. (``word_traces`` emits a row only for words in
    the engine's ``final``, so an ungraded word simply has no row.)
    """
    full_q = {(row["item"], row["scoring"], row["word"]): row["quality"] for row in full}
    clipped_q = {(row["item"], row["scoring"], row["word"]): row["quality"] for row in clipped}
    return [
        {"item": item, "scoring": scoring, "word": word,
         "full": full_q.get((item, scoring, word)),
         "clipped": clipped_q.get((item, scoring, word))}
        for item, scoring, word in sorted(full_q.keys() | clipped_q.keys())
        if full_q.get((item, scoring, word)) != clipped_q.get((item, scoring, word))
    ]

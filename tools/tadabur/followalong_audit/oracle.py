"""The Oracle arm: the reference phonemes placed on a clip's teacher word times (#307).

Muraja's checks are cut on its transcriber's schedule: a 5 s window once one is pending, with
its hop the first second, a preview of the pending audio every 200 ms, and a closing flush.
The app arm fills those checks from what the model decoded; this arm fills them from the
reference a perfect teacher would emit, with each word's Characters spread over that word's
phase A alignment (``tadabur.tlog_mining.word_times``). It is
:func:`training.muraja_policy.single_decode_checks`' schedule over timed characters instead of
one even spread over the clip, so a word is heard when the teacher said it, not when an even
spread would put it.

A clip whose alignment failed has no word times. It falls back to the even spread and is
marked ``aligned: false``, so the table never passes one off as timed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from training.decoding import (
    FLUSH,
    HOP_SAMPLES,
    PREVIEW,
    SAMPLE_RATE,
    WINDOW,
    check_schedule,
    clusters,
    make_check,
)
from training.muraja_policy import single_decode_checks


@dataclass(frozen=True)
class OracleArm:
    """One clip's oracle checks, and whether they were placed on real word times."""

    checks: list[dict]
    aligned: bool


def _validate_offsets(reference: str, offsets: Sequence[int]) -> None:
    """Refuse offsets that do not partition ``reference`` into words."""
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(reference):
        raise ValueError(f"offsets {list(offsets)} do not partition the reference")
    if any(later < earlier for earlier, later in zip(offsets, offsets[1:])):
        raise ValueError(f"offsets {list(offsets)} are not ascending")


def _placed_characters(
    reference: str, offsets: Sequence[int], word_times: Sequence[tuple[float, float]]
) -> list[tuple[str, float]]:
    """Every Swift Character of ``reference`` and the second its word's alignment puts it at.

    A Character is a consonant with its marks (``clusters``), so a cut between two of them can
    never separate a haraka from its consonant — the property even spreading by scalar would
    lose. Within a word the Characters are centred evenly across its ``(start, end)``.
    """
    words = [reference[start:end] for start, end in zip(offsets, offsets[1:])]
    if len(word_times) != len(words):
        raise ValueError(f"the clip has {len(word_times)} word times for {len(words)} words")
    placed: list[tuple[str, float]] = []
    for word, (start_s, end_s) in zip(words, word_times):
        characters = clusters(word)
        if not characters:
            continue
        if end_s < start_s:
            raise ValueError(f"the word time {start_s}-{end_s} ends before it starts")
        step = (end_s - start_s) / len(characters)
        placed.extend((char, start_s + (index + 0.5) * step)
                      for index, char in enumerate(characters))
    return placed


def oracle_checks(
    reference: str,
    offsets: Sequence[int],
    word_times: Sequence[tuple[float, float]] | None,
    n_samples: int,
) -> OracleArm:
    """``reference`` cut onto the transcriber's schedule at ``word_times``.

    ``offsets`` partition ``reference`` (``len(word_times) + 1`` entries, word ``j`` spanning
    ``[offsets[j], offsets[j + 1])``); ``word_times`` is one ``(start_s, end_s)`` per word, or
    ``None`` for a clip whose alignment failed. Each check holds the Characters whose placed
    time falls in the audio it decodes; the closing flush holds the rest as its hop, so the
    hops commit the whole reference once.
    """
    _validate_offsets(reference, offsets)
    if word_times is None:
        return OracleArm(single_decode_checks(reference, n_samples), aligned=False)

    placed = _placed_characters(reference, offsets, word_times)

    def text(start: int, end: int) -> str:
        low, high = start / SAMPLE_RATE, end / SAMPLE_RATE
        return "".join(char for char, at in placed if low <= at < high)

    checks: list[dict] = []
    for span in check_schedule(n_samples):
        if span.kind == WINDOW:
            hop_end = span.start + HOP_SAMPLES
            check = make_check(WINDOW, text(span.start, hop_end), text(hop_end, span.end))
        elif span.kind == PREVIEW:
            check = make_check(PREVIEW, "", text(span.start, span.end))
        else:  # the schedule's stop is this approximation's closing flush (#123)
            check = make_check(FLUSH, text(span.start, span.end), "")
        if check is not None:
            checks.append(check)
    return OracleArm(checks, aligned=True)

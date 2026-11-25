"""Amendment A1's audio-coverage gate: does the Teacher's transcript hold the word? (#310).

Coverage is engine-free. The Teacher arm's whole-clip decode is its checks' concatenated ``hop``
(the ``overlap`` repeats text the previous check held). That text is aligned to the ayah's
reference phonemes with :func:`training.muraja_policy._alignment` — the same spaces-ignored,
three-character-anchor alignment the tracer places sites by — and a word is *covered* when an
anchored character aligns inside its ``phoneme_groups`` span. No engine grade decides coverage:
the App arm is the thing being audited, and the Teacher arm runs through the same engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from training.muraja_policy import _alignment


def teacher_text(row: Mapping) -> str:
    """The Teacher's whole-clip decode: the ``hop`` of every check, in order, concatenated."""
    return "".join(check["hop"] for check in row["checks"])


def covered_words(text: str, reference: str,
                  spans: Sequence[tuple[int, int]]) -> list[bool]:
    """Whether an anchored character of ``text`` aligns inside each ``(start, end)`` span.

    ``spans`` are code-point spans of ``reference`` as the ayah stores it, spaces counted;
    :func:`_alignment` maps a character of ``text`` to that same reference index.
    """
    mapping, anchored = _alignment(text, reference)
    positions = [mapping[index] for index in anchored]
    return [any(start <= position < end for position in positions)
            for start, end in spans]


def ayah_coverage(teacher_row: Mapping, text, surah: int, ayah: int,
                  words: Iterable[int]) -> dict[int, bool]:
    """``{word: covered}`` for the ayah's phoneme ``words``, from the Teacher's decode.

    ``text`` offers ``phonemes(surah, ayah)`` and ``word_groups(surah, ayah, word)`` as
    :class:`training.muraja_policy.MurajaText` does. A word's span runs from the first start to
    the last end of its groups.
    """
    words = sorted(words)
    reference = text.phonemes(surah, ayah)
    spans = []
    for word in words:
        groups = text.word_groups(surah, ayah, word)
        spans.append((min(start for start, _ in groups), max(end for _, end in groups)))
    return dict(zip(words, covered_words(teacher_text(teacher_row), reference, spans)))

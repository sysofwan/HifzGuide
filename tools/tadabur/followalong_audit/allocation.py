"""The integer arithmetic the pre-registration fixes before any draw (#310, D2 and D6).

No item is drawn here: :func:`allocate` says how many snips each cell gets, and
:func:`primary_cause` gives each S1 word the one cause whose cell it is allocated in. The draw
itself is S5's, as one seeded permutation per cell.

The pre-registered order is **not** ``training.word_flags.CAUSES``: consonant identity comes
first because decision rule 3 turns on soft and target pairs, and the owner's priority is misheard
consonants (D2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

#: The causes in the pre-registered precedence order (D2).
PRIMARY_CAUSES = ("target_pair", "soft_pair", "other_consonant", "shaddah", "tashkeel", "skipped",
                  "unrecorded")


def primary_cause(causes: Iterable[str]) -> str:
    """The first of :data:`PRIMARY_CAUSES` a word carries.

    A word can carry several causes; its cell is the first in the pre-registered order, whatever
    order the word lists them in. No cause, or a cause outside the seven, is refused: an unknown
    cause would silently drop a word from every cell.
    """
    causes = list(causes)
    unknown = [cause for cause in causes if cause not in PRIMARY_CAUSES]
    if not causes or unknown:
        raise ValueError(f"no primary cause for {causes!r}: {unknown or 'no cause'} is not in "
                         f"{PRIMARY_CAUSES}")
    return next(cause for cause in PRIMARY_CAUSES if cause in causes)


def _apportion(sizes: Mapping, total: int) -> dict:
    """``total`` over ``sizes`` in proportion, by largest remainder; ties to the larger pool,
    then to the smaller key."""
    pool = sum(sizes.values())
    exact = {key: total * size / pool for key, size in sizes.items()}
    split = {key: int(share) for key, share in exact.items()}
    order = sorted(sizes, key=lambda key: (-(exact[key] - split[key]), -sizes[key], key))
    for key in order[:total - sum(split.values())]:
        split[key] += 1
    return split


def allocate(sizes: Mapping, total: int, *, floor: int = 0) -> dict:
    """Split ``total`` over the cells of ``sizes`` (cell -> pool size), by largest remainder.

    Every cell gets at least ``floor`` (never more than its pool), and the allocations sum to
    exactly ``total``. A cell whose floor binds takes its floor and the other cells are
    reapportioned without it, so the floor is paid for by the cells above it; a cell whose pool
    is below ``floor`` takes its whole pool. A ``total`` above the pools' sum, or one the floors
    cannot pay, is refused.
    """
    if total < 0 or floor < 0:
        raise ValueError(f"a negative total ({total}) or floor ({floor}) allocates nothing")
    if any(size <= 0 for size in sizes.values()):
        raise ValueError(f"every cell needs a positive pool: {dict(sizes)}")
    if total > sum(sizes.values()):
        raise ValueError(f"total {total} is more than the pools' {sum(sizes.values())}")

    floors = {key: min(floor, size) for key, size in sizes.items()}
    split = dict.fromkeys(sizes, 0)
    open_cells = list(sizes)
    while True:
        remaining = total - sum(split[key] for key in sizes if key not in open_cells)
        if remaining < 0:
            raise ValueError(f"total {total} cannot pay the {floor}-snip floors")
        if not open_cells:
            if remaining:
                raise ValueError(f"total {total} cannot pay the {floor}-snip floors")
            return split
        share = _apportion({key: sizes[key] for key in open_cells}, remaining)
        below = [key for key in open_cells if share[key] < floors[key]]
        if not below:
            split.update(share)
            return split
        for key in below:
            split[key] = floors[key]
            open_cells.remove(key)

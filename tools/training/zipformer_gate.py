"""The zipformer's card-reproduction gate (#245, step 3): pure rules, no model, no download.

The gate stops the screen unless the zipformer reproduces its card's PER on the card's own
benchmark (:data:`GATE_LIMIT_POINTS`). Three rules are pinned before any decode:

* **The unit rule.** The card's tokenizer is unpublished, and ``phoneme_units.json`` is the
  inventory, not the segmentation. :func:`encode_units` stands in with greedy longest match,
  and :func:`require_roundtrip` justifies it: re-tokenising a clip's emitted units must
  reproduce them on at least :data:`ROUNDTRIP_FLOOR` of clips, else :class:`GateStop` and no
  other rule is tuned against it.
* **The scale rule.** The card does not say whether kaldi fbank expects [-1, 1] or
  int16-scaled samples. :func:`pin_scale` keeps the scale with the lower PER on every
  benchmark source; sources that disagree, or a scale within :data:`SCALE_GAP_POINTS` on any
  source, stop the gate rather than letting the decode pick a scale after the fact.
* **Table misses.** A clip whose reference text is not in the card's table has no gold units;
  :func:`source_per` drops it from PER and counts it, as the card's own script does.
* **The card.** Once the three rules pass, :func:`require_card_reproduction` compares the run's
  per-source unit PER against the card's own (:data:`CARD_PER`) and stops outside
  :data:`GATE_LIMIT_POINTS`, or when a card source was not scored. A screen that is nowhere
  near the card is a broken decoder, not a comparison.

PER is pooled over clips per source (total edits over total reference units), never an average of
per-clip rates. :func:`source_per` is the card's unit PER; :func:`character_per` is the
unit-free character-level PER beside it, which the report carries for Q4.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

#: How much worse than the card a run may be, in PER points, before the screen stops.
GATE_LIMIT_POINTS = 0.5
#: The card's own unit PER by source (card "Results", v3.2 on quranic-asr-benchmark v1.1).
CARD_PER = {"studio": 1.43, "phone": 3.65, "unseen": 9.10}
#: The share of clips whose emitted units must re-tokenise to themselves.
ROUNDTRIP_FLOOR = 0.99
#: A scale must win a source by more than this many PER points to count for that source.
SCALE_GAP_POINTS = 2.0
#: The card's inventory key that is not a unit.
BLANK_UNIT = "<blank>"


class GateStop(RuntimeError):
    """The gate refused to go on: its inputs cannot support the claim it would make."""


def encode_units(text: str, units: Iterable[str]) -> list[str]:
    """Segment ``text`` into the card's units by greedy longest match, whitespace ignored.

    A character no unit can start with is refused rather than dropped: dropping it would
    score the model against a reference it never said.
    """
    inventory = sorted(
        {unit for unit in units if unit and unit != BLANK_UNIT},
        key=lambda unit: (-len(unit), unit),
    )
    encoded: list[str] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        for unit in inventory:
            if text.startswith(unit, index):
                encoded.append(unit)
                index += len(unit)
                break
        else:
            raise ValueError(f"no unit holds {text[index]!r} at index {index} in {text!r}")
    return encoded


def roundtrip_identical(emitted: Sequence[str], units: Iterable[str]) -> bool:
    """Whether re-tokenising a clip's emitted units gives back exactly those units."""
    return encode_units("".join(emitted), units) == list(emitted)


def require_roundtrip(clips: Iterable[Sequence[str]], units: Iterable[str],
                      minimum: float = ROUNDTRIP_FLOOR) -> float:
    """The round trip's share of ``clips``, or :class:`GateStop` below ``minimum``.

    ``clips`` is each clip's emitted units; no clips at all is an error, not a pass.
    """
    inventory = list(units)
    total = 0
    identical = 0
    for clip in clips:
        total += 1
        identical += roundtrip_identical(clip, inventory)
    if not total:
        raise ValueError("the round trip needs clips; an empty set proves nothing")
    share = identical / total
    if share < minimum:
        raise GateStop(
            f"only {identical} of {total} clips ({share:.4f}) re-tokenise to their emitted "
            f"units, below {minimum:.4f}: the longest-match stand-in is not the card's "
            "tokenizer, so no PER against it can be trusted"
        )
    return share


def edit_counts(reference: Sequence, hypothesis: Sequence) -> tuple[int, int]:
    """Levenshtein edits of ``hypothesis`` against ``reference``, and the reference's length."""
    reference = list(reference)
    hypothesis = list(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for row, ref in enumerate(reference, 1):
        current = [row]
        for column, hyp in enumerate(hypothesis, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (ref != hyp),
            ))
        previous = current
    return previous[-1], len(reference)


def _pooled_per(rows: Iterable[tuple[str, Sequence | None, Sequence]]) -> dict[str, dict]:
    """Pool edits over reference lengths per ``(source, gold, pred)``, counting ``gold`` None."""
    totals: dict[str, dict[str, int]] = {}
    for source, gold, pred in rows:
        entry = totals.setdefault(source, {"edits": 0, "units": 0, "clips": 0, "misses": 0})
        if gold is None:
            entry["misses"] += 1
            continue
        edits, units = edit_counts(gold, pred)
        entry["edits"] += edits
        entry["units"] += units
        entry["clips"] += 1
    return {
        source: {
            "per": 100.0 * entry["edits"] / entry["units"] if entry["units"] else None,
            "clips": entry["clips"],
            "misses": entry["misses"],
        }
        for source, entry in totals.items()
    }


def source_per(rows: Iterable[Mapping]) -> dict[str, dict]:
    """Pooled unit PER per source over ``rows`` (``source``, ``gold``, ``pred``).

    ``gold`` is ``None`` when the clip's reference text is not in the card's table: the clip
    is a miss, excluded from PER. A source left with no reference units has ``per`` ``None``.
    """
    return _pooled_per(
        (row["source"], row["gold"], row["pred"]) for row in rows)


def character_per(rows: Iterable[Mapping]) -> dict[str, dict]:
    """Pooled character PER beside :func:`source_per`: the units concatenated, codepoint edits.

    Same rows and same miss rule; the reference length is its characters, so no unit boundary
    can hide or create an error.
    """
    return _pooled_per(
        (row["source"],
         None if row["gold"] is None else "".join(row["gold"]),
         "".join(row["pred"]))
        for row in rows
    )


def require_card_reproduction(per: Mapping[str, Mapping], card: Mapping[str, float] = CARD_PER,
                              limit: float = GATE_LIMIT_POINTS) -> None:
    """Stop unless every card source's measured unit PER is within ``limit`` of the card's.

    ``per`` is :func:`source_per`'s output. A card source the run did not score, or scored with
    no reference units, stops too: a reproduction gate cannot pass on the sources it skipped.
    The boundary is inclusive -- exactly ``limit`` off is within it.
    """
    for source, expected in card.items():
        if source not in per:
            raise GateStop(
                f"the card reports {source} PER {expected}, but the run scored no {source} "
                "source; the gate cannot reproduce what it did not measure"
            )
        measured = per[source]["per"]
        if measured is None:
            raise GateStop(f"the run scored no reference units on {source}")
        gap = abs(measured - expected)
        if gap > limit:
            raise GateStop(
                f"{source}: the run's PER {measured:.2f} is {gap:.2f} points from the card's "
                f"{expected} (limit ±{limit}); this is not the card's model on this audio"
            )


def pin_scale(per: Mapping[float, Mapping[str, float | None]]) -> float:
    """The sample scale with the lower PER on every source, or :class:`GateStop`.

    ``per`` maps each candidate scale to its PER by source. Exactly two candidates are
    considered: the choice itself is binary ([-1, 1] or int16), and a third would need its own
    pre-declared rule.
    """
    scales = sorted(per)
    if len(scales) != 2:
        raise ValueError(f"the scale rule decides between two scales, got {scales}")
    low, high = scales
    sources = set(per[low])
    if not sources or set(per[high]) != sources:
        raise ValueError(
            f"the scales are measured on different sources: {sorted(per[low])} vs "
            f"{sorted(per[high])}"
        )
    winners: set[float] = set()
    for source in sorted(sources):
        low_per, high_per = per[low][source], per[high][source]
        if low_per is None or high_per is None:
            raise GateStop(f"{source} has no PER for one of the scales")
        gap = high_per - low_per
        if gap > SCALE_GAP_POINTS:
            winners.add(low)
        elif gap < -SCALE_GAP_POINTS:
            winners.add(high)
        else:
            raise GateStop(
                f"{source}: {low} and {high} are within {SCALE_GAP_POINTS} PER points "
                f"({low_per:.2f} vs {high_per:.2f}), so neither scale is pinned"
            )
    if len(winners) != 1:
        raise GateStop(f"the scales disagree by source on {sorted(sources)}: {sorted(winners)}")
    return winners.pop()

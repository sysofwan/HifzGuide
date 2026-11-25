"""The blind ids and the serve orders with their repeats (#311, draw amendment 1 §1 and §4).

Committed before ``tadabur/followalong_audit/draw.py`` exists and **locked**: later turns may add
tests but never weaken or rewrite these.

Contract this file pins (``tadabur.followalong_audit.draw``)
-----------------------------------------------------------
* ``item_id(seed, key, word) -> str``: the page's opaque id, 16 lowercase hex characters, a salted
  hash of the word's clip key and phoneme word (the salt is ``subseed(seed, "ids")``): unique per
  word, reproducible from the seed, and a function of nothing else — not the stratum, a cell, an
  arm or a weight.
* ``repeat_id(seed, item_id) -> str``: the id a re-ask of ``item_id`` is served under: 16 lowercase
  hex characters, different from ``item_id``, one per original, reproducible, and as opaque as an
  original's.
* ``plan_sitting(items, *, seed, sitting, repeats) -> list[Served]``: ``items`` are original ids
  (any order), ``sitting`` is ``"pilot"`` or ``"main"``. ``Served`` has ``.item`` (the original's id
  for both an original and its repeat) and ``.repeat`` (bool). The originals appear once each, in
  ``seeded_order(items, seed, f"order:{sitting}")``; ``repeats`` of them, drawn from the first
  ``len(items) - 20`` positions of that order (130 of the pilot's 150), appear a second time at a
  later position at least 20 places after the original, the choices seeded by
  ``f"repeats:{sitting}"``. The sitting is ``len(items) + repeats`` long.
"""

from __future__ import annotations

import re

import pytest

from tadabur.followalong_audit.draw import (
    item_id,
    plan_sitting,
    repeat_id,
    seeded_order,
)

SEED = 9665042067186208708
HEX16 = re.compile(r"^[0-9a-f]{16}$")


def ids(count: int) -> list[str]:
    return [item_id(SEED, f"{number // 5}_{number}.flac#0", number % 5 + 1)
            for number in range(count)]


# --- the ids ------------------------------------------------------------------------------------
def test_an_item_id_is_sixteen_lowercase_hex_characters():
    assert HEX16.match(item_id(SEED, "100_1_3073128032.flac#0", 4))


def test_an_item_id_is_reproducible_and_distinguishes_clip_and_word():
    first = item_id(SEED, "100_1_3073128032.flac#0", 4)
    assert item_id(SEED, "100_1_3073128032.flac#0", 4) == first
    assert item_id(SEED, "100_1_3073128032.flac#0", 5) != first
    assert item_id(SEED, "100_2_3073128032.flac#0", 4) != first


def test_an_item_id_depends_on_the_seed():
    assert item_id(SEED + 1, "100_1_3073128032.flac#0", 4) != item_id(
        SEED, "100_1_3073128032.flac#0", 4)


def test_the_ids_of_many_words_do_not_collide():
    words = [item_id(SEED, f"{n}_1_1.flac#0", w) for n in range(300) for w in range(1, 11)]
    assert len(set(words)) == 3000


def test_a_repeat_id_is_opaque_distinct_and_one_per_original():
    original = item_id(SEED, "100_1_3073128032.flac#0", 4)
    again = repeat_id(SEED, original)
    assert HEX16.match(again) and again != original
    assert repeat_id(SEED, original) == again
    assert repeat_id(SEED, item_id(SEED, "100_1_3073128032.flac#0", 5)) != again
    assert repeat_id(SEED + 1, original) != again


# --- the sitting --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pilot():
    items = ids(150)
    return items, plan_sitting(items, seed=SEED, sitting="pilot", repeats=7)


@pytest.fixture(scope="module")
def main():
    items = ids(390)
    return items, plan_sitting(items, seed=SEED, sitting="main", repeats=18)


def test_the_pilot_is_157_served_items_and_the_main_sitting_408(pilot, main):
    assert len(pilot[1]) == 157
    assert len(main[1]) == 408


def test_the_originals_are_the_items_in_their_seeded_order(pilot, main):
    for (items, plan), sitting in ((pilot, "pilot"), (main, "main")):
        originals = [served.item for served in plan if not served.repeat]
        assert originals == seeded_order(items, SEED, f"order:{sitting}")


def test_the_repeats_are_distinct_originals_each_served_a_second_time(pilot, main):
    for (items, plan), count in ((pilot, 7), (main, 18)):
        repeated = [served.item for served in plan if served.repeat]
        assert len(repeated) == count == len(set(repeated))
        assert set(repeated) <= set(items)


def test_every_repeat_follows_its_original_by_at_least_twenty_places(pilot, main):
    for _, plan in (pilot, main):
        for place, served in enumerate(plan):
            if not served.repeat:
                continue
            original = next(index for index, other in enumerate(plan)
                            if other.item == served.item and not other.repeat)
            assert place - original >= 20


def test_a_pilot_repeat_re_asks_one_of_the_first_130_originals(pilot):
    items, plan = pilot
    order = seeded_order(items, SEED, "order:pilot")
    for served in plan:
        if served.repeat:
            assert order.index(served.item) < 130


def test_a_main_repeat_re_asks_an_original_with_room_for_its_copy(main):
    items, plan = main
    order = seeded_order(items, SEED, "order:main")
    for served in plan:
        if served.repeat:
            assert order.index(served.item) < len(items) - 20


def test_the_plan_is_reproducible_from_the_seed_alone(pilot):
    items, plan = pilot
    assert plan_sitting(list(reversed(items)), seed=SEED, sitting="pilot", repeats=7) == plan
    assert plan_sitting(items, seed=SEED + 1, sitting="pilot", repeats=7) != plan


def test_the_two_sittings_order_and_repeat_independently():
    items = ids(60)
    pilot_plan = plan_sitting(items, seed=SEED, sitting="pilot", repeats=3)
    main_plan = plan_sitting(items, seed=SEED, sitting="main", repeats=3)
    assert pilot_plan != main_plan


def test_a_sitting_too_short_to_leave_twenty_places_is_refused():
    with pytest.raises(ValueError):
        plan_sitting(ids(20), seed=SEED, sitting="pilot", repeats=1)
    with pytest.raises(ValueError):
        plan_sitting(ids(60), seed=SEED, sitting="pilot", repeats=41)


def test_an_unknown_sitting_is_refused():
    with pytest.raises(ValueError):
        plan_sitting(ids(60), seed=SEED, sitting="later", repeats=1)

"""The character-start margin oracle (architect r18/r19, S5b #204).

The pinned competitor is a word-span search and ``docs/tlog-ident-margin-amendment.md``
discloses that it can overstate ``m``, never understate it. These tests pin the measured bound:
it reaches the different-class location the word-span search cannot (R4.1's two fixtures), it
never raises a margin, and :func:`tadabur.tlog_ident.oracle_pass` aggregates it without letting
it choose ``b_u``.

Test command, from ``tools/``::

    python -m pytest -q tadabur/test_tlog_ident_oracle.py
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="session")
def ident():
    return importlib.import_module("tadabur.tlog_ident")


@pytest.fixture
def shared_word(ident):
    """1:1 and 1:2 are one word each and share a class; 1:3 is its own class and starts inside
    that word when the decode begins four characters in (R4.1's fixture)."""
    return ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 2, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 3, "بتثج", (0, 1, 2, 3, 4)),
    ])


def test_the_oracle_reaches_the_location_inside_a_shared_word(ident, shared_word):
    """R4.1: the second perfect location starts at character 4, inside 1:2's single word, and
    belongs to 1:3 by aligned words. No word span can begin there, so the pin reports 0.5."""
    pinned = ident.identify("بتثجبتثج", database=shared_word)
    assert pinned.margin == pytest.approx(0.5)
    assert pinned.margin_class == shared_word.text_class(1, 3)
    assert ident.oracle_margin(shared_word, "بتثجبتثج", pinned=pinned.margin) == pytest.approx(
        0.0)


def test_the_oracle_finds_the_worse_location_inside_a_shared_word(ident):
    """R4.1's second fixture: the only different-class location scores 0.875, so the margin is
    at most 0.125 where the pin reports 0.625."""
    db = ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 2, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 3, "بتثح", (0, 1, 2, 3, 4)),
    ])
    pinned = ident.identify("بتثجبتثج", database=db)
    assert pinned.margin == pytest.approx(0.625)
    assert ident.oracle_margin(db, "بتثجبتثج", pinned=pinned.margin) == pytest.approx(0.125)


def test_the_oracle_never_raises_the_pinned_margin(ident, shared_word, monkeypatch):
    """The bound is ``min(pinned, best − char competitor)``: a competitor worse than the pinned
    one leaves the margin exactly as the pin reports it."""
    class Worse:
        score = 0.25
        text_class = 0

    monkeypatch.setattr(shared_word, "char_competitor", lambda *args, **kwargs: Worse())
    pinned = ident.identify("بتثجبتثج", database=shared_word)
    assert ident.oracle_margin(shared_word, "بتثجبتثج",
                               pinned=pinned.margin) == pytest.approx(pinned.margin)


def test_the_oracle_pass_records_the_bound_and_keeps_the_pinned_b_u(ident):
    """The artifact's ``b_u`` is re-derived from the pinned margins alone; the oracle's own
    re-derivation is recorded beside it, and the changed rows carry both margins."""
    db = ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 2, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 3, "بتثج", (0, 1, 2, 3, 4)),
    ])

    def decode(key, surah_ayah, spans):
        return ident.Decode(key, f"{key}.wav", surah_ayah, 1, 1, 0.0, spans)

    look = [decode("a", "1:1", "بتثجبتثج"), decode("b", "1:3", "بتثج")]
    payload = ident.oracle_pass(db, look, [], progress=lambda _: None)

    rows = [ident.identify(row.spans, database=db, filename_ayah=row.surah_ayah) for row in look]
    pinned = ident.derive_bars(rows, [])
    assert payload["b_u"] == pinned["b_u"]
    assert payload["b"] == pinned["b"]
    assert payload["clips"] == len(look)
    assert payload["clips_m_changed"] == len(payload["changed"])
    assert payload["assignments_changed_at_b_u"] == sum(
        item["crosses_b_u"] for item in payload["changed"])
    assert payload["assignments_at_b_u"]["pinned"] == payload["assignments_at_b_u"]["oracle"]
    for item in payload["changed"]:
        assert item["oracle_margin"] < item["pinned_margin"]
        assert item["assigned_pinned"] == (item["pinned_margin"] >= payload["b_u"])
    assert payload["definition"]["path"] == "docs/tlog-ident-margin-amendment.md"
    assert payload["share_m_changed"] == pytest.approx(
        payload["clips_m_changed"] / payload["clips"])

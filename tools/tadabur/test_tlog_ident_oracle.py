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
import json

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


@pytest.fixture
def mixed(ident):
    """Three clips over six ayat, one whose margin the bound lowers and two it leaves alone
    (reviewer r1's fixture). The changed clip is first, so a one-based index into the
    zero-based row list reports the second clip's margin to it."""
    return ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 2, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 3, "بتثج", (0, 1, 2, 3, 4)),
        ident.AyahReference(2, 1, "دذرزسطظع", (0, 8)),
        ident.AyahReference(2, 2, "دذرزسطظعغ", (0, 9)),
        ident.AyahReference(3, 1, "غفقكلمنه", (0, 8)),
    ])


def _decode(ident, key, surah_ayah, spans):
    return ident.Decode(key, f"{key}.wav", surah_ayah, 1, 1, 0.0, spans)


def test_each_changed_row_carries_its_own_margins_and_crossing(ident, mixed):
    """r1: `enumerate(..., 1)` into a zero-based row list gave the changed clip the *next*
    clip's bounded margin and counted 0 crossings where one assignment moves. The fixture's
    totals fall from 2 to 1 and the mechanical consequence is therefore ``owner``."""
    look = [_decode(ident, "changed", "1:1", "بتثجبتثج"),
            _decode(ident, "unchanged", "3:1", "غفقكلمنه"),
            _decode(ident, "wrong_ambiguous", "1:3", "دذرزسطظع")]
    payload = ident.oracle_pass(mixed, look, [], progress=lambda _: None)

    pinned = {item.key: ident.identify(item.spans, database=mixed,
                                       filename_ayah=item.surah_ayah) for item in look}
    assert pinned["changed"].margin == pytest.approx(0.5)
    assert payload["assignments_at_b_u"] == {"pinned": 2, "oracle": 1}
    assert payload["clips_m_changed"] == len(payload["changed"]) == 1
    assert payload["assignments_changed_at_b_u"] == 1

    item, = payload["changed"]
    assert item["key"] == "changed"
    assert item["pinned_margin"] == pytest.approx(pinned["changed"].margin)
    assert item["oracle_margin"] == pytest.approx(
        ident.oracle_margin(mixed, "بتثجبتثج", pinned=pinned["changed"].margin)) == 0.0
    assert item["oracle_margin"] < item["pinned_margin"]
    assert (item["assigned_pinned"], item["assigned_oracle"]) == (True, False)
    assert item["crosses_b_u"] is True
    assert ident.oracle_consequence(payload["b_u"], payload) == "owner"


def test_a_changed_last_row_is_aggregated_not_indexed_past(ident, mixed):
    """The same bound as a two-row population whose changed clip is last: the old index read
    past the list (IndexError) instead of recording the row."""
    look = [_decode(ident, "good", "3:1", "غفقكلمنه"),
            _decode(ident, "last", "1:1", "بتثجبتثج")]
    payload = ident.oracle_pass(mixed, look, [], progress=lambda _: None)

    item, = payload["changed"]
    assert item["key"] == "last"
    assert (item["pinned_margin"], item["oracle_margin"]) == pytest.approx((0.5, 0.0))
    assert item["crosses_b_u"] is False  # b_u is 0.0 here, so 0.0 still assigns
    assert payload["assignments_changed_at_b_u"] == sum(
        changed["crosses_b_u"] for changed in payload["changed"])


def test_the_committed_oracle_recorded_no_changed_row(ident):
    """The r1 index bug could only reach a changed row's evidence; ``b_u_oracle`` never read
    that list. The committed artifact has no changed row, so its recorded zero stands."""
    artifact = ident.oracle_artifact(ident.K)
    assert artifact["clips_m_changed"] == 0 == len(artifact["changed"])
    assert artifact["assignments_changed_at_b_u"] == 0
    assert artifact["assignments_at_b_u"]["pinned"] == artifact["assignments_at_b_u"]["oracle"]


def test_the_oracle_artifact_is_refused_when_a_recorded_input_is_stale(ident, tmp_path):
    """`b_u_oracle` is trusted only for the inputs the artifact names: the margin definition,
    the look decodes, their split, the real-mistake clips, and — from self-check r1 — its k."""
    base = json.loads(ident.ORACLE_PATH.read_text(encoding="utf-8"))
    assert ident.oracle_artifact(ident.K)["b_u_oracle"] == base["b_u_oracle"]

    def written(mutate):
        copy = json.loads(json.dumps(base))
        mutate(copy)
        path = tmp_path / "oracle.json"
        path.write_text(json.dumps(copy), encoding="utf-8")
        return path

    stale = [
        lambda a: a["definition"].__setitem__("sha256", "0" * 64),
        lambda a: a["population"]["decodes"].__setitem__("sha256", "0" * 64),
        lambda a: a["population"]["split"].__setitem__("sha256", "0" * 64),
        lambda a: a["population"]["mistakes"].__setitem__("sha256", "0" * 64),
        lambda a: a.__setitem__("k", ident.K * 2),
    ]
    for mutate in stale:
        with pytest.raises(ValueError):
            ident.oracle_artifact(ident.K, path=written(mutate))

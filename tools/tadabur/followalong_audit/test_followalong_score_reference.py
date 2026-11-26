"""The oracle's canonical text around an engine word, for F1-align (#311 PR2, architect r7).

Committed before the PR2 code exists and **locked**: later turns may add tests but never weaken or
rewrite these. They sit beside the locked ``test_followalong_prereg_attribution.py``, which pins
how ``attribution._aligned`` treats a reference it is given; these pin where that reference comes
from (prereg §5 l.284-288: "the word's reference R joined with a neighbour as the oracle text
joins them", ``sweep._request``: ``"".join(words)``).

Contract this file pins
-----------------------
``tadabur.followalong_audit.sweep``: ``AyahReference`` (the public name of ``_AyahReference``, same
fields: ``words``, ``engine_words``, ``word_of``, ``cluster_groups``, ``waqf``) and
``ayah_references(text) -> (label -> AyahReference)``, the cached accessor the oracle arm's own
``_References`` is built from (``text`` is a :class:`training.muraja_policy.MurajaText`).

``tadabur.followalong_audit.s5_arms.oracle_reference(reference, word)``, ``word`` a 1-based
engine word:

* ``None`` when no canonical word is mapped to ``word`` (``word_of``): F1-align cannot fire;
* else ``{"previous", "word", "next"}``: each the canonical words mapped to ``word - 1``, ``word``
  and ``word + 1``, concatenated in order with no separator; ``previous``/``next`` are ``None``
  at the ayah's first/last engine word, and when no canonical word is mapped to that neighbour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tadabur.followalong_audit.s5_arms import oracle_reference
from tadabur.followalong_audit.sweep import AyahReference, ayah_references
from training.muraja_policy import MurajaText

DB = Path(__file__).resolve().parents[3] / "data" / "quran.db"


def reference(words, word_of, engine_count=None) -> AyahReference:
    count = engine_count or max(word_of)
    return AyahReference(words=tuple(words), engine_words=tuple(f"e{n}" for n in range(count)),
                         word_of=tuple(word_of), cluster_groups=(), waqf=frozenset())


def test_a_plain_word_is_its_canonical_word_with_its_neighbours():
    ayah = reference(["aaa", "bbb", "ccc"], [1, 2, 3])
    assert oracle_reference(ayah, 2) == {"previous": "aaa", "word": "bbb", "next": "ccc"}


def test_the_ayahs_first_word_has_no_previous_and_its_last_no_next():
    ayah = reference(["aaa", "bbb", "ccc"], [1, 2, 3])
    assert oracle_reference(ayah, 1) == {"previous": None, "word": "aaa", "next": "bbb"}
    assert oracle_reference(ayah, 3) == {"previous": "bbb", "word": "ccc", "next": None}


def test_a_one_word_ayah_has_neither_neighbour():
    assert oracle_reference(reference(["aaaa"], [1]), 1) == {
        "previous": None, "word": "aaaa", "next": None}


def test_a_wasl_merged_engine_word_joins_its_canonical_words_without_a_separator():
    # canonical words 2 and 3 are recited in engine word 2
    ayah = reference(["aaa", "bbb", "ccc", "ddd"], [1, 2, 2, 3])
    assert oracle_reference(ayah, 2) == {"previous": "aaa", "word": "bbbccc", "next": "ddd"}
    assert oracle_reference(ayah, 1) == {"previous": None, "word": "aaa", "next": "bbbccc"}
    assert oracle_reference(ayah, 3) == {"previous": "bbbccc", "word": "ddd", "next": None}


def test_a_merged_neighbour_is_the_whole_merged_text_not_one_canonical_word():
    ayah = reference(["aaa", "bbb", "ccc", "ddd", "eee"], [1, 2, 2, 2, 3])
    assert oracle_reference(ayah, 3)["previous"] == "bbbcccddd"


def test_the_text_is_the_canonical_spelling_not_the_engines():
    # a pausal ta marbuta: the engine spells the last word differently from the phonetizer
    ayah = AyahReference(words=("aaa", "bbbه"), engine_words=("aaa", "bbbت"), word_of=(1, 2),
                         cluster_groups=(), waqf=frozenset())
    assert oracle_reference(ayah, 2) == {"previous": "aaa", "word": "bbbه", "next": None}


def test_an_engine_word_no_canonical_word_maps_to_has_no_reference():
    ayah = reference(["aaa", "bbb"], [1, 3], engine_count=3)
    assert oracle_reference(ayah, 2) is None


def test_a_neighbour_no_canonical_word_maps_to_is_none():
    ayah = reference(["aaa", "bbb"], [1, 3], engine_count=3)
    assert oracle_reference(ayah, 1) == {"previous": None, "word": "aaa", "next": None}
    assert oracle_reference(ayah, 3) == {"previous": None, "word": "bbb", "next": None}


def test_a_word_outside_the_ayah_is_refused():
    ayah = reference(["aaa", "bbb"], [1, 2])
    for word in (0, 3):
        with pytest.raises(ValueError):
            oracle_reference(ayah, word)


# --- the public accessor over the real text ---------------------------------------------------
@pytest.fixture(scope="module")
def references():
    return ayah_references(MurajaText(DB))


@pytest.mark.parametrize("label", ["1:1", "1:7", "2:255", "36:1", "112:1", "2:282"])
def test_the_accessor_maps_every_canonical_word_to_an_engine_word_in_order(references, label):
    ayah = references(label)
    assert len(ayah.word_of) == len(ayah.words)
    assert ayah.word_of[0] == 1
    assert list(ayah.word_of) == sorted(ayah.word_of)
    assert max(ayah.word_of) <= len(ayah.engine_words)


def test_the_accessor_caches_an_ayah(references):
    assert references("1:1") is references("1:1")


def test_a_real_ayahs_references_join_back_to_the_oracles_text(references):
    ayah = references("1:2")
    joined = "".join(oracle_reference(ayah, word)["word"]
                     for word in range(1, len(ayah.engine_words) + 1)
                     if oracle_reference(ayah, word) is not None)
    assert joined == "".join(ayah.words)

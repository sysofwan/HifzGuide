"""Planted mistakes (#308): each kind changes exactly one word, as specified.

``plant`` edits an ayah's spaceless phoneme words before the oracle places them on the clock
(``oracle_checks`` refuses a reference its checks do not commit), so the edit is a pure function
of the words and a seeded ``random.Random``. A Character is a consonant with its marks
(:func:`training.decoding.clusters`); every edit below is stated on Characters.
"""

from __future__ import annotations

import random

import pytest

from tadabur.followalong_audit.plant import (
    KINDS,
    PROTECTED_LETTERS,
    plant,
    plantable_words,
    planting_seed,
)
from tadabur.truth_sites import CONSONANTS, HARAKA_CHARS, TARGET_PAIRS
from training.decoding import clusters
from training.nonpro_eval import mushaf_form

#: 1:2, as ``hafs_word_reference`` spells it: no spaces, geminates doubled, madd as repeated letters.
WORDS = ("ءَلحَمدُ", "لِللَااهِ", "رَببِ", "لعَاالَمِۦۦن")
HARAKAT = frozenset(HARAKA_CHARS.values())
SEEDS = range(40)


def changed_clusters(before: str, after: str) -> list[tuple[str, str]]:
    """The ``(old, new)`` Characters that differ, for two words with the same Character count."""
    assert len(clusters(before)) == len(clusters(after))
    return [(old, new) for old, new in zip(clusters(before), clusters(after)) if old != new]


def base(character: str) -> str:
    return character[0]


def marks(character: str) -> str:
    return character[1:]


def only_changed_word(original: tuple[str, ...], planted) -> tuple[str, str]:
    """Assert every word but ``planted.word`` is untouched; the (old, new) of the one that moved."""
    assert len(planted.words) == len(original)
    differing = [i for i, (a, b) in enumerate(zip(original, planted.words)) if a != b]
    assert differing == [planted.word]
    return original[planted.word], planted.words[planted.word]


def test_the_kinds_are_each_target_pair_then_the_six_others():
    assert KINDS == (
        *(f"consonant_pair:{pair}" for pair in sorted(TARGET_PAIRS)),
        "consonant_random", "haraka_swap", "haraka_drop", "shaddah_drop", "skip_word", "add_word",
    )


def test_protected_letters_are_waw_alif_hamza_yaa():
    assert PROTECTED_LETTERS == frozenset("واءي")


@pytest.mark.parametrize("seed", SEEDS)
def test_a_pair_swap_replaces_one_consonant_by_its_partner_and_keeps_its_marks(seed):
    planted = plant("consonant_pair:ح↔ه", WORDS, random.Random(seed))
    old, new = only_changed_word(WORDS, planted)
    ((was, now),) = changed_clusters(old, new)
    assert {base(was), base(now)} == {"ح", "ه"}
    assert marks(was) == marks(now)
    assert planted.label == f"{base(was)}→{base(now)}"
    assert planted.kind == "consonant_pair:ح↔ه"


def test_a_pair_swap_is_only_plantable_where_a_letter_of_the_pair_stands():
    # ح stands in word 0 and ه in word 1; no other word carries either.
    assert plantable_words("consonant_pair:ح↔ه", WORDS) == [0, 1]
    assert plantable_words("consonant_pair:ت↔ط", WORDS) == []
    with pytest.raises(ValueError):
        plant("consonant_pair:ت↔ط", WORDS, random.Random(0))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_random_consonant_swap_changes_one_consonant_to_another_and_never_a_madd_letter(seed):
    planted = plant("consonant_random", WORDS, random.Random(seed))
    old, new = only_changed_word(WORDS, planted)
    ((was, now),) = changed_clusters(old, new)
    assert base(was) in CONSONANTS and base(now) in CONSONANTS and base(was) != base(now)
    assert marks(was) == marks(now)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_haraka_swap_puts_a_different_haraka_on_one_consonant(seed):
    planted = plant("haraka_swap", WORDS, random.Random(seed))
    old, new = only_changed_word(WORDS, planted)
    ((was, now),) = changed_clusters(old, new)
    assert base(was) == base(now)
    assert set(marks(was)) & HARAKAT and set(marks(now)) & HARAKAT
    assert marks(was) != marks(now)
    assert len(marks(was)) == len(marks(now))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_haraka_drop_removes_exactly_one_haraka_and_labels_the_exemption(seed):
    planted = plant("haraka_drop", WORDS, random.Random(seed))
    old, new = only_changed_word(WORDS, planted)
    ((was, now),) = changed_clusters(old, new)
    assert base(was) == base(now)
    assert now == "".join(char for char in was if char not in HARAKAT)
    assert len(was) - len(now) == 1
    assert planted.protected is (base(was) in PROTECTED_LETTERS)


@pytest.mark.parametrize("letter", ["و", "ء", "ي"])
def test_a_haraka_drop_on_a_protected_letter_is_labelled_protected(letter):
    planted = plant("haraka_drop", (letter + "َ", "لَ"), random.Random(0), word=0)
    assert planted.protected is True
    assert planted.words == (letter, "لَ")


@pytest.mark.parametrize("letter", ["ل", "ب", "م"])
def test_a_haraka_drop_on_any_other_letter_is_labelled_other(letter):
    planted = plant("haraka_drop", (letter + "ِ", "لَ"), random.Random(0), word=0)
    assert planted.protected is False
    assert planted.words == (letter, "لَ")


@pytest.mark.parametrize(
    "kind", ["haraka_swap", "shaddah_drop", "skip_word", "add_word", "consonant_random"])
def test_only_a_haraka_drop_carries_a_protected_label(kind):
    assert plant(kind, WORDS, random.Random(1)).protected is None


def test_a_shaddah_drop_collapses_a_geminate_to_one_consonant():
    # رَببِ holds the geminate ب(bare)+بِ; the lam of لِللَااهِ is one too (ل bare + لَ).
    assert plantable_words("shaddah_drop", WORDS) == [1, 2]
    one = plant("shaddah_drop", WORDS, random.Random(0), word=2)
    assert one.words == ("ءَلحَمدُ", "لِللَااهِ", "رَبِ", "لعَاالَمِۦۦن")
    other = plant("shaddah_drop", WORDS, random.Random(0), word=1)
    assert other.words == ("ءَلحَمدُ", "لِلَااهِ", "رَببِ", "لعَاالَمِۦۦن")


def test_a_shaddah_drop_collapses_a_four_consonant_ghunnah_geminate_to_one():
    # إِنَّ: the phonetizer spells the held noon as four. Dropping one leaves a geminate and
    # changes nothing the mushaf form sees; the whole run must collapse.
    assert plant("shaddah_drop", ("ءِننننَ",), random.Random(0)).words == ("ءِنَ",)
    assert plantable_words("shaddah_drop", ("ءِننننَ",)) == [0]


def test_a_shaddah_drop_is_one_site_per_run_and_always_changes_the_mushaf_form():
    # (word, how many Characters the collapse removes)
    corpus = {"ءِننننَ": 3, "رَببِ": 1, "لِللَااهِ": 1}
    for word, removed in corpus.items():
        planted = plant("shaddah_drop", (word,), random.Random(0))
        assert len(clusters(planted.words[0])) == len(clusters(word)) - removed
        assert mushaf_form(planted.words[0]) != mushaf_form(word)


def test_a_merged_word_boundary_and_a_madd_are_no_shaddah_site():
    # لِلكَاافِرِۦۦن recites لِ and ل from two words; لعَاالَمِۦۦن doubles ا and ۦ, no consonant.
    assert plantable_words("shaddah_drop", ("لِلكَاافِرِۦۦن",)) == []
    assert 3 not in plantable_words("shaddah_drop", WORDS)


def test_madd_and_a_repeated_vowel_letter_are_no_geminate():
    # لعَاالَمِۦۦن doubles ا and ۦ, neither a consonant.
    assert 3 not in plantable_words("shaddah_drop", WORDS)
    with pytest.raises(ValueError):
        plant("shaddah_drop", WORDS, random.Random(0), word=3)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_skipped_word_leaves_the_ayah_without_that_word_and_nothing_else(seed):
    planted = plant("skip_word", WORDS, random.Random(seed))
    assert planted.words == WORDS[: planted.word] + WORDS[planted.word + 1:]
    assert planted.inserted is None


def test_a_one_word_ayah_cannot_have_its_only_word_skipped():
    assert plantable_words("skip_word", ("رَببِ",)) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_an_added_word_is_a_copy_of_a_word_inserted_right_after_it(seed):
    planted = plant("add_word", WORDS, random.Random(seed))
    word = planted.word
    assert planted.inserted == word + 1
    assert planted.words == WORDS[: word + 1] + (WORDS[word],) + WORDS[word + 1:]


def test_planting_at_a_named_word_plants_there():
    assert plant("haraka_swap", WORDS, random.Random(5), word=3).word == 3


def test_a_named_word_that_cannot_carry_the_kind_is_refused():
    with pytest.raises(ValueError):
        plant("consonant_pair:ح↔ه", WORDS, random.Random(5), word=2)


def test_the_same_seed_plants_the_same_mistake_and_the_seeds_do_not_all_agree():
    for kind in ("consonant_random", "haraka_swap", "haraka_drop", "add_word"):
        assert plant(kind, WORDS, random.Random(9)) == plant(kind, WORDS, random.Random(9))
    spread = {plant("consonant_random", WORDS, random.Random(seed)).words for seed in SEEDS}
    assert len(spread) > 5


def test_the_planting_seed_names_the_run_the_clip_and_the_kind():
    assert planting_seed(7, "a.flac#0", "haraka_swap") == planting_seed(7, "a.flac#0", "haraka_swap")
    distinct = {planting_seed(7, "a.flac#0", "haraka_swap"),
                planting_seed(8, "a.flac#0", "haraka_swap"),
                planting_seed(7, "b.flac#0", "haraka_swap"),
                planting_seed(7, "a.flac#0", "haraka_drop")}
    assert len(distinct) == 4

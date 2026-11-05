"""Tests for #267's site mapping (step 2): Imlaei letters, alignment and site location.

Text only, no model and no audio. The reference is ``data/imlaei.json``'s word-by-word Imlaei
spelling; a model's undiacritised letters are aligned to it by edit distance; a truth site's
phonetic ``reference_index`` is mapped through its Uthmani word to one letter of that spelling.
The fixtures are small hand-built words plus the committed 11:30 site, whose realized reference
merges words (``مَيييَںںںصُرُنِۦۦ`` is two Uthmani words) and doubles geminates (``تتُ``, ``ككَ``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tadabur.fastconformer_screen.mapping import (
    Glyph,
    align,
    carried_at,
    glyphs,
    imlaei_words,
    letters_of,
    locate,
    locate_edit,
    locate_site,
    segment_letters,
    site_layout,
)
from tadabur.synthetic_edits import read_items
from tadabur.ship_a_verified import load_verified

# The committed 11:30 site: nine Uthmani words, realized into six space-separated words.
REFERENCE_11_30 = "وَيَااقَومِ مَيييَںںںصُرُنِۦۦ مِنَ للَااهِ ءِںںںطَرَتتُهُم ءَفَلَاا تَذَككَرُۥۥن"
OFFSETS_11_30 = [0, 12, 16, 30, 35, 43, 48, 59, 68, 80]
LETTERS_11_30 = "وياقوممنينصرنيمناللهءنطردتهمءفلاتذكرون"


# --- letters and glyphs --------------------------------------------------------------------


@pytest.mark.parametrize(("text", "letters"), [
    ("وَيَا قَوْمِ", "وياقوم"),                 # marks and the inner space go
    ("سُئِلَ مُؤْمِن", "سءلمءمن"),               # hamza seats ئ and ؤ read as ء
    ("أَإِنَّ", "ءءن"),                         # so do أ and إ
    ("طَرَدتُّهُمْ ۚ", "طردتهم"),               # a glued pause sign is no letter
    ("٣٠", ""),                                 # the ayah-number marker has no letters
])
def test_letters_of_keeps_only_undiacritised_letters(text, letters):
    assert letters_of(text) == letters


def test_segment_letters_joins_the_words_of_a_segment():
    assert segment_letters(["وَيَا قَوْمِ", "مَن"]) == "وياقوممن"


@pytest.mark.parametrize("text", ["تُّ", "تَّ", "تّ" + "َ"])
def test_glyphs_read_shadda_in_either_mark_order(text):
    assert glyphs(text) == (Glyph("ت", True),)


def test_glyphs_without_shadda_keep_a_repeated_letter_twice():
    assert glyphs("دَدَ") == (Glyph("د", False), Glyph("د", False))


# --- alignment by edit distance ------------------------------------------------------------


@pytest.mark.parametrize(("reference", "hypothesis", "expected"), [
    ("قمر", "قمر", [0, 1, 2]),
    ("قمر", "كمر", [0, 1, 2]),           # a substitution is still aligned
    ("قمر", "قر", [0, None, 1]),         # a deletion is a gap
    ("قمر", "قمار", [0, 1, 3]),          # an insertion shifts the hypothesis index
    ("قمر", "", [None, None, None]),
])
def test_align_maps_each_reference_letter_to_a_hypothesis_index(reference, hypothesis, expected):
    assert align(reference, hypothesis) == expected


# --- Imlaei words ----------------------------------------------------------------------------


def test_imlaei_words_are_the_segments_words_keyed_by_ayah_word():
    assert imlaei_words("11:30", 0, 2) == ["وَيَا قَوْمِ", "مَن"]


def test_imlaei_words_exclude_the_ayah_number_marker():
    assert len(imlaei_words("11:30", 0, 9)) == 9
    with pytest.raises(ValueError):
        imlaei_words("11:30", 0, 10)


# --- locating a site's letter ----------------------------------------------------------------


def test_locate_reads_a_letter_through_merged_words():
    words = imlaei_words("11:30", 0, 9)
    assert segment_letters(words) == LETTERS_11_30
    assert locate(words, OFFSETS_11_30, REFERENCE_11_30, 48) == 22                         # ط
    assert locate(words, OFFSETS_11_30, REFERENCE_11_30, REFERENCE_11_30.index("ص")) == 10  # ص, in the merged ينصرني


def test_locate_collapses_a_realized_geminate_to_one_letter():
    # realized تتُ is one Imlaei ت (written with a shadda); the letters after it keep their place
    words, reference, offsets = ["طَرَدتُّهُمْ"], "طَرَتتُهُم", [0, 10]
    assert [locate(words, offsets, reference, i) for i in (4, 5, 7)] == [3, 3, 4]


def test_locate_geminate_inside_the_committed_site():
    words = imlaei_words("11:30", 0, 9)
    assert {locate(words, OFFSETS_11_30, REFERENCE_11_30, i) for i in (72, 73)} == {34}   # ككَ -> ك
    assert locate(words, OFFSETS_11_30, REFERENCE_11_30, 75) == 35                         # ر after it


def test_locate_picks_a_repeated_letter_by_occurrence_order():
    words, reference, offsets = ["قَرَقَ"], "قَرَقَ", [0, 6]
    assert locate(words, offsets, reference, 0) == 0
    assert locate(words, offsets, reference, 4) == 2


def test_locate_reads_hamza_seats_as_hamza():
    assert locate(["أَفَلَا"], [0, 8], "ءَفَلَاا", 0) == 0
    assert locate(["مُؤْمِن"], [0, 6], "مُءمِن", 2) == 1


@pytest.mark.parametrize("index", [1, 2], ids=["haraka", "marker"])
def test_locate_refuses_a_carrier_that_is_no_consonant(index):
    assert locate(["مِنَ"], [0, 4], "مِںَ", index) is None


def test_locate_refuses_a_letter_the_imlaei_word_does_not_have():
    assert locate(["قَمَر"], [0, 6], "قَرَقَ", 4) is None


# --- reading the aligned letter or its shadda -------------------------------------------------


@pytest.mark.parametrize(("hypothesis", "expected"), [
    ("قَمَر", "ق"),
    ("كَمَر", "ك"),        # the model heard the partner letter
    ("مَر", None),         # nothing aligned at the carrier
    ("", None),
])
def test_carried_at_reads_the_letter_aligned_to_the_carrier(hypothesis, expected):
    assert carried_at("قمر", 0, hypothesis, shaddah=False) == expected


def test_carried_at_reads_hamza_as_hamza():
    assert carried_at("ءفلا", 0, "أَفَلَا", shaddah=False) == "ء"


@pytest.mark.parametrize(("hypothesis", "expected"), [
    ("طَرَدتُّهُمْ", "held"),
    ("طَرَدَتُهُمْ", "not_held"),
    ("طَرَدُهُمْ", None),
])
def test_carried_at_reads_the_shadda_of_the_aligned_letter(hypothesis, expected):
    assert carried_at("طردتهم", 3, hypothesis, shaddah=True) == expected


def test_a_shadda_on_another_consonant_is_not_the_sites_carrier():
    """The aligned glyph is a شّ, not the site's د: nothing is carried at this site."""
    assert carried_at("مدر", 1, "مَرَّرَ", shaddah=True) is None
    # the substitution stays readable on a consonant-pair site
    assert carried_at("مدر", 1, "مَرَّرَ", shaddah=False) == "ر"


def test_a_shaddaless_substitution_is_not_read_as_the_carriers_state():
    # the carrier د is unaligned; the aligned ر carries no shadda, but it is not this site
    assert carried_at("مدر", 1, "مَرَرَ", shaddah=True) is None
    assert carried_at("مدر", 1, "مَدَرَ", shaddah=True) == "not_held"


# --- the committed sites and edits --------------------------------------------------------------


def verified_sites():
    return load_verified()[0]


def test_every_new_audit_site_is_located_by_a_rebuilt_reference():
    # 16 verified sites have no stored word offsets; the phonetizer rebuilds them, and a site
    # is located only if the rebuilt reference equals the stored one exactly
    stored = _stored_site_ids()
    rebuilt = [site for site in verified_sites() if site.site_id not in stored]
    assert len(rebuilt) == 16
    for site in rebuilt:
        layout = site_layout(site)
        assert layout is not None, site.site_id
        assert layout.reference == site.reference


def _stored_site_ids() -> set[str]:
    """The sites whose word offsets are committed (``ship_a/sites.jsonl``)."""
    path = Path(__file__).parent / "ship_a" / "sites.jsonl"
    return {json.loads(line)["site_id"] for line in path.read_text(encoding="utf-8").splitlines()}


def test_a_reference_the_phonetizer_cannot_rebuild_is_unlocated():
    stored = _stored_site_ids()
    site = next(s for s in verified_sites() if s.site_id not in stored)
    broken = dataclasses.replace(site, reference=site.reference + "ا")
    assert site_layout(broken) is None
    assert locate_site(broken) is None


def test_the_committed_11_30_site_locates_its_tah():
    site = next(s for s in verified_sites() if s.reference == REFERENCE_11_30 and s.reference_index == 48)
    located = locate_site(site)
    assert (located.letters, located.index) == (LETTERS_11_30, 22)


def test_at_least_95_percent_of_ship_a_sites_are_located():
    sites = verified_sites()
    located = [locate_site(site) for site in sites]
    assert sum(item is not None for item in located) >= 0.95 * len(sites)
    for site, item in zip(sites, located):
        if item is not None:
            assert 0 <= item.index < len(item.letters), site.site_id


def test_a_sample_of_the_edits_is_located():
    items = read_items()[::10]
    assert all(locate_edit(item) is not None for item in items)

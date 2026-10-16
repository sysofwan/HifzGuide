"""Tests for tadabur.nonpro_sources: which rows are read and the words each clip recites (#156).

The datasets' rows are synthetic; the canonical words are quran-transcript's (local).
"""

from __future__ import annotations

import pytest

from tadabur.nonpro_sources import (
    DATASETS,
    DECODED,
    QDAT,
    QDAT_PASSAGE,
    RETASY,
    TARTEEL_V1,
    Clip,
    LeftOut,
    Unplaced,
    canonical_words,
    locate,
    locate_clips,
    select,
)
from tadabur.truth_sites import NONPRO_SOURCES


def _retasy(**overrides) -> dict:
    return {"audio": {"path": "u.wav", "bytes": b"x"}, "Surah": "Al-Faatihah",
            "Aya": "الرَّحْمَنِ الرَّحِيمِ", "final_label": "correct", "reciter_qiraah": "hafs",
            "reciter_id": "r1", "golden": False, **overrides}


def test_every_decoded_dataset_is_a_truth_site_source_and_pinned():
    assert {d.name for d in DECODED} == NONPRO_SOURCES
    assert all(len(d.revision) == 40 for d in DATASETS.values())
    assert [d.open for d in DECODED] == [True, False, True]  # RetaSy's licence is unknown


def test_a_whole_ayah_wins_over_a_run_inside_another():
    """``ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ`` is 1:3 whole, and words 3-4 of the basmala."""
    assert locate("الرَّحْمَنِ الرَّحِيمِ", 1, range(1, 8), canonical_words) == (3, 1, 2)


def test_retasy_s_orthographies_place_on_the_canonical_words():
    # Farsi ya, imlaei alif and hamza: 103:3, and Ayat al-Kursi's words 8-12 of 2:255.
    assert locate("إِنَّ ٱلۡإِنسَـٰنَ لَفِی خُسۡرٍ", 103, range(1, 4), canonical_words) == (2, 1, 4)
    assert locate("لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْمٌ", 2, [255], canonical_words) == (255, 8, 12)
    # 109:3 and 109:5 are one text: the first is taken, the words are the same either way.
    assert locate("وَلَاۤ أَنتُمۡ عَـٰبِدُونَ مَاۤ أَعۡبُدُ", 109, range(1, 7),
                  canonical_words) == (3, 1, 5)
    with pytest.raises(Unplaced):
        locate("أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيْمِ", 1, range(1, 8), canonical_words)


def test_retasy_reads_hafs_clips_labelled_correct_or_not_and_only_quran():
    clip = select(RETASY, _retasy())
    assert isinstance(clip, Clip)
    assert (clip.surah_ayah, clip.word_from, clip.word_to) == ("1:3", 1, 2)
    assert clip.labels == {"final_label": "correct", "golden": False}
    assert select(RETASY, _retasy(final_label="in_correct")).labels["final_label"] == "in_correct"
    for row, reason in ((_retasy(final_label="multiple_aya"), "label multiple_aya"),
                        (_retasy(reciter_qiraah="warsh"), "not hafs"),
                        (_retasy(Surah="Adhan"), "no Quran: Adhan"),
                        (_retasy(Aya="أَعُوذُ بِاللَّهِ"), "no ayah's words: Al-Faatihah")):
        assert select(RETASY, row) == LeftOut("u.wav", reason)


def test_tarteel_is_mapped_by_its_columns_to_the_whole_ayah_never_by_its_text():
    row = {"audio": {"path": "1_2_3.wav"}, "text": "anything", "sura": 1, "ayah": 2,
           "cer_nemo": 0.1, "cer_wav": 0.2, "completeness": 1.0}
    clip = select(TARTEEL_V1, row)
    assert (clip.audio_filename, clip.surah_ayah, clip.word_from, clip.word_to, clip.reciter) == \
        ("1_2_3.wav", "1:2", 1, 4, "")


def test_qdat_is_5_109_s_words_8_to_15_with_its_reciter_and_tajweed_labels():
    row = {"audio": {"path": "s7_3.wav"}, "id": "abcd1234", "original_id": "s7_3",
           "separate_tide": 1, "the_tight_noon": 1.0, "concealment": 0, "target": 0}
    clip = select(QDAT, row)
    assert (clip.audio_filename, clip.reciter) == ("abcd1234", "s7")
    assert (clip.surah, clip.ayah, clip.word_from, clip.word_to) == QDAT_PASSAGE
    words, _ = canonical_words(5, 109)
    assert words[QDAT_PASSAGE[2] - 1] == "قَالُوا۟" and words[QDAT_PASSAGE[3] - 1] == "ٱلْغُيُوبِ"
    assert clip.labels == {"separate_tide": 1, "the_tight_noon": 1, "concealment": 0, "target": 0}


def test_a_qdat_label_the_dataset_leaves_empty_stays_empty():
    row = {"audio": {"path": "s7_3.wav"}, "id": "abcd1234", "original_id": "s7_3",
           "separate_tide": 1, "the_tight_noon": None, "concealment": 0, "target": 0}
    assert select(QDAT, row).labels["the_tight_noon"] is None


def test_only_a_dataset_named_by_audio_path_is_located_by_it():
    """qdat names its clips by ``id``: its audio paths would locate nothing it names."""
    with pytest.raises(ValueError):
        locate_clips(QDAT, ["x.wav"])

"""Whose error each consonant false flag is (#155): where the letter sits, what the mushaf and
the recital say, the class, the rates and the doc. Torch-free, on synthetic phonemes."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import consonant_flags as cf
from training.consonant_flags_doc import render
from training.device_logs import committed_transcript
from training.word_flags import OTHER_CONSONANT, SHADDAH_CAUSE


def mismatch(expected: str, heard: str, group: int) -> dict:
    return {"kind": {"type": "mismatch", "expected": expected, "heard": heard}, "group_index": group}


def deletion(expected: str, group: int) -> dict:
    return {"kind": {"type": "deletion", "expected": expected}, "group_index": group}


# --- where the letter is ---------------------------------------------------------------------------


@pytest.mark.parametrize("word, units", [
    # the leading assimilated lam (مِّن لَّدُنۡهُ) is never scored
    ("للَدُنهُ", ["لَ", "دُ", "ن", "هُ"]),
    # the leading tanween bleed, the trailing connection waw and the madd marks are skipped
    ("ںںںذِرَ", ["ذِ", "رَ"]),
    ("ظُلُمَااتُوو", ["ظُ", "لُ", "مَ", "اا", "تُ"]),
    ("يَعلَمُۥۥن", ["يَ", "ع", "لَ", "مُ", "ن"]),
    # a run of one bare letter is one group; the voweled half of a geminate its own
    ("ءِننننَكَ", ["ءِ", "ننن", "نَ", "كَ"]),
])
def test_the_scored_units_are_the_groups_muraja_numbers(word, units):
    assert [u for _, u in cf.scored_units(word)] == units


def test_a_group_names_the_first_unit_at_or_after_it_that_holds_the_letter():
    word = "يَرجِعُۥۥن"  # an alignment that starts on its raa numbers ع as group 2, not 3
    assert cf.candidates(word, "ع", 2) == [5]
    assert cf.candidates(word, "ع", 3) == [5]
    assert cf.candidates(word, "ع", 4) == []
    # without a group, every unit holding the letter
    assert cf.candidates("مَددَمَدَ", "د", None) == [2, 3, 7]


# --- what the mushaf says ----------------------------------------------------------------------------


@pytest.mark.parametrize("text, index, reason", [
    ("مِّن لَّدُنْهُ", 3, "unmarked ن before a shaddah ل (complete idgham)"),
    ("بَل رَّفَعَهُ", 2, "unmarked ل before a shaddah ر (complete idgham)"),
    ("قَالُوا۟ ءَامَنَّا", 6, "a letter marked silent"),
    ("وَٱلْكِتَـٰبِ", 2, "hamzat al-wasl inside the ayah"),
    ("مِنۢ بَعْدِ", 2, "noon marked for iqlab"),
    ("هُدًۭى لِّلْمُتَّقِينَ", 3, "tanween marked as not izhar (idgham or ikhfa)"),
    ("لَهُم بِهِۦ", 4, "unmarked meem before baa (ikhfa shafawi)"),
    ("مِن قَبْلِ", 2, "unmarked noon (ikhfa, iqlab or idgham)"),
    ("قُلْنَا", 2, None),  # a sukun: pronounced
    ("عَدْنٍۢ تَجْرِى", 3, None),  # the iqlab mark is the tanween's; the noon is voweled
    ("عَلِيمٌۢ بِذَاتِ", 5, None),  # a letter with tanween is pronounced; the tanween is the noon
    ("ٱللَّهُ", 0, None),  # hamzat al-wasl opening the ayah is pronounced
])
def test_the_mushaf_marks_say_which_letters_are_not_pronounced_as_written(text, index, reason):
    found = cf.mushaf_reason(text, index)
    assert (found[0] if found else None) == reason


def test_a_phonetizer_slip_is_where_it_kept_a_letter_the_mushaf_assimilates():
    uthmani = "مِن لَّهُ"  # unmarked noon before a shaddah lam
    good, kept = "مِ للَهُ", "مِن لَهُ"
    good_spans = [(0, 1), (1, 2), (3, 4), None, (4, 5), (5, 6), None, (6, 7), (7, 8)]
    kept_spans = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), None, (6, 7), (7, 8)]
    assert cf.phonetizer_slips(uthmani, good, good_spans) == {}
    assert cf.phonetizer_slips(uthmani, kept, kept_spans) == {2: "unmarked ن before a shaddah ل (complete idgham)"}


# --- the classification -------------------------------------------------------------------------------

AYAH = "قَالَ رَببُ لعَاالَمِۦۦنَ"  # word 2 holds a geminate baa, word 3 a lam


def test_a_letter_the_realized_reference_lacks_is_the_reference_and_the_scorer():
    ayah = cf.Ayah("عَذَاابُن عَظِۦۦم")
    reading, _ = cf.classify(ayah, 1, deletion("ن", 4), cf.Recital("عَذَااب", ends_here=True), OTHER_CONSONANT)
    assert (reading.klass, reading.rule, reading.responsible) == (cf.REFERENCE, "waqf form", cf.MURAJA_SCORER)


def test_a_pronounced_letter_the_decode_holds_is_grading_one_it_does_not_hold_the_models():
    ayah = cf.Ayah(AYAH)
    realized = AYAH
    held = cf.classify(ayah, 3, mismatch("ل", "ر", 0), cf.Recital(realized, decode="قَالَرَببُلعَاالَمِۦۦنَ"),
                       OTHER_CONSONANT)[0]
    misheard = cf.classify(ayah, 3, mismatch("ل", "ر", 0), cf.Recital(realized, decode="قَالَرَببُرعَاالَمِۦۦنَ"),
                           OTHER_CONSONANT)[0]
    assert (held.klass, held.responsible) == (cf.GRADING, cf.MURAJA_SCORER)
    assert (misheard.klass, misheard.responsible, misheard.evidence) == (cf.MODEL, None, "decode")


def test_a_collapsed_geminate_is_grading_only_when_the_decode_holds_both_halves():
    ayah = cf.Ayah(AYAH)
    error = deletion("ب", 1)
    held = cf.classify(ayah, 2, error, cf.Recital(AYAH, decode="قَالَرَببُلعَاالَمِۦۦنَ"), SHADDAH_CAUSE)[0]
    single = cf.classify(ayah, 2, error, cf.Recital(AYAH, decode="قَالَرَبُلعَاالَمِۦۦنَ"), SHADDAH_CAUSE)[0]
    assert (held.klass, single.klass) == (cf.GRADING, cf.MODEL)


def test_with_no_recital_a_wasl_dependent_letter_is_unclear_and_a_slip_is_the_phonetizers():
    pause = {cf.WAQF_HERE: cf.Form("عَذَااب", 0, 9)}
    ayah = cf.Ayah("عَذَاابُن عَظِۦۦم", forms=lambda word: pause)
    reading, _ = cf.classify(ayah, 1, deletion("ن", 4), cf.Recital(), OTHER_CONSONANT)
    assert (reading.klass, reading.rule) == (cf.UNCLEAR, f"wasl-dependent: dropped on {cf.WAQF_HERE}")
    # a letter every form keeps is pronounced: the model's on the kept grade alone
    kept, _ = cf.classify(ayah, 1, mismatch("ذ", "ز", 1), cf.Recital(), OTHER_CONSONANT)
    assert (kept.klass, kept.evidence) == (cf.MODEL, "kept grade")
    slipped = cf.Ayah("مِن لَهُ", slips={2: "unmarked ن before a shaddah ل (complete idgham)"})
    reading, _ = cf.classify(slipped, 1, mismatch("ن", "ل", 1), cf.Recital(), OTHER_CONSONANT)
    assert (reading.klass, reading.responsible) == (cf.REFERENCE, cf.PHONETIZER)


def test_the_owners_verdict_stands_in_for_a_decode():
    reading, _ = cf.classify(cf.Ayah(AYAH), 3, mismatch("ل", "ر", 0), cf.Recital(verdict=cf.GRADING),
                             OTHER_CONSONANT)
    assert (reading.klass, reading.responsible, reading.evidence) == (cf.GRADING, cf.MURAJA_SCORER, "owner")


def test_a_letter_not_in_the_word_or_candidates_that_disagree_are_unclear():
    ayah = cf.Ayah("مَددَمَدَ")
    assert cf.classify(ayah, 1, mismatch("ق", "ك", 0), cf.Recital(), OTHER_CONSONANT)[0].rule == \
        "the letter is not in the word"
    # without a group, the three dals are candidates; the realized form lacks only the last
    realized = cf.Recital("مَددَمَ", decode="مَددَمَ")
    reading, unit = cf.classify(ayah, 1, cf.error_without_group("د→ت"), realized, OTHER_CONSONANT)
    assert (reading.klass, unit) == (cf.UNCLEAR, None)


def test_a_substitution_comes_back_as_an_error_with_no_group():
    assert cf.error_without_group("ه→∅") == {"kind": {"type": "deletion", "expected": "ه"},
                                             "group_index": cf.UNKNOWN_GROUP}
    assert cf.error_without_group("ل→ء")["kind"] == {"type": "mismatch", "expected": "ل", "heard": "ء"}


def test_the_context_brackets_the_letter_between_the_neighbouring_words():
    assert cf.context(AYAH, 2, 2) == "لَ|رَ[ب]بُ|ل"  # a geminate's bare half is its own group
    assert cf.context(AYAH, 1, None) == "#|قَالَ|رَ"


def test_the_decode_excerpt_is_what_the_decode_aligns_to_the_word():
    assert cf.decode_excerpt("قَالَرَبُلعَاالَمِۦۦنَ", AYAH, (5, 10)) == "رَبُ"


def test_a_session_transcript_is_its_updates_hops_in_order():
    log = ("x Update: type=window, hop=قَا, overlap=لَ رَب, didFlush=false\n"
           "x Update: type=preview, hop=, overlap=لَ, didFlush=false\n"
           "x Update: type=window, hop=لَ, overlap=, didFlush=false\n")
    assert committed_transcript(log) == "قَالَ"


# --- records, rates and the doc ------------------------------------------------------------------------


def record(word: str, substitution: str, klass: str, cluster: str, **kw) -> cf.FlagRecord:
    return cf.FlagRecord(
        population=kw.get("population", cf.QUD), model="h448", scoring=kw.get("scoring", "shipped_default"),
        word=word, printed="كلمة", cause=OTHER_CONSONANT, substitution=substitution, kind=cf.CONSONANT,
        group=0, unit=0, context="#|x|#", klass=klass, rule=kw.get("rule", "pronounced"),
        responsible=kw.get("responsible"), evidence="decode", cluster=cluster)


def population(clustered: bool) -> cf.Population:
    graded = {f"m:{s}": 100 for s in range(1, 31)}
    records = (record("1:1:1", "ن→ل", cf.MODEL, "m:1"), record("1:1:1", "ل→ر", cf.MODEL, "m:1"),
               record("2:3:1", "ن→ل", cf.MODEL, "m:2"), record("2:4:1", "ن→م", cf.MODEL, "m:2"),
               record("3:1:2", "ه→∅", cf.REFERENCE, "m:3", rule="waqf form", responsible=cf.MURAJA_SCORER),
               record("4:1:2", "د→ت", cf.GRADING, "m:4", responsible=cf.MURAJA_SCORER),
               record("5:1:2", "د→ت", cf.UNCLEAR, "m:5", rule="the letter is not in the word"))
    return cf.Population(cf.QUD, "h448", "shipped_default", graded, clustered, records)


def test_the_ranking_counts_words_per_1000_graded_with_a_clustered_interval():
    rows = cf.rate_rows(population(clustered=True), cf.MODEL, by_expected=False)
    assert [(r["substitution"], r["words"], r["per_1000"]) for r in rows] == [
        ("ن→ل", 2, 0.67), ("ل→ر", 1, 0.33), ("ن→م", 1, 0.33)]
    assert rows[0]["lower"] <= rows[0]["per_1000"] <= rows[0]["upper"]
    by_letter = cf.rate_rows(population(clustered=True), cf.MODEL, by_expected=True)
    assert [(r["expected"], r["words"]) for r in by_letter] == [("ن", 3), ("ل", 1)]
    assert "lower" not in cf.rate_rows(population(clustered=False), cf.MODEL, by_expected=False)[0]


def test_the_classes_and_the_not_model_table_name_the_responsible_component():
    pop = population(clustered=True)
    assert {k: v["words"] for k, v in cf.class_counts(pop).items()} == {
        cf.MODEL: 3, cf.GRADING: 1, cf.REFERENCE: 1, cf.UNCLEAR: 1}
    assert [(r["class"], r["responsible"], r["substitution"]) for r in cf.responsible_table(pop)] == [
        (cf.GRADING, cf.MURAJA_SCORER, "د→ت"), (cf.REFERENCE, cf.MURAJA_SCORER, "ه→∅")]


def test_the_owner_checks_list_each_class_top_substitutions_in_quran_order_a_word_once():
    flags = [record(f"{s}:1:1", "ن→ل", cf.MODEL, f"m:{s}") for s in range(12, 0, -1)]
    flags += [record("1:1:1", "ن→ل", cf.MODEL, "m:1", scoring="allowances_off")]
    checks = cf.owner_checks(flags)
    (only,) = checks[cf.MODEL]
    assert only["substitution"] == "ن→ل"
    assert [w["word"] for w in only["words"]] == [f"{s}:1:1" for s in range(1, cf.CHECK_WORDS + 1)]
    assert checks[cf.REFERENCE] == []


def test_the_doc_renders_every_population_and_the_owner_checks():
    pop = population(clustered=True)
    report = {"muraja_commit": "99c326f3fb", "populations": [cf.population_view(pop)],
              "phonetizer_check": {"ayat_checked": 6228, "not_checked": ["55:17"], "slips": []},
              "records": [cf.asdict(r) for r in pop.records],
              "owner_checks": [{"population": cf.QUD, "model": "h448", "checks": cf.owner_checks(pop.records)}]}
    doc = render(report)
    assert "| ن→ل | other_consonant | 2 | 0.67 [" in doc
    assert "| reference | Muraja word scorer | ه→∅ | waqf form | 1 | 0.33 |" in doc
    assert "## Owner-check list" in doc and "`1:1:1` كلمة" in doc
    assert "6228 ayat checked; 1 not checked" in doc
    assert "| QUD pilot | h448 | shipped_default | ن→ل | 2 | 0 | 0 | 0 |" in doc
    assert "grading (Muraja word scorer) 1; reference (Muraja word scorer) 1; unclear (none named) 1" in doc


QURAN_DB = Path(__file__).parent.parent.parent / "data" / "quran.db"


@pytest.mark.skipif(not QURAN_DB.exists(), reason="data/quran.db is not built here")
def test_the_forms_of_a_real_ayah_say_which_letters_a_pause_or_a_start_drops():
    pytest.importorskip("quran_transcript")
    from training.muraja_policy import MurajaText

    text = MurajaText(QURAN_DB)
    assert text.text_words(109, 6, 3) == [3]
    ayat = cf.Ayat(text)
    ayah = ayat.get(109, 6)  # لَكُمۡ دِينُكُمۡ وَلِيَ دِينِ
    assert ayah.phonemes == "لَكُم دِۦۦنُكُم وَلِيَ دِۦۦن"
    assert ayah.slips == {}
    # a pause on وَلِيَ makes its yaa a madd; a start at the last word keeps all of it
    yaa = cf.word_start(ayah.phonemes, 3) + 4
    reading, _ = cf.classify(ayah, 3, mismatch("ي", "ۦ", 2), cf.Recital(), OTHER_CONSONANT)
    assert (ayah.phonemes[yaa], reading.rule) == ("ي", f"wasl-dependent: dropped on {cf.WAQF_HERE}")
    assert cf.classify(ayah, 4, deletion("ن", 1), cf.Recital(), OTHER_CONSONANT)[0].klass == cf.MODEL

"""Whose error each consonant false flag is (#155): where the letter sits, what the mushaf and
the recital say, the class, the rates and the doc. Torch-free, on synthetic phonemes."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import consonant_flags as cf
from training.consonant_flags_doc import render
from training.decoding import committed_text
from training.device_logs import session_checks
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


@pytest.mark.parametrize("text, index, letter, spelled", [
    ("قُلْنَا", 2, "ل", cf.ITSELF_MARKED),       # sukun
    ("مِّن لَّدُنْهُ", 3, "ل", cf.ANOTHER_LETTER),  # the noon assimilated into the lam
    ("مِن قَبْلِ", 2, "ن", cf.ITSELF_UNMARKED),   # an ikhfa noon, unmarked
    ("مِّن لَّدُنْهُ", 7, "ل", cf.ITSELF_MARKED),   # the shaddah: the geminate's second lam
    ("عَلِيمٌ", 6, "ن", cf.A_MARK),                # the tanween's noon
    ("أَحَدٌ", 0, "ء", cf.ITSELF_MARKED),          # a hamza on its seat
])
def test_the_spelling_says_whether_the_mushaf_writes_and_marks_the_letter(text, index, letter, spelled):
    assert cf.spelling(text, index, letter) == spelled


def test_a_planted_wrong_letter_is_not_spelled_as_itself():
    """4:158 بَل رَّفَعَهُ: the reference's geminate raa is spelled as itself; a noon planted in its
    place is spelled as another letter, so the spelling check cannot pass it."""
    uthmani = "بَل رَّفَعَهُ"
    phonemes = "بَ ررَفَعَهُ"
    spans = [(0, 1), (1, 2), (3, 4), None, (4, 5), (5, 6), None, (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)]
    assert cf.spellings(uthmani, phonemes, spans)[3] == cf.ITSELF_MARKED
    assert cf.spellings(uthmani, phonemes.replace("رَ", "نَ"), spans)[3] == cf.ANOTHER_LETTER


# --- the classification -------------------------------------------------------------------------------

AYAH = "قَالَ رَببُ لعَاالَمِۦۦنَ"  # word 2 holds a geminate baa, word 3 a lam


def test_a_letter_the_realized_reference_lacks_is_the_reference_and_the_scorer():
    ayah = cf.Ayah("عَذَاابُن عَظِۦۦم")
    reading, _ = cf.classify(ayah, 1, deletion("ن", 4), cf.Recital("عَذَااب", ends_here=True), OTHER_CONSONANT)
    assert (reading.klass, reading.rule, reading.responsible) == (cf.REFERENCE, "waqf form", cf.MURAJA_SCORER)


def check(hop: str, overlap: str = "") -> dict:
    return {"hop": hop, "overlap": overlap, "flush": False}


def recital(realized: str | None, checks: list[dict], setting: tuple[int, ...] = (0,)) -> cf.Recital:
    return cf.Recital(realized, checks=tuple(checks), setting=lambda position: setting)


def test_a_check_may_have_aligned_any_count_of_its_pending_hops():
    """Past the 20 confirmed Characters, which hops were pending changes the query."""
    ba, ta, tha = "\u0628\u064e", "\u062a\u064e", "\u062b\u064e"
    checks = [check(ba * 25), check("", tha), check(ta, tha)]
    # its own hop is always pending: never folded into the confirmed tail
    assert cf.possible_queries(checks, 2) == [ba * 20 + ta + tha, ba * 25 + ta + tha]
    # a preview adds no hop: none or the one before it pending
    assert cf.possible_queries(checks, 1) == [ba * 20 + tha, ba * 25 + tha]
    # a stop has no overlap, and no more than MAX_PENDING_HOPS hops are ever pending
    many = [check(ba * 25) for _ in range(cf.MAX_PENDING_HOPS + 3)] + [{**check(ta), "stop": True}]
    queries = cf.possible_queries(many, len(many) - 1)
    assert len(queries) == cf.MAX_PENDING_HOPS and not queries[-1].endswith(tha)
    assert all(q.endswith(ba * 20 + ta) for q in queries)  # the stop's phonemes always pending


def test_a_pronounced_letter_every_possible_query_holds_is_grading_none_holds_the_models():
    error = mismatch("ل", "ر", 0)
    held = cf.classify(cf.Ayah(AYAH), 3, error, recital(AYAH, [check("قَالَرَببُلعَاالَمِۦۦنَ")]), OTHER_CONSONANT)[0]
    misheard = cf.classify(cf.Ayah(AYAH), 3, error, recital(AYAH, [check("قَالَرَببُرعَاالَمِۦۦنَ")]),
                           OTHER_CONSONANT)[0]
    assert (held.klass, held.responsible, held.evidence) == (cf.GRADING, cf.MURAJA_SCORER, "query")
    assert (misheard.klass, misheard.responsible, misheard.evidence) == (cf.MODEL, None, "query")


def test_the_setting_check_decides_not_the_committed_transcript():
    """The transcript commits the lam in a later hop; the preview that set the grade heard raa."""
    checks = [check("", "قَالَرَببُرعَاالَمِۦۦنَ"), check("قَالَرَببُلعَاالَمِۦۦنَ")]
    by_preview = cf.classify(cf.Ayah(AYAH), 3, mismatch("ل", "ر", 0), recital(AYAH, checks, (0,)), OTHER_CONSONANT)
    assert by_preview[0].klass == cf.MODEL
    both = cf.classify(cf.Ayah(AYAH), 3, mismatch("ل", "ر", 0), recital(AYAH, checks, (0, 1)), OTHER_CONSONANT)
    assert (both[0].klass, both[0].rule) == (cf.UNCLEAR, "pronounced; the queries that can have set the kept grade disagree")
    none = cf.classify(cf.Ayah(AYAH), 3, mismatch("ل", "ر", 0), recital(AYAH, checks, ()), OTHER_CONSONANT)
    assert none[0].klass == cf.UNCLEAR


def test_a_setting_check_that_cannot_be_read_is_never_masked_by_a_later_one():
    """Sol, #157 round 2 (40:9:8): check 0's query stops before the word's last letter, so it
    cannot be read there; check 1 holds it. Together they are unclear, not grading."""
    checks = [check("", "قَالَرَببُلعَاالَمِ"), check("قَالَرَببُلعَاالَمِۦۦنَ")]
    error = deletion("ن", 4)
    alone = cf.classify(cf.Ayah(AYAH), 3, error, recital(AYAH, checks, (1,)), OTHER_CONSONANT)[0]
    assert alone.klass == cf.GRADING
    both = cf.classify(cf.Ayah(AYAH), 3, error, recital(AYAH, checks, (0, 1)), OTHER_CONSONANT)[0]
    assert (both.klass, both.rule) == (
        cf.UNCLEAR, "pronounced; a check that can have set the kept grade has no query that reaches the letter")


def test_the_stop_aligns_its_phonemes_with_the_overlap_the_engine_kept():
    """``handleFinalFlush`` appends the stop's phonemes and aligns them with ``currentOverlap``
    from the update before it (``FollowAlongEngine.swift:899-913``); a silence flush leaves none
    of the hops before it pending."""
    ta, tha = "\u062a\u064e", "\u062b\u064e"
    checks = [check(ta, tha), {**check(ta), "stop": True}]
    assert cf.kept_overlap(checks, 1) == tha
    assert cf.possible_queries(checks, 1) == [ta + ta + tha]
    flushed = [check(ta, tha), {"hop": ta, "overlap": tha, "flush": True}, check(ta, tha)]
    assert cf.possible_queries(flushed, 1) == [ta + ta + tha]          # a flush aligns no pending hop
    assert cf.possible_queries(flushed, 2) == [ta * 3 + tha]           # the flushed hops stay confirmed


def test_a_collapsed_geminate_is_grading_only_when_the_query_holds_both_halves():
    error = deletion("ب", 1)
    held = cf.classify(cf.Ayah(AYAH), 2, error, recital(AYAH, [check("قَالَرَببُلعَاالَمِۦۦنَ")]), SHADDAH_CAUSE)[0]
    single = cf.classify(cf.Ayah(AYAH), 2, error, recital(AYAH, [check("قَالَرَبُلعَاالَمِۦۦنَ")]), SHADDAH_CAUSE)[0]
    assert (held.klass, single.klass) == (cf.GRADING, cf.MODEL)


def test_a_device_letter_is_read_in_every_check_whose_query_can_hold_it():
    checks = [check("قَالَ", "رَب"), check("رَببُ", "لعَا"), check("لعَاالَمِۦۦنَ")]
    position = cf.word_start(AYAH, 3)
    assert cf.covering_checks(checks, AYAH, position) == (1, 2)
    assert cf.covering_checks([check("زَيدُن")], AYAH, position) == ()


def test_with_no_recital_a_wasl_dependent_letter_is_unclear_and_a_slip_is_the_phonetizers():
    pause = {cf.WAQF_HERE: cf.Form("عَذَااب", 0, 9)}
    ayah = cf.Ayah("عَذَاابُن عَظِۦۦم", forms=lambda word: pause)
    reading, _ = cf.classify(ayah, 1, deletion("ن", 4), cf.Recital(), OTHER_CONSONANT)
    assert (reading.klass, reading.rule) == (cf.UNCLEAR, f"wasl-dependent: dropped on {cf.WAQF_HERE}")
    # a letter every form keeps is pronounced, but with no decode and no verdict it is unclear
    kept, _ = cf.classify(ayah, 1, mismatch("ذ", "ز", 1), cf.Recital(), OTHER_CONSONANT)
    assert (kept.klass, kept.rule) == (cf.UNCLEAR, "pronounced; no decode to read and no owner verdict")
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
    reading, unit = cf.classify(ayah, 1, cf.error_without_group("د→ت"), recital("مَددَمَ", [check("مَددَمَ")]),
                                OTHER_CONSONANT)
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


def test_a_sessions_checks_are_read_as_the_pinned_replay_parses_its_log():
    log = ("[10:00:00.000] [INFO] [FollowAlong] Update: type=window, hop=قَا, overlap=لَ رَب, didFlush=false\n"
           "Update: type=preview, hop=, overlap=لَ, didFlush=false\n"
           "Withheld update: hop=زَ, didFlush=false\n"
           "Update: type=flush, hop=لَ, overlap=, didFlush=true\n"
           "Stop: finalPhonemes=رَب\n")
    assert session_checks(log) == [
        {"hop": "قَا", "overlap": "لَ رَب", "flush": False}, {"hop": "", "overlap": "لَ", "flush": False},
        {"hop": "لَ", "overlap": "", "flush": True}, {"hop": "رَب", "overlap": "", "flush": False, "stop": True}]
    assert committed_text(session_checks(log)) == "قَالَرَب"
    legacy = ("[Engine] Window inference 302ms \u2014 confirmed: كَاا, unconfirmed: فهَا \n"
              "Preview inference 337ms \u2014 \n"
              "Preview inference 337ms \u2014 فهَاا\n")
    assert session_checks(legacy) == [{"hop": "كَاا", "overlap": "فهَا", "flush": False},
                                      {"hop": "", "overlap": "فهَاا", "flush": False}]
    assert session_checks("Position \u2192 1:1:1\n") == []


# --- records, rates and the doc ------------------------------------------------------------------------


def record(word: str, substitution: str, klass: str, cluster: str, **kw) -> cf.FlagRecord:
    return cf.FlagRecord(
        population=kw.get("population", cf.QUD), model="h448", scoring=kw.get("scoring", "shipped_default"),
        word=word, occurrence=kw.get("occurrence", f"{cluster}:{word}"), cause=OTHER_CONSONANT,
        substitution=substitution, kind=cf.CONSONANT, group=0, unit=0, spelled=kw.get("spelled", cf.ITSELF_MARKED),
        klass=klass, rule=kw.get("rule", "pronounced"), responsible=kw.get("responsible"), evidence="query",
        cluster=cluster, printed="كلمة", context="#|x|#")


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


def test_the_owner_checks_count_every_substitution_a_word_carries():
    """A word's last error must not hide its others (Sol, #157): ف→∅ rides on words that also
    carry another substitution."""
    flags = [record(f"{s}:1:1", sub, cf.GRADING, f"m:{s}") for s in range(1, 4) for sub in ("ف→∅", "ب→∅")]
    flags += [record("9:1:1", "ء→∅", cf.GRADING, "m:9")]
    assert [c["substitution"] for c in cf.owner_checks(flags)[cf.GRADING]] == ["ب→∅", "ف→∅", "ء→∅"]


def test_a_private_populations_checklist_holds_word_references_only():
    flags = [record("2:19:9", "ج→د", cf.MODEL, "69#1", population=cf.OWNER)]
    (entry,) = cf.owner_checks(flags)[cf.MODEL]
    assert entry["words"] == [{"word": "2:19:9", "rule": "pronounced"}]


def test_repeated_recitals_count_once_each_in_numerator_and_denominator():
    """16:80:15 recited in two items of one surah is two occurrences, as the graded words count it."""
    graded = {"m:16": 10}
    flags = (record("16:80:15", "ظ→ض", cf.MODEL, "m:16", occurrence="a#0:w15"),
             record("16:80:15", "ظ→ض", cf.MODEL, "m:16", occurrence="b#1:w15"))
    pop = cf.Population(cf.QUD, "h448", "shipped_default", graded, False, flags)
    assert cf.rate_rows(pop, cf.MODEL, by_expected=False)[0]["words"] == 2


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
    assert "| QUD pilot | h448 | shipped_default | ن→ل | 2 | 0 | 0 | 0 | 2 |" in doc
    assert "(`ن→ل`, `ل→ر`, `م→ب`): 3 of 3 of their errors sit" in doc
    assert "grading (Muraja word scorer) 1; reference (Muraja word scorer) 1; unclear (none named) 1" in doc


def test_the_committed_report_and_doc_keep_the_private_sessions_to_word_references():
    """#151's logs and #153's sessions are private: their records, views, checklists and doc
    sections hold no Arabic but a label or a one-letter expected letter (the invariant of
    ``test_device_logs`` and ``test_device_exports``)."""
    import json
    import re

    from training.test_device_logs import arabic_beyond_labels

    report = json.loads(cf.REPORT_PATH.read_text(encoding="utf-8"))
    private = [part for key in ("records", "populations", "owner_checks") for part in report[key]
               if part["population"] in cf.PRIVATE]
    assert private and not arabic_beyond_labels(private, longest=1)
    assert all("printed" not in r and "context" not in r and "decode" not in r
               for r in report["records"] if r["population"] in cf.PRIVATE)
    doc = cf.DOC_PATH.read_text(encoding="utf-8")
    sections = [m for name in cf.PRIVATE for pattern in (rf"\n## {re.escape(name)}\n(.*?)(?=\n## )",
                                                       rf"\n### {re.escape(name)}, [^\n]*\n(.*?)(?=\n##)")
                for m in re.findall(pattern, doc, flags=re.S)]
    assert len(sections) == 2 * len(cf.PRIVATE) and not arabic_beyond_labels(sections, longest=1)


def test_the_doc_is_rendered_from_the_committed_report():
    import json

    assert cf.DOC_PATH.read_text(encoding="utf-8") == render(json.loads(cf.REPORT_PATH.read_text(encoding="utf-8")))


QURAN_DB = Path(__file__).parent.parent.parent / "data" / "quran.db"


@pytest.mark.skipif(not QURAN_DB.exists(), reason="data/quran.db is not built here")
def test_the_forms_of_a_real_ayah_say_which_letters_a_pause_or_a_start_drops():
    pytest.importorskip("quran_transcript")
    from training.muraja_policy import MurajaText

    text = MurajaText(QURAN_DB)
    assert text.text_words(109, 6, 3) == [3]
    from training.consonant_flag_populations import Ayat

    ayat = Ayat(text)
    ayah = ayat.get(109, 6)  # لَكُمۡ دِينُكُمۡ وَلِيَ دِينِ
    assert ayah.phonemes == "لَكُم دِۦۦنُكُم وَلِيَ دِۦۦن"
    assert ayah.slips == {}
    # the lam of dinukum's neighbour is spelled as itself, voweled
    assert ayah.spelled[ayah.ordinal(cf.word_start(ayah.phonemes, 3) + 2)] == cf.ITSELF_MARKED
    # a pause on وَلِيَ makes its yaa a madd; a start at the last word keeps all of it
    yaa = cf.word_start(ayah.phonemes, 3) + 4
    reading, _ = cf.classify(ayah, 3, mismatch("ي", "ۦ", 2), cf.Recital(), OTHER_CONSONANT)
    assert (ayah.phonemes[yaa], reading.rule) == ("ي", f"wasl-dependent: dropped on {cf.WAQF_HERE}")
    assert cf.classify(ayah, 4, deletion("ن", 1), cf.Recital(verdict=cf.MODEL), OTHER_CONSONANT)[0].klass == cf.MODEL


def test_the_pinned_engine_aligns_the_stop_with_the_kept_overlap():
    """The engine itself, through the harness: a stop with no phonemes after a preview of 2:80's
    first three words runs one more alignment over the preview it kept, which moves the reader
    onto word 3; after an empty update, which clears the overlap, the same stop aligns nothing
    and the reader stays on word 1."""
    from muraja_harness.make_parity_fixtures import BALANCED, preview, words
    from training.muraja_policy import Harness

    harness = Harness.locate()
    if harness is None:
        pytest.skip("the Muraja harness is not built here")
    stop = {"hop": "", "overlap": "", "flush": False, "stop": True}
    empty = {"hop": "", "overlap": "", "flush": False}

    def positions(checks: list[dict]) -> list[str]:
        request = {"item": "kept overlap", "surah": 2, "ayah": 80, "start_word": 1, "report_words": [1, 2, 3],
                   "checks": checks, "scorings": [BALANCED]}
        _, (result,) = harness.run([request])
        return [c["position"] for c in result["checks"]]

    assert positions([preview(words(1, 3)), stop]) == ["2:80:1", "2:80:3"]
    assert positions([preview(words(1, 3)), empty, stop]) == ["2:80:1", "2:80:1", "2:80:1"]


@pytest.mark.skipif(not QURAN_DB.exists(), reason="data/quran.db is not built here")
@pytest.mark.parametrize("slug, model, item, word, error, check, klass", [
    # Sol, #157 round 3: the setting check's own hop is pending; with it folded into the confirmed
    # tail, 4:49:14 read as unclear (it is grading) and 38:22:19 as model (it is unclear)
    ("anas_almiman_yt", "h448", "qud_anas_almiman_yt_004_049.mp3#1", 14, mismatch("ل", "ن", 2), 42, cf.GRADING),
    ("abdul_hamid_ghraio_2026_yt", "base", "qud_abdul_hamid_ghraio_2026_yt_038_022.mp3#0", 19,
     mismatch("ط", "ق", 3), 62, cf.UNCLEAR),
])
def test_the_committed_qud_checks_read_as_the_engine_aligned_them(slug, model, item, word, error, check, klass):
    """The two QUD records, from the committed check sequence, read at the check that set the
    kept grade."""
    pytest.importorskip("quran_transcript")
    from tadabur.qud_text import item_reference
    from tadabur.waqf_segments import _uthmani_words, hafs_segment_reference
    from training.muraja_policy import MurajaText
    from training.qud_pilot import read_decodes
    from training.word_flags import TARGET_PAIR

    line = read_decodes(slug, model)[item]
    surah, ayah = map(int, line["surah_ayah"].split(":"))
    realized = item_reference(line["segment"], _uthmani_words(line["surah_ayah"]), hafs_segment_reference())
    muraja = cf.Ayah(MurajaText(QURAN_DB).phonemes(surah, ayah))
    recital = cf.Recital(realized.reference, checks=tuple(line["checks"]), setting=lambda position: (check,))
    cause = TARGET_PAIR if error["kind"]["expected"] == "ط" else OTHER_CONSONANT
    assert cf.classify(muraja, word, error, recital, cause)[0].klass == klass

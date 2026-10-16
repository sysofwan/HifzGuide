"""Tests for training.nonpro_eval: the bench's letters, the qdat and RetaSy views (#156).

Torch-free and harness-free: decodes, items and word grades are synthetic.
"""

from __future__ import annotations

import json

import pytest

from tadabur.qud_text import ItemReference
from training import nonpro_eval as ne
from training.external_corpus import WordGrade
from training.muraja_policy import SiteGrade
from training.word_flags import Flag

# Two words, ``قَالَ سَططَ``: a ق, a ل, a س and the geminate طط, each with a haraka.
REFERENCE = "قَالَ سَططَ"
OFFSETS = (0, 6, 11)


def _item(key="c#0", reciter="s1", labels=None) -> ne.ClipItem:
    return ne.ClipItem(key, "qdat", key.split("#")[0], 5, 109, 0, 16000, "e" * 64,
                       ItemReference(("قَالَ", "سَططَ"), (7, 8), REFERENCE, OFFSETS), reciter, 0,
                       labels or {})


def _grade(item, word_index, flag=None, model="h448", scoring="shipped_default", graded=True):
    return WordGrade(model, scoring, item, word_index - 7, 2, word_index + 1, word_index,
                     SiteGrade("flagged" if flag else "not_flagged" if graded else "not_graded"), flag)


def _flag(*consonants) -> Flag:
    return Flag(tuple(sorted({c for c, _ in consonants})), (), "minor", (),
                consonants=tuple(consonants))


def test_letters_agree_differ_or_are_left_empty_per_kind():
    out = ne.letter_outcomes("قَالِ سَطَ", REFERENCE)  # a kasra for a fatha; the طط collapsed
    assert out["consonant"] == [(0, ne.AGREE), (3, ne.AGREE), (6, ne.AGREE), (8, ne.AGREE)]
    assert dict(out["haraka"]) == {0: ne.AGREE, 3: ne.OTHER, 6: ne.AGREE, 9: ne.AGREE}
    assert out["shaddah"] == [(8, ne.OTHER)]
    assert ne.letter_outcomes("قَالَ", REFERENCE)["consonant"][2:] == [(6, ne.NONE), (8, ne.NONE)]


def test_the_tally_separates_agreement_from_committed_agreement():
    tally = ne._tally([ne.letter_outcomes("لَ شَططَ", REFERENCE)])  # no ق; ش for س
    assert tally["consonant"] == {"agree": 2, "other": 1, "none": 1, "agreement": 0.5,
                                  "agreement_committed": 0.6667}


def test_a_departure_is_placed_on_the_canonical_word_it_sits_in():
    assert ne.bench_departures(_item(), REFERENCE) == set()
    assert ne.bench_departures(_item(), "قَالَ شَططَ") == {8}  # the س of word 9
    assert ne.bench_departures(_item(), "قِالَ سَططَ") == {7}  # a haraka in word 8


def test_an_insertion_an_added_haraka_and_added_gemination_are_departures():
    """What the mushaf's own slots cannot see (Sol, #158): each departs at its word."""
    assert ne.bench_departures(_item(), "قَالَ سَبَططَ") == {8}  # an inserted ب
    assert ne.bench_departures(_item(), "قَالَ سَططَ بَ") == {8}  # an inserted word
    assert ne.bench_departures(_item(), "قَالَا سَططَ") == {7}  # an added madd
    assert ne.bench_departures(_item(), "قَالَ سَطَطَ") == {8}  # a haraka on the bare first ط
    assert ne.bench_departures(_item(), "قَاللَ سَططَ") == {7}  # added gemination
    assert ne.bench_departures(_item(), "قَالَ سَطَ") == {8}  # dropped gemination


def test_tajweed_lengths_and_the_hidden_noon_are_not_departures():
    assert ne.mushaf_form("لَنَاااا") == ne.mushaf_form("لَنَاا") == "لَنَا"
    assert ne.mushaf_form("ءِننننَكَ") == ne.mushaf_form("ءِننَكَ") != ne.mushaf_form("ءِنَكَ")
    assert ne.mushaf_form("ءَںںںتَ") == ne.mushaf_form("ءَنننتَ") == ne.mushaf_form("ءَنتَ")
    assert ne.mushaf_form("لغُيُۥۥبڇ") == ne.mushaf_form("لغُيُۥۥۥۥب")
    assert ne.mushaf_form("قَالُ") != ne.mushaf_form("قَاالُۥۥ")  # a madd dropped whole


def test_qdat_counts_reciters_per_substitution_and_splits_by_tajweed_label():
    a, b = _item("a#0", "s1", {"separate_tide": 1, "the_tight_noon": 1, "concealment": 0,
                               "target": 0}), \
        _item("b#0", "s2", {"separate_tide": 0, "the_tight_noon": 1, "concealment": 0,
                            "target": 0})
    nl = ("other_consonant", "ن→ل")
    grades = [_grade(a, 7, _flag(nl)), _grade(a, 8), _grade(b, 7, _flag(nl)),
              _grade(b, 8, _flag(("other_consonant", "س→ش")))]
    views = ne.qdat_views(grades)
    r = views["per_reciter"]["h448"]["shipped_default"]
    assert (r["reciters"], r["reciters_flagged"]) == (2, 2)
    assert r["by_substitution"][0] == {"cause": "other_consonant", "substitution": "ن→ل",
                                      "reciters": 2, "words": 2}
    assert r["by_cause"] == [{"cause": "other_consonant", "reciters": 2, "words": 3}]
    label = views["per_tajweed_label"]["h448"]["shipped_default"]["separate_tide"]
    assert (label["0"]["flagged"], label["0"]["graded"]) == (2, 2)
    assert (label["1"]["flagged"], label["1"]["graded"]) == (1, 2)
    word = views["per_word"]["h448"]["shipped_default"]
    assert [(w["canonical_word"], w["text"], w["flagged"]) for w in word] == \
        [(8, "قَالَ", 2), (9, "سَططَ", 1)]


def test_retasy_s_label_contrast_counts_clips_with_a_flag_per_label():
    ok, bad = _item("ok#0", labels={"final_label": "correct"}), \
        _item("bad#0", labels={"final_label": "in_correct"})
    grades = [_grade(ok, 7), _grade(ok, 8), _grade(bad, 7, _flag(("other_consonant", "ق→ك"))),
              _grade(bad, 8, graded=False)]
    view = ne.label_contrast(grades)["h448"]["shipped_default"]
    assert (view["correct"]["clips"], view["correct"]["clips_flagged"]) == (1, 0)
    assert (view["in_correct"]["clips_flagged"], view["in_correct"]["graded"]) == (1, 1)


def test_the_bench_s_app_view_places_each_flag_against_the_experts():
    item = _item("c#0")
    corpus = ne.Corpus({}, {"qdat": [item]}, {}, {})
    grades = [_grade(item, 7, _flag(("other_consonant", "ق→ك"))),
              _grade(item, 8, _flag(("target_pair", "س→ش")))]
    view = ne.bench_app(grades, corpus, {"c": "قَالَ شَططَ"})["h448"]["shipped_default"]
    assert view["flagged_where_experts_heard_a_departure"] == 1  # the س the experts heard as ش
    assert view["flagged_where_experts_heard_the_mushaf"] == 1


def test_a_reciter_clusters_its_clips_and_tarteel_clusters_by_surah():
    lines = [{"key": "x", "reciter": "s9", "surah_ayah": "2:3"},
             {"key": "y", "reciter": "s1", "surah_ayah": "2:4"}]
    assert ne.clusters(ne.QDAT, lines) == {"x": 1, "y": 0}
    assert ne.clusters(ne.DATASETS["tarteel_v1"], lines) == {"x": 2, "y": 2}


def test_restricted_datasets_write_under_the_ignored_directory():
    assert ne.part_dir(ne.RETASY).parent == ne.RESTRICTED_DIR
    assert ne.part_dir(ne.QDAT).parent == ne.NONPRO_DIR


@pytest.mark.skipif(not ne.REPORT_PATH.exists(), reason="the pilot's report is not built yet")
def test_the_committed_doc_is_what_the_renderer_writes():
    from training.nonpro_eval_doc import render

    report = json.loads(ne.REPORT_PATH.read_text(encoding="utf-8"))
    overlap = json.loads(ne.OVERLAP_PATH.read_text(encoding="utf-8"))
    assert ne.DOC_PATH.read_text(encoding="utf-8") == render(report, overlap)


def test_a_restricted_dataset_is_committed_as_aggregates_and_clip_ids_only():
    """RetaSy's licence is unknown and qdat_bench's unstated (#156): no committed file under
    ``nonpro_eval/`` (nor the doc) carries a restricted clip's checksum, its id outside the
    overlap sample's id list, or a per-clip result; their per-clip outputs are never tracked."""
    import subprocess

    from tadabur.exposure import load_registry
    from tadabur.nonpro_sources import DATASETS

    restricted = {d.repo for d in DATASETS.values() if not d.open}
    rows = [r for rows in load_registry().clips.values() for r in rows
            if r.dataset in restricted and r.dataset != DATASETS["qdat_bench"].repo]
    assert rows  # RetaSy's clips are registered: ids and checksums are provenance
    root = ne.DOC_PATH.parent.parent
    tracked = subprocess.run(["git", "ls-files", str(ne.NONPRO_DIR)], capture_output=True,
                             text=True, check=True, cwd=root).stdout.split()
    assert tracked and not any("/restricted/" in path for path in tracked)
    texts = {p: (root / p).read_text(encoding="utf-8")
             for p in tracked if p.endswith((".json", ".jsonl", ".md"))}
    texts["doc"] = ne.DOC_PATH.read_text(encoding="utf-8")
    overlap = json.loads(ne.OVERLAP_PATH.read_text(encoding="utf-8"))
    for name, dataset in DATASETS.items():
        if not dataset.open and name in overlap["datasets"]:
            assert all(isinstance(c, str) for c in overlap["datasets"][name]["clips"])
    for row in rows:
        for path, text in texts.items():
            assert row.audio_sha256 not in text, path
            if not path.endswith("overlap.json"):
                assert row.audio_filename not in text, path

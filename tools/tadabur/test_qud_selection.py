"""Tests for tadabur.qud_selection: the rules, the name tables and the committed files."""

from __future__ import annotations

import json

import pytest

from tadabur import qud_selection as qs

SOURCES = json.loads(qs.SOURCES_PATH.read_text(encoding="utf-8"))


def _row(**overrides) -> dict:
    row = {"slug": "x_yt", "reciter_id": "x", "name_ar": "س", "riwayah": qs.HAFS,
           "style": "Murattal", "recording_year": 2026, "total_duration_hours": 1.5}
    return {**row, **overrides}


def _sources(catalog: list[dict]) -> dict:
    return {
        "qud": {"revision": "r", "catalog": catalog,
                "shards": {r["slug"]: {f"d/{r['slug']}-00000-of-00001.parquet": 10}
                           for r in catalog}},
        "muaalem": {"reciters": [{"id": 0, "english_name": "Husari", "arabic_name": "ح"}],
                    "moshafs": [{"id": "0.0", "reciter_id": 0}]},
        "tadabur": {"sheikh_dict": {"عبدالرحمن  السديس": 15}},
    }


def test_normalize_name_folds_spacing_hamza_and_harakat():
    assert qs.normalize_name("عبد الرحمن السُّدَيس") == qs.normalize_name("عبدالرحمن السديس")
    assert qs.normalize_name("أحمد") == qs.normalize_name("احمد")
    assert qs.normalize_name("مصطفى") == qs.normalize_name("مصطفي")


@pytest.mark.parametrize(("overrides", "decision"), [
    ({"style": "Kids Repeat", "reciter_id": "mahmoud_khalil_al_husary"}, qs.EXCLUDED),
    ({"reciter_id": "mahmoud_khalil_al_husary", "recording_year": 2026}, qs.EXCLUDED),
    ({"recording_year": None}, qs.ELIGIBLE),
    ({"recording_year": qs.RECENT_FROM_YEAR - 1}, qs.ELIGIBLE),
    ({"recording_year": qs.RECENT_FROM_YEAR}, qs.SELECTED),
])
def test_decide_applies_the_rules_in_order(overrides, decision):
    assert qs.decide(_row(**overrides))[0] == decision


def test_select_lists_hafs_only_with_screens_and_totals():
    catalog = [
        _row(slug="a_yt", reciter_id="abdulrahman_al_sudais", name_ar="عبد الرحمن السديس"),
        _row(slug="b_yt", reciter_id="mahmoud_khalil_al_husary", name_ar="محمود خليل الحصري"),
        _row(slug="c_warsh", riwayah="Warsh A'n Nafi'"),
    ]
    sources = _sources(catalog)
    sources["muaalem"]["reciters"] = [{"id": 0, "english_name": "H", "arabic_name": "ح"}]
    selection = qs.select(sources, {15: {"h448.training": 970}})
    assert [m["slug"] for m in selection["mushafs"]] == ["a_yt", "b_yt"]
    a, b = selection["mushafs"]
    assert a["decision"] == qs.SELECTED and a["muaalem"] is None
    assert a["tadabur_name_screen"] == {"tadabur_id": 15, "uses": {"h448.training": 970}}
    assert b["decision"] == qs.EXCLUDED and b["muaalem"]["moshafs"] == ["0.0"]
    assert b["tadabur_name_screen"] is None
    assert selection["non_hafs_mushafs"] == 1
    assert selection["totals"][qs.SELECTED] == {"mushafs": 1, "reciters": 1, "hours": 1.5,
                                                "download_bytes": 10}


def test_every_muaalem_reciter_is_matched_or_listed_absent_once():
    ids = sorted(r["id"] for r in SOURCES["muaalem"]["reciters"])
    mapped = sorted(qs.MUAALEM_RECITER_OF.values())
    assert sorted(mapped + list(qs.MUAALEM_RECITERS_NOT_IN_QUD)) == ids
    qud_reciters = {r["reciter_id"] for r in SOURCES["qud"]["catalog"]}
    assert set(qs.MUAALEM_RECITER_OF) <= qud_reciters


def test_no_qud_reciter_with_a_muaalem_name_escapes_the_table():
    muaalem = {qs.normalize_name(r["arabic_name"]): r["id"] for r in SOURCES["muaalem"]["reciters"]}
    for row in SOURCES["qud"]["catalog"]:
        match = muaalem.get(qs.normalize_name(row["name_ar"]))
        if match is not None:
            assert qs.MUAALEM_RECITER_OF.get(row["reciter_id"]) == match, row["slug"]


def test_held_out_moshafs_exist_in_the_snapshot():
    moshafs = {m["id"] for m in SOURCES["muaalem"]["moshafs"]}
    assert set(qs.MUAALEM_HELD_OUT_MOSHAFS) <= moshafs


def test_committed_selection_follows_from_the_committed_sources():
    """Everything but the screen's exposure uses, which record the registry when ``select``
    ran and grow as later issues register recordings."""
    committed = json.loads(qs.SELECTION_PATH.read_text(encoding="utf-8"))
    uses_of = {m["tadabur_name_screen"]["tadabur_id"]: m["tadabur_name_screen"]["uses"]
               for m in committed["mushafs"] if m["tadabur_name_screen"]}
    rebuilt = json.loads(json.dumps(qs.select(SOURCES, uses_of), ensure_ascii=False))
    assert rebuilt == committed


def test_ranges_compacts_sorted_runs():
    assert qs.ranges([3, 1, 2, 7, 9, 8, 5]) == "1-3,5,7-9"
    assert qs.ranges([4]) == "4"
    assert qs.ranges([]) == ""


def test_summarize_coverage_counts_rows_ayat_and_sources():
    rows = [
        {"surah": 1, "ayah": 1, "duration_ms": 100, "source_url": "u1"},
        {"surah": 1, "ayah": 2, "duration_ms": 200, "source_url": "u1"},
        {"surah": 1, "ayah": 2, "duration_ms": 50, "source_url": "u1"},
        {"surah": 114, "ayah": 6, "duration_ms": 10, "source_url": "u2"},
    ]
    summary = qs.summarize_coverage(rows)
    assert summary["rows"] == 4 and summary["ayat"] == 3 and summary["duration_ms"] == 360
    assert summary["surahs"]["1"] == {"ayat": "1-2", "rows": 3, "duration_ms": 350,
                                      "sources": ["u1"]}
    assert list(summary["surahs"]) == ["1", "114"]


def test_date_check_sample_is_evenly_spaced_from_the_first():
    urls = [f"u{i:02d}" for i in range(12)]
    assert qs.date_check_sample(urls, 4) == ["u00", "u03", "u06", "u09"]
    assert qs.date_check_sample(urls[:2], 4) == ["u00", "u01"]


def test_youtube_fields_reads_title_and_upload_date():
    html = ('<title>سورة ق | رمضان ١٤٤٧ هـ - YouTube</title>'
            '"uploadDate":"2026-03-19T06:33:57-07:00"')
    assert qs.youtube_fields(html) == {"title": "سورة ق | رمضان ١٤٤٧ هـ",
                                       "upload_date": "2026-03-19T06:33:57-07:00"}
    assert qs.youtube_fields("<html></html>") == {"title": None, "upload_date": None}

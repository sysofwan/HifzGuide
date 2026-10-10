"""#61's Ship A worklist (#126): its composition, its freeze and its sittings."""

from __future__ import annotations

import json
from dataclasses import replace

from tadabur.listening_session import WORKLIST_PATH as SESSION_WORKLIST_PATH
from tadabur.listening_session import load_worklist
from tadabur.ship_a import load_sites, mine
from tadabur.ship_a_worklist import (
    EDIT_CHECK,
    GEMINATE,
    P35,
    PAIR_PARTNER,
    PARTS,
    REJECT_PILE,
    SITTINGS,
    SUMMARY_PATH,
    WORDS,
    WORKLIST_PATH,
    assemble,
    sha256_of,
)
from tadabur.test_power_frame import REFERENCE, scored_pool
from tadabur.word_sample import WORD_SAMPLE_PATH, load_words


def _carried(row, site_id: str, stratum: str):
    return replace(row, site=replace(row.site, site_id=site_id, stratum=stratum,
                                     stratum_population=5))


def test_the_worklist_takes_the_listened_sites_and_carries_p35_and_the_reject_pile():
    sites = mine(scored_pool(base=REFERENCE.replace("بب", "ب")))
    pair = next(s for s in sites if s.kind == "pair")
    session = [
        _carried(pair.row, "p35_fixture:x", "p35_fixture:ذ↔ز"),
        _carried(pair.row, pair.site.site_id, "new_audit:ذ→ز:base_partner"),
        _carried(pair.row, "new_audit:other", "new_audit:ذ→ز:base_partner"),
        _carried(pair.row, "new_audit:tashkeel", "new_audit:fatha:base_empty:h448_empty"),
    ]
    rows, parts, duplicates = assemble(sites, session)
    assert duplicates == 1
    assert parts == {pair.site.site_id: PAIR_PARTNER,
                     next(s for s in sites if s.kind == "geminate").site.site_id: GEMINATE,
                     "p35_fixture:x": P35, "new_audit:other": REJECT_PILE}
    assert [row.site.site_id for row in rows] == list(parts)


def test_a_held_geminate_stays_off_the_worklist():
    sites = mine(scored_pool())
    rows, parts, _ = assemble(sites, [])
    assert set(parts.values()) == {PAIR_PARTNER} and len(rows) == 1


def test_the_committed_worklist_is_frozen_and_its_summary_adds_up():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["frozen"] == {"ship_a/worklist.jsonl": sha256_of(WORKLIST_PATH),
                                 "ship_a/word_sample.jsonl": sha256_of(WORD_SAMPLE_PATH)}
    rows, words = load_worklist(WORKLIST_PATH), load_words()
    expected, parts, _ = assemble(load_sites(), load_worklist(SESSION_WORKLIST_PATH))
    assert [row.site.site_id for row in rows] == [row.site.site_id for row in expected]
    by_part = summary["parts"]
    assert set(by_part) == set(PARTS)
    assert by_part[WORDS]["rows"] == len(words)
    assert sum(by_part[p]["rows"] for p in PARTS if p not in (WORDS, EDIT_CHECK)) == len(rows)
    assert summary["rows"] == sum(entry["rows"] for entry in by_part.values())
    sittings = summary["sittings"]
    assert len(sittings) == SITTINGS
    assert sittings[0]["queue_positions"][0] == 1
    assert sittings[-1]["queue_positions"][1] == summary["rows"]
    for before, after in zip(sittings, sittings[1:]):
        assert after["queue_positions"][0] == before["queue_positions"][1] + 1
    assert sum(s["rows"] for s in sittings) == summary["rows"]

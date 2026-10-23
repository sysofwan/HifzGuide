"""The consonant round's unread QUD half (#175, set G1 from #174).

The complement of #95's salted half under the same salt: 10,261 decoded segments, disjoint from
the kept half and covering the pilot between them.
"""

from __future__ import annotations

import json

import pytest

from tadabur.qud_overlap import PILOT
from training import aug_readout as ar
from training import consonant_readout as cr
from training.qud_pilot import read_decodes


def _pilot_keys() -> set[str]:
    return {key for slug in PILOT for key in read_decodes(slug, "base")}


def test_the_consonant_qud_half_is_the_complement_of_95_s():
    kept = ar.load_qud_selection()
    other = cr.load_qud_complement()
    assert not kept & other  # disjoint
    assert kept | other == _pilot_keys()  # together they cover every decoded segment
    assert len(other) == 10_261


def test_the_committed_record_is_drawn_under_the_aug95_rule():
    record = json.loads(cr.QUD_SELECTION_PATH.read_text(encoding="utf-8"))
    assert (record["salt"], record["cutoff"]) == (ar.QUD_SALT, ar.QUD_CUTOFF)
    assert sum(record["kept_counts"].values()) == 10_261
    assert all(not ar.qud_kept(key) for keys in record["kept"].values() for key in keys)


def test_a_record_holding_a_kept_key_is_refused(tmp_path):
    record = json.loads(cr.QUD_SELECTION_PATH.read_text(encoding="utf-8"))
    slug = sorted(record["kept"])[0]
    record["kept"][slug].append(sorted(ar.load_qud_selection())[0])
    path = tmp_path / "qud_selection.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="another half"):
        cr.load_qud_complement(path)

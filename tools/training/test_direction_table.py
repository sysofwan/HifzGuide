"""The decode-level direction table (#90): directions, sides, families and the rates of one arm,
on synthetic sites and decodes. Torch-free."""

from __future__ import annotations

import pytest

from training import direction_table as d
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE
from training.test_site_outcomes import DAL, DHAKARA, DHAL, FATHA, MADDA, MEEM, RAA, ZAI, make
from training.truth_scorer import item_key, scored_sites

ZAKARA = DHAKARA.replace(DHAL, ZAI)  # the partner at the carrier
MADA_DECODE = MEEM + FATHA + DAL + FATHA + RAA + FATHA  # the geminate collapsed


def _site(clip: str, reference: str, mark: str, prescribed: str, heard: str):
    return make(reference, 2, mark, prescribed, heard, site_id=f"{clip}:{mark}",
                audio_filename=clip, stratum=f"s:{mark}", stratum_population=10)


def _rows(sites, decodes_by_clip, weights=None):
    decodes = {"m/spans": {item_key(s): decodes_by_clip.get(s.audio_filename, s.reference)
                           for s in sites}}
    reciters = {s.audio_filename: n for n, s in enumerate(sites)}
    return scored_sites(sites, reciters, decodes, design_weights=weights or {
        s.site_id: 1.0 for s in sites}).rows


def test_a_direction_names_the_mushaf_letter_then_its_partner():
    site = _site("a.wav", DHAKARA, "ذ↔ز", DHAL, DHAL)
    assert d.direction(site) == "ذ→ز" and d.partner(site) == ZAI
    assert d.direction(_site("b.wav", MADDA, "shaddah", "held", "held")) == d.GEMINATE
    assert d.direction(make(DHAKARA, 0, "fatha", "fatha", "fatha")) is None


def test_every_target_direction_and_side_has_a_row_and_soft_pairs_pool_apart():
    directions = {k.direction for k in d.KEYS}
    assert len(directions - {d.ALL_DIRECTIONS, d.GEMINATE}) == 18
    assert {k.family for k in d.KEYS} == {d.SOFT_PAIRS_FAMILY, "ذ↔ظ", "س↔ش", "ط↔ق", d.GEMINATE}
    assert all(k.side in (CORRECT_SIDE, MISTAKE_SIDE) for k in d.KEYS)
    # ط→ق pools with its own family, never with the soft pairs
    assert "ط→ق" in d.FAMILIES["ط↔ق"] and "ط→ق" not in d.FAMILIES[d.SOFT_PAIRS_FAMILY]
    assert "ط→ت" in d.FAMILIES[d.SOFT_PAIRS_FAMILY]


def test_swap_is_a_mishearing_on_the_correct_side_and_a_catch_on_the_mistake_side():
    sites = [_site("a.wav", DHAKARA, "ذ↔ز", DHAL, DHAL),   # correct, decoded ز: swapped
             _site("b.wav", DHAKARA, "ذ↔ز", DHAL, DHAL),   # correct, decoded ذ
             _site("c.wav", DHAKARA, "ذ↔ز", DHAL, ZAI),    # mistake, decoded ز: caught
             _site("d.wav", MADDA, "shaddah", "held", "held")]  # correct, collapsed
    rows = _rows(sites, {"a.wav": ZAKARA, "c.wav": ZAKARA, "d.wav": MADA_DECODE})
    table = {(e["direction"], e["side"]): e for e in d.table(rows, ["m/spans"])}
    correct = table[("ذ→ز", CORRECT_SIDE)]
    assert (correct["sites"], correct["insufficient_evidence"]) == (2, True)
    assert correct["arms"]["m/spans"]["swap"]["point"] == pytest.approx(0.5)
    assert correct["arms"]["m/spans"]["mushaf"]["point"] == pytest.approx(0.5)
    assert table[("ذ→ز", MISTAKE_SIDE)]["arms"]["m/spans"]["swap"]["point"] == 1.0
    assert table[(d.GEMINATE, CORRECT_SIDE)]["arms"]["m/spans"]["swap"]["point"] == 1.0
    assert "arms" not in table[("ز→ذ", CORRECT_SIDE)]  # no site: no rates
    pooled = next(e for e in d.table(rows, ["m/spans"]) if (e["family"], e["direction"],
                  e["side"]) == (d.SOFT_PAIRS_FAMILY, d.ALL_DIRECTIONS, CORRECT_SIDE))
    assert pooled["sites"] == 2


def test_rates_read_the_design_weights_and_a_difference_is_paired():
    sites = [_site("a.wav", DHAKARA, "ذ↔ز", DHAL, DHAL), _site("b.wav", DHAKARA, "ذ↔ز", DHAL, DHAL)]
    rows = _rows(sites, {"a.wav": ZAKARA}, {sites[0].site_id: 3.0, sites[1].site_id: 1.0})
    assert d.rates(rows, "m/spans")["swap"]["point"] == pytest.approx(0.75)
    assert d.difference(rows, "m/spans", "m/spans")["point"] == 0.0

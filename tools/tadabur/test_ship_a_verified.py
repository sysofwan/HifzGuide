"""Ship A's verdicts folded into truth sites (#90): frames, halves, design weights, and the
committed files against the committed worklist and verdicts. Torch-free."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tadabur import ship_a_verified as v
from tadabur.listening_session import Verdict, read_verdicts
from tadabur.ship_a import HALVES, reciter_half
from tadabur.truth_sites import NEW_AUDIT, P35_FIXTURE, load_truth_sites
from tadabur.word_sample import WORD_VERDICTS, load_words, verified_words
from training.test_site_outcomes import DHAKARA, DHAL, ZAI, make


def _row(n: int, stratum: str, source: str = NEW_AUDIT):
    return make(DHAKARA, 2, "ذ↔ز", DHAL, "pending", site_id=f"{source}:{n}", source=source,
                audio_filename=f"c{n}.wav", stratum=stratum, stratum_population=10)


def test_a_ship_a_site_and_a_reject_pile_site_each_keep_their_listening_chance():
    rows = [_row(1, "ship_a:ذ→ز:h448_partner:base_right"), _row(2, "new_audit:ذ→ز:base_partner"),
            _row(3, "p35_fixture:ذ↔ز", source=P35_FIXTURE)]
    verdicts = {r.site_id: Verdict(r.site_id, h) for r, h in zip(rows, (DHAL, ZAI, DHAL))}
    sites, designs = v.fold(rows, {rows[0].site_id: 0.25}, {rows[1].site_id: 0.5},
                            {"c1.wav": 7, "c2.wav": 8}, verdicts)
    assert [s.site_id for s in sites] == [rows[0].site_id, rows[1].site_id]  # no P3.5 row
    assert [s.heard for s in sites] == [DHAL, ZAI]
    assert [(d.frame, d.half, d.weight) for d in designs] == [
        (v.SHIP_A, reciter_half(7), 4.0), (v.REJECT_PILE, reciter_half(8), 2.0)]


def test_a_row_without_a_verdict_or_a_frame_is_refused():
    rows = [_row(1, "ship_a:x")]
    with pytest.raises(ValueError, match="no verdict"):
        v.fold(rows, {rows[0].site_id: 1.0}, {}, {"c1.wav": 1}, {})
    with pytest.raises(ValueError, match="neither"):
        v.fold(rows, {}, {}, {"c1.wav": 1}, {rows[0].site_id: Verdict(rows[0].site_id, DHAL)})


def test_a_design_refuses_an_impossible_chance():
    with pytest.raises(ValueError):
        v.Design("x", v.SHIP_A, HALVES[0], 0.0)
    with pytest.raises(ValueError):
        v.Design("x", "elsewhere", HALVES[0], 1.0)


def test_the_committed_fold_is_the_build_of_the_committed_inputs():
    sites, designs = v.load_verified()
    built_sites, built_designs = v.build()
    assert sites == built_sites and designs == {d.site_id: d for d in built_designs}
    assert len(sites) == 530 and {d.frame for d in designs.values()} == set(v.FRAMES)
    assert all(s.heard != "pending" for s in sites)
    assert load_truth_sites(v.SITES_PATH) == sites


def test_every_sampled_word_has_a_whole_word_verdict():
    words = verified_words(load_words(), read_verdicts())
    assert len(words) == 160 and {w.heard for w in words} <= set(WORD_VERDICTS)
    assert all(w.weight == 1 / w.row.draw_probability for w in words)
    first = load_words()[0]
    with pytest.raises(ValueError, match="no whole-word verdict"):
        verified_words([first], {})
    with pytest.raises(ValueError, match="no whole-word verdict"):
        verified_words([first], {first.site.site_id: Verdict(first.site.site_id, "held")})
    assert replace(words[0], heard="unclear").side is None

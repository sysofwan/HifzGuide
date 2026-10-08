"""Tests for re-locating the P3.5 fixtures on re-derived segments (#83).

Every case is a synthetic decode against a synthetic realized reference: the rule is
exercised end to end with no model, phonetizer or audio.
"""

from __future__ import annotations

import json

import pytest

from tadabur.eval_fixtures import EvalFixtureEntry
from tadabur.p35_truth_sites import (
    DROPPED_CONTRAST_ABSENT,
    DROPPED_NOT_EXPRESSIBLE,
    DROPPED_NOT_STAGED,
    DROPPED_SEGMENT_DROPPED,
    DROPPED_SEGMENT_MISSING,
    DROPPED_UNSEGMENTED,
    EXCLUDED_MARGINAL,
    KEPT,
    Segment,
    convert,
    outcome_table,
    read_segments,
    relocate,
)
from tadabur.staged_audio import StagedClip

DHAL, ZAI, LAM, KAF, RAA, BAA = "ذ", "ز", "ل", "ك", "ر", "ب"
FATHA, KASRA, ALIF = "َ", "ِ", "ا"
GHUNNA_NOON, NOON = "ں", "ن"

CLIP = "tadabur_spk0001_S1_A2_x_000001.wav"
SHA = "b" * 64
# ذَالِكَ رَب: a ذ at index 0, the single ب at index 10.
REFERENCE = DHAL + FATHA + ALIF + ALIF + LAM + KASRA + KAF + FATHA + " " + RAA + FATHA + BAA
SAID_ZAI = ZAI + REFERENCE[1:]


def _fixture(contrast="ذ↔ز", verdict="accept", seg=1, note="") -> EvalFixtureEntry:
    clip_id = f"{CLIP[:-4]}__seg{seg}.wav"
    return EvalFixtureEntry(clip_id, clip_id, "2:2", contrast, verdict, note)


def _segment(decode=SAID_ZAI, reference=REFERENCE, index=1, kept=True) -> Segment:
    return Segment(CLIP, index, 1.0, 2.5, kept,
                   reference if kept else None, decode if kept else None,
                   () if kept else ("short_segment",))


def _staged() -> dict[str, StagedClip]:
    return {CLIP: StagedClip(CLIP, 5, 9, 33, "2:2", 48000, SHA, ("p35_fixture",))}


def _relocate(fixture, segment=None, staged=None):
    segments = {} if segment is None else {(CLIP, segment.segment_index): segment}
    return relocate(fixture, segments, _staged() if staged is None else staged, {CLIP})


def test_an_accept_whose_contrast_reappears_becomes_a_site_said_as_the_mushaf():
    r = _relocate(_fixture(), _segment())
    assert r.outcome == KEPT
    (site,) = r.sites
    assert (site.mark, site.reference_index, site.prescribed, site.heard) == ("ذ↔ز", 0, DHAL, DHAL)
    assert (site.shard, site.audio_sha256, site.audio_filename) == (5, SHA, CLIP)
    assert (site.start_sample, site.end_sample) == (16000, 40000)
    assert site.reference == REFERENCE and site.surah_ayah == "2:2"
    assert site.stratum == "p35_fixture:ذ↔ز" and not site.assumes_competent_reciter


def test_a_reject_waits_for_a_site_level_verdict_instead_of_asserting_the_other_letter():
    (site,) = _relocate(_fixture(verdict="reject", note="Not hafs"), _segment()).sites
    assert (site.prescribed, site.heard) == (DHAL, "pending")


def test_a_contrast_that_no_longer_appears_drops_the_fixture():
    r = _relocate(_fixture(), _segment(decode=REFERENCE))
    assert (r.outcome, r.sites) == (DROPPED_CONTRAST_ABSENT, ())


def test_only_the_labelled_bucket_counts():
    r = _relocate(_fixture(contrast="ق↔ك"), _segment())
    assert r.outcome == DROPPED_CONTRAST_ABSENT


def test_a_dropped_shaddah_is_a_held_site_and_an_added_one_a_not_held_site():
    single, geminate = RAA + FATHA + BAA + KASRA + LAM, RAA + FATHA + BAA + BAA + KASRA + LAM
    held = _relocate(_fixture(contrast="shadda"), _segment(decode=single, reference=geminate))
    (site,) = held.sites
    assert (site.mark, site.reference_index, site.prescribed, site.heard) == (
        "shaddah", 2, "held", "held")

    added = _relocate(_fixture(contrast="shadda", verdict="reject"),
                      _segment(decode=geminate, reference=single))
    (site,) = added.sites
    assert (site.reference_index, site.prescribed, site.heard) == (2, "not_held", "pending")


def test_a_carrier_the_schema_cannot_hold_is_not_expressible():
    # A ghunna noon folds onto ن for alignment, but it is not a consonant a mark sits on.
    reference = LAM + FATHA + GHUNNA_NOON + GHUNNA_NOON + KASRA + KAF
    decode = LAM + FATHA + NOON + KASRA + KAF
    r = _relocate(_fixture(contrast="shadda"), _segment(decode=decode, reference=reference))
    assert r.outcome == DROPPED_NOT_EXPRESSIBLE


@pytest.mark.parametrize(
    "fixture, segment, staged, outcome",
    [
        (_fixture(contrast="marginal"), _segment(), None, EXCLUDED_MARGINAL),
        (_fixture(), _segment(), {}, DROPPED_NOT_STAGED),
        (_fixture(seg=2), _segment(), None, DROPPED_SEGMENT_MISSING),
        (_fixture(), _segment(kept=False), None, DROPPED_SEGMENT_DROPPED),
    ],
)
def test_every_other_fixture_is_dropped_with_its_reason(fixture, segment, staged, outcome):
    r = _relocate(fixture, segment, staged)
    assert (r.outcome, r.sites) == (outcome, ())


def test_a_clip_with_no_segments_is_reported_as_not_segmented():
    r = relocate(_fixture(), {}, _staged(), segmented_clips=set())
    assert r.outcome == DROPPED_UNSEGMENTED


def test_a_fixture_naming_another_ayah_than_its_staged_clip_is_an_error():
    fixture = EvalFixtureEntry(f"{CLIP[:-4]}__seg1.wav", "x", "3:3", "ذ↔ز", "accept")
    with pytest.raises(ValueError, match="different ayat"):
        _relocate(fixture, _segment())


def test_convert_is_ordered_and_gives_each_stratum_its_census_population():
    second = Segment(CLIP, 0, 0.0, 1.0, True, REFERENCE, SAID_ZAI, ())
    segments = {(CLIP, 1): _segment(), (CLIP, 0): second}
    fixtures = [_fixture(seg=1), _fixture(seg=0, verdict="reject"), _fixture(contrast="marginal")]
    sites, relocations = convert(fixtures, segments, _staged(), {CLIP})

    assert [(r.fixture.clip_id[-8:], r.fixture.contrast) for r in relocations] == [
        ("seg0.wav", "ذ↔ز"), ("seg1.wav", "marginal"), ("seg1.wav", "ذ↔ز")]
    assert [s.heard for s in sites] == ["pending", DHAL]
    assert {s.stratum_population for s in sites} == {2}
    assert len({s.site_id for s in sites}) == 2
    table = outcome_table(relocations)
    assert "| `ذ↔ز` | accept | 1 | 0 | 1 | 1 |" in table


def test_segments_are_read_from_a_resegment_output(tmp_path):
    (tmp_path / "segment_manifest.jsonl").write_text(json.dumps({
        "clip_audio_filename": CLIP, "segment_index": 1,
        "raw_reference_phonemes": REFERENCE, "predicted_phonemes": SAID_ZAI,
    }) + "\n", encoding="utf-8")
    (tmp_path / "segmentation.jsonl").write_text(json.dumps({
        "audio_filename": CLIP, "drops": {"short_segment": 1}, "segments": [
            {"segment_index": 0, "start_s": 0.0, "end_s": 0.5, "kept": False},
            {"segment_index": 1, "start_s": 1.0, "end_s": 2.5, "kept": True},
        ]}) + "\n", encoding="utf-8")
    segments = read_segments(tmp_path)
    assert segments[(CLIP, 1)] == _segment()
    assert segments[(CLIP, 0)].drops == ("short_segment",) and segments[(CLIP, 0)].decode is None

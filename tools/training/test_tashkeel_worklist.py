"""Tests for the mined tashkeel audit worklist (#60)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from training.tashkeel_eval import (
    DAMMA,
    FATHA,
    KASRA,
    MATCHED,
    OMITTED,
    SPURIOUS,
    SWAPPED,
    count_sites,
    score_vowels,
    vowel_sites,
)
from training.tashkeel_worklist import (
    RECOVERED,
    REGRESSED,
    TashkeelSite,
    discordant_sites,
    population_counts,
    read_worklist,
    sample_worklist,
    site_id,
    write_worklist,
)


@dataclass(frozen=True)
class FakeLabel:
    """The handful of ``WindowLabel`` fields the worklist copies through."""

    clip_audio_filename: str = "clip.wav"
    surah_ayah: str = "2:1"
    reciter_id: int = 7
    window_index: int = 0
    start_sample: int = 1600
    num_samples: int = 80000


def test_vowel_sites_and_score_vowels_cannot_disagree():
    reference = f"م{FATHA}ال{KASRA}ك{KASRA}"
    decode = f"م{FATHA}الك{DAMMA}"
    assert count_sites(vowel_sites(decode, reference)) == score_vowels(decode, reference)


def test_a_site_points_at_the_reference_position_it_classified():
    reference = f"م{FATHA}ال{KASRA}ك"
    sites = {s.reference_index: s for s in vowel_sites(reference, reference)}
    assert reference[sites[1].reference_index] == FATHA
    assert reference[sites[4].reference_index] == KASRA
    assert all(s.outcome == MATCHED for s in sites.values())


def test_an_omitted_vowel_is_located_on_its_carrier():
    reference = f"م{FATHA}ال{KASRA}ك"
    (site,) = [s for s in vowel_sites("مالك", reference) if s.outcome == OMITTED and s.reference_index == 1]
    assert site.carrier == "م"
    assert site.decoded_vowel is None
    assert site.decode_index is None


def test_an_edge_omission_outside_the_alignment_still_reports_its_carrier():
    # A decode matching only the head leaves the tail's vowels unaligned; they are still
    # omissions and must still name the letter they belonged to, or they cannot be audited.
    reference = f"م{FATHA}الك{KASRA}ب{DAMMA}"
    sites = [s for s in vowel_sites("م" + FATHA, reference) if s.outcome == OMITTED]
    assert {s.reference_vowel for s in sites} == {KASRA, DAMMA}
    assert all(s.carrier is not None for s in sites)


def test_a_spurious_vowel_carries_a_decode_index_and_no_reference_one():
    reference = "مالك"
    (site,) = [s for s in vowel_sites("م" + FATHA + "الك", reference) if s.outcome == SPURIOUS]
    assert site.decoded_vowel == FATHA
    assert site.reference_vowel is None
    assert site.decode_index == 1


def test_only_positions_where_exactly_one_model_matched_reach_the_worklist():
    reference = f"م{FATHA}ال{KASRA}ك{DAMMA}"
    # base drops the fatha; candidate drops the kasra; both keep the damma.
    base = f"مال{KASRA}ك{DAMMA}"
    candidate = f"م{FATHA}الك{DAMMA}"
    rows = discordant_sites(reference, base, candidate, FakeLabel())
    assert {(r.reference_vowel, r.direction) for r in rows} == {
        (FATHA, RECOVERED),
        (KASRA, REGRESSED),
    }


def test_positions_both_models_agree_on_are_excluded():
    reference = f"م{FATHA}ال{KASRA}ك"
    assert discordant_sites(reference, reference, reference, FakeLabel()) == []
    assert discordant_sites(reference, "مالك", "مالك", FakeLabel()) == []


def test_a_swap_counts_as_a_failure_not_a_match():
    # Wrong colour on the right carrier is the dangerous error; it must be auditable.
    reference = f"م{FATHA}الك"
    rows = discordant_sites(reference, f"م{KASRA}الك", reference, FakeLabel())
    assert [(r.direction, r.base_outcome) for r in rows] == [(RECOVERED, SWAPPED)]


def test_site_ids_are_stable_across_runs_and_distinct_across_positions():
    assert site_id("clip.wav", 0, 3) == site_id("clip.wav", 0, 3)
    assert site_id("clip.wav", 0, 3) != site_id("clip.wav", 1, 3)
    assert site_id("clip.wav", 0, 3) != site_id("clip.wav", 0, 4)


def test_population_counts_partition_every_reference_vowel():
    reference = f"م{FATHA}ال{KASRA}ك{DAMMA}"
    base = f"مال{KASRA}ك{DAMMA}"
    candidate = f"م{FATHA}الك{DAMMA}"
    rows = discordant_sites(reference, base, candidate, FakeLabel())
    counts = population_counts([reference], rows)
    assert counts["reference_vowels"] == 3
    assert counts[RECOVERED] == 1
    assert counts[REGRESSED] == 1
    assert counts["concordant"] == 1
    assert counts["strata"][RECOVERED]["fatha"] == 1
    assert counts["strata"][REGRESSED]["kasra"] == 1


def test_the_population_is_counted_per_colour_because_the_sample_is_drawn_per_colour():
    # sample_worklist caps each (direction, colour) bucket separately, so scaling an
    # audited share onto a direction *total* would weight the colours by sample size.
    reference = f"م{FATHA}ال{KASRA}ك{DAMMA}"
    rows = discordant_sites(reference, "مالك", reference, FakeLabel())
    strata = population_counts([reference], rows)["strata"]
    assert strata[RECOVERED] == {"fatha": 1, "damma": 1, "kasra": 1}
    assert strata[REGRESSED] == {"fatha": 0, "damma": 0, "kasra": 0}


def test_the_population_is_counted_off_the_same_rows_that_are_offered_for_audit():
    # Scaling an audited share onto a population computed by a *second* pass would let the
    # two drift; the sampled sites and the denominator must come from one partition.
    reference = f"م{FATHA}ال{KASRA}ك{DAMMA}"
    rows = discordant_sites(reference, "مالك", reference, FakeLabel())
    counts = population_counts([reference], rows)
    assert counts[RECOVERED] == len(rows) == 3
    assert counts["concordant"] == 0


def _row(index: int, direction: str, vowel: str) -> TashkeelSite:
    return TashkeelSite(
        site_id=site_id("clip.wav", 0, index),
        clip_audio_filename="clip.wav",
        surah_ayah="2:1",
        reciter_id=1,
        window_index=0,
        start_sample=0,
        num_samples=80000,
        reference="x",
        reference_index=index,
        reference_vowel=vowel,
        vowel_name={FATHA: "fatha", DAMMA: "damma", KASRA: "kasra"}[vowel],
        carrier="م",
        direction=direction,
        base_outcome=OMITTED,
        candidate_outcome=MATCHED,
        base_vowel=None,
        candidate_vowel=vowel,
    )


def test_sampling_caps_each_direction_and_colour_bucket_separately():
    rows = [_row(i, RECOVERED, FATHA) for i in range(50)]
    rows += [_row(100 + i, RECOVERED, KASRA) for i in range(3)]
    rows += [_row(200 + i, REGRESSED, FATHA) for i in range(50)]
    drawn = sample_worklist(rows, per_bucket=5)
    # Fatha would swamp the draw if the buckets were pooled; kasra keeps all three.
    assert len(drawn) == 5 + 3 + 5


def test_sampling_is_reproducible_but_not_ordered_by_bucket():
    rows = [_row(i, RECOVERED, FATHA) for i in range(20)]
    rows += [_row(100 + i, REGRESSED, KASRA) for i in range(20)]
    first = sample_worklist(rows, per_bucket=10)
    assert [r.site_id for r in first] == [r.site_id for r in sample_worklist(rows, per_bucket=10)]
    # A listener must not be able to read the direction off a run of consecutive rows.
    directions = [r.direction for r in first]
    assert len(set(directions[:10])) == 2


def test_a_worklist_round_trips_through_disk(tmp_path):
    rows = [_row(1, RECOVERED, FATHA), _row(2, REGRESSED, KASRA)]
    path = tmp_path / "worklist.jsonl"
    write_worklist(path, rows)
    assert read_worklist(path) == rows


def test_reading_rejects_a_row_missing_a_schema_field(tmp_path):
    path = tmp_path / "worklist.jsonl"
    path.write_text(json.dumps({"site_id": "abc"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worklist schema"):
        read_worklist(path)

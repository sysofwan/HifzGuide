"""Tests for the truth-site schema and its validating loader (#79).

Every check is driven through plain JSON rows on disk, the way a committed label file is
read, with no model, phonetizer or real audio.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from tadabur.truth_sites import (
    SCHEMA_FIELDS,
    STAGING_FIELDS,
    TruthSite,
    audio_sha256,
    load_truth_sites,
    write_truth_sites,
)


def _row(**overrides) -> dict:
    """A valid haraka site: fatha on the ب of ``كَتَبَ``."""
    row = {
        "site_id": "s1",
        "source": "new_audit",
        "assumes_competent_reciter": False,
        "audio_filename": "clip.wav",
        "shard": None,
        "start_sample": 0,
        "end_sample": None,
        "audio_sha256": None,
        "surah_ayah": "2:255",
        "reference": "كَتَبَ",
        "reference_index": 4,
        "mark": "fatha",
        "prescribed": "fatha",
        "heard": "fatha",
        "stratum": "stratum-a",
        "stratum_population": 10,
    }
    row.update(overrides)
    return row


def _write(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def _load(tmp_path, rows: list[dict], audio_dir=None) -> list[TruthSite]:
    path = tmp_path / "sites.jsonl"
    _write(path, rows)
    return load_truth_sites(path, audio_dir)


def _staged(sha: str, **overrides) -> dict:
    return _row(**{"shard": 3, "end_sample": 16000, "audio_sha256": sha, **overrides})


def test_schema_fields_match_the_documented_order():
    assert SCHEMA_FIELDS == (
        "site_id", "source", "assumes_competent_reciter", "audio_filename", "shard",
        "start_sample", "end_sample", "audio_sha256", "surah_ayah", "reference",
        "reference_index", "mark", "prescribed", "heard", "stratum", "stratum_population",
    )
    assert set(STAGING_FIELDS) <= set(SCHEMA_FIELDS)


def test_valid_rows_load_in_file_order(tmp_path):
    rows = [
        _row(),
        _row(site_id="s2", reference_index=2, mark="fatha", heard="unclear"),
        _row(site_id="s3", reference="كَتَب", reference_index=4, mark="sukun",
             prescribed="sukun", heard="fatha", audio_filename="other.wav"),
        _row(site_id="s4", reference="رَببِ", reference_index=2, mark="shaddah",
             prescribed="held", heard="not_held", audio_filename="third.wav"),
        _row(site_id="s5", reference="ذَ", reference_index=0, mark="ذ↔ز",
             prescribed="ذ", heard="ز", audio_filename="fourth.wav"),
    ]
    sites = _load(tmp_path, rows)
    assert [s.site_id for s in sites] == ["s1", "s2", "s3", "s4", "s5"]
    assert sites[0] == TruthSite(**rows[0])


def test_blank_and_comment_lines_are_ignored(tmp_path):
    path = tmp_path / "sites.jsonl"
    path.write_text("# header\n\n" + json.dumps(_row(), ensure_ascii=False) + "\n",
                    encoding="utf-8")
    assert [s.site_id for s in load_truth_sites(path)] == ["s1"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_truth_sites(tmp_path / "absent.jsonl")


# --- malformed rows ------------------------------------------------------------------


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "sites.jsonl"
    path.write_text('{"site_id": \n', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_truth_sites(path)


def test_a_non_object_row_is_rejected(tmp_path):
    path = tmp_path / "sites.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_truth_sites(path)


def test_a_missing_field_is_rejected(tmp_path):
    row = _row()
    del row["stratum"]
    with pytest.raises(ValueError, match=r"missing: \['stratum'\]"):
        _load(tmp_path, [row])


def test_an_unknown_field_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"unknown: \['note'\]"):
        _load(tmp_path, [_row(note="x")])


@pytest.mark.parametrize(
    "field, value",
    [
        ("reference_index", "4"),
        ("reference_index", 4.0),
        ("start_sample", True),
        ("assumes_competent_reciter", 1),
        ("stratum_population", None),
        ("mark", None),
    ],
)
def test_a_wrong_json_type_is_rejected(tmp_path, field, value):
    with pytest.raises(ValueError, match=f"{field} must be"):
        _load(tmp_path, [_row(**{field: value})])


# --- vocabulary ----------------------------------------------------------------------


def test_an_unknown_mark_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown mark 'vowel'"):
        _load(tmp_path, [_row(mark="vowel", prescribed="vowel", heard="vowel")])


def test_a_pair_outside_the_target_pairs_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown mark"):
        _load(tmp_path, [_row(reference="دَ", reference_index=0, mark="د↔ذ",
                              prescribed="د", heard="د")])


@pytest.mark.parametrize(
    "reference, prescribed, heard",
    [("ذَ", "ذ", "ذ"), ("ذَ", "ذ", "ظ"), ("ظَ", "ظ", "ذ"), ("ظَ", "ظ", "pending")],
)
def test_dhal_zah_is_a_target_pair_in_both_directions(tmp_path, reference, prescribed, heard):
    """ذ↔ظ is in scope by owner decision (acceptance rules §7) though Muraja does not
    forgive it, so it is a valid mark either way round, and its sites may wait."""
    (site,) = _load(tmp_path, [_row(reference=reference, reference_index=0, mark="ذ↔ظ",
                                    prescribed=prescribed, heard=heard)])
    assert (site.mark, site.prescribed, site.heard) == ("ذ↔ظ", prescribed, heard)


def test_a_dhal_zah_site_must_sit_on_its_prescribed_letter(tmp_path):
    with pytest.raises(ValueError, match="does not carry"):
        _load(tmp_path, [_row(reference="ظَ", reference_index=0, mark="ذ↔ظ",
                              prescribed="ذ", heard="ذ")])
    with pytest.raises(ValueError, match="cannot be heard as"):
        _load(tmp_path, [_row(reference="ذَ", reference_index=0, mark="ذ↔ظ",
                              prescribed="ذ", heard="ز")])


def test_an_unknown_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown source"):
        _load(tmp_path, [_row(source="model_output")])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"prescribed": "kasra"}, "cannot prescribe 'kasra'"),
        ({"heard": "none"}, "cannot be heard as 'none'"),
        ({"heard": "held"}, "cannot be heard as 'held'"),
        ({"reference": "رَببِ", "reference_index": 2, "mark": "shaddah",
          "prescribed": "unclear", "heard": "held"}, "cannot prescribe"),
        ({"reference": "ذَ", "reference_index": 0, "mark": "ذ↔ز",
          "prescribed": "ذ", "heard": "ظ"}, "cannot be heard as"),
    ],
)
def test_a_state_the_mark_cannot_take_is_rejected(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, [_row(**overrides)])


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"reference": "كَتَب", "reference_index": 4, "mark": "sukun", "prescribed": "sukun"},
        {"reference": "رَببِ", "reference_index": 2, "mark": "shaddah", "prescribed": "held"},
        {"reference": "ذَ", "reference_index": 0, "mark": "ذ↔ز", "prescribed": "ذ"},
    ],
)
def test_every_mark_can_wait_for_a_verdict(tmp_path, overrides):
    (site,) = _load(tmp_path, [_row(**overrides, heard="pending")])
    assert site.heard == "pending"


def test_pending_is_never_what_the_mushaf_prescribes(tmp_path):
    with pytest.raises(ValueError, match="cannot prescribe 'pending'"):
        _load(tmp_path, [_row(reference="ذَ", reference_index=0, mark="ذ↔ز",
                              prescribed="pending", heard="pending")])


@pytest.mark.parametrize("surah_ayah", ["2-255", "0:1", "115:1", "2:", ""])
def test_a_malformed_surah_ayah_is_rejected(tmp_path, surah_ayah):
    with pytest.raises(ValueError, match="surah_ayah"):
        _load(tmp_path, [_row(surah_ayah=surah_ayah)])


# --- provenance ----------------------------------------------------------------------


def test_an_empty_audio_filename_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="audio_filename is empty"):
        _load(tmp_path, [_row(audio_filename="")])


@pytest.mark.parametrize("field", ["audio_filename", "start_sample"])
def test_a_null_required_provenance_field_is_rejected(tmp_path, field):
    with pytest.raises(ValueError, match=f"{field} must be"):
        _load(tmp_path, [_row(**{field: None})])


@pytest.mark.parametrize("unset", STAGING_FIELDS)
def test_staging_fields_are_all_set_or_all_null(tmp_path, unset):
    with pytest.raises(ValueError, match="filled together"):
        _load(tmp_path, [_staged("a" * 64, **{unset: None})])


def test_fully_staged_provenance_loads(tmp_path):
    (site,) = _load(tmp_path, [_staged("a" * 64)])
    assert (site.shard, site.end_sample, site.audio_sha256) == (3, 16000, "a" * 64)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"shard": 385}, "shard 385"),
        ({"end_sample": 0}, "not after start_sample"),
        ({"audio_sha256": "A" * 64}, "64 lowercase hex"),
        ({"audio_sha256": "a" * 63}, "64 lowercase hex"),
    ],
)
def test_malformed_staged_provenance_is_rejected(tmp_path, overrides, message):
    row = _staged("a" * 64)
    row.update(overrides)
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, [row])


def test_a_negative_start_sample_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="negative"):
        _load(tmp_path, [_row(start_sample=-1)])


# --- shaddah: both gemination states ------------------------------------------------


@pytest.mark.parametrize(
    "reference, index, prescribed, heard",
    [
        ("رَببِ", 2, "held", "held"),  # held as written
        ("رَببِ", 2, "held", "not_held"),  # dropped shaddah
        ("رَبِ", 2, "not_held", "held"),  # added shaddah
        ("رَبِ", 2, "not_held", "not_held"),  # the un-held control, unchanged
        ("رَبِ", 2, "not_held", "unclear"),
        ("رَببِ", 2, "held", "unclear"),
    ],
)
def test_shaddah_sites_load_in_every_gemination_state(tmp_path, reference, index,
                                                       prescribed, heard):
    (site,) = _load(tmp_path, [_row(reference=reference, reference_index=index,
                                    mark="shaddah", prescribed=prescribed, heard=heard)])
    assert (site.prescribed, site.heard) == (prescribed, heard)


@pytest.mark.parametrize(
    "reference, index, prescribed",
    [
        ("رَبِ", 2, "held"),  # a held site needs a doubled carrier
        ("رَببِ", 2, "not_held"),  # an un-held site needs a single carrier...
        ("رَببِ", 3, "not_held"),  # ...not the second of a doubled pair either
    ],
)
def test_shaddah_carrier_must_match_the_prescribed_state(tmp_path, reference, index,
                                                          prescribed):
    with pytest.raises(ValueError, match="does not carry 'shaddah'"):
        _load(tmp_path, [_row(reference=reference, reference_index=index, mark="shaddah",
                              prescribed=prescribed, heard="held")])


# --- the reference index must carry the mark -----------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"reference_index": 6},  # past the end
        {"reference_index": 5},  # the haraka itself, not its carrier
        {"reference_index": 4, "reference": "كَتَبِ"},  # carrier bears kasra, not fatha
        {"reference": "كَتَبَ", "mark": "sukun", "prescribed": "sukun", "heard": "sukun"},
        {"reference": "زَ", "reference_index": 0, "mark": "ذ↔ز",
         "prescribed": "ذ", "heard": "ذ"},  # carrier is the other letter
        {"reference": "قَاا", "reference_index": 2, "mark": "sukun",
         "prescribed": "sukun", "heard": "sukun"},  # madd is not a carrier
    ],
)
def test_a_reference_index_that_does_not_carry_the_mark_is_rejected(tmp_path, overrides):
    with pytest.raises(ValueError, match="reference"):
        _load(tmp_path, [_row(**overrides)])


# --- whole-file invariants -----------------------------------------------------------


def test_duplicate_site_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate site_id"):
        _load(tmp_path, [_row(), _row(reference_index=2)])


def test_sites_on_one_item_must_share_its_reference(tmp_path):
    other = _row(site_id="s2", reference="كَتَبَ لَ", reference_index=7)
    with pytest.raises(ValueError, match="disagree"):
        _load(tmp_path, [_row(), other])


def test_sites_on_one_clip_must_share_its_checksum(tmp_path):
    # Two items (different start samples) in one clip: one file, so one checksum.
    rows = [_staged("a" * 64), _staged("b" * 64, site_id="s2", start_sample=100)]
    with pytest.raises(ValueError, match="disagree on its shard or checksum"):
        _load(tmp_path, rows)


def test_items_in_one_clip_may_differ_in_span_and_reference(tmp_path):
    rows = [_staged("a" * 64), _staged("a" * 64, site_id="s2", start_sample=100,
                                       end_sample=8000, reference="كَتَبَ لَ")]
    assert len(_load(tmp_path, rows)) == 2


def test_a_stratum_has_one_population(tmp_path):
    rows = [_row(), _row(site_id="s2", reference_index=2, stratum_population=11)]
    with pytest.raises(ValueError, match="populations"):
        _load(tmp_path, rows)


def test_a_population_smaller_than_its_sites_is_rejected(tmp_path):
    rows = [_row(stratum_population=1), _row(site_id="s2", reference_index=2,
                                             stratum_population=1)]
    with pytest.raises(ValueError, match="claims a population of 1"):
        _load(tmp_path, rows)


def test_a_non_positive_population_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        _load(tmp_path, [_row(stratum_population=0)])


# --- checksums against staged audio --------------------------------------------------


def test_audio_matching_its_checksum_verifies(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "clip.wav").write_bytes(b"RIFF staged clip")
    sha = audio_sha256(audio_dir / "clip.wav")
    assert len(_load(tmp_path, [_staged(sha)], audio_dir)) == 1


def test_a_checksum_mismatch_is_rejected_when_audio_is_supplied(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "clip.wav").write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="does not match the recorded"):
        _load(tmp_path, [_staged("0" * 64)], audio_dir)


def test_the_same_mismatch_passes_when_no_audio_is_supplied(tmp_path):
    assert len(_load(tmp_path, [_staged("0" * 64)])) == 1


def test_an_unstaged_site_cannot_be_verified(tmp_path):
    with pytest.raises(ValueError, match="no audio_sha256"):
        _load(tmp_path, [_row()], tmp_path)


def test_missing_staged_audio_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load(tmp_path, [_staged("0" * 64)], tmp_path / "empty")


# --- writing -------------------------------------------------------------------------


def test_write_then_load_round_trips(tmp_path):
    sites = [TruthSite(**_row()), TruthSite(**_row(site_id="s2", reference_index=2))]
    path = tmp_path / "out" / "sites.jsonl"
    write_truth_sites(sites, path)
    assert load_truth_sites(path) == sites
    assert "كَتَبَ" in path.read_text(encoding="utf-8")  # Arabic left readable


def test_write_rejects_an_invalid_site_before_touching_disk(tmp_path):
    path = tmp_path / "sites.jsonl"
    path.write_text("previous contents\n", encoding="utf-8")
    bad = dataclasses.replace(TruthSite(**_row()), mark="vowel")
    with pytest.raises(ValueError, match="unknown mark"):
        write_truth_sites([bad], path)
    assert path.read_text(encoding="utf-8") == "previous contents\n"

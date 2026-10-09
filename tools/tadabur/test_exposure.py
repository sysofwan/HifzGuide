"""Tests for the exposure registry: its schema, its consistency checks, the disjointness
check, and the committed registry against the sources it was built from (#89)."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

from tadabur.exposure import (
    CLIPS_V2_SHARDS,
    EVALSET_DEV,
    EVALSET_LEGACY,
    EVALSET_MANIFEST_PATH,
    EVALSET_TEST,
    H448_INIT_VALIDATION,
    H448_STREAM_SHARDS,
    H448_TRAINING,
    LABEL_FILES,
    LABEL_P35_FIXTURE,
    LABEL_TASHKEEL_COUNTERFACTUAL,
    MINING_POOL,
    MURAJA_CLIPS_PATH,
    MURAJA_REREAD_CORPUS,
    PROBE_TRAINING,
    SYNTHETIC_EDIT_SOURCE,
    TRUTH_SITE_USES,
    USES,
    Exposure,
    ExposureIncomplete,
    ExposureOverlap,
    ShardExposure,
    check_disjoint,
    COPY_SCREEN,
    EXPOSURE_DIR,
    PROBABLE_COPIES_NAME,
    RECORDING_ALIASES_NAME,
    copies_across_uses,
    copy_groups,
    duplicate_recordings,
    read_probable_copies,
    read_recording_aliases,
    relations,
    screen_agreement,
    write_probable_copies,
    write_recording_aliases,
    evalset_records,
    label_file_clips,
    load_registry,
    parse_exposure,
    require_complete,
    shard_exposure,
    staged_exposures,
    write_shard_use,
    write_use,
)
from tadabur.staged_audio import IndexRow, load_staged_clips

SHA = "b" * 64
WAQF = TRUTH_SITE_USES["waqf_boundary"]


def _row(name="a.wav", shard=3, row=7, reciter=12, span=(None, None), sha=SHA) -> Exposure:
    return Exposure(name, shard, row, reciter, span[0], span[1], sha)


def _shards(shards, rows, membership="exact", shared_baseline=True) -> ShardExposure:
    return ShardExposure(tuple(shards), rows, SHA, membership, shared_baseline, "test")


def _json(row: Exposure) -> dict:
    return dataclasses.asdict(row)


# --- the schema ------------------------------------------------------------------------


def test_parse_exposure_accepts_a_whole_clip_and_a_span():
    assert parse_exposure(_json(_row()), "t") == _row()
    assert parse_exposure(_json(_row(span=(0, 16000), sha=None)), "t").end_sample == 16000


@pytest.mark.parametrize("change", [
    {"reciter_id": True},                         # a bool is not an int
    {"start_sample": 0},                          # half a span
    {"start_sample": 5, "end_sample": 5},         # an empty span
    {"audio_sha256": "ABC"},
    {"audio_filename": "dir/a.wav"},
    {"shard": 385},
    {"extra": 1},
])
def test_parse_exposure_rejects_a_malformed_row(change):
    with pytest.raises(ValueError):
        parse_exposure({**_json(_row()), **change}, "t")


def test_write_and_load_round_trip_sorted_and_deduplicated(tmp_path):
    write_use(WAQF, [_row("b.wav", row=8), _row("a.wav"), _row("a.wav")], tmp_path)
    write_shard_use(H448_TRAINING, _shards((21, 22), {12: 3, 40: 1}), tmp_path)
    write_use(MINING_POOL, [], tmp_path)  # declared unused
    registry = load_registry(tmp_path)
    assert [r.audio_filename for r in registry.recordings[WAQF]] == ["a.wav", "b.wav"]
    assert registry.reciters(H448_TRAINING) == {12, 40}
    assert registry.reciters(MINING_POOL) == frozenset()
    assert registry.uses() == sorted([WAQF, H448_TRAINING, MINING_POOL])
    assert registry.uses_of(12) == sorted([WAQF, H448_TRAINING])
    assert registry.shared_baseline_uses() == [H448_TRAINING]


def test_a_use_with_no_file_is_missing_evidence_not_an_empty_use(tmp_path):
    write_use(WAQF, [_row()], tmp_path)
    registry = load_registry(tmp_path)
    assert MINING_POOL in registry.missing() and WAQF not in registry.missing()
    with pytest.raises(ExposureIncomplete, match="mining_pool has no file"):
        registry.reciters(MINING_POOL)
    with pytest.raises(ExposureIncomplete):
        check_disjoint(WAQF, MINING_POOL, registry=registry)
    with pytest.raises(ExposureIncomplete, match="no file for"):
        require_complete(registry)


def test_load_refuses_unknown_files_and_unsorted_rows(tmp_path):
    (tmp_path / "not_a_use.jsonl").write_text("")
    with pytest.raises(ValueError, match="unknown use"):
        load_registry(tmp_path)
    (tmp_path / "not_a_use.jsonl").unlink()
    rows = [_row("b.wav", row=8), _row("a.wav")]
    (tmp_path / f"{WAQF}.jsonl").write_text("".join(json.dumps(_json(r)) + "\n" for r in rows))
    with pytest.raises(ValueError, match="sorted"):
        load_registry(tmp_path)


@pytest.mark.parametrize("other", [
    _row("a.wav", reciter=99),             # one filename, two reciters
    _row("z.wav"),                         # two filenames, one shard row
    _row("a.wav", sha="c" * 64),           # one filename, two checksums
])
def test_a_recording_is_the_same_recording_in_every_use(tmp_path, other):
    write_use(WAQF, [_row("a.wav")], tmp_path)
    with pytest.raises(ValueError):
        write_use(MINING_POOL, [other], tmp_path)
    assert not (tmp_path / f"{MINING_POOL}.jsonl").exists()


def test_an_unknown_use_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown use"):
        write_use("panel_v2", [_row()], tmp_path)


# --- the check -------------------------------------------------------------------------


def _registry(tmp_path):
    write_use(WAQF, [_row("a.wav", reciter=1, sha="1" * 64),
                     _row("b.wav", row=8, reciter=2, sha="2" * 64)], tmp_path)
    write_use(MINING_POOL, [_row("c.wav", shard=39, row=0, reciter=2, sha="3" * 64)], tmp_path)
    write_use(SYNTHETIC_EDIT_SOURCE, [_row("d.wav", shard=21, row=0, reciter=3, sha="4" * 64)],
              tmp_path)
    write_shard_use(H448_TRAINING, _shards((21,), {3: 1, 7: 2}), tmp_path)
    return load_registry(tmp_path)


def test_check_disjoint_by_reciter_names_the_shared_reciters(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ExposureOverlap, match=r"mining_pool and truth_site.waqf_boundary share 1 reciter"):
        check_disjoint(MINING_POOL, WAQF, registry=registry)
    check_disjoint(MINING_POOL, SYNTHETIC_EDIT_SOURCE, registry=registry)
    check_disjoint(MINING_POOL, WAQF, by="source", registry=registry)  # different recordings


def test_check_disjoint_by_source_counts_a_shard_use_as_all_its_rows(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ExposureOverlap, match="share 1 sources"):
        check_disjoint(SYNTHETIC_EDIT_SOURCE, H448_TRAINING, by="source", registry=registry)
    check_disjoint(WAQF, H448_TRAINING, by="source", registry=registry)


def test_check_disjoint_by_source_sees_a_copy_under_another_row_by_its_checksum(tmp_path):
    """Tadabur files one recording under two speaker ids: two names and shard rows, one
    audio. Recordings without a checksum are compared by row alone."""
    write_use(WAQF, [_row("spk1.wav", sha="c" * 64), _row("x.wav", row=8, sha=None)], tmp_path)
    write_use(MINING_POOL, [_row("spk2.wav", shard=39, row=0, sha="c" * 64),
                            _row("y.wav", shard=39, row=1, sha=None)], tmp_path)
    registry = load_registry(tmp_path)
    with pytest.raises(ExposureOverlap, match=r"share 1 sources, e.g. \[\(39, 0\)\]"):
        check_disjoint(MINING_POOL, WAQF, by="source", registry=registry)
    assert duplicate_recordings(registry) == {
        "c" * 64: {(3, 7): (WAQF,), (39, 0): (MINING_POOL,)}}


def _index_row(
    name: str, shard: int, row: int, seconds: float = 8.0, reciter: int = 12
) -> IndexRow:
    return IndexRow(name, shard, row, reciter, "2:2", seconds)


def test_the_copy_screen_groups_one_reciter_and_one_duration_whatever_the_name():
    index = [
        _index_row("tadabur_spk0001_S1_A2_abcd0123_000004.wav", 3, 7),
        _index_row("tadabur_spk0002_S9_A5_ffff0000_000001.wav", 21, 5),  # another name
        _index_row("tadabur_spk0003_S1_A2_abcd0123_000004.wav", 22, 0, seconds=8.0004),
        _index_row("tadabur_spk0001_S1_A2_abcd0123_000005.wav", 22, 1, seconds=8.5),
        _index_row("tadabur_spk0001_S1_A2_abcd0123_000006.wav", 22, 2, reciter=13),
    ]
    assert copy_groups(index) == [((3, 7), (21, 5), (22, 0))]  # 8.0004 s is 8.000 s
    assert relations(copy_groups(index))[(21, 5)] == ((3, 7), (22, 0))


def test_screen_agreement_counts_recall_on_confirmed_groups_and_false_pairs():
    groups = [((0, 1), (0, 2), (0, 3)), ((1, 1), (1, 2))]
    checksums = {(0, 1): "a", (0, 2): "a", (0, 3): "b", (1, 1): "c", (2, 1): "c",
                 (1, 2): "d"}
    assert screen_agreement(groups, checksums) == {
        "checksummed_rows": 6, "confirmed_groups": 2, "confirmed_groups_found": 1,
        "grouped_pairs_same_checksum": 1, "grouped_pairs_different_checksum": 3}


_A, _B, _C = (f"tadabur_spk000{n}_S2_A2_abcd0123_00000{n}.wav" for n in (1, 2, 3))


def _copied_registry(tmp_path):
    """A truth site whose recording probably has a copy in a training shard (8 s), and a
    pool clip that probably is an edit source's recording under its own row (9 s)."""
    write_use(WAQF, [_row(_A, sha=None)], tmp_path)
    write_use(MINING_POOL, [_row(_B, shard=40, row=1, sha=None)], tmp_path)
    write_use(SYNTHETIC_EDIT_SOURCE, [_row(_C, shard=22, row=3, sha=None)], tmp_path)
    write_shard_use(H448_TRAINING, _shards((21,), {12: 1}), tmp_path)
    write_shard_use(H448_INIT_VALIDATION, _shards((0,), {12: 1}, "uncertain"), tmp_path)
    index = [_index_row(_A, 3, 7), _index_row("x.wav", 21, 5),
             _index_row(_B, 40, 1, seconds=9.0), _index_row(_C, 22, 3, seconds=9.0)]
    write_probable_copies(copy_groups(index), SHA, tmp_path)
    return load_registry(tmp_path)


def test_check_disjoint_by_source_counts_a_probable_copy_as_the_same_recording(tmp_path):
    registry = _copied_registry(tmp_path)
    # The site's recording is in no h448 training shard, but its probable copy is.
    with pytest.raises(ExposureOverlap,
                       match=r"truth_site.waqf_boundary and h448.training share 1"):
        check_disjoint(WAQF, H448_TRAINING, by="source", registry=registry)
    with pytest.raises(ExposureOverlap, match=r"share 1 sources, e.g. \[\(40, 1\)\]"):
        check_disjoint(MINING_POOL, SYNTHETIC_EDIT_SOURCE, by="source", registry=registry)
    check_disjoint(MINING_POOL, WAQF, by="source", registry=registry)
    assert copies_across_uses(registry) == {
        (MINING_POOL, SYNTHETIC_EDIT_SOURCE): 1, (SYNTHETIC_EDIT_SOURCE, MINING_POOL): 1,
        (WAQF, H448_TRAINING): 1}
    # Without the screen the copies pass: it is the screen that catches them.
    bare = dataclasses.replace(registry, copies={})
    check_disjoint(WAQF, H448_TRAINING, by="source", registry=bare)


def test_two_shard_uses_overlap_through_a_copy_spanning_their_shards(tmp_path):
    """Shards 0 and 21 are disjoint, but one recording sits in both."""
    write_shard_use(H448_TRAINING, _shards((21,), {12: 1}), tmp_path)
    write_shard_use(H448_INIT_VALIDATION, _shards((0,), {12: 1}, "uncertain"), tmp_path)
    check_disjoint(H448_TRAINING, H448_INIT_VALIDATION, by="source",
                   registry=load_registry(tmp_path))
    write_probable_copies([((0, 5), (21, 5))], SHA, tmp_path)
    with pytest.raises(ExposureOverlap, match=r"share 1 sources, e.g. \[\(21, 5\)\]"):
        check_disjoint(H448_TRAINING, H448_INIT_VALIDATION, by="source",
                       registry=load_registry(tmp_path))
    write_shard_use(H448_INIT_VALIDATION, _shards((0, 21), {12: 1}, "uncertain"), tmp_path)
    with pytest.raises(ExposureOverlap, match=r"e.g. \[21, \(21, 5\)\]"):  # a shared shard
        check_disjoint(H448_TRAINING, H448_INIT_VALIDATION, by="source",
                       registry=load_registry(tmp_path))


def test_aliases_merge_never_drop_and_make_rows_one_recording(tmp_path):
    first, second = _row(_A, sha="c" * 64), _row(_B, shard=40, row=1, sha="c" * 64)
    write_use(WAQF, [first], tmp_path)
    write_recording_aliases([first, second, _row(_C, shard=22, row=3, sha="d" * 64)], tmp_path)
    write_recording_aliases([], tmp_path)  # the use that brought the evidence may go
    registry = load_registry(tmp_path)
    assert registry.aliases == {(3, 7): ((40, 1),), (40, 1): ((3, 7),)}
    assert registry.same_recording((40, 1)) == {(3, 7), (40, 1)}
    write_use(MINING_POOL, [_row(_B, shard=40, row=1, sha=None)], tmp_path)
    with pytest.raises(ExposureOverlap, match="share 1 sources"):
        check_disjoint(MINING_POOL, WAQF, by="source", registry=load_registry(tmp_path))
    # An alias that names a recording another way than a use does is refused.
    with pytest.raises(ValueError):
        write_recording_aliases([_row(_A, sha="e" * 64)], tmp_path)


def test_the_uses_a_copy_touches_follow_every_write(tmp_path):
    """Only the relation is stored: a use written later is seen at once, no rewrite."""
    registry = _copied_registry(tmp_path)
    assert (PROBE_TRAINING, WAQF) not in copies_across_uses(registry)
    write_use(PROBE_TRAINING, [_row("tadabur_spk0004_S2_A2_abcd0123_000009.wav", shard=21,
                                    row=5, sha=None)], tmp_path)
    reloaded = load_registry(tmp_path)
    assert copies_across_uses(reloaded)[(PROBE_TRAINING, WAQF)] == 1
    with pytest.raises(ExposureOverlap):
        check_disjoint(PROBE_TRAINING, WAQF, by="source", registry=reloaded)


def test_the_committed_records_are_this_screen_current_and_complete():
    registry = load_registry()
    groups = read_probable_copies(EXPOSURE_DIR / PROBABLE_COPIES_NAME)
    assert groups and registry.copies == relations(groups)
    aliases = read_recording_aliases(EXPOSURE_DIR / RECORDING_ALIASES_NAME)
    # Every checksum two rows carry, in the staging registry or any use, is an alias.
    checksums = {a.source: a.audio_sha256 for a in aliases}
    checksums.update({(c.shard, c.row_index): c.audio_sha256 for c in load_staged_clips().values()})
    checksums.update({r.source: r.audio_sha256 for rows in registry.recordings.values()
                      for r in rows if r.audio_sha256 is not None})
    held = Counter(checksums.values())
    assert {s for s, sha in checksums.items() if held[sha] > 1} == set(registry.aliases)
    # The screen finds every confirmed group (its recall on what is known).
    agreement = screen_agreement(groups, checksums)
    assert agreement["confirmed_groups_found"] == agreement["confirmed_groups"] >= 32


def test_a_record_of_another_screen_is_refused(tmp_path):
    write_probable_copies([((0, 5), (21, 5))], SHA, tmp_path)
    path = tmp_path / PROBABLE_COPIES_NAME
    path.write_text(path.read_text().replace(COPY_SCREEN, "copy-screen-v1"), encoding="utf-8")
    with pytest.raises(ValueError, match="not the groups of the screen"):
        load_registry(tmp_path)


def test_check_disjoint_compares_only_the_first_use_with_the_others(tmp_path):
    registry = _registry(tmp_path)
    # truth sites and the pool share reciter 2, but neither is compared with the other.
    check_disjoint(SYNTHETIC_EDIT_SOURCE, WAQF, MINING_POOL, registry=registry)


def test_a_shard_use_needs_a_known_membership(tmp_path):
    with pytest.raises(ValueError, match="membership"):
        write_shard_use(H448_TRAINING, _shards((21,), {3: 1}, membership="some"), tmp_path)


def test_check_disjoint_refuses_unknown_names_and_modes(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="unknown use"):
        check_disjoint(WAQF, "the_panel", registry=registry)
    with pytest.raises(ValueError, match="by must be"):
        check_disjoint(WAQF, MINING_POOL, by="clip", registry=registry)


# --- the builders ----------------------------------------------------------------------


def test_shard_exposure_counts_rows_per_reciter_and_refuses_unindexed_shards(tmp_path):
    index = tmp_path / "index.jsonl"
    rows = [
        {"audio_filename": f"r{i}.wav", "shard": shard, "row_index": i, "reciter_id": reciter,
         "surah_id": 0, "ayah_id": 1, "ayah_duration_s": 3.0}
        for i, (shard, reciter) in enumerate([(21, 5), (21, 5), (22, 6), (39, 7)])
    ]
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kind = dict(membership="exact", shared_baseline=False, note="t")
    found = shard_exposure(index, [21, 22], **kind)
    assert (found.shards, dict(found.rows_per_reciter)) == ((21, 22), {5: 2, 6: 1})
    with pytest.raises(ValueError, match="does not index"):
        shard_exposure(index, [21, 23], **kind)


def test_evalset_records_reads_each_split_and_the_legacy_sample(tmp_path):
    manifest = tmp_path / "manifest.json"
    clips = [
        {"filename": "tadabur_sh039_i00012_S1_A2_.wav", "shard": 39, "split": "dev"},
        {"filename": "tadabur_sh058_i00003_S1_A2_.wav", "shard": 58, "split": "test"},
        {"filename": "tadabur_sh058_i00004_S1_A2_.wav", "shard": 58, "split": "dev",
         "in_population": False},
    ]
    manifest.write_text(json.dumps({"clips": clips}))
    assert evalset_records(manifest) == [
        ((39, 12), EVALSET_DEV), ((58, 3), EVALSET_TEST), ((58, 4), EVALSET_LEGACY),
    ]
    clips[0]["shard"] = 40
    manifest.write_text(json.dumps({"clips": clips}))
    with pytest.raises(ValueError, match="cannot read"):
        evalset_records(manifest)


def test_label_files_name_whole_clips():
    fixtures = label_file_clips(LABEL_P35_FIXTURE)
    assert len(fixtures) == 206 and not any("__seg" in name for name in fixtures)
    assert all(name.endswith(".wav") for name in label_file_clips(LABEL_TASHKEEL_COUNTERFACTUAL))


# --- the committed registry ------------------------------------------------------------


def test_h448_stream_shards_are_the_training_spec_and_avoid_every_unseen_shard():
    from tadabur.mining_pool import HELD_OUT_BLOCK
    from tadabur.shard_reader import parse_shard_spec
    from training.decode_evalset import gate_eval_shards, training_shard_spec

    assert H448_STREAM_SHARDS == training_shard_spec()
    trained = set(parse_shard_spec(H448_STREAM_SHARDS))
    assert not trained & (set(HELD_OUT_BLOCK) | set(gate_eval_shards()))
    assert load_registry().shard_uses[H448_TRAINING].shards == tuple(sorted(trained))


def test_the_committed_registry_is_complete():
    registry = load_registry()
    require_complete(registry)  # every declared use has a file, empty if unused
    assert set(registry.uses()) == USES
    assert registry.recordings[PROBE_TRAINING] == ()  # declared, unused until #94/#95


def test_h448s_own_exposures_are_the_shared_baseline():
    registry = load_registry()
    assert registry.shared_baseline_uses() == [H448_INIT_VALIDATION, H448_TRAINING]
    init = registry.shard_uses[H448_INIT_VALIDATION]
    assert init.shards == tuple(CLIPS_V2_SHARDS) and init.membership == "uncertain"
    assert registry.shard_uses[H448_TRAINING].membership == "exact"


def test_the_evalset_uses_match_the_committed_frozen_manifest():
    import hashlib

    assert hashlib.sha256(EVALSET_MANIFEST_PATH.read_bytes()).hexdigest().startswith("3d5d237a")
    registry = load_registry()
    recorded = {
        (row.source, use)
        for use in (EVALSET_DEV, EVALSET_TEST, EVALSET_LEGACY)
        for row in registry.recordings[use]
    }
    assert recorded == set(evalset_records(EVALSET_MANIFEST_PATH))
    manifest = json.loads(EVALSET_MANIFEST_PATH.read_text(encoding="utf-8"))
    by_source = {r.source: r for use in (EVALSET_DEV, EVALSET_TEST, EVALSET_LEGACY)
                 for r in registry.recordings[use]}
    for (source, _), clip in zip(evalset_records(EVALSET_MANIFEST_PATH), manifest["clips"]):
        assert by_source[source].reciter_id == clip["reciter_id"]


def test_the_muraja_use_matches_the_committed_clip_list():
    names = json.loads(MURAJA_CLIPS_PATH.read_text(encoding="utf-8"))
    recorded = {r.audio_filename for r in load_registry().recordings[MURAJA_REREAD_CORPUS]}
    assert recorded == set(names) and len(names) == 447


def test_the_committed_staged_uses_match_the_staged_registry_and_truth_sites():
    registry = load_registry()
    for use, rows in staged_exposures(load_staged_clips()).items():
        assert registry.recordings[use] == tuple(sorted(set(rows))), use


def test_every_label_file_clip_is_in_its_committed_use():
    registry = load_registry()
    for use in LABEL_FILES:
        recorded = {row.audio_filename for row in registry.recordings[use]}
        assert recorded == label_file_clips(use), use


def test_the_synthetic_edit_uses_match_88s_committed_edits():
    from tadabur.exposure import SYNTHETIC_EDIT_DONOR, synthetic_edit_exposures

    registry = load_registry()
    for use, rows in synthetic_edit_exposures().items():
        assert registry.recordings[use] == tuple(sorted(set(rows))), use
    assert registry.recordings[SYNTHETIC_EDIT_SOURCE] and registry.recordings[SYNTHETIC_EDIT_DONOR]


def test_edit_sources_and_donors_are_reciter_disjoint_from_every_evaluation_use():
    from tadabur.exposure import SEALED_PANEL, SYNTHETIC_EDIT_DONOR

    registry = load_registry()
    require_complete(registry)
    evaluation = [
        use for use in registry.uses()
        if use.startswith(("truth_site.", "human_label.", "decode_evalset."))
        or use in (MINING_POOL, SEALED_PANEL)
    ]
    assert SEALED_PANEL in evaluation and set(TRUTH_SITE_USES.values()) <= set(evaluation)
    for edits in (SYNTHETIC_EDIT_SOURCE, SYNTHETIC_EDIT_DONOR):
        check_disjoint(edits, *evaluation, by="reciter", registry=registry)
        check_disjoint(edits, *evaluation, by="source", registry=registry)

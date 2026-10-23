"""#176's baseline decodes: the QUD other half and Ship A's held-out verified sites.

These are the round's **baseline** decodes, read before any candidate exists: C95's check-protocol
decodes of every segment in #175's complement selection, and of every Ship A held-out verified
site (G6's eight real mistakes and the report-only marks) into the truth baseline's stores. The
entry point is :mod:`training.consonant_baseline`; the QUD half is decoded through
:mod:`training.aug_readout_decode`'s path, which must know ``qud_other`` as it knows ``qud_half``.

Locked (#176): later turns may add tests here but must not weaken these.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tadabur.qud_overlap import PILOT
from tadabur.ship_a import HELD_OUT
from tadabur.ship_a_verified import SHIP_A, load_designs
from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips
from training import aug_readout as ar
from training import aug_readout_decode as ad
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import consonant_readout as cr
from training import truth_baseline as tb
from training.external_corpus import read_decode_file

C95 = "c95"
QOTHER = ag.QUD_OTHER_SET


@pytest.fixture(scope="module")
def cb():
    """The module #176 adds; importing it is the first deliverable."""
    from training import consonant_baseline

    return consonant_baseline


def _harness() -> bool:
    from training.muraja_policy import Harness

    return Harness.locate() is not None


def _skip_without_harness() -> None:
    if not _harness():
        pytest.skip("the Muraja harness is not built here")


def _pin() -> dict:
    """The identity record a decode run of C95 on the round's sets must write."""
    return {"model": C95, "identity": ag.identity_record(asp.CONSONANT1.models[C95]),
            "fingerprints": ag.expected_fingerprints(asp.C95_REF)}


def _cache(path: Path, name: str, ref: str, identity: dict, items: dict,
           protocols: dict) -> None:
    """A truth-baseline cache file with the given identity and items."""
    path.write_text(json.dumps({
        "model": name, "identity": identity,
        "fingerprints": {p: tb.expected_fingerprint(ref, mode)
                         for p, mode in protocols.items()},
        "items": items}), encoding="utf-8")


def _probe_line(key: str) -> dict:
    """One decode line of a QUD segment, carrying the set's provenance."""
    return {"key": key, "audio_filename": f"{key}.mp3", "mp3_sha256": "a" * 64,
            "segment_index": 0, "start_sample": 0, "end_sample": 16000, "checks": []}


def _shards(tmp_path: Path, keys: list[str], identity: dict | None = None) -> Path:
    """A decodes directory as one decode pass leaves it: the identity record and one file per
    shard."""
    directory = tmp_path / "decodes"
    directory.mkdir()
    (directory / f"{QOTHER}.{C95}.identity.json").write_text(
        json.dumps(identity or _pin()), encoding="utf-8")
    for index, key in enumerate(keys):
        (directory / f"{QOTHER}.{index}.{C95}.jsonl").write_text(
            json.dumps(_probe_line(key), ensure_ascii=False) + "\n", encoding="utf-8")
    return directory


# --- the QUD other half's decode path -----------------------------------------------------------


def test_qud_other_reads_the_same_shards_as_the_half_qud_half_read():
    """One shard is one mushaf parquet: the other half is decoded by the same ten runs."""
    assert ad.shards(QOTHER) == ad.shards(ag.QUD_SET)
    assert len(ad.shards(QOTHER)) == 10


def test_qud_other_recordings_hold_only_the_complement(monkeypatch):
    """The complement selection is the set: a kept-half key never reaches C95's decode."""
    from training import qud_decode

    kept = sorted(ar.load_qud_selection())[0]
    held = sorted(cr.load_qud_complement())[:2]
    keys = (kept, *held)

    def fake(slug, shard, cache_dir):
        yield ad.Recording(keys, lambda: (None, [{"key": k, "start_sample": 0,
                                                  "end_sample": 1} for k in keys]))

    monkeypatch.setattr(qud_decode, "recordings", fake)
    (recording,) = list(ad.recordings(QOTHER, ad.shards(QOTHER)[0], Path("/unused"), None))
    assert recording.keys == tuple(held)
    _, records = recording.load()
    assert [record["key"] for record in records] == list(held)


def test_the_qother_identity_must_be_the_round_s_pin(cb):
    """C95 is a pinned model: the decode's identity is the pin, not merely the same bytes."""
    pin = _pin()
    assert cb.qother_identity(pin) == pin["identity"]
    for tampered in ({"identity": {**pin["identity"], "weights": tb.LIVE}},
                     {"identity": {**pin["identity"], "checkpoint_sha256": "0" * 64}},
                     {"identity": pin["identity"] | {"model_ref": "/elsewhere.pt"}},
                     {"fingerprints": {}}):
        with pytest.raises(ValueError):
            cb.qother_identity({**pin, **tampered})


def test_the_qother_merge_writes_the_whole_complement_deterministically(cb, tmp_path):
    """Exactly every complement key, once, as one byte-stable gzipped file."""
    keys = sorted(cr.load_qud_complement())[:2]
    directory = _shards(tmp_path, keys)
    first, second = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    models = tmp_path / "models.json"
    cb.merge_qother(directory, first, models_path=models, expected=2)
    cb.merge_qother(directory, second, models_path=models, expected=2)
    assert read_decode_file(first) == read_decode_file(second)
    assert first.read_bytes() == second.read_bytes()
    assert set(read_decode_file(first)) == set(keys)
    assert read_decode_file(first)[keys[0]]["mp3_sha256"] == "a" * 64


def test_the_qother_merge_refuses_a_short_or_stray_decode(cb, tmp_path):
    """A missing segment or one the other half owns is not this set's decode."""
    keys = sorted(cr.load_qud_complement())[:2]
    with pytest.raises(ValueError, match="items decoded"):
        cb.merge_qother(_shards(tmp_path, keys), tmp_path / "out.jsonl.gz",
                        models_path=tmp_path / "models.json", expected=3)
    strayed = tmp_path / "stray"
    strayed.mkdir()
    with pytest.raises(ValueError, match="complement"):
        cb.merge_qother(_shards(strayed, [*keys, sorted(ar.load_qud_selection())[0]]),
                        tmp_path / "out2.jsonl.gz",
                        models_path=tmp_path / "models.json", expected=3)


def test_the_qother_merge_appends_its_models_entry_and_never_rewrites_one(cb, tmp_path):
    """``models.json`` is the round's identity record: #176 adds this set, changes no entry."""
    keys = sorted(cr.load_qud_complement())[:2]
    directory = _shards(tmp_path, keys)
    models = tmp_path / "models.json"
    existing = {"qdat/c95": _pin()}
    models.write_text(json.dumps(existing), encoding="utf-8")
    cb.merge_qother(directory, tmp_path / "out.jsonl.gz", models_path=models, expected=2)
    after = json.loads(models.read_text(encoding="utf-8"))
    assert after["qdat/c95"] == existing["qdat/c95"]
    assert ag.validate_identity(QOTHER, C95, after[f"{QOTHER}/{C95}"],
                                spec=asp.CONSONANT1) == _pin()["identity"]
    drifted = _pin() | {"identity": {**_pin()["identity"], "checkpoint_sha256": "0" * 64}}
    (directory / f"{QOTHER}.{C95}.identity.json").write_text(json.dumps(drifted),
                                                             encoding="utf-8")
    with pytest.raises(ValueError):
        cb.merge_qother(directory, tmp_path / "out.jsonl.gz", models_path=models, expected=2)
    assert json.loads(models.read_text(encoding="utf-8")) == after


# --- the committed QUD decode (#176's artifact) -------------------------------------------------


def test_the_committed_qother_decode_is_the_complement_under_the_round_s_pin(cb):
    """The committed file holds every complement segment, on the audio the base decode read,
    and the round records C95's pinned identity for the set (`validate_identity`)."""
    from training.qud_pilot import read_decodes

    base = {}
    for slug in PILOT:
        base.update(read_decodes(slug, "base"))
    lines = ar.read_readout(QOTHER, C95, base, cr.load_qud_complement(),
                            spec=asp.CONSONANT1)
    assert set(lines) == cr.load_qud_complement()
    models = json.loads(cr.MODELS_PATH.read_text(encoding="utf-8"))
    assert ag.validate_identity(QOTHER, C95, models[f"{QOTHER}/{C95}"],
                                spec=asp.CONSONANT1) == _pin()["identity"]


def test_the_qud_other_coverage_is_the_frozen_population(cb):
    """The waqf/leen exclusions both models share: the reading's denominator."""
    population = cr.coverage()["populations"][QOTHER]
    assert (population["decoded"], population["evaluated"], population["excluded"]) \
        == (10261, 10085, 176)


# --- Ship A's held-out verified sites -----------------------------------------------------------


def test_the_held_out_selection_is_the_ship_a_held_out_frame(cb):
    """#126's split, ship A's frame only: G6's eight sites and the marks reported only."""
    sites = cb.held_out_sites()
    designs = load_designs()
    assert len(sites) == 375
    assert all((designs[site.site_id].frame, designs[site.site_id].half) == (SHIP_A, HELD_OUT)
               for site in sites)
    marks = {site.mark for site in sites}
    assert {"ط↔ق", "س↔ش", "س↔ص", "ض↔ظ"} <= marks
    assert set(cr.load_g6_sites()) <= {site.site_id for site in sites}


@pytest.mark.parametrize("name", ["base", "h448"])
def test_every_held_out_item_is_already_decoded_by_both_baselines(cb, name):
    """G6's context columns exist: C95 is the only model these clips lack."""
    registry = load_staged_clips(REGISTRY_PATH)
    items = cb.held_out_items(registry)
    assert len(items) == 363
    for store in (tb.DECODE_STORE, tb.CHECK_STORE):
        assert set(items) <= set(tb.read_cache(store.path(name))["items"])


def test_the_held_out_caches_must_be_the_pin_and_carry_check_sequences(cb, tmp_path):
    """The check store is what the harness replays: an item without its sequence is no decode."""
    items = {"one": object()}
    check, decoded = tmp_path / "checks", tmp_path / "decodes"
    check.mkdir()
    decoded.mkdir()
    with pytest.raises(SystemExit):
        cb.assert_held_out_caches(items, check_dir=check, decode_dir=decoded)
    _cache(check / f"{C95}.json", C95, asp.C95_REF,
           {**_pin()["identity"], "weights": tb.LIVE},
           {"one": {"checks": []}}, tb.CHECK_PROTOCOLS)
    _cache(decoded / f"{C95}.json", C95, asp.C95_REF, _pin()["identity"],
           {"one": {"spans": "", "stream_b0": ""}}, tb.PROTOCOLS)
    with pytest.raises(ValueError, match="pin"):
        cb.assert_held_out_caches(items, check_dir=check, decode_dir=decoded)
    _cache(check / f"{C95}.json", C95, asp.C95_REF, _pin()["identity"],
           {"one": {"spans": "", "stream_b0": ""}}, tb.CHECK_PROTOCOLS)
    with pytest.raises(ValueError, match="check"):
        cb.assert_held_out_caches(items, check_dir=check, decode_dir=decoded)
    _cache(check / f"{C95}.json", C95, asp.C95_REF, _pin()["identity"],
           {"one": {"checks": []}}, tb.CHECK_PROTOCOLS)
    assert cb.assert_held_out_caches(items, check_dir=check, decode_dir=decoded) \
        == _pin()["identity"]


def test_the_committed_held_out_caches_cover_every_site(cb):
    """The artifact: C95's two caches for every held-out item, under the round's pin."""
    items = cb.held_out_items(load_staged_clips(REGISTRY_PATH))
    assert cb.assert_held_out_caches(items) == _pin()["identity"]
    recorded = tb.read_cache(tb.CHECK_STORE.path(C95))
    assert set(items) <= set(recorded["items"])
    assert all("checks" in row for row in recorded["items"].values())


# --- the G6 read (baseline numbers) -------------------------------------------------------------


def test_g6_reads_the_frozen_sites_and_reports_c95_s_flag_count(cb, monkeypatch):
    """The read comes from #175's frozen list, never a live recomputation (#176 scoping)."""
    seen = {}

    def fake_catch(sites, models, **kwargs):
        seen["ids"] = [site.site_id for site in sites]
        seen["models"] = models
        return {"sites": seen["ids"],
                "models": {C95: {"graded": len(sites), "flagged": 3, "outcomes": {}}},
                "difference": 0}

    monkeypatch.setattr(cr, "catch_arm", fake_catch)
    table = cb.g6_baseline()
    assert seen["ids"] == cr.load_g6_sites()
    assert seen["models"] == (C95,)
    assert table["models"][C95]["flagged"] == 3
    assert table["models"][C95]["graded"] == len(cr.load_g6_sites()) == 8


def test_the_g6_read_cannot_certify_without_c95_s_caches(cb, tmp_path):
    """A missing cache is an unread input, never a zero."""
    assert cb.g6_baseline(cache_dir=tmp_path) is None


# --- the reading --------------------------------------------------------------------------------


def test_the_baseline_read_reports_the_qud_half_and_g6(cb):
    """The PR's numbers, from the committed decodes through the #175 harness."""
    _skip_without_harness()
    numbers = cb.read()
    qud = numbers["qud_other"]
    assert (qud["decoded"], qud["evaluated"], qud["excluded"]) == (10261, 10085, 176)
    assert qud["graded_words"] > 0
    assert 0.0 < qud["false_flags_per_1000"] < 100.0
    assert numbers["g6"]["sites"] == 8
    assert 0 <= numbers["g6"]["flagged"] <= 8

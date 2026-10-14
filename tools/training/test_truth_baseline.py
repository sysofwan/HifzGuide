"""The truth-baseline runner's torch-free parts: which files it scores, the arms it compares,
and the rendered report."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

from tadabur.edit_check_summary import CALLS
from tadabur.exclusions import REASONS, Exclusions
from tadabur.ship_a import HALVES
from tadabur.ship_a_verified import FRAMES, SHIP_A, Design
from tadabur.truth_sites import TRUTH_SITES_DIR
from training.decoding import APP_CHECKS, SPANS, DecodeFingerprint
from training.muraja_policy import FLAGGED, RECORDED_CHECKS, SHIPPED_DEFAULT, SINGLE_DECODE
from training.test_direction_table import ZAKARA, _site
from training.test_site_outcomes import DHAKARA, DHAL
from training.test_truth_scorer import RECITERS, app, decodes_for, sites  # noqa: F401  (a fixture)
from training.truth_baseline import (
    ARM_PROTOCOLS,
    CHECK_STORE,
    DECODE_STORE,
    DEFAULT_MODELS,
    DOC_PATH,
    PROTOCOLS,
    REPORT_PATH,
    cache_text,
    check_cache,
    comparisons,
    expected_fingerprint,
    file_sha256,
    model_identity,
    read_cache,
    scored,
    truth_site_files,
    verified_views,
)
from training.truth_baseline_doc import SPARSE_MARK, render
from training.truth_scorer import item_key, score


def test_every_committed_truth_site_file_and_nothing_else():
    names = [path.name for path in truth_site_files(TRUTH_SITES_DIR)]
    assert "waqf_boundaries.jsonl" in names and "p35_fixtures.jsonl" in names
    assert "p35_fixtures.relocation.jsonl" not in names


def test_new_truth_site_files_are_picked_up(tmp_path: Path):
    for name in ("new_audit.jsonl", "p35_fixtures.relocation.jsonl", "notes.txt"):
        (tmp_path / name).write_text("")
    assert [p.name for p in truth_site_files(tmp_path)] == ["new_audit.jsonl"]


def test_comparisons_pair_models_per_protocol_and_protocols_per_model():
    pairs = comparisons(["base", "h448"])
    assert ("h448/spans", "base/spans") in pairs and ("h448/stream_b0", "base/stream_b0") in pairs
    assert ("base/stream_b0", "base/spans") in pairs and ("h448/stream_b0", "h448/spans") in pairs
    assert PROTOCOLS["stream_b0"] == "confirmed-stream-v2-flush"


def test_the_recorded_check_sequence_is_paired_against_the_single_decode_stream():
    pairs = comparisons(["base", "h448"])
    assert ("h448/checks", "base/checks") in pairs
    assert ("base/checks", "base/stream_b0") in pairs and ("h448/checks", "h448/stream_b0") in pairs
    assert CHECK_STORE.protocols == {"checks": APP_CHECKS} and DECODE_STORE.protocols == PROTOCOLS


def test_render_marks_small_cells_and_names_every_arm(sites):  # noqa: F811
    decodes = {"base/spans": decodes_for(sites), "h448/spans": decodes_for(sites)}
    report = score(sites, RECITERS, decodes, [("h448/spans", "base/spans")])
    fingerprint = {
        "model": "m", "mode": "whole-spans", "weights_dtype": "bf16", "batch_size": 1,
        "device_type": "cuda", "autocast": True, "policy": "fp32-features-v1",
    }
    report["truth_site_files"] = ["x.jsonl"]
    report["fingerprints"] = {"base": {"spans": fingerprint}, "h448": {"spans": fingerprint}}
    text = render(report)
    assert SPARSE_MARK in text and "h448/spans - base/spans" in text
    assert "No real-mistake site" not in text  # the synthetic set has one
    assert "Sukun at a pause" in text and "Diagnostics only" in text


def test_render_states_each_arms_checks_and_pairs_recorded_against_single(sites):  # noqa: F811
    arms = ["h448/checks", "h448/stream_b0"]
    grades = app(sites, arms, {(SHIPPED_DEFAULT.name, "a.wav#2fathafatha"): FLAGGED})
    muraja = replace(grades, approximations={"h448/checks": RECORDED_CHECKS, "h448/stream_b0": SINGLE_DECODE})
    decodes = {a: decodes_for(sites) for a in arms}
    report = score(sites, RECITERS, decodes, [("h448/checks", "h448/stream_b0")], muraja)
    fingerprint = {
        "model": "m", "mode": "whole-spans", "weights_dtype": "bf16", "batch_size": 1,
        "device_type": "cuda", "autocast": True, "policy": "fp32-features-v1",
    }
    report["truth_site_files"] = ["x.jsonl"]
    report["fingerprints"] = {"h448": {"spans": fingerprint}}
    text = render(report)
    assert "**Recorded check sequence** (`h448/checks`)" in text
    assert "**Single-decode approximation** (`h448/stream_b0`)" in text
    assert "The recorded check sequence against the single-decode approximation" in text
    assert "| h448/checks - h448/stream_b0 |" in text


def test_the_report_path_needs_no_torch():
    code = (
        "import sys; import training.truth_baseline, training.truth_scorer; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)


# --- the decode cache is bound to one model, one protocol and one set of weights ----------


def _cache(model_ref: str, identity: dict) -> dict:
    return {
        "model": "m",
        "identity": identity,
        "fingerprints": {p: expected_fingerprint(model_ref, mode) for p, mode in PROTOCOLS.items()},
        "items": {"clip.wav@0:10": {"audio_sha256": "0" * 64, "spans": "", "stream_b0": ""}},
    }


def test_a_cache_refuses_another_model_reference_on_every_run(tmp_path: Path):
    cache = _cache("hub/model", {"model_ref": "hub/model", "hub_revision": "abc"})
    check_cache(tmp_path / "m.json", cache, "hub/model", PROTOCOLS)
    with pytest.raises(ValueError, match="holds decodes of"):
        check_cache(tmp_path / "m.json", cache, "hub/other", PROTOCOLS)


@pytest.mark.parametrize(
    "field, value",
    [
        ("model", "hub/other"),
        ("mode", "confirmed-stream-v1"),
        ("weights_dtype", "fp32"),
        ("batch_size", 8),
        ("device_type", "cpu"),
        ("autocast", False),
        ("policy", "legacy"),
    ],
)
@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
def test_a_cache_refuses_any_other_fingerprint_field(tmp_path: Path, protocol, field, value):
    cache = _cache("hub/model", {"model_ref": "hub/model", "hub_revision": "abc"})
    check_cache(tmp_path / "m.json", cache, "hub/model", PROTOCOLS)
    cache["fingerprints"][protocol][field] = value
    with pytest.raises(ValueError, match="was decoded as"):
        check_cache(tmp_path / "m.json", cache, "hub/model", PROTOCOLS)


def test_the_frozen_fingerprint_names_every_field():
    assert set(expected_fingerprint("m", SPANS)) == {f.name for f in fields(DecodeFingerprint)}


def test_a_cache_without_an_identity_is_refused(tmp_path: Path):
    cache = _cache("hub/model", None)
    with pytest.raises(ValueError, match="no model identity"):
        check_cache(tmp_path / "m.json", cache, "hub/model", PROTOCOLS)


def test_a_checkpoint_whose_bytes_changed_is_refused(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    identity = model_identity(str(checkpoint))
    assert identity == {"model_ref": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)}
    cache = _cache(str(checkpoint), identity)
    check_cache(tmp_path / "m.json", cache, str(checkpoint), PROTOCOLS)
    checkpoint.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="no longer has the bytes"):
        check_cache(tmp_path / "m.json", cache, str(checkpoint), PROTOCOLS)


def test_a_hub_identity_needs_the_loaded_revision():
    with pytest.raises(ValueError, match="loaded config"):
        model_identity("hub/model")


@pytest.mark.parametrize("store", [DECODE_STORE, CHECK_STORE], ids=["decodes", "checks"])
def test_the_committed_caches_are_bound_to_their_weights(store):
    for name, ref in DEFAULT_MODELS.items():
        cache = read_cache(store.path(name))
        assert cache["identity"]["model_ref"] == ref
        check_cache(store.path(name), cache, ref, store.protocols)
        assert cache["identity"] == read_cache(DECODE_STORE.path(name))["identity"]


@pytest.mark.parametrize("one_item_per_line", [False, True])
def test_a_cache_reads_back_as_written(one_item_per_line):
    checks = [{"hop": "بَ", "overlap": "تَ", "flush": False}, {"hop": "", "overlap": "", "flush": True}]
    cache = _cache("hub/model", {"model_ref": "hub/model", "hub_revision": "abc"})
    cache["items"]["other.wav@5:9"] = {"audio_sha256": "1" * 64, "checks": checks}
    text = cache_text(cache, one_item_per_line)
    assert json.loads(text) == cache and "بَ" in text
    if one_item_per_line:
        assert sum('"audio_sha256"' in line for line in text.splitlines()) == 2


def test_an_empty_cache_reads_back_as_written():
    cache = {"model": "m", "identity": None, "fingerprints": None, "items": {}}
    assert json.loads(cache_text(cache, True)) == cache


def test_the_report_and_the_power_inputs_both_leave_out_waqf_realized_sites(sites):  # noqa: F811
    """#134: the runner's one scoring step applies Ship A's waqf guard to both outputs and
    keeps ``today_system_arm``."""
    arms = {f"{model}/{protocol}": decodes_for(sites)
            for model in ("base", "h448") for protocol in ARM_PROTOCOLS}
    realized = Exclusions.of(waqf_realized={sites[0].site_id})
    report, inputs = scored(sites, RECITERS, arms, ["base", "h448"], app(sites, list(arms)),
                            realized)
    assert report["waqf_realized"] == inputs["waqf_realized"] == [sites[0].site_id]
    assert report["onset_trimmed"] == inputs["onset_trimmed"] == []
    assert inputs["today_system_arm"] == "h448/stream_b0"
    held = [c for c in inputs["cells"] for r in c["per_reciter"] if r == "1"]
    assert held == []  # reciter 1's only site was the realized one


def test_the_committed_doc_is_the_render_of_the_committed_report():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert DOC_PATH.read_text(encoding="utf-8") == render(report)


def test_the_committed_report_carries_the_verified_views():
    """#90: the word sample's causes; per frame and half (never pooled) the app outcome, the
    direction table, the per-stratum and per-teacher-outcome rates; the edit calls; and every
    exclusion reason, all read from the committed inputs."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert set(REASONS) <= set(report) and set(REASONS) <= set(report["word_sample"])
    assert set(report["verified"]) == set(FRAMES)
    assert all(set(halves) == set(HALVES) for halves in report["verified"].values())
    held_out = report["verified"]["ship_a"]["held_out"]
    assert {"app", "directions", "strata", "teacher"} <= set(held_out)
    assert {e["teacher"] for e in held_out["teacher"] if e["direction"] == "ط→ق"} == {
        "base_right", "base_wrong"}
    assert report["edit_check"]["type_verdicts"] == CALLS
    assert report["word_sample"]["words"] == 160
    assert "ship_a_verified.jsonl" in report["truth_site_files"]


def test_a_train_half_site_cannot_move_a_held_out_number():
    """Sol's round 2 on #145: each frame and half is scored alone, so changing what the models
    decoded at a train-half site leaves every held-out view (app outcome, directions, strata,
    teacher outcomes) exactly as it was."""
    sites = [replace(_site(f"{n}.wav", DHAKARA, "ذ↔ز", DHAL, DHAL),
                     stratum="ship_a:ذ→ز:h448_partner:base_wrong") for n in "abc"]
    designs = {s.site_id: Design(s.site_id, SHIP_A, half, 1.0)
               for s, half in zip(sites, ("train", "held_out", "held_out"))}
    reciters = {s.audio_filename: n for n, s in enumerate(sites)}

    def views(train_decode: str) -> dict:
        decodes = {f"{m}/{p}": {item_key(s): train_decode if s is sites[0] else s.reference
                                for s in sites}
                   for m in ("base", "h448") for p in ARM_PROTOCOLS}
        return verified_views(sites, reciters, decodes, app(sites, list(decodes)), designs,
                              Exclusions.of())

    before, after = views(DHAKARA), views(ZAKARA)
    assert before["ship_a"]["held_out"] == after["ship_a"]["held_out"]
    assert before["ship_a"]["train"] != after["ship_a"]["train"]

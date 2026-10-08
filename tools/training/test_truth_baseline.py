"""The truth-baseline runner's torch-free parts: which files it scores, the arms it compares,
and the rendered report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tadabur.truth_sites import TRUTH_SITES_DIR
from training.test_truth_scorer import RECITERS, decodes_for, sites  # noqa: F401  (a fixture)
from training.truth_baseline import (
    BATCH_SIZE,
    DECODES_DIR,
    DEFAULT_MODELS,
    PROTOCOLS,
    WEIGHTS_DTYPE,
    check_cache,
    comparisons,
    file_sha256,
    model_identity,
    read_decodes,
    truth_site_files,
)
from training.truth_baseline_doc import SPARSE_MARK, render
from training.truth_scorer import score


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


def test_the_report_path_needs_no_torch():
    code = (
        "import sys; import training.truth_baseline, training.truth_scorer; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)


# --- the decode cache is bound to one model, one protocol and one set of weights ----------


def _cache(model_ref: str, identity: dict) -> dict:
    def fingerprint(mode):
        return {"model": model_ref, "mode": mode, "weights_dtype": WEIGHTS_DTYPE,
                "batch_size": BATCH_SIZE, "device_type": "cuda", "autocast": True,
                "policy": "fp32-features-v1"}

    return {
        "model": "m",
        "identity": identity,
        "fingerprints": {p: fingerprint(mode) for p, mode in PROTOCOLS.items()},
        "items": {"clip.wav@0:10": {"audio_sha256": "0" * 64, "spans": "", "stream_b0": ""}},
    }


def test_a_cache_refuses_another_model_reference_on_every_run(tmp_path: Path):
    cache = _cache("hub/model", {"model_ref": "hub/model", "hub_revision": "abc"})
    check_cache(tmp_path / "m.json", cache, "hub/model")
    with pytest.raises(ValueError, match="holds decodes of"):
        check_cache(tmp_path / "m.json", cache, "hub/other")


def test_a_cache_refuses_another_protocol(tmp_path: Path):
    cache = _cache("hub/model", {"model_ref": "hub/model", "hub_revision": "abc"})
    cache["fingerprints"]["stream_b0"]["batch_size"] = 8
    with pytest.raises(ValueError, match="was decoded as"):
        check_cache(tmp_path / "m.json", cache, "hub/model")


def test_a_cache_without_an_identity_is_refused(tmp_path: Path):
    cache = _cache("hub/model", None)
    with pytest.raises(ValueError, match="no model identity"):
        check_cache(tmp_path / "m.json", cache, "hub/model")


def test_a_checkpoint_whose_bytes_changed_is_refused(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    identity = model_identity(str(checkpoint))
    assert identity == {"model_ref": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)}
    cache = _cache(str(checkpoint), identity)
    check_cache(tmp_path / "m.json", cache, str(checkpoint))
    checkpoint.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="no longer has the bytes"):
        check_cache(tmp_path / "m.json", cache, str(checkpoint))


def test_a_hub_identity_needs_the_loaded_revision():
    with pytest.raises(ValueError, match="loaded config"):
        model_identity("hub/model")


def test_the_committed_caches_are_bound_to_their_weights():
    for name, ref in DEFAULT_MODELS.items():
        cache = read_decodes(DECODES_DIR / f"{name}.json")
        assert cache["identity"]["model_ref"] == ref
        check_cache(DECODES_DIR / f"{name}.json", cache, ref)

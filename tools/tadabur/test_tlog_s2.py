"""Acceptance tests for the tlog S2 Tadabur validation (``tadabur.tlog_s2``, #186).

S2 draws a fixed-seed Tadabur sample from the training-complement shards only, embeds it with the
two pre-registered models (revisions pinned before the first embedding), validates the
pre-registered clustering rule on it, and commits the sample plus a generated report and page.
No tlog audio is read anywhere in the slice.

``sample.jsonl`` carries one row per clip: ``audio_filename``, ``shard``, ``row``,
``reciter_id``, ``surah_ayah``, the frame's ``ayah_duration_s`` and the exact embedding span
(``start_sample``/``end_sample``, the first 20 s at 16 kHz, cut to the clip's length). The
report's ``results`` hold, per model and per τ of the
pre-registered grid, the :func:`tadabur.speaker_cluster.metrics` entry with Wilson 95% intervals
on the two co-cluster rates; ``choice`` is exactly
:func:`tadabur.speaker_cluster.decide` over those entries, plus the chosen model's revision.

Test command, from the repo root::

    python -m pytest -q tools/tadabur/test_tlog_s2.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest

from tadabur import speaker_embed, tlog_s2
from tadabur.speaker_cluster import GRID, decide

#: The pre-registration's SHA-256, the same pin ``test_tlog_index`` holds.
PREREGISTRATION_SHA256 = "655c8c3e8c560a6e9a6b53b48b5e41c4bf45d3de270562b7140b6b757344096d"
#: The committed Tarteel v1 decodes whose spans give the look durations.
TARTEEL_DECODES_SHA256 = "ce45d840c73d34a1fc41d915c633ff8f6e30729f52b2aa7f2b9788c13ae18770"
#: The exact decile cut points of the 4,537 Tarteel v1 look clips, in seconds, top open.
TARTEEL_EDGES = (1.6254375, 4.181375, 5.2245, 6.229375, 7.267875, 8.5449375, 9.9149375,
                 11.4475, 13.4211875, 15.9985625)
TARTEEL_MAX_SECONDS = 19.992375
#: Every 17th training-complement shard, the first 20: the sample's frame.
SAMPLE_SHARDS = (21, 38, 56, 74, 92, 110, 128, 146, 164, 182,
                 200, 218, 236, 254, 272, 290, 308, 326, 344, 361)
SAMPLE_KEYS = frozenset({"audio_filename", "shard", "row", "reciter_id", "surah_ayah",
                         "ayah_duration_s", "start_sample", "end_sample"})


# --- the Tarteel look duration match ---------------------------------------------------------


def test_the_decile_cut_points_are_the_committed_tarteel_look_ones():
    edges = tlog_s2.bin_edges()
    assert len(edges) == 11
    assert edges[:10] == TARTEEL_EDGES
    assert edges[10] == float("inf")


def test_a_duration_maps_to_its_decile_and_nothing_outside_the_look_range():
    assert tlog_s2.bin_of(1.6254375) == 0
    assert tlog_s2.bin_of(4.181375) == 1
    assert tlog_s2.bin_of(15.9985625) == 9
    assert tlog_s2.bin_of(TARTEEL_MAX_SECONDS) == 9
    assert tlog_s2.bin_of(1.6) is None
    assert tlog_s2.bin_of(20.0) is None


def test_the_sample_constants_are_the_approved_ones():
    assert tlog_s2.SALT == "issue-186-tlog-s2-v1"
    assert tlog_s2.SAMPLE_SHARDS == SAMPLE_SHARDS
    assert (tlog_s2.PER_BIN, tlog_s2.COVERAGE_AYAT, tlog_s2.COVERAGE_RECITERS,
            tlog_s2.RECITER_CAP, tlog_s2.SPAN_SAMPLES) == (200, 60, 5, 20, 320_000)


# --- the draw itself -------------------------------------------------------------------------


def _synthetic_frame() -> list[dict]:
    """10 deciles x 5 ayat x 4 clips, 200 distinct reciters, one ayah-duration per bin."""
    seconds = (2.0, 4.5, 5.5, 6.5, 7.5, 9.0, 10.5, 12.0, 14.0, 17.0)
    frame, row = [], 0
    for decile, duration in enumerate(seconds):
        for ayah in range(1, 6):
            for offset in range(4):
                reciter = 1 + (decile * 20 + ayah * 4 + offset) % 200
                frame.append({
                    "audio_filename": f"tadabur_spk{reciter:04d}_S0_A{ayah}_"
                                      f"{decile:02d}{offset:02d}abcd_000001.wav",
                    "shard": SAMPLE_SHARDS[0], "row_index": row, "reciter_id": reciter,
                    "surah_id": 0, "ayah_id": ayah, "ayah_duration_s": duration,
                })
                row += 1
    return frame


#: Ten bins x 5 ayat x 4 reciters, one clip per reciter per bin: a large margin over the
#: 5-per-bin quota, so the draw's own ordering cannot run out of capacity.
def _draw(frame: list[dict]) -> list[dict]:
    return tlog_s2.draw_sample(frame, per_bin=5, coverage_ayat=2, coverage_reciters=2,
                               cap=10)


def test_the_draw_meets_every_constraint_on_the_frame():
    frame = _synthetic_frame()
    drawn = _draw(frame)
    assert len(drawn) == 50
    duration_of = {row["audio_filename"]: row["ayah_duration_s"] for row in frame}
    per_decile = np.bincount([tlog_s2.bin_of(duration_of[row["audio_filename"]])
                              for row in drawn], minlength=10)
    assert list(per_decile) == [5] * 10
    reciters = [row["reciter_id"] for row in drawn]
    assert max(np.bincount(reciters)) <= 10
    assert len({(row["reciter_id"], row["surah_ayah"]) for row in drawn}) == len(drawn)
    per_ayah: dict[str, set[int]] = {}
    for row in drawn:
        per_ayah.setdefault(row["surah_ayah"], set()).add(row["reciter_id"])
    assert sum(1 for reciters_of_ayah in per_ayah.values()
               if len(reciters_of_ayah) >= 2) >= 2
    assert drawn == sorted(drawn, key=lambda row: (row["shard"], row["row"]))
    assert len({row["audio_filename"] for row in drawn}) == len(drawn)


def test_the_draw_is_reproducible_from_its_salt_whatever_the_frame_order():
    frame = _synthetic_frame()
    first = _draw(frame)
    assert _draw(frame) == first
    shuffled = list(frame)
    random.Random(0).shuffle(shuffled)
    assert _draw(shuffled) == first


# --- the committed sample --------------------------------------------------------------------


def _committed_sample() -> list[dict]:
    return [json.loads(line) for line in
            tlog_s2.SAMPLE_PATH.read_text(encoding="utf-8").splitlines() if line]


def test_the_committed_sample_is_the_approved_draw():
    from training.distill_stream import held_out_shards

    rows = _committed_sample()
    assert len(rows) == 2000
    assert all(set(row) == SAMPLE_KEYS for row in rows)
    assert {row["shard"] for row in rows} == set(SAMPLE_SHARDS)
    assert not {row["shard"] for row in rows} & held_out_shards()
    assert all(row["start_sample"] == 0 for row in rows)
    # The span is the clip's true length cut at 20 s, not the full window: every clip is
    # shorter than 20 s (the frame is the Tarteel look range), so the field cannot be read as
    # "we embedded 20 s of a longer file" (selfcheck-r1 S1.5).
    assert all(row["end_sample"] == min(320_000, round(row["ayah_duration_s"] * 16_000))
               for row in rows)
    assert all(row["end_sample"] < 320_000 for row in rows)
    assert len({row["audio_filename"] for row in rows}) == len(rows)
    assert rows == sorted(rows, key=lambda row: (row["shard"], row["row"]))

    reciters = [row["reciter_id"] for row in rows]
    assert len(set(reciters)) >= 200
    assert max(np.bincount(reciters)) <= 20
    assert len({(row["reciter_id"], row["surah_ayah"]) for row in rows}) == len(rows)

    per_ayah: dict[str, set[int]] = {}
    for row in rows:
        per_ayah.setdefault(row["surah_ayah"], set()).add(row["reciter_id"])
    assert sum(1 for reciters_of_ayah in per_ayah.values()
               if len(reciters_of_ayah) >= 5) >= 50

    for row in rows:
        match = re.match(r"tadabur_spk\d+_S(\d+)_A(\d+)_", row["audio_filename"])
        assert match, row["audio_filename"]
        assert row["surah_ayah"] == f"{int(match.group(1)) + 1}:{int(match.group(2))}"


def test_the_embedding_run_refuses_a_sample_that_is_not_tadabur():
    from training.distill_stream import held_out_shards

    tlog_s2.assert_tadabur_sample(_committed_sample())
    tlog = {"audio_filename": "100_10_4115049010.flac", "shard": SAMPLE_SHARDS[0], "row": 0,
            "reciter_id": 1, "surah_ayah": "100:10", "start_sample": 0,
            "end_sample": 320_000}
    with pytest.raises(ValueError):
        tlog_s2.assert_tadabur_sample([tlog])
    held = {**tlog, "audio_filename": "tadabur_spk0001_S0_A1_abcd0000_000001.wav",
            "shard": min(held_out_shards())}
    with pytest.raises(ValueError):
        tlog_s2.assert_tadabur_sample([held])


# --- the pinned models and the embedding pass ------------------------------------------------


def test_the_candidate_models_are_pinned_in_the_preregistered_order():
    assert [model.key for model in speaker_embed.MODELS] == ["a", "b"]
    assert [model.repo_id for model in speaker_embed.MODELS] == [
        "speechbrain/spkrec-ecapa-voxceleb", "microsoft/wavlm-base-plus-sv"]
    assert [model.dim for model in speaker_embed.MODELS] == [192, 512]
    for model in speaker_embed.MODELS:
        assert re.fullmatch(r"[0-9a-f]{40}", model.revision)


def test_the_revisions_are_written_before_the_first_embedding(tmp_path):
    model = speaker_embed.MODELS[0]
    clip_ids = ["clip0", "clip1"]
    waveforms = [np.zeros(320_000, dtype=np.float32), np.ones(320_000, dtype=np.float32)]
    seen: list[list[str]] = []

    def encoder(batch):
        # The pin file must already exist, and name this model's revision, before any audio is
        # encoded: that is the "recorded before its first embedding" rule.
        pins = json.loads((tmp_path / "pins.json").read_text(encoding="utf-8"))
        assert pins["models"][model.key]["revision"] == model.revision
        assert pins["models"][model.key]["repo_id"] == model.repo_id
        assert pins["sample_sha256"] == "a" * 64
        seen.append(list(batch[0].shape))
        return np.array([[3.0] + [0.0] * (model.dim - 1),
                         [0.0, 2.0] + [0.0] * (model.dim - 2)])

    with pytest.raises(FileNotFoundError):
        tlog_s2.embed_sample(clip_ids, waveforms, model, tmp_path, encoder=encoder)

    tlog_s2.write_pins(tmp_path, sample_sha256="a" * 64, code_sha="b" * 40,
                       models=(model,))
    out = tlog_s2.embed_sample(clip_ids, waveforms, model, tmp_path, encoder=encoder)
    assert seen == [[320_000]]
    assert out.shape == (2, model.dim)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    stored = np.load(tmp_path / f"embeddings.{model.key}.npz")
    assert list(stored["clip_ids"]) == clip_ids
    assert np.allclose(stored["embeddings"], out)

    # A model whose revision is not pinned is refused, never embedded.
    with pytest.raises(ValueError):
        tlog_s2.embed_sample(clip_ids, waveforms, speaker_embed.MODELS[1], tmp_path,
                             encoder=lambda batch: np.zeros((2, 512)))


def test_the_pins_record_the_embedding_library_versions(tmp_path):
    # Model revisions alone do not reproduce the embeddings: B's pooling depends on the
    # transformers implementation, so the stack's versions are pinned beside them (#186 S1.6).
    tlog_s2.write_pins(tmp_path, sample_sha256="a" * 64, code_sha="b" * 40,
                       models=(speaker_embed.MODELS[0],))
    pins = json.loads((tmp_path / "pins.json").read_text(encoding="utf-8"))
    assert set(pins["libraries"]) == {"torch", "transformers", "speechbrain"}
    assert all(value is None or isinstance(value, str) for value in pins["libraries"].values())


# --- the committed report and page -----------------------------------------------------------


def _report() -> dict:
    return json.loads(tlog_s2.REPORT_PATH.read_text(encoding="utf-8"))


def test_the_report_records_the_frozen_inputs_and_the_pins():
    report = _report()
    rows = _committed_sample()
    assert report["preregistration"]["sha256"] == PREREGISTRATION_SHA256
    assert report["tarteel_look"]["sha256"] == TARTEEL_DECODES_SHA256
    assert report["tarteel_look"]["edges"] == list(TARTEEL_EDGES)
    assert report["tarteel_look"]["max_seconds"] == TARTEEL_MAX_SECONDS
    assert report["sample"]["sha256"] == hashlib.sha256(
        tlog_s2.SAMPLE_PATH.read_bytes()).hexdigest()
    assert report["sample"]["salt"] == tlog_s2.SALT
    assert report["sample"]["shards"] == list(SAMPLE_SHARDS)
    assert report["sample"]["clips"] == len(rows) == 2000
    assert report["sample"]["reciters"] >= 200
    assert report["sample"]["bins"] == [200] * 10
    assert report["grid"] == list(GRID)
    for model in speaker_embed.MODELS:
        record = report["models"][model.key]
        assert (record["repo_id"], record["revision"], record["dim"]) == (
            model.repo_id, model.revision, model.dim)


def test_every_model_has_every_tau_with_counted_rates_and_wilson_intervals():
    report = _report()
    assert list(report["results"]) == ["a", "b"]
    for model in ("a", "b"):
        assert set(report["results"][model]) == {f"{tau:.2f}" for tau in GRID}
        for entry in report["results"][model].values():
            assert entry["rows"] == 2000
            assert isinstance(entry["resplits"], int) and entry["resplits"] >= 0
            assert entry["largest_reciter"] >= 1
            for block in ("pre_cap", "post_cap"):
                found = entry[block]
                assert 1 <= found["largest_cluster"] <= entry["rows"]
                assert 0.0 < found["largest_cluster_share"] <= 1.0
                assert 0.0 <= found["recall"] <= 1.0
                for stratum in ("same_reciter", "same_ayah", "different_ayah"):
                    cell = found[stratum]
                    assert 0 <= cell["clustered"] <= cell["pairs"]
                    assert 0.0 <= cell["rate"] <= 1.0
                    low, high = cell["wilson"]
                    assert 0.0 <= low <= cell["rate"] <= high <= 1.0


def test_the_choice_is_the_preregistered_rule_and_nothing_else():
    report = _report()
    per_model = {model: {float(tau): entry
                         for tau, entry in report["results"][model].items()}
                 for model in report["results"]}
    recomputed = decide(per_model)
    choice = report["choice"]
    if "fallback" in choice:
        assert choice["fallback"] is True and recomputed == {"fallback": True}
    else:
        assert choice["model"] == recomputed["model"]
        assert choice["tau"] == recomputed["tau"]
        assert choice["revision"] == report["models"][choice["model"]]["revision"]


def test_the_report_names_each_model_s_rule_outcome():
    # A's failure is the recall bar, not the absence of an admissible tau, and the page must
    # say so: the fallback's cause is the number that decides it (selfcheck-r1 S1.4).
    report = _report()
    assert list(report["per_model"]) == ["a", "b"]
    for key, outcome in report["per_model"].items():
        if outcome["tau"] is None:
            assert outcome == {"tau": None, "recall": None, "pass": False}
        else:
            entry = report["results"][key][f"{outcome['tau']:.2f}"]
            assert outcome["recall"] == entry["pre_cap"]["recall"]
            assert outcome["pass"] == (outcome["recall"] >= 0.9)


def test_the_report_records_the_embedding_library_versions():
    libraries = _report()["embedding_run"]["libraries"]
    assert set(libraries) == {"torch", "transformers", "speechbrain"}
    assert libraries["transformers"] == "5.13.0"
    assert libraries["speechbrain"] == "1.1.2"
    assert libraries["torch"] == "2.11.0+cu128"


def test_the_page_states_each_model_s_outcome():
    report = _report()
    page = tlog_s2.DOC_PATH.read_text(encoding="utf-8")
    for key, outcome in report["per_model"].items():
        if outcome["tau"] is None:
            assert (f"No τ on the grid meets both chaining and text-independence for "
                    f"model {key.upper()}.") in page
        elif outcome["pass"]:
            assert (f"`*` marks τ = {outcome['tau']:.2f}, the value the rule selects for "
                    f"model {key.upper()}") in page
        else:
            assert (f"τ = {outcome['tau']:.2f} is model {key.upper()}'s lowest admissible "
                    f"value") in page


def test_the_page_is_generated_from_the_report():
    report = _report()
    page = tlog_s2.DOC_PATH.read_text(encoding="utf-8")
    assert page == tlog_s2.render(report)
    assert "Generated by `python -m tadabur.tlog_s2 report`" in page
    assert "do not edit by hand" in page
    assert "Reading of the pre-registration" in page
    if report["choice"].get("fallback") is True:
        assert "fallback" in page.lower()
    else:
        assert f"{report['choice']['tau']:.2f}" in page
        assert report["choice"]["revision"] in page

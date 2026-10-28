"""Acceptance tests for the tlog look decode of the four round-2 models (#212); no GPU, no model.

Committed before ``tadabur.tlog_eval_decode`` exists and **locked**: later turns may add tests but
never weaken or rewrite these. Nothing here grades, scores or reads a decode: the tests count
rows and compare strings the fake decoder made up.

Contract this file pins
-----------------------
``tadabur.tlog_eval_decode`` exposes:

* ``MODELS = ("base", "c95", "candidate_ema", "h448")``, ``PINNED_SHA256`` (model -> the full
  checkpoint SHA-256 of the three checkpoints) and ``PINNED_BASE_REVISION`` (the teacher's hub
  commit, ``01a1ef9f…``); ``model_specs() -> {model: training.external_decode.ModelSpec}`` (the
  teacher and h448 as the truth baseline loads them; C95 and the #178 candidate at their pinned
  checkpoint, EMA weights, step 10,000, recipe ``aug94-mixture-v2``).
  ``require_pinned_identities({model: identity})`` takes the identity records
  ``training.external_decode.load_decoders`` writes and raises ``ValueError`` naming the model
  when one is missing, extra, or not the pinned revision / checkpoint hash.
* ``read_manifest(role, *, tranches_dir=TRANCHES_DIR) -> list[dict]``: the manifest rows of
  ``role``, which must be ``"look"``. Any other role is a ``ValueError`` raised before any file
  is opened. A look row whose key ``(split, shard_file, row)`` appears in a
  ``confirm.<k>.jsonl`` of ``tranches_dir`` (k = 1..4) is a ``ValueError`` too.
* ``decodes_path(out_dir, shard_file, model)`` = ``out_dir / f"tlog.look.{shard_file}.{model}.jsonl"``,
  one file per shard and model, so shards decode side by side.
* ``STORES``: exactly ``checks`` and ``stream_b0`` (``training.truth_baseline.Store``): the app's
  check sequence and the b = 0 stream with its tail flushed. A decoder's whole-clip
  ``decode_spans`` is never called.
* ``recordings(rows, *, stream, word_counts, load_audio=None)`` -> ``Recording`` s of
  ``training.external_decode`` for the manifest ``rows`` of ONE shard. ``stream(name)`` yields
  the parquet rows of ``data/<shard_file>.parquet`` (called once per shard);
  ``word_counts(label)`` is the words in the ayah ``"s:a"``; ``load_audio(bytes)`` returns the
  16 kHz mono samples. A recording's ``load()`` verifies the audio SHA-256 first and returns the
  Tarteel v1 record shape (``key = <audio_path>#0``, ``audio_filename``, ``audio_sha256``,
  ``surah_ayah`` = the filename label, ``word_from`` 1, ``word_to`` the ayah's words,
  ``start_sample`` 0, ``end_sample``) plus ``split``, ``shard_file``, ``row``, ``stratum`` and
  ``assigned_label``; a row without a filename label keeps ``surah_ayah``, ``word_from`` and
  ``word_to`` ``None``.
* ``run_shard(shard_file, rows, out_dir, decoders, *, stream, word_counts, load_audio=None)
  -> timing``: reads the shard once and decodes each wanted row with every decoder in
  ``decoders`` (name -> decoder), appending to ``decodes_path`` (resumable); returns
  ``decode_recordings``'s timing.
* ``merge_decodes(parts_dir, out_dir, *, manifest_path, provenance, models=MODELS) -> dict``:
  gathers every ``decodes_path`` part into ``out_dir/tlog.look.<model>.jsonl.gz`` (sorted by key)
  and writes ``out_dir/meta.json`` (the ``provenance`` mapping plus ``manifest_sha256``,
  ``manifest_rows``, ``models`` and ``rows`` per model); returns ``{model: rows}``. A model that
  misses a manifest row, holds a row twice or holds one the manifest does not is a
  ``ValueError``.
* ``slice_verdict(slice_audio_s, slice_wall_s, total_audio_s, limit_hours=8.0) -> dict`` with
  ``projected_hours`` and ``within_budget`` (projected wall <= the limit, inclusive).

Test command, from ``tools/``::

    python -m pytest -q tadabur/test_tlog_eval_decode.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tadabur import tlog_eval_decode as ted
from tadabur.tlog_tranches import CONFIRM_ROLES, TRANCHES_DIR
from training.external_decode import ModelSpec
from training.truth_baseline import DEFAULT_MODELS

SHARD = "clean-00000-of-00411"
PINNED_SHA256 = {
    "c95": "8490b63d0664827d5a2e2e3121eb4af7b22069273db39268c7638cc7d8839cd6",
    "candidate_ema": "057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2",
    "h448": "eadc3f7504653cf4f88f4fb77482e1eb0f307e786ca5844d54ad72bc5483aa7a",
}
BASE_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"
SAMPLES_PER_BYTE = 1_000


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_row(row: int, *, split: str = "clean", shard_file: str = SHARD,
                  payload: bytes | None = None, label: str | None = "2:1",
                  stratum: str = "gating", **overrides) -> dict:
    payload = f"clip-{row}".encode() if payload is None else payload
    record = {"split": split, "shard_file": shard_file, "row": row,
              "audio_path": f"2_1_{row}.flac", "audio_sha256": _sha256(payload),
              "group_key": f"2_1_{row}.flac", "filename_label": label,
              "assigned_label": label, "label_source": "both" if label else "none",
              "straddles": False, "decode_mode": "whole_clip", "id_floor_ok": True,
              "stratum": stratum}
    record.update(overrides)
    return record


def _key(row) -> tuple[str, str, int]:
    return row["split"], row["shard_file"], row["row"]


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _tranches(tmp_path: Path, look, confirm=None) -> Path:
    """A tranches directory: ``look`` rows and all four confirm manifests.

    ``confirm`` is ``{role: rows}``; a role it leaves out holds one row of its own, disjoint
    from every look row used here.
    """
    directory = tmp_path / "tranches"
    _write_jsonl(directory / "look.jsonl", look)
    for index, role in enumerate(CONFIRM_ROLES, start=1):
        rows = (confirm or {}).get(role, [_manifest_row(900 + index, shard_file="clean-00900-of-00411")])
        _write_jsonl(directory / f"{role}.jsonl", rows)
    return directory


def _loaded(rows, payloads: dict[int, bytes], calls: list[str] | None = None) -> list[tuple]:
    """Each recording's ``(keys, samples, records)``, read in order as the decode loop reads."""
    return [(recording.keys, *recording.load()) for recording in ted.recordings(
        rows, stream=_stream(payloads, calls), word_counts=_word_counts, load_audio=_load_audio)]


def _stream(payloads: dict[int, bytes], calls: list[str] | None = None):
    """A shard stream of parquet-shaped rows (all rows of the shard, in row order)."""
    def stream(name: str):
        if calls is not None:
            calls.append(name)
        return iter({"audio": {"path": f"{index}.flac", "bytes": payload}}
                    for index, payload in sorted(payloads.items()))
    return stream


def _load_audio(data: bytes) -> np.ndarray:
    return np.zeros(len(data) * SAMPLES_PER_BYTE, dtype=np.float32)


class FakeDecoder:
    """Only the streaming decodes exist: a call to ``decode_spans`` is an AttributeError."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.checks = 0
        self.streams: list[tuple[int, bool]] = []

    def app_checks(self, span):
        self.checks += 1
        return [{"hop": f"{self.name}:{len(span)}", "overlap": "", "flush": True}]

    def decode_stream(self, span, block, flush_tail):
        self.streams.append((block, flush_tail))
        return f"{self.name}:stream:{len(span)}"


def _word_counts(label: str) -> int:
    return {"2:1": 3, "2:2": 5}[label]


# --- the models and their pins ------------------------------------------------------------------


def test_the_four_models_are_the_teacher_c95_the_candidate_and_h448():
    assert ted.MODELS == ("base", "c95", "candidate_ema", "h448")
    assert ted.PINNED_SHA256 == PINNED_SHA256
    assert ted.PINNED_BASE_REVISION == BASE_REVISION


def test_the_model_specs_pin_each_checkpoint_and_its_weights():
    specs = ted.model_specs()
    assert tuple(specs) == ted.MODELS
    assert all(isinstance(spec, ModelSpec) for spec in specs.values())
    assert specs["base"].ref == DEFAULT_MODELS["base"] and not specs["base"].use_ema
    assert specs["h448"].ref == DEFAULT_MODELS["h448"] and not specs["h448"].use_ema
    for name in ("c95", "candidate_ema"):
        assert specs[name].use_ema
        assert specs[name].step == 10_000
        assert specs[name].recipe == "aug94-mixture-v2"
    assert specs["c95"].ref != specs["candidate_ema"].ref


def _identities() -> dict:
    return {"base": {"model_ref": DEFAULT_MODELS["base"], "hub_revision": BASE_REVISION},
            **{name: {"model_ref": f"/ckpt/{name}.pt", "checkpoint_sha256": sha}
               for name, sha in PINNED_SHA256.items()}}


def test_the_pinned_identities_pass_and_each_wrong_one_is_refused_by_name():
    ted.require_pinned_identities(_identities())
    wrong_base = {**_identities(), "base": {"model_ref": "x", "hub_revision": "f" * 40}}
    with pytest.raises(ValueError, match="base"):
        ted.require_pinned_identities(wrong_base)
    for name in PINNED_SHA256:
        other = {**_identities(), name: {"model_ref": "x", "checkpoint_sha256": "0" * 64}}
        with pytest.raises(ValueError, match=name):
            ted.require_pinned_identities(other)


def test_a_missing_or_extra_model_identity_is_refused():
    missing = _identities()
    del missing["h448"]
    with pytest.raises(ValueError, match="h448"):
        ted.require_pinned_identities(missing)
    with pytest.raises(ValueError, match="candidate_live"):
        ted.require_pinned_identities({**_identities(), "candidate_live": {}})


# --- the look tranche only, and never a confirm row ---------------------------------------------


def test_the_look_manifest_is_read_whole(tmp_path):
    rows = [_manifest_row(3), _manifest_row(1, stratum="reported:unclean")]
    directory = _tranches(tmp_path, rows)
    assert ted.read_manifest("look", tranches_dir=directory) == rows


@pytest.mark.parametrize("role", ["confirm.1", "confirm.4", "train", "look.1", ""])
def test_any_role_but_look_is_refused_before_a_file_is_opened(tmp_path, role):
    empty = tmp_path / "empty"  # no file exists: a refusal must come first, not a missing file
    empty.mkdir()
    with pytest.raises(ValueError):
        ted.read_manifest(role, tranches_dir=empty)


def test_a_look_row_that_is_a_planted_confirm_row_is_refused(tmp_path):
    planted = _manifest_row(7)
    for role in CONFIRM_ROLES:
        directory = _tranches(tmp_path / role, [_manifest_row(1), planted],
                              {role: [_manifest_row(2, shard_file="clean-00002-of-00411"),
                                      dict(planted, group_key="other")]})
        with pytest.raises(ValueError, match="confirm"):
            ted.read_manifest("look", tranches_dir=directory)


def test_a_planted_confirm_row_never_reaches_a_shard_stream_or_a_decoder(tmp_path):
    """The refusal is at the manifest: nothing downstream runs with the planted tranche."""
    directory = _tranches(tmp_path, [_manifest_row(7)], {"confirm.1": [_manifest_row(7)]})
    calls: list[str] = []
    decoder = FakeDecoder()
    with pytest.raises(ValueError):
        rows = ted.read_manifest("look", tranches_dir=directory)
        ted.run_shard(SHARD, rows, tmp_path / "out", {"base": decoder},
                      stream=_stream({7: b"clip-7"}, calls), word_counts=_word_counts,
                      load_audio=_load_audio)
    assert calls == [] and decoder.checks == 0


def test_the_committed_look_manifest_holds_no_confirm_row():
    rows = ted.read_manifest("look")
    assert len(rows) == 9_710
    assert len({_key(row) for row in rows}) == len(rows)
    confirm = {_key(json.loads(line)) for role in CONFIRM_ROLES
               for line in (TRANCHES_DIR / f"{role}.jsonl").read_text(encoding="utf-8").splitlines()}
    assert not {_key(row) for row in rows} & confirm


# --- the record shape ---------------------------------------------------------------------------


def test_a_clip_is_one_item_in_the_tarteel_v1_record_shape():
    payloads = {0: b"skipped", 1: b"clip-1"}
    row = _manifest_row(1, label="2:2", stratum="gating")
    calls: list[str] = []
    ((keys, samples, (record,)),) = _loaded([row], payloads, calls)
    assert keys == ("2_1_1.flac#0",)
    assert len(samples) == len(b"clip-1") * SAMPLES_PER_BYTE
    assert record == {
        "key": "2_1_1.flac#0", "audio_filename": "2_1_1.flac",
        "audio_sha256": _sha256(b"clip-1"), "surah_ayah": "2:2", "word_from": 1, "word_to": 5,
        "start_sample": 0, "end_sample": len(samples), "split": "clean", "shard_file": SHARD,
        "row": 1, "stratum": "gating", "assigned_label": "2:2"}
    assert calls == [f"data/{SHARD}.parquet"]


def test_an_unlabelled_row_is_decoded_with_a_null_ayah_and_carries_its_label_and_stratum():
    row = _manifest_row(1, label=None, stratum="reported:unclean", assigned_label="2:1")
    ((_, _, (record,)),) = _loaded([row], {1: b"clip-1"})
    assert (record["surah_ayah"], record["word_from"], record["word_to"]) == (None, None, None)
    assert (record["assigned_label"], record["stratum"]) == ("2:1", "reported:unclean")


def test_the_shard_is_read_once_for_all_its_wanted_rows_in_row_order():
    payloads = {0: b"a", 1: b"bb", 2: b"ccc", 3: b"dddd"}
    wanted = [_manifest_row(3, payload=b"dddd"), _manifest_row(1, payload=b"bb")]
    calls: list[str] = []
    loaded = _loaded(wanted, payloads, calls)
    assert [keys for keys, _, _ in loaded] == [("2_1_1.flac#0",), ("2_1_3.flac#0",)]
    assert [records[0]["row"] for _, _, records in loaded] == [1, 3]
    assert len(calls) == 1


def test_a_wrong_audio_checksum_is_refused_before_anything_is_decoded(tmp_path):
    row = _manifest_row(1, audio_sha256=_sha256(b"another clip"))
    decoder = FakeDecoder()
    with pytest.raises(ValueError):
        ted.run_shard(SHARD, [row], tmp_path, {"base": decoder}, stream=_stream({1: b"clip-1"}),
                      word_counts=_word_counts, load_audio=_load_audio)
    assert decoder.checks == 0 and decoder.streams == []


def test_a_wanted_row_missing_from_the_streamed_shard_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        ted.run_shard(SHARD, [_manifest_row(5)], tmp_path, {"base": FakeDecoder()},
                      stream=_stream({0: b"a"}), word_counts=_word_counts,
                      load_audio=_load_audio)


# --- the stores and one shard read for every model ----------------------------------------------


def test_the_stores_are_the_check_sequence_and_the_flushed_b0_stream_and_no_whole_clip():
    assert set(ted.STORES) == {"checks", "stream_b0"}
    decoder = FakeDecoder()
    span = np.zeros(160, dtype=np.float32)
    line = {}
    for store in ted.STORES.values():
        line.update(store.decode(decoder, span))
    assert line == {"checks": [{"hop": "fake:160", "overlap": "", "flush": True}],
                    "stream_b0": "fake:stream:160"}
    assert decoder.streams == [(0, True)]


def test_one_shard_read_decodes_every_row_with_every_model_once(tmp_path):
    payloads = {0: b"a", 1: b"bb", 2: b"ccc"}
    rows = [_manifest_row(0, payload=b"a"), _manifest_row(2, payload=b"ccc", label=None,
                                                          stratum="reported:unclean")]
    decoders = {name: FakeDecoder(name) for name in ted.MODELS}
    calls: list[str] = []
    timing = ted.run_shard(SHARD, rows, tmp_path, decoders, stream=_stream(payloads, calls),
                           word_counts=_word_counts, load_audio=_load_audio)
    assert calls == [f"data/{SHARD}.parquet"]
    assert timing["items"] == 2
    for name, decoder in decoders.items():
        assert decoder.checks == 2 and len(decoder.streams) == 2
        lines = [json.loads(line) for line in
                 ted.decodes_path(tmp_path, SHARD, name).read_text(encoding="utf-8").splitlines()]
        assert [line["key"] for line in lines] == ["2_1_0.flac#0", "2_1_2.flac#0"]
        assert [line["stream_b0"] for line in lines] == [f"{name}:stream:1000",
                                                         f"{name}:stream:3000"]
        assert all("spans" not in line and line["checks"] for line in lines)
        assert lines[1]["surah_ayah"] is None and lines[0]["word_to"] == 3


def test_a_rerun_decodes_nothing_and_reads_no_audio(tmp_path):
    payloads = {0: b"a", 1: b"bb"}
    rows = [_manifest_row(0, payload=b"a"), _manifest_row(1, payload=b"bb")]
    decoders = {"base": FakeDecoder("base")}
    kwargs = {"word_counts": _word_counts, "load_audio": _load_audio}
    ted.run_shard(SHARD, rows, tmp_path, decoders, stream=_stream(payloads), **kwargs)
    audio_loads: list[bytes] = []
    again = ted.run_shard(SHARD, rows, tmp_path, decoders, stream=_stream(payloads),
                          word_counts=_word_counts,
                          load_audio=lambda data: audio_loads.append(data) or _load_audio(data))
    assert again["items"] == 0 and audio_loads == [] and decoders["base"].checks == 2


def test_no_audio_is_written_next_to_the_decodes(tmp_path):
    payload = b"unmistakable-audio-bytes"
    ted.run_shard(SHARD, [_manifest_row(0, payload=payload)], tmp_path,
                  {"base": FakeDecoder()}, stream=_stream({0: payload}),
                  word_counts=_word_counts, load_audio=_load_audio)
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written and all(path.suffix in (".jsonl", ".json") for path in written)
    assert all(payload not in path.read_bytes() for path in written)


# --- merging the parts --------------------------------------------------------------------------


def _parts(tmp_path: Path, rows, *, drop=None, repeat=None, extra=None) -> Path:
    """Per-shard parts for every model, written as ``run_shard`` writes them (shuffled order)."""
    parts = tmp_path / "parts"
    for model in ted.MODELS:
        lines = [{"key": f"{row['audio_path']}#0", "stream_b0": f"{model}:{row['row']}",
                  "checks": []} for row in rows]
        lines = lines[::-1]
        if model == drop:
            lines = lines[1:]
        if model == repeat:
            lines = lines + lines[:1]
        if model == extra:
            lines = lines + [{"key": "not-in-the-manifest.flac#0", "stream_b0": "", "checks": []}]
        _write_jsonl(ted.decodes_path(parts, SHARD, model), lines)
    return parts


def _manifest(tmp_path: Path, rows) -> Path:
    path = tmp_path / "look.jsonl"
    _write_jsonl(path, rows)
    return path


def test_the_merge_writes_one_sorted_file_per_model_and_a_meta(tmp_path):
    rows = [_manifest_row(index) for index in (4, 1, 3)]
    manifest = _manifest(tmp_path, rows)
    out = tmp_path / "decodes"
    provenance = {"code_sha": "c" * 40, "timings": {"wall_seconds": 12.5}}
    counts = ted.merge_decodes(_parts(tmp_path, rows), out, manifest_path=manifest,
                               provenance=provenance)
    assert counts == {model: 3 for model in ted.MODELS}
    for model in ted.MODELS:
        with gzip.open(out / f"tlog.look.{model}.jsonl.gz", "rt", encoding="utf-8") as handle:
            keys = [json.loads(line)["key"] for line in handle]
        assert keys == sorted(f"{row['audio_path']}#0" for row in rows)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["code_sha"] == "c" * 40 and meta["timings"] == {"wall_seconds": 12.5}
    assert meta["manifest_sha256"] == _sha256(manifest.read_bytes())
    assert meta["manifest_rows"] == 3
    assert meta["models"] == list(ted.MODELS)
    assert meta["rows"] == {model: 3 for model in ted.MODELS}


@pytest.mark.parametrize("fault", ["drop", "repeat", "extra"])
def test_the_merge_refuses_a_model_that_misses_repeats_or_adds_a_row(tmp_path, fault):
    rows = [_manifest_row(index) for index in (1, 2, 3)]
    parts = _parts(tmp_path, rows, **{fault: "c95"})
    with pytest.raises(ValueError, match="c95"):
        ted.merge_decodes(parts, tmp_path / "decodes", manifest_path=_manifest(tmp_path, rows),
                          provenance={})


# --- the slice gate -----------------------------------------------------------------------------


def test_the_slice_projects_the_full_wall_time_from_audio_seconds():
    verdict = ted.slice_verdict(slice_audio_s=500.0, slice_wall_s=1_000.0, total_audio_s=18_000.0)
    assert verdict["projected_hours"] == pytest.approx(10.0)
    assert verdict["within_budget"] is False


def test_the_budget_is_inclusive_at_the_limit():
    at_limit = ted.slice_verdict(500.0, 500.0, 28_800.0)  # exactly 8 h
    assert at_limit["projected_hours"] == pytest.approx(8.0) and at_limit["within_budget"] is True
    assert ted.slice_verdict(500.0, 500.0, 28_801.0)["within_budget"] is False
    assert ted.slice_verdict(500.0, 500.0, 36_000.0, limit_hours=10.0)["within_budget"] is True


# --- the narrowed parallel run (architect r26, r27) ---------------------------------------------


OTHER_SHARD = "clean-00001-of-00411"


def _parts_with_timing(tmp_path: Path, *, audio_s: float, items: int = 3,
                       shards: int = 12) -> Path:
    lines = [{"shard_file": f"clean-{index:05d}-of-00411", "items": items,
              "audio_seconds": audio_s / shards, "wall_seconds": 1.0,
              "decode_seconds": {"base": {"checks": 0.1}}} for index in range(shards)]
    _write_jsonl(tmp_path / "timing.jsonl", lines)
    return tmp_path


def test_the_run_is_the_three_gating_models_with_checks_only_and_omp_one():
    assert ted.RUN_MODELS == ("base", "c95", "candidate_ema")
    assert set(ted.RUN_STORES) == {"checks"}
    assert (ted.RUN_PROCESSES, ted.RUN_THREADS) == (6, 1)
    assert (ted.GATE_ROWS, ted.RUN_LIMIT_HOURS) == (12, 11.0)


def test_the_gating_selection_is_the_committed_gating_stratum():
    rows = ted.read_manifest("look")
    gating = ted.gating_rows(rows)
    assert len(gating) == 2_853
    assert all(row["stratum"] == "gating" for row in gating)


def test_the_selection_audio_joins_the_identified_tree_by_row_key():
    rows = ted.read_manifest("look")
    gating = ted.gating_rows(rows)
    keys = {_key(row) for row in gating}
    expected = 0.0
    for path in sorted(ted.IDENT_LOOK_DIR.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if _key(row) in keys:
                    expected += row["duration_s"]
    assert ted.selection_audio_seconds(gating) == pytest.approx(expected)
    assert 0 < expected < ted.selection_audio_seconds(rows)


@pytest.mark.parametrize("limit,within", [(11.0, True), (1.0, True), (0.9, False)])
def test_the_gate_projects_the_selection_from_the_wall_the_run_has_taken(
        tmp_path, monkeypatch, limit, within):
    parts = _parts_with_timing(tmp_path, audio_s=600.0)
    monkeypatch.setattr(ted, "read_manifest", lambda *args, **kwargs: [])
    monkeypatch.setattr(ted, "selection_audio_seconds", lambda rows: 3_600.0)
    record = ted.gate_verdict(parts, elapsed_seconds=600.0, limit_hours=limit,
                              processes=6, threads=1)
    assert record["projected_hours"] == pytest.approx(1.0)  # 600 s of audio covered in 600 s
    assert record["within_budget"] is within
    written = json.loads((tmp_path / "gate_timing.json").read_text(encoding="utf-8"))
    assert (written["shards"], written["processes"], written["threads"]) == (12, 6, 1)


def test_a_process_decodes_only_its_shard_subset_with_the_runs_checks(monkeypatch, tmp_path):
    shards = [SHARD, OTHER_SHARD, "clean-00002-of-00411", "clean-00003-of-00411"]
    rows = [_manifest_row(index, shard_file=shard, payload=bytes([index + 1]))
            for index, shard in enumerate(shards)]
    decoder = FakeDecoder("base")
    monkeypatch.setattr(ted, "read_manifest", lambda *args, **kwargs: rows)
    monkeypatch.setattr(ted, "load_models", lambda parts_dir, models: {"base": decoder})
    monkeypatch.setattr(ted, "ayah_word_counts", lambda: _word_counts)
    monkeypatch.setattr(ted, "default_audio", _load_audio)
    payloads = {f"data/{shard}.parquet": bytes([index + 1])
                for index, shard in enumerate(shards)}
    monkeypatch.setattr(ted, "shard_stream", lambda *args, **kwargs: lambda name: iter(
        [{"audio": {"bytes": payloads[name]}}]))
    result = ted.decode_shards(parts_dir=tmp_path, cache_dir=tmp_path, models=("base",),
                               shard_index=1, shard_count=2)
    assert result["shards"] == 2 and decoder.checks == 2 and decoder.streams == []
    assert sorted(path.name for path in tmp_path.glob("tlog.look.*.jsonl")) == [
        f"tlog.look.{OTHER_SHARD}.base.jsonl", f"tlog.look.{shards[3]}.base.jsonl"]


def test_the_merge_checks_the_gating_selection_not_the_whole_look_manifest(tmp_path):
    rows = [_manifest_row(1), _manifest_row(2, stratum="reported:unclean")]
    parts = tmp_path / "parts"
    _write_jsonl(ted.decodes_path(parts, SHARD, "base"),
                 [{"key": "2_1_1.flac#0", "checks": []}])
    counts = ted.merge_decodes(parts, tmp_path / "decodes",
                               manifest_path=_manifest(tmp_path, rows), provenance={},
                               models=("base",))
    assert counts == {"base": 1}
    meta = json.loads((tmp_path / "decodes" / "meta.json").read_text(encoding="utf-8"))
    assert meta["manifest_rows"] == 1 and meta["rows"] == {"base": 1}


def test_the_merge_refuses_a_part_row_outside_the_gating_selection(tmp_path):
    rows = [_manifest_row(1), _manifest_row(2, stratum="reported:unclean")]
    parts = tmp_path / "parts"
    _write_jsonl(ted.decodes_path(parts, SHARD, "base"),
                 [{"key": "2_1_1.flac#0", "checks": []},
                  {"key": "2_1_2.flac#0", "checks": []}])
    with pytest.raises(ValueError, match="base"):
        ted.merge_decodes(parts, tmp_path / "decodes", manifest_path=_manifest(tmp_path, rows),
                          provenance={}, models=("base",))

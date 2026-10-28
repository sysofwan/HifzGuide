"""Tests for the shared-feature wiring of ``tadabur.tlog_eval_decode`` and its parity record (#213).

Round 2 decodes the same look gating clips with three models. The check features do not depend
on the model, so the run featurises each clip once and feeds every model
(``training.external_decode.decode_recordings(share_features=True)``). The decode must stay
bit-identical to #212's committed look decodes, which this file pins three ways:

* the wiring: ``run_shard`` forwards the flag (off by default, so every existing caller is
  unchanged), ``decode_shards`` and the CLI turn it on unless told not to, and the per-shard
  timing carries the shared feature seconds into the run's totals;
* the comparison: :func:`parity_record` re-reads a shared re-decode's parts and compares them,
  line for line, with the committed decodes of the same shards;
* the record: the committed ``shared_parity.json`` of the real cuda-dev run on the first
  :data:`PARITY_SHARDS` gating shards, which must agree with the committed decodes it names.
  The real-data parity is an acceptance criterion, not something a fake can stand in for, so
  the record carries what shows the shared decode ran: the items it decoded, its timing and
  the digest of every part file it wrote.

The decoders are real ``Decoder`` s around the fakes of ``training.test_shared_check_features``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys

import pytest

from tadabur import tlog_eval_decode as ted
from tadabur.test_tlog_eval_decode import (
    LAUNCHER,
    OTHER_SHARD,
    SHARD,
    _launcher_call,
    _launcher_env,
    _load_audio,
    _manifest_row,
    _stream,
    _stub_launcher_box,
    _word_counts,
    _write_jsonl,
)
from tadabur.tlog_index import write_shard
from training import decoding as dc
from training.test_shared_check_features import _decoder

MODELS = ("base", "c95", "candidate_ema")
#: Two clips of 3 s and 4.5 s: each is several checks, so a repeated featurisation shows.
PAYLOADS = {0: b"a" * 3_000, 1: b"b" * 4_500}


def _decoders() -> dict:
    """Three models over equal extractors: batch sizes and weights differ, features do not."""
    return {"base": _decoder(0, 1, padding_value=1.0), "c95": _decoder(7, 3, padding_value=1.0),
            "candidate_ema": _decoder(19, 16, padding_value=1.0)}


def _rows(shard_file: str = SHARD) -> list[dict]:
    return [_manifest_row(index, shard_file=shard_file, payload=payload)
            for index, payload in PAYLOADS.items()]


def _pieces() -> int:
    return sum(len(dc.check_schedule(len(_load_audio(payload)))) for payload in PAYLOADS.values())


def _calls(decoders: dict) -> int:
    return sum(len(decoder.extractor.lengths) for decoder in decoders.values())


def _run_shard(directory, decoders: dict, **flag) -> dict:
    return ted.run_shard(SHARD, _rows(), directory, decoders, stream=_stream(PAYLOADS),
                         word_counts=_word_counts, load_audio=_load_audio,
                         stores=ted.RUN_STORES, **flag)


def _parts(directory) -> dict[str, bytes]:
    return {model: ted.decodes_path(directory, SHARD, model).read_bytes() for model in MODELS}


# --- run_shard ----------------------------------------------------------------------------------


def test_a_shard_is_decoded_unshared_unless_asked(tmp_path):
    decoders = _decoders()

    timing = _run_shard(tmp_path, decoders)

    assert _calls(decoders) == 3 * _pieces()
    assert "shared_feature_seconds" not in timing


def test_a_shared_shard_featurises_once_and_writes_the_same_bytes(tmp_path):
    unshared_decoders, shared_decoders = _decoders(), _decoders()
    _run_shard(tmp_path / "unshared", unshared_decoders, share_features=False)

    timing = _run_shard(tmp_path / "shared", shared_decoders, share_features=True)

    assert _calls(shared_decoders) == _pieces()
    assert _parts(tmp_path / "shared") == _parts(tmp_path / "unshared")
    assert set(timing["shared_feature_seconds"]) == {"checks"}


# --- decode_shards and the CLI ------------------------------------------------------------------


def _patch_run(monkeypatch, decoders: dict, rows: list[dict]) -> None:
    """Stand in for the manifest, the models and the shard stream, as the run's own tests do."""
    by_shard: dict[str, list[bytes]] = {}
    for row in rows:
        by_shard.setdefault(f"data/{row['shard_file']}.parquet", []).append(PAYLOADS[row["row"]])
    monkeypatch.setattr(ted, "read_manifest", lambda *args, **kwargs: rows)
    monkeypatch.setattr(ted, "load_models", lambda parts_dir, models: decoders)
    monkeypatch.setattr(ted, "ayah_word_counts", lambda: _word_counts)
    monkeypatch.setattr(ted, "default_audio", _load_audio)
    monkeypatch.setattr(ted, "shard_stream", lambda *args, **kwargs: lambda name: iter(
        [{"audio": {"bytes": payload}} for payload in by_shard[name]]))


def test_a_process_shares_features_by_default_and_not_when_told_not_to(monkeypatch, tmp_path):
    rows = _rows()
    shared, unshared = _decoders(), _decoders()

    _patch_run(monkeypatch, shared, rows)
    result = ted.decode_shards(parts_dir=tmp_path / "shared", cache_dir=tmp_path)
    _patch_run(monkeypatch, unshared, rows)
    plain = ted.decode_shards(parts_dir=tmp_path / "unshared", cache_dir=tmp_path,
                              share_features=False)

    assert _calls(shared) == _pieces() and _calls(unshared) == 3 * _pieces()
    assert set(result["shared_feature_seconds"]) == {"checks"}
    assert "shared_feature_seconds" not in plain
    assert _parts(tmp_path / "shared") == _parts(tmp_path / "unshared")
    timing = [json.loads(line) for line in
              (tmp_path / "shared" / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
    assert set(timing[0]["shared_feature_seconds"]) == {"checks"}


def test_a_process_can_be_restricted_to_named_gating_shards(monkeypatch, tmp_path):
    shards = [SHARD, OTHER_SHARD, "clean-00002-of-00411", "clean-00003-of-00411"]
    rows = [_manifest_row(0, shard_file=shard, payload=PAYLOADS[0]) for shard in shards]
    _patch_run(monkeypatch, _decoders(), rows)

    result = ted.decode_shards(parts_dir=tmp_path, cache_dir=tmp_path, models=MODELS,
                               shard_files=[shards[3], shards[1]])

    assert result["shards"] == 2
    written = sorted(path.name for path in tmp_path.glob("tlog.look.*.base.jsonl"))
    assert written == [f"tlog.look.{shards[1]}.base.jsonl", f"tlog.look.{shards[3]}.base.jsonl"]


def test_the_index_and_count_split_the_named_shards(monkeypatch, tmp_path):
    shards = [SHARD, OTHER_SHARD, "clean-00002-of-00411", "clean-00003-of-00411"]
    rows = [_manifest_row(0, shard_file=shard, payload=PAYLOADS[0]) for shard in shards]
    _patch_run(monkeypatch, _decoders(), rows)

    ted.decode_shards(parts_dir=tmp_path, cache_dir=tmp_path, models=MODELS,
                      shard_files=shards[:3], shard_index=1, shard_count=2)

    assert [path.name for path in tmp_path.glob("tlog.look.*.base.jsonl")] == [
        f"tlog.look.{shards[1]}.base.jsonl"]


def test_a_named_shard_that_holds_no_gating_row_is_refused(monkeypatch, tmp_path):
    _patch_run(monkeypatch, _decoders(), _rows())

    with pytest.raises(ValueError, match="not a gating shard"):
        ted.decode_shards(parts_dir=tmp_path, cache_dir=tmp_path, models=MODELS,
                          shard_files=[SHARD, OTHER_SHARD])
    assert not list(tmp_path.glob("tlog.look.*.jsonl"))


def _cli(monkeypatch, *argv: str, **stubs) -> dict:
    """Run ``main`` on ``argv`` with ``stubs`` standing in for the named functions; each stub
    records its keyword arguments and returns ``{}``."""
    seen = {}
    for name, result in stubs.items():
        monkeypatch.setattr(ted, name, lambda *args, _name=name, _result=result, **kwargs:
                            seen.update({_name: kwargs}) or _result)
    monkeypatch.setattr(sys, "argv", ["tlog_eval_decode", *argv])
    ted.main()
    return seen


def test_the_decode_command_shares_features_unless_told_not_to(monkeypatch, tmp_path):
    base = ["decode", "--parts-dir", str(tmp_path), "--cache-dir", str(tmp_path)]

    default = _cli(monkeypatch, *base, decode_shards={})["decode_shards"]
    off = _cli(monkeypatch, *base, "--no-share-features", decode_shards={})["decode_shards"]

    assert default["share_features"] is True and off["share_features"] is False
    assert default["shard_files"] is None


def test_the_decode_command_forwards_the_named_shards(monkeypatch, tmp_path):
    argv = ["decode", "--parts-dir", str(tmp_path), "--cache-dir", str(tmp_path),
            "--shard-files", SHARD, OTHER_SHARD]

    seen = _cli(monkeypatch, *argv, decode_shards={})["decode_shards"]

    assert list(seen["shard_files"]) == [SHARD, OTHER_SHARD]


# --- the shared seconds in the run's totals -----------------------------------------------------


def _timing_line(shard_file: str, shared: dict | None) -> dict:
    line = {"shard_file": shard_file, "items": 1, "audio_seconds": 1.0, "wall_seconds": 1.0,
            "decode_seconds": {"base": {"checks": 1.0}}}
    return line if shared is None else {**line, "shared_feature_seconds": shared}


def test_the_totals_sum_the_shared_feature_seconds_per_store(tmp_path):
    _write_jsonl(tmp_path / "timing.jsonl", [
        _timing_line(SHARD, {"checks": 1.5}), _timing_line(OTHER_SHARD, {"checks": 2.5}),
        _timing_line("clean-00002-of-00411", None)])  # a line from an unshared visit

    totals = ted.timing_totals(tmp_path, models=("base",))

    assert totals["shared_feature_seconds"] == {"checks": 4.0}
    assert totals["decode_seconds"] == {"base": {"checks": 3.0}}


def test_the_totals_of_an_unshared_run_keep_their_shape(tmp_path):
    _write_jsonl(tmp_path / "timing.jsonl", [_timing_line(SHARD, None)])

    totals = ted.timing_totals(tmp_path, models=("base",))

    assert "shared_feature_seconds" not in totals


# --- the parity comparison ----------------------------------------------------------------------


def _line(key: str, shard_file: str, checks: str, span=(1, 3)) -> dict:
    return {"key": key, "shard_file": shard_file, "audio_sha256": key * 2, "stratum": "gating",
            "checks": [{"hop": checks, "overlap": "", "flush": False}],
            "word_from": span[0], "word_to": span[1]}


def test_the_parity_digest_is_of_the_key_sorted_decode_lines():
    lines = [_line("b", SHARD, "x"), _line("a", SHARD, "y")]

    digest = ted.parity_digest(lines)

    assert digest == ted.parity_digest(list(reversed(lines)))
    assert digest == ted.parity_digest([dict(reversed(list(line.items()))) for line in lines])
    assert digest != ted.parity_digest([_line("b", SHARD, "x"), _line("a", SHARD, "z")])
    assert digest != ted.parity_digest(lines[:1])
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_the_parity_digest_leaves_out_the_span_the_merge_joins_in():
    """``word_from``/``word_to`` are the identified tree's, written at merge, not a decode."""
    first = [_line("a", SHARD, "x", span=(1, 3))]
    second = [_line("a", SHARD, "x", span=(4, 9))]

    assert ted.parity_digest(first) == ted.parity_digest(second)


def _committed(tmp_path, per_shard: dict[str, list[dict]]) -> None:
    write_shard([line for lines in per_shard.values() for line in lines],
                tmp_path / "committed" / "tlog.look.base.jsonl.gz")


def _shared_parts(tmp_path, per_shard: dict[str, list[dict]], timed: bool = True) -> None:
    """The parts a shared re-decode wrote, and the timing line each shard's visit appended
    (``timed=False``: parts that were put there, not decoded)."""
    for shard_file, lines in per_shard.items():
        _write_jsonl(ted.decodes_path(tmp_path / "parts", shard_file, "base"),
                     [{**line, "word_from": 1, "word_to": 99} for line in lines])
    if timed:
        _write_jsonl(tmp_path / "parts" / "timing.jsonl", [
            {"shard_file": shard_file, "items": len(lines), "audio_seconds": 10.0 * len(lines),
             "wall_seconds": 5.0, "decode_seconds": {"base": {"checks": 2.0}},
             "shared_feature_seconds": {"checks": 1.0}}
            for shard_file, lines in per_shard.items()])


def _parity(tmp_path, monkeypatch, shards=(SHARD, OTHER_SHARD)) -> dict:
    monkeypatch.setattr(ted, "head_sha", lambda *args, **kwargs: "abc123")
    return ted.parity_record(tmp_path / "parts", models=("base",), shards=list(shards),
                             decodes_dir=tmp_path / "committed", processes=6, threads=1)


def _per_shard() -> dict[str, list[dict]]:
    return {SHARD: [_line("a", SHARD, "x"), _line("b", SHARD, "y")],
            OTHER_SHARD: [_line("c", OTHER_SHARD, "z")],
            "clean-00002-of-00411": [_line("d", "clean-00002-of-00411", "w")]}


def test_identical_parts_are_the_committed_decodes_of_those_shards(tmp_path, monkeypatch):
    per_shard = _per_shard()
    _committed(tmp_path, per_shard)  # a third shard is committed but not compared
    _shared_parts(tmp_path, {s: per_shard[s] for s in (SHARD, OTHER_SHARD)})

    record = _parity(tmp_path, monkeypatch)

    assert record["models"] == ["base"] and record["shards"] == [SHARD, OTHER_SHARD]
    assert record["rows"] == {"base": 3}
    assert record["identical"] == {"base": True} and record["mismatches"] == {"base": 0}
    assert record["shared_sha256"] == record["committed_sha256"]
    assert (record["processes"], record["threads"], record["share_features"]) == (6, 1, True)
    assert record["code_sha"] == "abc123"
    assert record["items"] == 3 == record["rows"]["base"]
    assert record["timing"] == ted.timing_totals(tmp_path / "parts", models=("base",))
    assert record["timing"]["items"] == 3 and record["timing"]["shards"] == 2
    assert record["part_sha256"] == {"base": {
        shard: hashlib.sha256(ted.decodes_path(tmp_path / "parts", shard, "base")
                              .read_bytes()).hexdigest() for shard in (SHARD, OTHER_SHARD)}}


def test_one_changed_check_is_a_mismatch(tmp_path, monkeypatch):
    per_shard = _per_shard()
    _committed(tmp_path, per_shard)
    changed = {**per_shard, OTHER_SHARD: [_line("c", OTHER_SHARD, "Z")]}
    _shared_parts(tmp_path, {s: changed[s] for s in (SHARD, OTHER_SHARD)})

    record = _parity(tmp_path, monkeypatch)

    assert record["identical"] == {"base": False} and record["mismatches"] == {"base": 1}
    assert record["shared_sha256"] != record["committed_sha256"]


def test_a_row_the_shared_decode_lacks_is_a_mismatch(tmp_path, monkeypatch):
    per_shard = _per_shard()
    _committed(tmp_path, per_shard)
    short = {SHARD: per_shard[SHARD][:1], OTHER_SHARD: per_shard[OTHER_SHARD]}
    _shared_parts(tmp_path, short)

    record = _parity(tmp_path, monkeypatch)

    assert record["identical"] == {"base": False} and record["mismatches"] == {"base": 1}


def test_parts_that_were_never_decoded_record_no_items(tmp_path, monkeypatch):
    """A parts directory seeded from the committed decodes (or kept from an earlier run) is
    identical to them, but nothing was decoded: the record's items are what tell."""
    per_shard = _per_shard()
    _committed(tmp_path, per_shard)
    _shared_parts(tmp_path, {s: per_shard[s] for s in (SHARD, OTHER_SHARD)}, timed=False)

    record = _parity(tmp_path, monkeypatch)

    assert record["identical"] == {"base": True} and record["rows"] == {"base": 3}
    assert record["items"] == 0 != record["rows"]["base"]


def test_the_parity_shards_are_the_first_twenty_gating_shards():
    gating = sorted({row["shard_file"] for row in ted.gating_rows(ted.read_manifest("look"))})

    assert ted.PARITY_SHARDS == 20
    assert ted.parity_shards() == gating[:20]


def test_the_parity_command_writes_the_record_of_the_first_twenty_shards(monkeypatch, tmp_path):
    out = tmp_path / "parity.json"
    record = {"models": ["base"], "identical": {"base": True}}
    argv = ["parity", "--parts-dir", str(tmp_path / "parts"), "--out", str(out)]

    seen = _cli(monkeypatch, *argv, parity_record=record)["parity_record"]

    assert json.loads(out.read_text(encoding="utf-8")) == record
    assert list(seen["shards"]) == ted.parity_shards()
    assert list(seen["models"]) == list(ted.RUN_MODELS)
    assert (seen["processes"], seen["threads"]) == (ted.RUN_PROCESSES, ted.RUN_THREADS)


# --- the committed record of the real-data parity -----------------------------------------------


def _committed_decodes(model: str) -> list[dict]:
    with gzip.open(ted.DECODES_ROOT / f"tlog.look.{model}.jsonl.gz", "rt",
                   encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_the_committed_parity_record_matches_the_committed_decodes_of_the_first_twenty_shards():
    record = json.loads(ted.PARITY_PATH.read_text(encoding="utf-8"))
    shards = ted.parity_shards()

    assert record["shards"] == shards and len(shards) == 20
    assert record["models"] == list(ted.RUN_MODELS)
    assert record["share_features"] is True
    assert (record["processes"], record["threads"]) == (ted.RUN_PROCESSES, ted.RUN_THREADS)
    assert re.fullmatch(r"[0-9a-f]{40}", record["code_sha"])
    for model in ted.RUN_MODELS:
        committed = [line for line in _committed_decodes(model) if line["shard_file"] in shards]
        assert committed and record["rows"][model] == len(committed)
        assert record["committed_sha256"][model] == ted.parity_digest(committed)
        assert record["shared_sha256"][model] == record["committed_sha256"][model]
        assert record["identical"][model] is True and record["mismatches"][model] == 0
        assert record["items"] == record["rows"][model]  # every row was decoded by this run
        digests = record["part_sha256"][model]
        assert list(digests) == shards and len(set(digests.values())) == len(shards)
        assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in digests.values())
    timing = record["timing"]
    assert (timing["items"], timing["shards"]) == (record["items"], len(shards))
    assert timing["audio_seconds"] > 0 and timing["wall_seconds_summed_over_processes"] > 0
    assert set(timing["decode_seconds"]) == set(ted.RUN_MODELS)
    assert all(seconds["checks"] > 0 for seconds in timing["decode_seconds"].values())
    assert timing["shared_feature_seconds"]["checks"] > 0  # the features were shared
    parts = [digest for digests in record["part_sha256"].values() for digest in digests.values()]
    assert len(set(parts)) == len(parts)  # one file per model and shard, none copied


# --- the launcher -------------------------------------------------------------------------------


def _decode_workers(tmp_path, **overrides) -> list[str]:
    env = _launcher_env(tmp_path, **overrides)
    binaries, log = _stub_launcher_box(tmp_path, env)
    assert _launcher_call(tmp_path, "decode", "write_runner abc123", env).returncode == 0
    run = subprocess.run(["bash", str(tmp_path / "root" / "run-decode.sh")],
                         env={**env, "PATH": f"{binaries}:{env['PATH']}"},
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return [line for line in log.read_text(encoding="utf-8").splitlines()
            if "tadabur.tlog_eval_decode decode" in line]


def test_the_launcher_workers_share_features_by_default(tmp_path):
    workers = _decode_workers(tmp_path)

    assert len(workers) == 3
    assert not any("--no-share-features" in line for line in workers)


def test_the_launcher_can_turn_sharing_off_for_a_benchmark(tmp_path):
    workers = _decode_workers(tmp_path, SHARE_FEATURES="0")

    assert len(workers) == 3 and all("--no-share-features" in line for line in workers)


def test_the_launchers_default_parallelism_is_the_decoders():
    """One measured setting (#213 benchmark), written in both places: they must agree."""
    text = LAUNCHER.read_text(encoding="utf-8")
    defaults = {name: int(re.search(rf"^{name}=\$\{{{name}:-(\d+)\}}", text, re.M).group(1))
                for name in ("PROCESSES", "THREADS")}

    assert defaults == {"PROCESSES": ted.RUN_PROCESSES, "THREADS": ted.RUN_THREADS}

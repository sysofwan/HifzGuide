"""Tests for ``decode_recordings``' opt-in ``share_features`` flag (#213); no GPU, no download.

Off (the default, and what every round-1 script runs), the loop is unchanged: each model
featurises each check of each item itself. On, the ``checks`` store featurises an item once and
every decoder still pending on it runs on those features -- same records, one feature pass.
The decoders are real :class:`training.decoding.Decoder` s around the tiny fakes of
``test_shared_check_features``, so the loop is exercised end to end without a model; the check
sequences are what ``Decoder.app_checks`` returns, which the shared path must reproduce byte for
byte. Other stores are not shared: they are called per decoder, as before.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tadabur.audio import TARGET_SAMPLE_RATE
from training import decoding as dc
from training import external_decode
from training.external_decode import Recording, decode_recordings
from training.test_shared_check_features import _audio, _decoder
from training.truth_baseline import CHECK_STORE, CHECKS, SharedDecode, Store

STORES = {CHECKS: CHECK_STORE}
#: One recording of two items, over different sub-spans of it (neither starts at 0 and the
#: spans differ in length), so a decoder featurising the whole recording would show.
SPANS = {"a#0": (0, 4 * TARGET_SAMPLE_RATE), "b#0": (TARGET_SAMPLE_RATE, 6 * TARGET_SAMPLE_RATE)}


def _recording() -> Recording:
    samples = _audio(6.0, seed=4)
    records = [{"key": key, "start_sample": start, "end_sample": end}
               for key, (start, end) in SPANS.items()]
    return Recording(tuple(SPANS), lambda: (samples, records))


def _decoders() -> dict:
    return {"base": _decoder(0, 1, padding_value=1.0), "c95": _decoder(7, 3, padding_value=1.0)}


def _run(directory, decoders=None, stores=STORES, **flag) -> tuple[dict, dict]:
    decoders = _decoders() if decoders is None else decoders
    paths = {name: directory / f"{name}.jsonl" for name in decoders}
    directory.mkdir(parents=True, exist_ok=True)
    return decode_recordings([_recording()], decoders, paths, "test", stores, **flag), paths


def _extractor_calls(decoders: dict) -> int:
    return sum(len(decoder.extractor.lengths) for decoder in decoders.values())


def _pieces() -> int:
    return sum(len(dc.check_schedule(end - start)) for start, end in SPANS.values())


def _expected(name: str) -> list[dict]:
    """What each decoder's file holds, from an unshared ``app_checks`` over each item's span."""
    decoder = _decoders()[name]
    samples = _audio(6.0, seed=4)
    return [{"key": key, "start_sample": start, "end_sample": end,
             CHECKS: decoder.app_checks(samples[start:end])}
            for key, (start, end) in SPANS.items()]


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_model_featurises_every_check_itself_by_default(tmp_path):
    decoders = _decoders()

    timing, paths = _run(tmp_path, decoders)

    assert _extractor_calls(decoders) == 2 * _pieces()
    assert all(_lines(paths[name]) == _expected(name) for name in decoders)
    assert "shared_feature_seconds" not in timing
    assert set(timing["decode_seconds"]) == set(decoders)


def test_sharing_off_is_the_default(tmp_path):
    off, off_paths = _run(tmp_path / "off", share_features=False)
    default, default_paths = _run(tmp_path / "default")

    assert off_paths.keys() == default_paths.keys()
    assert all(off_paths[name].read_bytes() == default_paths[name].read_bytes()
               for name in off_paths)
    assert "shared_feature_seconds" not in off and "shared_feature_seconds" not in default


def test_sharing_on_writes_byte_identical_records(tmp_path):
    _, plain = _run(tmp_path / "plain")
    _, shared = _run(tmp_path / "shared", share_features=True)

    assert all(plain[name].read_bytes() == shared[name].read_bytes() for name in plain)
    assert all(_lines(shared[name]) == _expected(name) for name in shared)


def test_sharing_on_featurises_each_check_once_for_all_models(tmp_path):
    decoders = _decoders()

    _run(tmp_path, decoders, share_features=True)

    assert _extractor_calls(decoders) == _pieces()


def test_sharing_on_times_the_features_apart_from_each_models_decode(tmp_path):
    decoders = _decoders()

    timing, _ = _run(tmp_path, decoders, share_features=True)

    assert set(timing["decode_seconds"]) == set(decoders)
    assert all(set(arms) == {CHECKS} for arms in timing["decode_seconds"].values())
    assert set(timing["shared_feature_seconds"]) == {CHECKS}


def test_a_resumed_item_with_one_model_left_is_decoded_as_before(tmp_path):
    _, paths = _run(tmp_path / "first")
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    done = _lines(paths["base"])[:1]  # base already holds a#0; c95 holds nothing
    (resumed / "base.jsonl").write_text("".join(json.dumps(line, ensure_ascii=False,
                                                            separators=(",", ":")) + "\n"
                                                for line in done), encoding="utf-8")
    decoders = _decoders()

    _, again = _run(resumed, decoders, share_features=True)

    assert _lines(again["base"]) == _expected("base")
    assert _lines(again["c95"]) == _expected("c95")
    # a#0 has only c95 pending: its own extractor featurises it, unshared. b#0 has both
    # pending: the first decoder's extractor featurises it once for both.
    pieces = {key: [s.end - s.start for s in dc.check_schedule(end - start)]
              for key, (start, end) in SPANS.items()}
    assert decoders["c95"].extractor.lengths == pieces["a#0"]
    assert decoders["base"].extractor.lengths == pieces["b#0"]


def test_the_feature_seconds_are_timed_apart_from_each_models_decode(monkeypatch, tmp_path):
    clock = [0.0]

    def featurize(extractor, span):
        clock[0] += 10.0  # a feature pass costs 10 s
        return len(span)

    def decode_shared(decoder, features):
        clock[0] += 1.0  # each model's own pass costs 1 s
        return {"x": features}

    monkeypatch.setattr(external_decode, "time", SimpleNamespace(time=lambda: clock[0]))
    stores = {"x": Store(tmp_path, {}, lambda decoder, span: {"x": len(span)},
                         shared=SharedDecode(featurize, decode_shared))}

    timing, _ = _run(tmp_path / "out", stores=stores, share_features=True)

    assert timing["shared_feature_seconds"] == {"x": 20.0}  # two items, one pass each
    assert timing["decode_seconds"] == {"base": {"x": 2.0}, "c95": {"x": 2.0}}


def test_other_stores_are_called_per_decoder_as_before(tmp_path):
    calls = []

    def decode(decoder, span):
        calls.append((decoder.model_ref, len(span)))
        return {"extra": len(span)}

    stores = {CHECKS: CHECK_STORE, "extra": Store(tmp_path, {}, decode)}
    decoders = _decoders()

    timing, paths = _run(tmp_path / "out", decoders, stores, share_features=True)

    sizes = [end - start for start, end in SPANS.values()]
    assert calls == [(decoder.model_ref, size) for size in sizes for decoder in decoders.values()]
    assert all(line["extra"] == size for name in paths
               for line, size in zip(_lines(paths[name]), sizes))
    assert set(timing["shared_feature_seconds"]) == {CHECKS}  # the extra store shares nothing


def test_decoders_with_different_extractors_are_refused_before_anything_is_written(tmp_path):
    decoders = {"base": _decoder(0, 1, padding_value=1.0), "c95": _decoder(7, 3, padding_value=0.0)}

    with pytest.raises(ValueError, match="feature extractor"):
        _run(tmp_path, decoders, share_features=True)

    assert not any(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.jsonl"))
    assert _extractor_calls(decoders) == 0


def test_unshared_decoding_does_not_compare_extractors(tmp_path):
    decoders = {"base": _decoder(0, 1, padding_value=1.0), "c95": _decoder(7, 3, padding_value=0.0)}

    _, paths = _run(tmp_path, decoders)

    assert all(_lines(paths[name]) for name in decoders)

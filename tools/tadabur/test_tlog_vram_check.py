"""Tests for the P3 VRAM check's record and its pass/fallback rule (#216).

The committed ``vram_check.json`` is the measurement round 2 launches on: it must name the
parity selection, the four models, one thread and the peaks of every setting tried, and the
count P3 runs at must be the largest whose peak fits the card's budget. The real reading
cannot be re-derived by a fake, so the record is pinned as it was written; the fallback rule
is a unit test, because the failure that matters is retrying at the wrong count.
"""

from __future__ import annotations

import json
import re

from tadabur import tlog_eval_decode as ted
from tadabur import tlog_vram_check as tv


def test_the_committed_vram_check_records_a_passing_setting_of_the_parity_decode():
    record = json.loads(tv.VRAM_PATH.read_text(encoding="utf-8"))

    assert re.fullmatch(r"[0-9a-f]{40}", record["code_sha"])
    assert record["pull_request"] is None or re.fullmatch(
        rf"refs/pull/{record['pull_request']}/head", record["pull_request_ref"])
    assert record["models"] == list(ted.MODELS)
    assert (record["threads"], record["share_features"]) == (tv.THREADS, True)
    assert record["shards"] == ted.parity_shards() and len(record["shards"]) == 20
    assert record["total_mib"] == 16311
    assert (record["pass_peak_mib"], record["chosen_processes"]) == (14500, 6)
    settings = record["settings"]
    assert [setting["processes"] for setting in settings] == [6]  # one look decode passed
    for setting in settings:
        assert setting["items"] > 0 and setting["samples"] >= 2
        assert setting["peak_mib"] >= setting["baseline_mib"] > 0
        assert setting["peak_mib"] <= record["pass_peak_mib"]
        assert setting["passed"] is True


def test_the_first_passing_setting_is_chosen_and_the_rest_are_not_run(monkeypatch, tmp_path):
    peaks = {6: 15000, 5: 14000, 4: 13000}
    tried = []

    def fake_measure(processes, **kwargs):
        tried.append(processes)
        return {"processes": processes, "peak_mib": peaks[processes], "baseline_mib": 2,
                "samples": 3, "seconds": 1.0, "shards": 1, "items": 1}

    monkeypatch.setattr(tv, "parity_shards", lambda: ["clean-00000-of-00411"])
    monkeypatch.setattr(tv, "head_sha", lambda: "0" * 40)
    monkeypatch.setattr(tv, "measure", fake_measure)
    record = tv.vram_check(parts_root=tmp_path, cache_dir=tmp_path, clip_cache=tmp_path,
                           out=tmp_path / "vram.json", memory=lambda field: 16311)

    assert tried == [6, 5]
    assert [setting["passed"] for setting in record["settings"]] == [False, True]
    assert record["chosen_processes"] == 5
    assert json.loads((tmp_path / "vram.json").read_text(encoding="utf-8")) == record

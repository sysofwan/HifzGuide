"""Round 2's support surface (#227): the identity writer, the excluded-rows disclosure and the
reported shaddah G3 line.

The acceptance tests pin what each phase reads and writes; these pin the three edges a reader
would otherwise have to trust: nothing produced ``models.json`` before, a row the item builder
cannot grade vanished from the population, and "shaddah reported only" implied dropped.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import aug_readout_verdict as av
from training import round2_fixtures as fx
from training import tlog_eval


def test_the_models_writer_builds_the_look_manifest_from_the_pins(tmp_path):
    from training.consonant2_readout import write_models

    spec = fx.filled_spec(root=tmp_path / "readout")
    records = write_models(spec)

    assert set(records) == {f"tlog.look/{model}" for model in
                            (*fx.R2_MODELS, asp.C95, "base")}
    for model, record in ((key.rsplit("/", 1)[1], value) for key, value in records.items()):
        assert ag.validate_identity("tlog.look", model, record, spec=spec) == record["identity"]
    with pytest.raises(SystemExit):
        write_models(spec)


def test_the_models_writer_refuses_an_unpinned_arm(tmp_path):
    from training.consonant2_readout import write_models

    with pytest.raises(SystemExit, match="not pinned"):
        write_models(asp.CONSONANT2, tmp_path / "models.json")


def test_the_look_artifact_names_the_rows_it_left_out(tmp_path):
    repo = fx.init_repo(tmp_path / "repo")
    path, sha = fx.commit_prereg(repo)
    spec = fx.filled_spec(root=repo / "readout", prereg=(path, sha))
    layout = ag.Layout(spec.root)
    layout.root.mkdir()
    layout.models.write_text(json.dumps(fx.look_manifest(spec)), encoding="utf-8")
    items = [fx.tlog_item(n, 1 + n % 3) for n in range(3)]
    sources = fx.TlogSources(tmp_path / "files", items, lambda model, i, p: None, spec=spec)
    read = sources.tlog
    sources.tlog = lambda role, models: dataclasses.replace(
        read(role, models), excluded={"106_1_1482117566.flac#0": "2:254"})

    ar.run_look(sources, layout, spec=spec)

    look = json.loads(layout.look.read_text(encoding="utf-8"))
    assert look["excluded"] == {"keys": ["106_1_1482117566.flac#0"], "count": 1,
                                "ayat": ["2:254"], "reason": tlog_eval.EXCLUDED_REASON}


def test_round_2_shows_the_shaddah_g3_line_beside_the_verdict():
    comparison = av.comparison_name(asp.CANDIDATE, asp.C95)
    report = {"guards": {"qdat_bench": {"shaddah": {
        "comparisons": {comparison: {
            "point": -5.0, "interval": {"lower": -7.0, "upper": -3.0,
                                        "method": "percentile"}}}}}}}
    (line,) = av.consonant_reported(report, asp.CONSONANT2)
    assert line["name"] == "g3.qdat_bench.shaddah" and line["verdict"] == av.FAIL
    assert line["bound"] == -7.0
    assert av.consonant_reported(report, asp.CONSONANT1) == []

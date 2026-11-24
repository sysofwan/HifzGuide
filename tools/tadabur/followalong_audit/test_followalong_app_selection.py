"""A named clip list selects from the claim-gated population, it never replaces it (#309).

``app-run --clips`` is how the P2 and P3 subsets are named. The population
(:func:`tadabur.followalong_audit.clips.app_arm_rows`) is the one authority on which clips the
audit grades and against which ayah, so a named file must name population clips and repeat their
provenance exactly: an arbitrary list — the look tranche's rows, an unopened tranche, a valid key
with a rewritten ``filename_label`` — would otherwise be graded and persisted as this audit's
population.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tadabur.followalong_audit import __main__ as cli
from tadabur.followalong_audit.clips import PROVENANCE_FIELDS, selected_clips


def row(index: int, **changes) -> dict:
    clip = {"audio_path": f"{index}_1_{index}.flac", "audio_sha256": f"{index:064x}",
            "split": "clean", "shard_file": f"clean-{index:05d}-of-00411", "row": index,
            "filename_label": "1:2", "stratum": "gating", "group_key": f"{index}_1_{index}.flac"}
    return {**clip, **changes}


POPULATION = [row(1), row(2)]


def write_list(path, records) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                    encoding="utf-8")


def test_a_named_list_selects_the_populations_own_rows(tmp_path):
    named = [{**POPULATION[1], "assigned_label": "1:2", "decode_mode": "whole_clip"},
             POPULATION[0]]
    chosen = selected_clips(POPULATION, named)
    assert chosen == [POPULATION[1], POPULATION[0]]
    assert sorted(chosen[0]) == sorted(PROVENANCE_FIELDS)


def test_a_clip_outside_the_population_is_refused():
    look = row(3, stratum="reported:below_bar")
    with pytest.raises(ValueError, match="selects from that population"):
        selected_clips(POPULATION, [look])


def test_a_changed_ayah_reference_is_refused():
    with pytest.raises(ValueError, match="disagrees with the population on \\['filename_label'\\]"):
        selected_clips(POPULATION, [{**POPULATION[0], "filename_label": "1:3"}])


def test_a_changed_source_row_is_refused():
    with pytest.raises(ValueError, match="'row'"):
        selected_clips(POPULATION, [{**POPULATION[0], "row": 999}])


def test_a_repeated_clip_is_refused():
    with pytest.raises(ValueError, match="appears twice"):
        selected_clips(POPULATION, [POPULATION[0], POPULATION[0]])


def test_the_app_run_takes_its_named_list_from_the_population(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "app_arm_rows", lambda: POPULATION)
    path = tmp_path / "clips.jsonl"
    write_list(path, [POPULATION[1]])
    args = SimpleNamespace(clips=path, size=None, seed=7)
    assert cli._selected_clips(args) == [POPULATION[1]]


def test_the_app_run_refuses_a_named_list_outside_the_population(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "app_arm_rows", lambda: POPULATION)
    path = tmp_path / "clips.jsonl"
    write_list(path, [row(3)])
    args = SimpleNamespace(clips=path, size=None, seed=7)
    with pytest.raises(ValueError, match="selects from that population"):
        cli._selected_clips(args)

"""Acceptance tests for tlog S8's overlay v1 documentation (#193).

Committed before the code exists and **locked**: later turns may add tests but never weaken or
rewrite these. The overlay is the committed per-row metadata of ``tarteel-ai/tlog`` and may be
published later (S9, the owner's call), so its format is written down once, in
``tools/tadabur/tlog/README.md``, and checked against the committed rows.

Contract this file pins
-----------------------
``tadabur.tlog_schema`` is new and holds the format as data:

* ``STAGES``: stage directory name -> the tuple of every field of that stage's rows, for the four
  row trees ``index``, ``pass1``, ``split`` and ``overlap``. Each equals the key set of **every**
  row of the committed trees (``schema_version`` included where the rows carry it; ``split``
  rows do not).
* ``PUBLIC_DROPPED = ("speaker_cluster", "group_key", "nn_train_sim", "role")``: the fields the
  public subset drops until a tranche is spent (``nn_train_sim`` is listed "if present": no
  committed stage has it). The teacher decodes (``ident/decodes``) are dropped as a whole tree.
* ``public_subset() -> dict[str, tuple[str, ...]]``: every stage's fields without
  ``PUBLIC_DROPPED``, in ``STAGES`` order.

``tools/tadabur/tlog/README.md`` gains a ``## Overlay v1`` section, with these parts:

* a line ``- `<stage>/`: `field`, `field`, ...`` per stage, listing exactly ``STAGES[stage]``,
  before the ``### Public subset`` subsection;
* ``### Public subset``: a line ``- `<stage>`: `field`, ...`` per stage listing exactly
  ``public_subset()[stage]``, and a ``- dropped:`` line naming every ``PUBLIC_DROPPED`` field and
  the teacher decodes;
* every stage directory of ``tools/tadabur/tlog/`` by name (``index/``, ``pass1/``, ``split/``,
  ``overlap/``, ``ident/``, ``tranches/``, ``audit/``, ``s2/``), ``schema_version``,
  ``meta.json`` for the provenance, the join key ``(split, shard_file, row)``, the attribution
  (``tarteel-ai/tlog``, the pinned revision, ``Tarteel``, ``CC BY 4.0``) and that rows are never
  deleted, only excluded with a reason.
"""

from __future__ import annotations

import gzip
import importlib
import json
import re
from pathlib import Path

import pytest

from tadabur.nonpro_sources import TLOG

TLOG_DIR = Path(__file__).parent / "tlog"
README = TLOG_DIR / "README.md"
ROW_STAGES = ("index", "pass1", "split", "overlap")
FIRST_SHARDS = {"clean": "clean-00000-of-00411", "unclean": "unclean-00000-of-00332"}
DROPPED = ("speaker_cluster", "group_key", "nn_train_sim", "role")


@pytest.fixture(scope="module")
def tlog_schema():
    return importlib.import_module("tadabur.tlog_schema")


def _section() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Overlay v1")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def _tokens(line: str) -> list[str]:
    return re.findall(r"`([^`]+)`", line)


def _split_section() -> tuple[str, str]:
    section = _section()
    head, _, public = section.partition("### Public subset")
    assert public, "the Overlay v1 section has no '### Public subset' subsection"
    return head, public


def _stage_lines(text: str, *, directory: bool) -> dict[str, list[str]]:
    pattern = r"^- `(\w+)/`:(.*)$" if directory else r"^- `(\w+)`:(.*)$"
    return {name: _tokens(rest) for name, rest in re.findall(pattern, text, flags=re.M)}


def test_the_public_dropped_fields_are_the_issues_list(tlog_schema):
    assert tuple(tlog_schema.PUBLIC_DROPPED) == DROPPED


def test_the_schema_covers_exactly_the_four_row_trees(tlog_schema):
    assert set(tlog_schema.STAGES) == set(ROW_STAGES)
    assert all(len(set(fields)) == len(fields) for fields in tlog_schema.STAGES.values())


@pytest.mark.parametrize("stage", ROW_STAGES)
@pytest.mark.parametrize("split", ["clean", "unclean"])
def test_each_stages_fields_are_the_keys_of_every_committed_row(stage, split, tlog_schema):
    path = TLOG_DIR / stage / split / f"{FIRST_SHARDS[split]}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows
    assert {frozenset(row) for row in rows} == {frozenset(tlog_schema.STAGES[stage])}


def test_the_join_key_is_in_every_stage_and_schema_version_is_where_rows_carry_it(tlog_schema):
    for stage, fields in tlog_schema.STAGES.items():
        assert {"split", "shard_file", "row"} <= set(fields), stage
    assert "schema_version" in tlog_schema.STAGES["index"]
    assert "schema_version" in tlog_schema.STAGES["pass1"]
    assert "schema_version" in tlog_schema.STAGES["overlap"]
    assert "schema_version" not in tlog_schema.STAGES["split"]


def test_the_public_subset_is_the_schema_without_the_dropped_fields(tlog_schema):
    public = tlog_schema.public_subset()
    assert set(public) == set(tlog_schema.STAGES)
    for stage, fields in tlog_schema.STAGES.items():
        assert public[stage] == tuple(field for field in fields if field not in DROPPED)
    assert not {field for fields in public.values() for field in fields} & set(DROPPED)
    assert {"speaker_cluster", "group_key", "role"} <= set(tlog_schema.STAGES["split"])
    assert set(public["split"]) == {"audio_path", "row", "shard_file", "split"}


def test_the_readme_has_the_overlay_v1_section_and_it_is_the_committed_one(tlog_schema):
    assert "## Overlay v1" in README.read_text(encoding="utf-8")
    assert "tlog_schema" in _section()  # where the format lives as data


def test_the_readme_lists_every_stages_fields_exactly_as_the_schema_does(tlog_schema):
    head, _ = _split_section()
    documented = _stage_lines(head, directory=True)
    assert set(ROW_STAGES) <= set(documented)
    for stage in ROW_STAGES:
        assert set(documented[stage]) == set(tlog_schema.STAGES[stage]), stage
        assert len(documented[stage]) == len(set(documented[stage])), stage


def test_the_readme_public_subset_field_list_matches_the_schema(tlog_schema):
    _, public_text = _split_section()
    documented = _stage_lines(public_text, directory=False)
    expected = tlog_schema.public_subset()
    for stage in ROW_STAGES:
        assert set(documented[stage]) == set(expected[stage]), stage
    [dropped] = [line for line in public_text.splitlines() if line.startswith("- dropped:")]
    assert set(DROPPED) <= set(_tokens(dropped))
    assert "teacher decodes" in dropped
    # a dropped field is never listed as public, in the prose or in a stage line
    assert not {field for stage in ROW_STAGES for field in documented[stage]} & set(DROPPED)


def test_the_readme_names_every_stage_directory_of_the_tlog_tree():
    section = _section()
    directories = sorted(path.name for path in TLOG_DIR.iterdir() if path.is_dir())
    assert {"index", "pass1", "split", "overlap", "ident", "tranches", "audit", "s2"} <= set(
        directories)
    missing = [name for name in directories if f"`{name}/`" not in section]
    assert missing == []


def test_the_readme_documents_the_join_provenance_and_schema_version():
    section = _section()
    assert "(split, shard_file, row)" in section
    assert "`schema_version`" in section
    assert "`meta.json`" in section
    for stage in ROW_STAGES:
        assert (TLOG_DIR / stage / "meta.json").is_file()  # the provenance the README points at


def test_the_readme_attributes_the_source_dataset_and_its_licence():
    section = _section()
    assert TLOG.repo == "tarteel-ai/tlog" and "tarteel-ai/tlog" in section
    assert TLOG.revision in section
    assert "Tarteel" in section
    assert "CC BY 4.0" in section


def test_the_readme_says_rows_are_never_deleted_only_excluded_with_a_reason():
    section = _section().lower()
    assert "never deleted" in section
    assert "excluded" in section and "reason" in section


def test_the_readme_says_the_public_subset_drops_role_until_a_tranche_is_spent():
    _, public_text = _split_section()
    assert "tranche is spent" in public_text

"""Acceptance tests for tlog S7: the confirm decodes, the tranche manifests and the eval uses (#192).

Committed before ``tadabur.tlog_tranches`` exists and **locked**: later turns may add tests but
never weaken or rewrite these. The rules are ``docs/tlog-gating-amendment.md`` (owner-approved
2026-10-28; it REPLACES the frozen pre-registration's gating rule) and architect r24's decisions
in the brief. The tests cite the amendment by SHA-256 and recompute its numbers.

**Sealed discipline.** The confirm tranches are sealed: no assertion here puts a row, a label or
a decode into a failure message. Every comparison is between counts, booleans or row keys
``(split, shard_file, row)``, so a failing run prints aggregates only.

Contract this file pins
-----------------------
``tadabur.tlog_tranches`` exposes:

* ``ROLES = ("look", "confirm.1", ..., "confirm.4")``, ``CONFIRM_ROLES`` (the last four), the
  stratum names ``GATING = "gating"`` and ``STRATA`` (the six names in summary
  order, which is NOT the precedence: ``gating``, ``reported:unclean``, ``reported:undecided``,
  ``reported:straddles``, ``reported:identified``, ``reported:below_bar``; the precedence is
  ``stratum_of``'s branch order, below), ``CLAIM_FIELDS = ("role", "issue", "commit", "date")`` and
  ``TRANCHES_DIR = tools/tadabur/tlog/tranches``.
* ``stratum_of(record) -> str``: one stratum per ident row, the first matching of r24's table
  (unclean; ``label_source == none``; straddles; ``both``; ``identified``; ``filename``). An
  unknown ``label_source`` is a ``ValueError``.
* ``manifest_rows(ident_rows, group_keys) -> list[dict]``: one manifest row per ident row, sorted
  by ``(split, shard_file, row)``, with exactly :data:`MANIFEST_FIELDS` below (the ident row's
  values copied, ``group_key`` looked up by row key, ``stratum`` from :func:`stratum_of`). A row
  with no group key is refused (``KeyError`` or ``ValueError``).
* ``select(role, *, d_max, rows=None)``: the usable rows of one confirm role at or below
  ``d_max`` (``.selected``) and above it (``.over_d_max``, each with ``decode_mode =
  "not_decoded:over_d_max"`` and ``label_source = "none"``), both in row-key order. ``look`` is
  refused (S6's decodes are reused, never redone), as is an unknown role. ``rows`` replaces
  ``iter_pool_rows()``. ``require_d_max_share(selection)`` raises ``RuntimeError`` when strictly
  more than 1% of the role's usable rows are above ``d_max`` (checked per tranche).
* ``exposure_rows(rows) -> list[ClipExposure]``: the sorted, de-duplicated clip exposures of
  manifest rows (``dataset = tarteel-ai/tlog`` at the pinned revision, ``audio_filename =
  audio_path``, ``surah_ayah`` = ``assigned_label`` else ``"unknown"``, ``reciter = "unknown"``).
  One ``(audio_path, audio_sha256)`` with two different ``surah_ayah`` is a ``ValueError``.
  ``require_disjoint_uses(uses)`` takes ``{use: [ClipExposure]}`` and raises ``ValueError`` when one
  ``(audio_filename, audio_sha256)`` is in two uses (cross-tranche leakage).
* ``tranche_summary(rows) -> dict`` with ``rows``, ``entries`` (``len(exposure_rows(rows))``),
  ``strata`` (every stratum, zero-filled), ``gating``, ``gating_id_floor_ok`` (the amendment §5
  sensitivity line, ``both`` and the recitation floor passed) and ``gating_surahs`` (distinct
  surahs of the gating rows' ``assigned_label``: the bootstrap cluster count).
* ``require_claim_order(role, tranches_dir=TRANCHES_DIR)``: ``confirm.1`` needs no claim;
  ``confirm.k`` needs ``<tranches_dir>/confirm.(k-1).claim.json``, a JSON object with exactly
  ``CLAIM_FIELDS`` whose ``role`` is ``confirm.(k-1)``; ``look`` and unknown roles are refused;
  every refusal is a ``ValueError`` naming the missing predecessor. S7 itself writes no claim.

``tadabur.exposure`` gains ``TLOG_USES`` (``tlog.look``, ``tlog.confirm.1`` ... ``.4``): clip uses
in ``CLIP_USES`` and ``EXTERNAL_USES``, one ``exposure/<use>.clips.jsonl`` each.

Committed data, from the S6 layout: ``ident/decodes/confirm.k/{clean,unclean}.jsonl.gz`` and
``ident/confirm.k/{clean,unclean}.jsonl.gz`` with a ``meta.json`` in each directory;
``tranches/<role>.jsonl`` for all five roles and ``tranches/meta.json`` with ``schema_version``,
``model``, ``revision``, ``code_sha``, the SHAs ``preregistration_sha256``,
``gating_amendment_sha256``, ``calibration_sha256`` and ``decode_limit_sha256``, ``overlay``
(``split_code_sha``, ``overlap_code_sha``), ``slice_timing`` (``rows``, ``seconds``) and
``tranches`` (role -> ``tranche_summary``).

Test command, from ``tools/``::

    python -m pytest -q tadabur/test_tlog_tranches.py
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import re
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from tadabur import exposure
from tadabur import tlog_pool as pool
from tadabur.nonpro_sources import TLOG
from tadabur.tlog_ident import BASE_MODEL, BASE_REVISION, load_identifier
from tadabur.tlog_overlap import iter_pool_rows
from tadabur.tlog_pass1 import read_jsonl_gz

ROOT = Path(__file__).resolve().parents[2]
TLOG_DIR = ROOT / "tools" / "tadabur" / "tlog"
FROZEN = ROOT / "docs" / "tlog-holdout-preregistration.md"
FROZEN_SHA256 = "204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086"
GATING_DOC = ROOT / "docs" / "tlog-gating-amendment.md"
GATING_SHA256 = "23d6bf7abc9882909e9686d19b2bdcd8b19fb19ed49a38829c4db9e58a4ce085"
#: S6's committed look trees, reused verbatim by S7 (r24 item 6): never re-decoded.
LOOK_TREE_SHA256 = {
    "decodes/look/clean.jsonl.gz":
        "2195e1ee860d0fc8e308f25724de3acd2fec47f2cf3b25ddf25f3db9433934af",
    "decodes/look/unclean.jsonl.gz":
        "81ab90f3791d070b9be9037b233e525065b8b328757adcc6bbc57f9d2303071f",
    "look/clean.jsonl.gz":
        "50b4437a7cd45f0f5d72ae785b4efbe44892a8b333b14665bd31158d0c6f5c53",
    "look/unclean.jsonl.gz":
        "2d9003dcb9c4b7b30fc03869f2ddc42dfaf8fa0f5ac04a414a3bb6ddcabac2f2",
}
ROLES = ("look", "confirm.1", "confirm.2", "confirm.3", "confirm.4")
CONFIRM_ROLES = ROLES[1:]
# The summary order, not the precedence: precedence is `_expected_stratum`'s branch order.
STRATA = ("gating", "reported:unclean", "reported:undecided", "reported:straddles",
          "reported:identified", "reported:below_bar")
MANIFEST_FIELDS = frozenset({
    "split", "shard_file", "row", "audio_path", "audio_sha256", "group_key", "filename_label",
    "assigned_label", "label_source", "straddles", "decode_mode", "id_floor_ok", "stratum"})
IDENTIFICATION_KEYS = frozenset({
    "surah", "ayah_from", "word_from", "ayah_to", "word_to", "straddles", "score", "length",
    "class_label", "decoy", "exhaustive_decoy", "delta", "margin", "margin_class", "id_floor_ok",
    "overlaps_filename", "filename_label", "label_source", "assigned_label", "floor_rule",
    "decode_mode"})
PROVENANCE = ("split", "shard_file", "row", "audio_path", "audio_sha256", "duration_s")
#: Look's strata under the amendment, recomputed from S6's committed look tree (§1 gives 2,853).
LOOK_STRATA = {"gating": 2_853, "reported:unclean": 3_383, "reported:undecided": 0,
               "reported:straddles": 2_154, "reported:identified": 466,
               "reported:below_bar": 854}
TLOG_SHARDS = 743
REPLAY_ROWS = 100
CLAIM_COMMIT = "a" * 40


@pytest.fixture(scope="session")
def tranches():
    return importlib.import_module("tadabur.tlog_tranches")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key(row) -> tuple[str, str, int]:
    return row["split"], row["shard_file"], row["row"]


def _keyset(rows) -> set[tuple[str, str, int]]:
    return {_key(row) for row in rows}


def _manifest_path(role: str) -> Path:
    return TLOG_DIR / "tranches" / f"{role}.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_tree(root: Path, name: str) -> list[dict]:
    """One committed tree's rows, ``clean`` then ``unclean``."""
    return [row for split in ("clean", "unclean")
            for row in read_jsonl_gz(root / name / f"{split}.jsonl.gz")]


@lru_cache(maxsize=1)
def _pool() -> dict[str, dict[tuple[str, str, int], dict]]:
    """Each role's usable rows keyed by row key, with the frozen facts the trees must agree with."""
    rows: dict[str, dict] = {role: {} for role in ROLES}
    for row in iter_pool_rows():
        if row["role"] in rows and row["usable"]:
            rows[row["role"]][_key(row)] = {field: row[field] for field in (
                "audio_path", "audio_sha256", "duration_s", "filename_label")}
    return rows


@lru_cache(maxsize=1)
def _group_keys() -> dict[tuple[str, str, int], str]:
    keys = {}
    for path in sorted((TLOG_DIR / "split").glob("*/*.jsonl.gz")):
        for row in read_jsonl_gz(path):
            if row["role"] in ROLES:
                keys[_key(row)] = row["group_key"]
    return keys


def _expected_stratum(row) -> str:
    """r24's table, restated here so the committed manifests are checked against the rule itself."""
    if row["split"] == "unclean":
        return "reported:unclean"
    if row["label_source"] == "none":
        return "reported:undecided"
    if row["straddles"]:
        return "reported:straddles"
    return {"both": "gating", "identified": "reported:identified",
            "filename": "reported:below_bar"}[row["label_source"]]


def _ident(split="clean", label_source="both", straddles=False, **overrides) -> dict:
    """A synthetic ident-tree row (the key set of ``tlog_pool.with_keys``)."""
    row = {"split": split, "shard_file": f"{split}-00000-of-00411", "row": 1,
           "audio_path": "100_10_1.flac", "audio_sha256": _sha256(b"1"), "duration_s": 3.0,
           "filename_label": None if split == "unclean" else "100:10",
           "assigned_label": "100:10", "label_source": label_source, "straddles": straddles,
           "id_floor_ok": True, "decode_mode": pool.WHOLE_CLIP}
    row.update(overrides)
    return row


def _pool_row(role="confirm.1", row=1, **overrides) -> dict:
    """A synthetic ``iter_pool_rows`` row."""
    out = {"split": "clean", "shard_file": "clean-00000-of-00411", "row": row, "role": role,
           "usable": True, "duration_s": 5.0, "audio_path": f"100_10_{row}.flac",
           "audio_sha256": _sha256(str(row).encode()), "filename_label": "100:10"}
    out.update(overrides)
    return out


# --- the stratum rule (r24 item 2) -------------------------------------------------------------


@pytest.mark.parametrize(("record", "stratum"), [
    (dict(split="clean", label_source="both", straddles=False), "gating"),
    (dict(split="clean", label_source="both", straddles=True), "reported:straddles"),
    (dict(split="clean", label_source="identified", straddles=False), "reported:identified"),
    (dict(split="clean", label_source="identified", straddles=True), "reported:straddles"),
    (dict(split="clean", label_source="filename", straddles=False), "reported:below_bar"),
    (dict(split="clean", label_source="filename", straddles=True), "reported:straddles"),
    (dict(split="clean", label_source="none", straddles=None), "reported:undecided"),
    (dict(split="clean", label_source="none", straddles=True), "reported:undecided"),
    (dict(split="unclean", label_source="both", straddles=False), "reported:unclean"),
    (dict(split="unclean", label_source="none", straddles=None), "reported:unclean"),
    (dict(split="unclean", label_source="identified", straddles=True), "reported:unclean"),
])
def test_stratum_of_applies_the_first_matching_row_of_the_precedence_table(tranches, record,
                                                                          stratum):
    assert tranches.stratum_of(_ident(**record)) == stratum


def test_the_strata_are_the_six_names_in_summary_order(tranches):
    assert tuple(tranches.STRATA) == STRATA and tranches.GATING == "gating"
    assert tuple(tranches.ROLES) == ROLES and tuple(tranches.CONFIRM_ROLES) == CONFIRM_ROLES


def test_an_unknown_label_source_is_refused_not_given_a_stratum(tranches):
    with pytest.raises(ValueError):
        tranches.stratum_of(_ident(label_source="guess"))


def test_gating_ignores_the_floor_and_the_relabel_bar(tranches):
    """Amendment §1: a floor-failing ``both`` row gates; a non-``both`` row never does."""
    assert tranches.stratum_of(_ident(id_floor_ok=False)) == "gating"
    assert tranches.stratum_of(_ident(label_source="filename", id_floor_ok=True)) != "gating"


# --- manifest rows ------------------------------------------------------------------------------


def test_manifest_rows_carry_exactly_the_pinned_fields_in_row_key_order(tranches):
    later = _ident(row=9, audio_path="b.flac", audio_sha256=_sha256(b"9"))
    earlier = _ident(row=2, audio_path="a.flac", audio_sha256=_sha256(b"2"), id_floor_ok=False,
                     label_source="filename", assigned_label="100:10")
    keys = {_key(later): "g-later", _key(earlier): "g-earlier"}
    rows = tranches.manifest_rows([later, earlier], keys)
    assert [_key(row) for row in rows] == [_key(earlier), _key(later)]
    assert all(set(row) == MANIFEST_FIELDS for row in rows)
    assert [row["group_key"] for row in rows] == ["g-earlier", "g-later"]
    assert [row["stratum"] for row in rows] == ["reported:below_bar", "gating"]
    assert rows[0]["id_floor_ok"] is False and rows[0]["audio_sha256"] == earlier["audio_sha256"]


def test_a_row_without_a_group_key_is_refused(tranches):
    with pytest.raises((KeyError, ValueError)):
        tranches.manifest_rows([_ident()], {})


def test_the_summary_counts_every_stratum_and_the_sensitivity_line(tranches):
    rows = tranches.manifest_rows([
        _ident(row=1, audio_path="a.flac", assigned_label="1:1"),
        _ident(row=2, audio_path="b.flac", assigned_label="1:2", id_floor_ok=False),
        _ident(row=3, audio_path="c.flac", assigned_label="2:5"),
        _ident(row=4, audio_path="d.flac", straddles=True),
        _ident(row=5, audio_path="e.flac", label_source="none", straddles=None,
               assigned_label=None, id_floor_ok=None, decode_mode=pool.NOT_DECODED),
        _ident(row=6, audio_path="f.flac", split="unclean", filename_label=None,
               assigned_label=None, label_source="none"),
    ], {(r["split"], r["shard_file"], r["row"]): f"g{r['row']}" for r in (
        _ident(row=n, split="unclean" if n == 6 else "clean") for n in range(1, 7))})
    summary = tranches.tranche_summary(rows)
    assert set(summary) == {"rows", "entries", "strata", "gating", "gating_id_floor_ok",
                            "gating_surahs"}
    assert summary["rows"] == 6 and summary["entries"] == 6
    assert summary["strata"] == {"gating": 3, "reported:unclean": 1, "reported:undecided": 1,
                                 "reported:straddles": 1, "reported:identified": 0,
                                 "reported:below_bar": 0}
    assert summary["gating"] == 3 and summary["gating_id_floor_ok"] == 2
    assert summary["gating_surahs"] == 2


# --- exposure rows and uses (r24 item 3) --------------------------------------------------------


def _manifest(tranches, *records) -> list[dict]:
    keys = {_key(r): f"g{r['row']}" for r in records}
    return tranches.manifest_rows(list(records), keys)


def test_exposure_rows_list_every_row_with_the_pinned_dataset_fields(tranches):
    rows = _manifest(
        tranches,
        _ident(row=1, audio_path="a.flac", audio_sha256=_sha256(b"a"), assigned_label="3:7"),
        _ident(row=2, audio_path="b.flac", audio_sha256=_sha256(b"b"), label_source="none",
               straddles=None, assigned_label=None, id_floor_ok=None,
               decode_mode=pool.NOT_DECODED),
        _ident(row=3, audio_path="c.flac", audio_sha256=_sha256(b"c"), split="unclean",
               filename_label=None, assigned_label=None, label_source="none"))
    uses = tranches.exposure_rows(rows)
    assert uses == sorted(uses) and len(uses) == 3
    assert {(u.dataset, u.revision, u.reciter) for u in uses} == {
        ("tarteel-ai/tlog", TLOG.revision, "unknown")}
    assert {u.audio_filename: u.surah_ayah for u in uses} == {
        "a.flac": "3:7", "b.flac": "unknown", "c.flac": "unknown"}
    assert {u.audio_filename: u.audio_sha256 for u in uses}["a.flac"] == _sha256(b"a")


def test_rows_sharing_a_path_and_checksum_collapse_to_one_entry(tranches):
    shared = {"audio_path": "s.flac", "audio_sha256": _sha256(b"s")}
    rows = _manifest(tranches, _ident(row=1, **shared), _ident(row=2, **shared),
                     _ident(row=3, audio_path="s.flac", audio_sha256=_sha256(b"other")))
    assert len(rows) == 3 and len(tranches.exposure_rows(rows)) == 2
    assert tranches.tranche_summary(rows)["entries"] == 2


def test_one_clip_with_two_ayahs_in_a_use_fails_loudly(tranches):
    shared = {"audio_path": "s.flac", "audio_sha256": _sha256(b"s")}
    rows = _manifest(tranches, _ident(row=1, assigned_label="1:1", **shared),
                     _ident(row=2, assigned_label="1:2", **shared))
    with pytest.raises(ValueError):
        tranches.exposure_rows(rows)


def _clip(name="a.flac", sha="a", ayah="1:1") -> exposure.ClipExposure:
    return exposure.ClipExposure("tarteel-ai/tlog", TLOG.revision, name, ayah, _sha256(sha.encode()),
                                 "unknown")


def test_a_clip_in_two_tranche_uses_is_cross_tranche_leakage(tranches):
    tranches.require_disjoint_uses({"tlog.confirm.1": [_clip("a.flac")],
                                    "tlog.confirm.2": [_clip("b.flac")]})
    with pytest.raises(ValueError):
        tranches.require_disjoint_uses({"tlog.confirm.1": [_clip("a.flac")],
                                        "tlog.confirm.2": [_clip("a.flac")]})
    # the same name with another checksum is another recording, so it is not leakage
    tranches.require_disjoint_uses({"tlog.confirm.1": [_clip("a.flac", "x")],
                                    "tlog.confirm.2": [_clip("a.flac", "y")]})


def test_the_five_tlog_uses_are_declared_clip_uses():
    uses = {f"tlog.{role}" for role in ROLES}
    assert exposure.TLOG_USES == uses
    assert uses <= exposure.CLIP_USES and uses <= exposure.EXTERNAL_USES
    assert exposure.NONPRO_EVALUATION in exposure.CLIP_USES


@pytest.mark.parametrize("use", [f"tlog.{role}" for role in ROLES])
def test_each_tlog_use_is_refused_by_check_disjoint_like_the_nonpro_use(use, tmp_path):
    exposure.write_clip_use(use, [_clip()], tmp_path)
    exposure.write_use(exposure.LABEL_WAQF_EVENTS,
                       [exposure.Exposure("w.wav", 1, 1, 1, None, None, None)], tmp_path)
    registry = exposure.load_registry(tmp_path)
    assert [c.audio_filename for c in registry.clips[use]] == ["a.flac"]
    with pytest.raises(ValueError, match="evaluation only"):
        exposure.check_disjoint(exposure.LABEL_WAQF_EVENTS, use, registry=registry)
    with pytest.raises(ValueError, match="evaluation only"):
        exposure.check_disjoint(use, exposure.LABEL_WAQF_EVENTS, registry=registry)


def test_every_tlog_use_is_missing_until_its_file_is_written(tmp_path):
    exposure.write_use(exposure.LABEL_WAQF_EVENTS,
                       [exposure.Exposure("w.wav", 1, 1, 1, None, None, None)], tmp_path)
    assert exposure.TLOG_USES <= set(exposure.load_registry(tmp_path).missing())


# --- selection, the D_max stop and the decoder (r24 items 5-6) ----------------------------------


def test_select_takes_the_usable_rows_of_one_confirm_role_up_to_d_max(tranches):
    rows = [_pool_row(row=3), _pool_row(row=1), _pool_row(row=2, duration_s=95.0),
            _pool_row(row=4, usable=False), _pool_row(row=5, role="confirm.2"),
            _pool_row(row=6, role="look"), _pool_row(row=7, role="train"),
            _pool_row(row=8, duration_s=90.0)]
    selection = tranches.select("confirm.1", d_max=90.0, rows=rows)
    assert [r["row"] for r in selection.selected] == [1, 3, 8]
    assert [r["row"] for r in selection.over_d_max] == [2]
    assert selection.over_d_max[0]["decode_mode"] == pool.NOT_DECODED
    assert selection.over_d_max[0]["label_source"] == "none"
    assert {r["role"] for r in (*selection.selected, *selection.over_d_max)} == {"confirm.1"}


@pytest.mark.parametrize("role", ["look", "train", "confirm.5", "confirm"])
def test_select_refuses_look_train_and_unknown_roles(tranches, role):
    with pytest.raises(ValueError):
        tranches.select(role, d_max=90.0, rows=[_pool_row()])


def test_the_d_max_stop_is_checked_per_tranche_and_is_strict(tranches):
    def selection(over: int, total: int = 100):
        rows = [_pool_row(row=n, duration_s=100.0 if n < over else 5.0) for n in range(total)]
        return tranches.select("confirm.2", d_max=90.0, rows=[{**r, "role": "confirm.2"}
                                                              for r in rows])

    tranches.require_d_max_share(selection(0))
    tranches.require_d_max_share(selection(1))  # exactly 1%: not above the stop
    with pytest.raises(RuntimeError):
        tranches.require_d_max_share(selection(2))


def test_the_module_reaches_the_teacher_only_through_the_pool_and_no_student_model():
    """The code check for "no student, C95 or candidate model decodes any confirm row"."""
    source = (ROOT / "tools" / "tadabur" / "tlog_tranches.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            imported |= {base, *(f"{base}.{alias.name}".lstrip(".") for alias in node.names)}
    parts = {part for name in imported for part in name.split(".") if part}
    forbidden = {"training", "inference", "torch", "transformers", "tlog_ident_decode",
                 "muaalem", "muaalem_audio", "eval_harness", "decode_evalset"}
    assert not parts & forbidden
    assert "tlog_pool" in parts
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {"load_decoder", "Decoder", "AutoModel", "from_pretrained"}
    assert "decoder" in names  # it does decode, through tlog_pool.decoder()


def test_no_reader_globs_the_confirm_prefix_so_s5bs_confirm_directory_stays_apart():
    """``ident/confirm/`` (S5b's Common Voice and Freesound) is not a tranche tree."""
    pattern = re.compile(r"glob\(\s*f?[\"']confirm")
    offenders = [path.name for path in (ROOT / "tools" / "tadabur").glob("tlog_*.py")
                 if pattern.search(path.read_text(encoding="utf-8"))]
    assert offenders == []
    assert (pool.IDENT_DIR / "confirm").is_dir()
    with pytest.raises(FileNotFoundError):
        pool.read_ident_tree("confirm")  # a population name is exact, never a prefix


# --- the claim order (r24 item 4) ---------------------------------------------------------------


def _claim(tmp_path: Path, role: str, **overrides) -> Path:
    record = {"role": role, "issue": 195, "commit": CLAIM_COMMIT, "date": "2026-11-02"}
    record.update(overrides)
    path = tmp_path / f"{role}.claim.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_the_claim_schema_and_tranche_directory_are_pinned(tranches):
    assert tuple(tranches.CLAIM_FIELDS) == ("role", "issue", "commit", "date")
    assert tranches.TRANCHES_DIR == TLOG_DIR / "tranches"


def test_confirm_one_needs_no_claim(tranches, tmp_path):
    tranches.require_claim_order("confirm.1", tmp_path)


def test_a_tranche_cannot_be_claimed_before_its_predecessor(tranches, tmp_path):
    for role in ("confirm.2", "confirm.3", "confirm.4"):
        with pytest.raises(ValueError, match=r"confirm\.\d"):
            tranches.require_claim_order(role, tmp_path)
    _claim(tmp_path, "confirm.1")
    tranches.require_claim_order("confirm.2", tmp_path)


def test_a_skipped_tranche_blocks_the_ones_after_it(tranches, tmp_path):
    _claim(tmp_path, "confirm.1")
    with pytest.raises(ValueError, match=r"confirm\.2"):
        tranches.require_claim_order("confirm.3", tmp_path)
    with pytest.raises(ValueError):
        tranches.require_claim_order("confirm.4", tmp_path)
    tranches.require_claim_order("confirm.2", tmp_path)


def test_one_round_may_claim_two_consecutive_tranches_in_order(tranches, tmp_path):
    tranches.require_claim_order("confirm.1", tmp_path)
    _claim(tmp_path, "confirm.1")
    tranches.require_claim_order("confirm.2", tmp_path)
    _claim(tmp_path, "confirm.2")
    tranches.require_claim_order("confirm.3", tmp_path)
    with pytest.raises(ValueError):
        tranches.require_claim_order("confirm.4", tmp_path)


def test_a_predecessor_claim_must_follow_the_pinned_schema(tranches, tmp_path):
    path = _claim(tmp_path, "confirm.1")
    path.write_text(json.dumps({"role": "confirm.1", "issue": 195}), encoding="utf-8")
    with pytest.raises(ValueError):
        tranches.require_claim_order("confirm.2", tmp_path)
    _claim(tmp_path, "confirm.2")  # a claim file in another's name does not count
    path.write_text(json.dumps({"role": "confirm.2", "issue": 195, "commit": CLAIM_COMMIT,
                                "date": "2026-11-02"}), encoding="utf-8")
    with pytest.raises(ValueError):
        tranches.require_claim_order("confirm.2", tmp_path)


@pytest.mark.parametrize("role", ["look", "confirm.5", "train"])
def test_look_and_unknown_roles_are_not_claimable(tranches, tmp_path, role):
    with pytest.raises(ValueError):
        tranches.require_claim_order(role, tmp_path)


def test_no_claim_file_is_committed_for_any_tranche():
    assert sorted(path.name for path in (TLOG_DIR / "tranches").glob("*.claim.json")) == []


# --- the committed confirm trees (r24 item 5; issue item 1) -------------------------------------


@pytest.mark.parametrize("role", CONFIRM_ROLES)
def test_the_confirm_trees_follow_the_s6_layout(role):
    for root in (pool.DECODES_DIR, pool.IDENT_DIR):
        assert (root / role / "meta.json").is_file()
        for split in ("clean", "unclean"):
            assert (root / role / f"{split}.jsonl.gz").is_file()


@pytest.mark.parametrize("role", CONFIRM_ROLES)
def test_the_confirm_trees_cover_exactly_the_role(role):
    d_max = pool.decode_limit()["d_max_seconds"]
    usable = _pool()[role]
    decoded = _keyset(_read_tree(pool.DECODES_DIR, role))
    identified = _keyset(_read_tree(pool.IDENT_DIR, role))
    within = {key for key, row in usable.items() if row["duration_s"] <= d_max}
    assert len(usable) > 9_000 and len(within) < len(usable)
    assert decoded == within
    assert identified == set(usable)
    assert (len(usable) - len(within)) / len(usable) <= 0.01  # the per-tranche D_max stop


@pytest.mark.parametrize("role", CONFIRM_ROLES)
def test_the_confirm_rows_are_whole_clip_or_not_decoded_and_agree_with_the_frozen_rows(role):
    d_max = pool.decode_limit()["d_max_seconds"]
    usable = _pool()[role]
    decodes = _read_tree(pool.DECODES_DIR, role)
    idents = _read_tree(pool.IDENT_DIR, role)
    assert [_key(r) for r in decodes] == sorted(_key(r) for r in decodes)
    assert [_key(r) for r in idents] == sorted(_key(r) for r in idents)
    assert {r["decode_mode"] for r in decodes} == {pool.WHOLE_CLIP}
    for rows in (decodes, idents):
        stale = sum(1 for r in rows if (r["audio_sha256"], r["audio_path"], r["duration_s"],
                                        r["filename_label"]) != (
            usable[_key(r)]["audio_sha256"], usable[_key(r)]["audio_path"],
            usable[_key(r)]["duration_s"], usable[_key(r)]["filename_label"]))
        assert stale == 0
    unmarked = sum(1 for r in idents
                   if (r["decode_mode"] == pool.NOT_DECODED) != (r["duration_s"] > d_max))
    keyless = sum(1 for r in idents if not IDENTIFICATION_KEYS | set(PROVENANCE) <= set(r))
    undecided = sum(1 for r in idents if r["decode_mode"] == pool.NOT_DECODED
                    and (r["label_source"] != "none" or r["id_floor_ok"] is not None))
    assert (unmarked, keyless, undecided) == (0, 0, 0)


@pytest.mark.parametrize("role", CONFIRM_ROLES)
def test_the_confirm_identifications_replay_from_the_decodes_with_s5s_engine(role):
    """A spread sample of the committed decodes, identified again, must equal the committed rows."""
    database, bars, k = load_identifier()
    decodes = _read_tree(pool.DECODES_DIR, role)
    idents = {_key(r): r for r in _read_tree(pool.IDENT_DIR, role)}
    sample = decodes[::max(1, len(decodes) // REPLAY_ROWS)]
    assert len(sample) >= REPLAY_ROWS
    replayed = pool.identify_rows(sample, database=database, bars=bars, k=k)
    assert sum(1 for row in replayed if row != idents[_key(row)]) == 0


@pytest.mark.parametrize("role", CONFIRM_ROLES)
def test_each_confirm_tree_meta_records_the_base_teacher_and_its_own_census(role):
    pins = {"calibration_sha256": _sha256((pool.IDENT_DIR / "ident_calibration.json").read_bytes()),
            "decode_limit_sha256": _sha256(pool.DECODE_LIMIT_PATH.read_bytes())}
    for root in (pool.DECODES_DIR, pool.IDENT_DIR):
        meta = json.loads((root / role / "meta.json").read_text(encoding="utf-8"))
        rows = _read_tree(root, role)
        assert (meta["model"], meta["revision"]) == (BASE_MODEL, BASE_REVISION)
        assert {name: meta[name] for name in pins} == pins
        assert re.fullmatch(r"[0-9a-f]{40}", meta["code_sha"])
        assert meta["rows"] == {split: sum(1 for r in rows if r["split"] == split)
                                for split in ("clean", "unclean")}
        assert len(meta["census"]) == TLOG_SHARDS and sum(meta["census"].values()) == len(rows)


def test_s6s_look_trees_are_reused_unchanged():
    for name, digest in LOOK_TREE_SHA256.items():
        assert _sha256((pool.IDENT_DIR / name).read_bytes()) == digest


# --- the committed manifests (acceptance: five manifests, gating reproducible) ------------------


def test_the_gating_amendment_is_the_pinned_rule_these_tests_apply():
    text = GATING_DOC.read_text(encoding="utf-8")
    assert _sha256(GATING_DOC.read_bytes()) == GATING_SHA256
    assert "`label_source == both`" in text and "2,853" in text
    assert _sha256(FROZEN.read_bytes()) == FROZEN_SHA256 and FROZEN_SHA256 in text


@pytest.mark.parametrize("role", ROLES)
def test_a_manifest_is_the_identification_tree_under_the_amendments_rule(tranches, role):
    """Re-applies amendment §1 and r24's precedence to the committed ident tree of the role."""
    idents = pool.read_ident_tree(role)
    manifest = _read_jsonl(_manifest_path(role))
    assert _keyset(manifest) == set(_pool()[role]) == _keyset(idents)
    assert [_key(r) for r in manifest] == sorted(_key(r) for r in manifest)
    assert all(set(r) == MANIFEST_FIELDS for r in manifest)
    by_key = {_key(r): r for r in idents}
    group_keys = _group_keys()
    wrong = Counter()
    for row in manifest:
        source = by_key[_key(row)]
        wrong["stratum"] += row["stratum"] != _expected_stratum(source)
        wrong["group_key"] += row["group_key"] != group_keys[_key(row)]
        wrong["copied"] += any(row[f] != source[f] for f in MANIFEST_FIELDS
                               if f not in ("group_key", "stratum"))
        wrong["gating"] += (row["stratum"] == "gating") != (
            row["split"] == "clean" and source["filename_label"] is not None
            and not source["straddles"] and source["label_source"] == "both")
    assert sum(wrong.values()) == 0
    rebuilt = tranches.manifest_rows(idents, group_keys)
    assert len(rebuilt) == len(manifest)
    assert sum(1 for mine, theirs in zip(rebuilt, manifest, strict=True) if mine != theirs) == 0


def test_look_gating_reproduces_the_amendments_2853():
    counts = Counter(row["stratum"] for row in _read_jsonl(_manifest_path("look")))
    assert {name: counts.get(name, 0) for name in STRATA} == LOOK_STRATA
    assert sum(counts.values()) == 9_710


def test_gating_rows_are_clean_labelled_non_straddling_and_both():
    for role in ROLES:
        gating = [r for r in _read_jsonl(_manifest_path(role)) if r["stratum"] == "gating"]
        assert len(gating) > 2_000
        bad = sum(1 for r in gating if r["split"] != "clean" or r["label_source"] != "both"
                  or r["straddles"] is not False or r["filename_label"] is None
                  or r["assigned_label"] != r["filename_label"])
        assert bad == 0


def test_the_tranche_meta_records_counts_pins_and_the_slice_timing(tranches):
    meta = json.loads((TLOG_DIR / "tranches" / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert (meta["model"], meta["revision"]) == (BASE_MODEL, BASE_REVISION)
    assert re.fullmatch(r"[0-9a-f]{40}", meta["code_sha"])
    assert meta["preregistration_sha256"] == FROZEN_SHA256
    assert meta["gating_amendment_sha256"] == GATING_SHA256
    assert meta["calibration_sha256"] == _sha256(
        (pool.IDENT_DIR / "ident_calibration.json").read_bytes())
    assert meta["decode_limit_sha256"] == _sha256(pool.DECODE_LIMIT_PATH.read_bytes())
    overlay = {"split_code_sha": json.loads((TLOG_DIR / "split" / "meta.json").read_text())["code_sha"],
               "overlap_code_sha": json.loads(
                   (TLOG_DIR / "overlap" / "meta.json").read_text())["code_sha"]}
    assert meta["overlay"] == overlay
    assert 400 <= meta["slice_timing"]["rows"] <= 600 and meta["slice_timing"]["seconds"] > 0
    assert set(meta["tranches"]) == set(ROLES)
    for role in ROLES:
        rows = _read_jsonl(_manifest_path(role))
        assert meta["tranches"][role] == tranches.tranche_summary(rows)
        assert meta["tranches"][role]["rows"] == len(_pool()[role])
        gating = [r for r in rows if r["stratum"] == "gating"]
        assert meta["tranches"][role]["gating_surahs"] == len(
            {r["assigned_label"].split(":")[0] for r in gating})
    assert meta["tranches"]["look"]["gating"] == 2_853


def test_the_manifests_are_not_gitignored():
    paths = [f"tools/tadabur/tlog/tranches/{role}.jsonl" for role in ROLES]
    for path in paths:
        ignored = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode
        assert ignored == 1, f"{path} is gitignored; add its `!` entry"


# --- the committed exposure uses ----------------------------------------------------------------


def test_the_five_exposure_uses_are_committed_and_loaded(tranches):
    registry = exposure.load_registry()
    assert exposure.TLOG_USES <= set(registry.clips)
    for role in ROLES:
        use = f"tlog.{role}"
        expected = set(tranches.exposure_rows(_read_jsonl(_manifest_path(role))))
        committed = registry.clips[use]
        assert len(committed) == len(expected)
        differing = len(set(committed) ^ expected)  # a count: a failure prints no clip
        assert differing == 0
        meta_entries = json.loads(
            (TLOG_DIR / "tranches" / "meta.json").read_text())["tranches"][role]["entries"]
        assert meta_entries == len(committed)
        assert {(c.dataset, c.revision, c.reciter) for c in committed} == {
            ("tarteel-ai/tlog", TLOG.revision, "unknown")}


def test_no_clip_has_two_ayahs_in_a_use_or_sits_in_two_tranches(tranches):
    registry = exposure.load_registry()
    uses = {f"tlog.{role}": registry.clips[f"tlog.{role}"] for role in ROLES}
    for use, clips in uses.items():
        ayahs: dict[tuple[str, str], set[str]] = {}
        for clip in clips:
            ayahs.setdefault((clip.audio_filename, clip.audio_sha256), set()).add(clip.surah_ayah)
        assert sum(1 for seen in ayahs.values() if len(seen) > 1) == 0, use
    tranches.require_disjoint_uses(uses)
    seen = Counter(key for clips in uses.values()
                   for key in {(c.audio_filename, c.audio_sha256) for c in clips})
    assert sum(1 for count in seen.values() if count > 1) == 0

"""The round-2 confirm decode (#230, P3): ``tlog_eval_decode`` reads confirm.1 and confirm.2.

``read_manifest`` stays look-only (its tests are locked). The new interface:

* ``read_confirm_manifest(role, *, tranches_dir=TRANCHES_DIR, layout=None, spec=CONSONANT2)``
  returns the committed manifest rows of ``confirm.1`` or ``confirm.2`` (every stratum; the
  caller takes :func:`gating_rows`). It raises before it opens a file for any other role
  (``ValueError``); for a role whose claim is not committed in HEAD or whose predecessor's is
  not (``ValueError``); and, claim or not, when ``choice.json`` of the round's ``layout`` (default
  the spec's root) is not committed (``SystemExit``, "not committed"). A row of the role that is
  also in any other tranche's manifest (look, or another confirm) is a planted row and refused
  (``ValueError`` naming the other tranche), so a row of an unclaimed tranche is never decoded.
* ``decodes_path(out_dir, shard_file, model, role="look")``: the look's name is unchanged
  (``tlog.look.<shard>.<model>.jsonl``); a confirm role's is ``tlog.<role>.<shard>.<model>.jsonl``.
* ``decode_shards(..., role="look")`` and ``cache_selection(..., role="look")``: with a confirm
  role they read ``read_confirm_manifest(role)``'s **gating** rows only, never ``read_manifest``.
* ``merge_confirm_decodes(parts_dir, out_dir, *, manifests, provenance, models=...)``: the pooled
  confirm file. ``manifests`` is ``{"confirm.1": rows, "confirm.2": rows}`` (exactly those two,
  from :func:`read_confirm_manifest`); every model must hold each tranche's gating row exactly
  once (``ValueError`` naming the model otherwise) and each line is written with the ``tranche``
  it came from. One gzipped file per model, ``tlog.confirm.<model>.jsonl.gz``, sorted by key,
  and ``meta.confirm.json`` (never ``meta.json``, which is the look's).

Every git state is real; nothing decodes.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tadabur import tlog_eval_decode as ted
from tadabur.test_tlog_eval_decode import _manifest_row, _sha256, _stream, _write_jsonl
from training import round2_fixtures as fx
from training import round2_verdict_fixtures as rv

SHARDS = {"look": "clean-00001-of-00411", "confirm.1": "clean-00101-of-00411",
          "confirm.2": "clean-00102-of-00411", "confirm.3": "clean-00103-of-00411",
          "confirm.4": "clean-00104-of-00411"}
ROLES = tuple(SHARDS)


def _rows(role: str) -> list[dict]:
    """Three gating rows and one reported row of ``role``'s own shard."""
    shard = SHARDS[role]
    return [*(_manifest_row(n, shard_file=shard) for n in (1, 2, 3)),
            _manifest_row(4, shard_file=shard, stratum="reported:unclean")]


def _manifests(round_: rv.Round, rows: dict[str, list[dict]] | None = None) -> None:
    rows = rows or {}
    paths = []
    for role in ROLES:
        _write_jsonl(round_.tranches / f"{role}.jsonl", rows.get(role, _rows(role)))
        paths.append(round_.tranches / f"{role}.jsonl")
    round_.commit(*paths, message="the tranche manifests")


def _claim(round_: rv.Round, role: str, *, commit: bool = True) -> Path:
    path = round_.tranches / f"{role}.claim.json"
    path.write_text(json.dumps({"role": role, "issue": 227, "commit": rv.CLAIM_COMMIT,
                                "date": "2026-11-02"}), encoding="utf-8")
    if commit:
        round_.commit(path, message=f"claim {role}")
    return path


def _read(round_: rv.Round, role: str, layout=None):
    return ted.read_confirm_manifest(role, tranches_dir=round_.tranches,
                                     layout=layout or round_.layout, spec=round_.spec)


@pytest.fixture
def claimed(tmp_path) -> rv.Round:
    """Both claims committed, the choice committed, every manifest committed."""
    round_ = rv.committed_round(tmp_path)
    _manifests(round_)
    _claim(round_, "confirm.1")
    _claim(round_, "confirm.2")
    return round_


# --- the gates ------------------------------------------------------------------------------------


def test_a_claimed_tranche_is_read_whole_once_the_choice_is_committed(claimed):
    assert _read(claimed, "confirm.1") == _rows("confirm.1")
    assert _read(claimed, "confirm.2") == _rows("confirm.2")


@pytest.mark.parametrize("role", ["confirm.3", "confirm.4", "look", "train", "confirm", ""])
def test_no_other_role_is_read_even_when_everything_is_committed(claimed, role):
    with pytest.raises(ValueError):
        _read(claimed, role)


def test_a_role_is_refused_before_any_file_is_opened(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        ted.read_confirm_manifest("confirm.3", tranches_dir=empty)


def test_an_unclaimed_role_is_refused(tmp_path):
    round_ = rv.committed_round(tmp_path)
    _manifests(round_)
    with pytest.raises(ValueError, match="claim"):
        _read(round_, "confirm.1")


def test_a_claim_that_is_written_but_not_committed_opens_nothing(tmp_path):
    round_ = rv.committed_round(tmp_path)
    _manifests(round_)
    _claim(round_, "confirm.1", commit=False)
    with pytest.raises(ValueError, match="claim"):
        _read(round_, "confirm.1")


def test_confirm_2_is_refused_while_claim_1_is_not_committed(tmp_path):
    round_ = rv.committed_round(tmp_path)
    _manifests(round_)
    _claim(round_, "confirm.1", commit=False)
    _claim(round_, "confirm.2")
    with pytest.raises(ValueError, match="confirm.1"):
        _read(round_, "confirm.2")


def test_a_committed_claim_is_not_enough_without_the_committed_choice(claimed):
    nowhere = rv.ag.Layout(claimed.repo / "no-choice-here")
    nowhere.root.mkdir()
    with pytest.raises(SystemExit, match="not committed"):
        _read(claimed, "confirm.1", layout=nowhere)


def test_a_choice_edited_since_its_commit_opens_nothing(claimed):
    claimed.layout.choice.write_text('{"edited": true}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="not committed"):
        _read(claimed, "confirm.1")


def test_no_claim_and_no_choice_is_refused(tmp_path):
    repo = fx.init_repo(tmp_path / "bare")
    (repo / "tranches").mkdir()
    _write_jsonl(repo / "tranches" / "confirm.1.jsonl", _rows("confirm.1"))
    with pytest.raises((ValueError, SystemExit)):
        ted.read_confirm_manifest("confirm.1", tranches_dir=repo / "tranches",
                                  layout=rv.ag.Layout(repo / "round"))


@pytest.mark.parametrize("other", ["look", "confirm.2", "confirm.3", "confirm.4"])
def test_a_row_planted_from_any_other_tranche_is_refused(tmp_path, other):
    round_ = rv.committed_round(tmp_path)
    planted = _rows(other)[0]
    own = [*_rows("confirm.1")[:3], dict(planted, group_key="other")]
    _manifests(round_, {"confirm.1": own})
    _claim(round_, "confirm.1")
    _claim(round_, "confirm.2")
    with pytest.raises(ValueError, match=other.replace(".", r"\.")):
        _read(round_, "confirm.1")


# --- one tranche's parts and clips ----------------------------------------------------------------


def test_a_confirm_part_is_named_for_its_tranche_and_the_look_s_is_unchanged(tmp_path):
    assert ted.decodes_path(tmp_path, "s", "base") == tmp_path / "tlog.look.s.base.jsonl"
    assert ted.decodes_path(tmp_path, "s", "base", role="confirm.2") == (
        tmp_path / "tlog.confirm.2.s.base.jsonl")


def test_the_decode_reads_the_confirm_gating_rows_and_never_the_look_manifest(
        tmp_path, monkeypatch):
    seen = {}

    def read_confirm(role, **kwargs):
        seen["role"] = role
        return _rows("confirm.1")

    monkeypatch.setattr(ted, "read_confirm_manifest", read_confirm)
    monkeypatch.setattr(ted, "read_manifest",
                        lambda *args, **kwargs: pytest.fail("the look manifest was read"))
    monkeypatch.setattr(ted, "load_models", lambda parts_dir, models: {})
    monkeypatch.setattr(ted, "ayah_word_counts", lambda: lambda label: 3)

    def decode(shards, by_shard, **kwargs):
        seen["shards"] = list(shards)
        seen["rows"] = {name: [row["row"] for row in rows] for name, rows in by_shard.items()}
        return []

    monkeypatch.setattr(ted, "_decode_shards", decode)
    ted.decode_shards(parts_dir=tmp_path / "parts", cache_dir=tmp_path / "cache",
                      models=("base", "c95"), role="confirm.1")
    assert seen["role"] == "confirm.1"
    assert seen["shards"] == [SHARDS["confirm.1"]]
    assert seen["rows"] == {SHARDS["confirm.1"]: [1, 2, 3]}  # the reported row is not decoded


def test_the_clip_cache_holds_a_confirm_tranches_gating_clips_only(tmp_path, monkeypatch):
    monkeypatch.setattr(ted, "read_confirm_manifest", lambda role, **kwargs: _rows(role))
    monkeypatch.setattr(ted, "read_manifest",
                        lambda *args, **kwargs: pytest.fail("the look manifest was read"))
    payloads = {0: b"unwanted", **{n: f"clip-{n}".encode() for n in (1, 2, 3, 4)}}
    cache = tmp_path / "clips"
    result = ted.cache_selection(cache, tmp_path / "shards", role="confirm.2",
                                 stream=_stream(payloads))
    assert result["rows"] == 3 and result["written"] == 3
    assert sorted(path.name for path in cache.iterdir()) == sorted(
        f"{_sha256(payloads[n])}.audio" for n in (1, 2, 3))


# --- the pooled merge -----------------------------------------------------------------------------


MODELS = ("base", "c95")


def _lines(rows, model: str, *, drop=(), repeat=False, extra=False) -> list[dict]:
    lines = [{"key": f"{row['audio_path']}#0", "split": row["split"],
              "shard_file": row["shard_file"], "row": row["row"], "checks": [model]}
             for row in rows if row["stratum"] == "gating" and row["row"] not in drop]
    if repeat:
        lines.append(lines[0])
    if extra:
        lines.append({"key": "not-in-a-manifest.flac#0", "checks": []})
    return lines


def _parts(tmp_path: Path, manifests, faults=None) -> Path:
    faults = faults or {}
    parts = tmp_path / "parts"
    for role, rows in manifests.items():
        for model in MODELS:
            fault = faults.get((role, model), {})
            _write_jsonl(ted.decodes_path(parts, SHARDS[role], model, role=role),
                         _lines(rows, model, **fault))
    return parts


def _manifest_rows() -> dict[str, list[dict]]:
    # Distinct clips across the tranches: ``audio_path`` embeds the row number.
    return {"confirm.1": [_manifest_row(n, shard_file=SHARDS["confirm.1"]) for n in (3, 1)],
            "confirm.2": [_manifest_row(n, shard_file=SHARDS["confirm.2"]) for n in (12, 11)]}


def test_the_merge_pools_both_tranches_into_one_sorted_file_per_model(tmp_path):
    manifests = _manifest_rows()
    out = tmp_path / "decodes"
    counts = ted.merge_confirm_decodes(_parts(tmp_path, manifests), out, manifests=manifests,
                                       provenance={"code_sha": "c" * 40}, models=MODELS)
    assert counts == {"base": 4, "c95": 4}
    for model in MODELS:
        with gzip.open(out / f"tlog.confirm.{model}.jsonl.gz", "rt", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle]
        assert [line["key"] for line in lines] == sorted(line["key"] for line in lines)
        assert {line["tranche"] for line in lines} == {"confirm.1", "confirm.2"}
        by_tranche = {role: {line["key"] for line in lines if line["tranche"] == role}
                      for role in manifests}
        assert by_tranche == {role: {f"{row['audio_path']}#0" for row in rows}
                              for role, rows in manifests.items()}
    meta = json.loads((out / "meta.confirm.json").read_text(encoding="utf-8"))
    assert meta["code_sha"] == "c" * 40 and meta["models"] == list(MODELS)
    assert meta["rows"] == {"base": 4, "c95": 4}
    assert not (out / "meta.json").exists()


@pytest.mark.parametrize("role, fault", [
    ("confirm.1", {"drop": (1,)}), ("confirm.1", {"repeat": True}), ("confirm.1", {"extra": True}),
    ("confirm.2", {"drop": (11,)}), ("confirm.2", {"repeat": True}),
    ("confirm.2", {"extra": True})])
def test_the_merge_refuses_a_model_that_misses_repeats_or_adds_a_row(tmp_path, role, fault):
    manifests = _manifest_rows()
    parts = _parts(tmp_path, manifests, {(role, "c95"): fault})
    with pytest.raises(ValueError, match="c95"):
        ted.merge_confirm_decodes(parts, tmp_path / "decodes", manifests=manifests,
                                  provenance={}, models=MODELS)
    assert not (tmp_path / "decodes").exists() or not list((tmp_path / "decodes").iterdir())


def test_the_merge_needs_both_tranches_and_no_third(tmp_path):
    manifests = _manifest_rows()
    parts = _parts(tmp_path, manifests)
    with pytest.raises(ValueError, match="confirm.2"):
        ted.merge_confirm_decodes(parts, tmp_path / "a", manifests={
            "confirm.1": manifests["confirm.1"]}, provenance={}, models=MODELS)
    with pytest.raises(ValueError, match="confirm.3"):
        ted.merge_confirm_decodes(parts, tmp_path / "b", manifests={
            **manifests, "confirm.3": _rows("confirm.3")}, provenance={}, models=MODELS)


def test_the_merge_refuses_a_clip_in_both_tranches(tmp_path):
    manifests = _manifest_rows()
    manifests["confirm.2"] = [dict(manifests["confirm.1"][0], shard_file=SHARDS["confirm.2"]),
                              *manifests["confirm.2"]]
    with pytest.raises(ValueError, match="both"):
        ted.merge_confirm_decodes(_parts(tmp_path, manifests), tmp_path / "decodes",
                                  manifests=manifests, provenance={}, models=MODELS)

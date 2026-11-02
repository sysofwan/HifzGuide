"""Which rows phase A may touch (#263): refusals, on real git states and synthetic manifests.

The interface, in ``tadabur.tlog_mining.selection``:

* ``require_phase(phase)``: ``"A"`` passes; any other phase (``"B"``, ``"C"``, anything) is a
  ``ValueError`` naming the phase. Phase B waits for #262's amendment and phase C for round 3's
  verdict, so the launcher does not run them.
* ``phase_a_rows(tranche, *, tranches_dir=TRANCHES_DIR, split_dir=SPLIT_DIR, layout=None,
  spec=CONSONANT2) -> list[dict]``: the **gating** rows of ``look``, ``confirm.1`` or
  ``confirm.2`` and nothing else. It goes through ``tlog_eval_decode``'s own readers, so the
  confirm tranches still need their committed claims and the committed ``choice.json``. Any
  other tranche (``confirm.3``, ``confirm.4``, ``train``, anything) is a ``ValueError`` before a
  file is opened. Every returned row must also be a row of that tranche's role in the committed
  split tree (``split_dir/*/*.jsonl.gz``, rows ``{split, shard_file, row, role, ...}``): a row
  the tree gives another role (``train`` above all) or does not hold is a ``ValueError``
  naming the role.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tadabur.test_tlog_eval_confirm import _claim, _manifests, _rows
from tadabur.test_tlog_eval_decode import _manifest_row, _write_jsonl
from tadabur.tlog_mining.selection import phase_a_rows, require_phase
from training import round2_verdict_fixtures as rv

ROLES = ("look", "confirm.1", "confirm.2", "confirm.3", "confirm.4")


def _split_tree(directory: Path, roles: dict[str, list[dict]], *, override: dict | None = None):
    """The committed split tree for ``roles``' rows; ``override`` maps a row key to its role."""
    override = override or {}
    rows = []
    for role, manifest in roles.items():
        for row in manifest:
            key = (row["split"], row["shard_file"], row["row"])
            rows.append({"audio_path": row["audio_path"], "group_key": row["group_key"],
                         "role": override.get(key, role), "row": row["row"],
                         "shard_file": row["shard_file"], "speaker_cluster": None,
                         "split": row["split"]})
    path = directory / "clean" / "part.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    return directory


@pytest.fixture
def claimed(tmp_path):
    round_ = rv.committed_round(tmp_path)
    _manifests(round_)
    _claim(round_, "confirm.1")
    _claim(round_, "confirm.2")
    return round_


def _select(round_, tranche, tmp_path, **tree):
    split_dir = _split_tree(tmp_path / "split", {role: _rows(role) for role in ROLES}, **tree)
    return phase_a_rows(tranche, tranches_dir=round_.tranches, split_dir=split_dir,
                        layout=round_.layout, spec=round_.spec)


def test_only_phase_a_is_runnable():
    require_phase("A")
    for phase in ("B", "C", "D", "", "a b"):
        with pytest.raises(ValueError, match="phase"):
            require_phase(phase)


@pytest.mark.parametrize("tranche", ["look", "confirm.1", "confirm.2"])
def test_a_phase_a_tranche_yields_its_gating_rows_and_no_reported_row(claimed, tmp_path, tranche):
    rows = _select(claimed, tranche, tmp_path)
    assert rows == [row for row in _rows(tranche) if row["stratum"] == "gating"]
    assert len(rows) == 3


@pytest.mark.parametrize("tranche", ["confirm.3", "confirm.4", "train", "confirm", "", "all"])
def test_no_other_tranche_is_selected(claimed, tmp_path, tranche):
    with pytest.raises(ValueError):
        _select(claimed, tranche, tmp_path)


def test_a_refused_tranche_opens_no_file(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        phase_a_rows("confirm.3", tranches_dir=empty, split_dir=empty)


def test_confirm_still_needs_its_committed_claim(tmp_path):
    round_ = rv.committed_round(tmp_path)
    _manifests(round_)
    with pytest.raises(ValueError, match="claim"):
        _select(round_, "confirm.1", tmp_path)


def test_a_row_the_split_tree_gives_to_train_is_refused(claimed, tmp_path):
    victim = _rows("confirm.1")[0]
    key = (victim["split"], victim["shard_file"], victim["row"])
    with pytest.raises(ValueError, match="train"):
        _select(claimed, "confirm.1", tmp_path, override={key: "train"})


def test_a_row_the_split_tree_gives_to_another_tranche_is_refused(claimed, tmp_path):
    victim = _rows("look")[0]
    key = (victim["split"], victim["shard_file"], victim["row"])
    with pytest.raises(ValueError, match="confirm.3"):
        _select(claimed, "look", tmp_path, override={key: "confirm.3"})


def test_a_row_the_split_tree_does_not_hold_is_refused(claimed, tmp_path):
    split_dir = _split_tree(tmp_path / "split", {role: _rows(role)[:2] for role in ROLES})
    with pytest.raises(ValueError):
        phase_a_rows("look", tranches_dir=claimed.tranches, split_dir=split_dir,
                     layout=claimed.layout, spec=claimed.spec)


def test_a_look_row_planted_in_a_confirm_manifest_is_refused(claimed, tmp_path):
    planted = _rows("look")[0]
    _write_jsonl(claimed.tranches / "confirm.3.jsonl", [planted])
    claimed.commit(claimed.tranches / "confirm.3.jsonl", message="plant")
    with pytest.raises(ValueError):
        _select(claimed, "look", tmp_path)


def test_the_real_split_tree_is_the_default_tree():
    from tadabur import tlog_tranches
    from tadabur.tlog_mining import selection

    assert selection.SPLIT_DIR == tlog_tranches.SPLIT_DIR
    assert selection.TRANCHES_DIR == tlog_tranches.TRANCHES_DIR


def test_the_committed_look_gating_rows_are_selectable_from_the_real_trees():
    """No synthetic state: the repository's own look manifest and split tree agree."""
    rows = phase_a_rows("look")
    assert len(rows) == 2853 and {row["stratum"] for row in rows} == {"gating"}

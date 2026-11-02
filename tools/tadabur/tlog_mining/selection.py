"""Which rows phase A may touch (#263): the gating stratum of look and confirm.1/2, nothing else.

Phase A aligns the gating rows of the three tranches round 2 already read: ``look`` and the two
claimed confirm tranches. Every other tranche — ``confirm.3``/``confirm.4`` (sealed for #266's
pilot), ``train``, or a name that is no tranche at all — is refused before a file is opened.
Phases B and C are refused for the same reason: B waits for #262's amendment and C for round 3's
verdict, so this launcher implements neither.

The rows come from :mod:`tadabur.tlog_eval_decode`'s own readers, so a confirm tranche still
needs its committed claim and the committed ``choice.json``, and a row planted in another
tranche's manifest stays refused. Every selected row is then checked against the committed split
tree: the tree's ``role`` for that row must be the tranche itself, so a row the tree gives to
``train`` (or to another confirm tranche) is never aligned here even if a manifest names it, and
a row the tree does not hold is refused rather than silently skipped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from tadabur import tlog_eval_decode as ted
from tadabur.tlog_pass1 import read_jsonl_gz
from tadabur.tlog_pool import row_key
from tadabur.tlog_tranches import SPLIT_DIR, TRANCHES_DIR
from training.tlog_eval import CONFIRM_TRANCHES
from training.aug_readout_spec import CONSONANT2, RoundSpec

#: The only phase this package implements.
PHASE = "A"
#: The tranches phase A reads, in the launcher's order.
PHASE_A_TRANCHES = (ted.LOOK, CONFIRM_TRANCHES[0], CONFIRM_TRANCHES[1])


def require_phase(phase: str) -> None:
    """Refuse any phase but ``A``: B waits for #262 and C for round 3's verdict."""
    if phase != PHASE:
        raise ValueError(f"phase {phase} is not runnable here: phase {PHASE} only "
                         f"(phase B waits for #262, phase C for round 3)")


def phase_a_rows(tranche: str, *, tranches_dir: Path = TRANCHES_DIR,
                 split_dir: Path = SPLIT_DIR, layout=None,
                 spec: RoundSpec = CONSONANT2) -> list[dict]:
    """``tranche``'s gating rows, each one the committed split tree also gives to that tranche."""
    if tranche not in PHASE_A_TRANCHES:
        raise ValueError(f"{tranche!r} is not a phase A tranche {list(PHASE_A_TRANCHES)}: "
                         f"phase A reads the look and confirm.1/2 gating rows only")
    rows = ted.gating_rows(_manifest_rows(tranche, tranches_dir, layout, spec))
    _require_split_roles(rows, tranche, split_dir)
    return rows


def _manifest_rows(tranche: str, tranches_dir: Path, layout, spec: RoundSpec) -> list[dict]:
    if tranche == ted.LOOK:
        return ted.read_manifest(tranche, tranches_dir=tranches_dir)
    return ted.read_confirm_manifest(tranche, tranches_dir=tranches_dir, layout=layout,
                                     spec=spec)


def _require_split_roles(rows: Sequence[Mapping], tranche: str, split_dir: Path) -> None:
    wanted = {row_key(row) for row in rows}
    roles = _split_roles(wanted, split_dir)
    for row in rows:
        key = row_key(row)
        role = roles.get(key)
        if role is None:
            raise ValueError(f"{key} is not a row of the committed split tree {split_dir}")
        if role != tranche:
            raise ValueError(f"{key} is a {role} row of the committed split tree, not a "
                             f"{tranche} row: a {role} row is never aligned here")


def _split_roles(keys: Iterable[tuple[str, str, int]], split_dir: Path) -> dict:
    """The split tree's ``role`` for each wanted row key, file by file."""
    wanted = set(keys)
    roles: dict[tuple[str, str, int], str] = {}
    for path in sorted(Path(split_dir).glob("*/*.jsonl.gz")):
        for row in read_jsonl_gz(path):
            key = row_key(row)
            if key in wanted:
                roles[key] = row["role"]
    return roles

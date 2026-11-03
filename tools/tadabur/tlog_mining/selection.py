"""Which rows each phase may touch (#263): phase A's gating rows, phase B's labelled reported rows.

Phase A aligns the gating rows of the three tranches round 2 already read: ``look`` and the two
claimed confirm tranches. Phase B decodes the **labelled** ``reported:*`` rows of those same two
confirm tranches — the straddles, the identified, the below-bar and the unclean rows that carry a
reference — and lists the rows whose ``assigned_label`` the phonetizer cannot read (#248) instead
of decoding them. Every other tranche — ``confirm.3``/``confirm.4`` (sealed for #266's pilot),
``train``, or a name that is no tranche at all — is refused before a file is opened. Phase C
waits for round 3's verdict, so this package implements neither it nor its readout.

Phase B also requires slice 1's amendment at its **pinned bytes**: :func:`require_phase` hashes
``docs/tlog-mining-amendment.md`` and refuses a missing or changed file, so no exploit of the
mined tranches can run under an amendment the principal architect did not cross-check. The pin is
the file's SHA-256 after the Fable AGREE replaced its pending line; editing the amendment means
re-pinning it here, deliberately.

The rows come from :mod:`tadabur.tlog_eval_decode`'s own readers, so a confirm tranche still
needs its committed claim and the committed ``choice.json``, and a row planted in another
tranche's manifest stays refused. Every selected row is then checked against the committed split
tree: the tree's ``role`` for that row must be the tranche itself, so a row the tree gives to
``train`` (or to another confirm tranche) is never read here even if a manifest names it, and a
row the tree does not hold is refused rather than silently skipped.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from tadabur import tlog_eval_decode as ted
from tadabur.tlog_pass1 import read_jsonl_gz
from tadabur.tlog_pool import row_key
from tadabur.tlog_tranches import SPLIT_DIR, TRANCHES_DIR
from tadabur.waqf_segments import hafs_word_reference
from training.aug_readout_spec import CONSONANT2, RoundSpec
from training.tlog_eval import CONFIRM_TRANCHES, _canonical_words

#: Phase A, the default: word times for the gating clips.
PHASE_A = "A"
#: Phase B: the pool extension decode of the labelled reported rows.
PHASE_B = "B"
#: The phases this package implements; the CLI's default phase is A.
PHASE = PHASE_A
#: The tranches phase A reads, in the launcher's order.
PHASE_A_TRANCHES = (ted.LOOK, CONFIRM_TRANCHES[0], CONFIRM_TRANCHES[1])
#: The tranches phase B pools, in the launcher's order.
PHASE_B_TRANCHES = (CONFIRM_TRANCHES[0], CONFIRM_TRANCHES[1])
#: The reported strata phase B decodes. ``reported:unclean`` joins them only when it carries a
#: label source other than ``none`` (the 2,949 unlabelled unclean rows are skipped).
PHASE_B_STRATA = ("reported:straddles", "reported:below_bar", "reported:identified")
#: Slice 1's amendment and the SHA-256 of the bytes phase B requires: the file after the
#: cross-check comment replaced its pending line.
MINING_AMENDMENT_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "tlog-mining-amendment.md")
MINING_AMENDMENT_SHA256 = "f02bae6271bc1472262164131dee9eaf8e1e80d6f8c1419720421d890ea9313c"


def require_phase(phase: str, *, amendment_path: Path = MINING_AMENDMENT_PATH) -> None:
    """Refuse phase C and anything but ``A``/``B``; ``B`` also needs its pinned amendment."""
    if phase == PHASE_A:
        return
    if phase == PHASE_B:
        _require_amendment(amendment_path)
        return
    if phase == "C":
        raise ValueError("phase C is not runnable here: it waits for round 3's verdict and its "
                         "committed live/EMA choice, so this package neither runs it nor reads "
                         "its output")
    raise ValueError(f"phase {phase} is not a runnable phase: {PHASE_A}, and {PHASE_B} under "
                     f"its pinned amendment")


def _require_amendment(path: Path) -> None:
    """Refuse unless ``path`` holds exactly the pinned amendment bytes."""
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        raise ValueError(f"the mining amendment {path} is missing: phase {PHASE_B} requires it "
                         f"at its pinned SHA-256 {MINING_AMENDMENT_SHA256}") from None
    if digest != MINING_AMENDMENT_SHA256:
        raise ValueError(f"the mining amendment {path} is {digest}, not the pinned SHA-256 "
                         f"{MINING_AMENDMENT_SHA256}: phase {PHASE_B} requires the amendment "
                         f"slice 1 pinned")


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


def phase_b_rows(tranche: str, *, tranches_dir: Path = TRANCHES_DIR,
                 split_dir: Path = SPLIT_DIR, layout=None, spec: RoundSpec = CONSONANT2,
                 phonetizable: Callable[[str], bool] | None = None
                 ) -> tuple[list[dict], list[dict]]:
    """``tranche``'s decodable rows, and the rows whose reference cannot be phonetized (#248).

    A row is decodable when it is a labelled ``reported:*`` row — see
    :data:`PHASE_B_STRATA` and the ``reported:unclean`` rule — and ``phonetizable`` accepts its
    **``assigned_label``**: the ayah the row is read against, never the filename it claims.
    ``phonetizable=None`` is the real phonetizer. A rejecting row is listed, not decoded, so a
    run can never silently drop one. The split-tree check covers both lists.
    """
    if tranche not in PHASE_B_TRANCHES:
        raise ValueError(f"{tranche!r} is not a phase B tranche {list(PHASE_B_TRANCHES)}: "
                         f"phase B decodes the labelled reported rows of confirm.1/2 only")
    rows = [row for row in _manifest_rows(tranche, tranches_dir, layout, spec)
            if _mineable(row)]
    _require_split_roles(rows, tranche, split_dir)
    readable = _phonetizer() if phonetizable is None else phonetizable
    decodable: list[dict] = []
    listed: list[dict] = []
    for row in rows:
        label = row.get("assigned_label")
        (decodable if label is not None and readable(label) else listed).append(row)
    return decodable, listed


def _mineable(row: Mapping) -> bool:
    """A labelled ``reported:*`` row: a straddle, identified, below-bar, or labelled unclean."""
    return (row["stratum"] in PHASE_B_STRATA
            or (row["stratum"] == "reported:unclean"
                and row.get("label_source") not in (None, "none")))


def _phonetizer() -> Callable[[str], bool]:
    """The real phonetizer's verdict on one ``surah:ayah``: its reference builds, or it does not.

    The same two failures phase A's aligner records as ``aligned: false`` (a missing ayah, an
    index outside the text) are the ones that make a row unreadable, so a listed row and an
    unaligned clip name the same ayat (#248).
    """
    reference = hafs_word_reference()

    def readable(label: str) -> bool:
        try:
            reference(_canonical_words(label))
        except (KeyError, IndexError):
            return False
        return True

    return readable


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
                             f"{tranche} row: a {role} row is never read here")


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

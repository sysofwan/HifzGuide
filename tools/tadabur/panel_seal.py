"""The sealed panel's seal, enforced where audio enters the tools (#89).

The sealed held-out panel (:mod:`tadabur.sealed_panel`) is scored once, by the ship
criterion (#97), with the owner's authorization (acceptance rules §3). A wrapper that
asks politely is not a seal: any tool that globs a directory of WAVs or loads a staging
registry could decode panel audio without passing through it. So every shared input
boundary calls :func:`refuse_sealed` (one file) or :func:`refuse_sealed_paths` (a glob)
**before** the audio loses its identity, and :func:`tadabur.staged_audio.load_staged_clips`
refuses a registry that lists a panel clip.

**What counts as panel audio** is decided by provenance against the panel's committed
staging registry (``sealed_panel/staged_clips.jsonl``), never by a path: a file whose name
is a panel clip's ``audio_filename``, or whose bytes have a panel clip's SHA-256 (a copy
under another name). Only files whose size is a panel clip's PCM_16 WAV size are hashed,
so the check costs nothing on ordinary audio.

**Unsealing** is lexically scoped: inside ``with unsealed(authorization):`` the checks pass,
for exactly two authorizations. :data:`SHIP_CRITERION_AUTHORIZATION` is #97's one-time
score. :data:`PREPARATION_AUTHORIZATION` is the panel's own preparation in
:mod:`tadabur.sealed_panel`: staging its audio, and the base teacher's segmentation and
decode cache, with no gate and no scoring. A lint test pins the modules that may name
either.

This module is import-light (stdlib only) so every loader can call it.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

PANEL_REGISTRY_PATH = Path(__file__).parent / "sealed_panel" / "staged_clips.jsonl"

#: #97's one-time, owner-authorized score of the panel (acceptance rules §3).
SHIP_CRITERION_AUTHORIZATION = (
    "issue-97: the owner authorized the one-time score of the sealed panel"
)
#: The panel's own staging, segmentation and base-teacher decode cache (#89). No scoring.
PREPARATION_AUTHORIZATION = "issue-89: stage and segment the sealed panel, teacher cache only"
_AUTHORIZATIONS = frozenset({SHIP_CRITERION_AUTHORIZATION, PREPARATION_AUTHORIZATION})

#: A canonical PCM_16 mono WAV's header, as soundfile writes the staged clips.
_WAV_HEADER_BYTES = 44

_UNSEALED: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sealed_panel_unsealed", default=None
)


class SealedPanelError(RuntimeError):
    """Something tried to read the sealed panel without an authorization."""


@dataclass(frozen=True)
class PanelProvenance:
    """What identifies panel audio: its filenames, checksums and staged file sizes."""

    filenames: frozenset[str]
    checksums: frozenset[str]
    file_sizes: frozenset[int]


@functools.lru_cache(maxsize=4)
def panel_provenance(path: Path = PANEL_REGISTRY_PATH) -> PanelProvenance:
    """The panel's provenance from its staging registry (empty if there is no panel)."""
    if not path.exists():
        return PanelProvenance(frozenset(), frozenset(), frozenset())
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(raw) for raw in f if raw.strip()]
    return PanelProvenance(
        frozenset(row["audio_filename"] for row in rows),
        frozenset(row["audio_sha256"] for row in rows),
        frozenset(_WAV_HEADER_BYTES + 2 * row["num_samples"] for row in rows),
    )


@contextlib.contextmanager
def unsealed(authorization: str | None) -> Iterator[None]:
    """Let panel audio through inside the block, for an authorized purpose only."""
    if authorization not in _AUTHORIZATIONS:
        raise SealedPanelError(_REFUSAL)
    token = _UNSEALED.set(authorization)
    try:
        yield
    finally:
        _UNSEALED.reset(token)


def is_panel_audio(path: Path, provenance: PanelProvenance | None = None) -> bool:
    """Whether ``path`` names, or byte-for-byte copies, a panel clip."""
    provenance = provenance or panel_provenance()
    path = Path(path)
    if path.name in provenance.filenames:
        return True
    if not provenance.checksums or not path.is_file():
        return False
    if path.stat().st_size not in provenance.file_sizes:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() in provenance.checksums


def refuse_sealed(path: Path, provenance: PanelProvenance | None = None) -> Path:
    """``path``, unless it is panel audio read outside an authorized block."""
    if _UNSEALED.get() is None and is_panel_audio(path, provenance):
        raise SealedPanelError(f"{path} is sealed-panel audio. {_REFUSAL}")
    return path


def refuse_sealed_paths(
    paths: Iterable[Path], provenance: PanelProvenance | None = None
) -> list[Path]:
    """Every path, checked: what a glob loader returns."""
    return [refuse_sealed(path, provenance) for path in paths]


def refuse_sealed_names(
    names: Iterable[str], checksums: Iterable[str | None] = (),
    provenance: PanelProvenance | None = None,
) -> None:
    """Refuse a registry or manifest that names a panel clip, or records a panel checksum."""
    if _UNSEALED.get() is not None:
        return
    provenance = provenance or panel_provenance()
    hits = sorted(set(names) & provenance.filenames)
    hits += sorted(set(checksums) & provenance.checksums)
    if hits:
        raise SealedPanelError(f"{hits[:3]} belong to the sealed panel. {_REFUSAL}")


_REFUSAL = (
    "The sealed held-out panel (#89) is scored only by the ship criterion (#97), once, "
    "with the owner's authorization (acceptance rules §3). Select, tune and diagnose on "
    "the mining pool or decode_evalset dev."
)

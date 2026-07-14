"""Sample a waqf event-adjudication worklist for the P7.F0 HITL gate (#27).

The waqf head is distilled from a silence VAD, so nothing in the pipeline knows
whether a detected silence is a real **waqf**, a mid-word stop-consonant/hamza
**closure**, or whether a *continued* word boundary was a genuine **wasl** — those
are the three errors ADR-0004's event-level eval must measure, and only a human can
call them. This module draws the labelling worklist: from a candidate-boundary
manifest (produced by the segmentation/VAD pass, the waqf analogue of the poison
audit's filter manifest), it samples up to ``per_class`` boundaries **per predicted
class** so the fixture is stocked across all three error types, not dominated by the
common case.

Sampling is pure and reproducible, mirroring :mod:`tadabur.audit_sampler`: each
class draws from an independent ``(seed, predicted)`` RNG over records sorted by
``(clip_id, boundary_index)``, so the same manifest + seed always yields the same
worklist regardless of class order. ``local_audio_path`` is the filename the
boundary's **whole** clip audio is served under — the adjudication UI plays the clip
and seeks to ``(start_s, end_s)``. Unlike the poison audit — which exports its own
audio and so hashes ``audio_ref`` into a fresh collision-proof name — the waqf whole
clips are already staged on disk by :mod:`tadabur.waqf_segments` (the same 16 kHz
clips the VAD/segmentation pass that produced these candidates analysed), each under
its raw ``audio_filename``. ``local_audio_path`` is therefore just ``audio_ref``, so
the UI's ``--audio-dir`` *is* that staging directory and every served name already
exists in it — no separate export step, and no rows the human cannot adjudicate.

Usage:
  python -m tadabur.waqf_event_sampler --candidates candidates.jsonl \
    --clips waqf_clip_worklist.jsonl --seed 0 [--per-class 50]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .waqf_event_fixtures import WAQF_EVENT_CLASSES

DEFAULT_PER_CLASS = 50
_CLASSES = frozenset(WAQF_EVENT_CLASSES)


@dataclass(frozen=True)
class WaqfCandidate:
    """One detector-proposed candidate boundary, the sampler's input unit.

    ``clip_id`` / ``audio_ref`` name the whole clip; ``boundary_index`` orders the
    candidate boundaries within it (with ``clip_id``, the stable key). ``word_index``
    is the Uthmani word the boundary falls after and ``(start_s, end_s)`` its time
    span in the clip. ``predicted`` is the detector's class (one of
    :data:`~tadabur.waqf_event_fixtures.WAQF_EVENT_CLASSES`) — the stratum this
    candidate is sampled into.
    """

    clip_id: str
    audio_ref: str
    surah_ayah: str
    boundary_index: int
    word_index: int
    start_s: float
    end_s: float
    predicted: str


@dataclass(frozen=True)
class WaqfCandidateItem:
    """One worklist row: a :class:`WaqfCandidate` plus its clip's audio filename.

    ``local_audio_path`` is the filename the boundary's whole clip audio is served
    under. The waqf whole clips are staged on disk by :mod:`tadabur.waqf_segments`
    under their raw ``audio_filename``, so this is simply ``audio_ref`` — the UI's
    ``--audio-dir`` is that staging directory and every row's audio already exists
    in it (no separate export, unlike :mod:`tadabur.audit_sampler`).
    """

    clip_id: str
    audio_ref: str
    surah_ayah: str
    boundary_index: int
    word_index: int
    start_s: float
    end_s: float
    predicted: str
    local_audio_path: str


def read_candidates(path: Path) -> list[WaqfCandidate]:
    """Read the candidate-boundary manifest (JSONL) into records, in file order.

    Fails loudly on an unknown ``predicted`` class so a malformed manifest cannot
    silently drop a whole stratum from the worklist. Blank lines are skipped.
    """
    candidates: list[WaqfCandidate] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            candidate = WaqfCandidate(**json.loads(line))
            if candidate.predicted not in _CLASSES:
                raise ValueError(
                    f"{path}:{lineno}: candidate {candidate.clip_id!r}#"
                    f"{candidate.boundary_index} has unknown predicted class "
                    f"{candidate.predicted!r}, expected one of {list(WAQF_EVENT_CLASSES)}"
                )
            candidates.append(candidate)
    return candidates


def _sample(records: list[WaqfCandidate], n: int, seed: object) -> list[WaqfCandidate]:
    """Up to ``n`` candidates drawn reproducibly from ``records`` for one class.

    ``records`` is sorted by ``(clip_id, boundary_index)`` first so the draw depends
    only on the manifest contents and ``seed``, not on iteration order. When a class
    has ``n`` or fewer records all are taken. The result is returned in that same
    stable order.
    """
    ordered = sorted(records, key=lambda c: (c.clip_id, c.boundary_index))
    if len(ordered) <= n:
        return ordered
    chosen = random.Random(f"{seed}:{n}").sample(ordered, n)
    return sorted(chosen, key=lambda c: (c.clip_id, c.boundary_index))


def _item(candidate: WaqfCandidate) -> WaqfCandidateItem:
    return WaqfCandidateItem(
        **asdict(candidate),
        local_audio_path=candidate.audio_ref,
    )


def sample_worklist(
    candidates: list[WaqfCandidate],
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = 0,
) -> list[WaqfCandidateItem]:
    """Build the deterministic per-class waqf-boundary adjudication worklist.

    For each of the three predicted classes, sample up to ``per_class`` candidates
    from an independent ``(seed, predicted)`` RNG, so the worklist is reproducible
    and class-order-independent and every stratum is represented. Rows are returned
    grouped by class in :data:`WAQF_EVENT_CLASSES` order.
    """
    by_class: dict[str, list[WaqfCandidate]] = {c: [] for c in WAQF_EVENT_CLASSES}
    for candidate in candidates:
        by_class[candidate.predicted].append(candidate)
    items: list[WaqfCandidateItem] = []
    for predicted in WAQF_EVENT_CLASSES:
        for candidate in _sample(by_class[predicted], per_class, f"{seed}:{predicted}"):
            items.append(_item(candidate))
    return items


def write_worklist(items: list[WaqfCandidateItem], path: Path) -> None:
    """Write the worklist as JSONL, one :class:`WaqfCandidateItem` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")


def sample_clips(
    candidates: list[WaqfCandidate],
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = 0,
) -> list[str]:
    """The distinct clips the correction-based per-clip UI walks through.

    The UI (:mod:`tadabur.waqf_audit_ui`) reviews a whole **clip** at a time —
    the candidate manifest is the assumed-correct baseline and the human only marks
    the boundaries the detector got wrong. The worklist is therefore a set of clips,
    not boundaries. We reuse :func:`sample_worklist`'s stratified per-class draw so
    every clip that carries a sampled ``waqf`` / ``wasl`` / ``mid_word_closure``
    boundary is included, then collapse to the distinct clips in first-seen order.
    Deterministic in ``(seed, per_class)``.
    """
    seen: dict[str, None] = {}
    for item in sample_worklist(candidates, per_class=per_class, seed=seed):
        seen.setdefault(item.clip_id, None)
    return list(seen)


def write_clip_worklist(clip_ids: list[str], path: Path) -> None:
    """Write the clip worklist as JSONL, one ``{"clip_id": ...}`` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for clip_id in clip_ids:
            f.write(json.dumps({"clip_id": clip_id}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Candidate-boundary manifest (JSONL of WaqfCandidate rows).")
    parser.add_argument("--worklist", type=Path, default=None,
                        help="Output per-boundary worklist (JSONL of WaqfCandidateItem rows).")
    parser.add_argument("--clips", type=Path, default=None,
                        help="Output clip worklist (JSONL of {\"clip_id\": ...}) for the "
                             "correction-based per-clip UI.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS,
                        help=f"Candidates to sample per predicted class (default: {DEFAULT_PER_CLASS}).")
    args = parser.parse_args()

    if args.worklist is None and args.clips is None:
        parser.error("at least one of --worklist or --clips is required")

    candidates = read_candidates(args.candidates)
    if args.worklist is not None:
        items = sample_worklist(candidates, per_class=args.per_class, seed=args.seed)
        write_worklist(items, args.worklist)
        print(f"Wrote {len(items)} worklist rows from {len(candidates)} candidate boundaries to {args.worklist}.")
    if args.clips is not None:
        clip_ids = sample_clips(candidates, per_class=args.per_class, seed=args.seed)
        write_clip_worklist(clip_ids, args.clips)
        print(f"Wrote {len(clip_ids)} clips from {len(candidates)} candidate boundaries to {args.clips}.")


if __name__ == "__main__":
    main()

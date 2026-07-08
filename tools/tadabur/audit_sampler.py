"""Per-contrast audit sampler — the AFK precursor to the P3.5 poison audit (#6).

From an existing filter manifest (``tadabur.manifest``), draw a deterministic,
seeded labelling worklist: ~N randomly-admitted clips **per contrast** (the seven
buckets from ``contrast_attribution.all_contrasts`` — six soft pairs + shadda),
plus a configurable sample from the marginal ``match_ratio`` band just above the
gate threshold (ADR-0001's ~0.65–0.72 glance). Each worklist row is
``(clip_id, contrast, match_ratio, audio_ref)`` so a human can bucket clips B
(gold) vs C (poison) per contrast.

Sampling is pure and reproducible: each bucket draws from an independent
per-bucket RNG seeded by ``(seed, contrast)``, over records sorted by
``audio_filename``, so the same manifest + seed always yields the same worklist
regardless of bucket order. Exporting the sampled audio for listening streams the
source dataset once (lazy, optional) and is kept out of the pure sampling path so
the worklist can be produced — and unit-tested — without the dataset.

Usage:
  python -m tadabur.audit_sampler --manifest passing_subset.jsonl \
    --worklist audit_worklist.jsonl --seed 0 [--per-contrast 30] \
    [--marginal-n 30] [--threshold 0.65] [--marginal-band 0.07] \
    [--audio-dir audit_audio/ --config-name preview]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .contrast_attribution import MARGINAL_CONTRAST, all_contrasts
from .manifest import ManifestRecord, read_records

DEFAULT_PER_CONTRAST = 30
DEFAULT_MARGINAL_N = 30
# Gate threshold and the width of the marginal band audited above it (ADR-0001's
# ~0.65–0.72 glance): records with threshold ≤ match_ratio ≤ threshold + band.
DEFAULT_THRESHOLD = 0.65
DEFAULT_MARGINAL_BAND = 0.07


@dataclass(frozen=True)
class WorklistItem:
    """One clip to audit for one contrast bucket."""

    clip_id: str
    contrast: str
    match_ratio: float
    audio_ref: str


def _sample(records: list[ManifestRecord], n: int, seed: object) -> list[ManifestRecord]:
    """Up to ``n`` records drawn reproducibly from ``records`` for one bucket.

    ``records`` is sorted by ``audio_filename`` first so the draw depends only on
    the manifest contents and ``seed``, not on file/iteration order. When a bucket
    has ``n`` or fewer records all are taken. The result is returned in stable
    ``audio_filename`` order.
    """
    ordered = sorted(records, key=lambda r: r.audio_filename)
    if len(ordered) <= n:
        return ordered
    chosen = random.Random(f"{seed}:{n}").sample(ordered, n)
    return sorted(chosen, key=lambda r: r.audio_filename)


def _bucket(records: list[ManifestRecord], contrast: str) -> list[ManifestRecord]:
    return [r for r in records if contrast in r.contrasts]


def _marginal_bucket(
    records: list[ManifestRecord], threshold: float, band: float
) -> list[ManifestRecord]:
    return [r for r in records if threshold <= r.match_ratio <= threshold + band]


def sample_worklist(
    records: list[ManifestRecord],
    per_contrast: int = DEFAULT_PER_CONTRAST,
    marginal_n: int = DEFAULT_MARGINAL_N,
    threshold: float = DEFAULT_THRESHOLD,
    marginal_band: float = DEFAULT_MARGINAL_BAND,
    seed: int = 0,
) -> list[WorklistItem]:
    """Build the deterministic per-contrast + marginal audit worklist.

    For each of the seven contrast buckets, sample up to ``per_contrast`` admitted
    clips; then sample up to ``marginal_n`` from the ``[threshold, threshold+band]``
    ``match_ratio`` band. A clip that exhibits several contrasts is audited once per
    contrast it exhibits (its purpose in each bucket differs). Buckets draw from
    independent ``(seed, contrast)`` RNGs, so the worklist is reproducible and
    bucket-order-independent.
    """
    items: list[WorklistItem] = []
    for contrast in all_contrasts():
        for record in _sample(_bucket(records, contrast), per_contrast, f"{seed}:{contrast}"):
            items.append(
                WorklistItem(record.audio_filename, contrast, record.match_ratio, record.audio_filename)
            )
    marginal = _marginal_bucket(records, threshold, marginal_band)
    for record in _sample(marginal, marginal_n, f"{seed}:{MARGINAL_CONTRAST}"):
        items.append(
            WorklistItem(record.audio_filename, MARGINAL_CONTRAST, record.match_ratio, record.audio_filename)
        )
    return items


def write_worklist(items: list[WorklistItem], path: Path) -> None:
    """Write the worklist as JSONL, one :class:`WorklistItem` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")


def _safe_name(audio_ref: str) -> str:
    """A flat, collision-free filename for an ``audio_ref`` (no path traversal)."""
    return audio_ref.replace("/", "__").replace("\\", "__").lstrip(".")


def export_audio(
    items: list[WorklistItem],
    out_dir: Path,
    dataset_id: str,
    config_name: str | None,
    split: str,
) -> set[str]:
    """Stream the source dataset once and write each sampled clip's audio to disk.

    Reads audio with ``decode=False`` (raw bytes, no ``torchcodec``), writing one
    file per distinct ``audio_ref`` in the worklist so a human can listen. Returns
    the set of ``audio_ref``s found. Imports ``datasets`` lazily so the pure
    sampling path stays dependency-light.
    """
    from datasets import Audio, load_dataset

    from .filter import AUDIO_COLUMN

    wanted = {item.audio_ref for item in items}
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_id, name=config_name, split=split, streaming=True)
    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(decode=False))

    found: set[str] = set()
    for row in dataset:
        name = row.get("audio_filename")
        if name in wanted and name not in found:
            (out_dir / _safe_name(name)).write_bytes(row[AUDIO_COLUMN]["bytes"])
            found.add(name)
            if len(found) == len(wanted):
                break
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Input filter manifest (JSONL).")
    parser.add_argument("--worklist", type=Path, required=True, help="Output worklist (JSONL).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    parser.add_argument("--per-contrast", type=int, default=DEFAULT_PER_CONTRAST,
                        help=f"Clips to sample per contrast (default: {DEFAULT_PER_CONTRAST}).")
    parser.add_argument("--marginal-n", type=int, default=DEFAULT_MARGINAL_N,
                        help=f"Clips to sample from the marginal band (default: {DEFAULT_MARGINAL_N}).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Gate pass threshold (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--marginal-band", type=float, default=DEFAULT_MARGINAL_BAND,
                        help=f"Width of the marginal band above threshold (default: {DEFAULT_MARGINAL_BAND}).")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="If set, export sampled clips' audio here for listening.")
    parser.add_argument("--dataset", default="FaisaI/tadabur", help="HF dataset id (for --audio-dir).")
    parser.add_argument("--config-name", default=None, help="Dataset config name (for --audio-dir).")
    parser.add_argument("--split", default="train", help="Dataset split (for --audio-dir).")
    args = parser.parse_args()

    records = read_records(args.manifest)
    items = sample_worklist(
        records,
        per_contrast=args.per_contrast,
        marginal_n=args.marginal_n,
        threshold=args.threshold,
        marginal_band=args.marginal_band,
        seed=args.seed,
    )
    write_worklist(items, args.worklist)
    print(f"Wrote {len(items)} worklist rows from {len(records)} manifest passers to {args.worklist}.")

    if args.audio_dir is not None:
        found = export_audio(items, args.audio_dir, args.dataset, args.config_name, args.split)
        print(f"Exported audio for {len(found)} distinct clips to {args.audio_dir}.")


if __name__ == "__main__":
    main()

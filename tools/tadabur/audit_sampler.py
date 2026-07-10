"""Per-contrast audit sampler — the AFK precursor to the P3.5 poison audit (#6).

From an existing filter manifest (``tadabur.manifest``), draw a deterministic,
seeded labelling worklist: ~N randomly-admitted clips **per contrast** (the seven
buckets from ``contrast_attribution.all_contrasts`` — six soft pairs + shadda),
plus a configurable sample from the marginal ``match_ratio`` band just above the
gate threshold (ADR-0001's ~0.65–0.72 glance). Each worklist row is
``(clip_id, contrast, match_ratio, audio_ref, local_audio_path)`` so a human can
bucket clips B (gold) vs C (poison) per contrast, and can locate every sampled
clip's exported audio by its ``local_audio_path``.

Sampling is pure and reproducible: each bucket draws from an independent
per-bucket RNG seeded by ``(seed, contrast)``, over records sorted by
``audio_filename``, so the same manifest + seed always yields the same worklist
regardless of bucket order. ``local_audio_path`` is a pure, deterministic function
of ``audio_ref`` (a SHA-256 prefix + sanitized basename), so the worklist names
each clip's export target up front — independent of whether audio is exported.

Exporting the sampled audio for listening streams the source dataset once (lazy,
optional) and is kept out of the pure sampling path so the worklist can be produced
— and unit-tested — without the dataset. Export **fails loudly** if any sampled
``audio_ref`` is absent from the chosen dataset/config/split, so a worklist is never
silently left with unlistenable rows.

Usage:
  python -m tadabur.audit_sampler --manifest passing_subset.jsonl \
    --worklist audit_worklist.jsonl --seed 0 [--per-contrast 30] \
    [--marginal-n 30] [--threshold 0.65] [--marginal-band 0.07] \
    [--audio-dir audit_audio/ --config-name preview]
"""

from __future__ import annotations

import argparse
import hashlib
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


def local_audio_path(audio_ref: str) -> str:
    """A collision-proof, deterministic local filename for ``audio_ref``.

    The 16-hex SHA-256 prefix of the *full* ref guarantees two distinct refs never
    map to the same file — unlike a flat ``/``→``__`` rewrite, where ``a/b.wav`` and
    ``a__b.wav`` would collide and silently overwrite one sampled clip with another.
    The trailing sanitized basename keeps the name human-legible, and the flat
    result (no separators) is immune to path traversal.
    """
    digest = hashlib.sha256(audio_ref.encode("utf-8")).hexdigest()[:16]
    basename = audio_ref.replace("\\", "/").rsplit("/", 1)[-1]
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in basename)
    return f"{digest}_{safe}" if safe else digest


@dataclass(frozen=True)
class WorklistItem:
    """One clip to audit for one contrast bucket.

    ``local_audio_path`` is the deterministic filename this clip's audio is exported
    to under ``--audio-dir`` (see :func:`local_audio_path`), so #6 can locate every
    sampled row's audio even though export is optional and runs separately.
    """

    clip_id: str
    contrast: str
    match_ratio: float
    audio_ref: str
    local_audio_path: str


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


def _item(record: ManifestRecord, contrast: str) -> WorklistItem:
    return WorklistItem(
        clip_id=record.audio_filename,
        contrast=contrast,
        match_ratio=record.match_ratio,
        audio_ref=record.audio_filename,
        local_audio_path=local_audio_path(record.audio_filename),
    )


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
            items.append(_item(record, contrast))
    marginal = _marginal_bucket(records, threshold, marginal_band)
    for record in _sample(marginal, marginal_n, f"{seed}:{MARGINAL_CONTRAST}"):
        items.append(_item(record, MARGINAL_CONTRAST))
    return items


def write_worklist(items: list[WorklistItem], path: Path) -> None:
    """Write the worklist as JSONL, one :class:`WorklistItem` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")


def _require_all_exported(wanted: set[str], found: set[str], dataset_id: str, split: str) -> None:
    """Raise if any sampled ``audio_ref`` never turned up in the stream.

    A worklist with unlistenable rows is worse than a loud failure: the P3.5 audit
    (#6) would silently skip the missing clips. Listing the misses points the
    operator at the wrong dataset/config/split (or a stale manifest).
    """
    missing = sorted(wanted - found)
    if missing:
        preview = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ValueError(
            f"{len(missing)} sampled clip(s) were not found in dataset "
            f"{dataset_id!r} split {split!r}; cannot export their audio: "
            f"{preview}{more}. Check --dataset/--config-name/--split match the "
            f"manifest, or that the manifest is not stale."
        )


def export_audio(
    items: list[WorklistItem],
    out_dir: Path,
    dataset_id: str,
    config_name: str | None,
    split: str,
) -> dict[str, str]:
    """Stream the source dataset once and write each sampled clip's audio to disk.

    Reads audio with ``decode=False`` (raw bytes, no ``torchcodec``), writing one
    file per distinct ``audio_ref`` to its :func:`local_audio_path` under ``out_dir``
    — a collision-proof, deterministic name so two refs never overwrite each other.
    Returns a mapping ``audio_ref -> local_audio_path`` for the clips written, and
    **raises** if any sampled ref is missing so the worklist is never left with
    rows a human cannot listen to. Imports ``datasets`` lazily so the pure sampling
    path stays dependency-light.
    """
    from datasets import Audio, load_dataset

    from .dataset_source import AUDIO_COLUMN, resolve_audio_filename

    wanted = {item.audio_ref: item.local_audio_path for item in items}
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_id, name=config_name, split=split, streaming=True)
    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(decode=False))

    exported: dict[str, str] = {}
    for row in dataset:
        name = resolve_audio_filename(row)
        if name in wanted and name not in exported:
            (out_dir / wanted[name]).write_bytes(row[AUDIO_COLUMN]["bytes"])
            exported[name] = wanted[name]
            if len(exported) == len(wanted):
                break

    _require_all_exported(set(wanted), set(exported), dataset_id, split)
    return exported


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
        exported = export_audio(items, args.audio_dir, args.dataset, args.config_name, args.split)
        print(f"Exported audio for {len(exported)} distinct clips to {args.audio_dir}.")


if __name__ == "__main__":
    main()

"""The staged-audio registry: every Tadabur clip re-staged for labels or mining, with provenance.

``audit_run/`` was cleared on 2026-10-08 and took the audio behind every label with it. What
makes audio re-stageable is not the audio but its **provenance**: which parquet shard and
row of ``FaisaI/tadabur`` it came from, who recited it, how long it is once decoded to
16 kHz mono, and the SHA-256 of the staged file so a re-download can be proven identical.
This module owns that record (#83):

* :class:`StagedClip` — one staged whole clip: ``audio_filename``, ``shard``,
  ``row_index``, the canonical ``reciter_id`` (Tadabur's own ``reciter_id`` column, which
  disagrees with the filename's ``spkNNNN`` on ~1% of clips, so the filename is never
  parsed for it), ``surah_ayah``, ``num_samples`` and ``audio_sha256``, plus the
  ``uses`` it was staged for (:data:`USES`);
* :func:`load_staged_clips` / :func:`write_staged_clips` — the committed registry
  (``staged_audio/clips.jsonl``), validated on both paths;
* :func:`fill_staging` — copies a clip's ``shard``, ``audio_sha256`` and (for a
  whole-clip item) ``end_sample`` into the truth sites that sit on it;
* :func:`stage_clips` — the re-staging itself: download each needed shard once, decode
  the wanted rows with :func:`tadabur.audio.decode_to_mono_16k`, write them as 16 kHz mono
  **PCM_16** WAV (the format the labels were made on and ``truth_sites.audio_sha256``
  hashes), hash them, and delete the shard.

The registry is the clip half of the exposure registry the acceptance rules ask for: the
reciter id, source recording and checksum of every clip, with each use it has had. Spans
within a clip live with the sites that use them.

Usage (from ``tools/``; ``stage`` downloads ~2.4 GB per shard, one at a time)::

  python -m tadabur.staged_audio index --shards 0-20,39,58,77 --out stage/shard_index.jsonl
  python -m tadabur.staged_audio stage --index stage/shard_index.jsonl \\
      --pool-selection stage/pool_selection.jsonl \\
      --audio-dir stage/clips --shard-cache stage/hf_cache
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from .dataset_source import canonical_surah_ayah, resolve_audio_filename
from .truth_sites import (
    NUM_SHARDS,
    P35_FIXTURE,
    WAQF_BOUNDARY,
    TruthSite,
    audio_sha256,
)

STAGED_AUDIO_DIR = Path(__file__).parent / "staged_audio"
REGISTRY_PATH = STAGED_AUDIO_DIR / "clips.jsonl"

#: The clip was staged for the h448-unseen mining pool (:mod:`tadabur.mining_pool`).
MINING_POOL = "mining_pool"
#: Why a clip was staged: truth sites of one source, or the mining pool.
USES = frozenset({WAQF_BOUNDARY, P35_FIXTURE, MINING_POOL})

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class StagedClip:
    """One re-staged whole clip. See the module docstring."""

    audio_filename: str
    shard: int
    row_index: int
    reciter_id: int
    surah_ayah: str
    num_samples: int
    audio_sha256: str
    uses: tuple[str, ...]


_FIELD_TYPES = {
    "audio_filename": str,
    "shard": int,
    "row_index": int,
    "reciter_id": int,
    "surah_ayah": str,
    "num_samples": int,
    "audio_sha256": str,
    "uses": list,
}
assert set(_FIELD_TYPES) == {f.name for f in fields(StagedClip)}


def parse_staged_clip(data: dict, where: str) -> StagedClip:
    """One validated :class:`StagedClip` from its JSON object; ``where`` prefixes errors."""
    if set(data) != set(_FIELD_TYPES):
        raise ValueError(
            f"{where}: fields {sorted(data)} do not match {sorted(_FIELD_TYPES)}"
        )
    for name, expected in _FIELD_TYPES.items():
        if type(data[name]) is not expected:  # exact: a bool is not an int
            raise ValueError(f"{where}: {name} must be {expected.__name__}")
    clip = StagedClip(**{**data, "uses": tuple(data["uses"])})
    if not clip.audio_filename.endswith(".wav") or "/" in clip.audio_filename:
        raise ValueError(f"{where}: audio_filename must be a bare .wav name")
    if not 0 <= clip.shard < NUM_SHARDS:
        raise ValueError(f"{where}: shard {clip.shard} is outside [0, {NUM_SHARDS})")
    if clip.row_index < 0 or clip.reciter_id < 0 or clip.num_samples < 1:
        raise ValueError(f"{where}: row_index, reciter_id and num_samples must be sane")
    if not re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", clip.surah_ayah):
        raise ValueError(f"{where}: surah_ayah {clip.surah_ayah!r} is not 'surah:ayah'")
    if not _SHA256.fullmatch(clip.audio_sha256):
        raise ValueError(f"{where}: audio_sha256 is not 64 lowercase hex characters")
    if not clip.uses or list(clip.uses) != sorted(set(clip.uses)) or not set(clip.uses) <= USES:
        raise ValueError(
            f"{where}: uses must be a sorted, non-empty, duplicate-free subset of "
            f"{sorted(USES)}"
        )
    return clip


def load_staged_clips(path: Path = REGISTRY_PATH) -> dict[str, StagedClip]:
    """The registry keyed by ``audio_filename``; every row validated, no duplicates."""
    clips: dict[str, StagedClip] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            clip = parse_staged_clip(json.loads(raw), f"{path}:{lineno}")
            if clip.audio_filename in clips:
                raise ValueError(f"{path}:{lineno}: duplicate {clip.audio_filename}")
            clips[clip.audio_filename] = clip
    _check_unique_sources(clips.values(), str(path))
    return clips


def _check_unique_sources(clips: Iterable[StagedClip], where: str) -> None:
    seen: dict[tuple[int, int], str] = {}
    for clip in clips:
        other = seen.setdefault((clip.shard, clip.row_index), clip.audio_filename)
        if other != clip.audio_filename:
            raise ValueError(
                f"{where}: {clip.audio_filename} and {other} claim the same shard row"
            )


def write_staged_clips(clips: Iterable[StagedClip], path: Path = REGISTRY_PATH) -> None:
    """Atomically write the registry, sorted by ``audio_filename``, after validating it."""
    ordered = sorted(clips, key=lambda c: c.audio_filename)
    for clip in ordered:
        parse_staged_clip(_as_json(clip), f"write_staged_clips[{clip.audio_filename}]")
    if len({c.audio_filename for c in ordered}) != len(ordered):
        raise ValueError("write_staged_clips: duplicate audio_filename")
    _check_unique_sources(ordered, "write_staged_clips")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for clip in ordered:
            f.write(json.dumps(_as_json(clip), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _as_json(clip: StagedClip) -> dict:
    return {**asdict(clip), "uses": list(clip.uses)}


def fill_staging(
    sites: list[TruthSite], clips: Mapping[str, StagedClip]
) -> tuple[list[TruthSite], list[str]]:
    """Sites with their staging fields copied from the registry, and the unstaged clips.

    A site on a staged clip takes the clip's ``shard`` and ``audio_sha256``. Its item's
    ``end_sample`` is the clip's length when the item is the whole clip (``start_sample``
    0 and no end yet); an item with a recorded span keeps it, and must fit inside the
    clip. A site on a clip the registry lacks keeps all three fields ``null`` and its clip
    is listed, sorted, so nothing is invented for audio that could not be re-staged.
    """
    filled: list[TruthSite] = []
    missing: set[str] = set()
    for site in sites:
        clip = clips.get(site.audio_filename)
        if clip is None:
            missing.add(site.audio_filename)
            filled.append(site)
            continue
        end = site.end_sample
        if end is None:
            if site.start_sample != 0:
                raise ValueError(f"{site.site_id}: a span item needs its own end_sample")
            end = clip.num_samples
        if end > clip.num_samples:
            raise ValueError(
                f"{site.site_id}: end_sample {end} is past the staged clip's "
                f"{clip.num_samples} samples"
            )
        filled.append(
            replace(site, shard=clip.shard, end_sample=end, audio_sha256=clip.audio_sha256)
        )
    return filled, sorted(missing)


# --- re-staging from the shards --------------------------------------------------------


@dataclass(frozen=True)
class IndexRow:
    """Where one clip lives in Tadabur's full config (:func:`iter_shard_metadata`)."""

    audio_filename: str
    shard: int
    row_index: int
    reciter_id: int
    surah_ayah: str
    duration_s: float


def read_shard_index(path: Path) -> dict[str, IndexRow]:
    """A shard index written by ``index``, keyed by ``audio_filename``."""
    rows: dict[str, IndexRow] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            row = json.loads(raw)
            entry = IndexRow(
                audio_filename=row["audio_filename"],
                shard=row["shard"],
                row_index=row["row_index"],
                reciter_id=row["reciter_id"],
                surah_ayah=canonical_surah_ayah(row["surah_id"], row["ayah_id"]),
                duration_s=row["ayah_duration_s"],
            )
            if rows.setdefault(entry.audio_filename, entry) != entry:
                raise ValueError(f"{path}: {entry.audio_filename} appears twice")
    return rows


#: Yields the ``datasets``-shaped rows of one shard, in row order.
ShardRows = Callable[[int], Iterator[dict]]


def stage_clips(
    requests: Mapping[str, frozenset[str]],
    index: Mapping[str, IndexRow],
    audio_dir: Path,
    shard_rows: ShardRows,
    registry: Mapping[str, StagedClip] | None = None,
    on_shard_done: Callable[[dict[str, StagedClip]], None] | None = None,
) -> tuple[dict[str, StagedClip], list[str]]:
    """Stage every requested clip once, and list the ones no indexed shard holds.

    ``requests`` maps each wanted ``audio_filename`` to its uses. Clips already in
    ``registry`` with their WAV present under ``audio_dir`` are not staged again (a
    resumed run); their uses are merged. The rest are grouped by shard and each shard is
    read once, in shard order, through ``shard_rows``. A row is staged only after its
    filename and reciter are checked against the index, so a reordered shard can never
    attach one clip's provenance to another's audio, and a clip the registry already
    knows must re-stage to the checksum it records. ``on_shard_done`` receives the
    registry after each shard so a long run can checkpoint it.
    """
    staged = dict(registry or {})
    unlocatable = sorted(name for name in requests if name not in index)
    pending: dict[int, dict[int, IndexRow]] = {}
    for name, uses in sorted(requests.items()):
        if name in staged and (audio_dir / name).exists():
            merged = tuple(sorted(set(staged[name].uses) | uses))
            staged[name] = replace(staged[name], uses=merged)
        elif name in index:
            entry = index[name]
            pending.setdefault(entry.shard, {})[entry.row_index] = entry

    audio_dir.mkdir(parents=True, exist_ok=True)
    for shard in sorted(pending):
        wanted = pending[shard]
        for row_index, row in enumerate(shard_rows(shard)):
            entry = wanted.get(row_index)
            if entry is None:
                continue
            name = resolve_audio_filename(row)
            if name != entry.audio_filename or int(row["reciter_id"]) != entry.reciter_id:
                raise ValueError(
                    f"shard {shard} row {row_index} is {name} (reciter "
                    f"{row['reciter_id']}), but the index says {entry.audio_filename} "
                    f"(reciter {entry.reciter_id})"
                )
            clip = _stage_row(row, entry, audio_dir, requests[name])
            previous = staged.get(name)
            if previous is not None:
                if previous.audio_sha256 != clip.audio_sha256:
                    raise ValueError(
                        f"{name} re-staged with sha256 {clip.audio_sha256}, but the "
                        f"registry records {previous.audio_sha256}"
                    )
                clip = replace(clip, uses=tuple(sorted(set(previous.uses) | set(clip.uses))))
            staged[name] = clip
        missed = sorted(e.audio_filename for e in wanted.values() if e.audio_filename not in staged)
        if missed:
            raise ValueError(f"shard {shard} ended before rows for {missed[:3]}")
        if on_shard_done is not None:
            on_shard_done(staged)
    return staged, unlocatable


def _stage_row(row: dict, entry: IndexRow, audio_dir: Path, uses: frozenset[str]) -> StagedClip:
    """Decode one row to 16 kHz mono, write it as PCM_16 WAV and record its provenance."""
    import soundfile as sf

    from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k

    waveform = decode_to_mono_16k(row["audio"]["bytes"])
    path = audio_dir / entry.audio_filename
    sf.write(path, waveform, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return StagedClip(
        audio_filename=entry.audio_filename,
        shard=entry.shard,
        row_index=entry.row_index,
        reciter_id=entry.reciter_id,
        surah_ayah=entry.surah_ayah,
        num_samples=sf.info(path).frames,
        audio_sha256=audio_sha256(path),
        uses=tuple(sorted(uses)),
    )


def labelled_clip_requests() -> dict[str, frozenset[str]]:
    """The whole clips behind the committed labels: the waqf-boundary truth sites and the
    P3.5 fixtures (whose ``<clip>__seg<n>.wav`` ids name a segment of a whole clip)."""
    from .eval_fixtures import load_should_accept, load_should_reject
    from .segment_score import parse_segment_id
    from .truth_sites import TRUTH_SITES_DIR, load_truth_sites

    requests: dict[str, set[str]] = {}
    for site in load_truth_sites(TRUTH_SITES_DIR / "waqf_boundaries.jsonl"):
        requests.setdefault(site.audio_filename, set()).add(WAQF_BOUNDARY)
    for entry in [*load_should_accept(), *load_should_reject()]:
        clip, _ = parse_segment_id(entry.clip_id)
        requests.setdefault(clip, set()).add(P35_FIXTURE)
    return {name: frozenset(uses) for name, uses in requests.items()}


def _pool_requests(path: Path) -> dict[str, frozenset[str]]:
    with open(path, encoding="utf-8") as f:
        names = [json.loads(raw)["audio_filename"] for raw in f if raw.strip()]
    return {name: frozenset({MINING_POOL}) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="index shards' rows without their audio")
    index.add_argument("--shards", required=True, help="shard spec, e.g. '0-20,39,58'")
    index.add_argument("--out", type=Path, required=True)

    stage = commands.add_parser("stage", help="re-stage the labelled clips and the pool")
    stage.add_argument("--index", type=Path, required=True)
    stage.add_argument("--pool-selection", type=Path, default=None,
                       help="JSONL of the mining pool's clips (tadabur.mining_pool select)")
    stage.add_argument("--audio-dir", type=Path, required=True)
    stage.add_argument("--shard-cache", type=Path, required=True,
                       help="a cache directory of this run's own; each shard is deleted "
                       "from it once staged")
    stage.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()

    from .shard_reader import iter_shard_metadata, iter_shard_rows, parse_shard_spec

    if args.command == "index":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for row in iter_shard_metadata(parse_shard_spec(args.shards)):
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"Indexed shards {args.shards} into {args.out}")
        return

    requests = labelled_clip_requests()
    if args.pool_selection is not None:
        for name, uses in _pool_requests(args.pool_selection).items():
            requests[name] = requests.get(name, frozenset()) | uses
    registry = load_staged_clips(args.registry) if args.registry.exists() else {}

    def shard_rows(shard: int) -> Iterator[dict]:
        return iter_shard_rows(
            [shard], cache_dir=args.shard_cache, delete_after=True,
            columns=["audio", "reciter_id"],
        )

    def checkpoint(staged: dict[str, StagedClip]) -> None:
        write_staged_clips(staged.values(), args.registry)
        print(f"  {len(staged)} clips staged so far", flush=True)

    staged, unlocatable = stage_clips(
        requests, read_shard_index(args.index), args.audio_dir, shard_rows, registry,
        on_shard_done=checkpoint,
    )
    write_staged_clips(staged.values(), args.registry)
    print(f"Staged {len(staged)} clips into {args.audio_dir}; registry {args.registry}")
    if unlocatable:
        print(f"{len(unlocatable)} requested clips are in no indexed shard:")
        for name in unlocatable:
            print(f"  {name}")


if __name__ == "__main__":
    main()

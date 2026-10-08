"""The mining pool: a fixed set of whole clips from shards ``h448`` never trained on (#83).

The diagnostics (#85, #86) and the listening session's site mining (#87) all need the same
thing: real recitations the shipped student has never seen, re-staged with provenance,
segmented into waqf segments with realized references and word times, and decoded once by
the frozen base teacher. This module fixes **which clips** (:func:`select_pool`) and
writes the committed manifest of what they became (:func:`build_manifest`); the audio
stays on the GPU box and is re-stageable from :mod:`tadabur.staged_audio`.

Selection is a pure function of the shard index:

* **Frame** -- the strided reserve (:func:`training.decode_evalset.gate_eval_shards`)
  past the contiguous held-out block 0-20, so shard 20, which earlier runs used heavily,
  is not drawn twice. Of those rows, the eligible ones are: not in the frozen
  ``decode_evalset`` (its dev half is the teacher-agreement guard, so a pool clip that
  later feeds training cannot leak into it), 1.5-50 s long (the evalset's bounds, which
  match the staging filter's cap), and on an ayah the phonetizer can realize
  (``generate_phonemes.FALLBACK_PHONEMES`` lists the eight it cannot).
* **Reciter-balanced, reciter-contiguous** -- reciters are ranked by a salted hash of the
  canonical ``reciter_id``, clips within a reciter by a salted hash of the filename, and
  the pool takes up to :data:`PER_RECITER_CAP` clips from each reciter in rank order
  until :data:`POOL_SIZE` clips are drawn. The cap keeps one prolific reciter (one holds
  ~800 of the reserve's clips) from dominating a reciter-clustered interval; taking whole
  reciters in rank order leaves every lower-ranked reciter untouched, so the sealed panel
  (#89) still has reciters that no part of this work has used.

Usage (from ``tools/``)::

  python -m tadabur.mining_pool select --index stage/shard_index.jsonl \\
      --evalset-manifest tadabur/gate_eval/manifest.json --out stage/pool_selection.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from . import phoneme_sifat
from .staged_audio import IndexRow, read_shard_index

MINING_POOL_DIR = Path(__file__).parent / "mining_pool"
SELECTION_PATH = MINING_POOL_DIR / "selection.json"

#: About 2,000 clips (#83): enough for #87's ~200 haraka and ~60 geminate sites many
#: times over; the summary's capacity counts show what each stratum actually holds.
POOL_SIZE = 2000
#: At most this many clips per reciter. With ~500 eligible reciters this draws ~380 of
#: them, leaving ~160 reciters (none in any truth site) for the sealed panel.
PER_RECITER_CAP = 8
SALT = "issue-83-mining-pool-v1"
#: Shards ``h448`` never trained on besides the strided reserve: the block 0-20.
HELD_OUT_BLOCK = range(0, 21)

_EVALSET_FILENAME = re.compile(r"tadabur_sh(\d{3})_i(\d{5})_")


def pool_shards() -> list[int]:
    """The strided reserve shards the pool is drawn from."""
    from training.decode_evalset import gate_eval_shards

    return [shard for shard in gate_eval_shards() if shard not in HELD_OUT_BLOCK]


def read_evalset_rows(manifest_path: Path) -> set[tuple[int, int]]:
    """``(shard, row_index)`` of every clip in a ``decode_evalset`` manifest.

    The manifest names each clip ``tadabur_sh<shard>_i<row>_...`` (its
    ``_clip_filename``); the shard is also checked against its own field.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: set[tuple[int, int]] = set()
    for clip in manifest["clips"]:
        match = _EVALSET_FILENAME.match(clip["filename"])
        if match is None or int(match.group(1)) != clip["shard"]:
            raise ValueError(f"{manifest_path}: cannot read {clip['filename']!r}")
        rows.add((int(match.group(1)), int(match.group(2))))
    return rows


def _rank(text: str) -> str:
    return hashlib.sha256(f"{SALT}:{text}".encode("utf-8")).hexdigest()


def eligible(
    rows: Iterable[IndexRow], evalset_rows: set[tuple[int, int]]
) -> tuple[list[IndexRow], Counter]:
    """The pool's frame, and why every other row in the pool shards was left out."""
    import generate_phonemes
    from training.decode_evalset import MAX_CLIP_SECONDS, MIN_CLIP_SECONDS

    shards = set(pool_shards())
    kept: list[IndexRow] = []
    excluded: Counter = Counter()
    for row in rows:
        if row.shard not in shards:
            continue
        if (row.shard, row.row_index) in evalset_rows:
            excluded["in_decode_evalset"] += 1
        elif not MIN_CLIP_SECONDS <= row.duration_s <= MAX_CLIP_SECONDS:
            excluded["duration"] += 1
        elif row.surah_ayah in generate_phonemes.FALLBACK_PHONEMES:
            excluded["phonetizer_unsupported"] += 1
        else:
            kept.append(row)
    return kept, excluded


def select_pool(
    rows: Iterable[IndexRow], size: int = POOL_SIZE, cap: int = PER_RECITER_CAP
) -> list[IndexRow]:
    """Up to ``cap`` clips per reciter, reciters in salted-hash order, until ``size``.

    Deterministic in the rows' content, never their order. The last reciter drawn may
    contribute fewer than ``cap`` clips so the pool is exactly ``size`` when the frame
    allows; otherwise every capped clip is taken.
    """
    by_reciter: dict[int, list[IndexRow]] = {}
    for row in rows:
        by_reciter.setdefault(row.reciter_id, []).append(row)
    pool: list[IndexRow] = []
    for reciter in sorted(by_reciter, key=lambda r: _rank(f"reciter:{r}")):
        clips = sorted(by_reciter[reciter], key=lambda r: _rank(f"clip:{r.audio_filename}"))
        pool.extend(clips[: min(cap, size - len(pool))])
        if len(pool) == size:
            break
    return pool


# --- the committed manifest ------------------------------------------------------------

CLIPS_PATH = MINING_POOL_DIR / "clips.jsonl"
DECODES_PATH = MINING_POOL_DIR / "base_decodes.json"
SUMMARY_PATH = MINING_POOL_DIR / "summary.json"


@dataclass(frozen=True)
class PoolSegment:
    """One waqf segment of a pool clip: its word range, its exact sample span in the
    staged clip, its realized reference with per-word offsets into it, and whether
    :mod:`tadabur.segment_score`'s drop rules kept it."""

    segment_index: int
    word_start: int
    word_end: int
    start_sample: int
    end_sample: int
    reference: str
    raw_word_offsets: tuple[int, ...]
    kept: bool


@dataclass(frozen=True)
class PoolClip:
    """One pool clip as segmented: the :class:`tadabur.clip_status.ClipStatus` fields
    (word times included) and every segment. Provenance (shard, row, checksum, length)
    is the clip's row in the staged-clip registry."""

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    n_words: int
    skip_reason: str | None
    re_reads: int
    recited_words: int | None
    recitation_start_s: float
    recitation_end_s: float
    word_times: tuple[float, ...]
    segments: tuple[PoolSegment, ...]


def segment_key(audio_filename: str, segment_index: int) -> str:
    """The key a segment's decode is stored under in ``base_decodes.json``."""
    return f"{audio_filename}#{segment_index}"


def build_manifest(
    statuses: list[dict], segmentation: list[dict], staged: dict
) -> list[PoolClip]:
    """The pool manifest from :mod:`tadabur.resegment`'s ``clip_status`` and
    ``segmentation`` rows. Every clip must be staged for the pool; segment spans are
    converted to samples by the same rule that sliced them for decoding."""
    from .segment_score import segment_sample_bounds
    from .staged_audio import MINING_POOL

    segments_of = {row["audio_filename"]: row["segments"] for row in segmentation}
    clips = []
    for status in sorted(statuses, key=lambda s: s["audio_filename"]):
        name = status["audio_filename"]
        clip = staged.get(name)
        if clip is None or MINING_POOL not in clip.uses:
            raise ValueError(f"{name} is not staged for the mining pool")
        segments = []
        for seg in segments_of.get(name, []):
            start, end = segment_sample_bounds(clip.num_samples, seg["start_s"], seg["end_s"])
            segments.append(
                PoolSegment(
                    segment_index=seg["segment_index"],
                    word_start=seg["word_start"],
                    word_end=seg["word_end"],
                    start_sample=start,
                    end_sample=end,
                    reference=seg["reference"],
                    raw_word_offsets=tuple(seg["raw_word_offsets"]),
                    kept=seg["kept"],
                )
            )
        clips.append(
            PoolClip(
                audio_filename=name,
                surah_ayah=status["surah_ayah"],
                reciter_id=status["reciter_id"],
                n_words=status["n_words"],
                skip_reason=status["skip_reason"],
                re_reads=status["re_reads"],
                recited_words=status["recited_words"],
                recitation_start_s=status["recitation_start_s"],
                recitation_end_s=status["recitation_end_s"],
                word_times=tuple(status["word_times"]),
                segments=tuple(segments),
            )
        )
    return clips


def write_manifest(clips: list[PoolClip], path: Path = CLIPS_PATH) -> None:
    """One clip per line, sorted by filename, keys sorted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for clip in sorted(clips, key=lambda c: c.audio_filename):
            f.write(json.dumps(asdict(clip), ensure_ascii=False, sort_keys=True) + "\n")


def load_manifest(path: Path = CLIPS_PATH, registry: dict | None = None) -> list[PoolClip]:
    """The committed pool, checked against the staged-clip registry: every clip staged
    for the pool with the same reciter and ayah, and every segment span inside it."""
    from .staged_audio import MINING_POOL, load_staged_clips

    staged = load_staged_clips() if registry is None else registry
    clips = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            data["word_times"] = tuple(data["word_times"])
            data["segments"] = tuple(
                PoolSegment(**{**seg, "raw_word_offsets": tuple(seg["raw_word_offsets"])})
                for seg in data["segments"]
            )
            clip = PoolClip(**data)
            entry = staged.get(clip.audio_filename)
            where = f"{path}:{lineno}"
            if entry is None or MINING_POOL not in entry.uses:
                raise ValueError(f"{where}: {clip.audio_filename} is not staged for the pool")
            if (entry.reciter_id, entry.surah_ayah) != (clip.reciter_id, clip.surah_ayah):
                raise ValueError(f"{where}: reciter or ayah disagrees with the registry")
            for seg in clip.segments:
                if not 0 <= seg.start_sample <= seg.end_sample <= entry.num_samples:
                    raise ValueError(f"{where}: segment {seg.segment_index} span is outside")
            clips.append(clip)
    return clips


def load_base_decodes(path: Path = DECODES_PATH) -> tuple[dict, dict[str, str]]:
    """The base teacher's decode of every kept pool segment, keyed by
    :func:`segment_key`, and the :class:`~training.decoding.DecodeFingerprint` (as a
    dict) they were made under."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["decode_fingerprint"], data["decodes"]


#: The pairs the listening session mines (acceptance rules §7): the six soft pairs and
#: ذ↔ظ, as ``a↔b`` labels in codepoint order.
TARGET_PAIRS = tuple(sorted(phoneme_sifat.soft_pair_contrasts() | {"\u0630\u2194\u0638"}))


def capacity(clips: list[PoolClip], decodes: dict[str, str]) -> dict:
    """How many candidate sites each listening-session stratum holds in the pool.

    Counted on kept segments from the base decode against the realized reference, so it
    sizes mining and says nothing about truth: per haraka, the reference harakat the base
    decode left empty (``omitted``) and matched; for shaddah, the reference geminates and
    the gemination mismatches by direction; per target pair, the reference carriers of
    either letter and the sites where the base decode has the other letter; and the
    mid-word consonants with no haraka after them (the prescribed-sukun stratum's
    population, a geminate's first half excluded).
    """
    from training.tashkeel_eval import MATCHED, OMITTED, vowel_sites
    from training.tashkeel_worklist import VOWEL_NAMES

    from .contrast_attribution import ADDED, DROPPED, SHADDA_CONTRAST, contrast_sites
    from .truth_sites import CONSONANTS, HARAKA_CHARS

    harakat = set(HARAKA_CHARS.values())
    haraka = {name: Counter() for name in sorted(HARAKA_CHARS)}
    shaddah: Counter = Counter()
    pairs = {pair: Counter() for pair in TARGET_PAIRS}
    sukun_mid_word = 0
    for clip in clips:
        for seg in clip.segments:
            if not seg.kept:
                continue
            reference, decode = seg.reference, decodes[segment_key(clip.audio_filename, seg.segment_index)]
            for site in vowel_sites(decode, reference):
                if site.outcome in (MATCHED, OMITTED) and site.reference_vowel in VOWEL_NAMES:
                    haraka[VOWEL_NAMES[site.reference_vowel]][site.outcome] += 1
            for site in contrast_sites(decode, reference, SHADDA_CONTRAST):
                shaddah[{DROPPED: "base_single_at_geminate", ADDED: "base_double_at_single"}[site.change]] += 1
            for i, char in enumerate(reference[:-1]):
                following = reference[i + 1]
                if char not in CONSONANTS:
                    continue
                if following == char:
                    shaddah["reference_geminates"] += 1
                elif following in CONSONANTS and (i == 0 or reference[i - 1] != char):
                    sukun_mid_word += 1
            for pair in TARGET_PAIRS:
                letters = set(pair.split("\u2194"))
                pairs[pair]["reference_carriers"] += sum(c in letters for c in reference)
                pairs[pair]["base_other_letter"] += len(contrast_sites(decode, reference, pair))
    return {
        "haraka": {name: dict(sorted(c.items())) for name, c in haraka.items()},
        "shaddah": dict(sorted(shaddah.items())),
        "sukun_mid_word_carriers": sukun_mid_word,
        "pairs": {pair: dict(sorted(c.items())) for pair, c in pairs.items()},
    }


def summarize(clips: list[PoolClip], staged: dict, capacity_counts: dict, run: dict) -> dict:
    """Counts per shard and reciter, segmentation outcomes, staged size, and capacity."""
    entries = [staged[c.audio_filename] for c in clips]
    segments = [seg for c in clips for seg in c.segments]
    per_reciter = Counter(c.reciter_id for c in clips)
    return {
        "clips": len(clips),
        "reciters": len(per_reciter),
        "clips_per_shard": {str(k): v for k, v in sorted(Counter(e.shard for e in entries).items())},
        "clips_per_reciter": {str(k): v for k, v in sorted(per_reciter.items())},
        "clips_per_reciter_histogram": {
            str(k): v for k, v in sorted(Counter(per_reciter.values()).items())
        },
        "audio_hours": round(sum(e.num_samples for e in entries) / 16000 / 3600, 2),
        "clip_skip_reasons": dict(sorted(Counter(c.skip_reason or "none" for c in clips).items())),
        "segments": len(segments),
        "segments_kept": sum(seg.kept for seg in segments),
        "decode_fingerprint": run["decode_fingerprint"],
        "pausal_taa_marbuta": run["pausal_taa_marbuta"],
        "vad": run["vad"],
        "capacity": capacity_counts,
    }


def _build(seg_dir: Path, registry_path: Path | None) -> None:
    from .staged_audio import load_staged_clips

    def rows(name: str) -> list[dict]:
        with open(seg_dir / name, encoding="utf-8") as f:
            return [json.loads(raw) for raw in f if raw.strip()]

    staged = load_staged_clips() if registry_path is None else load_staged_clips(registry_path)
    run = json.loads((seg_dir / "run.json").read_text(encoding="utf-8"))
    clips = build_manifest(rows("clip_status.jsonl"), rows("segmentation.jsonl"), staged)
    decodes = {
        segment_key(row["clip_audio_filename"], row["segment_index"]): row["predicted_phonemes"]
        for row in rows("segment_manifest.jsonl")
    }
    write_manifest(clips)
    DECODES_PATH.write_text(
        json.dumps(
            {"decode_fingerprint": run["decode_fingerprint"], "decodes": decodes},
            ensure_ascii=False, indent=0, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    load_manifest(registry=staged)  # what was written must load against the registry
    summary = summarize(clips, staged, capacity(clips, decodes), run)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "clips_per_reciter"},
                     indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="draw the pool from a shard index")
    select.add_argument("--index", type=Path, required=True)
    select.add_argument("--evalset-manifest", type=Path, required=True,
                        help="the frozen decode_evalset manifest.json whose clips to avoid")
    select.add_argument("--out", type=Path, required=True)
    build = commands.add_parser("build", help="write the committed pool manifest")
    build.add_argument("--seg-dir", type=Path, required=True,
                       help="output of `tadabur.resegment --use mining_pool`")
    build.add_argument("--registry", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "build":
        _build(args.seg_dir, args.registry)
        return


    frame, excluded = eligible(
        read_shard_index(args.index).values(), read_evalset_rows(args.evalset_manifest)
    )
    pool = select_pool(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in sorted(pool, key=lambda r: r.audio_filename):
            f.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    selection = {
        "salt": SALT,
        "pool_size": POOL_SIZE,
        "per_reciter_cap": PER_RECITER_CAP,
        "shards": pool_shards(),
        "frame_clips": len(frame),
        "frame_reciters": len({r.reciter_id for r in frame}),
        "excluded": dict(sorted(excluded.items())),
        "decode_evalset_manifest_sha256": hashlib.sha256(
            args.evalset_manifest.read_bytes()
        ).hexdigest(),
        "pool_clips": len(pool),
        "pool_reciters": len({r.reciter_id for r in pool}),
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

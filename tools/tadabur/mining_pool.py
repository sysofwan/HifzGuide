"""The mining pool: a fixed set of whole clips from shards ``h448`` never trained on (#83).

The diagnostics (#85, #86) and the listening session's site mining (#87) all need the same
thing: real recitations the shipped student has never seen, re-staged with provenance,
segmented into waqf segments with realized references and word times, and decoded once by
the frozen base teacher. This module fixes **which clips** and writes the committed
manifest of what they became; the audio stays on the GPU box and is re-stageable from
:mod:`tadabur.staged_audio`.

**The frame** is the strided reserve (:func:`training.decode_evalset.gate_eval_shards`)
past the contiguous held-out block 0-20, so shard 20, which earlier runs used heavily, is
not drawn twice. Of those rows, the eligible ones are: not in the frozen ``decode_evalset``
(its dev half is the teacher-agreement guard, so a pool clip that later feeds training
cannot leak into it), 1.5-50 s long (the evalset's bounds, which match the staging
filter's cap), and on an ayah the phonetizer can realize
(``generate_phonemes.FALLBACK_PHONEMES`` lists the eight it cannot).

**Stratum ``uniform``** (:func:`select_pool`) is reciter-balanced and reciter-contiguous:
reciters are ranked by a salted hash of the canonical ``reciter_id``, clips within a
reciter by a salted hash of the filename, and up to :data:`PER_RECITER_CAP` clips are
taken from each reciter in rank order until :data:`POOL_SIZE`. The cap keeps one prolific
reciter (one holds ~800 of the reserve's clips) from dominating a reciter-clustered
interval; taking whole reciters in rank order leaves every lower-ranked reciter untouched,
so the sealed panel (#89) still has reciters no part of this work has used.

**Strata ``consonant_pair`` and ``geminate``** are censuses. The uniform draw holds too
few of the rare events #87 mines (on the 2,000 uniform clips the base teacher heard the
other letter of a target pair at 21 sites, none for ``ذ↔ظ``, and left a geminate single at
56), so every frame clip **of the drawn reciters** is decoded whole by the base teacher
(:func:`scan_events`, ``scan``) and every clip with a pair substitution or a gemination
mismatch joins the pool. The census is complete within that sub-frame, so each stratum's
population is its clip count (weight 1), and no reciter outside the uniform draw is
touched.

Usage (from ``tools/``; ``scan`` downloads the 19 pool shards again, one at a time)::

  python -m tadabur.mining_pool select --index stage/shard_index.jsonl \\
      --evalset-manifest tadabur/gate_eval/manifest.json --out stage/pool_uniform.jsonl
  python -m tadabur.mining_pool scan --index stage/shard_index.jsonl \\
      --evalset-manifest tadabur/gate_eval/manifest.json --uniform stage/pool_uniform.jsonl \\
      --out stage/scan.jsonl --shard-cache stage/hf_cache
  python -m tadabur.mining_pool select ... --scan stage/scan.jsonl --out stage/pool_selection.jsonl
  python -m tadabur.mining_pool build --selection stage/pool_selection.jsonl --seg-dir stage/seg_pool
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
from .contrast_attribution import ADDED, DROPPED, SHADDA_CONTRAST, contrast_sites
from .staged_audio import IndexRow, read_shard_index

MINING_POOL_DIR = Path(__file__).parent / "mining_pool"
SELECTION_PATH = MINING_POOL_DIR / "selection.json"
FRAME_PATH = MINING_POOL_DIR / "frame.json"
CLIPS_PATH = MINING_POOL_DIR / "clips.jsonl"
DECODES_PATH = MINING_POOL_DIR / "base_decodes.json"
SUMMARY_PATH = MINING_POOL_DIR / "summary.json"

#: The uniform stratum's size (#83: "about 2,000 clips").
POOL_SIZE = 2000
#: At most this many uniform clips per reciter: the draw takes 394 of the frame's 500
#: reciters.
PER_RECITER_CAP = 8
SALT = "issue-83-mining-pool-v1"
#: Shards ``h448`` never trained on besides the strided reserve: the block 0-20.
HELD_OUT_BLOCK = range(0, 21)

#: Why a clip is in the pool (module docstring). A clip may be in several.
UNIFORM = "uniform"
CONSONANT_PAIR = "consonant_pair"
GEMINATE = "geminate"
STRATA = (UNIFORM, CONSONANT_PAIR, GEMINATE)

#: The pairs the listening session mines (acceptance rules §7): the six soft pairs and
#: ذ↔ظ, as ``a↔b`` labels in codepoint order.
TARGET_PAIRS = tuple(sorted(phoneme_sifat.soft_pair_contrasts() | {"ذ↔ظ"}))

_EVALSET_FILENAME = re.compile(r"tadabur_sh(\d{3})_i(\d{5})_")


# --- the draw --------------------------------------------------------------------------


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
) -> tuple[list[IndexRow], list[tuple[IndexRow, str]]]:
    """The pool's frame, and every other row in the pool shards with why it was left out."""
    import generate_phonemes
    from training.decode_evalset import MAX_CLIP_SECONDS, MIN_CLIP_SECONDS

    shards = set(pool_shards())
    kept: list[IndexRow] = []
    excluded: list[tuple[IndexRow, str]] = []
    for row in rows:
        if row.shard not in shards:
            continue
        if (row.shard, row.row_index) in evalset_rows:
            excluded.append((row, "in_decode_evalset"))
        elif not MIN_CLIP_SECONDS <= row.duration_s <= MAX_CLIP_SECONDS:
            excluded.append((row, "duration"))
        elif row.surah_ayah in generate_phonemes.FALLBACK_PHONEMES:
            excluded.append((row, "phonetizer_unsupported"))
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


def read_selection(path: Path) -> list[IndexRow]:
    """The clips a ``select`` output lists (its ``strata`` field, if any, is ignored)."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(raw) for raw in f if raw.strip()]
    return [IndexRow(**{k: row[k] for k in IndexRow.__dataclass_fields__}) for row in rows]


def census_frame(frame: Iterable[IndexRow], uniform: Iterable[IndexRow]) -> list[IndexRow]:
    """The frame clips of the reciters the uniform draw took: what ``scan`` decodes."""
    reciters = {row.reciter_id for row in uniform}
    return sorted((r for r in frame if r.reciter_id in reciters), key=lambda r: r.audio_filename)


def scan_events(decode: str, reference: str) -> dict:
    """The rare events in one whole-clip base decode against the clip's reference: per
    target pair, the sites where the decode has the other letter, and the gemination
    mismatches by direction (:func:`tadabur.contrast_attribution.contrast_sites`)."""
    pairs = {}
    for pair in TARGET_PAIRS:
        if found := len(contrast_sites(decode, reference, pair)):
            pairs[pair] = found
    changes = Counter(site.change for site in contrast_sites(decode, reference, SHADDA_CONTRAST))
    return {"pairs": pairs, "shaddah": dict(sorted(changes.items()))}


def event_strata(events: dict) -> tuple[str, ...]:
    """The census strata a scanned clip's events put it in."""
    hits = ((CONSONANT_PAIR, events["pairs"]), (GEMINATE, events["shaddah"]))
    return tuple(stratum for stratum, found in hits if found)


def stratify(
    uniform: list[IndexRow], census: list[IndexRow], scanned: dict[str, dict]
) -> dict[str, tuple[str, ...]]:
    """Every pool clip and its strata: the uniform draw, plus each census clip whose scan
    found an event. Refuses a scan that does not cover the census frame exactly."""
    expected = {row.audio_filename for row in census}
    if set(scanned) != expected:
        raise ValueError(
            f"the scan covers {len(scanned)} clips but the census frame holds "
            f"{len(expected)} ({len(expected - set(scanned))} unscanned)"
        )
    strata: dict[str, set[str]] = {row.audio_filename: {UNIFORM} for row in uniform}
    for name, events in scanned.items():
        for stratum in event_strata(events):
            strata.setdefault(name, set()).add(stratum)
    return {name: tuple(s for s in STRATA if s in found) for name, found in sorted(strata.items())}


def inclusion_probabilities(
    frame: list[IndexRow], uniform: list[IndexRow], strata: dict[str, tuple[str, ...]]
) -> dict[str, float]:
    """Each pool clip's inclusion probability, **conditional on the drawn reciters**.

    Within a drawn reciter the uniform stratum takes ``k`` of the reciter's ``n`` frame
    clips by a salted hash, a simple random sample of them: ``k / n``. A census stratum
    takes every clip of the drawn reciters with an event, so a clip in one has
    probability 1 whatever its uniform chance (the union counts the overlap once). The
    reciters themselves are the first 394 of 500 in salted-hash order; an analysis that
    reaches past them treats that as a simple random sample of reciters
    (``frame.json`` records both counts).
    """
    eligible_clips = Counter(row.reciter_id for row in frame)
    drawn = Counter(row.reciter_id for row in uniform)
    reciter = {row.audio_filename: row.reciter_id for row in frame}
    return {
        name: 1.0 if set(found) - {UNIFORM}
        else drawn[reciter[name]] / eligible_clips[reciter[name]]
        for name, found in strata.items()
    }


def frame_record(
    frame: list[IndexRow], excluded: list[tuple[IndexRow, str]], uniform: list[IndexRow]
) -> dict:
    """The frame the draw was made from, per shard and per reciter: eligible clips, the
    rows excluded and why, and how many clips the uniform stratum took from each."""
    shards: dict[int, Counter] = {shard: Counter() for shard in pool_shards()}
    for row in frame:
        shards[row.shard]["eligible"] += 1
    for row, reason in excluded:
        shards[row.shard][f"excluded_{reason}"] += 1
    reciters: dict[int, Counter] = {}
    for row in frame:
        reciters.setdefault(row.reciter_id, Counter())["eligible"] += 1
    for row in uniform:
        reciters[row.reciter_id]["uniform"] += 1
    drawn = {row.reciter_id for row in uniform}
    return {
        "per_shard": {str(k): dict(sorted(v.items())) for k, v in sorted(shards.items())},
        "per_reciter": {
            str(k): {**dict(sorted(v.items())), "drawn": k in drawn}
            for k, v in sorted(reciters.items())
        },
        "reciters_eligible": len(reciters),
        "reciters_drawn": len(drawn),
    }


def scan_identity(census: list[IndexRow], fingerprint: dict) -> dict:
    """What a scan's rows are a function of: the census frame (every clip's name, shard
    and row) and the decode fingerprint. Events are not stored, so the phonetizer that
    turns decodes into events is recorded by ``select``, not here."""
    frame = "\n".join(f"{r.audio_filename}\t{r.shard}\t{r.row_index}" for r in census)
    return {
        "census_clips": len(census),
        "census_sha256": hashlib.sha256(frame.encode("utf-8")).hexdigest(),
        "decode_fingerprint": fingerprint,
    }


def _scan_run_path(scan: Path) -> Path:
    return scan.with_suffix(".run.json")


def _scan(args) -> None:
    """Decode every census clip whole with the base teacher; one ``{audio_filename,
    decode}`` row each.

    Each clip's audio is decoded exactly as staging would leave it (16 kHz mono through a
    PCM_16 round trip, :func:`tadabur.staged_audio.as_staged`), so a selected clip's scan
    decode is the whole-clip decode its staged file gives. The scan's identity
    (:func:`scan_identity`) is written beside it when it starts; a resumed scan appends
    only after checking its identity is the stored one, and refuses a row outside the
    census frame, so earlier rows are never relabelled as a different run's.
    """
    from training.decoding import SPANS, Decoder

    from .audio import decode_to_mono_16k
    from .resegment import BASE_TEACHER, DECODE_BATCH_SIZE, WEIGHTS_DTYPE
    from .shard_reader import iter_shard_rows
    from .staged_audio import as_staged

    frame, _ = eligible(
        read_shard_index(args.index).values(), read_evalset_rows(args.evalset_manifest)
    )
    census = census_frame(frame, read_selection(args.uniform))
    decoder = Decoder.load(
        BASE_TEACHER, args.device, weights_dtype=WEIGHTS_DTYPE, batch_size=DECODE_BATCH_SIZE
    )
    identity = scan_identity(census, decoder.fingerprint(SPANS).as_dict())
    run_path = _scan_run_path(args.out)
    done: set[str] = set()
    if args.out.exists():
        if not run_path.exists() or json.loads(run_path.read_text()) != identity:
            raise SystemExit(f"{args.out} was scanned under another identity; refusing to resume")
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(raw)["audio_filename"] for raw in f if raw.strip()}
        if not done <= {row.audio_filename for row in census}:
            raise SystemExit(f"{args.out} holds clips outside the census frame")
    else:
        run_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    wanted: dict[int, dict[int, IndexRow]] = {}
    for row in census:
        if row.audio_filename not in done:
            wanted.setdefault(row.shard, {})[row.row_index] = row
    print(f"census frame {len(census)} clips; {len(done)} already scanned", flush=True)

    with open(args.out, "a", encoding="utf-8") as out:
        for shard in sorted(wanted):
            rows = iter_shard_rows([shard], cache_dir=args.shard_cache, delete_after=True,
                                   columns=["audio", "reciter_id"])
            for row_index, row in enumerate(rows):
                entry = wanted[shard].get(row_index)
                if entry is None:
                    continue
                if row["audio"]["path"] != entry.audio_filename:
                    raise ValueError(f"shard {shard} row {row_index} is not {entry.audio_filename}")
                samples = as_staged(decode_to_mono_16k(row["audio"]["bytes"]))
                (decode,) = decoder.decode_spans([samples])
                record = {"audio_filename": entry.audio_filename, "decode": decode}
                out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            print(f"  shard {shard} scanned", flush=True)


def whole_ayah_references(surah_ayahs: Iterable[str]) -> dict[str, str | None]:
    """Each ayah's realized reference recited whole (wasl inside, waqf at its end), or
    ``None`` where quran-transcript cannot phonetize it."""
    from .waqf_segments import _uthmani_words, hafs_segment_reference

    segment_reference = hafs_segment_reference()
    references: dict[str, str | None] = {}
    for surah_ayah in sorted(set(surah_ayahs)):
        try:
            references[surah_ayah] = segment_reference(_uthmani_words(surah_ayah))[0]
        except (KeyError, IndexError):
            references[surah_ayah] = None
    return references


def _select(args) -> None:
    import hafs_phonetizer

    frame, excluded = eligible(
        read_shard_index(args.index).values(), read_evalset_rows(args.evalset_manifest)
    )
    uniform = select_pool(frame)
    selection = {
        "salt": SALT,
        "pool_size": POOL_SIZE,
        "per_reciter_cap": PER_RECITER_CAP,
        "shards": pool_shards(),
        "frame_clips": len(frame),
        "frame_reciters": len({r.reciter_id for r in frame}),
        "excluded": dict(sorted(Counter(reason for _, reason in excluded).items())),
        "decode_evalset_manifest_sha256": hashlib.sha256(
            args.evalset_manifest.read_bytes()
        ).hexdigest(),
        "uniform_clips": len(uniform),
        "uniform_reciters": len({r.reciter_id for r in uniform}),
    }
    by_name = {row.audio_filename: row for row in frame}
    if args.scan is None:
        strata = {row.audio_filename: (UNIFORM,) for row in uniform}
    else:
        census = census_frame(frame, uniform)
        identity = json.loads(_scan_run_path(args.scan).read_text(encoding="utf-8"))
        if {k: v for k, v in scan_identity(census, {}).items() if k != "decode_fingerprint"} \
                != {k: v for k, v in identity.items() if k != "decode_fingerprint"}:
            raise SystemExit(f"{args.scan} was not scanned over this census frame")
        with open(args.scan, encoding="utf-8") as f:
            decodes = {r["audio_filename"]: r["decode"] for r in map(json.loads, f)}
        references = whole_ayah_references(by_name[n].surah_ayah for n in decodes)
        scanned = {}
        for name, decode in sorted(decodes.items()):
            reference = references[by_name[name].surah_ayah]
            scanned[name] = (
                {"pairs": {}, "shaddah": {}} if reference is None else scan_events(decode, reference)
            )
        strata = stratify(uniform, census, scanned)
        populations = Counter(s for found in strata.values() for s in found)
        events = Counter()
        for found in scanned.values():
            events.update(found["pairs"])
            events.update({f"shaddah_{k}": v for k, v in found["shaddah"].items()})
        selection["census"] = {
            "frame_clips": len(census),
            "clips_without_reference": sum(
                references[by_name[n].surah_ayah] is None for n in decodes
            ),
            "scan": identity,
            "phonetizer_revision": hafs_phonetizer.REVISION,
            "events": dict(sorted(events.items())),
            "stratum_clips": {s: populations[s] for s in STRATA},
        }
    selection["pool_clips"] = len(strata)
    selection["pool_reciters"] = len({by_name[n].reciter_id for n in strata})
    probabilities = inclusion_probabilities(frame, uniform, strata)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for name, found in strata.items():
            row = {**asdict(by_name[name]), "strata": list(found),
                   "inclusion_probability": probabilities[name]}
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(
        json.dumps(selection, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    FRAME_PATH.write_text(
        json.dumps(frame_record(frame, excluded, uniform), indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, sort_keys=True, ensure_ascii=False))


# --- the committed manifest ------------------------------------------------------------


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
    """One pool clip as segmented: the strata it was drawn in and its inclusion
    probability given the drawn reciters (:func:`inclusion_probabilities`), the
    :class:`tadabur.clip_status.ClipStatus` fields (word times included) and every
    segment. Provenance (shard, row, checksum, length) is the clip's row in the
    staged-clip registry."""

    audio_filename: str
    strata: tuple[str, ...]
    inclusion_probability: float
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


@dataclass(frozen=True)
class Selected:
    """Why one clip is in the pool, as the selection recorded it."""

    strata: tuple[str, ...]
    inclusion_probability: float


def read_selected(path: Path) -> dict[str, Selected]:
    """A ``select`` output's strata and inclusion probability per clip."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(raw) for raw in f if raw.strip()]
    return {
        row["audio_filename"]: Selected(tuple(row["strata"]), row["inclusion_probability"])
        for row in rows
    }


def build_manifest(
    statuses: list[dict],
    segmentation: list[dict],
    selected: dict[str, Selected],
    staged: dict,
) -> list[PoolClip]:
    """The pool manifest from :mod:`tadabur.resegment`'s ``clip_status`` and
    ``segmentation`` rows and the selection. Every selected clip must have been
    segmented and staged for the pool; segment spans are converted to samples by the
    same rule that sliced them for decoding."""
    from .segment_score import segment_sample_bounds
    from .staged_audio import MINING_POOL

    by_name = {status["audio_filename"]: status for status in statuses}
    if set(by_name) != set(selected):
        raise ValueError("the segmentation and the selection name different clips")
    segments_of = {row["audio_filename"]: row["segments"] for row in segmentation}
    clips = []
    for name in sorted(by_name):
        status, clip = by_name[name], staged.get(name)
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
                strata=selected[name].strata,
                inclusion_probability=selected[name].inclusion_probability,
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
    for the pool with the same reciter and ayah, in known strata, and every segment span
    inside it."""
    from .staged_audio import MINING_POOL, load_staged_clips

    staged = load_staged_clips() if registry is None else registry
    clips = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            data["strata"] = tuple(data["strata"])
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
            if not clip.strata or not set(clip.strata) <= set(STRATA):
                raise ValueError(f"{where}: strata {clip.strata} are not {STRATA}")
            if not 0 < clip.inclusion_probability <= 1:
                raise ValueError(f"{where}: inclusion_probability is not in (0, 1]")
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

    from .truth_sites import CONSONANTS, HARAKA_CHARS

    haraka = {name: Counter() for name in sorted(HARAKA_CHARS)}
    shaddah: Counter = Counter()
    pairs = {pair: Counter() for pair in TARGET_PAIRS}
    sukun_mid_word = 0
    directions = {DROPPED: "base_single_at_geminate", ADDED: "base_double_at_single"}
    for clip in clips:
        for seg in clip.segments:
            if not seg.kept:
                continue
            reference = seg.reference
            decode = decodes[segment_key(clip.audio_filename, seg.segment_index)]
            for site in vowel_sites(decode, reference):
                if site.outcome in (MATCHED, OMITTED) and site.reference_vowel in VOWEL_NAMES:
                    haraka[VOWEL_NAMES[site.reference_vowel]][site.outcome] += 1
            for site in contrast_sites(decode, reference, SHADDA_CONTRAST):
                shaddah[directions[site.change]] += 1
            for i, char in enumerate(reference[:-1]):
                following = reference[i + 1]
                if char not in CONSONANTS:
                    continue
                if following == char:
                    shaddah["reference_geminates"] += 1
                elif following in CONSONANTS and (i == 0 or reference[i - 1] != char):
                    sukun_mid_word += 1
            for pair in TARGET_PAIRS:
                letters = set(pair.split("↔"))
                pairs[pair]["reference_carriers"] += sum(c in letters for c in reference)
                pairs[pair]["base_other_letter"] += len(contrast_sites(decode, reference, pair))
    return {
        "haraka": {name: dict(sorted(c.items())) for name, c in haraka.items()},
        "shaddah": dict(sorted(shaddah.items())),
        "sukun_mid_word_carriers": sukun_mid_word,
        "pairs": {pair: dict(sorted(c.items())) for pair, c in pairs.items()},
    }


def summarize(clips: list[PoolClip], staged: dict, decodes: dict[str, str], run: dict) -> dict:
    """Counts per shard, reciter and stratum, segmentation outcomes, and the capacity of
    the whole pool and of its uniform stratum alone."""
    entries = [staged[c.audio_filename] for c in clips]
    segments = [seg for c in clips for seg in c.segments]
    per_reciter = Counter(c.reciter_id for c in clips)
    return {
        "clips": len(clips),
        "reciters": len(per_reciter),
        "clips_per_stratum": {s: sum(s in c.strata for c in clips) for s in STRATA},
        "clips_per_shard": {
            str(k): v for k, v in sorted(Counter(e.shard for e in entries).items())
        },
        "clips_per_reciter": {str(k): v for k, v in sorted(per_reciter.items())},
        "clips_per_reciter_histogram": {
            str(k): v for k, v in sorted(Counter(per_reciter.values()).items())
        },
        "audio_hours": round(sum(e.num_samples for e in entries) / 16000 / 3600, 2),
        "clip_skip_reasons": dict(
            sorted(Counter(c.skip_reason or "none" for c in clips).items())
        ),
        "segments": len(segments),
        "segments_kept": sum(seg.kept for seg in segments),
        "decode_fingerprint": run["decode_fingerprint"],
        "phonetizer_revision": run["phonetizer_revision"],
        "vad": run["vad"],
        "capacity": capacity(clips, decodes),
        "capacity_uniform_only": capacity([c for c in clips if UNIFORM in c.strata], decodes),
    }


def _build(selection_path: Path, seg_dir: Path, registry_path: Path | None) -> None:
    from .staged_audio import load_staged_clips

    def rows(path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(raw) for raw in f if raw.strip()]

    staged = load_staged_clips() if registry_path is None else load_staged_clips(registry_path)
    run = json.loads((seg_dir / "run.json").read_text(encoding="utf-8"))
    clips = build_manifest(
        rows(seg_dir / "clip_status.jsonl"), rows(seg_dir / "segmentation.jsonl"),
        read_selected(selection_path), staged,
    )
    decodes = {
        segment_key(row["clip_audio_filename"], row["segment_index"]): row["predicted_phonemes"]
        for row in rows(seg_dir / "segment_manifest.jsonl")
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
    summary = summarize(clips, staged, decodes, run)
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
    for name, text in (("select", "draw the pool from a shard index"),
                       ("scan", "decode the census frame and record its events")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--index", type=Path, required=True)
        command.add_argument("--evalset-manifest", type=Path, required=True,
                             help="the frozen decode_evalset manifest.json whose clips to avoid")
        command.add_argument("--out", type=Path, required=True)
    select = commands.choices["select"]
    select.add_argument("--scan", type=Path, default=None,
                        help="the census scan; without it only the uniform stratum is drawn")
    scan = commands.choices["scan"]
    scan.add_argument("--uniform", type=Path, required=True,
                      help="a `select` output without --scan: the uniform draw")
    scan.add_argument("--shard-cache", type=Path, required=True)
    scan.add_argument("--device", default="cuda")
    build = commands.add_parser("build", help="write the committed pool manifest")
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--seg-dir", type=Path, required=True,
                       help="output of `tadabur.resegment --use mining_pool`")
    build.add_argument("--registry", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "select":
        _select(args)
    elif args.command == "scan":
        _scan(args)
    else:
        _build(args.selection, args.seg_dir, args.registry)


if __name__ == "__main__":
    main()

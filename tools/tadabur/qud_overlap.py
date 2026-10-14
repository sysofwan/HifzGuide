"""Rule out that a pilot QUD mushaf is audio Tadabur or Muaalem already holds (#144).

The models must never have heard the pilot's recordings. Names prove nothing (Tadabur files
reciters under ids, and a recording can be filed under another), so the check is by audio, in
two stages, and every decision line is fixed from **controls** committed before any pilot
recording is compared with Tadabur.

**Stage 1, the timing screen (no audio).** One recording keeps its rhythm whoever aligned it.
For every ayah of a pilot mushaf, its QUD word start times are compared with those of every
Tadabur row of the same ayah (Tadabur's ``metadata.word_alignments``, read from the shards'
metadata column, :func:`tadabur.shard_reader.iter_shard_word_alignments`): after removing the
offset between the two cuts (the median difference), the **timing distance** is the median
absolute residual, in seconds (:func:`timing_distance`). Only ayat of at least
:data:`MIN_WORDS` words, recited without a repeat, whose two word counts agree, are compared.
A pair within the screen's line is a **candidate**.

**Stage 2, the fingerprint.** Each candidate's Tadabur audio and the QUD ayah are fingerprinted
(:mod:`tadabur.qud_fingerprint`) and compared by bit error rate at the best offset. A pair
within the fingerprint's line is a **match**; one match drops the mushaf whole.

**Muaalem.** Every Muaalem reciter is already out by name, so an overlap would be a
misattributed recording. The first row group of each of the 27 Muaalem moshafs' first shard
(the opening of al-Fatiha and al-Baqarah) is fingerprinted against the same ayat of each pilot
mushaf, a QUD ayah slid along each Muaalem segment.

**Controls** (``qud_eval/overlap_controls.json``, written by ``controls`` and committed first;
none involves a pilot mushaf's audio against Tadabur, and no pilot reciter's Tadabur rows):

* *synthetic positives*: staged Tadabur clips re-encoded the way QUD's audio arrives (44.1 kHz
  stereo MP3) with the start cut by 100-500 ms: one recording, so the fingerprint must match;
* *hard negatives*: the same reciter, the same ayah, two recordings (QUD mushafs of one reciter
  in two styles, or two years' taraweeh): the fingerprint must not match;
* *cross-aligner pairs*: staged Tadabur clips of reciters QUD also holds (name screen), each
  against the same ayah in each of that reciter's QUD mushafs. The fingerprint (its line fixed
  by the two sets above) says which pairs are one recording; those are the timing screen's
  positives, the rest of the same reciter its hard negatives, and a different reciter's
  same ayah its plain negatives.

**Decision lines** (fixed here, applied by ``calibrate`` to the controls' scores, and written
with them to ``qud_eval/overlap_rules.json`` before ``screen`` runs):

* fingerprint: halfway between the highest BER of a synthetic positive and the lowest BER of a
  hard negative, valid only if the first is below the second (else the check fails);
* timing screen: :data:`TIMING_MARGIN` times the largest timing distance of a cross-aligner
  positive, so every control positive passes; the share of same-reciter negatives that pass
  too is reported as the screen's false-candidate rate.

Usage (from ``tools/``)::

  python -m tadabur.qud_overlap controls                     # write and commit first
  python -m tadabur.qud_overlap calibrate --clips-dir <staged Tadabur clips>
  python -m tadabur.qud_overlap screen --timings <cache>     # stage 1
  python -m tadabur.qud_overlap confirm --cache-dir <dir>    # stage 2 and Muaalem
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .audio import decode_to_mono_16k, read_audio_bytes
from .dataset_source import canonical_surah_ayah
from .qud_fingerprint import BANDS, FRAME_S, HIGH_HZ, HOP_S, LOW_HZ, MIN_OVERLAP
from .qud_fingerprint import best_match, fingerprint
from .qud_selection import QUD_DIR, SELECTION_PATH, SOURCES_PATH
from .qud_source import QUD_DATASET, QudRow, mp3_sha256, parquet_table
from .qud_text import Unmapped, map_words

CONTROLS_PATH = QUD_DIR / "overlap_controls.json"
RULES_PATH = QUD_DIR / "overlap_rules.json"
SCREEN_PATH = QUD_DIR / "overlap_screen.json"
RESULTS_PATH = QUD_DIR / "overlap_results.json"

#: The pilot (#144, owner approval 2026-10-14), and the fallbacks in the order they are taken.
PILOT = ("abdul_hamid_ghraio_2026_yt", "anas_almiman_yt")
FALLBACKS = {"abdul_hamid_ghraio_2026_yt": "abdul_hamid_ghraio_2025_yt",
             "anas_almiman_yt": "abdulrahman_al_sudais_yt"}
SALT = "qud-overlap-controls-2026-10"
SYNTHETIC_CLIPS = 60
CUT_MS = (100, 500)
#: Same reciter, two recordings that cannot be one: two styles, or two Ramadans' taraweeh.
HARD_NEGATIVE_PAIRS = (
    ("mahmoud_khalil_al_husary_mujawwad_tarteel", "mahmoud_khalil_al_husary_qdc_128k"),
    ("mohammed_siddiq_al_minshawi_mujawwad_mp3quran", "mohammed_siddiq_al_minshawi_mp3quran"),
    ("abdulbasit_abdulsamad_mujawwad_tarteel", "abdulbasit_abdulsamad_tarteel"),
    ("maher_al_muaiqly_qdc", "maher_al_muaiqly_tarteel"),
    ("abdul_hamid_ghraio_2025_yt", "abdul_hamid_ghraio_2026_yt"),
)
HARD_NEGATIVE_AYAT = 25
#: Cross-aligner controls: Tadabur reciters (by name screen) with at least this many staged
#: clips, at most :data:`CROSS_CLIPS` clips each.
CROSS_MIN_STAGED = 20
CROSS_CLIPS = 12
MIN_WORDS = 4
TIMING_MARGIN = 1.5


def salted_rank(*parts: object) -> str:
    return hashlib.sha256(":".join([SALT, *map(str, parts)]).encode()).hexdigest()


# --- timing (pure) -----------------------------------------------------------------------


def timing_distance(qud_starts: Sequence[float], tadabur_starts: Sequence[float]) -> float:
    """The median absolute residual of the word start differences after removing their median
    (the offset between the two cuts), in seconds."""
    if len(qud_starts) != len(tadabur_starts) or len(qud_starts) < MIN_WORDS:
        raise ValueError("timing is compared only on equal word counts of at least MIN_WORDS")
    differences = [t - q for q, t in zip(qud_starts, tadabur_starts)]
    offset = statistics.median(differences)
    return round(statistics.median(abs(d - offset) for d in differences), 4)


def qud_word_starts(row: QudRow, canonical: Sequence[str]) -> list[float] | None:
    """A QUD ayah's word start times (s) when it can be compared: every canonical word
    recited once, in order (no repeat, :func:`tadabur.qud_text.map_words`), one timestamp per
    word, and at least :data:`MIN_WORDS` words."""
    words = row.text_uthmani.split()
    if len(words) < MIN_WORDS or len(words) != len(canonical):
        return None
    try:
        if map_words(words, canonical).repeats:
            return None
    except Unmapped:
        return None
    if [w[0] for w in row.word_timestamps] != list(range(1, len(words) + 1)):
        return None
    return [w[1] / 1000 for w in row.word_timestamps]


# --- the Tadabur timing cache --------------------------------------------------------------


def write_tadabur_timings(path: Path, shards: Iterable[int]) -> None:
    """Every Tadabur row's identity and word start times (gzipped JSON lines), read from the
    shards' metadata columns."""
    from .shard_reader import iter_shard_word_alignments

    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in iter_shard_word_alignments(shards):
            f.write(json.dumps({
                "shard": row["shard"], "row_index": row["row_index"],
                "audio_filename": row["audio_filename"], "reciter_id": row["reciter_id"],
                "surah_ayah": canonical_surah_ayah(row["surah_id"], row["ayah_id"]),
                "starts": [round(w["start"], 3) for w in row["word_alignments"]],
            }, ensure_ascii=False) + "\n")


def read_tadabur_timings(path: Path) -> dict[str, list[dict]]:
    """The timing cache by ``surah:ayah``."""
    by_ayah: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_ayah[row["surah_ayah"]].append(row)
    return by_ayah


# --- the controls (definitions, committed before anything is compared) ---------------------


def _canonical_words(surah_ayah: str) -> list[str]:
    from .waqf_segments import _uthmani_words

    return _uthmani_words(surah_ayah)


def define_controls(selection: Mapping, registry: Mapping, ayah_rows) -> dict:
    """The controls (module docstring), drawn by salted rank. ``ayah_rows(slug)`` gives a
    mushaf's row index by ``surah:ayah``; ``registry`` is the staged-clip registry."""
    from .staged_audio import SEALED_PANEL

    mushafs = {m["slug"]: m for m in selection["mushafs"]}
    pilot_ids = {mushafs[s]["tadabur_name_screen"]["tadabur_id"] for s in PILOT
                 if mushafs[s]["tadabur_name_screen"]}
    usable = {name: clip for name, clip in registry.items()
              if SEALED_PANEL not in clip.uses and clip.reciter_id not in pilot_ids}
    long_enough = {name: clip for name, clip in usable.items()
                   if len(_canonical_words(clip.surah_ayah)) >= MIN_WORDS}

    synthetic = sorted(usable, key=lambda n: salted_rank("synthetic", n))[:SYNTHETIC_CLIPS]
    positives = [{"audio_filename": n, "tadabur_reciter": usable[n].reciter_id,
                  "cut_ms": CUT_MS[0] + int(salted_rank("cut", n)[:8], 16)
                  % (CUT_MS[1] - CUT_MS[0] + 1)} for n in synthetic]

    rows_of: dict[str, dict[str, int]] = {}

    def rows(slug: str) -> dict[str, int]:
        if slug not in rows_of:
            rows_of[slug] = ayah_rows(slug)
        return rows_of[slug]

    negatives = []
    for a, b in HARD_NEGATIVE_PAIRS:
        shared = [k for k in rows(a) if k in rows(b)
                  and len(_canonical_words(k)) >= MIN_WORDS]
        for key in sorted(shared, key=lambda k: salted_rank("negative", a, b, k)
                          )[:HARD_NEGATIVE_AYAT]:
            negatives.append({"pair": [a, b], "surah_ayah": key,
                              "rows": {a: rows(a)[key], b: rows(b)[key]}})

    by_tadabur: dict[int, list[str]] = defaultdict(list)
    for slug, m in mushafs.items():
        if m["tadabur_name_screen"] and slug not in PILOT:
            by_tadabur[m["tadabur_name_screen"]["tadabur_id"]].append(slug)
    staged_by_reciter: dict[int, list[str]] = defaultdict(list)
    for name, clip in long_enough.items():
        staged_by_reciter[clip.reciter_id].append(name)
    reciters = {t: slugs for t, slugs in sorted(by_tadabur.items())
                if t not in pilot_ids and len(staged_by_reciter[t]) >= CROSS_MIN_STAGED}
    cross = []
    for tadabur_id, slugs in reciters.items():
        # A different reciter's same ayah comes from another control reciter's mushafs.
        others = sorted(s for t, group in reciters.items() if t != tadabur_id for s in group)
        clips = sorted(staged_by_reciter[tadabur_id], key=lambda n: salted_rank("cross", n))
        for name in clips[:CROSS_CLIPS]:
            key = usable[name].surah_ayah
            qud = [{"slug": s, "row": rows(s)[key]} for s in sorted(slugs) if key in rows(s)]
            other = min(others, key=lambda s: salted_rank("other", name, s))
            cross.append({"audio_filename": name, "tadabur_reciter": tadabur_id,
                          "surah_ayah": key, "qud": qud,
                          "other": {"slug": other, "row": rows(other).get(key)}})
    return {
        "issue": 144,
        "salt": SALT,
        "qud_metadata_revision": selection["qud_revision"],
        "qud_audio": "the dataset viewer's rows API (tadabur.qud_source.viewer_row); the "
                     "revision it served is recorded with each score",
        "pilot": list(PILOT),
        "pilot_tadabur_ids_left_out": sorted(pilot_ids),
        "fingerprint": {"frame_s": FRAME_S, "hop_s": HOP_S, "bands": BANDS, "low_hz": LOW_HZ,
                        "high_hz": HIGH_HZ, "min_overlap_frames": MIN_OVERLAP},
        "synthetic_encoding": "resampled to 44.1 kHz, duplicated to stereo, scaled by 0.7, "
                              "written as MP3 by libsndfile's defaults, the first cut_ms cut",
        "min_words": MIN_WORDS,
        "timing_margin": TIMING_MARGIN,
        "decision_lines": {
            "fingerprint": "halfway between the highest BER of a synthetic positive and the "
                           "lowest BER of a hard negative; the check fails unless the first "
                           "is below the second",
            "timing": "timing_margin x the largest timing distance of a cross-aligner pair "
                      "the fingerprint matches",
        },
        "synthetic_positives": positives,
        "hard_negatives": negatives,
        "cross_aligner": cross,
    }


# --- scoring the controls, and the lines ---------------------------------------------------


def synthetic_copy(samples: np.ndarray, cut_ms: int) -> np.ndarray:
    """A 16 kHz clip as QUD would carry it: 44.1 kHz stereo MP3, its start cut, decoded back."""
    import librosa
    import soundfile as sf

    from .audio import TARGET_SAMPLE_RATE

    cut = samples[round(cut_ms * TARGET_SAMPLE_RATE / 1000):]
    up = librosa.resample(cut, orig_sr=TARGET_SAMPLE_RATE, target_sr=44100)
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([up, up], axis=1) * 0.7, 44100, format="MP3")
    return decode_to_mono_16k(buffer.getvalue())


def _clip(clips_dir: Path, registry: Mapping, name: str) -> np.ndarray:
    from .staged_audio import verify_staged

    verify_staged(registry[name], clips_dir)
    return decode_to_mono_16k(read_audio_bytes(clips_dir / name))


def _match(a: np.ndarray, b: np.ndarray) -> dict:
    found = best_match(fingerprint(a), fingerprint(b))
    return {"ber": None if found is None else found.ber,
            "offset_frames": None if found is None else found.offset,
            "overlap_frames": None if found is None else found.overlap}


def score_controls(controls: Mapping, registry: Mapping, clips_dir: Path,
                   timings: Mapping[str, dict], fetch) -> dict:
    """Every control's fingerprint BER (and, for cross-aligner pairs, timing distance).
    ``fetch(slug, row)`` gives a :class:`~tadabur.qud_source.ViewerRow`; ``timings`` is the
    Tadabur timing cache by ``audio_filename``."""
    from .qud_source import decode_ayah

    def qud(slug: str, row: int, expected: str) -> tuple:
        got = fetch(slug, row)
        if got.row.surah_ayah != expected:
            raise ValueError(f"{slug} row {row} is {got.row.surah_ayah}, not {expected}")
        return got, decode_ayah(got.row)

    positives = []
    for p in controls["synthetic_positives"]:
        samples = _clip(clips_dir, registry, p["audio_filename"])
        positives.append({**p, **_match(samples, synthetic_copy(samples, p["cut_ms"]))})
    negatives = []
    for n in controls["hard_negatives"]:
        (a, wa), (b, wb) = (qud(s, n["rows"][s], n["surah_ayah"]) for s in n["pair"])
        negatives.append({**n, **_match(wa, wb), "revisions": [a.revision, b.revision],
                          "mp3_sha256": [mp3_sha256(a.row.mp3), mp3_sha256(b.row.mp3)]})
    cross = []
    for c in controls["cross_aligner"]:
        samples = _clip(clips_dir, registry, c["audio_filename"])
        canonical = _canonical_words(c["surah_ayah"])
        tadabur_starts = timings[c["audio_filename"]]["starts"]
        pairs = [*({**q, "same_reciter": True} for q in c["qud"]),
                 *([{**c["other"], "same_reciter": False}] if c["other"]["row"] is not None
                   else [])]
        for q in pairs:
            got, wave = qud(q["slug"], q["row"], c["surah_ayah"])
            starts = qud_word_starts(got.row, canonical)
            distance = (timing_distance(starts, tadabur_starts)
                        if starts is not None and len(starts) == len(tadabur_starts) else None)
            cross.append({"audio_filename": c["audio_filename"],
                          "tadabur_reciter": c["tadabur_reciter"], "surah_ayah": c["surah_ayah"],
                          **q, "revision": got.revision, "mp3_sha256": mp3_sha256(got.row.mp3),
                          "timing_distance": distance, **_match(samples, wave)})
    return {"synthetic_positives": positives, "hard_negatives": negatives,
            "cross_aligner": cross}


def decide_lines(scores: Mapping) -> dict:
    """The fingerprint and timing lines from the controls' scores (module docstring)."""
    pos = [p["ber"] for p in scores["synthetic_positives"]]
    neg = [n["ber"] for n in scores["hard_negatives"] if n["ber"] is not None]
    fingerprint_line = (max(pos) + min(neg)) / 2 if max(pos) < min(neg) else None
    out: dict = {"fingerprint": {"line": fingerprint_line, "synthetic_positive_max": max(pos),
                                 "hard_negative_min": min(neg),
                                 "synthetic_positives": len(pos), "hard_negatives": len(neg)}}
    if fingerprint_line is None:
        out["timing"] = {"line": None, "why": "the fingerprint has no line"}
        return out
    cross = [c for c in scores["cross_aligner"] if c["ber"] is not None]
    same = [c for c in cross if c["ber"] <= fingerprint_line]
    timed = [c["timing_distance"] for c in same if c["timing_distance"] is not None]
    line = TIMING_MARGIN * max(timed) if timed else None
    same_reciter_neg = [c for c in cross if c["same_reciter"] and c["ber"] > fingerprint_line
                        and c["timing_distance"] is not None]
    other_neg = [c for c in cross if not c["same_reciter"] and c["timing_distance"] is not None]
    out["timing"] = {
        "line": None if line is None else round(line, 4),
        "positives": len(same), "positives_timed": len(timed),
        "positive_max": max(timed) if timed else None,
        "fingerprint_matches_by_reciter": dict(sorted(
            _count(c["tadabur_reciter"] for c in same).items())),
        "same_reciter_negatives_timed": len(same_reciter_neg),
        "same_reciter_negatives_passing": None if line is None else
        sum(c["timing_distance"] <= line for c in same_reciter_neg),
        "other_reciter_negatives_timed": len(other_neg),
        "other_reciter_negatives_passing": None if line is None else
        sum(c["timing_distance"] <= line for c in other_neg),
        "different_reciter_fingerprint_min": min((c["ber"] for c in cross
                                                  if not c["same_reciter"]), default=None),
    }
    return out


def _count(values: Iterable) -> dict:
    counts: dict = defaultdict(int)
    for v in values:
        counts[str(v)] += 1
    return counts


# --- stage 1: the timing screen --------------------------------------------------------------


def screen_mushaf(rows: Sequence[QudRow], by_ayah: Mapping[str, list[dict]],
                  line: float) -> dict:
    """Every comparable ayah of a pilot mushaf against every Tadabur row of that ayah: how
    many were compared, each Tadabur reciter's closest distances, and the candidates."""
    compared_ayat = pairs = 0
    candidates = []
    closest: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        starts = qud_word_starts(row, _canonical_words(row.surah_ayah))
        if starts is None:
            continue
        compared_ayat += 1
        for other in by_ayah.get(row.surah_ayah, []):
            if len(other["starts"]) != len(starts):
                continue
            pairs += 1
            distance = timing_distance(starts, other["starts"])
            closest[other["reciter_id"]].append(distance)
            if distance <= line:
                candidates.append({"surah_ayah": row.surah_ayah, "distance": distance,
                                   **{k: other[k] for k in ("audio_filename", "shard",
                                                            "row_index", "reciter_id")}})
    per_reciter = {
        str(reciter): {"pairs": len(d), "min": min(d), "median": statistics.median(d),
                       "within_line": sum(x <= line for x in d)}
        for reciter, d in sorted(closest.items())
    }
    return {"ayat": len(rows), "ayat_compared": compared_ayat, "pairs": pairs,
            "candidates": sorted(candidates, key=lambda c: (c["shard"], c["row_index"])),
            "per_tadabur_reciter": per_reciter}


# --- stage 2 and Muaalem: fingerprints ---------------------------------------------------------


def confirm_candidates(candidates: Sequence[Mapping], line: float, cache_dir: Path,
                       fetch, row_of_ayah: Mapping[str, int], slug: str) -> list[dict]:
    """Each candidate's Tadabur audio (its shard read once, :func:`tadabur.shard_reader.
    iter_selected_rows`) fingerprinted against the QUD ayah. A sealed-panel row cannot be
    read and is recorded as not compared."""
    from .panel_seal import SealedPanelError
    from .qud_source import decode_ayah
    from .shard_reader import iter_selected_rows

    by_shard: dict[int, list[Mapping]] = defaultdict(list)
    for c in candidates:
        by_shard[c["shard"]].append(c)
    waves: dict[str, np.ndarray] = {}
    out = []
    for shard, group in sorted(by_shard.items()):
        wanted = {c["row_index"]: c for c in group}
        rows = iter_selected_rows(shard, set(wanted), cache_dir=cache_dir,
                                  columns=["audio", "surah_id", "ayah_id", "reciter_id"])
        for position, record in enumerate(rows):
            if record is None:
                continue
            c = wanted[position]
            try:
                tadabur = decode_to_mono_16k(record["audio"]["bytes"])
            except SealedPanelError:
                out.append({**c, "ber": None, "match": None, "why": "sealed panel"})
                continue
            if c["surah_ayah"] not in waves:
                got = fetch(slug, row_of_ayah[c["surah_ayah"]])
                waves[c["surah_ayah"]] = decode_ayah(got.row)
            found = _match(waves[c["surah_ayah"]], tadabur)
            out.append({**c, **found, "match": found["ber"] is not None
                        and found["ber"] <= line})
    return out


#: Muaalem segments compared per moshaf: the first row group of its first file.
MUAALEM_ROW_GROUP = 0


def muaalem_check(slug: str, fetch, row_of_ayah: Mapping[str, int], line: float) -> list[dict]:
    """Per Muaalem moshaf, the lowest BER of any of the pilot mushaf's ayat against the
    Muaalem segments that cover it (first row group of the first file)."""
    from .muaalem_audio import first_shard, row_group_segments
    from .qud_source import decode_ayah

    sources = _load(SOURCES_PATH)
    waves: dict[str, np.ndarray] = {}
    out = []
    for moshaf in sorted(m["id"] for m in sources["muaalem"]["moshafs"]):
        path = first_shard(moshaf)
        best = None
        compared = 0
        for segment in row_group_segments(moshaf, path, MUAALEM_ROW_GROUP):
            if segment.first is None or segment.last is None or segment.first[0] != segment.last[0]:
                continue
            surah = segment.first[0]
            ayat = [f"{surah}:{a}" for a in range(segment.first[1], segment.last[1] + 1)]
            ayat = [k for k in ayat if k in row_of_ayah]
            if not ayat:
                continue
            reference = fingerprint(decode_to_mono_16k(segment.audio))
            for key in ayat:
                if key not in waves:
                    waves[key] = decode_ayah(fetch(slug, row_of_ayah[key]).row)
                found = best_match(reference, fingerprint(waves[key]))
                compared += 1
                if found is not None and (best is None or found.ber < best["ber"]):
                    best = {"ber": found.ber, "surah_ayah": key,
                            "segment_index": segment.segment_index}
        out.append({"moshaf": moshaf, "file": path, "row_group": MUAALEM_ROW_GROUP,
                    "comparisons": compared, "closest": best,
                    "match": best is not None and best["ber"] <= line})
    return out


# --- running it ----------------------------------------------------------------------------


def _dump(path: Path, payload: Mapping) -> None:
    from training.tashkeel_eval import write_text_atomically

    write_text_atomically(path, json.dumps(payload, ensure_ascii=False, indent=1,
                                           sort_keys=True) + "\n")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ayah_rows(selection: Mapping):
    """``slug -> {surah:ayah: row index}`` from the shards' ``surah`` / ``ayah`` columns."""
    shards = {m["slug"]: m["shards"] for m in selection["mushafs"]}

    def read(slug: str) -> dict[str, int]:
        keys = [f"{r['surah']}:{r['ayah']}" for shard in shards[slug]
                for r in parquet_table(QUD_DATASET, selection["qud_revision"], shard,
                                       ["surah", "ayah"]).to_pylist()]
        return {key: i for i, key in enumerate(keys)}

    return read


def timings_by_filename(path: Path) -> dict[str, dict]:
    return {row["audio_filename"]: row for rows in read_tadabur_timings(path).values()
            for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("controls", help="define the controls (commit before calibrating)")
    calibrate = commands.add_parser("calibrate", help="score the controls and fix the lines")
    calibrate.add_argument("--clips-dir", type=Path, required=True)
    calibrate.add_argument("--timings", type=Path, required=True,
                           help="the Tadabur timing cache (write-timings)")
    timings = commands.add_parser("write-timings", help="cache Tadabur's word start times")
    timings.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from .staged_audio import load_staged_clips

    selection = _load(SELECTION_PATH)
    if args.command == "controls":
        _dump(CONTROLS_PATH, define_controls(selection, load_staged_clips(),
                                             _ayah_rows(selection)))
    elif args.command == "write-timings":
        from .shard_reader import NUM_SHARDS

        write_tadabur_timings(args.out, range(NUM_SHARDS))
    elif args.command == "calibrate":
        from .qud_source import viewer_row

        scores = score_controls(_load(CONTROLS_PATH), load_staged_clips(), args.clips_dir,
                                timings_by_filename(args.timings), viewer_row)
        _dump(RULES_PATH, {"lines": decide_lines(scores), "scores": scores})
        print(json.dumps(_load(RULES_PATH)["lines"], indent=1))


if __name__ == "__main__":
    main()

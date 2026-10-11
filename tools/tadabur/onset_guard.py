"""Guard Ship A's sites against a clip whose opening sound is cut off (#138).

While listening (#61), the owner found clips whose first sound is missing: the ``ء`` of
``ءَلَّذِينَ`` or the ``ث`` of ``ثُمَّ`` at the start of an ayah, so the shaddah after it cannot
be judged. The cut is in Tadabur's source audio (the excerpt starts at the file's sample 0)
and the app never hears audio cut that way, so such a site tests the clip, not the model.

**Which items are exposed** (:func:`exposure`). Every Ship A site, every row of #61's frozen
worklist and every word of the word sample (the item model of :mod:`tadabur.waqf_guard`) is
exposed when its prescribed mark's evidence (a geminate and the letter before it, a
letter's carrier, a tashkeel carrier; a sampled word whole) lies in the first word of its
segment. An excerpt plays the carrier's word and one word either side, so that is also the
first word the owner hears. The segment, the span the model decodes, starts at a cut that
may have removed speech: the file's sample 0 (``file_start``, also when only the excerpt
starts there), or a segment start the segmenter placed (``segmenter_cut``).

**Was the onset cut** (the detector). Each exposed segment is scored by five candidates,
each read from the start of the span the model decodes (:data:`CANDIDATES`):

* ``ctc_onset_llr``: the base teacher's CTC log-likelihood of the segment's reference less
  that of the reference without its first letter (the letter and its marks): low when the
  audio is better explained without that letter;
* ``ctc_first_posterior``: the base teacher's posterior of the reference's first letter on
  its best (Viterbi) path, the highest over its frames;
* ``decode_leading_missing``: how many reference characters precede the first stretch
  (two characters or more) the base teacher's free decode matches;
* ``onset_level_db``: the level of the span's first 10 ms above the file's noise floor (its
  10th-percentile 10 ms frame): a cut mid-waveform starts loud, a clean cut fades in;
* ``vad_speech_frame0``: the recitation VAD's speech probability for the span's first frame.

**Calibration** (:func:`synthetic_clips`, :func:`calibrate`). Synthetic trims of clean
segments. Tadabur's files start mid-sound (their first 10 ms frame is speech-level almost
always), so the clean onsets are the segments that start after a pause: a kept segment of a
mining-pool clip (never a sealed-panel reciter, never a clip holding an exposed item) whose
first letter's onset, read off the waveform (:func:`clear_onset`), follows a clear silence
inside its pause; one segment per canonical reciter. Each is cut at :data:`DEPTHS_MS` from
the onset (``-100``: 100 ms of the silence kept; ``0``: at the onset; the rest remove that
much of the first letter) to its end, the reference unchanged. A cut that removes
any of the letter (depth > 0) is a trim. Clips are split by canonical reciter
(:data:`SPLIT_SALT`): each candidate's operating point is the pre-stated rule on the
calibration half, the highest recall with precision at least
:data:`tadabur.local_pause.MIN_PRECISION` (ties: higher precision, then the stricter
threshold); the test half is only reported. The candidate the guard uses is the one with the
highest calibration recall at its operating point (ties: higher precision, then
:data:`CANDIDATES` order).

**Real-world check** (:func:`owner_check`). At the current verdicts, the owner's trim notes on
exposed items are positives and the answered exposed items without one are weak negatives
(the owner was not asked about trims). The detector **decides** only when the chosen
candidate is not contradicted there: it must call at least :data:`MIN_OWNER_FLAG_RECALL` of
the owner-flagged exposed items trimmed, at a precision of at least
:data:`tadabur.local_pause.MIN_PRECISION` (the floor its operating point was held to).
Otherwise its scores are kept, the answered items any candidate calls trimmed are listed for
re-listening, and only the owner's notes decide. The precision condition was added after the
first build (:data:`OWNER_CHECK_HISTORY`): the recall-only rule written down first would have
let the chosen candidate decide, so the detector being off rests on that added condition (and
on the synthetic clean cuts it calls trimmed), not on a rule fixed before any result.

**The owner decides first** (:func:`owner_says`). A note with ``trim``, ``cut`` or ``split``
(any word starting so: "Trimming issues", "Incurrect split") marks the item ``onset_trimmed``;
a negated one ("no trim", "not cut") marks it clean, overriding the detector both ways
(:func:`tadabur.owner_notes.says`, negation scoped to its clause). A note decides on any item, exposed or not. Notes are read
from the verdicts file at every use (:func:`trimmed_ids`).

**Treatment.** An item judged ``onset_trimmed`` is an excerpt fault: it leaves every evaluation
denominator through the one exclusion the waqf guard uses (:mod:`tadabur.exclusions`), after
the design weights are computed over every site, and it is no training target. Its verdict is
kept on record.

Files (``tools/tadabur/ship_a/``):

* ``onset_guard.synthetic.jsonl``: the synthetic clips: provenance (clip, checksum, shard,
  canonical reciter, split), the segment, the first letter's onset sample and the levels it
  was read at; ``onset_guard.synthetic.json``: the selection's counts and rules;
* ``onset_guard.scores.jsonl``: every span scored (each exposed segment and each synthetic
  cut), with the candidates' scores and the head of the free decode; ``onset_guard.run.json``:
  the decode and VAD pass's provenance;
* ``onset_guard.jsonl``: one row per item: its exposure, cut, span, scores and the detector's
  decision;
* ``onset_guard.curves.json``: every candidate's calibration curve;
* ``onset_guard.summary.json``: counts per part and per stratum at the current verdicts, the
  operating points with calibration, test and owner-flag numbers, and the inputs' checksums.

Usage (from ``tools/``)::

  python -m tadabur.onset_guard synthetic --audio-dir stage/clips      # CPU, reads audio
  python -m tadabur.onset_guard spans --out spans.jsonl
  python -m tadabur.onset_guard score --spans spans.jsonl --audio-dir stage/clips \\
      --audio-dir ../issue-126/stage/clips                             # GPU, minutes
  python -m tadabur.onset_guard build                                  # torch-free
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from training.acceptance_stats import reciter_hash
from training.tashkeel_eval import write_text_atomically

from .audio import TARGET_SAMPLE_RATE
from .listening_session import VERDICTS_PATH, Verdict, read_verdicts
from .local_pause import MIN_PRECISION, point
from .owner_notes import TRIM, says
from .phoneme_vocab import PHONEME_CHAR_TO_ID
from .ship_a import SHIP_A_DIR, SITES_PATH
from .ship_a_worklist import WORKLIST_PATH
from .waqf_guard import WORD, WORD_SAMPLE, Item, all_items, sha256_of
from .word_sample import WORD_SAMPLE_PATH

GUARD_PATH = SHIP_A_DIR / "onset_guard.jsonl"
SUMMARY_PATH = SHIP_A_DIR / "onset_guard.summary.json"
CURVES_PATH = SHIP_A_DIR / "onset_guard.curves.json"
SCORES_PATH = SHIP_A_DIR / "onset_guard.scores.jsonl"
RUN_PATH = SHIP_A_DIR / "onset_guard.run.json"
SYNTHETIC_PATH = SHIP_A_DIR / "onset_guard.synthetic.jsonl"
SYNTHETIC_RUN_PATH = SHIP_A_DIR / "onset_guard.synthetic.json"

ONSET_TRIMMED = "onset_trimmed"
SITE = "site"
EXPOSED = "exposed"
NOT_EXPOSED = "not_exposed"
#: Where an exposed item's segment starts.
FILE_START = "file_start"
SEGMENTER_CUT = "segmenter_cut"
#: Who decided an item's status.
OWNER_NOTE = "owner_note"
DETECTOR = "detector"
#: A scored span: an exposed segment, or a synthetic cut.
CUT = "cut"
SYNTHETIC = "synthetic"

#: The owner check (#138): the detector decides only if it calls at least this share of the
#: owner-flagged exposed items trimmed (and at the operating point's precision floor).
MIN_OWNER_FLAG_RECALL = 0.75
#: How the owner check came to be, recorded with every summary.
OWNER_CHECK_HISTORY = (
    "written down before any result: contradicted only when the chosen candidate finds fewer "
    "than MIN_OWNER_FLAG_RECALL of the owner's flags; that rule passed the chosen candidate "
    "(onset_level_db, calling 99 of 187 exposed items trimmed at owner-check precision 0.18). "
    "The precision floor (MIN_PRECISION, the operating point's) was added after that first "
    "build; detector-off rests on it.")

# --- synthetic trims ---------------------------------------------------------------------------

#: Where each clip is cut, in ms from its first letter's onset; a depth above 0 is a trim.
DEPTHS_MS = (-100, 0, 20, 40, 80, 120)
#: The reciter split of the synthetic clips: a reciter is ``calibration`` when its
#: :func:`training.acceptance_stats.reciter_hash` under this salt is below 2**63.
SPLIT_SALT = "issue-138-onset-trim-2026-10"
SPLIT_CUTOFF = 2 ** 63
CALIBRATION = "calibration"
TEST = "test"
SPLITS = (CALIBRATION, TEST)
#: The selection's ranking salt (which of a reciter's eligible clips is taken).
SELECTION_SALT = "issue-138-onset-clips-2026-10"
#: Level frames: 10 ms, not overlapping.
LEVEL_FRAME = TARGET_SAMPLE_RATE // 100
#: Levels are read against the pause's floor: the 10th-percentile frame of the (up to)
#: :data:`FLOOR_S` of pause before the first word's aligned onset. The first word's 95th
#: percentile over :data:`CONTRAST_S` must stand this far above it.
FLOOR_S = 0.4
CONTRAST_S = 0.5
MIN_CONTRAST_DB = 20.0
#: A frame this far above the floor is speech; one at most this far above it is silence.
SPEECH_ABOVE_FLOOR_DB = 12.0
SILENCE_ABOVE_FLOOR_DB = 6.0
#: The first speech frame is looked for this close to the first word's aligned onset
#: (frames before it, frames after it); the onset walks back from it while frames are not
#: silent, and must stay this close to the aligned onset (seconds before, after).
SEARCH_FRAMES = (10, 5)
ONSET_WINDOW_S = (0.1, 0.05)
#: A clear leading silence: this long before the onset, inside the segment's own pause, its
#: 95th-percentile frame silent.
MIN_LEAD_S = 0.2


def reciter_split(reciter_id: int) -> str:
    return CALIBRATION if reciter_hash(reciter_id, SPLIT_SALT) < SPLIT_CUTOFF else TEST


def frame_levels(waveform: np.ndarray) -> np.ndarray:
    """Each whole 10 ms frame's level in dB (mean square, floored at -120 dB)."""
    frames = len(waveform) // LEVEL_FRAME
    squares = np.asarray(waveform[:frames * LEVEL_FRAME], dtype=np.float64) ** 2
    power = squares.reshape(frames, LEVEL_FRAME).mean(axis=1)
    return 10 * np.log10(np.maximum(power, 1e-12))


@dataclass(frozen=True)
class Onset:
    """A first letter's onset read off the waveform, and the levels it was read at."""

    sample: int
    floor_db: float
    peak_db: float
    lead_db: float


def clear_onset(levels: np.ndarray, word_s: float, pause_start: int) -> Onset | None:
    """The onset of a segment's first letter after a clear leading silence, or ``None``.

    The pause runs from frame ``pause_start`` (the previous segment's end, or 0) to the first
    word's aligned onset ``word_s``; its floor and the word's contrast are read as
    :data:`FLOOR_S` and :data:`MIN_CONTRAST_DB` say. The first speech frame near ``word_s``
    (:data:`SEARCH_FRAMES`) is walked back while frames are not silent; the onset must lie
    within :data:`ONSET_WINDOW_S` of ``word_s`` with :data:`MIN_LEAD_S` of silence before it,
    inside the pause."""
    word, lead = round(word_s * 100), round(MIN_LEAD_S * 100)
    lo = max(pause_start, word - round(FLOOR_S * 100))
    if word - lo < lead:
        return None
    floor = float(np.percentile(levels[lo:word], 10))
    peak = float(np.percentile(levels[word:word + round(CONTRAST_S * 100)], 95))
    first = max(0, word - SEARCH_FRAMES[0])
    near = np.flatnonzero(levels[first:word + SEARCH_FRAMES[1]] >= floor + SPEECH_ABOVE_FLOOR_DB)
    if peak - floor < MIN_CONTRAST_DB or not len(near):
        return None
    onset = first + int(near[0])
    while onset > 0 and levels[onset - 1] > floor + SILENCE_ABOVE_FLOOR_DB:
        onset -= 1
    if (onset - lead < pause_start
            or not word_s - ONSET_WINDOW_S[0] <= onset / 100 <= word_s + ONSET_WINDOW_S[1]):
        return None
    lead_db = float(np.percentile(levels[onset - lead:onset], 95))
    if lead_db > floor + SILENCE_ABOVE_FLOOR_DB:
        return None
    return Onset(onset * LEVEL_FRAME, round(floor, 2), round(peak, 2), round(lead_db, 2))


def _rank(*parts) -> str:
    return hashlib.sha256(":".join((SELECTION_SALT, *map(str, parts))).encode()).hexdigest()


def synthetic_clips(audio_dirs: Sequence[Path]) -> tuple[list[dict], dict]:
    """One clean segment per canonical reciter of the mining pool (module docstring), with
    its provenance and onset, and the selection's counts."""
    from .audio import read_audio
    from .exposure import MINING_POOL, excluded_reciters
    from .mining_pool import load_manifest
    from .staged_audio import load_staged_clips, verify_staged

    registry = load_staged_clips()
    barred = excluded_reciters(MINING_POOL)
    exposed_clips = {i.audio_filename for i in all_items() if exposure(i) == EXPOSED}
    counts = Counter()
    by_reciter: dict[int, list] = defaultdict(list)
    for clip in load_manifest(registry=registry):
        if clip.reciter_id in barred:
            raise ValueError(f"{clip.audio_filename}: a sealed-panel reciter in the mining pool")
        if clip.audio_filename in exposed_clips:
            counts["clip_holds_an_exposed_item"] += 1
        elif clip.re_reads or clip.skip_reason or not clip.word_times:
            counts["clip_not_placed"] += 1
        else:
            pauses = (0, *(s.end_sample // LEVEL_FRAME for s in clip.segments[:-1]))
            by_reciter[clip.reciter_id].extend(
                (clip, segment, pause) for segment, pause in zip(clip.segments, pauses)
                if segment.kept)
    rows: list[dict] = []
    levels_of: dict[str, np.ndarray] = {}
    for reciter in sorted(by_reciter):
        ranked = sorted(by_reciter[reciter],
                        key=lambda c: _rank(c[0].audio_filename, c[1].segment_index))
        for clip, segment, pause in ranked:
            staged = registry[clip.audio_filename]
            if clip.audio_filename not in levels_of:
                audio_dir = next(d for d in audio_dirs if (d / clip.audio_filename).exists())
                verify_staged(staged, audio_dir)
                levels_of[clip.audio_filename] = frame_levels(
                    read_audio(audio_dir / clip.audio_filename, dtype="float32")[0])
            word_s = clip.word_times[segment.word_start]
            onset = clear_onset(levels_of[clip.audio_filename], word_s, pause)
            counts["segments_read"] += 1
            if onset is None:
                counts["segment_without_a_clear_onset"] += 1
                continue
            rows.append({"audio_filename": clip.audio_filename,
                         "audio_sha256": staged.audio_sha256, "shard": staged.shard,
                         "row_index": staged.row_index, "reciter_id": clip.reciter_id,
                         "split": reciter_split(clip.reciter_id),
                         "surah_ayah": clip.surah_ayah, "segment_index": segment.segment_index,
                         "word_start": segment.word_start,
                         "segment_start_sample": segment.start_sample,
                         "segment_end_sample": segment.end_sample,
                         "reference": segment.reference, "word_s": word_s,
                         "onset_sample": onset.sample, "floor_db": onset.floor_db,
                         "peak_db": onset.peak_db, "lead_db": onset.lead_db})
            break
        else:
            counts["reciters_without_a_clean_segment"] += 1
    record = {"pool_reciters": len(by_reciter), "segments": len(rows),
              "per_split": dict(sorted(Counter(r["split"] for r in rows).items())),
              "counts": dict(sorted(counts.items())), "depths_ms": list(DEPTHS_MS),
              "split_salt": SPLIT_SALT, "selection_salt": SELECTION_SALT,
              "rules": {"floor_s": FLOOR_S, "contrast_s": CONTRAST_S,
                        "min_contrast_db": MIN_CONTRAST_DB,
                        "speech_above_floor_db": SPEECH_ABOVE_FLOOR_DB,
                        "silence_above_floor_db": SILENCE_ABOVE_FLOOR_DB,
                        "search_frames": list(SEARCH_FRAMES),
                        "onset_window_s": list(ONSET_WINDOW_S), "min_lead_s": MIN_LEAD_S},
              "barred_reciters": len(barred)}
    return rows, record


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    write_text_atomically(path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


# --- which items are exposed -------------------------------------------------------------------


def exposure(item: Item) -> str:
    """Whether the item's mark is read in the first word of its segment (module docstring).
    A mark is read inside one word (:func:`tadabur.waqf_guard.classify` checks it), so its
    carrier places it; a sampled word's carrier is its first character."""
    return EXPOSED if item.reference_index < item.word_offsets[1] else NOT_EXPOSED


def cut_of(item: Item) -> str:
    """Where an exposed item's segment starts: at the file's sample 0, or at the segmenter's cut."""
    return FILE_START if 0 in (item.start_sample, item.excerpt_start_sample) else SEGMENTER_CUT


def span_key(audio_filename: str, start: int, end: int) -> str:
    return f"{CUT}:{audio_filename}@{start}-{end}"


def synthetic_key(clip: Mapping, depth_ms: int) -> str:
    return f"{SYNTHETIC}:{clip['audio_filename']}#{clip['segment_index']}@{depth_ms}"


def spans(items: Iterable[Item], synthetic: Iterable[Mapping]) -> list[dict]:
    """Every span the detector scores, by key: each exposed item's segment (once per
    segment) and each synthetic clip cut at each depth."""
    cuts: dict[str, dict] = {}
    for item in items:
        if exposure(item) != EXPOSED:
            continue
        key = span_key(item.audio_filename, item.start_sample, item.end_sample)
        row = {"key": key, "kind": CUT, "audio_filename": item.audio_filename,
               "start_sample": item.start_sample, "end_sample": item.end_sample,
               "reference": item.reference}
        if cuts.setdefault(key, row) != row:
            raise ValueError(f"{key}: two items read one segment against two references")
    rows = list(cuts.values())
    for clip in synthetic:
        for depth in DEPTHS_MS:
            rows.append({"key": synthetic_key(clip, depth), "kind": SYNTHETIC,
                         "audio_filename": clip["audio_filename"],
                         "start_sample": clip["onset_sample"] + depth * TARGET_SAMPLE_RATE // 1000,
                         "end_sample": clip["segment_end_sample"],
                         "reference": clip["reference"], "split": clip["split"],
                         "reciter_id": clip["reciter_id"], "depth_ms": depth,
                         "trimmed": depth > 0})
    return sorted(rows, key=lambda r: r["key"])


# --- the candidates' scores (pure, from posteriors and samples) --------------------------------


@dataclass(frozen=True)
class Candidate:
    """A detector score and its direction: ``low`` when a lower score says trimmed."""

    name: str
    trimmed_when: str

    def fires(self, score: float, threshold: float) -> bool:
        return score <= threshold if self.trimmed_when == "low" else score >= threshold

    def strictness(self, threshold: float) -> float:
        """Larger for a threshold that calls fewer spans trimmed."""
        return -threshold if self.trimmed_when == "low" else threshold


CANDIDATES = (
    Candidate("ctc_onset_llr", "low"),
    Candidate("ctc_first_posterior", "low"),
    Candidate("decode_leading_missing", "high"),
    Candidate("onset_level_db", "high"),
    Candidate("vad_speech_frame0", "high"),
)
#: The free decode's head kept with a score, for reading.
DECODE_HEAD = 16


def labels_of(reference: str) -> tuple[list[int], int]:
    """The reference's CTC labels (spaces dropped) and how many its first letter and that
    letter's marks take (the letter, then every combining character after it)."""
    chars = reference.replace(" ", "")
    unknown = sorted(set(chars) - set(PHONEME_CHAR_TO_ID))
    if unknown:
        raise ValueError(f"reference characters outside the phoneme vocabulary: {unknown}")
    first = 1
    while first < len(chars) and unicodedata.combining(chars[first]):
        first += 1
    return [PHONEME_CHAR_TO_ID[c] for c in chars], first


def leading_missing(decode: str, reference: str) -> int:
    """How many reference characters (spaces dropped) precede the first stretch of two or
    more that the decode matches (difflib's matching blocks); the whole reference when none."""
    chars = reference.replace(" ", "")
    for block in SequenceMatcher(None, chars, decode.replace(" ", ""),
                                 autojunk=False).get_matching_blocks():
        if block.size >= 2:
            return block.a
    return len(chars)


def ctc_scores(log_posteriors: np.ndarray, reference: str) -> dict:
    """The CTC candidates from one span's log-posteriors (module docstring)."""
    from training.ctc_paths import ctc_best_path, ctc_log_likelihood
    from training.decoding import scan_ctc, tokens_to_phonemes

    labels, first = labels_of(reference)
    llr = ctc_log_likelihood(log_posteriors, labels) - ctc_log_likelihood(
        log_posteriors, labels[first:])
    lo, hi = ctc_best_path(log_posteriors, labels)[0]
    posterior = float(np.exp(log_posteriors[lo:hi + 1, labels[0]].max()))
    decode = tokens_to_phonemes(s.token_id for s in scan_ctc(log_posteriors.argmax(axis=1)))
    return {"ctc_onset_llr": round(float(llr), 4), "ctc_first_posterior": round(posterior, 6),
            "decode_leading_missing": leading_missing(decode, reference),
            "decode_head": decode[:DECODE_HEAD]}


def onset_level_db(span: np.ndarray, file_levels: np.ndarray) -> float:
    """The span's first 10 ms level above the file's noise floor (its 10th-percentile frame)."""
    return round(float(frame_levels(span[:LEVEL_FRAME])[0] - np.percentile(file_levels, 10)), 3)


# --- the decode and VAD pass (GPU) -------------------------------------------------------------

#: The base teacher, decoded as #84 decodes it (bf16 weights, batch size 1).
BASE_MODEL = "obadx/muaalem-model-v3_2"
WEIGHTS_DTYPE = "bf16"
BATCH_SIZE = 1


def score_spans(rows: Sequence[dict], audio_dirs: Sequence[Path], device: str,
                vad_dtype: str) -> tuple[list[dict], dict]:
    """Every span's candidate scores, and the pass's provenance."""
    import torch

    from training.decoding import SPANS, Decoder
    from training.shaddah_probe_run import hub_revision

    from .audio import read_audio
    from .local_pause import CHUNK_SAMPLES, silence_probabilities
    from .staged_audio import load_staged_clips, verify_staged
    from .vad import _load_vad

    registry = load_staged_clips()
    decoder = Decoder.load(BASE_MODEL, device, weights_dtype=WEIGHTS_DTYPE,
                           batch_size=BATCH_SIZE)
    vad_device, dtype = torch.device(device), getattr(torch, vad_dtype)
    vad, processor = _load_vad(vad_device, dtype)
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_clip[row["audio_filename"]].append(row)
    out = []
    for name in sorted(by_clip):
        audio_dir = next(d for d in audio_dirs if (d / name).exists())
        verify_staged(registry[name], audio_dir)
        waveform = read_audio(audio_dir / name, dtype="float32")[0]
        levels = frame_levels(waveform)
        for row in by_clip[name]:
            span = waveform[row["start_sample"]:row["end_sample"]]
            (log_posteriors,) = decoder.span_log_posteriors([span])
            silence = silence_probabilities(span[:CHUNK_SAMPLES], vad, processor,
                                            vad_device, dtype)
            out.append({**row, "scores": {
                **ctc_scores(log_posteriors, row["reference"]),
                "onset_level_db": onset_level_db(span, levels),
                "vad_speech_frame0": round(1.0 - float(silence[0]), 6)}})
    provenance = {"decoder": decoder.fingerprint(SPANS).as_dict(),
                  "decoder_hub_revision": hub_revision(BASE_MODEL),
                  "vad_model": vad.config._name_or_path,
                  "vad_revision": getattr(vad.config, "_commit_hash", None),
                  "vad_dtype": vad_dtype, "level_frame_samples": LEVEL_FRAME,
                  "spans": len(out), "clips": len(by_clip),
                  "per_kind": dict(sorted(Counter(r["kind"] for r in out).items()))}
    return sorted(out, key=lambda r: r["key"]), provenance


# --- calibration -------------------------------------------------------------------------------


def curve(candidate: Candidate, rows: Sequence[Mapping]) -> list[dict]:
    """The candidate's precision and recall on ``rows`` (each with ``scores`` and ``trimmed``)
    at every score it takes there, as a threshold."""
    truth = [r["trimmed"] for r in rows]
    values = [r["scores"][candidate.name] for r in rows]
    return [{"threshold": t, **point((candidate.fires(v, t) for v in values), truth).as_dict()}
            for t in sorted(set(values))]


def operating_point(candidate: Candidate, points: Sequence[dict]) -> dict | None:
    """The pre-stated rule: the highest recall with precision at least
    :data:`MIN_PRECISION`; ties: higher precision, then the stricter threshold."""
    eligible = [p for p in points
                if p["precision"] is not None and p["precision"] >= MIN_PRECISION]
    if not eligible:
        return None
    return max(eligible, key=lambda p: (p["recall"], p["precision"],
                                        candidate.strictness(p["threshold"])))


def at(candidate: Candidate, threshold: float, rows: Sequence[Mapping]) -> dict:
    """Precision and recall at one threshold, with the share called trimmed per depth."""
    called = [candidate.fires(r["scores"][candidate.name], threshold) for r in rows]
    by_depth = defaultdict(list)
    for row, fired in zip(rows, called):
        by_depth[row["depth_ms"]].append(fired)
    return {**point(called, (r["trimmed"] for r in rows)).as_dict(),
            "called_trimmed_per_depth_ms": {str(d): f"{sum(v)}/{len(v)}"
                                            for d, v in sorted(by_depth.items())}}


def calibrate(synthetic: Sequence[Mapping]) -> tuple[dict, dict]:
    """Every candidate's operating point on the calibration half with both halves' numbers,
    the candidate chosen, and the full calibration curves."""
    by_split = {s: [r for r in synthetic if r["split"] == s] for s in SPLITS}
    reports, curves = {}, {}
    for candidate in CANDIDATES:
        curves[candidate.name] = curve(candidate, by_split[CALIBRATION])
        chosen = operating_point(candidate, curves[candidate.name])
        reports[candidate.name] = {
            "trimmed_when": candidate.trimmed_when,
            "operating_point": None if chosen is None else chosen["threshold"],
            **({} if chosen is None else {
                s: at(candidate, chosen["threshold"], rows) for s, rows in by_split.items()}),
        }
    ranked = [c for c in CANDIDATES if reports[c.name]["operating_point"] is not None]
    best = max(ranked, key=lambda c: (reports[c.name][CALIBRATION]["recall"],
                                      reports[c.name][CALIBRATION]["precision"],
                                      -CANDIDATES.index(c)), default=None)
    return {"rule": f"per candidate, the highest calibration recall with precision >= "
                    f"{MIN_PRECISION} (ties: higher precision, stricter threshold); the "
                    "candidate with the highest calibration recall at its point (ties: higher "
                    "precision, candidate order)",
            "trimmed": "a synthetic cut at depth > 0 ms (any of the first letter removed)",
            "spans": {s: len(rows) for s, rows in by_split.items()},
            "reciters": {s: len({r["reciter_id"] for r in rows}) for s, rows in by_split.items()},
            "candidates": reports,
            "chosen": None if best is None else best.name}, curves


# --- the decisions -----------------------------------------------------------------------------


def owner_says(note: str) -> bool | None:
    """What a verdict's note says about the onset: ``True`` cut, ``False`` clean (a negated
    trim word), ``None`` neither or ambiguous (:func:`tadabur.owner_notes.says`,
    :data:`tadabur.owner_notes.TRIM`)."""
    return says(note, TRIM)


def guard_row(item: Item, scores: Mapping[str, dict], decider: tuple[Candidate, float] | None
              ) -> dict:
    """The item's sidecar row: its exposure and, where exposed, its cut, span, scores and the
    detector's decision (``None`` where the detector does not decide)."""
    row = {"item_id": item.item_id, "parts": list(item.parts), "stratum": item.stratum,
           "mark": item.mark, "prescribed": item.prescribed, "exposure": exposure(item)}
    if row["exposure"] != EXPOSED:
        return row
    key = span_key(item.audio_filename, item.start_sample, item.end_sample)
    if key not in scores:
        raise ValueError(f"{key} was never scored: re-run spans and score")
    found = scores[key]
    return {**row, "cut": cut_of(item), "span": [item.start_sample, item.end_sample],
            "excerpt_start_sample": item.excerpt_start_sample, "scores": found,
            "detector_trimmed": None if decider is None
            else decider[0].fires(found[decider[0].name], decider[1])}


def resolve(row: Mapping, verdict: Verdict | None) -> tuple[str, str | None]:
    """An item's status and who decided it: an owner note first (on any item), else the
    detector where it decides for an exposed item."""
    owner = owner_says(verdict.note) if verdict is not None else None
    if owner is not None:
        return (ONSET_TRIMMED if owner else SITE), OWNER_NOTE
    if row.get("detector_trimmed") is None:
        return SITE, None
    return (ONSET_TRIMMED if row["detector_trimmed"] else SITE), DETECTOR


def trimmed_ids(guard_path: Path = GUARD_PATH,
                verdicts_path: Path = VERDICTS_PATH) -> frozenset[str]:
    """Every item ``onset_trimmed`` at the current verdicts."""
    verdicts = read_verdicts(verdicts_path)
    return frozenset(row["item_id"] for row in read_jsonl(guard_path)
                     if resolve(row, verdicts.get(row["item_id"]))[0] == ONSET_TRIMMED)


# --- the real-world check and the summary ------------------------------------------------------


def calls(row: Mapping, calibration: Mapping) -> list[str]:
    """The candidates that call an exposed item trimmed at their operating points."""
    found = calibration["candidates"]
    return [c.name for c in CANDIDATES if found[c.name]["operating_point"] is not None
            and c.fires(row["scores"][c.name], found[c.name]["operating_point"])]


def owner_check(rows: Sequence[Mapping], verdicts: Mapping[str, Verdict],
                calibration: Mapping) -> dict:
    """Each candidate at its operating point against the owner: exposed items with a trim note
    are positives, answered exposed items without one negatives (weak unless the note says
    clean). The chosen candidate is contradicted when it calls trimmed less than
    :data:`MIN_OWNER_FLAG_RECALL` of the flags, or its precision here is below
    :data:`MIN_PRECISION`. The negatives any candidate calls trimmed are listed for
    re-listening, with the candidates that do."""
    judged = [r for r in rows if r["exposure"] == EXPOSED and r["item_id"] in verdicts]
    says = [owner_says(verdicts[r["item_id"]].note) for r in judged]
    truth = [s is True for s in says]
    called = [calls(r, calibration) for r in judged]
    candidates = {c.name: point((c.name in k for k in called), truth).as_dict()
                  for c in CANDIDATES
                  if calibration["candidates"][c.name]["operating_point"] is not None}
    found = candidates.get(calibration["chosen"])
    return {"answered_exposed": len(judged), "owner_flags": sum(truth),
            "owner_clean_notes": sum(s is False for s in says),
            "weak_negatives": sum(s is None for s in says), "candidates": candidates,
            "contradicted": found is None or found["recall"] is None
            or found["recall"] < MIN_OWNER_FLAG_RECALL
            or found["precision"] is None or found["precision"] < MIN_PRECISION,
            "relisten": [{"item_id": r["item_id"], "candidates": k}
                         for r, k, flagged in sorted(zip(judged, called, truth),
                                                     key=lambda x: (-len(x[1]), x[0]["item_id"]))
                         if k and not flagged]}


def decider(calibration: Mapping, check: Mapping) -> tuple[Candidate, float] | None:
    """The candidate and threshold that decide, or ``None`` (module docstring)."""
    if check["contradicted"]:
        return None
    candidate = next(c for c in CANDIDATES if c.name == calibration["chosen"])
    return candidate, calibration["candidates"][candidate.name]["operating_point"]


def _tally(rows: Sequence[Mapping], verdicts: Mapping[str, Verdict], calibration: Mapping
           ) -> dict:
    counts = Counter()
    for row in rows:
        status, decided_by = resolve(row, verdicts.get(row["item_id"]))
        counts["items"] += 1
        counts[row["exposure"]] += 1
        counts[row.get("cut")] += 1
        if row["exposure"] == EXPOSED:
            counts["chosen_candidate_calls_trimmed"] += calibration["chosen"] in calls(
                row, calibration)
        counts["detector_trimmed"] += row.get("detector_trimmed") is True
        counts["decided_by_owner_note"] += decided_by == OWNER_NOTE
        counts[status] += 1
    return {k: counts[k] for k in ("items", EXPOSED, FILE_START, SEGMENTER_CUT,
                                   "chosen_candidate_calls_trimmed", "detector_trimmed",
                                   "decided_by_owner_note", ONSET_TRIMMED, SITE)}


def summarize(rows: Sequence[Mapping], verdicts: Mapping[str, Verdict], calibration: Mapping
              ) -> dict:
    """Counts per part and per stratum (the word sample's per-reciter strata as one), and the
    owner's notes, at ``verdicts``."""
    by_part, by_stratum = defaultdict(list), defaultdict(list)
    for row in rows:
        for part in row["parts"]:
            by_part[part].append(row)
        by_stratum[WORD_SAMPLE if row["mark"] == WORD else row["stratum"]].append(row)
    notes = Counter()
    for row in rows:
        verdict = verdicts.get(row["item_id"])
        says = owner_says(verdict.note) if verdict is not None else None
        if says is not None:
            notes[f"{'trim' if says else 'clean'} note, {row['exposure']}"] += 1
    return {"parts": {p: _tally(r, verdicts, calibration) for p, r in sorted(by_part.items())},
            "strata": {s: _tally(r, verdicts, calibration)
                       for s, r in sorted(by_stratum.items())},
            "all": _tally(rows, verdicts, calibration),
            "owner_notes": dict(sorted(notes.items()))}


def build(verdicts_path: Path = VERDICTS_PATH) -> dict:
    """Calibrate the candidates, check them against the owner, decide every item, and write
    the sidecar, the curves and the summary."""
    scored = read_jsonl(SCORES_PATH)
    synthetic = read_jsonl(SYNTHETIC_PATH)
    expected = spans(all_items(), synthetic)
    if [{k: v for k, v in r.items() if k != "scores"} for r in scored] != expected:
        raise ValueError("the scored spans are not this build's: re-run spans and score")
    calibration, curves = calibrate([r for r in scored if r["kind"] == SYNTHETIC])
    cut_scores = {r["key"]: r["scores"] for r in scored if r["kind"] == CUT}
    items = all_items()
    verdicts = read_verdicts(verdicts_path)
    unguarded = [guard_row(item, cut_scores, None) for item in items]
    check = owner_check(unguarded, verdicts, calibration)
    deciding = decider(calibration, check)
    rows = [guard_row(item, cut_scores, deciding) for item in items]
    write_jsonl(GUARD_PATH, rows)
    write_text_atomically(CURVES_PATH, json.dumps(curves, indent=1, sort_keys=True) + "\n")
    summary = {
        "treatment": "an exposed item judged trimmed (by an owner note first, else by the "
                     "detector where it decides) is onset_trimmed: an excerpt fault, out of "
                     "every evaluation denominator and training-target export; its verdict "
                     "stays on record",
        "inputs": {name: sha256_of(path) for name, path in (
            ("ship_a/sites.jsonl", SITES_PATH), ("ship_a/worklist.jsonl", WORKLIST_PATH),
            ("ship_a/word_sample.jsonl", WORD_SAMPLE_PATH),
            ("ship_a/onset_guard.synthetic.jsonl", SYNTHETIC_PATH),
            ("ship_a/onset_guard.scores.jsonl", SCORES_PATH),
            ("listening_session/verdicts.jsonl", verdicts_path))},
        "scoring": json.loads(RUN_PATH.read_text(encoding="utf-8")),
        "synthetic_selection": json.loads(SYNTHETIC_RUN_PATH.read_text(encoding="utf-8")),
        "calibration": calibration,
        "owner_check": {**check, "min_flag_recall": MIN_OWNER_FLAG_RECALL,
                        "min_precision": MIN_PRECISION, "history": OWNER_CHECK_HISTORY},
        "detector_decides": None if deciding is None
        else {"candidate": deciding[0].name, "threshold": deciding[1]},
        **summarize(rows, verdicts, calibration),
    }
    write_text_atomically(SUMMARY_PATH, json.dumps(summary, indent=1, ensure_ascii=False,
                                                   sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic", help="pick the clean clips to trim (reads audio)")
    synthetic.add_argument("--audio-dir", type=Path, action="append", required=True)
    listed = commands.add_parser("spans", help="list every span the detector scores")
    listed.add_argument("--out", type=Path, required=True)
    score = commands.add_parser("score", help="score every listed span (GPU)")
    score.add_argument("--spans", type=Path, required=True)
    score.add_argument("--audio-dir", type=Path, action="append", required=True,
                       help="a directory of staged clips (repeatable; the first holding a "
                            "clip is used)")
    score.add_argument("--device", default="cuda")
    score.add_argument("--vad-dtype", default="bfloat16")
    commands.add_parser("build", help="write the sidecar and its summary")
    args = parser.parse_args()
    if args.command == "synthetic":
        rows, record = synthetic_clips(args.audio_dir)
        write_jsonl(SYNTHETIC_PATH, rows)
        write_text_atomically(SYNTHETIC_RUN_PATH, json.dumps(record, indent=1, sort_keys=True)
                              + "\n")
        print(json.dumps(record))
    elif args.command == "spans":
        rows = spans(all_items(), read_jsonl(SYNTHETIC_PATH))
        write_jsonl(args.out, rows)
        print(f"{len(rows)} spans ({sum(r['kind'] == CUT for r in rows)} exposed segments)")
    elif args.command == "score":
        rows, provenance = score_spans(read_jsonl(args.spans), args.audio_dir, args.device,
                                       args.vad_dtype)
        write_jsonl(SCORES_PATH, rows)
        write_text_atomically(RUN_PATH, json.dumps(provenance, indent=1, sort_keys=True) + "\n")
        print(json.dumps(provenance))
    else:
        summary = build()
        print(json.dumps({"all": summary["all"], "chosen": summary["calibration"]["chosen"],
                          "detector_decides": summary["detector_decides"],
                          "owner_check": summary["owner_check"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

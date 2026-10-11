"""Ship A's random word sample (#126): whole words judged by ear, drawn with no model.

ADR-0012 judges a candidate by the app outcome, and its false flags on correctly recited
words, and the new errors it introduces, need a denominator no model chose. This module
draws it: whole words of held-out reciters, each asked one question, **was this word
recited correctly?** (:data:`WORD_VERDICTS`: ``correct``, ``mistake`` or ``unclear``, the
acceptance rules' sides; ``unclear`` leaves every denominator).

**Frame.** Every Uthmani word of every kept segment of the mining pool's ``uniform`` clips
(#83: ``k`` of a drawn reciter's ``n`` eligible clips by salted hash) whose canonical reciter
is in Ship A's held-out half (:func:`tadabur.ship_a.reciter_half`), less the clips tied to a
recording ``h448`` trained on, ``decode_evalset`` or the sealed panel, even by copy
(:func:`tadabur.power_frame.unseen_clips`). Sealed-panel reciters are refused (the exposure registry's :func:`tadabur.exposure.excluded_reciters`; the pool is
reciter-disjoint from the panel already). The census strata of the pool (clips where the
base teacher heard a pair swap or a gemination mismatch) are left out, since a decode
selected them. No decode enters the draw; the segments are #83's, whose pause placement and
drop rules read the base teacher's decode, as every pool site's do.

**Draw.** Reciters of the frame are ranked by ``sha256("<salt>:reciter:<id>")`` and the first
:data:`RECITERS` taken; from each, the word ranked first by ``sha256("<salt>:word:<word id>")``.
One word per drawn reciter, so the verdicts are independent and equally weighted (the
Wilson and Tango bounds apply, acceptance rules §1).

**Inclusion probability** of a word: its clip's chance in the uniform draw (``k / n``) times the
word's chance here, ``R / H`` (``R`` of the ``H`` frame reciters drawn) times ``1 / N_r`` (one of
its reciter's ``N_r`` frame words). Like #87's, it is conditional on the realized pool.

**Excerpt and time.** The word and one word either side, as #87 places every excerpt
(:func:`tadabur.listening_session.excerpt_span`); the whole segment where the clip's word
times cannot place them. A whole-word judgement weighs every letter and mark, so it is
estimated at the excerpt played twice plus :data:`ANSWER_SECONDS` (#87's site question:
twice plus 4 s).

Files (``tools/tadabur/ship_a/``): ``word_sample.jsonl`` (one word per line, its provenance,
place and design) and ``word_sample.summary.json`` (the frame per reciter, the draw, minutes).

Usage (from ``tools/``; torch-free, a few seconds)::

  python -m tadabur.word_sample draw
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE
from training.tashkeel_eval import write_text_atomically

from .audio import TARGET_SAMPLE_RATE
from .exposure import MINING_POOL, excluded_reciters
from .listening_session import POOL_CLIPS_PATH, PLAYS_PER_SITE, excerpt_span
from .mining_pool import FRAME_PATH as POOL_FRAME_PATH
from .mining_pool import UNIFORM, PoolClip, PoolSegment, load_manifest
from .power_frame import unseen_clips
from .ship_a import HELD_OUT, SHIP_A_DIR, reciter_half
from .staged_audio import StagedClip, load_staged_clips
from .truth_sites import UNCLEAR

WORD_SAMPLE_PATH = SHIP_A_DIR / "word_sample.jsonl"
WORD_SUMMARY_PATH = SHIP_A_DIR / "word_sample.summary.json"

SALT = "ship-a-word-sample-2026-10"
#: Reciters drawn, one word each: about 38 minutes at the frame's mean time per word.
RECITERS = 160
#: Seconds to judge a whole word after hearing it (#87's one-mark question takes 4).
ANSWER_SECONDS = 6.0

WORD_MODE = "word"
WORD_VERDICTS = (CORRECT_SIDE, MISTAKE_SIDE, UNCLEAR)


def _rank(*parts) -> str:
    return hashlib.sha256(":".join((SALT, *map(str, parts))).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SampledWord:
    """What a word verdict is about: the clip (with its provenance), the segment it is
    recited in (span and realized reference), the word (its ayah word index and where it
    starts in the reference), and the stratum (its reciter) with that stratum's population."""

    site_id: str
    audio_filename: str
    shard: int
    audio_sha256: str
    reciter_id: int
    surah_ayah: str
    segment_index: int
    start_sample: int
    end_sample: int
    reference: str
    word_index: int
    reference_index: int
    stratum: str
    stratum_population: int


@dataclass(frozen=True)
class WordSite:
    """One drawn word as the listening session asks it: the word, its sampling design, the
    segment's word offsets and the excerpt (the shape of a worklist row)."""

    site: SampledWord
    clip_inclusion_probability: float
    draw_probability: float
    inclusion_probability: float
    word_start: int
    word_offsets: tuple[int, ...]
    excerpt_start_sample: int
    excerpt_end_sample: int

    @property
    def mode(self) -> str:
        return WORD_MODE

    @property
    def word_span(self) -> tuple[int, int]:
        """The word's ``[start, end)`` in the reference."""
        k = self.site.word_index - self.word_start
        return self.word_offsets[k], self.word_offsets[k + 1]

    def listening_seconds(self) -> float:
        excerpt = (self.excerpt_end_sample - self.excerpt_start_sample) / TARGET_SAMPLE_RATE
        return PLAYS_PER_SITE * excerpt + ANSWER_SECONDS


def word_id(audio_filename: str, segment_index: int, word_index: int) -> str:
    """A model-free id: the clip, the segment and the word's index in its ayah."""
    return f"word_sample:{audio_filename}#{segment_index}@{word_index}"


def uniform_probabilities(pool_frame: dict) -> dict[int, float]:
    """Per drawn reciter, a clip's chance in the uniform stratum: ``k / n`` (#83)."""
    return {
        int(reciter): entry["uniform"] / entry["eligible"]
        for reciter, entry in pool_frame["per_reciter"].items()
        if entry["drawn"]
    }


@dataclass(frozen=True)
class FrameWord:
    """One word of the frame before the draw: its clip, its segment and its ayah index."""

    clip: PoolClip
    segment: PoolSegment
    word_index: int

    @property
    def word_id(self) -> str:
        return word_id(self.clip.audio_filename, self.segment.segment_index, self.word_index)

    @property
    def reference_index(self) -> int:
        """Where the word starts in its segment's realized reference."""
        return self.segment.raw_word_offsets[self.word_index - self.segment.word_start]


def frame_words(clips: list[PoolClip], barred: frozenset[int]) -> dict[int, list[FrameWord]]:
    """Per held-out reciter, every word of its uniform clips' kept segments (module doc)."""
    words: dict[int, list[FrameWord]] = {}
    for clip in clips:
        if UNIFORM not in clip.strata or reciter_half(clip.reciter_id) != HELD_OUT:
            continue
        if clip.reciter_id in barred:
            raise ValueError(f"{clip.audio_filename}: a sealed-panel reciter is in the pool")
        for segment in clip.segments:
            if segment.kept:
                words.setdefault(clip.reciter_id, []).extend(
                    FrameWord(clip, segment, w)
                    for w in range(segment.word_start, segment.word_end)
                )
    return dict(sorted(words.items()))


def draw(
    words: dict[int, list[FrameWord]],
    uniform: dict[int, float],
    registry: dict[str, StagedClip],
    reciters: int = RECITERS,
) -> list[WordSite]:
    """One word from each of the first ``reciters`` frame reciters in hash order, sorted by id."""
    drawn = sorted(words, key=lambda r: _rank("reciter", r))[:reciters]
    reciter_share = len(drawn) / len(words)
    rows = []
    for reciter in drawn:
        word = min(words[reciter], key=lambda w: _rank("word", w.word_id))
        clip, segment, staged = word.clip, word.segment, registry[word.clip.audio_filename]
        excerpt = excerpt_span(segment, clip.word_times, clip.re_reads, word.reference_index,
                               staged.num_samples)
        draw_probability = reciter_share / len(words[reciter])
        rows.append(WordSite(
            site=SampledWord(
                site_id=word.word_id,
                audio_filename=clip.audio_filename,
                shard=staged.shard,
                audio_sha256=staged.audio_sha256,
                reciter_id=reciter,
                surah_ayah=clip.surah_ayah,
                segment_index=segment.segment_index,
                start_sample=segment.start_sample,
                end_sample=segment.end_sample,
                reference=segment.reference,
                word_index=word.word_index,
                reference_index=word.reference_index,
                stratum=f"word_sample:reciter:{reciter}",
                stratum_population=len(words[reciter]),
            ),
            clip_inclusion_probability=uniform[reciter],
            draw_probability=draw_probability,
            inclusion_probability=uniform[reciter] * draw_probability,
            word_start=segment.word_start,
            word_offsets=segment.raw_word_offsets,
            excerpt_start_sample=excerpt[0],
            excerpt_end_sample=excerpt[1],
        ))
    return sorted(rows, key=lambda row: row.site.site_id)


# --- the file --------------------------------------------------------------------------------

_SITE_FIELDS = tuple(f.name for f in fields(SampledWord))
_DESIGN_FIELDS = tuple(f.name for f in fields(WordSite) if f.name != "site")


def word_json(row: WordSite) -> dict:
    return {**asdict(row.site), **{name: getattr(row, name) for name in _DESIGN_FIELDS},
            "word_offsets": list(row.word_offsets)}


def parse_word(data: dict, where: str) -> WordSite:
    """One validated word row: every field present with its type, the word inside its
    segment's reference, the design consistent, the excerpt a span of samples."""
    if not isinstance(data, dict) or set(data) != {*_SITE_FIELDS, *_DESIGN_FIELDS}:
        raise ValueError(f"{where}: a word row has exactly the fields {sorted(data)}")
    row = WordSite(SampledWord(**{k: data[k] for k in _SITE_FIELDS}),
                   **{**{k: data[k] for k in _DESIGN_FIELDS},
                      "word_offsets": tuple(data["word_offsets"])})
    site, offsets = row.site, row.word_offsets
    if not (offsets and all(type(o) is int for o in offsets) and offsets[0] == 0
            and offsets[-1] == len(site.reference) and list(offsets) == sorted(offsets)):
        raise ValueError(f"{where}: word_offsets do not partition the reference")
    k = site.word_index - row.word_start
    if not (0 <= k < len(offsets) - 1 and offsets[k] == site.reference_index
            and offsets[k] < offsets[k + 1]):
        raise ValueError(f"{where}: word {site.word_index} is not a word of the segment")
    probabilities = (row.clip_inclusion_probability, row.draw_probability,
                     row.inclusion_probability)
    if not all(type(p) is float and 0 < p <= 1 for p in probabilities) or abs(
            row.inclusion_probability - row.clip_inclusion_probability * row.draw_probability
    ) > 1e-12:
        raise ValueError(f"{where}: inclusion probabilities are not floats in (0, 1], clip x draw")
    if not (0 <= site.start_sample < site.end_sample
            and 0 <= row.excerpt_start_sample < row.excerpt_end_sample):
        raise ValueError(f"{where}: the segment or the excerpt is not a span of samples")
    if site.site_id != word_id(site.audio_filename, site.segment_index, site.word_index):
        raise ValueError(f"{where}: site_id does not name the clip, segment and word")
    return row


def write_words(rows: list[WordSite], path: Path = WORD_SAMPLE_PATH) -> None:
    """Validate every row (ids unique), then write atomically, one JSON object per line."""
    for number, row in enumerate(rows, 1):
        parse_word(word_json(row), f"write_words[{number}]")
    if len({row.site.site_id for row in rows}) != len(rows):
        raise ValueError("write_words: two rows share a site id")
    write_text_atomically(path, "".join(
        json.dumps(word_json(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def load_words(path: Path = WORD_SAMPLE_PATH) -> list[WordSite]:
    """The word sample in file order, every row checked and ids unique."""
    with open(path, encoding="utf-8") as f:
        rows = [parse_word(json.loads(raw), f"{path}:{n}") for n, raw in enumerate(f, 1)
                if raw.strip()]
    if len({row.site.site_id for row in rows}) != len(rows):
        raise ValueError(f"{path}: two rows share a site id")
    return rows


def summarize(words: dict[int, list[FrameWord]], rows: list[WordSite]) -> dict:
    """The frame per reciter, the draw and the estimated minutes."""
    minutes = sum(row.listening_seconds() for row in rows) / 60
    return {
        "salt": SALT,
        "listening_model": {"plays_per_word": PLAYS_PER_SITE, "answer_seconds": ANSWER_SECONDS},
        "frame": {
            "reciters": len(words),
            "words": sum(map(len, words.values())),
            "words_per_reciter": {str(r): len(w) for r, w in words.items()},
        },
        "drawn_reciters": len(rows),
        "words": len(rows),
        "listening_minutes": round(minutes, 1),
        "excerpts_whole_segment": sum(
            (r.excerpt_start_sample, r.excerpt_end_sample) == (r.site.start_sample,
                                                               r.site.end_sample)
            for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("draw", help="draw the word sample and write it with its summary")
    parser.parse_args()
    registry = load_staged_clips()
    clips, _ = unseen_clips(load_manifest(POOL_CLIPS_PATH, registry), registry)
    words = frame_words(clips, excluded_reciters(MINING_POOL))
    uniform = uniform_probabilities(json.loads(POOL_FRAME_PATH.read_text(encoding="utf-8")))
    rows = draw(words, uniform, registry)
    write_words(rows)
    summary = summarize(words, rows)
    write_text_atomically(WORD_SUMMARY_PATH, json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "frame"}, indent=1))


if __name__ == "__main__":
    main()

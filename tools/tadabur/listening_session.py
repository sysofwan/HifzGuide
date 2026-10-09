"""The owner's listening session (#87 for #61): which sites to listen to, and the verdicts.

ADR-0011 judges every model against truth sites (:mod:`tadabur.truth_sites`). This module
mines the sites the owner's one sitting adjudicates by ear, from the mining pool (#83), and
stores what was heard.

**What is mined.** Eligibility is decided by the realized reference alone (acceptance rules
§1). Two frozen decodes of each kept pool segment only assign an eligible site to a
**stratum**: the base teacher's (``mining_pool/base_decodes.json``) and, for tashkeel, the
shipped student's, ``h448`` streamed at ``b=0`` with no bias (:mod:`tadabur.pool_stream`),
because that is today's system in the ship criterion (§3). No candidate model is involved,
so the selection never has to be redone for a new checkpoint.

* **tashkeel** (one question for haraka and sukun: "which mark did you hear"), stratified by
  both decodes:

  - a **mid-word haraka** on a consonant, per haraka. Each decode left the slot empty
    (``empty``: the carrier aligned, no haraka after it), emitted that haraka
    (``matched``), or anything else (``other``: another haraka, or a misheard or unaligned
    carrier), read by :func:`training.tashkeel_eval.vowel_sites`' carrier anchoring;
  - a **mid-word prescribed sukun**: a single consonant followed, inside its word, by another
    consonant or the qalqala mark, so the mushaf writes sukun on it. Each decode emitted
    nothing after the aligned carrier (``empty``), a haraka (``haraka``), or misheard it
    (``other``) (:func:`carrier_marks`).

  *Mid-word* means a letter of the same word follows the mark (``raw_word_offsets`` give the
  words). A word whose Uthmani text carries tanween ends at its last haraka: the ``ن`` /
  ``ں`` / assimilated letter after it is the tanween's realization, not a letter of the
  word, so the case ending is never mid-word. Word-final marks depend on waqf and wasl and
  are out of scope (the weak-label rows of acceptance rules §1).
* **shaddah**: every reference geminate (the first of the doubled consonant) inside one word,
  where the base decoded one consonant (``held:base_single``) or not (``held:base_rest``). A
  geminate whose halves fall in two words (idgham across a word boundary) is its own stratum,
  ``held:cross_word``, drawn 0 by default: it exists only if the reciter did not pause there,
  and #86 found the segmentation missed a real pause at 18 of 20 such sites it sampled, so a
  "not held" there is usually a correct pause. And every single
  consonant, where the base doubled it (``not_held:base_double``, the site of an added
  shaddah) or not (``not_held:base_rest``). Gemination mismatches come from
  :func:`tadabur.contrast_attribution.contrast_sites`, the rule the pool's census used.
* **consonant**: per target pair (the six soft pairs and ``ذ↔ظ``, acceptance rules §7) and
  **per direction**, every carrier of the prescribed letter where the base decoded the
  partner (``base_partner``, the reject-pile sites) or did not (``base_rest``).

Plus the 23 nominal P3.5 rejects (``truth_sites/p35_fixtures.jsonl``, ``heard: pending``),
re-adjudicated at site level (acceptance rules §8). They keep their own site ids and strata.

**How it is drawn.** Within each stratum, sites are ranked by a salted hash of their site id
and the top ``n`` taken (``training.tashkeel_worklist.sample_worklist``'s rule): uniform, and
stable when the population changes, so a re-mine re-draws the same sites and verdicts carry
over. ``n`` per stratum is a parameter (``--sizes``); the post-#84 power simulation (#105)
sets it. :data:`DEFAULT_SIZES` realizes the ~1.5 h plan of #61.

**What each row records.** A row is a truth-site skeleton (``heard: pending``; provenance
from the staged-clip registry; the segment as the item; ``stratum`` and its
``stratum_population`` in the pool) plus the sampling design: the clip's inclusion
probability in the pool (given the drawn reciters, ``mining_pool/clips.jsonl``), the
within-stratum draw probability ``n / N``, their product ``inclusion_probability``, the
segment's words (``word_start``, ``word_offsets``) and the excerpt the UI plays. A site's
design weight is ``1 / inclusion_probability``.

**Verdicts** are written by the blind UI (:mod:`tadabur.tashkeel_audit_ui`) to
``listening_session/verdicts.jsonl``, a tracked file, keyed by site id. A verdict's ``heard``
takes the truth-site vocabulary of the site's mark; a synthetic-edit check item's verdict
(:mod:`tadabur.edit_check`, #107) also records whether it sounded ``natural``.

**Audio.** ``fetch`` copies every clip the worklist plays and every edit-check item from
the GPU box into one local directory with a single rsync, and verifies each checksum.

Usage (from ``tools/``; torch-free, needs ``quran-transcript`` for the Uthmani words)::

  python -m tadabur.listening_session mine --p35-seg-dir stage/seg_p35 [--sizes sizes.json]
  python -m tadabur.listening_session clips    # the clip names the worklist plays
  python -m tadabur.listening_session fetch --audio-dir ~/hifzguide-listening/clips
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import posixpath
import random
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from training.tashkeel_eval import MATCHED as VOWEL_MATCHED
from training.tashkeel_eval import (
    OMITTED,
    SHORT_VOWELS,
    vowel_sites,
    write_text_atomically,
)

from .audio import TARGET_SAMPLE_RATE
from .contrast_attribution import ADDED, DROPPED, SHADDA_CONTRAST, contrast_sites
from .mining_pool import (
    CLIPS_PATH as POOL_CLIPS_PATH,
    DECODES_PATH as POOL_DECODES_PATH,
    PoolClip,
    PoolSegment,
    load_base_decodes,
    load_manifest,
    segment_key,
)
from .pool_stream import H448_DECODES_PATH, load_stream_decodes
from .smith_waterman import smith_waterman
from .staged_audio import StagedClip, load_staged_clips, verify_staged
from .truth_sites import (
    CONSONANTS,
    HARAKA_CHARS,
    HELD,
    NEW_AUDIT,
    NOT_HELD,
    PENDING,
    SHADDAH,
    SUKUN,
    TARGET_PAIRS,
    TASHKEEL_MARKS,
    TRUTH_SITES_DIR,
    UNCLEAR,
    TruthSite,
    _check_file,
    label_states,
    load_truth_sites,
    parse_site,
)

if TYPE_CHECKING:
    from .edit_check import EditCheckSite

SESSION_DIR = Path(__file__).parent / "listening_session"
WORKLIST_PATH = SESSION_DIR / "worklist.jsonl"
SUMMARY_PATH = SESSION_DIR / "worklist.summary.json"
VERDICTS_PATH = SESSION_DIR / "verdicts.jsonl"
P35_SITES_PATH = TRUTH_SITES_DIR / "p35_fixtures.jsonl"

SALT = "issue-87-listening-session-v1"

# --- the three questions the UI asks ---------------------------------------------------
TASHKEEL_MODE = "tashkeel"
SHADDAH_MODE = "shaddah"
CONSONANT_MODE = "consonant"


def mode_of(mark: str) -> str:
    """The question a site's mark is asked as: haraka and sukun share one, so the page
    cannot tell a prescribed sukun from a prescribed haraka."""
    if mark in TASHKEEL_MARKS:
        return TASHKEEL_MODE
    return SHADDAH_MODE if mark == SHADDAH else CONSONANT_MODE


# --- strata ----------------------------------------------------------------------------
#: What a decode did at a tashkeel site: left it empty, emitted the prescribed haraka,
#: emitted a haraka at a prescribed sukun, or anything else.
EMPTY = "empty"
MATCHED = "matched"
HARAKA = "haraka"
OTHER = "other"
HARAKA_OUTCOMES = (EMPTY, MATCHED, OTHER)
SUKUN_OUTCOMES = (EMPTY, HARAKA, OTHER)
#: The two frozen decodes a tashkeel stratum is crossed by.
BASE = "base"
H448 = "h448"

BASE_SINGLE = "base_single"
BASE_DOUBLE = "base_double"
BASE_PARTNER = "base_partner"
BASE_REST = "base_rest"
#: A geminate whose two halves fall in different words (#86): see :func:`_crosses_words`.
CROSS_WORD = "cross_word"

HARAKA_NAMES = tuple(sorted(HARAKA_CHARS))
_CHAR_TO_HARAKA = {char: name for name, char in HARAKA_CHARS.items()}
#: The qalqala mark the phonetizer writes after a sakin qalqala letter: a sukun carrier.
QALQALA = "ڇ"
#: Madd letters: after a haraka they lengthen it, so they reveal it on the page.
MADD = frozenset("اۥۦ")
_TANWEEN = frozenset("\u064b\u064c\u064d")


def _stratum(*parts: str) -> str:
    return ":".join((NEW_AUDIT, *parts))


def tashkeel_stratum(mark: str, base: str, h448: str) -> str:
    """``new_audit:<mark>:base_<outcome>:h448_<outcome>``."""
    return _stratum(mark, f"{BASE}_{base}", f"{H448}_{h448}")


def _direction(prescribed: str, partner: str) -> str:
    return f"{prescribed}→{partner}"


def _pair_directions() -> list[tuple[str, str, str]]:
    """``(pair, prescribed, partner)`` for both directions of every target pair."""
    return [
        (pair, a, b)
        for pair in sorted(TARGET_PAIRS)
        for a, b in (pair.split("↔"), reversed(pair.split("↔")))
    ]


STRATA: tuple[str, ...] = (
    *(tashkeel_stratum(h, b, s) for h in HARAKA_NAMES
      for b in HARAKA_OUTCOMES for s in HARAKA_OUTCOMES),
    *(tashkeel_stratum(SUKUN, b, s) for b in SUKUN_OUTCOMES for s in SUKUN_OUTCOMES),
    _stratum(SHADDAH, HELD, BASE_SINGLE),
    _stratum(SHADDAH, HELD, BASE_REST),
    _stratum(SHADDAH, HELD, CROSS_WORD),
    _stratum(SHADDAH, NOT_HELD, BASE_DOUBLE),
    _stratum(SHADDAH, NOT_HELD, BASE_REST),
    *(_stratum(_direction(a, b), o) for _, a, b in _pair_directions()
      for o in (BASE_PARTNER, BASE_REST)),
)

#: Default draw per haraka, by (base outcome, h448 outcome). Weighted toward the slots the
#: shipped model leaves empty, above all where the base teacher heard the haraka (the
#: failure the owner sees in the app); every other cell keeps a small positive draw, so each
#: outcome stratum has positive inclusion probability. A cell smaller than its size is
#: taken whole.
HARAKA_SIZES = {
    (EMPTY, EMPTY): 15, (MATCHED, EMPTY): 25, (OTHER, EMPTY): 5,
    (EMPTY, MATCHED): 5, (MATCHED, MATCHED): 10, (OTHER, MATCHED): 3,
    (EMPTY, OTHER): 3, (MATCHED, OTHER): 4, (OTHER, OTHER): 3,
}
#: The same for prescribed sukun: weighted toward a haraka the shipped model adds, with the
#: concordant-empty cell as the controls the sukun floors are read on.
SUKUN_SIZES = {
    (EMPTY, EMPTY): 15, (EMPTY, HARAKA): 15, (EMPTY, OTHER): 5,
    (HARAKA, EMPTY): 4, (HARAKA, HARAKA): 8, (HARAKA, OTHER): 2,
    (OTHER, EMPTY): 3, (OTHER, HARAKA): 2, (OTHER, OTHER): 3,
}

#: Sites drawn per stratum by default, sized to the ~1.5 h plan of #61 with the parts #82
#: added; #105's power simulation sets the final sizes. Tashkeel as above; 60 within-word
#: geminates the base decoded single (cross-word ones are 0, #86); 5 per consonant direction the base heard as its partner (~40 in
#: all, the rare directions take what they have). Every other stratum is 0: its population
#: is recorded, and #105 may size it.
DEFAULT_SIZES: dict[str, int] = {
    **{tashkeel_stratum(h, b, s): n for h in HARAKA_NAMES for (b, s), n in HARAKA_SIZES.items()},
    **{tashkeel_stratum(SUKUN, b, s): n for (b, s), n in SUKUN_SIZES.items()},
    _stratum(SHADDAH, HELD, BASE_SINGLE): 60,
    **{_stratum(_direction(a, b), BASE_PARTNER): 5 for _, a, b in _pair_directions()},
}

#: Listening-time model for the estimate in the summary: each site's excerpt is played
#: this many times, plus a fixed time to answer.
PLAYS_PER_SITE = 2
ANSWER_SECONDS = 4.0

#: An excerpt is the carrier's word with one word either side, padded by this much.
EXCERPT_PAD_S = 0.3
#: A shorter excerpt is taken as a failed word alignment: the whole segment plays instead.
MIN_EXCERPT_S = 1.0


# --- finding sites in one segment (pure, over strings) ---------------------------------


@dataclass(frozen=True)
class Found:
    """One eligible site in a segment: its stratum and what the truth site will test."""

    stratum: str
    mark: str
    prescribed: str
    reference_index: int


def carrier_marks(decode: str, reference: str) -> dict[int, str | None]:
    """For each reference consonant the base decode matched, the haraka it emitted right
    after it (``None`` for none). Carriers the decode misheard or never reached are absent."""
    alignment = smith_waterman(decode, reference)
    marks: dict[int, str | None] = {}
    for local, query in enumerate(alignment.ref_to_query):
        index = alignment.ref_start + local
        if reference[index] in CONSONANTS and query >= 0 and decode[query] == reference[index]:
            following = decode[query + 1] if query + 1 < len(decode) else None
            marks[index] = following if following in SHORT_VOWELS else None
    return marks


def word_limits(reference: str, offsets: list[int], tanween: list[bool]) -> list[int]:
    """Per reference position, the end of the letters its word owns.

    ``offsets`` are the segment's ``raw_word_offsets`` (word ``j`` spans ``[offsets[j],
    offsets[j + 1])``, the last is the reference's length). A word owns its span, except
    that a word with tanween owns only up to its last haraka (the case ending): what follows
    is the tanween's realization."""
    if len(offsets) != len(tanween) + 1 or offsets[-1] != len(reference):
        raise ValueError("word offsets do not partition the reference into the words given")
    limits = [0] * len(reference)
    for word, (start, end) in enumerate(zip(offsets, offsets[1:])):
        limit = end
        if tanween[word]:
            harakat = [i for i in range(start, end) if reference[i] in SHORT_VOWELS]
            limit = harakat[-1] if harakat else end
        for i in range(start, end):
            limits[i] = limit
    return limits


def _mid_word(reference: str, limits: list[int], mark_end: int) -> bool:
    """Whether a letter of the same word follows a mark ending at ``mark_end``."""
    return any(reference[j] in CONSONANTS for j in range(mark_end + 1, limits[mark_end]))


def run_start(reference: str, index: int) -> int:
    """The first index of the run of identical characters that ``index`` is in."""
    while index > 0 and reference[index - 1] == reference[index]:
        index -= 1
    return index


def haraka_outcomes(decode: str, reference: str) -> dict[int, str]:
    """Per haraka carrier (the letter before the haraka): :data:`MATCHED` when the decode
    emitted that haraka on the aligned carrier, :data:`EMPTY` when the carrier aligned and no
    haraka followed it, else :data:`OTHER`. The anchoring is
    :func:`training.tashkeel_eval.vowel_sites`', the repo's one rule for haraka outcomes."""
    outcomes = {}
    for site in vowel_sites(decode, reference):
        if site.reference_vowel is None:
            continue
        if site.outcome == VOWEL_MATCHED:
            outcome = MATCHED
        elif site.outcome == OMITTED and site.carrier_matched:
            outcome = EMPTY
        else:
            outcome = OTHER
        outcomes[site.reference_index - 1] = outcome
    return outcomes


def sukun_outcomes(decode: str, reference: str) -> dict[int, str]:
    """Per aligned consonant: :data:`EMPTY` when the decode emitted no haraka right after
    it, :data:`HARAKA` when it did. A carrier missing here is :data:`OTHER`."""
    return {
        index: EMPTY if mark is None else HARAKA
        for index, mark in carrier_marks(decode, reference).items()
    }


def _tashkeel_found(
    reference: str, decodes: Mapping[str, str], limits: list[int]
) -> list[Found]:
    harakat = {model: haraka_outcomes(d, reference) for model, d in decodes.items()}
    sukun = {model: sukun_outcomes(d, reference) for model, d in decodes.items()}
    found: list[Found] = []
    for i, carrier in enumerate(reference):
        if carrier not in CONSONANTS:
            continue
        following = reference[i + 1] if i + 1 < len(reference) else ""
        if following in SHORT_VOWELS and _mid_word(reference, limits, i + 1):
            name = _CHAR_TO_HARAKA[following]
            stratum = tashkeel_stratum(name, harakat[BASE][i], harakat[H448][i])
            found.append(Found(stratum, name, name, i))
        single = run_start(reference, i) == i and following != carrier
        if single and (following in CONSONANTS or following == QALQALA) \
                and _mid_word(reference, limits, i):
            stratum = tashkeel_stratum(SUKUN, sukun[BASE].get(i, OTHER),
                                       sukun[H448].get(i, OTHER))
            found.append(Found(stratum, SUKUN, SUKUN, i))
    return found


def _crosses_words(reference: str, offsets: list[int], start: int) -> bool:
    """Whether the run of identical characters starting at ``start`` spans a word boundary:
    an idgham across words, which exists only if the reciter did not pause there."""
    end = start
    while end < len(reference) and reference[end] == reference[start]:
        end += 1
    return bisect.bisect_right(offsets, start) != bisect.bisect_right(offsets, end - 1)


def _shaddah_found(reference: str, decode: str, offsets: list[int]) -> list[Found]:
    changes = contrast_sites(decode, reference, SHADDA_CONTRAST)
    dropped = {run_start(reference, s.reference_index) for s in changes if s.change == DROPPED}

    def held(i: int) -> str:
        if _crosses_words(reference, offsets, i):
            return _stratum(SHADDAH, HELD, CROSS_WORD)
        return _stratum(SHADDAH, HELD, BASE_SINGLE if i in dropped else BASE_REST)

    found = [
        Found(held(i), SHADDAH, HELD, i)
        for i, char in enumerate(reference[:-1])
        if char in CONSONANTS and reference[i + 1] == char and run_start(reference, i) == i
    ]
    doubled = {s.reference_index for s in changes if s.change == ADDED}
    found += [
        Found(_stratum(SHADDAH, NOT_HELD, BASE_DOUBLE if i in doubled else BASE_REST),
              SHADDAH, NOT_HELD, i)
        for i, char in enumerate(reference)
        if char in CONSONANTS and char not in (reference[i - 1:i], reference[i + 1:i + 2])
    ]
    return found


def _pair_found(reference: str, decode: str) -> list[Found]:
    found: list[Found] = []
    for pair in sorted(TARGET_PAIRS):
        letters = pair.split("↔")
        partnered = {s.reference_index for s in contrast_sites(decode, reference, pair)}
        for i, char in enumerate(reference):
            if char not in letters or run_start(reference, i) != i:
                continue  # a geminate is one site, on its first half
            (partner,) = set(letters) - {char}
            outcome = BASE_PARTNER if i in partnered else BASE_REST
            found.append(Found(_stratum(_direction(char, partner), outcome), pair, char, i))
    return found


def segment_sites(
    reference: str, base: str, h448: str, offsets: list[int], tanween: list[bool]
) -> list[Found]:
    """Every eligible site in one kept segment, stratified by the base decode and, for
    tashkeel, by the ``h448`` b=0 stream's text for the segment too."""
    limits = word_limits(reference, offsets, tanween)
    return (
        _tashkeel_found(reference, {BASE: base, H448: h448}, limits)
        + _shaddah_found(reference, base, offsets)
        + _pair_found(reference, base)
    )


# --- one site's provenance and excerpt -------------------------------------------------


def question(mark: str) -> str:
    """The site-id suffix: the question asked, never the prescribed mark."""
    return "tashkeel" if mark in TASHKEEL_MARKS else mark


def site_id(audio_filename: str, segment_index: int, reference_index: int, mark: str) -> str:
    """A model-free id: the clip, the segment, the carrier and the question asked there."""
    return f"{NEW_AUDIT}:{audio_filename}#{segment_index}@{reference_index}:{question(mark)}"


def excerpt_span(
    segment: PoolSegment,
    word_times: tuple[float, ...],
    re_reads: int,
    reference_index: int,
    clip_samples: int,
) -> tuple[int, int]:
    """The ``[start, end)`` samples the UI plays: the carrier's word and one word either
    side, padded by :data:`EXCERPT_PAD_S` on each side and clamped only to the clip, so it
    never cuts inside those words.

    The word times are the clip's whole-clip alignment (``ClipStatus.word_times``), the only
    timings the pool has. They place *words*, so an excerpt is a function of the reference
    and that alignment, never of the mark a decode emitted at the carrier. The whole
    segment plays instead when the times cannot place the words: a clip with re-reads (its
    words are recited more than once, so a word's time names one pass), no word times, a
    word span that does not overlap the segment, or an excerpt shorter than
    :data:`MIN_EXCERPT_S`."""
    whole = (segment.start_sample, segment.end_sample)
    if re_reads or len(word_times) <= segment.word_end:
        return whole
    word = segment.word_start + bisect.bisect_right(segment.raw_word_offsets, reference_index) - 1
    first, last = max(segment.word_start, word - 1), min(segment.word_end, word + 2)
    onset, offset = word_times[first] * TARGET_SAMPLE_RATE, word_times[last] * TARGET_SAMPLE_RATE
    if not (onset < offset and onset < segment.end_sample and offset > segment.start_sample):
        return whole
    start = max(0, round(onset - EXCERPT_PAD_S * TARGET_SAMPLE_RATE))
    end = min(clip_samples, round(offset + EXCERPT_PAD_S * TARGET_SAMPLE_RATE))
    if end - start < MIN_EXCERPT_S * TARGET_SAMPLE_RATE:
        return whole
    return start, end


@dataclass(frozen=True)
class Candidate:
    """One eligible site before the draw: a truth-site skeleton whose population is not
    known yet, its clip's pool inclusion probability, its segment's words and its excerpt."""

    site: TruthSite
    clip_inclusion_probability: float
    word_start: int
    word_offsets: tuple[int, ...]
    excerpt: tuple[int, int]


def pool_candidates(
    clips: list[PoolClip],
    base_decodes: Mapping[str, str],
    h448_decodes: Mapping[str, str],
    registry: Mapping[str, StagedClip],
    tanween_of: Callable[[str], list[bool]],
) -> list[Candidate]:
    """Every eligible site on the kept segments of the pool, with provenance."""
    candidates: list[Candidate] = []
    for clip in clips:
        staged = registry[clip.audio_filename]
        tanween = tanween_of(clip.surah_ayah)
        for segment in clip.segments:
            if not segment.kept:
                continue
            key = segment_key(clip.audio_filename, segment.segment_index)
            for found in segment_sites(
                segment.reference, base_decodes[key], h448_decodes[key],
                list(segment.raw_word_offsets),
                tanween[segment.word_start:segment.word_end],
            ):
                site = TruthSite(
                    site_id=site_id(clip.audio_filename, segment.segment_index,
                                    found.reference_index, found.mark),
                    source=NEW_AUDIT,
                    assumes_competent_reciter=False,
                    audio_filename=clip.audio_filename,
                    shard=staged.shard,
                    start_sample=segment.start_sample,
                    end_sample=segment.end_sample,
                    audio_sha256=staged.audio_sha256,
                    surah_ayah=clip.surah_ayah,
                    reference=segment.reference,
                    reference_index=found.reference_index,
                    mark=found.mark,
                    prescribed=found.prescribed,
                    heard=PENDING,
                    stratum=found.stratum,
                    stratum_population=0,  # set once the stratum is counted
                )
                excerpt = excerpt_span(segment, clip.word_times, clip.re_reads,
                                       found.reference_index, staged.num_samples)
                candidates.append(Candidate(site, clip.inclusion_probability,
                                            segment.word_start, segment.raw_word_offsets,
                                            excerpt))
    return candidates


def p35_segments(
    clip_status: list[dict], segmentation: list[dict], registry: Mapping[str, StagedClip]
) -> dict[tuple[str, int], tuple[PoolSegment, tuple[float, ...], int, int]]:
    """The P3.5 re-location's segments (``tadabur.resegment`` output) keyed by clip and
    start sample, each with its clip's word times, re-read count and length."""
    from .segment_score import segment_sample_bounds

    status = {row["audio_filename"]: row for row in clip_status}
    segments = {}
    for row in segmentation:
        name = row["audio_filename"]
        clip_samples = registry[name].num_samples
        for seg in row["segments"]:
            start, end = segment_sample_bounds(clip_samples, seg["start_s"], seg["end_s"])
            segment = PoolSegment(
                segment_index=seg["segment_index"], word_start=seg["word_start"],
                word_end=seg["word_end"], start_sample=start, end_sample=end,
                reference=seg["reference"], raw_word_offsets=tuple(seg["raw_word_offsets"]),
                kept=seg["kept"],
            )
            segments[(name, start)] = (
                segment, tuple(status[name]["word_times"]), status[name]["re_reads"],
                clip_samples,
            )
    return segments


def p35_candidates(
    sites: list[TruthSite],
    segments: Mapping[tuple[str, int], tuple[PoolSegment, tuple[float, ...], int, int]],
) -> list[Candidate]:
    """The P3.5 sites still ``pending``: a census of the nominal rejects (inclusion
    probability 1), each on the re-location segment it was defined on."""
    candidates = []
    for site in sites:
        if site.heard != PENDING:
            continue
        segment, word_times, re_reads, clip_samples = segments[
            (site.audio_filename, site.start_sample)
        ]
        if (segment.end_sample, segment.reference) != (site.end_sample, site.reference):
            raise ValueError(f"{site.site_id}: its segment no longer matches the re-location")
        excerpt = excerpt_span(segment, word_times, re_reads, site.reference_index,
                               clip_samples)
        candidates.append(Candidate(site, 1.0, segment.word_start, segment.raw_word_offsets,
                                    excerpt))
    return candidates


# --- the worklist ----------------------------------------------------------------------


@dataclass(frozen=True)
class SessionSite:
    """One worklist row: a truth-site skeleton plus its sampling design and excerpt."""

    site: TruthSite
    clip_inclusion_probability: float
    draw_probability: float
    inclusion_probability: float
    word_start: int
    word_offsets: tuple[int, ...]
    excerpt_start_sample: int
    excerpt_end_sample: int

    @property
    def mode(self) -> str:
        return mode_of(self.site.mark)


DESIGN_FIELDS = tuple(f.name for f in fields(SessionSite) if f.name != "site")
_Row = TypeVar("_Row")


def _rank(stratum: str, site_id_: str) -> str:
    return hashlib.sha256(f"{SALT}:{stratum}:{site_id_}".encode("utf-8")).hexdigest()


def draw(candidates: list[Candidate], sizes: Mapping[str, int]) -> list[SessionSite]:
    """Up to ``sizes[stratum]`` pool sites per stratum, ranked by a salted hash of the site
    id, each with its stratum's population and its inclusion probability."""
    unknown = set(sizes) - set(STRATA)
    if unknown:
        raise ValueError(f"sizes name unknown strata: {sorted(unknown)}")
    by_stratum: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_stratum.setdefault(candidate.site.stratum, []).append(candidate)
    rows: list[SessionSite] = []
    for stratum, members in sorted(by_stratum.items()):
        ranked = sorted(members, key=lambda c: _rank(stratum, c.site.site_id))
        chosen = ranked[:sizes.get(stratum, 0)]
        draw_probability = len(chosen) / len(members)
        rows += [
            _session_site(replace(c.site, stratum_population=len(members)), c,
                          draw_probability)
            for c in chosen
        ]
    return rows


def _session_site(site: TruthSite, candidate: Candidate, draw_probability: float) -> SessionSite:
    return SessionSite(
        site=site,
        clip_inclusion_probability=candidate.clip_inclusion_probability,
        draw_probability=draw_probability,
        inclusion_probability=candidate.clip_inclusion_probability * draw_probability,
        word_start=candidate.word_start,
        word_offsets=candidate.word_offsets,
        excerpt_start_sample=candidate.excerpt[0],
        excerpt_end_sample=candidate.excerpt[1],
    )


def census(candidates: list[Candidate]) -> list[SessionSite]:
    """Every candidate, each certain to be listened to: the P3.5 safeguards, which keep
    their own strata and populations (acceptance rules §1, *Targeted safeguards*)."""
    return [_session_site(c.site, c, 1.0) for c in candidates]


def shuffled(rows: list[_Row]) -> list[_Row]:
    """One queue across every mode and stratum, in a seeded order that depends only on the
    site ids, so the position of a site says nothing about its stratum. Any rows carrying a
    ``site`` are shuffled alike: the UI queues the synthetic-edit check with the worklist."""
    queue = sorted(rows, key=lambda r: r.site.site_id)
    random.Random(f"{SALT}:order").shuffle(queue)
    return queue


def _row_json(row: SessionSite) -> dict:
    design = {name: getattr(row, name) for name in DESIGN_FIELDS}
    return {**asdict(row.site), **design, "word_offsets": list(row.word_offsets)}


def parse_row(data: dict, where: str) -> SessionSite:
    """One validated worklist row: a valid truth site plus a consistent design."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: a worklist row is a JSON object")
    design = {name: data.get(name) for name in DESIGN_FIELDS}
    site = parse_site({k: v for k, v in data.items() if k not in DESIGN_FIELDS}, where)
    if site.heard != PENDING:
        raise ValueError(f"{where}: a worklist row is a skeleton; heard must be pending")
    offsets = design["word_offsets"]
    if not (isinstance(offsets, list) and offsets and all(type(o) is int for o in offsets)
            and offsets[0] == 0 and offsets[-1] == len(site.reference)
            and offsets == sorted(offsets) and type(design["word_start"]) is int
            and design["word_start"] >= 0):
        raise ValueError(f"{where}: word_start and word_offsets do not describe the reference")
    row = SessionSite(site=site, **{**design, "word_offsets": tuple(offsets)})
    probabilities = (row.clip_inclusion_probability, row.draw_probability,
                     row.inclusion_probability)
    if not all(type(p) is float and 0 < p <= 1 for p in probabilities):
        raise ValueError(f"{where}: inclusion probabilities must be floats in (0, 1]")
    if abs(row.inclusion_probability - row.clip_inclusion_probability
           * row.draw_probability) > 1e-12:
        raise ValueError(f"{where}: inclusion_probability is not clip x draw")
    if not (type(row.excerpt_start_sample) is int and type(row.excerpt_end_sample) is int
            and 0 <= row.excerpt_start_sample < row.excerpt_end_sample):
        raise ValueError(f"{where}: the excerpt is not a span of samples")
    return row


def write_worklist(rows: list[SessionSite], path: Path = WORKLIST_PATH) -> None:
    """Validate every row and the file, then write atomically, one JSON object per line."""
    for number, row in enumerate(rows, 1):
        parse_row(_row_json(row), f"write_worklist[{number}]")
    _check_file([row.site for row in rows], "write_worklist")
    write_text_atomically(path, "".join(
        json.dumps(_row_json(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def load_worklist(path: Path = WORKLIST_PATH) -> list[SessionSite]:
    """The worklist in file order, every row and the truth-site file invariants checked."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if raw.strip():
                rows.append(parse_row(json.loads(raw), f"{path}:{lineno}"))
    _check_file([row.site for row in rows], str(path))
    return rows


# --- the summary sidecar ---------------------------------------------------------------


def listening_seconds(row: SessionSite) -> float:
    """The listening-time model: :data:`PLAYS_PER_SITE` plays plus :data:`ANSWER_SECONDS`."""
    excerpt = (row.excerpt_end_sample - row.excerpt_start_sample) / TARGET_SAMPLE_RATE
    return PLAYS_PER_SITE * excerpt + ANSWER_SECONDS


def summarize(
    population: list[TruthSite], rows: list[SessionSite], registry: Mapping[str, StagedClip]
) -> dict:
    """Per stratum: its population (every site of ``population`` in it) and their reciters,
    the sites drawn and their reciters, the draw and inclusion probabilities, and the
    estimated listening minutes. Every stratum of :data:`STRATA` is listed, drawn from or
    not, then the census strata the rows come from (P3.5). Per mode and in total: sites
    and minutes."""
    def reciters(names) -> int:
        return len({registry[name].reciter_id for name in names})

    members_of: dict[str, list[str]] = {stratum: [] for stratum in STRATA}
    for site in population:
        members_of.setdefault(site.stratum, []).append(site.audio_filename)
    drawn: dict[str, list[SessionSite]] = {}
    for row in rows:
        drawn.setdefault(row.site.stratum, []).append(row)
    strata = {}
    for stratum in [*STRATA, *sorted(set(drawn) - set(STRATA))]:
        chosen, members = drawn.get(stratum, []), members_of[stratum]
        if any(row.site.stratum_population != len(members) for row in chosen):
            raise ValueError(f"{stratum}: the rows disagree with the population given")
        strata[stratum] = {
            "population": len(members),
            "population_reciters": reciters(members),
            "sampled": len(chosen),
            "sampled_reciters": reciters(r.site.audio_filename for r in chosen),
            "draw_probability": chosen[0].draw_probability if chosen else 0.0,
            "inclusion_probability_min": min((r.inclusion_probability for r in chosen),
                                             default=0.0),
            "inclusion_probability_max": max((r.inclusion_probability for r in chosen),
                                             default=0.0),
            "listening_minutes": round(sum(map(listening_seconds, chosen)) / 60, 1),
        }
    modes: dict[str, list[SessionSite]] = {}
    for row in rows:
        modes.setdefault(row.mode, []).append(row)
    return {
        "salt": SALT,
        "listening_model": {"plays_per_site": PLAYS_PER_SITE, "answer_seconds": ANSWER_SECONDS,
                            "excerpt_pad_s": EXCERPT_PAD_S, "min_excerpt_s": MIN_EXCERPT_S},
        "strata": strata,
        "modes": {
            mode: {"sites": len(members),
                   "listening_minutes": round(sum(map(listening_seconds, members)) / 60, 1)}
            for mode, members in sorted(modes.items())
        },
        "sites": len(rows),
        "listening_minutes": round(sum(map(listening_seconds, rows)) / 60, 1),
        "excerpts_whole_segment": sum(
            (r.excerpt_start_sample, r.excerpt_end_sample)
            == (r.site.start_sample, r.site.end_sample) for r in rows
        ),
    }


# --- verdicts --------------------------------------------------------------------------


#: How a synthetic-edit check item sounded (#107): a real recitation, or not.
NATURAL = "natural"
UNNATURAL = "unnatural"
#: The naturalness answers, in the order the page offers them.
NATURALNESS = (NATURAL, UNNATURAL, UNCLEAR)


@dataclass(frozen=True)
class Verdict:
    """What the listener heard at one site, in the truth-site vocabulary of its mark, and
    for a synthetic-edit check item also whether it sounded ``natural`` (one of
    :data:`NATURALNESS`). ``natural`` is ``None`` for every other site and is then left out
    of the file, so a verdicts file without it reads and rewrites byte for byte."""

    site_id: str
    heard: str
    note: str = ""
    natural: str | None = None

    def as_json(self) -> dict:
        data = asdict(self)
        if self.natural is None:
            del data["natural"]
        return data


_VERDICT_KEYS = frozenset({"site_id", "heard", "note"})


def hearable(mark: str) -> frozenset[str]:
    """The answers a listener may give for ``mark``: everything but ``pending``."""
    return label_states(mark)[1] - {PENDING}


def read_verdicts(path: Path = VERDICTS_PATH) -> dict[str, Verdict]:
    """Every verdict keyed by site id; empty when the file is absent or empty."""
    if not path.exists():
        return {}
    verdicts: dict[str, Verdict] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) - {"natural"} != _VERDICT_KEYS:
                raise ValueError(
                    f"{path}:{lineno}: a verdict is {{site_id, heard, note}} and optionally "
                    f"natural")
            if "natural" in data and data["natural"] not in NATURALNESS:
                raise ValueError(f"{path}:{lineno}: natural must be one of {NATURALNESS}")
            if data["site_id"] in verdicts:
                raise ValueError(f"{path}:{lineno}: a second verdict for {data['site_id']}")
            verdicts[data["site_id"]] = Verdict(**data)
    return verdicts


def write_verdicts(verdicts: Mapping[str, Verdict], path: Path = VERDICTS_PATH) -> None:
    """Rewrite the whole file atomically, sorted by site id, so an interrupted save leaves
    the previous verdicts intact and a re-save never reorders the diff."""
    write_text_atomically(path, "".join(
        json.dumps(verdicts[key].as_json(), ensure_ascii=False, sort_keys=True) + "\n"
        for key in sorted(verdicts)
    ))


def adjudicated(sites: list[TruthSite], verdicts: Mapping[str, Verdict]) -> list[TruthSite]:
    """``sites`` with each one's verdict, matched by site id, as what was heard.

    This is how the 23 P3.5 rejects, which keep their own site ids in the session, get their
    site-level verdicts. A verdict fills a ``pending`` site; one for a site that already has a
    verdict must agree with it. Verdicts for other sites are ignored.
    """
    result = []
    for site in sites:
        verdict = verdicts.get(site.site_id)
        if verdict is None:
            result.append(site)
            continue
        if verdict.heard not in hearable(site.mark):
            raise ValueError(f"{site.site_id}: {verdict.heard!r} is not an answer for {site.mark}")
        if site.heard not in (PENDING, verdict.heard):
            raise ValueError(
                f"{site.site_id}: heard {site.heard!r} in its file, {verdict.heard!r} in the session"
            )
        result.append(replace(site, heard=verdict.heard))
    return result


# --- the session's audio ---------------------------------------------------------------

#: The GPU box, and where the session's clips (#83) and the synthetic-edit check's items
#: (#88, ``tadabur.edit_check.EDIT_AUDIO_REMOTE``) sit on it.
GPU_HOST = "root@cuda-dev"
SESSION_CLIPS_REMOTE = "/root/scratch/issue-83/stage/clips"


def fetch_list(sources: Mapping[str, set[str]]) -> tuple[str, list[str]]:
    """For ``{remote directory: file names}``, the directory they share and each file's
    path under it: what one ``rsync --files-from`` copies into a single local directory."""
    names = [name for group in sources.values() for name in group]
    if len(names) != len(set(names)):
        raise ValueError("two remote directories hold files of one name")
    root = posixpath.commonpath(list(sources))
    return root, [posixpath.join(posixpath.relpath(remote, root), name)
                  for remote, group in sorted(sources.items()) for name in sorted(group)]


def fetch(
    rows: list[SessionSite], items: list[EditCheckSite], audio_dir: Path,
    registry: Mapping[str, StagedClip], host: str = GPU_HOST,
    sources: Mapping[str, str] | None = None, run: Callable[..., object] = subprocess.run,
) -> None:
    """Copy every clip the worklist plays and every edit-check item from ``host`` into
    ``audio_dir`` with one rsync (files already there are kept), then verify each against
    its recorded checksum:
    the clips through the staged-clip registry (and their length), the items through the
    blind check. ``sources`` maps ``"session"`` and ``"edit_check"`` to their remote
    directories."""
    from .edit_check import EDIT_AUDIO_REMOTE
    from .edit_check import verify_audio as verify_edit_check_audio

    remote = {"session": SESSION_CLIPS_REMOTE, "edit_check": EDIT_AUDIO_REMOTE,
              **(sources or {})}
    clips = {row.site.audio_filename for row in rows}
    root, files = fetch_list({remote["session"]: clips,
                              remote["edit_check"]: {i.site.audio_filename for i in items}})
    audio_dir.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--no-relative", "--files-from=-", f"{host}:{root}/", f"{audio_dir}/"],
        input="".join(f"{path}\n" for path in files), text=True, check=True)
    for name in sorted(clips):
        verify_staged(registry[name], audio_dir)
    verify_edit_check_audio(items, audio_dir)
    print(f"{len(clips)} session clips and {len(items)} edit-check items verified in "
          f"{audio_dir}")


# --- CLI -------------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(raw) for raw in f if raw.strip()]


def _uthmani_tanween() -> Callable[[str], list[bool]]:
    from .waqf_segments import _uthmani_words

    cache: dict[str, list[bool]] = {}

    def tanween_of(surah_ayah: str) -> list[bool]:
        if surah_ayah not in cache:
            cache[surah_ayah] = [
                any(c in _TANWEEN for c in word) for word in _uthmani_words(surah_ayah)
            ]
        return cache[surah_ayah]

    return tanween_of


def _mine(args) -> None:
    registry = load_staged_clips()
    clips = load_manifest(POOL_CLIPS_PATH, registry)
    fingerprint, decodes = load_base_decodes(POOL_DECODES_PATH)
    h448_provenance, h448_decodes = load_stream_decodes(H448_DECODES_PATH)
    sizes = DEFAULT_SIZES if args.sizes is None else json.loads(args.sizes.read_text())
    candidates = pool_candidates(clips, decodes, h448_decodes, registry, _uthmani_tanween())
    p35 = load_truth_sites(P35_SITES_PATH)
    safeguards = p35_candidates(
        p35,
        p35_segments(_jsonl(args.p35_seg_dir / "clip_status.jsonl"),
                     _jsonl(args.p35_seg_dir / "segmentation.jsonl"), registry),
    )
    rows = shuffled(draw(candidates, sizes) + census(safeguards))
    re_adjudicated = {c.site.stratum for c in safeguards}
    population = [c.site for c in candidates] + [s for s in p35 if s.stratum in re_adjudicated]
    summary = {
        "inputs": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("mining_pool/clips.jsonl", POOL_CLIPS_PATH),
                               ("mining_pool/base_decodes.json", POOL_DECODES_PATH),
                               ("mining_pool/h448_stream_decodes.json", H448_DECODES_PATH),
                               ("truth_sites/p35_fixtures.jsonl", P35_SITES_PATH))
        },
        "base_decode": fingerprint,
        "h448_decode": h448_provenance,
        "sizes": {stratum: sizes.get(stratum, 0) for stratum in STRATA},
        **summarize(population, rows, registry),
    }
    write_worklist(rows)
    write_text_atomically(
        SUMMARY_PATH, json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    for stratum, counts in summary["strata"].items():
        if counts["population"]:
            print(f"{stratum:52s} {counts['sampled']:4d} / {counts['population']:6d}"
                  f"  {counts['listening_minutes']:5.1f} min")
    print(json.dumps({k: summary[k] for k in ("modes", "sites", "listening_minutes")},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    mine = commands.add_parser("mine", help="draw the worklist and write its summary")
    mine.add_argument("--p35-seg-dir", type=Path, required=True,
                      help="tadabur.resegment output for the P3.5 clips (clip_status.jsonl, "
                           "segmentation.jsonl): the word times of their excerpts")
    mine.add_argument("--sizes", type=Path, default=None,
                      help="JSON {stratum: sites to draw}; default DEFAULT_SIZES")
    commands.add_parser("clips", help="print the clip names the worklist plays, one per line")
    fetch_ = commands.add_parser(
        "fetch", help="copy the worklist's clips and the edit-check items from the GPU box "
                      "into one local directory, and verify every checksum")
    fetch_.add_argument("--audio-dir", type=Path, required=True)
    fetch_.add_argument("--host", default=GPU_HOST)
    args = parser.parse_args()
    if args.command == "mine":
        _mine(args)
    elif args.command == "fetch":
        from .edit_check import load_edit_check

        fetch(load_worklist(), load_edit_check(), args.audio_dir.expanduser(),
              load_staged_clips(), args.host)
    else:
        print("\n".join(sorted({row.site.audio_filename for row in load_worklist()})))


if __name__ == "__main__":
    main()

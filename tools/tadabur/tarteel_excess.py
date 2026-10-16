"""The owner's listening on Tarteel v1: words ``h448`` flags and the teacher does not (#160).

On Tarteel v1 app-user audio (#156) ``h448`` flags 57.0 words per 1,000 against the base
teacher's 21.3, and the recitations are unverified. This module asks the owner, by ear,
whether ``h448``'s **excess** flags are model errors or the users' real mistakes, beside
**control** words neither model flags, which measure how often the owner hears a mistake on
an unflagged word.

**Frame.** The graded words of the Tarteel v1 items in the split's **look** half
(:mod:`tadabur.tarteel_split`; the confirm half is never read), under Muraja's shipped default,
both models' recorded checks replayed through Muraja's engine exactly as #156 graded them
(:func:`training.external_corpus.word_grades`, the committed decodes):

* an **excess** word: ``h448`` flags it and the base teacher grades it correct
  (``not_flagged``);
* a **control** word (:data:`CONTROL`): both grade it correct.

A word is a Muraja phoneme word; it covers one canonical word, or several where Muraja joins
them.

**Strata.** A word's **top substitution** is, of the consonant errors its ``h448`` flag
carries (``training.word_flags.Flag.consonants``: cause and ``expected→heard``), the one
ranked highest in #156's tally of ``h448``'s Tarteel v1 flags (shipped default, ``report.json``,
most words first). Its stratum is that substitution when it is one of :data:`NAMED`, the
tally's most frequent substitutions the issue names, else :data:`OTHER` (as is a flag with no
consonant error: tashkeel, skipped, unrecorded).

**Draw (seeded, at most one word per clip).** Two stages, each by a salted SHA-256 rank
(:data:`SALT`), so the draw depends on nothing but the ids:

1. each clip with excess words keeps one, its **representative**: the word ranked first
   (chance ``1 / m``, ``m`` the clip's excess words);
2. per stratum, the representatives in it are ranked and the first :data:`SIZES` taken
   (chance ``n / N``, ``N`` the stratum's representatives; all of them when ``N <= n``).

A word's inclusion probability is the product; its design weight the inverse, so the
weighted shares estimate the frame's excess words (:mod:`tadabur.tarteel_excess_report`).

**Controls (seeded, one per clip, never an excess draw's clip).** Of the look-half clips with
a control word, less the clips the excess draw took, the first :data:`CONTROLS` by salted rank
(chance ``c / C``, ``C`` those clips); in each, the control word ranked first (chance
``1 / k``, ``k`` the clip's control words). The probability is conditional on the excess draw,
which is fixed by the ids. The controls estimate the owner's baseline on unflagged words of
the clips the sitting could have drawn.

**Excerpt: the whole clip.** No corpus here times its words (#156), and #156's decodes keep
strings, not the CTC frames a ``TimedClip``-style span would need: a span would mean
re-decoding with the teacher and trusting its alignment exactly where ``h448`` heard
something else (a dropped hamza or alif at a word's edge is where a cut goes wrong). A
Tarteel v1 clip is one ayah, 9.3 s on average, so the whole clip, played twice, fits the
sitting (:func:`listen_seconds`): the page highlights the word in the ayah's text, and the
listener judges it in context.

**Blind (ADR-0007).** The page is :mod:`tadabur.tashkeel_audit_ui`'s word mode ("Was the
highlighted word recited correctly?": Yes ``correct``, No ``mistake``, Unclear ``unclear``,
with a note). It receives what any sampled word sends (the ayah's text, the highlight, the
answer); never a model's output, the stratum, the flag or the design, and so not whether a
word is a control: excess and control words come in one seeded order of their ids.

**Audio.** :func:`stage` re-reads only the drawn clips from ``greentechapps/user_recited_
verses_full`` at #156's pinned revision (the row groups holding them, by HTTP range requests;
run it with ``HF_HOME`` in a scratch directory), checks each clip's bytes against #156's
SHA-256, decodes them as #156 did (16 kHz mono), checks the length against #156's decode and
writes a PCM_16 WAV. ``staged.jsonl`` records where each came from and the staged file's
checksum and length; :func:`session` refuses audio that does not match it. A clip already
staged and matching its record is kept. :func:`push` copies the staged clips to the GPU box
(where the owner's sessions are hosted) and re-verifies each checksum there. No audio is
committed.

Files (``tools/tadabur/tarteel_excess/``): ``worklist.jsonl`` (one drawn word per line, with
its provenance, stratum and design), ``worklist.summary.json`` (the frame per stratum, the
inputs' and the worklist's SHA-256), ``staged.jsonl``, ``verdicts.jsonl`` and ``review.jsonl``
(tracked, written by the page).

Usage (from ``tools/``; ``draw`` needs the Muraja harness)::

  python -m tadabur.tarteel_excess draw
  HF_HOME=<scratch> python -m tadabur.tarteel_excess stage --audio-dir <session audio>
  python -m tadabur.tarteel_excess push --audio-dir <session audio>   # to cuda-dev
  python -m tadabur.tarteel_excess serve --audio-dir <session audio> --host 0.0.0.0 [--port 8000]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

from training.tashkeel_eval import write_text_atomically

from .audio import TARGET_SAMPLE_RATE
from .listening_session import PLAYS_PER_SITE, read_review, read_verdicts
from .nonpro_sources import TARTEEL_V1, audio_sha256, locate_clips, row_group
from .tarteel_split import LOOK, SPLIT_PATH, half
from .word_sample import ANSWER_SECONDS, SampledWord, WordSite

if TYPE_CHECKING:
    from training.external_corpus import WordGrade

    from .tashkeel_audit_ui import SessionState

EXCESS_DIR = Path(__file__).parent / "tarteel_excess"
WORKLIST_PATH = EXCESS_DIR / "worklist.jsonl"
SUMMARY_PATH = EXCESS_DIR / "worklist.summary.json"
STAGED_PATH = EXCESS_DIR / "staged.jsonl"
VERDICTS_PATH = EXCESS_DIR / "verdicts.jsonl"
REVIEW_PATH = EXCESS_DIR / "review.jsonl"
NONPRO_REPORT = Path(__file__).parent / "nonpro_eval" / "report.json"
#: The mushaf's Uthmani words, which the page shows in place of the phonetic reference.
QURAN_DB = Path(__file__).resolve().parents[2] / "data" / "quran.db"

SALT = "tarteel-v1-excess-listening-2026-10"
#: The tally's most frequent ``h448`` substitutions the issue names, each a stratum.
NAMED = ("ء→∅", "ا→∅", "ج→ش", "ل→ن", "ن→∅", "د→ت", "ن→ل", "ز→س")
OTHER = "other"
EXCESS_STRATA = (*NAMED, OTHER)
#: The control words' stratum: graded correct by both models.
CONTROL = "control"
STRATA = (*EXCESS_STRATA, CONTROL)
#: Excess words drawn per stratum, 80 in all. ``other`` holds about four in five of the frame's
#: words (1,374 of 1,724 in the look half), so it takes 32 and each named stratum 6: enough to
#: see where ``h448`` errs per substitution, without resting the pooled share on a few heavy
#: words.
SIZES = {**{s: 6 for s in NAMED}, OTHER: 32}
#: Control words drawn: the owner's baseline on unflagged words.
CONTROLS = 20
SCORING = "shipped_default"
#: Where the owner's sessions are hosted (``tadabur.listening_session``), and this one's clips.
GPU_HOST = "root@cuda-dev"
REMOTE_CLIPS = "/root/scratch/issue-160/clips"


def _rank(*parts: object) -> str:
    return hashlib.sha256(":".join((SALT, *map(str, parts))).encode("utf-8")).hexdigest()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listen_seconds(start_sample: int, end_sample: int) -> float:
    """The clip played :data:`PLAYS_PER_SITE` times, plus a whole-word answer's time."""
    return PLAYS_PER_SITE * (end_sample - start_sample) / TARGET_SAMPLE_RATE + ANSWER_SECONDS


# --- the frame ---------------------------------------------------------------------------------


def substitution_ranks(report: Mapping) -> dict[tuple[str, str], int]:
    """#156's tally of ``h448``'s Tarteel v1 flags (shipped default) as ``(cause,
    substitution)`` → rank, most words first."""
    tally = report["app"]["outcome"][TARTEEL_V1.name]["h448"][SCORING]["substitutions"]
    return {(t["cause"], t["substitution"]): k for k, t in enumerate(tally)}


def stratum_of(consonants: Sequence[tuple[str, str]], ranks: Mapping[tuple[str, str], int]) -> str:
    """The stratum of a flag carrying ``consonants`` (module docstring): its top substitution
    when that is :data:`NAMED`, else :data:`OTHER`. A substitution the tally lacks ranks
    last."""
    if not consonants:
        return OTHER
    _, top = min(consonants, key=lambda c: (ranks.get(c, len(ranks)), c))
    return top if top in NAMED else OTHER


@dataclass(frozen=True)
class FrameWord:
    """One frame word: ``h448``'s grade of it (its item, its Muraja phoneme word and, for an
    excess word, the flag), the canonical words it covers (``[position_from, position_to)`` in
    the item) and its stratum (:data:`CONTROL` for a control word)."""

    grade: WordGrade
    position_from: int
    position_to: int
    stratum: str

    @property
    def site_id(self) -> str:
        return f"{TARTEEL_V1.name}:{self.grade.item.key}:w{self.grade.word}"

    @property
    def clip(self) -> str:
        return self.grade.item.audio_filename


def frame_words(grades: Sequence[WordGrade], ranks: Mapping[tuple[str, str], int]
                ) -> list[FrameWord]:
    """The frame (module docstring) among ``grades``, sorted by id: the look half's words the
    base teacher grades ``not_flagged`` under :data:`SCORING`, excess where ``h448`` flags them,
    control where it does not either."""
    from training.muraja_policy import NOT_FLAGGED

    shipped = [g for g in grades if g.scoring == SCORING and half(g.item.audio_filename) == LOOK]
    base = {(g.item.key, g.word): g.outcome for g in shipped if g.model == "base"}
    starts: dict[str, list[int]] = defaultdict(list)
    for g in shipped:
        if g.model == "h448":
            starts[g.item.key].append(g.word_index)
    out = []
    for g in shipped:
        if g.model != "h448" or base[(g.item.key, g.word)] != NOT_FLAGGED or not (
                g.flagged or g.outcome == NOT_FLAGGED):
            continue
        first = g.item.reference.word_indices[0]
        later = [s for s in starts[g.item.key] if s > g.word_index]
        end = min(later) if later else first + len(g.item.reference.word_indices)
        stratum = stratum_of(g.reason.consonants, ranks) if g.flagged else CONTROL
        out.append(FrameWord(g, g.word_index - first, end - first, stratum))
    return sorted(out, key=lambda w: w.site_id)


# --- the worklist ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ListeningWord:
    """One worklist row: the drawn word and its clip's provenance (#156's ``audio_sha256`` of
    the published bytes; the whole clip ``[start_sample, end_sample)`` as #156 decoded it), its
    item's realized reference and word offsets, the canonical words the Muraja word covers,
    ``h448``'s flag (empty for a control word), its stratum, and the design: the clip's frame
    words of its kind (``clip_words``), the stratum's frame words, the population its clip was
    drawn from (``stratum_population``: the stratum's representatives, or the controls'
    eligible clips), the word's chance in its clip (``representative_probability``), the
    clip's chance in that population (``draw_probability``) and their product."""

    site_id: str
    audio_filename: str
    audio_sha256: str
    surah_ayah: str
    cluster: int
    start_sample: int
    end_sample: int
    reference: str
    word_offsets: tuple[int, ...]
    word: int
    position_from: int
    position_to: int
    h448_flag: dict
    stratum: str
    stratum_words: int
    stratum_population: int
    clip_words: int
    representative_probability: float
    draw_probability: float
    inclusion_probability: float

    @property
    def weight(self) -> float:
        return 1 / self.inclusion_probability

    def word_site(self, words: Sequence[str] | None = None) -> WordSite:
        """The row as the listening page's word mode asks it: the whole clip, the text with
        the Muraja word's canonical words joined into the one word highlighted. With
        ``words`` (the ayah's Uthmani words, :func:`uthmani_words`) the page shows the mushaf's
        text, which the owner reads more easily (2026-10-15); otherwise the phonetic
        reference."""
        if words is None:
            text, cuts = self.reference, self.word_offsets
        else:
            text, cuts = uthmani_text(words, len(self.word_offsets) - 1)
        offsets = (*cuts[:self.position_from + 1], *cuts[self.position_to:])
        return WordSite(
            site=SampledWord(
                site_id=self.site_id, audio_filename=self.audio_filename, shard=-1,
                audio_sha256=self.audio_sha256, reciter_id=self.cluster,
                surah_ayah=self.surah_ayah, segment_index=0, start_sample=self.start_sample,
                end_sample=self.end_sample, reference=text,
                word_index=self.position_from,
                reference_index=cuts[self.position_from], stratum=self.stratum,
                stratum_population=self.stratum_population),
            clip_inclusion_probability=self.representative_probability,
            draw_probability=self.draw_probability,
            inclusion_probability=self.inclusion_probability,
            word_start=0, word_offsets=offsets,
            excerpt_start_sample=self.start_sample, excerpt_end_sample=self.end_sample)


def uthmani_words(surah_ayah: str, db: Path = QURAN_DB) -> list[str]:
    """The ayah's Uthmani words from ``quran.db``, in order, without the ayah-number marker
    the table stores as its last word."""
    import sqlite3

    surah, ayah = (int(part) for part in surah_ayah.split(":"))
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
        words = [text for (text,) in connection.execute(
            "SELECT text FROM words WHERE surah = ? AND ayah = ? ORDER BY word", (surah, ayah))]
    if not words or any(ch.isalpha() for ch in words[-1]):
        raise ValueError(f"{surah_ayah}: quran.db's last word is not the ayah-number marker")
    return words[:-1]


def uthmani_text(words: Sequence[str], count: int) -> tuple[str, tuple[int, ...]]:
    """The ayah's words joined by spaces and the offset where each starts (plus the end), as
    the reference's ``word_offsets`` run: word k spans ``[cuts[k], cuts[k + 1])`` with its
    trailing space. Refused unless the ayah has the reference's ``count`` canonical words,
    so the highlight lands on the word the verdict is about."""
    if len(words) != count:
        raise ValueError(f"the mushaf has {len(words)} words where the reference has {count}")
    text = " ".join(words)
    cuts, at = [0], 0
    for word in words[:-1]:
        at += len(word) + 1
        cuts.append(at)
    cuts.append(len(text))
    return text, tuple(cuts)


_TUPLES = ("word_offsets",)


def parse_row(data: dict, where: str) -> ListeningWord:
    """One validated worklist row: every field, the word inside its reference, the design
    probabilities in (0, 1] and consistent."""
    names = {f.name for f in fields(ListeningWord)}
    if not isinstance(data, dict) or set(data) != names:
        raise ValueError(f"{where}: a row has exactly the fields {sorted(names)}")
    row = ListeningWord(**{k: tuple(v) if k in _TUPLES else v for k, v in data.items()})
    offsets = row.word_offsets
    if not (offsets[0] == 0 and offsets[-1] == len(row.reference)
            and list(offsets) == sorted(offsets)
            and 0 <= row.position_from < row.position_to < len(offsets)):
        raise ValueError(f"{where}: the word is not a run of the reference's words")
    if row.stratum not in STRATA or not 0 <= row.start_sample < row.end_sample:
        raise ValueError(f"{where}: unknown stratum or no span of samples")
    if (row.stratum == CONTROL) != (row.h448_flag == {}):
        raise ValueError(f"{where}: a control word carries no flag, an excess word one")
    probabilities = (row.representative_probability, row.draw_probability,
                     row.inclusion_probability)
    if not all(type(p) is float and 0 < p <= 1 for p in probabilities) or abs(
            row.inclusion_probability - row.representative_probability * row.draw_probability
    ) > 1e-12:
        raise ValueError(f"{where}: probabilities are not floats in (0, 1], representative x draw")
    return row


def _row(word: FrameWord, stratum_words: int, population: int, clip_words: int,
         draw_probability: float) -> ListeningWord:
    item, grade = word.grade.item, word.grade
    flag = {} if grade.reason is None else {
        "causes": list(grade.reason.causes),
        "substitutions": [list(c) for c in grade.reason.consonants],
        "quality": grade.reason.quality, "errors": list(grade.reason.errors)}
    return ListeningWord(
        site_id=word.site_id, audio_filename=item.audio_filename,
        audio_sha256=item.audio_sha256, surah_ayah=item.surah_ayah, cluster=item.cluster,
        start_sample=item.start_sample, end_sample=item.end_sample,
        reference=item.reference.reference, word_offsets=tuple(item.reference.offsets),
        word=grade.word, position_from=word.position_from, position_to=word.position_to,
        h448_flag=flag, stratum=word.stratum, stratum_words=stratum_words,
        stratum_population=population, clip_words=clip_words,
        representative_probability=1 / clip_words, draw_probability=draw_probability,
        inclusion_probability=draw_probability / clip_words)


def draw(frame: Sequence[FrameWord], sizes: Mapping[str, int] = SIZES,
         controls: int = CONTROLS) -> list[ListeningWord]:
    """The excess words (at most one per clip, then up to ``sizes[stratum]`` per stratum) and
    the controls (``controls`` clips the excess draw did not take, one word each), as the
    module docstring draws them, sorted by id."""
    by_clip: dict[tuple[bool, str], list[FrameWord]] = defaultdict(list)
    stratum_words: dict[str, int] = defaultdict(int)
    for word in frame:
        by_clip[(word.stratum == CONTROL, word.clip)].append(word)
        stratum_words[word.stratum] += 1

    def representative(control: bool, clip: str) -> FrameWord:
        return min(by_clip[(control, clip)], key=lambda w: _rank("word", w.site_id))

    by_stratum: dict[str, list[FrameWord]] = defaultdict(list)
    for control, clip in by_clip:
        if not control:
            word = representative(False, clip)
            by_stratum[word.stratum].append(word)
    rows = []
    for stratum, members in sorted(by_stratum.items()):
        chosen = sorted(members, key=lambda w: _rank("draw", stratum, w.site_id))[:sizes[stratum]]
        rows += [_row(w, stratum_words[stratum], len(members),
                      len(by_clip[(False, w.clip)]), len(chosen) / len(members)) for w in chosen]
    taken = {r.audio_filename for r in rows}
    eligible = sorted(clip for control, clip in by_clip if control and clip not in taken)
    chosen_clips = sorted(eligible, key=lambda c: _rank("control", c))[:controls]
    rows += [_row(representative(True, clip), stratum_words[CONTROL], len(eligible),
                  len(by_clip[(True, clip)]), len(chosen_clips) / len(eligible))
             for clip in chosen_clips]
    return sorted(rows, key=lambda r: r.site_id)


def row_json(row: ListeningWord) -> dict:
    return {**asdict(row), "word_offsets": list(row.word_offsets)}


def write_worklist(rows: Sequence[ListeningWord], path: Path = WORKLIST_PATH) -> None:
    for number, row in enumerate(rows, 1):
        parse_row(row_json(row), f"write_worklist[{number}]")
    if len({r.site_id for r in rows}) != len(rows) or len({r.audio_filename for r in rows}) != len(rows):
        raise ValueError("write_worklist: two rows share a word or a clip")
    write_text_atomically(path, "".join(json.dumps(row_json(r), ensure_ascii=False, sort_keys=True)
                                        + "\n" for r in rows))


def load_worklist(path: Path = WORKLIST_PATH, summary: Path = SUMMARY_PATH) -> list[ListeningWord]:
    """The frozen worklist: its SHA-256 must be the one the summary froze."""
    frozen = json.loads(summary.read_text(encoding="utf-8"))["frozen"][path.name]
    if sha256_of(path) != frozen:
        raise ValueError(f"{path} is not the worklist {summary.name} froze")
    with open(path, encoding="utf-8") as f:
        return [parse_row(json.loads(raw), f"{path}:{n}") for n, raw in enumerate(f, 1)
                if raw.strip()]


def summarize(grades: Sequence[WordGrade], frame: Sequence[FrameWord],
              rows: Sequence[ListeningWord]) -> dict:
    """The frame and the draw per stratum, and what the sitting costs."""
    look = [g for g in grades if g.scoring == SCORING and g.model == "h448"
            and half(g.item.audio_filename) == LOOK]
    clips = defaultdict(set)
    for word in frame:
        clips[word.stratum].add(word.clip)
    drawn = defaultdict(list)
    for row in rows:
        drawn[row.stratum].append(row)
    return {
        "look_half": {"items": len({g.item.key for g in look}),
                      "h448_graded": sum(g.graded for g in look),
                      "h448_flagged": sum(g.flagged for g in look),
                      "excess_words": sum(w.stratum != CONTROL for w in frame),
                      "clips_with_excess": len({w.clip for w in frame if w.stratum != CONTROL}),
                      "control_words": sum(w.stratum == CONTROL for w in frame),
                      "clips_with_a_control_word": len(clips[CONTROL])},
        "strata": {s: {"words": sum(w.stratum == s for w in frame),
                       "clips_with_a_word": len(clips[s]),
                       # The excess strata's representatives; the controls' eligible clips.
                       "population": drawn[s][0].stratum_population if drawn[s] else 0,
                       "drawn": len(drawn[s]),
                       "draw_probability": drawn[s][0].draw_probability if drawn[s] else None}
                   for s in STRATA},
        "drawn": len(rows),
        "drawn_excess": sum(r.stratum != CONTROL for r in rows),
        "drawn_controls": len(drawn[CONTROL]),
        "listening_minutes": round(sum(listen_seconds(r.start_sample, r.end_sample)
                                       for r in rows) / 60, 1),
    }


def build(harness) -> tuple[list[ListeningWord], dict]:
    """Grade the look half as #156 did, and draw. Returns the rows and the summary (without
    the worklist's checksum)."""
    from training.carrier_windowing import MODELS
    from training.external_corpus import read_decode_file, word_grades
    from training.nonpro_eval import build_items, decodes_file

    from .waqf_segments import hafs_segment_reference

    decodes = {m: read_decode_file(decodes_file(TARTEEL_V1, m)) for m in MODELS}
    lines = [line for line in decodes["base"].values() if half(line["audio_filename"]) == LOOK]
    items, unsupported = build_items(TARTEEL_V1, lines, hafs_segment_reference())
    muraja, grades = word_grades(items, decodes, harness)
    report = json.loads(NONPRO_REPORT.read_text(encoding="utf-8"))
    frame = frame_words(grades, substitution_ranks(report))
    rows = draw(frame)
    summary = {
        "issue": 160,
        "salt": SALT,
        "scoring": SCORING,
        "named_strata": list(NAMED),
        "sizes": SIZES,
        "controls": CONTROLS,
        "muraja_build": muraja,
        "inputs": {name: sha256_of(path) for name, path in (
            ("nonpro_eval/tarteel_v1/decodes.base.jsonl.gz", decodes_file(TARTEEL_V1, "base")),
            ("nonpro_eval/tarteel_v1/decodes.h448.jsonl.gz", decodes_file(TARTEEL_V1, "h448")),
            ("nonpro_eval/tarteel_v1/split.json", SPLIT_PATH),
            ("nonpro_eval/report.json", NONPRO_REPORT))},
        "phonetizer_unsupported": len(unsupported),
        **summarize(grades, frame, rows),
    }
    return rows, summary


# --- the audio ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class StagedTarteelClip:
    """Where a staged clip came from (#156's dataset at its revision: shard and row group; the
    published bytes' SHA-256) and the staged 16 kHz PCM_16 WAV's SHA-256 and length."""

    audio_filename: str
    audio_sha256: str
    shard: str
    row_group: int
    staged_sha256: str
    num_samples: int


def load_staged(path: Path = STAGED_PATH) -> dict[str, StagedTarteelClip]:
    with open(path, encoding="utf-8") as f:
        clips = [StagedTarteelClip(**json.loads(raw)) for raw in f if raw.strip()]
    return {c.audio_filename: c for c in clips}


def write_staged(clips: Mapping[str, StagedTarteelClip], path: Path = STAGED_PATH) -> None:
    write_text_atomically(path, "".join(json.dumps(asdict(clips[k]), sort_keys=True) + "\n"
                                        for k in sorted(clips)))


def staged_file_matches(clip: StagedTarteelClip, audio_dir: Path) -> bool:
    """Whether ``audio_dir`` holds exactly the staged file ``clip`` records (checksum and
    length)."""
    import soundfile as sf

    from .panel_seal import refuse_sealed
    from .truth_sites import audio_sha256 as file_sha256

    path = refuse_sealed(audio_dir / clip.audio_filename)
    return path.exists() and (file_sha256(path), sf.info(path).frames) == (
        clip.staged_sha256, clip.num_samples)


def verify_audio(rows: Sequence[ListeningWord], staged: Mapping[str, StagedTarteelClip],
                 audio_dir: Path) -> None:
    """Fail unless every row's clip is staged under #156's checksum, and ``audio_dir`` holds
    exactly the staged file (checksum and length, as long as #156's decode)."""
    for row in rows:
        clip = staged.get(row.audio_filename)
        if clip is None or clip.audio_sha256 != row.audio_sha256 or clip.num_samples != row.end_sample:
            raise ValueError(f"{row.audio_filename}: not staged from #156's clip")
        if not staged_file_matches(clip, audio_dir):
            raise ValueError(f"{audio_dir / row.audio_filename} is not the staged clip "
                             f"staged.jsonl records")


def stage(rows: Sequence[ListeningWord], audio_dir: Path,
          recorded: Mapping[str, StagedTarteelClip] | None = None,
          locate: Callable = locate_clips, read_group: Callable = row_group
          ) -> dict[str, StagedTarteelClip]:
    """Re-read each row's clip from its row group (each group once), check its bytes against
    #156's SHA-256, decode it to 16 kHz mono as #156 did, check the length against #156's
    decode and write it as a PCM_16 WAV into ``audio_dir``. A clip ``recorded`` (a committed
    ``staged.jsonl``) whose file in ``audio_dir`` matches its record is kept as it is; one that
    is re-read must re-stage to the file it records. Returns the record of every row's clip."""
    import soundfile as sf

    from .audio import decode_to_mono_16k
    from .truth_sites import audio_sha256 as file_sha256

    recorded = recorded or {}
    out = {r.audio_filename: recorded[r.audio_filename] for r in rows
           if r.audio_filename in recorded
           and staged_file_matches(recorded[r.audio_filename], audio_dir)}
    wanted = {r.audio_filename: r for r in rows if r.audio_filename not in out}
    if not wanted:
        return out
    places = locate(TARTEEL_V1, wanted)
    if missing := sorted(set(wanted) - set(places)):
        raise ValueError(f"no row group holds {missing[:3]}")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for name, place in places.items():
        groups[place].add(name)
    audio_dir.mkdir(parents=True, exist_ok=True)
    for (shard, group), names in sorted(groups.items()):
        found = {r["audio"]["path"]: r["audio"]["bytes"]
                 for r in read_group(TARTEEL_V1, shard, group) if r["audio"]["path"] in names}
        for name in sorted(names):
            row, data = wanted[name], found[name]
            if audio_sha256(data) != row.audio_sha256:
                raise ValueError(f"{name}: the dataset's bytes are not #156's (SHA-256)")
            waveform = decode_to_mono_16k(data)
            if len(waveform) != row.end_sample:
                raise ValueError(f"{name}: {len(waveform)} samples, #156 decoded {row.end_sample}")
            path = audio_dir / name
            sf.write(path, waveform, TARGET_SAMPLE_RATE, subtype="PCM_16")
            clip = StagedTarteelClip(name, row.audio_sha256, shard, group, file_sha256(path),
                                     sf.info(path).frames)
            if recorded.get(name, clip) != clip:
                path.unlink()
                raise ValueError(f"{name} re-staged as {clip}, staged.jsonl records "
                                 f"{recorded[name]}")
            out[name] = clip
        print(f"{shard} row group {group}: {len(names)} clips", flush=True)
    return out


_SAFE_NAME = re.compile(r"[0-9A-Za-z_.-]+\.wav")


def push(rows: Sequence[ListeningWord], staged: Mapping[str, StagedTarteelClip],
         audio_dir: Path, host: str = GPU_HOST, remote: str = REMOTE_CLIPS,
         run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
    """Copy every row's verified clip from ``audio_dir`` to ``remote`` on ``host`` with one
    rsync (files already there are compared and kept), then re-read each one's SHA-256 there
    and fail unless it is the staged file's."""
    verify_audio(rows, staged, audio_dir)
    names = sorted(r.audio_filename for r in rows)
    if not all(_SAFE_NAME.fullmatch(n) for n in names):
        raise ValueError("a clip name is not a plain .wav name")
    run(["ssh", host, f"mkdir -p {remote}"], check=True)
    run(["rsync", "-a", "--files-from=-", f"{audio_dir}/", f"{host}:{remote}/"],
        input="".join(f"{n}\n" for n in names), text=True, check=True)
    listing = run(["ssh", host, f"cd {remote} && sha256sum -- {' '.join(names)}"],
                  capture_output=True, text=True, check=True).stdout
    found = {name: digest for digest, name in (line.split() for line in listing.splitlines())}
    if wrong := [n for n in names if found.get(n) != staged[n].staged_sha256]:
        raise ValueError(f"{host}:{remote} holds other bytes for {wrong[:3]}")
    print(f"{len(names)} clips copied to {host}:{remote} and verified there")


# --- the session -------------------------------------------------------------------------------


def session(audio_dir: Path, directory: Path = EXCESS_DIR,
            words: Callable[[str], Sequence[str]] = uthmani_words) -> SessionState:
    """The listening page's state for the frozen worklist in ``directory``, once every clip is
    verified: word mode only, in one seeded order of the ids
    (:func:`tadabur.tashkeel_audit_ui.queue`), each ayah shown in the mushaf's Uthmani text
    that ``words`` gives (:func:`uthmani_words`); the verdicts and the review list are that
    directory's."""
    from .tashkeel_audit_ui import SessionState, queue

    rows = load_worklist(directory / WORKLIST_PATH.name, directory / SUMMARY_PATH.name)
    verify_audio(rows, load_staged(directory / STAGED_PATH.name), audio_dir)
    verdicts, review = directory / VERDICTS_PATH.name, directory / REVIEW_PATH.name
    return SessionState(rows=queue([], [], [r.word_site(words(r.surah_ayah)) for r in rows]),
                        verdicts_path=verdicts, audio_dir=audio_dir,
                        verdicts=read_verdicts(verdicts), review_path=review,
                        review=read_review(review))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("draw", help="grade the look half, draw and freeze the worklist")
    staged = commands.add_parser("stage", help="re-read the drawn clips from the Hub, verify "
                                               "and write them as 16 kHz WAVs")
    staged.add_argument("--audio-dir", type=Path, required=True)
    pushed = commands.add_parser("push", help="copy the staged clips to the GPU box and verify "
                                              "them there")
    pushed.add_argument("--audio-dir", type=Path, required=True)
    pushed.add_argument("--host", default=GPU_HOST)
    pushed.add_argument("--remote", default=REMOTE_CLIPS)
    served = commands.add_parser("serve", help="the blind listening page on the worklist")
    served.add_argument("--audio-dir", type=Path, required=True)
    served.add_argument("--port", type=int, default=8000)
    served.add_argument("--host", default="127.0.0.1",
                        help="bind address; 0.0.0.0 to answer from a phone on the LAN")
    args = parser.parse_args()
    if args.command == "draw":
        from training.muraja_policy import require_harness

        rows, summary = build(require_harness())
        write_worklist(rows)
        summary["frozen"] = {WORKLIST_PATH.name: sha256_of(WORKLIST_PATH)}
        write_text_atomically(SUMMARY_PATH, json.dumps(summary, ensure_ascii=False, indent=1,
                                                       sort_keys=True) + "\n")
        print(json.dumps({k: summary[k] for k in ("look_half", "strata", "drawn", "drawn_excess",
                                                  "drawn_controls", "listening_minutes")},
                         ensure_ascii=False, indent=1))
    elif args.command == "stage":
        recorded = load_staged() if STAGED_PATH.exists() else None
        clips = stage(load_worklist(), args.audio_dir.expanduser(), recorded)
        write_staged(clips)
        print(f"{len(clips)} clips staged and verified in {args.audio_dir}")
    elif args.command == "push":
        push(load_worklist(), load_staged(), args.audio_dir.expanduser(), args.host, args.remote)
    else:
        from .audit_http import serve
        from .tashkeel_audit_ui import SessionHandler

        state = session(args.audio_dir.expanduser())
        progress = state.progress()
        print(f"{progress['answered']} of {progress['total']} answered. "
              f"Serving on http://{args.host}:{args.port}/", flush=True)
        serve(SessionHandler, state, args.port, args.host).serve_forever()


if __name__ == "__main__":
    main()

"""The owner's listening on Tarteel v1: words ``h448`` flags and the teacher does not (#160).

On Tarteel v1 app-user audio (#156) ``h448`` flags 57.0 words per 1,000 against the base
teacher's 21.3, and the recitations are unverified. This module asks the owner, by ear,
whether ``h448``'s **excess** flags are model errors or the users' real mistakes.

**Frame.** Every graded word of every Tarteel v1 item in the split's **look** half
(:mod:`tadabur.tarteel_split`; the confirm half is never read) that ``h448``'s recorded checks
flag under Muraja's shipped default **and** the base teacher's checks grade correct
(``not_flagged``), both replayed through Muraja's engine exactly as #156 graded them
(:func:`training.external_corpus.word_grades`, the committed decodes). A word is a Muraja
phoneme word; it covers one canonical word, or several where Muraja joins them.

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
answer); never a model's output, the stratum, the flag or the design.

**Audio.** :func:`stage` re-reads only the drawn clips from ``greentechapps/user_recited_
verses_full`` at #156's pinned revision (the row groups holding them, by HTTP range requests;
run it with ``HF_HOME`` in a scratch directory), checks each clip's bytes against #156's
SHA-256, decodes them as #156 did (16 kHz mono), checks the length against #156's decode and
writes a PCM_16 WAV. ``staged.jsonl`` records where each came from and the staged file's
checksum and length; :func:`session` refuses audio that does not match it. No audio is
committed.

Files (``tools/tadabur/tarteel_excess/``): ``worklist.jsonl`` (one drawn word per line, with
its provenance, stratum and design), ``worklist.summary.json`` (the frame per stratum, the
inputs' and the worklist's SHA-256), ``staged.jsonl``, ``verdicts.jsonl`` and ``review.jsonl``
(tracked, written by the page).

Usage (from ``tools/``; ``draw`` needs the Muraja harness)::

  python -m tadabur.tarteel_excess draw
  HF_HOME=<scratch> python -m tadabur.tarteel_excess stage --audio-dir <session audio>
  python -m tadabur.tarteel_excess serve --audio-dir <session audio> --host 0.0.0.0 [--port 8000]
"""

from __future__ import annotations

import argparse
import hashlib
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

SALT = "tarteel-v1-excess-listening-2026-10"
#: The tally's most frequent ``h448`` substitutions the issue names, each a stratum.
NAMED = ("ء→∅", "ا→∅", "ج→ش", "ل→ن", "ن→∅", "د→ت", "ن→ل", "ز→س")
OTHER = "other"
STRATA = (*NAMED, OTHER)
#: Words drawn per stratum, 100 in all. ``other`` holds about four in five of the frame's words
#: (1,374 of 1,724 in the look half), so it takes 44 and each named stratum 7: enough to see
#: where ``h448`` errs per substitution, without resting the pooled share on a few heavy words.
SIZES = {**{s: 7 for s in NAMED}, OTHER: 44}
SCORING = "shipped_default"


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
class ExcessWord:
    """One frame word: its item, its Muraja phoneme word, the canonical words it covers
    (``[position_from, position_to)`` in the item), its stratum and ``h448``'s flag."""

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


def excess_words(grades: Sequence[WordGrade], ranks: Mapping[tuple[str, str], int]
                 ) -> list[ExcessWord]:
    """The frame (module docstring) among ``grades``, sorted by id: the look half's words
    ``h448`` flags under :data:`SCORING` and the base teacher grades ``not_flagged``."""
    from training.muraja_policy import NOT_FLAGGED

    shipped = [g for g in grades if g.scoring == SCORING and half(g.item.audio_filename) == LOOK]
    base = {(g.item.key, g.word): g.outcome for g in shipped if g.model == "base"}
    starts: dict[str, list[int]] = defaultdict(list)
    for g in shipped:
        if g.model == "h448":
            starts[g.item.key].append(g.word_index)
    out = []
    for g in shipped:
        if g.model != "h448" or not g.flagged or base[(g.item.key, g.word)] != NOT_FLAGGED:
            continue
        first = g.item.reference.word_indices[0]
        later = [s for s in starts[g.item.key] if s > g.word_index]
        end = min(later) if later else first + len(g.item.reference.word_indices)
        out.append(ExcessWord(g, g.word_index - first, end - first,
                              stratum_of(g.reason.consonants, ranks)))
    return sorted(out, key=lambda w: w.site_id)


# --- the worklist ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ListeningWord:
    """One worklist row: the drawn word and its clip's provenance (#156's ``audio_sha256`` of
    the published bytes; the whole clip ``[start_sample, end_sample)`` as #156 decoded it), its
    item's realized reference and word offsets, the canonical words the Muraja word covers,
    ``h448``'s flag, its stratum, and the design: the clip's excess words, the stratum's frame
    words and representatives, and each stage's probability."""

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

    def word_site(self) -> WordSite:
        """The row as the listening page's word mode asks it: the whole clip, the reference
        with the Muraja word's canonical words joined into the one word highlighted."""
        offsets = (*self.word_offsets[:self.position_from + 1],
                   *self.word_offsets[self.position_to:])
        return WordSite(
            site=SampledWord(
                site_id=self.site_id, audio_filename=self.audio_filename, shard=-1,
                audio_sha256=self.audio_sha256, reciter_id=self.cluster,
                surah_ayah=self.surah_ayah, segment_index=0, start_sample=self.start_sample,
                end_sample=self.end_sample, reference=self.reference,
                word_index=self.position_from,
                reference_index=self.word_offsets[self.position_from], stratum=self.stratum,
                stratum_population=self.stratum_population),
            clip_inclusion_probability=self.representative_probability,
            draw_probability=self.draw_probability,
            inclusion_probability=self.inclusion_probability,
            word_start=0, word_offsets=offsets,
            excerpt_start_sample=self.start_sample, excerpt_end_sample=self.end_sample)


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
    probabilities = (row.representative_probability, row.draw_probability,
                     row.inclusion_probability)
    if not all(type(p) is float and 0 < p <= 1 for p in probabilities) or abs(
            row.inclusion_probability - row.representative_probability * row.draw_probability
    ) > 1e-12:
        raise ValueError(f"{where}: probabilities are not floats in (0, 1], representative x draw")
    return row


def draw(frame: Sequence[ExcessWord], sizes: Mapping[str, int] = SIZES) -> list[ListeningWord]:
    """At most one word per clip, then up to ``sizes[stratum]`` per stratum (module
    docstring), sorted by id."""
    by_clip: dict[str, list[ExcessWord]] = defaultdict(list)
    stratum_words: dict[str, int] = defaultdict(int)
    for word in frame:
        by_clip[word.clip].append(word)
        stratum_words[word.stratum] += 1
    by_stratum: dict[str, list[ExcessWord]] = defaultdict(list)
    for words in by_clip.values():
        representative = min(words, key=lambda w: _rank("word", w.site_id))
        by_stratum[representative.stratum].append(representative)
    rows = []
    for stratum, members in sorted(by_stratum.items()):
        chosen = sorted(members, key=lambda w: _rank("draw", stratum, w.site_id))[:sizes[stratum]]
        draw_probability = len(chosen) / len(members)
        for word in chosen:
            item, grade = word.grade.item, word.grade
            m = len(by_clip[word.clip])
            rows.append(ListeningWord(
                site_id=word.site_id, audio_filename=item.audio_filename,
                audio_sha256=item.audio_sha256, surah_ayah=item.surah_ayah,
                cluster=item.cluster, start_sample=item.start_sample,
                end_sample=item.end_sample, reference=item.reference.reference,
                word_offsets=tuple(item.reference.offsets), word=grade.word,
                position_from=word.position_from, position_to=word.position_to,
                h448_flag={"causes": list(grade.reason.causes),
                           "substitutions": [list(c) for c in grade.reason.consonants],
                           "quality": grade.reason.quality, "errors": list(grade.reason.errors)},
                stratum=stratum, stratum_words=stratum_words[stratum],
                stratum_population=len(members), clip_words=m,
                representative_probability=1 / m, draw_probability=draw_probability,
                inclusion_probability=draw_probability / m))
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


def summarize(grades: Sequence[WordGrade], frame: Sequence[ExcessWord],
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
                      "excess_words": len(frame),
                      "clips_with_excess": len({w.clip for w in frame})},
        "strata": {s: {"words": sum(w.stratum == s for w in frame),
                       "clips_with_a_word": len(clips[s]),
                       "representatives": drawn[s][0].stratum_population if drawn[s] else 0,
                       "drawn": len(drawn[s]),
                       "draw_probability": drawn[s][0].draw_probability if drawn[s] else None}
                   for s in STRATA},
        "drawn": len(rows),
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
    frame = excess_words(grades, substitution_ranks(report))
    rows = draw(frame)
    summary = {
        "issue": 160,
        "salt": SALT,
        "scoring": SCORING,
        "named_strata": list(NAMED),
        "sizes": SIZES,
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


def verify_audio(rows: Sequence[ListeningWord], staged: Mapping[str, StagedTarteelClip],
                 audio_dir: Path) -> None:
    """Fail unless every row's clip is staged under #156's checksum, and ``audio_dir`` holds
    exactly the staged file (checksum and length, as long as #156's decode)."""
    import soundfile as sf

    from .panel_seal import refuse_sealed
    from .truth_sites import audio_sha256 as file_sha256

    for row in rows:
        clip = staged.get(row.audio_filename)
        if clip is None or clip.audio_sha256 != row.audio_sha256 or clip.num_samples != row.end_sample:
            raise ValueError(f"{row.audio_filename}: not staged from #156's clip")
        path = refuse_sealed(audio_dir / row.audio_filename)
        if not path.exists():
            raise FileNotFoundError(f"{path} is not staged")
        if (file_sha256(path), sf.info(path).frames) != (clip.staged_sha256, clip.num_samples):
            raise ValueError(f"{path} is not the staged clip staged.jsonl records")


def stage(rows: Sequence[ListeningWord], audio_dir: Path,
          recorded: Mapping[str, StagedTarteelClip] | None = None,
          locate: Callable = locate_clips, read_group: Callable = row_group
          ) -> dict[str, StagedTarteelClip]:
    """Re-read each row's clip from its row group (each group once), check its bytes against
    #156's SHA-256, decode it to 16 kHz mono as #156 did, check the length against #156's
    decode and write it as a PCM_16 WAV into ``audio_dir``. With ``recorded`` (a committed
    ``staged.jsonl``), every clip must re-stage to the file it records. Returns every clip's
    record."""
    import soundfile as sf

    from .audio import decode_to_mono_16k
    from .truth_sites import audio_sha256 as file_sha256

    wanted = {r.audio_filename: r for r in rows}
    places = locate(TARTEEL_V1, wanted)
    if missing := sorted(set(wanted) - set(places)):
        raise ValueError(f"no row group holds {missing[:3]}")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for name, place in places.items():
        groups[place].add(name)
    audio_dir.mkdir(parents=True, exist_ok=True)
    out = {}
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
            if recorded is not None and recorded.get(name) != clip:
                path.unlink()
                raise ValueError(f"{name} re-staged as {clip}, staged.jsonl records "
                                 f"{recorded[name]}")
            out[name] = clip
        print(f"{shard} row group {group}: {len(names)} clips", flush=True)
    return out


# --- the session -------------------------------------------------------------------------------


def session(audio_dir: Path, directory: Path = EXCESS_DIR) -> SessionState:
    """The listening page's state for the frozen worklist in ``directory``, once every clip is
    verified: word mode only, in one seeded order of the ids
    (:func:`tadabur.tashkeel_audit_ui.queue`); the verdicts and the review list are that
    directory's."""
    from .tashkeel_audit_ui import SessionState, queue

    rows = load_worklist(directory / WORKLIST_PATH.name, directory / SUMMARY_PATH.name)
    verify_audio(rows, load_staged(directory / STAGED_PATH.name), audio_dir)
    verdicts, review = directory / VERDICTS_PATH.name, directory / REVIEW_PATH.name
    return SessionState(rows=queue([], [], [r.word_site() for r in rows]),
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
        print(json.dumps({k: summary[k] for k in ("look_half", "strata", "drawn",
                                                  "listening_minutes")},
                         ensure_ascii=False, indent=1))
    elif args.command == "stage":
        recorded = load_staged() if STAGED_PATH.exists() else None
        clips = stage(load_worklist(), args.audio_dir.expanduser(), recorded)
        if recorded is None:
            write_staged(clips)
        print(f"{len(clips)} clips staged and verified in {args.audio_dir}")
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

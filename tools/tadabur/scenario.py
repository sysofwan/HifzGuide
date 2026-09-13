"""The scenario manifest — what HifzGuide hands Muraja for one mined re-read.

This is the cross-repo interface of Muraja ADR-0016. Mining and staging are HifzGuide
(Python, CUDA); transcription, assertion and the scoreboard are Muraja (Swift, macOS);
between them travel a ``scenario.jsonl`` and a directory of 16 kHz mono WAVs. One record
describes one clip: where its audio is, which words of its ayah the oracle may assert
over, where the re-read sits, and — when the excision survived its re-gate — the paired
control clip with the repeat removed.

Three of ADR-0016's decisions land in the record's shape.

* **Decision 1 — the oracle is word-space only.** ``word_start`` / ``word_end`` are the
  half-open Uthmani word range of *this ayah* that the decode actually covered, obtained
  by mapping the alignment's reference span through the phonetizer's per-word phoneme
  boundaries (:func:`tadabur.waqf_segments.hafs_normalized_word_reference`). **No
  timestamp from Tadabur's ``word_alignments`` or from ``ClipStatus.word_times`` appears
  anywhere in this file.** The seconds that do appear are the clip's own duration and the
  spans cut out of it — neither is an assertion, and both are validated by re-gating the
  audio they describe.
* **Decision 3 — bleed is clipped, not absorbed.** The staged WAV is already re-cut
  (#67/#68): ``recitation_start_s`` / ``recitation_end_s`` say what was kept of the source
  clip and ``recut_applied`` whether anything was. The ``early_start`` flag survives from
  the superseded policy as a *signal*, not a rule — Muraja #128 reads it rather than
  starting every session an ayah early.
* **Decision 4 — the differential is excised and self-validated.** ``excised_audio`` is
  present only for a pair whose control clip came back clean from the ``.balanced`` gate
  (:mod:`tadabur.excision`). A cut that failed leaves the record intact with
  ``excised_audio`` null and the refusal in ``excision.reason``, so a discard is a counted
  outcome rather than a missing row.

The record is versioned (:data:`SCHEMA_VERSION`) the way
:mod:`tadabur.reference_phonemes` versions its cache — ``v1`` plus the normalization
algorithm — because every phoneme-space offset in it, the seams and the covered reference
span alike, is an index into a string ``normalize_phonemes`` produced.

Usage:
  python -m tadabur.scenario --rejects reject_run/rejects.jsonl \
      --clips bleed_run/clips/ --recuts bleed_run/recuts.jsonl \
      --out scenario_run/ [--limit N] [--batch-size 4]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import normalization
from .excision import ExcisionPlan, cut_time
from .seam import SeamCoverage

# Schema version for ``scenario.jsonl``. Follows
# :data:`tadabur.reference_phonemes.CACHE_VERSION`'s shape and, like it, is tied to
# :data:`tadabur.normalization.ALGORITHM_VERSION`: the seams and the covered reference
# span are offsets into a normalized string, so a normalization change moves them even
# when nothing in this module does. Bump the ``v1`` part when the record's own layout
# changes.
SCHEMA_VERSION = f"v1+norm{normalization.ALGORITHM_VERSION}"

# Where the two audio directories sit relative to ``scenario.jsonl``. Paths in the record
# are relative to it, so the whole bundle moves to the Mac as one directory.
AUDIO_DIR = "audio"
EXCISED_DIR = "excised"

# Leading query phonemes the aligner trimmed before an **early start** is worth flagging
# (ADR-0016 decision 3). The bar matches the trim baseline in
# :mod:`tadabur.bleed_detect`, which is where a trim was last asked to mean something.
# Since #68 re-cuts the bleed instead of absorbing it, this should now be rare — the flag
# exists so Muraja #128 can act on the clips where it is not, rather than applying a
# blanket one-ayah-early policy to a corpus that no longer needs one.
EARLY_START_TRIM = 5


@dataclass(frozen=True)
class Seam:
    """One re-read in a clip, in phoneme space and in the clip's own seconds.

    ``query_start`` / ``query_end`` index the **normalized decode**
    (``predicted_phonemes`` normalized), and ``ref_position`` is the reference phoneme the
    reciter doubled back to. ``start_s`` / ``end_s`` are where an excision cut, and are
    present because the pair's audio is unreadable without them — they are provenance for
    a cut the re-gate already validated, never an input to an assertion.
    """

    query_start: int
    query_end: int
    ref_position: int
    phonemes: int
    start_s: float
    end_s: float
    pause_anchored_start: bool
    pause_anchored_end: bool


@dataclass(frozen=True)
class ScenarioRecord:
    """One mined re-read, as Muraja consumes it.

    ``clip_id`` is the identity: the staged WAV is ``audio/<clip_id>.wav`` and the join
    key back to the reject sink and the recut records is ``<clip_id>.wav``. The gate
    fields are :class:`~tadabur.scorer.GateResult` over the **staged** (post-re-cut)
    audio, not the source clip's — the re-cut changes them, which is the point of it.

    ``excision`` always carries a ``reason``; it carries the full re-gate numbers only
    when a cut was actually made and re-decoded. ``excised_phonemes`` is that re-decode,
    and is kept for a **refused** pair as well as a kept one — a discard whose decode is
    thrown away cannot be explained, only counted.
    """

    schema_version: str
    clip_id: str
    audio: str
    surah_ayah: str
    reciter_id: int
    duration_s: float

    match_ratio: float
    max_insertion_run: int
    leading_trim: int
    trailing_trim: int
    added_shadda: bool
    predicted_phonemes: str

    word_start: int
    word_end: int
    ref_covered_start: int
    ref_covered_end: int
    ref_length: int
    uncovered_head: int
    uncovered_tail: int

    early_start: bool
    recut_applied: bool
    recitation_start_s: float
    recitation_end_s: float

    seams: tuple[Seam, ...] = ()
    excised_audio: str | None = None
    excised_duration_s: float | None = None
    excised_phonemes: str | None = None
    excision: dict | None = None

    @property
    def words_covered(self) -> int:
        """Words of this ayah the oracle may assert over."""
        return max(0, self.word_end - self.word_start)

    @property
    def has_pair(self) -> bool:
        """Whether a validated excision differential shipped with this clip."""
        return self.excised_audio is not None


def covered_word_range(
    reference: str, word_offsets: list[int], ref_start: int, ref_end: int
) -> tuple[int, int]:
    """The half-open word range lying wholly inside reference span ``[ref_start, ref_end)``.

    A word is covered only when **every** phoneme of it is: a word the decode reached
    halfway into is a word the oracle cannot grade, and admitting it would manufacture the
    false-skip class ADR-0016 exists to avoid. Word-separator spaces are ignored on both
    sides, because the query has none and the aligner never matches them — counting them
    would drop the last covered word whenever the phonetizer happened to leave a space
    after it.

    Returns ``(0, 0)`` when the span covers no whole word.
    """
    start = end = -1
    for word in range(len(word_offsets) - 1):
        positions = [
            index
            for index in range(
                max(0, word_offsets[word]), min(word_offsets[word + 1], len(reference))
            )
            if reference[index] != " "
        ]
        if not positions:
            continue
        if positions[0] >= ref_start and positions[-1] < ref_end:
            if start < 0:
                start = word
            end = word + 1
    return (0, 0) if start < 0 else (start, end)


def warrants_early_start(leading_trim: int) -> bool:
    """Whether this clip's lead-in is long enough to be worth starting an ayah early."""
    return leading_trim >= EARLY_START_TRIM


def build_seams(seams: list[SeamCoverage]) -> tuple[Seam, ...]:
    """The record's seam list, from the seams found — not from the cuts that succeeded.

    A clip whose excision was refused still *has* its re-read, and a record that reported
    seams only for the pairs that survived would make the corpus look like it contained
    fewer re-reads than it does. The times are where a cut would land
    (:func:`tadabur.excision.cut_time`), which is what makes the seam locatable in the
    audio whether or not one was made.
    """
    return tuple(
        Seam(
            query_start=seam.query_start,
            query_end=seam.query_end,
            ref_position=seam.ref_position,
            phonemes=seam.repeat_phonemes,
            start_s=round(cut_time(seam.start, seam.span_start_s), 4),
            end_s=round(cut_time(seam.end, seam.span_end_s), 4),
            pause_anchored_start=seam.start.anchored,
            pause_anchored_end=seam.end.anchored,
        )
        for seam in seams
    )


def write_scenario_records(path: Path, records: list[ScenarioRecord]) -> None:
    """Write the manifest as deterministic, key-sorted JSONL in ``clip_id`` order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in sorted(records, key=lambda record: record.clip_id):
            f.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def read_scenario_records(path: Path) -> list[ScenarioRecord]:
    """Load every :class:`ScenarioRecord` from ``path`` in file order."""
    records: list[ScenarioRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["seams"] = tuple(Seam(**seam) for seam in data.get("seams", ()))
            records.append(ScenarioRecord(**data))
    return records


@dataclass
class StagingTally:
    """What one staging run selected, skipped and produced.

    Skips are named rather than summed into one number: an ayah the phonetizer cannot
    handle and an ayah whose word offsets disagree with the cached reference are different
    problems, and only the second would mean something is wrong.
    """

    selected: int = 0
    staged: int = 0
    phonetizer_unsupported: int = 0
    reference_mismatch: int = 0
    missing_audio: int = 0
    recuts_applied: int = 0
    early_starts: int = 0
    truncated: int = 0
    seams: int = 0
    clips_with_multiple_seams: int = 0
    pairs_attempted: int = 0
    pairs_kept: int = 0
    pair_refusals: dict[str, int] = field(default_factory=dict)

    def refuse(self, reason: str) -> None:
        self.pair_refusals[reason] = self.pair_refusals.get(reason, 0) + 1


def verify_bundle(directory: Path) -> list[str]:
    """Everything wrong with a shipped scenario bundle, in reading order.

    The check that turns "copy the files" into a handoff: it is what Muraja runs on the
    Mac after the transfer, so it reads the manifest and the filesystem and **nothing
    else** — no torch, no CUDA, no reference cache. An empty list means every record
    names audio that is present, inside the bundle, and describes a clip the oracle can
    actually assert over.

    The last two checks are the ones a truncated transfer would not trip but a broken
    stager would: a record covering no words, or carrying no seam, is a row the corpus
    has no use for, and finding it here is better than finding it as a silent zero in the
    scoreboard.
    """
    directory = Path(directory)
    manifest = directory / "scenario.jsonl"
    if not manifest.exists():
        return [f"missing {manifest}"]

    records = read_scenario_records(manifest)
    if not records:
        return [f"{manifest} is empty"]

    problems: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record.schema_version != SCHEMA_VERSION:
            problems.append(
                f"{record.clip_id}: schema {record.schema_version}, "
                f"expected {SCHEMA_VERSION}"
            )
        if record.clip_id in seen:
            problems.append(f"{record.clip_id}: duplicate record")
        seen.add(record.clip_id)
        for name, relative in (("audio", record.audio), ("excised_audio", record.excised_audio)):
            if relative is None:
                continue
            if relative.startswith("/") or ".." in relative.split("/"):
                problems.append(f"{record.clip_id}: {name} escapes the bundle ({relative})")
            elif not (directory / relative).exists():
                problems.append(f"{record.clip_id}: {name} is missing ({relative})")
        if record.words_covered <= 0:
            problems.append(f"{record.clip_id}: covers no words")
        if not record.seams:
            problems.append(f"{record.clip_id}: carries no re-read seam")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejects", type=Path, help="Reject sink JSONL (the predicate).")
    parser.add_argument("--clips", type=Path, help="Directory of staged 16 kHz WAVs.")
    parser.add_argument(
        "--recuts",
        type=Path,
        help="Recut records JSONL (#68). Without it, clips are staged whole.",
    )
    parser.add_argument("--out", type=Path, help="Scenario bundle directory to write.")
    parser.add_argument(
        "--verify",
        type=Path,
        help="Check a shipped bundle and exit. Needs no model, GPU or reference cache.",
    )
    parser.add_argument(
        "--limit", type=int, help="Stage only the first N clips in clip_id order."
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Model batch size.")
    args = parser.parse_args()

    if args.verify is not None:
        problems = verify_bundle(args.verify)
        for problem in problems:
            print(problem)
        manifest = args.verify / "scenario.jsonl"
        records = len(read_scenario_records(manifest)) if manifest.exists() else 0
        print(
            f"{args.verify}: {records} records, "
            f"{len(problems)} problem{'' if len(problems) == 1 else 's'}"
        )
        raise SystemExit(1 if problems else 0)

    if not (args.rejects and args.clips and args.out):
        parser.error("--rejects, --clips and --out are required unless --verify is given")

    import soundfile as sf
    import torch

    from .audio import TARGET_SAMPLE_RATE
    from .bleed_detect import BleedDetector
    from .bleed_recut import normalized_onsets, read_recut_records
    from .excision import ExcisionValidation, excise, plan_excision, validate_excision
    from .inference import MuaalemPhonemeModel
    from .normalization import normalize_phonemes
    from .reference_phonemes import load_reference_phonemes
    from .rejects import read_reject_records
    from .scorer import BALANCED_SCORER
    from .seam import seam_coverage
    from .smith_waterman import smith_waterman
    from .vad import (
        DEFAULT_MIN_SILENCE_MS,
        DEFAULT_MIN_SPEECH_MS,
        DEFAULT_PAD_MS,
        _clip_intervals,
        _load_vad,
        pauses_from_intervals,
    )
    from .waqf_detect import collapse_with_times
    from .waqf_segments import _uthmani_words, hafs_normalized_word_reference

    references = load_reference_phonemes()
    detector = BleedDetector.of(references)
    word_reference = hafs_normalized_word_reference()
    recuts = (
        {record.audio_filename: record for record in read_recut_records(args.recuts)}
        if args.recuts
        else {}
    )
    tally = StagingTally()

    # --- select, in a fixed order, so a --limit run is a prefix and not a sample -------
    selected = sorted(
        (
            record
            for record in read_reject_records(args.rejects)
            if record.is_clean_re_read
        ),
        key=lambda record: record.audio_filename,
    )
    if args.limit is not None:
        selected = selected[: args.limit]
    tally.selected = len(selected)

    # --- materialize the staged audio: the source clip, cut to its recitation span -----
    audio_dir = args.out / AUDIO_DIR
    excised_dir = args.out / EXCISED_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[object, object, dict]] = []
    for record in selected:
        source = args.clips / record.audio_filename
        if not source.exists():
            tally.missing_audio += 1
            continue
        waveform, rate = sf.read(source, dtype="float32")
        if rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"{source} is {rate} Hz; the staged corpus is 16 kHz mono")
        recut = recuts.get(record.audio_filename)
        start_s = recut.recitation_start_s if recut else 0.0
        end_s = recut.recitation_end_s if recut else len(waveform) / rate
        clipped = waveform[int(start_s * rate) : int(end_s * rate)]
        staged.append(
            (
                record,
                clipped,
                {
                    "recut_applied": bool(recut and recut.accepted),
                    "recitation_start_s": start_s,
                    "recitation_end_s": end_s,
                },
            )
        )
        tally.recuts_applied += int(bool(recut and recut.accepted))
    print(f"staged {len(staged)} of {tally.selected} clean re-reads", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # --- the VAD, then the phoneme model: the two never co-reside on the GPU -----------
    intervals: list[list[tuple[float, float]]] = []
    vad_model, vad_processor = _load_vad(device, dtype)
    try:
        for start in range(0, len(staged), args.batch_size):
            batch = staged[start : start + args.batch_size]
            intervals.extend(
                _clip_intervals(
                    [waveform for _, waveform, _ in batch],
                    vad_model,
                    vad_processor,
                    device=device,
                    dtype=dtype,
                    batch_size=args.batch_size,
                    min_silence_ms=DEFAULT_MIN_SILENCE_MS,
                    min_speech_ms=DEFAULT_MIN_SPEECH_MS,
                    pad_ms=DEFAULT_PAD_MS,
                )
            )
    finally:
        del vad_model, vad_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"vad done for {len(intervals)} clips", flush=True)

    model = MuaalemPhonemeModel.load(device=device, dtype=dtype)
    decodes = []
    for start in range(0, len(staged), args.batch_size):
        batch = staged[start : start + args.batch_size]
        decodes.extend(
            model.decode_batch(
                [waveform for _, waveform, _ in batch], TARGET_SAMPLE_RATE
            )
        )
        print(f"decoded {len(decodes)}/{len(staged)}", flush=True)

    # --- one pass over the staged clips: gate, coverage, seams, excision plan ----------
    pending: list[tuple[ScenarioRecord, ExcisionPlan, object, object]] = []
    for (record, waveform, provenance), decoded, speech in zip(
        staged, decodes, intervals
    ):
        reference = references[record.surah_ayah]
        try:
            word_ref, word_offsets = word_reference(_uthmani_words(record.surah_ayah))
        except (KeyError, IndexError):
            tally.phonetizer_unsupported += 1
            continue
        if word_ref != reference:
            tally.reference_mismatch += 1
            continue

        duration_s = len(waveform) / TARGET_SAMPLE_RATE
        decode, decode_times = collapse_with_times(
            list(decoded.class_ids), duration_s / max(len(decoded.class_ids), 1)
        )
        # Two alignments, deliberately. The **three-ayah** one (#67) is what the covered
        # span is read from: it is the drag-resistant view, and a clip that ran into its
        # neighbour would otherwise have that neighbour's phonemes counted as coverage of
        # this ayah. The **two-ayah** one is what the seams come off, because an insertion
        # run measured against `prev | this | next` would lose any repeat the neighbour's
        # text happens to match — and it is the alignment the gate itself scores, so the
        # runs found here are the runs `max_insertion_run` mined the clip on.
        gate = BALANCED_SCORER.gate(decode, reference)
        verdict = detector.detect(decode, record.surah_ayah)
        alignment = smith_waterman(
            query=normalize_phonemes(decode).normalized, reference=reference
        )
        pauses = pauses_from_intervals([(float(a), float(b)) for a, b in speech])
        seams = seam_coverage(
            record.audio_filename,
            record.surah_ayah,
            record.reciter_id,
            alignment,
            normalized_onsets(decode, decode_times),
            duration_s,
            pauses,
        )
        plan = plan_excision(seams, duration_s)
        word_start, word_end = covered_word_range(
            reference, word_offsets, verdict.ref_covered_start, verdict.ref_covered_end
        )

        clip_id = Path(record.audio_filename).stem
        sf.write(
            audio_dir / f"{clip_id}.wav", waveform, TARGET_SAMPLE_RATE, subtype="PCM_16"
        )
        tally.staged += 1
        tally.seams += len(seams)
        tally.clips_with_multiple_seams += int(len(seams) > 1)
        tally.early_starts += int(warrants_early_start(gate.leading_trim))
        tally.truncated += int(verdict.uncovered_tail > 0)
        pending.append(
            (
                ScenarioRecord(
                    schema_version=SCHEMA_VERSION,
                    clip_id=clip_id,
                    audio=f"{AUDIO_DIR}/{clip_id}.wav",
                    surah_ayah=record.surah_ayah,
                    reciter_id=record.reciter_id,
                    duration_s=round(duration_s, 4),
                    match_ratio=gate.match_ratio,
                    max_insertion_run=gate.max_insertion_run,
                    leading_trim=gate.leading_trim,
                    trailing_trim=gate.trailing_trim,
                    added_shadda=gate.added_shadda,
                    predicted_phonemes=decode,
                    word_start=word_start,
                    word_end=word_end,
                    ref_covered_start=verdict.ref_covered_start,
                    ref_covered_end=verdict.ref_covered_end,
                    ref_length=verdict.ref_length,
                    uncovered_head=verdict.uncovered_head,
                    uncovered_tail=verdict.uncovered_tail,
                    early_start=warrants_early_start(gate.leading_trim),
                    recut_applied=provenance["recut_applied"],
                    recitation_start_s=round(provenance["recitation_start_s"], 4),
                    recitation_end_s=round(provenance["recitation_end_s"], 4),
                    seams=build_seams(seams),
                ),
                plan,
                waveform,
                gate,
            )
        )

    # --- the excision differential: cut, re-decode, re-gate, keep or discard -----------
    attempts = [item for item in pending if item[1].usable]
    tally.pairs_attempted = len(attempts)
    if attempts:
        excised_dir.mkdir(parents=True, exist_ok=True)
    cut_waveforms = [
        excise(waveform, TARGET_SAMPLE_RATE, plan) for _, plan, waveform, _ in attempts
    ]
    cut_decodes = []
    for start in range(0, len(cut_waveforms), args.batch_size):
        cut_decodes.extend(
            model.decode_batch(
                cut_waveforms[start : start + args.batch_size], TARGET_SAMPLE_RATE
            )
        )

    validations: dict[str, tuple[ExcisionValidation, object, str]] = {}
    for (scenario, _, _, gate), cut, decoded in zip(attempts, cut_waveforms, cut_decodes):
        after = BALANCED_SCORER.gate(decoded.phonemes, references[scenario.surah_ayah])
        validations[scenario.clip_id] = (
            validate_excision(gate, after),
            cut,
            decoded.phonemes,
        )

    records: list[ScenarioRecord] = []
    for scenario, plan, _, _ in pending:
        validated = validations.get(scenario.clip_id)
        if validated is None:
            records.append(
                replace_excision(scenario, None, None, None, {"reason": plan.reason})
            )
            tally.refuse(plan.reason)
            continue
        validation, cut, phonemes = validated
        if not validation.accepted:
            records.append(
                replace_excision(scenario, None, None, phonemes, asdict(validation))
            )
            tally.refuse(validation.reason)
            continue
        sf.write(
            excised_dir / f"{scenario.clip_id}.wav", cut, TARGET_SAMPLE_RATE,
            subtype="PCM_16",
        )
        tally.pairs_kept += 1
        records.append(
            replace_excision(
                scenario,
                f"{EXCISED_DIR}/{scenario.clip_id}.wav",
                round(len(cut) / TARGET_SAMPLE_RATE, 4),
                phonemes,
                asdict(validation),
            )
        )

    write_scenario_records(args.out / "scenario.jsonl", records)
    with open(args.out / "staging.json", "w", encoding="utf-8") as f:
        json.dump(
            {"schema_version": SCHEMA_VERSION, **asdict(tally)},
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    print(
        f"staged {tally.staged} clips, {tally.seams} seams; "
        f"{tally.pairs_kept} of {tally.pairs_attempted} excision pairs kept "
        f"(refusals: {tally.pair_refusals or 'none'})"
    )
    print(f"wrote {args.out / 'scenario.jsonl'}")


def replace_excision(
    record: ScenarioRecord,
    excised_audio: str | None,
    excised_duration_s: float | None,
    excised_phonemes: str | None,
    excision: dict | None,
) -> ScenarioRecord:
    """``record`` with its excision outcome filled in."""
    return replace(
        record,
        excised_audio=excised_audio,
        excised_duration_s=excised_duration_s,
        excised_phonemes=excised_phonemes,
        excision=excision,
    )


if __name__ == "__main__":
    main()

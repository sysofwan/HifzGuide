"""The tracer's seeded clip draw and the provenance a committed clip list carries (#307).

The clips come from the claim-gated phase A rows of ``confirm.1`` and ``confirm.2`` — never the
raw tranche files (``tadabur.tlog_mining.selection.phase_a_rows``) — so a draw can only name
rows the committed claims and split tree give to those tranches. The draw is one shuffle of the
distinct rows under a seed and a prefix of it, so 100 clips keep the first 20 a 20-clip draw
picked, and a rerun reproduces the list exactly.
"""

from __future__ import annotations

import json
import random
import wave
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

#: What a clip's provenance line holds: the source row's own fields, unchanged, so a clip can
#: be fetched (``audio_path``/``audio_sha256``), checked against the input it came from
#: (``split``/``shard_file``/``row``), and referenced (``filename_label`` names its ayah).
PROVENANCE_FIELDS = (
    "audio_path", "split", "shard_file", "row", "audio_sha256",
    "filename_label", "stratum", "group_key",
)

#: The two tranches the owner's PRD #305 opens for the audit (``CONFIRM_TRANCHES[:2]``).
TRANCHES = ("confirm.1", "confirm.2")

#: The one sample rate the driver accepts, and the source clips' own.
SAMPLE_RATE = 16_000

#: How the committed word times record a clip whose ayah the phonetizer cannot read. Such a
#: clip has no reference phonemes at all, so the Oracle arm cannot exist for it; the recorded
#: fallback (:attr:`tadabur.followalong_audit.oracle.OracleArm.aligned`) needs a reference.
UNREADABLE_REFERENCE = "the phonetizer cannot handle"


def _row_key(row: Mapping) -> tuple[str, str, int]:
    return (row["split"], row["shard_file"], row["row"])


def draw_clips(rows: Iterable[Mapping], size: int, *, seed: int) -> list[dict]:
    """``size`` distinct source rows drawn from ``rows`` under ``seed``, in a fixed order.

    Rows are deduplicated by their source key and sorted before the seeded shuffle, so the
    draw depends on the set of rows and the seed only — not on the order the caller read them
    in. Taking a prefix keeps a larger draw an extension of a smaller one.
    """
    distinct = {_row_key(row): row for row in rows}
    if size > len(distinct):
        raise ValueError(f"cannot draw {size} clips from {len(distinct)} distinct rows")
    ordered = [distinct[key] for key in sorted(distinct)]
    random.Random(seed).shuffle(ordered)
    return [{name: row[name] for name in PROVENANCE_FIELDS} for row in ordered[:size]]


def gating_rows(tranches: Sequence[str] = TRANCHES) -> list[dict]:
    """Every phase A gating row of ``tranches``, through the claim-gated reader."""
    from tadabur.tlog_mining.selection import phase_a_rows

    return [row for tranche in tranches for row in phase_a_rows(tranche)]


def oracle_eligible(rows: Iterable[Mapping], word_times: Mapping[str, Mapping]) -> list[dict]:
    """The rows whose Oracle arm can exist: their committed word times hold a reference.

    A clip whose ayah the phonetizer cannot read has no reference to place, so it can never
    produce an Oracle table; the draw leaves it out rather than report a clip that is bound to
    fail. #248's exclusion is pinned once in
    ``tadabur/consonant2_readout/report.json`` (``coverage.graded.excluded_clips``, read by
    :func:`tadabur.tlog_mining.candidate_inventory.excluded_248`): the twenty clips of seven ayat
    (``55:17``, ``90:8``, ``90:10``, ``106:1``–``106:4``) measured on this same phase A
    population. The committed word times record the same exclusion as a reason, so a row either
    source refuses is left out: the pin cannot be edited away, and a word time that has no
    reference cannot be graded whatever the report says. A clip whose alignment failed for
    another reason (posteriors too short) keeps its reference and stays eligible: its Oracle arm
    falls back to the even spread, marked.
    """
    from tadabur.tlog_mining.candidate_inventory import excluded_248
    from training.nonpro_decode import item_key

    excluded = excluded_248()
    eligible = []
    for row in rows:
        key = item_key(row["audio_path"])
        record = word_times.get(key)
        if record is None:
            continue
        if key in excluded or str(record.get("reason") or "").startswith(UNREADABLE_REFERENCE):
            continue
        eligible.append(row)
    return eligible


def app_arm_rows(records: Mapping[str, Mapping] | None = None) -> list[dict]:
    """The App-arm population: the claim-gated gating rows of confirm.1/2 the audit can grade.

    #248's twenty clips (seven ayat the phonetizer cannot read) are left out, exactly as the
    tracer's draw does — they have no reference at all, so no arm can grade them. ``records`` is
    the committed word-times table (``{item: {"reason": ...}}``), read from the committed phase A
    shards when omitted. The tranche reader is :func:`gating_rows`, so an unclaimed tranche stops
    the population instead of shrinking it.
    """
    if records is None:
        from tadabur.followalong_audit.tracer import load_word_times

        records = load_word_times()
    return oracle_eligible(gating_rows(), records)


def selection_record(rows: Iterable[Mapping], eligible: Iterable[Mapping], *,
                     size: int, seed: int) -> dict:
    """What a draw selected from: the population, #248's exclusion, and the seed.

    Written beside the clip list, so the committed output states its own selection instead of
    leaving ``--seed``/``--size`` to a shell history.
    """
    from training.nonpro_decode import item_key

    rows = list(rows)
    kept = {item_key(row["audio_path"]) for row in eligible}
    excluded = [row for row in rows if item_key(row["audio_path"]) not in kept]
    return {
        "seed": seed,
        "size": size,
        "phase_a_rows": len(rows),
        "eligible_rows": len(kept),
        "excluded_rows": len(excluded),
        "excluded_ayat": sorted({row["filename_label"] for row in excluded}),
        "excluded_items": sorted(item_key(row["audio_path"]) for row in excluded),
        "exclusion": "#248, tadabur/consonant2_readout/report.json coverage.graded.excluded_clips",
    }


def write_clips(clips: Sequence[Mapping], path: Path) -> None:
    """Write ``clips`` as one JSON object per line, keys sorted, UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(clip, ensure_ascii=False, sort_keys=True) + "\n" for clip in clips)
    path.write_text(text, encoding="utf-8")


def read_clips(path: Path) -> list[dict]:
    """The clip list ``write_clips`` wrote."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def selected_clips(population: Sequence[Mapping], named: Sequence[Mapping]) -> list[dict]:
    """The named list's clips as the population's own rows, refusing anything else.

    A named list is a *selection from* :func:`app_arm_rows`, never a substitute for it: the
    claim-gated population is the only authority on which clips this audit grades and against
    which ayah, so every record must name a population clip and repeat its
    :data:`PROVENANCE_FIELDS` exactly (extra fields are ignored, as a tranche file's are). An
    unknown record, a repeat, or a changed provenance/reference is refused — with the WAVs
    present, silently accepting one would grade an unopened tranche's clip or another ayah.
    """
    by_sha = {row["audio_sha256"]: row for row in population}
    chosen: dict[str, dict] = {}
    for record in named:
        sha = record.get("audio_sha256")
        row = by_sha.get(sha)
        if row is None:
            raise ValueError(
                f"{record.get('audio_path', record)} is not one of the population's "
                f"{len(population)} claim-gated confirm.1/confirm.2 clips: a named clip list "
                "selects from that population, it does not replace it")
        if sha in chosen:
            raise ValueError(f"{row['audio_path']} appears twice in the clip list")
        differing = [field for field in PROVENANCE_FIELDS if record.get(field) != row[field]]
        if differing:
            raise ValueError(
                f"{row['audio_path']} disagrees with the population on {differing}: "
                f"{ {field: record.get(field) for field in differing} }, the population says "
                f"{ {field: row[field] for field in differing} }")
        chosen[sha] = row
    return list(chosen.values())


def draw_record_path(clip_list: Path) -> Path:
    """Where a clip list's selection record sits: ``clips.jsonl`` → ``clips.draw.json``."""
    return Path(clip_list).with_suffix(".draw.json")


def wav_samples(path: Path) -> int:
    """The frame count of a 16 kHz mono 16-bit WAV, the one format the driver decodes.

    Refusing another rate here keeps a wrong clip out of the run rather than letting the driver
    transcribe it at the wrong speed.
    """
    with wave.open(str(path), "rb") as source:
        if source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path} is {source.getframerate()} Hz, not {SAMPLE_RATE} Hz")
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"{path} is not mono 16-bit PCM")
        return source.getnframes()


def verify_wav(path: Path, expected: Mapping | None) -> str:
    """Refuse a WAV whose bytes or length differ from the fetch's digest manifest; its SHA-256.

    The manifest is written where the audio was fetched (``<sha>.wav`` → WAV SHA-256 and sample
    count) and committed with the run, so a truncated rsync, a re-encode or a substituted file
    cannot enter the table the audit reports. The clip list's own ``audio_sha256`` names the
    source clip; the WAV is a different container and is bound by this manifest.
    """
    if expected is None:
        raise ValueError(f"{path} has no entry in the fetch's digest manifest")
    from tadabur.tlog_index import sha256_file

    found = sha256_file(path)
    if found != expected.get("wav_sha256"):
        raise ValueError(f"{path} is {found}, not the fetched {expected.get('wav_sha256')}")
    frames = wav_samples(path)
    if expected.get("samples") is not None and frames != expected["samples"]:
        raise ValueError(f"{path} holds {frames} samples, not the fetched {expected['samples']}")
    return found

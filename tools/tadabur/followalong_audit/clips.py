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
    fail. This is exactly #248's exclusion — the twenty clips of seven ayat (``55:17``,
    ``90:8``, ``90:10``, ``106:1``–``106:4``) that
    ``tadabur/consonant2_readout/report.json``'s ``coverage.graded.excluded_clips`` pins at
    SHA-256 ``2149ea0f…``, measured on the same phase A population. A clip whose alignment
    failed for another reason (posteriors too short) keeps its reference and stays eligible:
    its Oracle arm falls back to the even spread, marked.
    """
    from training.nonpro_decode import item_key

    eligible = []
    for row in rows:
        record = word_times.get(item_key(row["audio_path"]))
        if record is None:
            continue
        if str(record.get("reason") or "").startswith(UNREADABLE_REFERENCE):
            continue
        eligible.append(row)
    return eligible


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

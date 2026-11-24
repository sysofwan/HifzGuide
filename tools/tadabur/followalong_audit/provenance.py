"""The committed audio provenance of the App-arm population (#309).

One line per clip: where the source audio lives (``split``/``shard_file``/``row``), what it is
(``audio_path``/``audio_sha256``) and the WAV the App arm graded (``wav_sha256``/``samples``).
The claim-gated tranche rows are the authority; the file is checked against them, so a manifest
that names a row the tranches do not, misses one, or describes one differently than the tranche
says is refused. The WAV digests come from the fetch's own manifest, so the committed file also
binds the audio that was graded to the bytes that were fetched.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.followalong_audit.atomic import write_bytes_atomically

#: The committed manifest: beside the run's other tables, one gzipped JSON line per clip.
PROVENANCE_PATH = Path(__file__).parent / "app_arm" / "audio_manifest.jsonl.gz"

#: What a line holds: where the audio lives, what it is, and the WAV that was graded.
PROVENANCE_COLUMNS = (
    "audio_path", "split", "shard_file", "row", "samples", "audio_sha256", "wav_sha256",
)

#: The fields a line takes from the tranche row itself (the rest come from the fetch's manifest).
_SOURCE_FIELDS = ("audio_path", "split", "shard_file", "row", "audio_sha256")


def _source_key(line: Mapping) -> tuple[str, str, int]:
    return (line["split"], line["shard_file"], line["row"])


def provenance_rows(rows: Sequence[Mapping],
                    wav_manifest: Mapping[str, Mapping]) -> list[dict]:
    """The manifest's lines, in source order (split, shard, row), whatever the input's order.

    Every row must have been fetched: a missing WAV entry is an error naming the clip's
    ``audio_path``, never a line with a made-up digest.
    """
    lines = []
    for row in rows:
        fetched = wav_manifest.get(row["audio_sha256"])
        if fetched is None:
            raise ValueError(f"{row['audio_path']} has no WAV in the fetch's manifest")
        lines.append({
            "audio_path": row["audio_path"],
            "split": row["split"],
            "shard_file": row["shard_file"],
            "row": row["row"],
            "samples": fetched["samples"],
            "audio_sha256": row["audio_sha256"],
            "wav_sha256": fetched["wav_sha256"],
        })
    return sorted(lines, key=_source_key)


def check_provenance(lines: Sequence[Mapping], rows: Sequence[Mapping]) -> None:
    """Refuse a manifest that is not exactly ``rows``, one line each, fields included."""
    expected = {_source_key(row): row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("the tranche rows repeat a source row")
    found: dict[tuple[str, str, int], Mapping] = {}
    for line in lines:
        if set(line) != set(PROVENANCE_COLUMNS):
            raise ValueError(f"{line.get('audio_path', line)}: a line holds "
                             f"{sorted(line)}, not {sorted(PROVENANCE_COLUMNS)}")
        key = _source_key(line)
        if key in found:
            raise ValueError(f"{key} appears twice in the manifest")
        found[key] = line
    missing, extra = sorted(expected.keys() - found.keys()), sorted(found.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f"the manifest is not the tranche's rows: {len(missing)} missing "
                         f"({missing[:3]}), {len(extra)} extra ({extra[:3]})")
    for key, line in found.items():
        row = expected[key]
        differing = [field for field in _SOURCE_FIELDS if line[field] != row[field]]
        if differing:
            raise ValueError(f"{row['audio_path']}: the manifest disagrees on {differing}: "
                             f"{ {field: line[field] for field in differing} }, the tranche "
                             f"says { {field: row[field] for field in differing} }")


def read_provenance(path: Path = PROVENANCE_PATH) -> list[dict]:
    """The committed manifest's lines, in file order."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_provenance(lines: Sequence[Mapping], path: Path = PROVENANCE_PATH) -> None:
    """Write the lines as one gzipped JSON object each, keys sorted, fixed mtime: byte-identical."""
    body = "".join(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n" for line in lines)
    write_bytes_atomically(Path(path), gzip.compress(body.encode("utf-8"), mtime=0))

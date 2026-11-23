"""A committed decode read through the candidate inventory's SHA-256 pins (#309).

The detector and bridge decodes are named and hashed in
``tadabur.tlog_mining.candidate_inventory.PINS``: bytes that do not hash to the pin are refused,
so a re-run never grades a file the report did not name. This module only parses; the pins stay
in the module that owns them.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tadabur.tlog_mining.candidate_inventory import ROOT, read_pinned


def pinned_records(name: str, root: Path = ROOT) -> list[dict]:
    """The pinned decode's JSON lines, in file order, refused unless its bytes match the pin."""
    data = read_pinned(name, root)
    try:
        text = gzip.decompress(data).decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is not a gzipped UTF-8 decode file") from error
    return [json.loads(line) for line in text.splitlines() if line.strip()]

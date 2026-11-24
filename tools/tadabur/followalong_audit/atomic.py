"""Atomic writes for the audit's tables (#309).

A killed run must leave the previous file or the new one, never a torn one. The temporary
sibling carries the writer's pid, so two pool workers writing their own traces, results or
manifests can never collide on one temporary name.
"""

from __future__ import annotations

import os
from pathlib import Path


def write_text_atomically(path: Path, text: str) -> None:
    """``text`` to ``path`` through a pid-suffixed sibling, then over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_bytes_atomically(path: Path, data: bytes) -> None:
    """``data`` to ``path`` through a pid-suffixed sibling, then over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)

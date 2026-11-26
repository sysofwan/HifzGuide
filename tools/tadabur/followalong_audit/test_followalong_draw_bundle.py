"""A published draw is frozen: a re-derivation either matches it or is refused (#311, PR 1).

``build`` derives every file into a staging directory and then :func:`build.publish`es it, so a
draw someone is listening to can never be silently replaced: an id a verdict names would then
belong to another word. The sample (frame, gate, key, worklist, plans, snips) must match byte
for byte; only the ``draw_seed.json`` record may be re-recorded, and a bundle that already
matches writes nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tadabur.followalong_audit import build as builder
from tadabur.followalong_audit.build import RECORD_FILES, SAMPLE_FILES, publish

BUNDLE = (*SAMPLE_FILES, *RECORD_FILES)


def write_bundle(directory: Path, *, body: bytes = b"same", **overrides: bytes) -> Path:
    """A staged bundle whose files all hold ``body`` unless a name overrides it."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE:
        (directory / name).write_bytes(overrides.get(name, body))
    return directory


@pytest.fixture
def writes(monkeypatch) -> list[str]:
    """Every publishing write, without touching the disk twice."""
    written: list[str] = []
    real = builder.write_bytes_atomically

    def record(path: Path, data: bytes) -> None:
        written.append(path.name)
        real(path, data)

    monkeypatch.setattr(builder, "write_bytes_atomically", record)
    return written


def test_a_bundle_that_already_matches_writes_nothing(tmp_path, writes):
    out = write_bundle(tmp_path / "out")
    assert publish(write_bundle(tmp_path / "staged"), out) == []
    assert writes == []


def test_a_changed_sample_is_refused_before_anything_is_written(tmp_path, writes):
    out = write_bundle(tmp_path / "out")
    staged = write_bundle(tmp_path / "staged", **{"key.jsonl": b"another draw"})
    with pytest.raises(ValueError, match="key.jsonl"):
        publish(staged, out)
    assert writes == []
    assert sorted(path.name for path in out.iterdir()) == sorted(BUNDLE)
    assert (out / "key.jsonl").read_bytes() == b"same"


def test_a_changed_snip_sidecar_is_refused_too(tmp_path):
    out = write_bundle(tmp_path / "out")
    staged = write_bundle(tmp_path / "staged", **{"snips.jsonl": b"other audio"})
    with pytest.raises(ValueError, match="snips.jsonl"):
        publish(staged, out)


def test_the_record_may_be_re_recorded_without_a_redraw(tmp_path, writes):
    out = write_bundle(tmp_path / "out")
    staged = write_bundle(tmp_path / "staged", **{"draw_seed.json": b'{"draw_commit": "b"}'})
    assert publish(staged, out) == ["draw_seed.json"]
    assert writes == ["draw_seed.json"]
    assert (out / "draw_seed.json").read_bytes() == b'{"draw_commit": "b"}'


def test_an_empty_directory_gets_the_whole_bundle(tmp_path, writes):
    out = tmp_path / "out"
    out.mkdir()
    assert publish(write_bundle(tmp_path / "staged"), out) == list(BUNDLE)
    assert sorted(path.name for path in out.iterdir()) == sorted(BUNDLE)

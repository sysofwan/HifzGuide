"""The driver's model assets are release v1.0.7's, verified against its ``manifest.json`` (#307)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tadabur.followalong_audit.assets import verify_assets

FILES = {"models.zip": b"model bytes", "mel_filters.bin": b"mel", "window.bin": b"window"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(**overrides: str) -> dict:
    """A release manifest in ``manifest.json``'s shape; it lists files the driver never reads
    too, which must not be required."""
    assets = [{"filename": name, "sha256": overrides.get(name, sha256(data)), "size": len(data)}
              for name, data in FILES.items()]
    assets += [{"filename": "quran.db", "sha256": sha256(b"db"), "size": 2},
               {"filename": "fonts.zip", "sha256": sha256(b"fonts"), "size": 5}]
    return {"version": "v1.0.7", "assets": assets}


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    for name, data in FILES.items():
        (tmp_path / name).write_bytes(data)
    return tmp_path


def test_assets_that_match_the_manifest_are_accepted(assets):
    verify_assets(assets, manifest())


@pytest.mark.parametrize("name", sorted(FILES))
def test_an_asset_whose_checksum_differs_is_refused_by_name(assets, name):
    (assets / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match=name):
        verify_assets(assets, manifest())


@pytest.mark.parametrize("name", sorted(FILES))
def test_a_manifest_checksum_that_differs_from_the_file_is_refused(assets, name):
    with pytest.raises(ValueError, match=name):
        verify_assets(assets, manifest(**{name: sha256(b"another file")}))


@pytest.mark.parametrize("name", sorted(FILES))
def test_a_missing_asset_is_refused(assets, name):
    (assets / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=name):
        verify_assets(assets, manifest())


def test_an_asset_the_manifest_does_not_list_is_refused(assets):
    listed = manifest()
    listed["assets"] = [a for a in listed["assets"] if a["filename"] != "window.bin"]
    with pytest.raises(ValueError, match="window.bin"):
        verify_assets(assets, listed)


def test_a_manifest_of_another_release_is_refused(assets):
    with pytest.raises(ValueError, match=r"v1\.0\.7"):
        verify_assets(assets, {**manifest(), "version": "v1.0.6"})

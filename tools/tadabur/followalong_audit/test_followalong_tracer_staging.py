"""Staging the verified release for the driver (#307).

``MuaalemInference`` reads ``<HIFZGUIDE_ASSETS_DIR>/MuaalemStudent_h448_6BIT.mlmodelc`` — not the
release's ``models.zip`` — so verifying the archive is not enough: the staged directory must hold
the model the manifest's checksums bind. These tests stage a small stand-in release.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from tadabur.followalong_audit.assets import MODEL_PACKAGE, stage_assets, tree_sha256

FILTERS = {"mel_filters.bin": b"mel", "window.bin": b"window"}
MODEL_MEMBER = f"{MODEL_PACKAGE}.mlmodelc/coremldata.bin"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def release(tmp_path: Path, members: dict[str, bytes] | None = None) -> tuple[Path, dict]:
    directory = tmp_path / "release"
    directory.mkdir()
    for name, data in FILTERS.items():
        (directory / name).write_bytes(data)
    with zipfile.ZipFile(directory / "models.zip", "w") as archive:
        for name, data in (members or {MODEL_MEMBER: b"model"}).items():
            archive.writestr(name, data)
    files = {**FILTERS, "models.zip": (directory / "models.zip").read_bytes()}
    manifest = {"version": "v1.0.7",
                "assets": [{"filename": name, "sha256": sha256(data), "size": len(data)}
                           for name, data in files.items()]}
    return directory, manifest


def test_the_staged_directory_is_the_one_the_driver_reads(tmp_path):
    directory, manifest = release(tmp_path)
    staged = stage_assets(directory, manifest, tmp_path / "staging")
    assert staged.directory == tmp_path / "staging"
    assert staged.model == f"{MODEL_PACKAGE}.mlmodelc"
    assert (staged.directory / MODEL_MEMBER).read_bytes() == b"model"
    assert (staged.directory / "mel_filters.bin").read_bytes() == FILTERS["mel_filters.bin"]
    assert (staged.directory / "window.bin").read_bytes() == FILTERS["window.bin"]
    assert staged.model_sha256 == tree_sha256(staged.directory / staged.model)
    assert staged.checksums == {asset["filename"]: asset["sha256"] for asset in manifest["assets"]}


def test_a_rerun_cannot_leave_another_releases_files_behind(tmp_path):
    directory, manifest = release(tmp_path)
    staging = tmp_path / "staging"
    stage_assets(directory, manifest, staging)
    (staging / "stale.bin").write_bytes(b"old release")
    stage_assets(directory, manifest, staging)
    assert not (staging / "stale.bin").exists()


def test_the_model_digest_changes_with_the_models_bytes(tmp_path):
    directory, manifest = release(tmp_path)
    first = stage_assets(directory, manifest, tmp_path / "staging")
    other = tmp_path / "other"
    other.mkdir()
    for name, data in FILTERS.items():
        (other / name).write_bytes(data)
    with zipfile.ZipFile(other / "models.zip", "w") as archive:
        archive.writestr(MODEL_MEMBER, b"other model")
    changed = {"version": "v1.0.7", "assets": [
        {"filename": name, "sha256": sha256(data), "size": len(data)}
        for name, data in {**FILTERS, "models.zip": (other / "models.zip").read_bytes()}.items()]}
    second = stage_assets(other, changed, tmp_path / "staging2")
    assert first.model_sha256 != second.model_sha256


def test_a_zip_without_the_model_package_is_refused(tmp_path):
    directory, manifest = release(tmp_path, members={"something_else.bin": b"x"})
    with pytest.raises(ValueError, match=MODEL_PACKAGE):
        stage_assets(directory, manifest, tmp_path / "staging")


def test_a_release_whose_bytes_do_not_match_is_refused_before_staging(tmp_path):
    directory, manifest = release(tmp_path)
    (directory / "window.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="window.bin"):
        stage_assets(directory, manifest, tmp_path / "staging")
    assert not (tmp_path / "staging").exists()

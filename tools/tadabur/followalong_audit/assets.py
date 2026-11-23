"""The app arm's model assets: verified against a release, then laid out as the driver reads them.

The driver runs the app's CoreML model, so it must run the release the app ships: v1.0.7's
``models.zip``, ``mel_filters.bin`` and ``window.bin``. A truncated or substituted download
decodes silently worse, which no later stage can tell from a real grade difference, so the
files are refused on any mismatch rather than trusted.

Verification alone is not enough: ``MuaalemInference`` resolves
``<HIFZGUIDE_ASSETS_DIR>/MuaalemStudent_h448_6BIT.mlmodelc`` (or ``.mlpackage``) itself, and the
release ships the model as a zip, so the verified archive is not the directory the driver
reads (:meth:`TranscribeAssets.makeInference`). :func:`stage_assets` therefore expands the
verified zip and copies the verified filters beside it, and the driver is handed that directory
— the bytes graded are the bytes the manifest names.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tadabur.tlog_index import sha256_file

#: The release the audit pins: the same one the Muraja checkout's ``tools/lib/release.env``
#: pins and the harness was built against.
RELEASE = "v1.0.7"

#: The release assets the driver reads. The release ships others (``quran.db``, ``fonts.zip``)
#: that the driver never touches, so they are not required here.
REQUIRED = ("models.zip", "mel_filters.bin", "window.bin")

#: The model package ``TranscribeAssets`` loads, and the two extensions it accepts.
MODEL_PACKAGE = "MuaalemStudent_h448_6BIT"
MODEL_EXTENSIONS = ("mlmodelc", "mlpackage")


@dataclass(frozen=True)
class StagedAssets:
    """The directory the driver reads, and what the release's checksums bind it to."""

    directory: Path
    model: str
    model_sha256: str
    checksums: dict[str, str]


def verify_assets(directory: Path, manifest: Mapping) -> None:
    """Refuse unless ``directory`` holds each :data:`REQUIRED` file at its manifest checksum.

    ``manifest`` is a HifzGuide release's ``manifest.json``; it must be :data:`RELEASE`'s, and
    must list each required file (a manifest that lists a different release, or omits one of
    the three, is refused rather than partially checked).
    """
    version = manifest.get("version")
    if version != RELEASE:
        raise ValueError(f"the assets are release {version!r}, not {RELEASE!r}: the audit runs "
                         f"the release the app ships")
    listed = {asset.get("filename"): asset.get("sha256") for asset in manifest.get("assets", ())}
    directory = Path(directory)
    for name in REQUIRED:
        if name not in listed:
            raise ValueError(f"{name} is not in the {RELEASE} manifest: the driver needs it")
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing from {directory}")
        found = sha256_file(path)
        if found != listed[name]:
            raise ValueError(f"{name} is {found}, not the manifest's {listed[name]}: "
                             f"{directory} is not release {RELEASE}'s assets")


def staged_model(directory: Path) -> Path | None:
    """The model package ``TranscribeAssets`` would load from ``directory``, as it chooses."""
    return next((directory / f"{MODEL_PACKAGE}.{extension}"
                 for extension in MODEL_EXTENSIONS
                 if (directory / f"{MODEL_PACKAGE}.{extension}").is_dir()), None)


def tree_sha256(directory: Path) -> str:
    """A digest over a directory's files: relative path and content, sorted.

    ``.mlmodelc`` is a directory, so no single file names the model that was graded; this
    digest lets the run record it, and differ if a later run stages different bytes.
    """
    digest = hashlib.sha256()
    directory = Path(directory)
    for path in sorted(path for path in directory.rglob("*") if path.is_file()):
        digest.update(str(path.relative_to(directory)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def stage_assets(release_dir: Path, manifest: Mapping, staging_dir: Path) -> StagedAssets:
    """Verify ``release_dir``'s files and build the directory the driver reads from them.

    ``models.zip`` is expanded into ``staging_dir`` (re-created, so a rerun cannot mix
    releases), the verified ``mel_filters.bin``/``window.bin`` are copied next to it, and the
    staged copy of each is re-hashed against the manifest — the driver loads these bytes, so
    these are the bytes that must match. Refuses a zip that holds no ``MuaalemStudent_*``
    package at its root: the driver would fail later with a less specific error.
    """
    verify_assets(Path(release_dir), manifest)
    listed = {asset["filename"]: asset["sha256"] for asset in manifest["assets"]}
    staging_dir = Path(staging_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    with zipfile.ZipFile(Path(release_dir) / "models.zip") as archive:
        archive.extractall(staging_dir)
    model = staged_model(staging_dir)
    if model is None or not any(path.is_file() for path in model.rglob("*")):
        raise ValueError(f"models.zip holds no {MODEL_PACKAGE} package at its root: "
                         f"{staging_dir} cannot back the driver")
    for name in ("mel_filters.bin", "window.bin"):
        shutil.copyfile(Path(release_dir) / name, staging_dir / name)
        found = sha256_file(staging_dir / name)
        if found != listed[name]:
            raise ValueError(f"the staged {name} is {found}, not the manifest's {listed[name]}")
    return StagedAssets(
        directory=staging_dir,
        model=model.name,
        model_sha256=tree_sha256(model),
        checksums={name: listed[name] for name in REQUIRED},
    )

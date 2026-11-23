"""The app arm's model assets, verified against a release's ``manifest.json`` (#307).

The driver runs the app's CoreML model, so it must run the release the app ships: v1.0.7's
``models.zip``, ``mel_filters.bin`` and ``window.bin``. A truncated or substituted download
decodes silently worse, which no later stage can tell from a real grade difference, so the
files are refused on any mismatch rather than trusted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

#: The release the audit pins: the same one the Muraja checkout's ``tools/lib/release.env``
#: pins and the harness was built against.
RELEASE = "v1.0.7"

#: The release assets the driver reads. The release ships others (``quran.db``, ``fonts.zip``)
#: that the driver never touches, so they are not required here.
REQUIRED = ("models.zip", "mel_filters.bin", "window.bin")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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

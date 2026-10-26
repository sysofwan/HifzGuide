"""S5b (#204): fetch the two fresh negative sources and write their manifests.

Two sources, one draw rule each, both fixed by ``docs/tlog-ident-floor-preregistration.md``
(whose SHA-256 is pinned in :mod:`tadabur.tlog_ident`):

* **Freesound** (``freesound``): stream the 100 shared shards of
  ``obadx/freesound-commercial-50k-noise-only`` @ the pinned revision in salted-rank order,
  keep the eligible recordings that are in neither committed pool, stop once 1,000 are eligible
  (or at 10 shards, reported), delete each shard after reading it, and write
  ``tools/tadabur/tlog/ident/confirm/freesound.json``. The manifest carries the full shard
  ranking, the drawn prefix, the eligible rows of those shards, the per-shard exclusions by
  reason and the licence mix, so the selection is re-derivable offline from the salt.
* **Common Voice** (``common-voice``): download the six primary MDC archives with ``curl -C -``
  into scratch, verifying each against the SHA-256 and size pinned in the pre-registration. The
  key never leaves ``cuda-dev``: the Mac asks it for a presigned URL (12 h, re-requested on
  expiry) and passes the URL to curl through a 0600 config file, so neither the key nor the URL
  reaches a log or an argv. The draw itself is the next step; this command only fetches bytes.

Both write into scratch, never into the repo (except the Freesound manifest) and commit no audio.

Usage (from ``tools/``)::

    python -m tadabur.tlog_ident_fetch freesound --cache-dir /tmp/s5b-freesound
    python -m tadabur.tlog_ident_fetch common-voice \\
        --archives-dir /Volumes/NetBackup/HifzGuide/scratch/common-voice/archives
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from tadabur import tlog_ident_confirm as confirm
from tadabur.audio import decode_to_mono_16k
from tadabur.nonpro_sources import Dataset, iter_rows
from tadabur.panel_seal import SealedPanelError
from tadabur.tlog_ident import (FLOOR_PREREGISTRATION_PATH, FLOOR_PREREGISTRATION_SHA256)
from tadabur.tlog_index import repo_relative, sha256_file
from training.perturbation import NOISE_LICENCES, NoiseRecording, pool_eligible

ROOT = Path(__file__).resolve().parents[2]
NONPRO_GAP_POOL = ROOT / "tools" / "tadabur" / "nonpro_gap" / "noise_pool.json"
AUGMENTATION_POOL = ROOT / "tools" / "tadabur" / "augmentation" / "noise_pool.json"

#: Mozilla Data Collective: the presigned-URL endpoint and where the key lives on the GPU box.
MDC_API = "https://mozilladatacollective.com/api"
MDC_REMOTE = "root@cuda-dev"
#: The six primary locales, in the order the doc lists them.
CV_PRIMARY_LOCALES = ("ar", *confirm.PRIMARY_LOCALES)
#: How many times a download may re-request an expired URL before the command fails.
DOWNLOAD_ATTEMPTS = 6

#: One row of the pre-registration's §4d archive-pin table.
_ARCHIVE_PIN = re.compile(
    r"^\|\s*`(?P<locale>[a-z]{2,3})`\s*\|\s*(?P<role>\w+)\s*\|"
    r"\s*`(?P<dataset_id>[^`]+)`\s*\|\s*`(?P<filename>[^`]+)`\s*\|"
    r"\s*`(?P<sha256>[0-9a-f]{64})`\s*\|\s*(?P<size>[\d,]+)\s*\|$")


# --- the pre-registration this fetch pins ------------------------------------------------------


def _provenance(doc_path: Path = FLOOR_PREREGISTRATION_PATH,
                repo: Path = ROOT) -> dict:
    """The doc's path, SHA-256 and the commit that last touched it, plus the PR pair.

    ``pull_request`` stays null until the slice opens its PR; the tests accept the null pair and
    order the chain only when the recorded commits exist in the clone.
    """
    relative = repo_relative(doc_path, repo)
    result = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%H", "--", relative],
        check=True, capture_output=True, text=True)
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError(f"{relative} has no commit to cite")
    sha = sha256_file(doc_path)
    if sha != FLOOR_PREREGISTRATION_SHA256:
        raise RuntimeError(f"{relative} is {sha}, not the pinned {FLOOR_PREREGISTRATION_SHA256}")
    return {"doc": {"path": relative, "sha256": sha, "commit": commit},
            "pull_request": None, "pull_request_ref": None}


def archive_pins(doc_path: Path = FLOOR_PREREGISTRATION_PATH) -> dict[str, dict]:
    """The 11 MDC archive pins from the doc's §4d table, keyed by locale."""
    pins: dict[str, dict] = {}
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        match = _ARCHIVE_PIN.match(line.strip())
        if match is None:
            continue
        locale = match["locale"]
        pins[locale] = {"locale": locale, "role": match["role"],
                        "dataset_id": match["dataset_id"], "filename": match["filename"],
                        "sha256": match["sha256"],
                        "size_bytes": int(match["size"].replace(",", ""))}
    expected = {"ar", *confirm.PRIMARY_LOCALES, *confirm.SUBSTITUTES.values()}
    if set(pins) != expected or len(pins) != len(expected):
        raise RuntimeError(f"the doc pins {sorted(pins)}, not {sorted(expected)}")
    return pins


# --- Freesound ---------------------------------------------------------------------------------


def _pool_keys() -> tuple[set[int], set[str], set[int], set[str]]:
    """The ids and ``samples_sha256``s of the two committed pools, one pair each."""
    keys = []
    for path in (NONPRO_GAP_POOL, AUGMENTATION_POOL):
        recordings = json.loads(path.read_text(encoding="utf-8"))["recordings"]
        keys.append({row["freesound_id"] for row in recordings})
        keys.append({row["samples_sha256"] for row in recordings})
    return keys[0], keys[1], keys[2], keys[3]


def _shard_dataset(shard: str) -> Dataset:
    return Dataset("freesound_confirm", confirm.FREESOUND_REPO, confirm.FREESOUND_REVISION,
                   (shard,), "per recording", False)


def freesound_manifest(cache_dir: Path, *,
                       size: int = confirm.FREESOUND_SIZE,
                       max_shards: int = confirm.MAX_SHARDS) -> dict:
    """Stream the ranked shards until ``size`` recordings are eligible, then the manifest."""
    nonpro_ids, nonpro_samples, augmentation_ids, augmentation_samples = _pool_keys()
    ranked = confirm.shard_ranking(confirm.candidate_shards())
    eligible: list[dict] = []
    drawn: list[str] = []
    left_out: dict[str, dict[str, int]] = {}
    for shard in ranked[:max_shards]:
        counts = {"undecodable": 0, "ineligible": 0, "in_nonpro_gap_pool": 0,
                  "in_augmentation_pool": 0}
        for row in iter_rows(_shard_dataset(shard), shard, cache_dir):
            try:
                samples = decode_to_mono_16k(row["audio"]["bytes"])
            except SealedPanelError:  # a seal violation is never a corrupt input
                raise
            except Exception:  # any format libsndfile cannot read is counted
                counts["undecodable"] += 1
                continue
            if not pool_eligible(samples):
                counts["ineligible"] += 1
                continue
            record = NoiseRecording(int(row["freesound_id"]), NOISE_LICENCES[row["license"]],
                                    row["username"], samples).as_dict()
            if (record["freesound_id"] in nonpro_ids
                    or record["samples_sha256"] in nonpro_samples):
                counts["in_nonpro_gap_pool"] += 1
                continue
            if (record["freesound_id"] in augmentation_ids
                    or record["samples_sha256"] in augmentation_samples):
                counts["in_augmentation_pool"] += 1
                continue
            eligible.append({**record, "shard": shard})
        left_out[shard] = counts
        drawn.append(shard)
        print(f"{shard}: {len(eligible)} eligible after {len(drawn)} shard(s), "
              f"left out {counts}", flush=True)
        if len(eligible) >= size:
            break
    recordings = confirm.freesound_select(eligible, count=size)
    return {
        "salt": confirm.SALT,
        "dataset": {"repo": confirm.FREESOUND_REPO,
                    "revision": confirm.FREESOUND_REVISION,
                    "shards": list(ranked)},
        "drawn_shards": drawn,
        "eligible": eligible,
        "recordings": recordings,
        "short": size - len(recordings),
        "licences": dict(sorted(Counter(row["licence"] for row in recordings).items())),
        "left_out": left_out,
        "provenance": _provenance(),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def freesound(args: argparse.Namespace) -> None:
    confirm.require_committed([FLOOR_PREREGISTRATION_PATH])
    manifest = freesound_manifest(Path(args.cache_dir))
    _write_json(Path(args.out), manifest)
    if manifest["short"]:
        print(f"stopped at {len(manifest['drawn_shards'])} shards, "
              f"{manifest['short']} short of {confirm.FREESOUND_SIZE}", flush=True)
    print(f"wrote {args.out}: {len(manifest['recordings'])} recordings from "
          f"{len(manifest['drawn_shards'])} shard(s)", flush=True)


# --- Common Voice: the archives, and only the archives ------------------------------------------


def presigned_url(dataset_id: str, *, remote: str = MDC_REMOTE) -> str:
    """Ask ``cuda-dev`` to mint a download URL; the key is read there and never printed."""
    remote_command = (
        'set -euo pipefail; '
        'key=$(cat ~/.config/mozilla-data-collective/api_key); '
        'curl -sS -f -X POST -H "Authorization: Bearer $key" '
        f"'{MDC_API}/datasets/{dataset_id}/download'")
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", remote, remote_command],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"the presigned URL request failed for {dataset_id}: "
                           f"{result.stderr.strip()[:400]}")
    try:
        return json.loads(result.stdout)["downloadUrl"]
    except (json.JSONDecodeError, KeyError) as error:
        raise RuntimeError(f"the presigned URL response for {dataset_id} had no downloadUrl") \
            from error


def download_archive(locale: str, pin: dict, archives_dir: Path, *,
                     remote: str = MDC_REMOTE, attempts: int = DOWNLOAD_ATTEMPTS) -> None:
    """Fetch one archive with ``curl -C -``, re-requesting an expired URL, then verify the pin."""
    destination = archives_dir / pin["filename"]
    if destination.is_file() and sha256_file(destination) == pin["sha256"]:
        print(f"{locale}: present and verified", flush=True)
        return

    for attempt in range(1, attempts + 1):
        present = destination.stat().st_size if destination.is_file() else 0
        print(f"{locale}: {pin['filename']} ({pin['size_bytes']} bytes), "
              f"resuming from {present}", flush=True)
        url = presigned_url(pin["dataset_id"], remote=remote)
        config = tempfile.NamedTemporaryFile("w", dir=archives_dir, prefix=".curl-",
                                             suffix=".conf", delete=False)
        try:
            os.chmod(config.name, 0o600)
            config.write(f'url = "{url}"\n')
            config.close()
            result = subprocess.run(
                ["curl", "-fLsS", "-C", "-", "--retry", "2", "--retry-delay", "5",
                 "-K", config.name, "-o", str(destination)])
        finally:
            os.unlink(config.name)
        if result.returncode == 0:
            break
        if result.returncode == 33:  # the server ignored the range: restart from zero
            destination.unlink(missing_ok=True)
        print(f"{locale}: attempt {attempt} exited {result.returncode}; re-requesting",
              flush=True)
    else:
        raise RuntimeError(f"{locale}: {attempts} download attempts failed")

    if destination.stat().st_size != pin["size_bytes"]:
        raise RuntimeError(f"{locale}: {destination.stat().st_size} bytes, pinned "
                           f"{pin['size_bytes']}")
    digest = sha256_file(destination)
    if digest != pin["sha256"]:
        raise RuntimeError(f"{locale}: SHA-256 {digest}, pinned {pin['sha256']}")
    print(f"{locale}: verified {pin['size_bytes']} bytes, {digest}", flush=True)


def common_voice(args: argparse.Namespace) -> None:
    pins = archive_pins()
    requested = tuple(args.locales) if args.locales else CV_PRIMARY_LOCALES
    archives_dir = Path(args.archives_dir)
    archives_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for locale in requested:
        try:
            download_archive(locale, pins[locale], archives_dir, remote=args.remote)
        except (OSError, RuntimeError) as error:  # one locale must not hide the others' state
            failures += 1
            print(f"{locale}: FAILED {error}", flush=True)
    if failures:
        raise SystemExit(f"{failures} of {len(requested)} archives failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    freesound_parser = commands.add_parser(
        "freesound", help="stream the ranked shards and write the Freesound manifest")
    freesound_parser.add_argument("--cache-dir", type=Path, required=True,
                                  help="scratch for one shard at a time (deleted after reading)")
    freesound_parser.add_argument("--out", type=Path,
                                  default=confirm.FREESOUND_MANIFEST_PATH)
    freesound_parser.set_defaults(run=freesound)

    cv_parser = commands.add_parser(
        "common-voice", help="download and verify the primary MDC archives")
    cv_parser.add_argument("--archives-dir", type=Path, required=True)
    cv_parser.add_argument("--locales", nargs="+", default=None,
                           help=f"default: {' '.join(CV_PRIMARY_LOCALES)}")
    cv_parser.add_argument("--remote", default=MDC_REMOTE,
                           help="the host that holds the MDC key")
    cv_parser.set_defaults(run=common_voice)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

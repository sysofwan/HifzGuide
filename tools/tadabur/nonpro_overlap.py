"""A cheap overlap check for the non-professional corpora: a sample against EveryAyah (#156).

Tarteel v1's, RetaSy's and qdat's clips are app-user and amateur recordings, so they should be
independent of the professional audio the models were trained on (Tadabur's 618 named
reciters; Muaalem's 27 named moshafs, ``qud_eval/sources.json``). What a copy would look like
is a "user" clip that is in fact a professional recording, the same ayah from a published
mushaf (played into the phone, or filed in the dataset by mistake). #156 asks for a **cheap check
only**, so this is a sample, not a screen:

* **the sample**: per dataset, a few parquet row groups drawn by a salted hash of their place
  (:data:`ROW_GROUPS`), and in them up to :data:`CLIPS_PER_GROUP` decoded clips, the first by the
  same hash; only those row groups are read (HTTP range requests), never a whole shard;
* **the professionals**: every reciter folder of EveryAyah (``everyayah.com/data``, one per
  reciter name, :func:`everyayah_folders`), the same ayah, downloaded and never kept;
* **the comparison**: :mod:`tadabur.qud_fingerprint`'s fingerprint at the best offset, against
  #144's line (``qud_eval/overlap_rules.json``, fixed from #144's controls and not retuned).

Tadabur is not read: its rows of one ayah are spread over its 385 shards, and #144's screen
needed a full read of every shard (8 h) to fingerprint them; its reciters are all named
professionals (``sheikh_dict.json``), as EveryAyah's are, so EveryAyah stands in for "a published
professional recording". A clip that matches no EveryAyah recording could still be a copy of a
recording EveryAyah lacks: **no overlap detected in the sample**, not proven absence.

Usage (from ``tools/``, on a machine with network; CPU only)::

  python -m tadabur.nonpro_overlap --out tadabur/nonpro_eval/overlap.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audio import decode_to_mono_16k
from .nonpro_sources import DECODED, Dataset, audio_sha256
from .qud_fingerprint import best_match, fingerprint

EVERYAYAH = "https://everyayah.com/data"
SALT = "nonpro-overlap-2026-10"
ROW_GROUPS = {"tarteel_v1": 3, "retasy": 3, "qdat": 2}
CLIPS_PER_GROUP = 10
RULES_PATH = Path(__file__).parent / "qud_eval" / "overlap_rules.json"
#: Concurrent EveryAyah downloads.
FETCHERS = 8


def salted(*parts: object) -> str:
    return hashlib.sha256(":".join([SALT, *map(str, parts)]).encode()).hexdigest()


def everyayah_folders() -> list[str]:
    """One EveryAyah folder per reciter name (the first listed), in listing order."""
    import urllib.request

    with urllib.request.urlopen(f"{EVERYAYAH}/recitations.js", timeout=60) as response:
        listing = json.loads(response.read())
    seen, folders = set(), []
    for key, entry in listing.items():
        if key.isdigit() and entry["name"] not in seen:
            seen.add(entry["name"])
            folders.append(entry["subfolder"])
    return folders


def fetch(url: str) -> bytes | None:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def sampled_rows(dataset: Dataset, decoded: set[str]) -> list[dict]:
    """The sample's rows of one dataset, audio included (module docstring)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from .nonpro_sources import LeftOut, select

    groups = []
    fs = HfFileSystem()
    for shard in dataset.shards:
        with fs.open(f"datasets/{dataset.repo}@{dataset.revision}/{shard}") as f:
            groups += [(shard, g) for g in range(pq.ParquetFile(f).metadata.num_row_groups)]
    chosen = sorted(groups, key=lambda sg: salted(dataset.name, *sg))[:ROW_GROUPS[dataset.name]]
    rows = []
    for shard, group in sorted(chosen):
        with fs.open(f"datasets/{dataset.repo}@{dataset.revision}/{shard}",
                     block_size=32 << 20) as f:
            records = pq.ParquetFile(f).read_row_group(group).to_pylist()
        clips = [(select(dataset, r), r) for r in records]
        clips = [(c, r) for c, r in clips if not isinstance(c, LeftOut)
                 and c.audio_filename in decoded]
        clips.sort(key=lambda cr: salted(dataset.name, cr[0].audio_filename))
        rows += [{"clip": c, "audio": r["audio"]["bytes"], "shard": shard, "row_group": group}
                 for c, r in clips[:CLIPS_PER_GROUP]]
    return rows


def compare(row: dict, folders: list[str], line: float) -> dict:
    clip = row["clip"]
    mine = fingerprint(decode_to_mono_16k(row["audio"]))
    urls = [f"{EVERYAYAH}/{folder}/{clip.surah:03d}{clip.ayah:03d}.mp3" for folder in folders]
    with ThreadPoolExecutor(FETCHERS) as pool:
        audio = list(pool.map(fetch, urls))
    scores = []
    for folder, data in zip(folders, audio):
        if data is None:
            continue
        match = best_match(mine, fingerprint(decode_to_mono_16k(data)))
        if match is not None:
            scores.append((match.ber, folder))
    scores.sort()
    return {"audio_filename": clip.audio_filename, "audio_sha256": audio_sha256(row["audio"]),
            "surah_ayah": clip.surah_ayah, "shard": row["shard"], "row_group": row["row_group"],
            "recordings_compared": len(scores),
            "lowest_ber": scores[0][0] if scores else None,
            "nearest": scores[0][1] if scores else None,
            "matches": [folder for ber, folder in scores if ber <= line]}


def run(decoded: dict[str, set[str]]) -> dict:
    line = json.loads(RULES_PATH.read_text(encoding="utf-8"))["lines"]["fingerprint"]["line"]
    folders = everyayah_folders()
    out = {"everyayah_folders": len(folders), "fingerprint_line": line, "datasets": {}}
    for dataset in DECODED:
        rows = sampled_rows(dataset, decoded[dataset.name])
        clips = [compare(row, folders, line) for row in rows]
        out["datasets"][dataset.name] = {
            "clips": clips if dataset.open else [{k: c[k] for k in (
                "audio_filename", "recordings_compared", "lowest_ber", "matches")}
                for c in clips],
            "sampled": len(clips),
            "comparisons": sum(c["recordings_compared"] for c in clips),
            "lowest_ber": min((c["lowest_ber"] for c in clips if c["lowest_ber"] is not None),
                              default=None),
            "matches": sum(len(c["matches"]) for c in clips),
        }
        print(dataset.name, out["datasets"][dataset.name]["sampled"],
              out["datasets"][dataset.name]["lowest_ber"], flush=True)
    return out


def main() -> None:
    from training.external_corpus import read_decode_file
    from training.nonpro_eval import decodes_file

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decodes-dir", type=Path,
                        help="the GPU box's per-shard decodes, before they are merged")
    args = parser.parse_args()
    if args.decodes_dir:
        decoded = {d.name: {json.loads(line)["audio_filename"]
                            for path in args.decodes_dir.glob(f"{d.name}.*.base.jsonl")
                            for line in path.read_text(encoding="utf-8").splitlines()}
                   for d in DECODED}
    else:
        decoded = {d.name: {line["audio_filename"]
                            for line in read_decode_file(decodes_file(d, "base")).values()}
                   for d in DECODED}
    args.out.write_text(json.dumps(run(decoded), ensure_ascii=False, indent=1, sort_keys=True)
                        + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

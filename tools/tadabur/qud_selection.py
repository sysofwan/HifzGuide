"""Which QUD Hafs mushafs the correct-recitation eval set is built from (#144, phase 1).

#144 builds an **evaluation-only** set of truth sites from professional recitation that
neither model has heard: QUD (`QUD-Technologies/quranic-universal-ayahs`, CC BY 4.0 for the
timing data; the card says the audio "is not relicensed, and remains the property of
upstream sources/reciters"), one row per ayah with its MP3, the recited Uthmani text and
word times. This module records what the selection was made from and applies its rules.

**Sources** (``qud_eval/sources.json``, written by ``snapshot`` at pinned revisions):

* QUD's ``mushafs`` catalog (one row per mushaf: reciter, riwayah, style, recording context
  and year, channel, codec, hours) and every mushaf's parquet shards with their sizes;
* Muaalem's training data, `obadx/muaalem-annotated-v3`: its ``reciters_metadata`` (22
  reciters) and ``moshaf_metadata`` (27 moshafs with their source URLs). arXiv 2509.00094
  says "We trained on all available Moshaf datasets, reserving three Moshaf (19.0, 29.0,
  30.0) for comprehensive testing" and names no reciter; the model card of
  `obadx/muaalem-model-v3_2` lists the dataset and says nothing of a held-out split;
* Tadabur's ``sheikh_dict.json`` (Arabic name -> id), for a name screen only.

**Rules** (:func:`select`), in order; the first that applies decides:

1. not Hafs: out (only Hafs mushafs are listed);
2. a style that is not one reciter's continuous recitation (:data:`EXCLUDED_STYLES`): out;
3. the reciter is one of Muaalem's 22 (:data:`MUAALEM_RECITER_OF`, matched by hand): out,
   held-out moshaf or not. Whether ``muaalem-model-v3_2`` was trained without 19.0, 29.0 and
   30.0 is not stated anywhere the model is published, so those reciters count as seen;
4. no recording year, or one before :data:`RECENT_FROM_YEAR`: eligible, not selected;
5. otherwise: **selected**.

The year rule is a preference, not the overlap test. ``muaalem-annotated-v3`` was created on
2025-08-24 and ``muaalem-model-v3_2`` last changed on 2025-09-04; Tadabur, which ``h448``
streamed, was first published on 2025-11-28. A catalog year cannot be ordered within 2025,
so a 2025 recording counts as recent here, and phase 2 fingerprints every selected
recording against Tadabur and Muaalem's audio anyway (#144, "rule out overlap by audio").

**Tadabur name screen.** Each mushaf's reciter is looked up in ``sheikh_dict.json`` by
normalized Arabic name (:func:`normalize_name`) and, where found, the exposure registry's
uses of that Tadabur id are listed. Whether the dict's ids are Tadabur's canonical
``reciter_id`` is undocumented (it holds 618 names; Tadabur has 671 reciter ids), and a
name that is not found proves nothing, so the screen never decides; it says where the
fingerprint has to look hardest and which selected reciters ``h448`` may have heard in
other recordings. The uses are the registry's when ``select`` ran.

**Coverage** (``qud_eval/coverage.json``, written by ``coverage``): per selected mushaf and
surah, the ayat present, their total duration and the source URLs, read from the parquet
shards' ``surah``, ``ayah``, ``duration_ms`` and ``source_url`` columns only (no audio).

Usage (from ``tools/``; torch-free; ``snapshot`` and ``coverage`` read the Hub)::

  python -m tadabur.qud_selection snapshot   # a few seconds
  python -m tadabur.qud_selection select
  python -m tadabur.qud_selection coverage   # ~10 min: column reads of the selected shards
  python -m tadabur.qud_selection date-check # titles and upload dates of sampled videos
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .dataset_source import DATASET_ID as TADABUR_DATASET

QUD_DATASET = "QUD-Technologies/quranic-universal-ayahs"
MUAALEM_DATASET = "obadx/muaalem-annotated-v3"
MUAALEM_MODEL = "obadx/muaalem-model-v3_2"
MUAALEM_PAPER = "https://arxiv.org/abs/2509.00094"

QUD_DIR = Path(__file__).parent / "qud_eval"
SOURCES_PATH = QUD_DIR / "sources.json"
SELECTION_PATH = QUD_DIR / "selection.json"
COVERAGE_PATH = QUD_DIR / "coverage.json"
DATE_CHECK_PATH = QUD_DIR / "date_check.json"

HAFS = "Hafs A'n Assem"
SELECTED = "selected"
ELIGIBLE = "eligible_not_selected"
EXCLUDED = "excluded"

#: Styles that are not one reciter's continuous recitation.
EXCLUDED_STYLES = {
    "Kids Repeat": "children repeat each phrase after the reciter: two voices, and the "
                   "children's are not a professional recitation",
}

#: The first recording year counted as recent (see the module docstring).
RECENT_FROM_YEAR = 2025

#: The moshafs arXiv 2509.00094 reserves for testing.
MUAALEM_HELD_OUT_MOSHAFS = ("19.0", "29.0", "30.0")

#: QUD ``reciter_id`` -> ``muaalem-annotated-v3`` reciter id, matched by hand on both
#: datasets' Arabic and English names. Where the normalized Arabic names differ, the
#: difference is noted; every other pair matches exactly under :func:`normalize_name`.
MUAALEM_RECITER_OF = {
    "mahmoud_khalil_al_husary": 0,
    "mohammed_siddiq_al_minshawi": 1,
    "abdulbasit_abdulsamad": 2,
    "mahmoud_ali_al_banna": 3,
    "mustafa_ismail": 4,
    "ahmed_amer": 5,  # QUD أحمد عامر, Muaalem أحمد محمد عامر (both Egypt)
    "ayman_swed": 6,  # QUD أيمن سويد, Muaalem أيمن رشدي سويد (both Syria)
    "ahmed_issa_al_maasaraawi": 7,  # QUD المعصراوي, Muaalem's spelling المعصرواوي
    "ali_al_huthaifi": 8,  # QUD علي بن عبد الرحمن الحذيفي, Muaalem علي الحذيفي
    "mahmoud_abdul_hakam": 9,
    "abdur_rashid_sufi": 11,
    "mohammed_ayyub": 12,
    "ibrahim_al_akhdar": 13,
    "mishary_rashid_al_afasy": 19,  # QUD مشاري بن راشد العفاسي, Muaalem's spelling الفاسي
    "ahmed_shaheen": 22,
    "khalifa_al_tunaiji": 24,
    "saber_abdulhakam": 26,
    "ahmed_nuayna": 29,
    "ali_hajjaj_al_souasi": 30,
}
#: Muaalem reciters with no mushaf in QUD (checked by name against the whole catalog).
MUAALEM_RECITERS_NOT_IN_QUD = (25, 27, 28)

_DIACRITICS = re.compile("[ً-ٰٟـ]")
_LETTER_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})


def normalize_name(name: str) -> str:
    """An Arabic name with harakat, tatweel, alif and ya variants, taa marbuta and spaces
    folded, so ``عبد الرحمن`` and ``عبدالرحمن`` compare equal."""
    return _DIACRITICS.sub("", name).translate(_LETTER_VARIANTS).replace(" ", "")


# --- the rules (pure) ------------------------------------------------------------------


def _muaalem_entry(reciter_id: int, muaalem: Mapping) -> dict:
    reciter = next(r for r in muaalem["reciters"] if r["id"] == reciter_id)
    moshafs = sorted(m["id"] for m in muaalem["moshafs"] if m["reciter_id"] == reciter_id)
    return {
        "reciter_id": reciter_id,
        "english_name": reciter["english_name"],
        "arabic_name": reciter["arabic_name"],
        "moshafs": moshafs,
        "moshafs_held_out_by_paper": [m for m in moshafs if m in MUAALEM_HELD_OUT_MOSHAFS],
    }


def tadabur_screen(name_ar: str, sheikh_ids: Mapping[str, int],
                   uses_of: Mapping[int, Mapping[str, int]]) -> dict | None:
    """The Tadabur id whose ``sheikh_dict`` name equals ``name_ar`` once normalized, and
    the exposure uses of that id with their row counts (``uses_of``); ``None`` when no name
    matches, which is no evidence of absence."""
    by_name = {normalize_name(name): tid for name, tid in sheikh_ids.items()}
    tid = by_name.get(normalize_name(name_ar))
    if tid is None:
        return None
    return {"tadabur_id": tid, "uses": dict(sorted(uses_of.get(tid, {}).items()))}


def decide(row: Mapping) -> tuple[str, str]:
    """``(decision, reason)`` for one Hafs catalog row, by the module's rules 2-5."""
    if row["style"] in EXCLUDED_STYLES:
        return EXCLUDED, f"style {row['style']}: {EXCLUDED_STYLES[row['style']]}"
    if row["reciter_id"] in MUAALEM_RECITER_OF:
        return EXCLUDED, "the reciter is in Muaalem's training data (muaalem-annotated-v3)"
    year = row["recording_year"]
    if year is None:
        return ELIGIBLE, "no recording year: it cannot be placed after the models' training"
    if year < RECENT_FROM_YEAR:
        return ELIGIBLE, f"recorded in {year}, before {RECENT_FROM_YEAR}"
    return SELECTED, (f"Hafs, reciter not in Muaalem's training data, recorded in {year} "
                      f"(>= {RECENT_FROM_YEAR})")


def select(sources: Mapping, uses_of: Mapping[int, Mapping[str, int]]) -> dict:
    """Every Hafs mushaf of the snapshot with its provenance, the Muaalem and Tadabur
    screens and the decision, sorted by slug; and the totals per decision."""
    qud, muaalem = sources["qud"], sources["muaalem"]
    mushafs = []
    for row in sorted(qud["catalog"], key=lambda r: r["slug"]):
        if row["riwayah"] != HAFS:
            continue
        decision, reason = decide(row)
        shards = qud["shards"][row["slug"]]
        muaalem_id = MUAALEM_RECITER_OF.get(row["reciter_id"])
        mushafs.append({
            **{key: row[key] for key in sorted(row)},
            "decision": decision,
            "reason": reason,
            "muaalem": None if muaalem_id is None else _muaalem_entry(muaalem_id, muaalem),
            "tadabur_name_screen": tadabur_screen(row["name_ar"],
                                                  sources["tadabur"]["sheikh_dict"], uses_of),
            "shards": sorted(shards),
            "download_bytes": sum(shards.values()),
        })
    totals: dict[str, dict] = {}
    for decision in (SELECTED, ELIGIBLE, EXCLUDED):
        group = [m for m in mushafs if m["decision"] == decision]
        totals[decision] = {
            "mushafs": len(group),
            "reciters": len({m["reciter_id"] for m in group}),
            "hours": round(sum(m["total_duration_hours"] for m in group), 1),
            "download_bytes": sum(m["download_bytes"] for m in group),
        }
    return {
        "qud_revision": qud["revision"],
        "rules": {
            "riwayah": HAFS,
            "excluded_styles": EXCLUDED_STYLES,
            "recent_from_year": RECENT_FROM_YEAR,
            "muaalem_held_out_moshafs": list(MUAALEM_HELD_OUT_MOSHAFS),
            "muaalem_reciters_not_in_qud": list(MUAALEM_RECITERS_NOT_IN_QUD),
        },
        "non_hafs_mushafs": sum(r["riwayah"] != HAFS for r in qud["catalog"]),
        "totals": totals,
        "mushafs": mushafs,
    }


def tadabur_uses_of(registry) -> dict[int, dict[str, int]]:
    """Per Tadabur reciter id, each exposure use that touched it and how many of its rows:
    the shard uses' row counts, and the recording uses' recordings."""
    uses: dict[int, dict[str, int]] = defaultdict(dict)
    for use, exposure in registry.shard_uses.items():
        for reciter, rows in exposure.rows_per_reciter.items():
            uses[reciter][use] = rows
    for use, rows in registry.recordings.items():
        for row in rows:
            uses[row.reciter_id][use] = uses[row.reciter_id].get(use, 0) + 1
    return dict(uses)


# --- coverage (pure over rows) ---------------------------------------------------------


def ranges(numbers: Iterable[int]) -> str:
    """``"1-7,9"`` for ``[1, 2, ..., 7, 9]``: a sorted set of ayah numbers, compactly."""
    out: list[str] = []
    values = sorted(set(numbers))
    start = prev = None
    for n in values + [None]:
        if start is not None and n == prev + 1:
            prev = n
            continue
        if start is not None:
            out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    return ",".join(out)


def summarize_coverage(rows: Iterable[Mapping]) -> dict:
    """One mushaf's coverage from its rows' ``surah``, ``ayah``, ``duration_ms`` and
    ``source_url``: per surah the ayat present, their rows (an ayah may appear twice),
    duration and source URLs; and the totals."""
    per_surah: dict[int, dict] = defaultdict(lambda: {"ayat": [], "duration_ms": 0,
                                                      "sources": set()})
    for row in rows:
        entry = per_surah[row["surah"]]
        entry["ayat"].append(row["ayah"])
        entry["duration_ms"] += row["duration_ms"]
        entry["sources"].add(row["source_url"])
    surahs = {
        str(surah): {"ayat": ranges(e["ayat"]), "rows": len(e["ayat"]),
                     "duration_ms": e["duration_ms"], "sources": sorted(e["sources"])}
        for surah, e in sorted(per_surah.items())
    }
    return {
        "rows": sum(s["rows"] for s in surahs.values()),
        "ayat": sum(len(set(per_surah[int(k)]["ayat"])) for k in surahs),
        "duration_ms": sum(s["duration_ms"] for s in surahs.values()),
        "surahs": surahs,
    }


# --- the Hub (network) -----------------------------------------------------------------

COVERAGE_COLUMNS = ["surah", "ayah", "duration_ms", "source_url"]
#: A mushaf config's shard: ``<riwayah dir>/<slug>-00000-of-00004.parquet``.
_SHARD_NAME = re.compile(r"(.+)-\d{5}-of-\d{5}\.parquet")


def _parquet_table(repo: str, revision: str, path: str, columns: list[str] | None = None):
    """A parquet file of a Hub dataset at a pinned revision, reading only ``columns``
    (the footer and those column chunks, by HTTP range requests)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{repo}@{revision}/{path}", block_size=1 << 16) as f:
        return pq.ParquetFile(f).read(columns=columns)


def snapshot() -> dict:
    """Everything :func:`select` reads, fetched at each dataset's current revision."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    qud = api.dataset_info(QUD_DATASET)
    files = [f for f in api.list_repo_tree(QUD_DATASET, recursive=True, revision=qud.sha,
                                           repo_type="dataset")
             if f.path.endswith(".parquet") and not f.path.startswith("mushafs/")]
    shards: dict[str, dict[str, int]] = defaultdict(dict)
    for f in files:
        shards[_SHARD_NAME.fullmatch(f.path.split("/")[-1]).group(1)][f.path] = f.size
    catalog = _parquet_table(QUD_DATASET, qud.sha, "mushafs/all-00000-of-00001.parquet")
    muaalem = api.dataset_info(MUAALEM_DATASET)
    reciters = _parquet_table(MUAALEM_DATASET, muaalem.sha,
                              "reciters_metadata/train-00000-of-00001.parquet")
    moshafs = _parquet_table(MUAALEM_DATASET, muaalem.sha,
                             "moshaf_metadata/train-00000-of-00001.parquet",
                             ["id", "reciter_id", "name", "publisher", "sources",
                              "recitation_speed"])
    model = api.model_info(MUAALEM_MODEL)
    tadabur = api.dataset_info(TADABUR_DATASET)
    sheikh_path = hf_hub_download(TADABUR_DATASET, "sheikh_dict.json", repo_type="dataset",
                                  revision=tadabur.sha)
    sheikh = json.loads(Path(sheikh_path).read_text(encoding="utf-8-sig"))
    return {
        "qud": {"dataset": QUD_DATASET, "revision": qud.sha,
                "last_modified": qud.last_modified.isoformat(),
                "catalog": sorted(catalog.to_pylist(), key=lambda r: r["slug"]),
                "shards": {slug: dict(sorted(s.items())) for slug, s in sorted(shards.items())}},
        "muaalem": {"dataset": MUAALEM_DATASET, "revision": muaalem.sha,
                    "created_at": muaalem.created_at.isoformat(),
                    "last_modified": muaalem.last_modified.isoformat(),
                    "model": MUAALEM_MODEL, "model_revision": model.sha,
                    "model_last_modified": model.last_modified.isoformat(),
                    "paper": MUAALEM_PAPER,
                    "reciters": sorted(reciters.to_pylist(), key=lambda r: r["id"]),
                    "moshafs": sorted(moshafs.to_pylist(), key=lambda r: float(r["id"]))},
        "tadabur": {"dataset": TADABUR_DATASET, "revision": tadabur.sha,
                    "created_at": tadabur.created_at.isoformat(),
                    "last_modified": tadabur.last_modified.isoformat(),
                    "sheikh_dict": dict(sorted(sheikh.items(), key=lambda kv: kv[1]))},
    }


def coverage(selection: Mapping) -> dict:
    """:func:`summarize_coverage` of every selected mushaf, read column-only from its
    shards at the selection's QUD revision."""
    out = {}
    for mushaf in selection["mushafs"]:
        if mushaf["decision"] != SELECTED:
            continue
        rows: list[dict] = []
        for shard in mushaf["shards"]:
            rows += _parquet_table(QUD_DATASET, selection["qud_revision"], shard,
                                   COVERAGE_COLUMNS).to_pylist()
        out[mushaf["slug"]] = summarize_coverage(rows)
        print(f"{mushaf['slug']}: {out[mushaf['slug']]['rows']} rows", flush=True)
    return {"qud_revision": selection["qud_revision"], "columns": COVERAGE_COLUMNS,
            "mushafs": out}


#: Source videos sampled per selected mushaf by ``date-check``.
DATE_CHECK_SAMPLES = 6
_YOUTUBE_FIELDS = {
    "title": re.compile(r"<title>([^<]*?)(?: - YouTube)?</title>"),
    "upload_date": re.compile(r'"uploadDate":"([^"]+)"'),
}


def date_check_sample(sources: list[str], k: int = DATE_CHECK_SAMPLES) -> list[str]:
    """``k`` of a mushaf's sorted source URLs, evenly spaced from the first."""
    k = min(k, len(sources))
    return [sources[i * len(sources) // k] for i in range(k)]


def youtube_fields(html: str) -> dict[str, str | None]:
    """A YouTube watch page's title and upload date (``None`` where the page lacks one)."""
    found = {name: pattern.search(html) for name, pattern in _YOUTUBE_FIELDS.items()}
    return {name: m.group(1) if m else None for name, m in found.items()}


def date_check(coverage_summary: Mapping) -> dict:
    """Per selected mushaf, the title and upload date of :data:`DATE_CHECK_SAMPLES` of its
    source videos: evidence for the catalog's recording year. An upload date bounds the
    recording from above only; a title naming the Ramadan (1446 AH = March 2025, 1447 AH =
    February-March 2026) bounds it from below."""
    import urllib.request

    out = {}
    for slug, mushaf in sorted(coverage_summary["mushafs"].items()):
        urls = sorted({u for surah in mushaf["surahs"].values() for u in surah["sources"]})
        checked = {}
        for url in date_check_sample(urls):
            request = urllib.request.Request(f"https://{url}", headers={
                "User-Agent": "Mozilla/5.0", "Accept-Language": "en"})
            with urllib.request.urlopen(request, timeout=60) as response:
                checked[url] = youtube_fields(response.read().decode("utf-8", "replace"))
        out[slug] = checked
    return {"samples_per_mushaf": DATE_CHECK_SAMPLES, "mushafs": out}


def _dump(path: Path, payload: Mapping) -> None:
    write_text_atomically(path, json.dumps(payload, ensure_ascii=False, indent=1,
                                           sort_keys=True) + "\n")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("snapshot", "select", "coverage", "date-check"))
    args = parser.parse_args()
    if args.command == "snapshot":
        _dump(SOURCES_PATH, snapshot())
    elif args.command == "select":
        from .exposure import load_registry

        selection = select(_load(SOURCES_PATH), tadabur_uses_of(load_registry()))
        _dump(SELECTION_PATH, selection)
        for decision, total in selection["totals"].items():
            print(f"{decision:22s} {total}")
    elif args.command == "coverage":
        _dump(COVERAGE_PATH, coverage(_load(SELECTION_PATH)))
    else:
        _dump(DATE_CHECK_PATH, date_check(_load(COVERAGE_PATH)))


if __name__ == "__main__":
    main()

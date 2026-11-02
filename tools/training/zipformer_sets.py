"""The #245 screen's report-only sets, and the experts' qdat_bench read.

Three sets are not verdict sets: QUD's unread half and Tarteel v1's confirm half sit in the
sets table's "report only" row because zipformer probably trained on them, and qdat_bench is the
catch-side corpus the card's own transcript covers. Their outcome is the truth-site one
(:mod:`training.truth_scorer`): does the decode *commit* the carrier the experts heard, on the
correct and the mistake sites separately -- commit rate, committed accuracy, decode-level flags,
missed mistakes and silent corrections, with the clustered bootstrap. Nothing here gates the
screen's verdict.

* **One arm per model, and it is the same kind of arm for all four.** The corpora's committed
  decodes carry the recorded check sequence (``checks``), so each model's text is that
  sequence's committed transcript (:func:`training.decoding.committed_text`), ours from its own
  per-check decodes and zipformer from its single stream. The streams differ in what the app
  would see (zipformer's checks are slices of one stream; ours are independent looks), and that
  asymmetry is stated beside the numbers rather than hidden by a mixed arm.
* **QUD's unread half**: every segment of the pilot's two mushafs that #95 did not read
  (:func:`training.consonant_readout.load_qud_complement`), scored against the corpus's truth
  sites. The round-2 candidate has no committed QUD decode, so its arm is named as missing --
  never dropped silently, and never replaced by another model's.
* **Tarteel v1's confirm half only**: the look half is decoded for three of the four arms (round
  2 read the confirm half once, by design), so scoring the look half would change the population
  from model to model.
* **qdat_bench**: the 159 clips are qdat's own audio, so their decodes are qdat's; what the
  corpus adds is the experts' ``phonetic_transcript``, which lives on the gated Hub and is
  *read, never committed*. :func:`write_qdat_bench` computes the agreement there (on the box,
  beside the token) and writes a small committed summary this module and the readout read.

Usage (from ``tools/``)::

    python -m training.zipformer_sets qdat_bench --out \\
        tadabur/zipformer_screen/qdat_bench.json       # on cuda-dev, with the Hub token
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from training.zipformer_decode import ZIPFORMER_DIR
from training.zipformer_readout import C95, CANDIDATE, TEACHER, read_rows

#: The report-only corpora's committed decodes, per model and arm store.
QUD_DIR = Path(__file__).resolve().parent.parent / "tadabur"
QDAT_BENCH_PATH = ZIPFORMER_DIR / "qdat_bench.json"
#: The arm label zipformer's report-only reads use: the store its checks came from.
ZIPFORMER_CHECKS = "zipformer-checks"


def _item(row: Mapping) -> str:
    """A decode line's truth-site item key: clip and span, as ``truth_scorer.item_key``."""
    return f"{row['audio_filename']}@{row['start_sample']}:{row['end_sample']}"


def _committed(lines: Sequence[Mapping]) -> dict[str, str]:
    """Each item's committed transcript: every check's hop, the stop's included."""
    from training.decoding import committed_text

    return {_item(row): committed_text(row["checks"]) for row in lines}


def _clusters(sites: Sequence, mushaf: bool) -> dict[str, int]:
    """Each clip's bootstrap cluster: the mushaf and surah, or Tarteel v1's surah alone.

    QUD audio is two mushafs' segments, so the surah alone would merge two reciters; Tarteel
    v1's clips are app sessions and cluster by surah as its readouts do.
    """
    names: dict[str, int] = {}
    out: dict[str, int] = {}
    for site in sites:
        surah = str(site.surah_ayah).split(":")[0]
        name = f"{str(site.stratum).split(':')[1]}:{surah}" if mushaf else surah
        out[site.audio_filename] = names.setdefault(name, len(names))
    return out


def truth_site_view(sites: Sequence, decodes: Mapping[str, Mapping[str, str]],
                    clusters: Mapping[str, int], comparisons: Sequence[tuple[str, str]],
                    design_weights: Mapping[str, float] | None = None) -> dict:
    """The pooled truth-site cells, one per side, for every arm in ``decodes``.

    ``clusters`` maps each site's ``audio_filename`` to its bootstrap cluster id; the site
    design weights (:func:`training.truth_scorer.site_weights`) hold each site's stratum share,
    as the truth-baseline readout reads them.
    """
    from tadabur.exclusions import NONE
    from training.site_outcomes import CORRECT_SIDE
    from training.truth_scorer import ALL, score

    report = score(sites, clusters, decodes, comparisons, None, design_weights, NONE)
    cells = {}
    for cell in report["cells"]:
        if cell["family"] != ALL or cell["heard"] != ALL:
            continue
        if not cell.get("arms"):
            continue
        cells[cell["side"]] = {
            "sites": cell["sites"], "reciters": cell["reciters"],
            "arms": {arm: row["rates"] for arm, row in cell["arms"].items()},
            "differences": cell["differences"],
        }
    return {
        "sites": len(sites),
        "correct_sites": cells.get(CORRECT_SIDE, {}).get("sites"),
        "cells": cells,
        "bootstrap": report["bootstrap"],
    }


def _weights(sites: Sequence) -> dict[str, float]:
    from training.truth_scorer import site_weights

    return site_weights(sites)


def _scored_set(name: str, sites: Sequence, decodes: Mapping[str, Mapping[str, str]],
                clusters: Mapping[str, int], missing: Sequence[str], rule: str,
                arm_note: str) -> dict:
    from collections import Counter

    from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE, side

    present = {arm: rows for arm, rows in decodes.items() if rows is not None}
    comparisons = [(arm, TEACHER) for arm in present if arm != TEACHER]
    view = truth_site_view(sites, present, clusters, comparisons, _weights(sites))
    sides = view["cells"]
    counts = Counter(side(site) for site in sites)
    return {
        "rule": rule, "arm_note": arm_note, "set": name,
        "sites": view["sites"], "clips": len(clusters),
        "sides": {CORRECT_SIDE: counts.get(CORRECT_SIDE, 0),
                  MISTAKE_SIDE: counts.get(MISTAKE_SIDE, 0),
                  "no_side": counts.get(None, 0)},
        "arms": sorted(present), "missing_arms": sorted(missing),
        "correct_side": sides.get(CORRECT_SIDE), "mistake_side": sides.get(MISTAKE_SIDE),
        "bootstrap": view["bootstrap"],
    }


# --- QUD's unread half -----------------------------------------------------------------------

def qud_other() -> dict:
    """Every truth site of QUD's unread half, scored for the arms that have a committed decode.

    The half is the frozen complement's own items (:func:`load_qud_complement`, whose keys are
    ``<clip>#<segment>``; a site's id carries that key), and one site is one segment, so a kept
    segment never leaks into this population.
    """
    from tadabur.qud_overlap import PILOT
    from training.consonant_readout import load_qud_complement
    from tadabur.truth_sites import load_truth_sites
    from training.qud_pilot import decodes_file as pilot_decodes

    complement = load_qud_complement()
    pilot_sites = QUD_DIR / "qud_eval" / "pilot" / "truth_sites"
    sites = [site for slug in PILOT
             for site in load_truth_sites(pilot_sites / f"{slug}.jsonl.gz")
             if site.site_id.split(":")[1] in complement]
    decodes = {
        TEACHER: _committed([row for slug in PILOT
                             for row in read_rows(pilot_decodes(slug, "base"))]),
        C95: _committed(read_rows(QUD_DIR / "consonant_readout" / "decodes"
                                  / "qud_other.c95.jsonl.gz")),
        CANDIDATE: None,
        ZIPFORMER_CHECKS: _committed(read_rows(ZIPFORMER_DIR / "decodes"
                                               / "qud-other.zipformer.jsonl.gz")),
    }
    return _scored_set(
        "qud_other", sites, decodes, _clusters(sites, mushaf=True), [CANDIDATE],
        "QUD's unread half, the frozen complement's segments of the pilot's two mushafs, "
        "scored against the corpus's truth sites (report only: zipformer probably trained on "
        "it). The round-2 candidate has no committed QUD decode, so its arm is named missing",
        "each arm's committed transcript from its recorded check sequence; zipformer's is the "
        "single stream's checks, ours the app's own per-check decodes")


# --- Tarteel v1's confirm half ---------------------------------------------------------------

def tarteel_confirm() -> dict:
    """Every truth site of Tarteel v1's confirm half, for all four arms."""
    from tadabur.nonpro_sources import DATASETS
    from tadabur.tarteel_split import load_split
    from tadabur.truth_sites import load_truth_sites
    from training.nonpro_eval import decodes_file, part_dir

    confirm = set(load_split()["confirm"])
    sites = [site for site in
             load_truth_sites(part_dir(DATASETS["tarteel_v1"]) / "truth_sites.jsonl.gz")
             if site.audio_filename in confirm]
    decodes = {
        TEACHER: _committed([row for row in
                             read_rows(decodes_file(DATASETS["tarteel_v1"], "base"))
                             if row["audio_filename"] in confirm]),
        C95: _committed(read_rows(QUD_DIR / "aug_readout" / "decodes"
                                  / "tarteel_v1.confirm.candidate_ema.jsonl.gz")),
        CANDIDATE: _committed(read_rows(QUD_DIR / "consonant2_readout" / "decodes"
                                        / "tarteel_v1.confirm.r2_candidate_ema.jsonl.gz")),
        ZIPFORMER_CHECKS: _committed(read_rows(ZIPFORMER_DIR / "decodes"
                                               / "tarteel.zipformer.jsonl.gz")),
    }
    return _scored_set(
        "tarteel_v1_confirm", sites, decodes, _clusters(sites, mushaf=False), [],
        "Tarteel v1's confirm half (the look half is decoded for three of the four arms), "
        "scored against the corpus's truth sites (report only: zipformer probably trained on "
        "it)",
        "each arm's committed transcript from its recorded check sequence; zipformer's is the "
        "single stream's checks, ours the app's own per-check decodes")


# --- qdat_bench, the experts' transcript -----------------------------------------------------

#: The qdat decodes the expert read uses: each model's committed text on qdat's own clips (the
#: bench clips are qdat's audio, so no bench-specific decode exists or is needed).
QDAT_ARMS = {
    TEACHER: "nonpro_eval/qdat/decodes.base.jsonl.gz",
    C95: "aug_readout/decodes/qdat.candidate_ema.jsonl.gz",
    CANDIDATE: "consonant2_readout/decodes/qdat.r2_candidate_ema.jsonl.gz",
    ZIPFORMER_CHECKS: "zipformer_screen/decodes/qdat.zipformer.jsonl.gz",
}


def write_qdat_bench(out: Path) -> dict:
    """The experts' per-class agreement and departure words, over qdat_bench's own clips.

    Runs where the Hub token is (cuda-dev): the experts' ``phonetic_transcript`` is read and
    never committed, so only the summary is. Each model's text is its qdat decode's committed
    transcript on the bench clip's own audio; the expert transcript is also compared with the
    mushaf, and where it leaves the mushaf is recorded word by word, which the corpus's own
    readout reports descriptively.
    """
    from tadabur.nonpro_sources import DATASETS
    from tadabur.waqf_segments import hafs_segment_reference
    from training.nonpro_eval import (AGREE, LETTER_KINDS, NONE, OTHER, bench_departures,
                                      bench_transcripts, build_items, decodes_file,
                                      letter_outcomes)

    transcripts = bench_transcripts()
    # only the bench clips' items: the corpus's other datasets (RetaSy's restricted decodes
    # among them) are not needed to read this corpus, and their absence is not an error here
    dataset = DATASETS["qdat"]
    lines = {row["key"]: row for row in read_rows(decodes_file(dataset, "base"))}
    selected = [line for _, line in sorted(lines.items())
                if line["audio_filename"] in transcripts]
    items, unsupported = build_items(dataset, selected, hafs_segment_reference())
    if unsupported:
        raise SystemExit(f"qdat_bench: {len(unsupported)} clip(s) have no phonetizer "
                         f"reference, e.g. {unsupported[:3]}")
    texts: dict[str, dict[str, str]] = {}
    for arm, relative in QDAT_ARMS.items():
        lines = read_rows(QUD_DIR / relative)
        committed = _committed(lines)
        texts[arm] = {row["audio_filename"]: committed[_item(row)] for row in lines}
    missing = [item.audio_filename for item in items
               if any(item.audio_filename not in rows for rows in texts.values())]
    if missing:
        raise SystemExit(f"qdat_bench: {len(missing)} clip(s) have no decode for some arm, "
                         f"e.g. {missing[:3]}")
    agreement: dict[str, dict] = {}
    for arm, rows in texts.items():
        agreement[arm] = {}
        for kind in LETTER_KINDS:
            agree = other = none = 0
            for item in items:
                transcript = transcripts[item.audio_filename]
                for _, outcome in letter_outcomes(rows[item.audio_filename],
                                                  transcript)[kind]:
                    agree += outcome == AGREE
                    other += outcome == OTHER
                    none += outcome == NONE
            total = agree + other + none
            committed = agree + other
            agreement[arm][kind] = {
                "agree": agree, "other": other, "none": none,
                "agreement": round(agree / total, 4) if total else None,
                "agreement_committed": round(agree / committed, 4) if committed else None,
            }
    departures = {item.audio_filename:
                  sorted(bench_departures(item, transcripts[item.audio_filename]))
                  for item in items}
    record = {
        "issue": 245,
        "set": "qdat_bench",
        "rule": "the experts' phonetic_transcript, read on the Hub and never committed: each "
                "model's committed text on the clip's own audio, letter by letter over the "
                "transcript's slots, per class; and where the transcript leaves the mushaf "
                "(the departure words), which no model is scored on here",
        "clips": len(items),
        "arms": {arm: {"source": f"tadabur/{relative}", "protocol": "checks"}
                 for arm, relative in QDAT_ARMS.items()},
        "agreement": agreement,
        "departures": {
            "clips": sum(bool(words) for words in departures.values()),
            "words": sum(len(words) for words in departures.values()),
            "per_clip": {clip: words for clip, words in departures.items() if words},
        },
    }
    Path(out).write_text(
        json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"wrote {out}: {len(items)} clips, "
          f"{record['departures']['words']} departure words", flush=True)
    return record


def qdat_bench() -> dict:
    """The committed experts' summary."""
    return json.loads(QDAT_BENCH_PATH.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("qdat_bench",))
    parser.add_argument("--out", type=Path, default=QDAT_BENCH_PATH)
    args = parser.parse_args(argv)
    write_qdat_bench(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

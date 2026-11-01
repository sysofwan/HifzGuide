"""Score #245's zipformer screen from its committed decodes, and write `report.json` + the doc.

The screen compares the Quran-Lab zipformer (v3.2 fp32, chunk 24) against the base teacher, the
round-2 candidate and C95 on sets our models never trained on. Every metric reads the arm
:func:`training.zipformer_decode.metric_arm` names for its model — app outcomes from
``checks-w1`` (zipformer) and our recorded ``checks``, whole items from ``spans`` /
``stream_b0`` — and the report carries that map.

* **Correct side.** qdat's all-cause false flags per 1,000 graded words (replayed through
  Muraja's own engine) and the benchmark's per-class phoneme error against the ayah reference,
  per source; tlog_holdout is the verdict source.
* **Catch side.** The synthetic edits' heard rate and decoys-left-unchanged, and Ship A's
  mistake sites flagged through the engine.
* **Match rule** (owner, 2026-10-31): the paired 95% interval of single-stream ``checks`` minus
  ``checks-w1``, on qdat false flags and Ship A catches. Zero inside it means one stream suffices.
* **Explanation, never a gate.** Look diversity (rescue rate, ratchet-hidden catches, ratchet
  gain against the single-decode approximation) and the compute table.
* **Report only.** The ratchet-free lines: whole-item per-class error on qdat, and whole-item
  Ship A outcomes, beside the verdict they are read next to.
* **Step 6's carrier lines.** The empty-slot rate by the mushaf's prescription (haraka versus
  sukun), the commit rate and committed accuracy separately, and teacher agreement for our
  students only; the correct and catch sides are never pooled.

The verdict is the pre-registered rule (issue #245, unchanged, ``checks-w1`` the verdict arm):
**GO** when zipformer closes at least half the candidate−teacher gap on qdat and tlog_holdout
and its catch side is not worse than the candidate's (edits heard within 5 points with decoys
≥ 95%, Ship A mistake sites flagged within 5 points); **PRIOR** when the correct side passes and
the catch side does not; **STOP** when the correct side fails.

Usage (from ``tools/``, torch-free; the decodes are committed and the Muraja harness is built)::

    python -m training.zipformer_readout report    # tools/tadabur/zipformer_screen/report.json
    python -m training.zipformer_readout doc       # docs/zipformer-comparison.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from tadabur.tlog_index import write_json
from training.zipformer_decode import METRICS, ZIPFORMER_DIR, metric_arm

DECODES = ZIPFORMER_DIR / "decodes"
REPORT_PATH = ZIPFORMER_DIR / "report.json"
PROBE_PATH = ZIPFORMER_DIR / "probe.json"
DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "zipformer-comparison.md"

#: The four models this screen reports, and the three it compares against zipformer.
MODELS = ("base", "zipformer", "r2_candidate_ema", "c95")
STUDENTS = ("base", "r2_candidate_ema", "c95")
TEACHER, CANDIDATE = "base", "r2_candidate_ema"
#: The arm each model's whole-item text is read from (`metric_arm`'s answer); the committed
#: decodes carry it.
WHOLE_ITEM = {"base": "stream_b0", "c95": "stream_b0", "r2_candidate_ema": "stream_b0",
              "zipformer": "spans"}
#: The zipformer check arms, which live in a second store.
ZIP_CHECKS, ZIP_STREAM, ZIP_WINDOWS = "checks-w1", "checks", "checks-w1-windows"
#: The ``decodes[model][key]`` arm name the shared word-grading loop reads (it asks for
#: ``"checks"`` whatever the model, so a zipformer line is handed over under that key).
SHARED_CHECKS = "checks"

PER_1000, PERCENT = 1000.0, 100.0
#: The pre-registered thresholds (issue #245): half the gap; five points on the catch side;
#: decoys at 95%.
GAP_CLOSED = 0.5
CATCH_POINTS = 5.0
DECOY_FLOOR = 95.0
#: The probe's horizon (architect r2, AC 6), the card's chunk length, and the sample rate.
HORIZON_HOURS = 12.0
CHUNK_SECONDS = 0.48
SAMPLE_RATE = 16000

_MARKS = re.compile(r"[\u064B-\u0652\u0653-\u0655\u0670\u06D6-\u06ED\u08F0-\u08FF\u0640]")


# --- reading the committed decodes ---------------------------------------------------------


def read_rows(path: Path) -> list[dict]:
    """A committed record's rows (``.jsonl`` or ``.jsonl.gz``), in file order."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_lines(path: Path) -> dict[str, dict]:
    """A committed decode file by item key."""
    return {row["key"]: row for row in read_rows(path)}


def zipformer_lines(set_name: str, store: str = "") -> dict[str, dict]:
    suffix = f".{store}" if store else ""
    return read_lines(DECODES / f"{set_name}.zipformer{suffix}.jsonl.gz")


def our_lines(set_name: str, model: str) -> dict[str, dict]:
    return read_lines(DECODES / f"{set_name}.{model}.jsonl.gz")


def whole_item_texts(set_name: str) -> dict[str, dict[str, str]]:
    """Each model's whole-item text by item key, from the arm its metric reads."""
    out = {"zipformer": {key: line["spans"] for key, line in zipformer_lines(set_name).items()}}
    for model in STUDENTS:
        out[model] = {key: line[WHOLE_ITEM[model]]
                      for key, line in our_lines(set_name, model).items()}
    return out


# --- the generic rate view ----------------------------------------------------------------


def rate_view(observations: Mapping[str, Sequence[tuple[object, int, int]]],
              comparisons: Sequence[tuple[str, str]], scale: float) -> dict:
    """Each arm's rate and interval, and each paired difference, on ``scale``; one observation
    per cluster member as ``(cluster, numerator, denominator)``, the same clusters every arm."""
    from training.aug_readout_metrics import Terms, compare

    clusters = None
    by_model = {}
    for arm, rows in observations.items():
        if clusters is None:
            clusters = [row[0] for row in rows]
        elif [row[0] for row in rows] != clusters:
            raise ValueError(f"{arm}: not the same clusters as the other arms")
        by_model[arm] = (np.array([row[1] for row in rows], dtype=float),
                         np.array([row[2] for row in rows], dtype=float))
    if not clusters:
        raise ValueError("no observations to rate")
    index = {cluster: position for position, cluster in
             enumerate(sorted(set(clusters), key=str))}
    return compare(Terms(tuple(index[cluster] for cluster in clusters), by_model),
                   scale, comparisons)


def paired(view: Mapping, a: str, b: str) -> dict | None:
    """``a − b`` from a :func:`rate_view` view."""
    from training.aug_readout_metrics import comparison_name

    return view.get("comparisons", {}).get(comparison_name(a, b))


# --- the benchmark's per-class phoneme error ----------------------------------------------


def _bare(text: str) -> str:
    """Text with its marks and spaces folded, for matching an ayah."""
    return " ".join(_MARKS.sub("", text).split())


def ayah_index() -> dict[str, tuple[int, int]]:
    """Each ayah's imlaey words (marks folded) to its ``(surah, ayah)``: the benchmark's
    ``reference_text`` is imlaey, which the phonetizer cannot read, so a clip whose id carries
    no ayah is matched to the Qur'an through this index. Built once per process: it is a
    function of the Qur'an alone, and the report reads it from two sections."""
    if not _AYAH_INDEX:
        _AYAH_INDEX.append(_build_ayah_index())
    return _AYAH_INDEX[0]


_AYAH_INDEX: list[dict[str, tuple[int, int]]] = []


def _build_ayah_index() -> dict[str, tuple[int, int]]:
    from quran_transcript import Aya

    index: dict[str, tuple[int, int]] = {}
    for surah in range(1, 115):
        for ayah in range(1, 300):
            try:
                words = Aya(surah, ayah).get().imlaey_words
            except Exception:
                break
            index.setdefault(_bare(" ".join(words)), (surah, ayah))
    return index


def clip_ayah(key: str, reference_text: str, index: Mapping[str, tuple[int, int]]):
    """A benchmark clip's ayah: its source's id carries it (``tlog_holdout/72_19_…``,
    ``qul_alnufais/26_189``), else the imlaey index matches the text."""
    parts = key.split("/", 1)[-1].split("_")
    if key.startswith(("tlog_holdout/", "qul_alnufais/")):
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None
    return index.get(_bare(reference_text))


def benchmark_references(lines: Mapping[str, dict]) -> tuple[dict[str, str], dict[str, int],
                                                            list[str]]:
    """Each clip's ayah reference — the phonetizer over the canonical Uthmani words, the same
    realized reference our decodes are scored against — its surah (the bootstrap cluster), and
    the clips the phonetizer cannot handle (the repo's known Quraysh ayat)."""
    from quran_transcript import Aya

    from tadabur.waqf_segments import phonetize

    index = ayah_index()
    references: dict[str, str] = {}
    surahs: dict[str, int] = {}
    unsupported: list[str] = []
    for key, line in sorted(lines.items()):
        where = clip_ayah(key, line["reference_text"], index)
        if where is None:
            unsupported.append(key)
            continue
        surah, ayah = where
        try:
            references[key] = phonetize(" ".join(Aya(surah, ayah).get().uthmani_words)).phonemes
        except (KeyError, IndexError):
            unsupported.append(key)
            continue
        surahs[key] = surah
    return references, surahs, unsupported


def _class_errors(rates: Mapping) -> dict:
    """Each class's error (1 − agreement) with its interval, on the percent scale."""
    out = {}
    for name, row in rates.items():
        if row is None:
            out[name] = None
            continue
        out[name] = {"error": round(100 * (1 - row["point"]), 2),
                     "interval": {"lower": round(100 * (1 - row["high"]), 2),
                                  "upper": round(100 * (1 - row["low"]), 2),
                                  "method": "bootstrap", "reason": ""},
                     "clusters": row["clusters"], "slots": row["slots"]}
    return out


def benchmark_rates(verdict: set[str], same_group: set[str]) -> dict:
    """The verdict clips' per-source, per-model per-class phoneme error, surah-clustered; and
    the tlog_holdout verdict clips a training row shares a group with, read apart (architect
    r3's sensitivity split)."""
    from training.nonpro_gap import rates, tally

    lines = zipformer_lines("benchmark")
    references, surahs, unsupported = benchmark_references(lines)
    sources = {key: line["source"] for key, line in lines.items()}
    texts = whole_item_texts("benchmark")
    scored = sorted(verdict & set(references))
    out = {"clips": {"verdict": len(verdict), "scored": len(scored),
                     "unsupported": sorted(set(unsupported) & verdict)},
           "arms": WHOLE_ITEM, "sources": {}}

    def errors(model: str, keys: Sequence[str]) -> dict:
        rows = [(surahs[key], tally(texts[model][key], references[key])) for key in keys]
        return _class_errors(rates(rows))

    for source in sorted({sources[key] for key in scored}):
        keys = [key for key in scored if sources[key] == source]
        out["sources"][source] = {"clips": len(keys)}
        for model in MODELS:
            out["sources"][source][model] = errors(model, keys)

    holdout = [key for key in scored if sources[key] == "tlog_holdout"]
    split = [key for key in holdout if key in same_group]
    apart = [key for key in holdout if key not in same_group]
    out["sensitivity"] = {
        "rule": "the tlog_holdout verdict clips that share a group with a training row, read "
                "apart: each model's per-class error with them and without them (architect "
                "r3's sensitivity split); every other source stays whole",
        "source": "tlog_holdout", "clips": len(split), "without_clips": len(apart),
        "models": {model: {"with": errors(model, holdout), "without": errors(model, apart)}
                   for model in MODELS},
    }
    return out


# --- qdat: the app outcome on correct recitation -----------------------------------------


def qdat_set():
    """qdat's items and our three models' decodes, plus zipformer's two check arms."""
    from training.aug_readout import Sources
    from training.aug_readout_spec import CONSONANT2

    data = Sources(CONSONANT2).qdat(STUDENTS)
    decodes = {model: data.decodes[model] for model in STUDENTS}
    for arm, lines in ((ZIP_CHECKS, zipformer_lines("qdat", "w1")),
                       (ZIP_STREAM, zipformer_lines("qdat"))):
        decodes[f"zipformer-{arm}"] = {key: {SHARED_CHECKS: line[arm]}
                                       for key, line in lines.items()}
    return data.items, decodes


def qdat_false_flags(harness) -> dict:
    """The all-cause false flags per 1,000 graded words, per arm, and the paired differences."""
    from training.aug_readout_metrics import compare, false_flag_terms
    from training.external_corpus import word_grades
    from training.muraja_policy import SHIPPED_DEFAULT

    items, decodes = qdat_set()
    arms = (*STUDENTS, f"zipformer-{ZIP_CHECKS}", f"zipformer-{ZIP_STREAM}")
    build, grades = word_grades(items, decodes, harness, arms, [SHIPPED_DEFAULT])
    terms = false_flag_terms(grades, arms, SHIPPED_DEFAULT.name)
    view = compare(terms, PER_1000, [(CANDIDATE, TEACHER),
                                     (f"zipformer-{ZIP_CHECKS}", TEACHER),
                                     (f"zipformer-{ZIP_CHECKS}", CANDIDATE),
                                     ("c95", TEACHER),
                                     (f"zipformer-{ZIP_STREAM}", f"zipformer-{ZIP_CHECKS}")])
    return {"harness": build, "items": len(items), "arms": list(arms), "words": len(terms.clusters),
            **view}


# --- the synthetic edits -------------------------------------------------------------------


def edits_rates() -> dict:
    """Each model's heard rate on the 150 edits and unchanged rate on the 150 decoys, paired by
    the edit's ``pair_id`` cluster."""
    from tadabur.synthetic_edits import read_items, teacher_agrees

    items = read_items()
    texts = whole_item_texts("edits")
    out = {"items": len(items), "arms": {}, "differences": {}}
    for role in ("edit", "decoy"):
        observations = {
            model: [(item["pair_id"],
                     int(teacher_agrees(item, texts[model][item["item_id"]])), 1)
                    for item in items if item["role"] == role]
            for model in MODELS}
        view = rate_view(observations,
                         [(model, CANDIDATE) for model in MODELS if model != CANDIDATE], PERCENT)
        out["arms"][role] = {model: view["models"][model] for model in MODELS}
        out["differences"][role] = view["comparisons"]
    return out


# --- step 6's carrier lines -----------------------------------------------------------------

#: Each line's (numerator, denominator) in :func:`_mark_counts`; the empty-slot share is the
#: rate the mushaf's own prescription sets, and a committed unit is one that carried *any*
#: explicit mark on the carrier (zipformer writes the haraka inside the consonant token).
CARRIER_METRICS = {
    "empty_slot_haraka": ("haraka_empty", "haraka_slots"),
    "empty_slot_sukun": ("sukun_empty", "sukun_slots"),
    "commit_rate": ("committed", "haraka_slots"),
    "committed_accuracy": ("accurate", "committed"),
}


def _mark_counts(decode: str, reference: str) -> dict:
    """One clip's carrier counts, read off the carrier-anchored alignment.

    A reference carrier's prescription is the mark written after it (past a shaddah) — one of
    the three harakat, or no mark, which is how the mushaf writes a sukun. A carrier the local
    alignment never reached is counted apart (``unaligned``), not as an empty slot: its slot
    may be missing with its consonant, which the per-class error already measures.
    """
    from training.acceptance_stats import SHADDAH_MARK
    from training.tashkeel_eval import SHORT_VOWELS, carrier_readings

    readings = carrier_readings(decode, reference)
    out = {"haraka_slots": 0, "haraka_empty": 0, "sukun_slots": 0, "sukun_empty": 0,
           "committed": 0, "accurate": 0, "unaligned": 0}
    for index, char in enumerate(reference):
        if unicodedata.combining(char):
            continue
        reading = readings.get(index)
        if reading is None:
            out["unaligned"] += 1
            continue
        after = index + 1
        if after < len(reference) and reference[after] == SHADDAH_MARK:
            after += 1
        prescribed = reference[after] if after < len(reference) else None
        marks = reading.marks
        if prescribed in SHORT_VOWELS:
            out["haraka_slots"] += 1
            out["haraka_empty"] += not marks
            out["committed"] += bool(marks)
            out["accurate"] += prescribed in marks
        else:
            out["sukun_slots"] += 1
            out["sukun_empty"] += not marks
    return out


def _carrier_rates(texts: Mapping[str, Mapping[str, str]], references: Mapping[str, str],
                   clusters: Mapping[str, object]) -> dict:
    """Each model's carrier lines over the same clips, one observation per clip and cluster."""
    keys = sorted(references)
    counts = {model: [_mark_counts(texts[model][key], references[key]) for key in keys]
              for model in texts}
    out = {}
    for metric, (num, den) in CARRIER_METRICS.items():
        observations = {model: [(clusters[key], counts[model][index][num],
                                 counts[model][index][den]) for index, key in enumerate(keys)]
                        for model in texts}
        view = rate_view(observations, [(model, TEACHER) for model in texts if model != TEACHER],
                         PERCENT)
        out[metric] = {"models": view["models"], "comparisons": view["comparisons"]}
    return out


def commit_lines() -> dict:
    """Issue #245 step 6's three lines, never pooled across the two sides.

    * **Empty-slot rate** where the mushaf prescribes a haraka, and where it prescribes a sukun
      (no written haraka): the share of carriers the decode left bare. The first is an omission,
      the second the expected form.
    * **Commit rate and committed accuracy** separately: a unit that carries the haraka inside
      its consonant token still commits the slot; only an emitted mark can be accurate.
    * **Teacher agreement** for our students only, on the correct side: the projected edit
      distance to ``base`` over the same clips, a clone diagnostic, never a reference a student
      is judged against and never computed for zipformer.
    """
    from training.acceptance_stats import AgreementRecord, Resample, agreement_terms
    from training.truth_scorer import estimate

    plan = json.loads((ZIPFORMER_DIR / "disjointness.json").read_text(encoding="utf-8"))["plan"]
    lines = zipformer_lines("benchmark")
    references, surahs, _ = benchmark_references(lines)
    scored = {key: references[key] for key in plan["verdict"] if key in references}
    texts = whole_item_texts("benchmark")
    correct = {"set": "benchmark", "clips": len(scored),
               "rates": _carrier_rates({model: texts[model] for model in MODELS}, scored, surahs)}
    for model in STUDENTS:
        if model == TEACHER:
            continue
        records = [AgreementRecord(surahs[key], texts[TEACHER][key], texts[model][key])
                   for key in sorted(scored)]
        clusters, num, den = agreement_terms(records)
        correct["rates"].setdefault("teacher_agreement", {})[model] = estimate(
            clusters, (num, den), np.ones(len(clusters)), Resample.by_cluster(clusters))

    from tadabur.synthetic_edits import read_items

    items = [item for item in read_items() if item["role"] == "edit"]
    edit_texts = whole_item_texts("edits")
    catch = {"set": "edits", "items": len(items),
             "rates": _carrier_rates({model: edit_texts[model] for model in MODELS},
                                     {item["item_id"]: item["labelled_reference"]
                                      for item in items},
                                     {item["item_id"]: item["pair_id"] for item in items})}
    return {"rule": "Step 6's carrier lines, never pooled across the correct and catch sides: "
                    "the empty-slot rate where the mushaf prescribes a haraka (an omission) and "
                    "where it prescribes a sukun (no written haraka, so an empty slot is the "
                    "expected form); the commit rate and committed accuracy separately, because "
                    "a unit may carry the haraka inside its consonant token; and teacher "
                    "agreement for our students only, projected onto the teacher's vocabulary, "
                    "a clone diagnostic that is never computed for zipformer",
            "correct_side": correct, "catch_side": catch}


# --- Ship A ---------------------------------------------------------------------------------


def ship_a_set():
    """Ship A's sites, and each model's whole-item text and recorded check sequence, on the
    items every compared arm holds (the candidate's committed cache covers 363 of 512)."""
    from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips
    from tadabur.truth_sites import TRUTH_SITES_DIR, load_truth_sites
    from training.truth_baseline import (CHECKS as TB_CHECKS, CHECK_STORE, DECODE_STORE,
                                         item_key, items_of, read_cache)

    sites = load_truth_sites(TRUTH_SITES_DIR / "ship_a_verified.jsonl")
    items = items_of(sites, load_staged_clips(REGISTRY_PATH))
    caches = {model: read_cache(CHECK_STORE.path(model))["items"] for model in STUDENTS}
    shared = set(caches[CANDIDATE]) & set(caches[TEACHER]) & set(items)
    kept = [site for site in sites if item_key(site) in shared]
    decodes, sequences = {}, {}
    for model in STUDENTS:
        decode = read_cache(DECODE_STORE.path(model))["items"]
        decodes[model] = {key: decode[key][WHOLE_ITEM[model]] for key in shared}
        sequences[model] = {key: caches[model][key][TB_CHECKS] for key in shared}
    w1, spans = zipformer_lines("ship-a", "w1"), zipformer_lines("ship-a")
    spans_text = {key: spans[key]["spans"] for key in shared}
    for arm, source in ((ZIP_CHECKS, w1), (ZIP_STREAM, spans)):
        decodes[f"zipformer-{arm}"] = spans_text
        sequences[f"zipformer-{arm}"] = {key: source[key][arm] for key in shared}
    return kept, decodes, sequences


def _site_rows(subset, outcome_of, arm, clusters) -> list[tuple[int, int, int]]:
    return sorted((clusters[site.site_id], int(outcome_of(arm, site)), 1) for site in subset)


def ship_a_outcomes(harness) -> dict:
    """Each arm's flagged counts on Ship A's mistake and correct sites, the paired differences,
    and the ratchet-free line from each model's one whole-item decode."""
    from training.muraja_policy import SHIPPED_DEFAULT, WORD_FLAGGED
    from training.site_outcomes import MISTAKE_SIDE, side
    from training.truth_baseline import app_outcomes

    sites, decodes, sequences = ship_a_set()
    _, kept = app_outcomes(sites, [], decodes, sequences, harness)
    scoring = SHIPPED_DEFAULT.name
    mistakes = [site for site in sites if side(site) == MISTAKE_SIDE]
    correct = [site for site in sites if side(site) != MISTAKE_SIDE]

    def flagged(arm, site):
        return kept.grades[scoring][arm][site.site_id].outcome in WORD_FLAGGED

    arms = ("base", CANDIDATE, f"zipformer-{ZIP_CHECKS}", f"zipformer-{ZIP_STREAM}")
    clusters = {site.site_id: index
                for index, site in enumerate(sorted(sites, key=lambda s: s.site_id))}
    return {"sites": len(sites), "mistakes": len(mistakes), "correct": len(correct),
            "harness": harness.build,
            "mistakes_flagged": rate_view(
                {arm: _site_rows(mistakes, flagged, arm, clusters) for arm in arms},
                [(arm, CANDIDATE) for arm in arms if arm != CANDIDATE]
                + [(f"zipformer-{ZIP_STREAM}", f"zipformer-{ZIP_CHECKS}")], PERCENT),
            "correct_flagged": rate_view(
                {arm: _site_rows(correct, flagged, arm, clusters) for arm in arms}, [], PERCENT),
            "whole_item": _whole_item_outcomes(sites, decodes["base"],
                                               decodes[f"zipformer-{ZIP_STREAM}"],
                                               mistakes, correct)}


def _whole_item_outcomes(sites, base, zipformer, mistakes, correct) -> dict:
    """The ratchet-free line: each model's one whole-item decode graded as checks of its own (the
    single-decode approximation), flagged on the same sites."""
    from training.muraja_policy import SHIPPED_DEFAULT, WORD_FLAGGED
    from training.truth_baseline import app_outcomes

    _, kept = app_outcomes(sites, [], {"base": base, "zipformer": zipformer}, {}, _harness())
    scoring = SHIPPED_DEFAULT.name

    def count(arm, subset):
        grades = kept.grades[scoring][arm]
        return {"flagged": sum(grades[site.site_id].outcome in WORD_FLAGGED for site in subset),
                "den": len(subset)}

    return {"harness": _harness().build,
            **{arm: {"mistakes": count(arm, mistakes), "correct": count(arm, correct)}
               for arm in ("base", "zipformer")}}


_HARNESS: list = []


def _harness():
    from training.muraja_policy import require_harness

    if not _HARNESS:
        _HARNESS.append(require_harness())
    return _HARNESS[0]


# --- look diversity (explanation only) ------------------------------------------------------


def _raw_looks(items, placements) -> dict[str, tuple[dict[int, list], dict[int, dict]]]:
    """Each item's per-word look qualities and kept status, from one harness run (shipped
    default only, the scoring every verdict reads). ``items`` are harness items whose keys
    carry their arm (``arm#item``)."""
    from training.muraja_policy import SHIPPED_DEFAULT, request

    _, results = _harness().run([request(item, placements[item.key], [SHIPPED_DEFAULT])
                                 for item in items])
    out: dict[str, tuple[dict[int, list], dict[int, dict]]] = {}
    for result in results:
        looks: dict[int, list] = defaultdict(list)
        for check in result["checks"]:
            for status in check["statuses"]:
                looks[status["word"]].append(status["quality"])
        out[result["item"]] = (dict(looks), {s["word"]: s for s in result["final"]})
    return out


def _look_metrics(looks, kept) -> tuple[int, int, int, int]:
    """``(words, flagged_by_any, rescued, multi_look)`` for one item's words."""
    from training.muraja_policy import CORRECT, PENDING, SHIPPED_DEFAULT, displayed

    words = flagged = rescued = multi = 0
    for word, qualities in looks.items():
        scored = [quality for quality in qualities if quality != PENDING]
        if not scored:
            continue
        words += 1
        multi += len(scored) >= 2
        bad = any(displayed(quality, SHIPPED_DEFAULT) != CORRECT for quality in scored)
        flagged += bad
        status = kept.get(word)
        rescued += bool(bad and status is not None
                        and displayed(status["quality"], SHIPPED_DEFAULT) == CORRECT)
    return words, flagged, rescued, multi


def _qdat_looks() -> dict:
    """qdat's look diversity: each arm's rescue rate and its ratchet gain against the
    single-decode approximation of the same model's whole-item text."""
    from training.external_corpus import placement, placed_words
    from training.muraja_policy import Item, MurajaText, single_decode_checks

    items, decodes = qdat_set()
    text = MurajaText(_harness().quran_db)
    placed = {item.key: placed_words(item, text.phoneme_word) for item in items}
    at = {key: placement(item, placed[key]) for key, item in
          ((item.key, item) for item in items)}
    arms = {"base": ("checks", "stream_b0"), "c95": ("checks", "stream_b0"),
            "r2_candidate_ema": ("checks", "stream_b0"),
            "zipformer-checks-w1": (ZIP_CHECKS, "spans"),
            "zipformer-checks": (ZIP_STREAM, "spans"),
            "zipformer-checks-w1-windows": (ZIP_WINDOWS, "spans")}
    w1, spans = zipformer_lines("qdat", "w1"), zipformer_lines("qdat")
    files = {"zipformer-checks-w1": w1, "zipformer-checks": spans,
             "zipformer-checks-w1-windows": w1}
    table = {}
    for arm, (check_arm, whole_arm) in arms.items():
        source = files.get(arm)
        checks = {item.key: (source[item.key][check_arm] if source is not None
                             else decodes[arm][item.key][check_arm]) for item in items}
        whole = {item.key: (spans[item.key]["spans"] if arm.startswith("zipformer")
                            else decodes[arm][item.key][whole_arm]) for item in items}
        runs = _raw_looks(
            [Item(f"{arm}#{item.key}", tuple(word for word, _ in placed[item.key]),
                  tuple(checks[item.key])) for item in items],
            {f"{arm}#{key}": where for key, where in at.items()})
        singles = _raw_looks(
            [Item(f"{arm}#{item.key}", tuple(word for word, _ in placed[item.key]),
                  tuple(single_decode_checks(whole[item.key],
                                             item.end_sample - item.start_sample)))
             for item in items],
            {f"{arm}#{key}": where for key, where in at.items()})
        words = flagged = rescued = multi = flagged_single = 0
        for item in items:
            w, f, r, m = _look_metrics(*runs[f"{arm}#{item.key}"])
            words, flagged, rescued, multi = words + w, flagged + f, rescued + r, multi + m
            flagged_single += _look_metrics(*singles[f"{arm}#{item.key}"])[1]
        table[arm] = {"words": words, "flagged": flagged,
                      "flagged_single_decode": flagged_single,
                      "rescue_rate": round(100 * rescued / flagged, 2) if flagged else None,
                      "ratchet_gain": flagged - flagged_single,
                      "multi_look_share": round(100 * multi / words, 2) if words else None}
    return table


def _ship_a_hidden() -> dict:
    """Ship A's ratchet-hidden catches: mistake sites flagged by at least one look whose kept
    grade is correct (the ratchet's saving, on the catch side)."""
    from training.muraja_policy import (CORRECT, Item, MurajaText, PENDING, SHIPPED_DEFAULT,
                                        displayed, place)
    from training.site_outcomes import MISTAKE_SIDE, side
    from training.truth_baseline import item_key

    sites, _, sequences = ship_a_set()
    mistakes = [site for site in sites if side(site) == MISTAKE_SIDE]
    by_item: dict[str, list] = defaultdict(list)
    for site in mistakes:
        by_item[item_key(site)].append(site)
    text = MurajaText(_harness().quran_db)
    out = {}
    for arm in (CANDIDATE, f"zipformer-{ZIP_CHECKS}"):
        requests = [Item(f"{arm}#{key}", tuple(sites_), tuple(sequences[arm][key]))
                    for key, sites_ in sorted(by_item.items())]
        at = {request.key: place(request, text) for request in requests}
        runs = _raw_looks(requests, at)
        graded = flagged = hidden = 0
        for key in by_item:
            looks, kept = runs[f"{arm}#{key}"]
            for site in by_item[key]:
                where = at[f"{arm}#{key}"].where[site.site_id]
                if where is None or where.word not in looks:
                    continue
                graded += 1
                qualities = [q for q in looks[where.word] if q != PENDING]
                bad = any(displayed(q, SHIPPED_DEFAULT) != CORRECT for q in qualities)
                status = kept.get(where.word)
                flagged += bad
                hidden += bool(bad and status is not None
                               and displayed(status["quality"], SHIPPED_DEFAULT) == CORRECT)
        out[arm] = {"graded": graded, "flagged_any": flagged, "hidden_catches": hidden,
                    "hidden_rate": round(100 * hidden / graded, 2) if graded else None}
    return out


def look_diversity() -> dict:
    """The look-diversity table: qdat's per-arm rescue rate and ratchet gain, and Ship A's
    ratchet-hidden catches. Explanation only; nothing here gates."""
    return {"qdat": _qdat_looks(), "ship_a": _ship_a_hidden()}


# --- compute, exposure, gates ---------------------------------------------------------------


def compute_table() -> dict:
    """Decoded audio per audio-second and calls per audio-second per arm, from the schedule; the
    batch's measured wall per job sits under ``measured``."""
    from training.decoding import WINDOW, check_schedule
    from training.zipformer_decode import decoded_audio_ratio

    samples = 60 * SAMPLE_RATE
    schedule = check_schedule(samples)
    checks = len(schedule) / 60.0
    windows = sum(1 for span in schedule if span.kind == WINDOW) / 60.0
    arms = {
        "ours": {"decoded_audio_seconds_per_audio_second": decoded_audio_ratio("ours", samples),
                 "inferences_per_audio_second": checks},
        "zipformer-single-stream": {"decoded_audio_seconds_per_audio_second": 1.0,
                                    "inferences_per_audio_second": 1.0 / CHUNK_SECONDS},
        ZIP_CHECKS: {"decoded_audio_seconds_per_audio_second":
                     decoded_audio_ratio(ZIP_CHECKS, samples),
                     "inferences_per_audio_second": checks},
        # only its windows are inferred: its previews and stop are cut from the one stream
        ZIP_WINDOWS: {"decoded_audio_seconds_per_audio_second":
                      decoded_audio_ratio(ZIP_WINDOWS, samples) + 1.0,
                      "inferences_per_audio_second": windows},
    }
    for row in arms.values():
        row["decoded_audio_seconds_per_audio_second"] = round(
            row["decoded_audio_seconds_per_audio_second"], 2)
        row["inferences_per_audio_second"] = round(row["inferences_per_audio_second"], 2)
        if row["decoded_audio_seconds_per_audio_second"] > 1.0:
            row["onnx_chunk_calls_per_audio_second"] = round(
                row["decoded_audio_seconds_per_audio_second"] / CHUNK_SECONDS, 1)
    return {"arms": arms, "measured": _measured_wall()}


def _measured_wall() -> dict:
    """Each batch job's measured wall and rate, from the committed timing files.

    ``decoded_audio_seconds`` is seconds in both the record and its files
    (``decoded_audio_unit``); a row written in samples predates the unit fix and is refused
    rather than silently divided.
    """
    out = {}
    for path in sorted(DECODES.glob("*.zipformer*.timing.jsonl.gz")):
        for row in read_rows(path):
            if row.get("decoded_audio_unit") != "seconds":
                raise ValueError(f"{path}: {row['set']}/{row.get('store')} has no "
                                 f"decoded_audio_unit: seconds")
            store = row.get("store", "decode")
            name = f"{row['set']}/{store}"
            out[name] = {"items": row["items"], "audio_seconds": row["audio_seconds"],
                         "decoded_audio_seconds": row["decoded_audio_seconds"],
                         "wall_seconds": row["wall_seconds"], "workers": row["workers"]}
    return out


def exposure() -> dict:
    """The synthetic edits' source and donor recordings, and the share of each lineage's Tadabur
    training reciters that touches them: the reciter-in-training screen, the only one the
    registry runs on our models. The teacher and zipformer trained outside Tadabur."""
    from tadabur.exposure import (AUG_TRAINING, CONSONANT_TRAINING, CONSONANT2_TRAINING,
                                  SYNTHETIC_EDIT_DONOR, SYNTHETIC_EDIT_SOURCE, load_registry)

    registry = load_registry()
    uses = {"base": None, "c95": (AUG_TRAINING, CONSONANT_TRAINING),
            "r2_candidate_ema": (CONSONANT_TRAINING, CONSONANT2_TRAINING),
            "zipformer": None}
    out = {"rule": "share of the edit set's reciters in the lineage's Tadabur training uses; "
                   "the teacher (Muaalem) and zipformer trained outside Tadabur, so their "
                   "exposure is unknown by name, not zero",
           "uses": {model: list(use) if use is not None else None
                    for model, use in uses.items()}, "models": {}}
    for corpus in (SYNTHETIC_EDIT_SOURCE, SYNTHETIC_EDIT_DONOR):
        reciters = {row.reciter_id for row in registry.recordings[corpus]}
        for model, model_uses in uses.items():
            trained = (None if model_uses is None
                       else {r for use in model_uses for r in registry.reciters(use)})
            out["models"].setdefault(model, {})[corpus] = {
                "reciters": len(reciters), "exposed": None if trained is None
                else len(reciters & trained),
                "share": None if trained is None
                else round(100 * len(reciters & trained) / len(reciters), 1)}
    return out


def gates() -> dict:
    """The three gates in the order they ran, from their committed records."""
    out = {}
    for name, filename in (("card_gate_v3_2_baseline", "card_gate.json"),
                           ("card_gate_v3_decoder_check", "card_gate.v3.json")):
        record = json.loads((ZIPFORMER_DIR / filename).read_text(encoding="utf-8"))
        pinned = record["scales"][str(record["pinned_scale"])]
        roundtrip = pinned["roundtrip"]
        out[name] = {"model": record["model"]["name"], "sha256": record["model"]["sha256"],
                     "role": record["role"], "passed": record["passed"],
                     "pinned_scale": record["pinned_scale"], "stop": record.get("stop"),
                     "roundtrip": {"identical": roundtrip["identical"],
                                   "clips": roundtrip["clips"],
                                   "minimum": roundtrip["minimum"],
                                   "accepted": roundtrip["accepted"],
                                   "failures": {kind: sum(1 for failure in
                                                          roundtrip["failures"]
                                                          if failure["classification"] == kind)
                                                for kind in sorted({failure["classification"]
                                                                    for failure in
                                                                    roundtrip["failures"]})}},
                     "character_per": {scale: {source: row["character_per"][source]["per"]
                                               for source in sorted(row["character_per"])}
                                       for scale, row in record["scales"].items()}}
    parity = json.loads((ZIPFORMER_DIR / "parity.json").read_text(encoding="utf-8"))
    out["parity"] = {"passed": parity["passed"], "rule": parity["rule"]}
    return out


def disjointness() -> dict:
    """The committed disjointness screen's counts, and the decode plan's verdict set."""
    record = json.loads((ZIPFORMER_DIR / "disjointness.json").read_text(encoding="utf-8"))
    plan = record["plan"]
    return {"clips": record["benchmark"]["clips"],
            "benchmark": {key: record["benchmark"][key]
                          for key in ("rows", "clips", "duplicate_rows", "manifest_sha256")},
            "plan": {name: len(keys) for name, keys in plan.items() if name != "by_role"},
            "roles": len(plan["by_role"]),
            "verdict": len(plan["verdict"]), "trained": len(plan["trained"]),
            "same_group": len(plan["same_group"]), "sealed": len(plan["sealed"]),
            "tlog": {key: record["tlog"][key] for key in
                     ("rows", "training", "consumed", "indexed_rows", "unfingerprinted_rows")},
            "name_screen": record["name_screen"]}


def arm_map() -> dict:
    """Every model's arm per metric, from :func:`training.zipformer_decode.metric_arm`."""
    return {model: {metric: metric_arm(model, metric) for metric in METRICS} for model in MODELS}


# --- the verdict ----------------------------------------------------------------------------


def closure(teacher: float, candidate: float, arm: float) -> dict:
    """Whether ``arm`` closes at least half the candidate−teacher gap on a lower-is-better
    metric: it must reach ``teacher + 0.5 × (candidate − teacher)``. A gap at or below zero (the
    candidate already at or below the teacher) is closed by anything at or below the candidate."""
    gap = candidate - teacher
    target = candidate if gap <= 0 else teacher + GAP_CLOSED * gap
    return {"teacher": teacher, "candidate": candidate, "zipformer": arm,
            "gap": round(gap, 2), "target": round(target, 2),
            "closed": bool(arm <= target)}


def verdict(record: Mapping) -> dict:
    """The pre-registered rule over the record's numbers (issue #245, ``checks-w1`` arm)."""
    qdat = record["qdat"]["models"]
    tlog = record["benchmark"]["sources"]["tlog_holdout"]
    correct = {
        "qdat_false_flags": closure(qdat[TEACHER]["point"], qdat[CANDIDATE]["point"],
                                    qdat[f"zipformer-{ZIP_CHECKS}"]["point"]),
        "tlog_holdout_consonant": closure(tlog[TEACHER]["consonant"]["error"],
                                          tlog[CANDIDATE]["consonant"]["error"],
                                          tlog["zipformer"]["consonant"]["error"]),
        "tlog_holdout_haraka": closure(tlog[TEACHER]["haraka"]["error"],
                                       tlog[CANDIDATE]["haraka"]["error"],
                                       tlog["zipformer"]["haraka"]["error"]),
    }
    edits = record["edits"]["arms"]
    heard, candidate_heard = edits["edit"]["zipformer"]["point"], edits["edit"][CANDIDATE]["point"]
    decoys = edits["decoy"]["zipformer"]["point"]
    catch = {
        "edits_heard": {"zipformer": heard, "candidate": candidate_heard,
                        "bar": round(candidate_heard - CATCH_POINTS, 2),
                        "passed": bool(heard >= candidate_heard - CATCH_POINTS)},
        "edit_decoys_unchanged": {"zipformer": decoys,
                                  "candidate": edits["decoy"][CANDIDATE]["point"],
                                  "bar": DECOY_FLOOR,
                                  "passed": bool(decoys >= DECOY_FLOOR)},
        "ship_a_mistakes_flagged": _ship_a_bar(record),
    }
    correct_passed = all(row["closed"] for row in correct.values())
    catch_passed = all(row["passed"] for row in catch.values())
    return {"rule": "GO = the correct side closes half the gap on qdat and tlog_holdout and the "
                    "catch side is within the candidate's minus 5 points (decoys >= 95%); "
                    "PRIOR = correct passes, catch fails; STOP = correct fails",
            "correct_side": correct, "catch_side": catch,
            "correct_passed": correct_passed, "catch_passed": catch_passed,
            "verdict": ("GO" if correct_passed and catch_passed else
                        "PRIOR" if correct_passed else "STOP")}


def _ship_a_bar(record: Mapping) -> dict:
    view = record["ship_a"]["mistakes_flagged"]["models"]
    arm = view[f"zipformer-{ZIP_CHECKS}"]["point"]
    bar = view[CANDIDATE]["point"] - CATCH_POINTS
    return {"zipformer": arm, "candidate": view[CANDIDATE]["point"], "bar": round(bar, 2),
            "passed": bool(arm >= bar),
            "note": "on the candidate's 363 committed items (46 of 75 mistake sites)"}


def match_rule(record: Mapping) -> dict:
    """The owner's pre-declared rule (`.orch/tasks/zip-run/owner-confirm.md`): single-stream
    ``checks`` matches ``checks-w1`` when the paired 95% interval of their difference contains
    zero for both qdat false flags and Ship A catches."""
    rows = {"qdat_false_flags": paired(record["qdat"], f"zipformer-{ZIP_STREAM}",
                                       f"zipformer-{ZIP_CHECKS}"),
            "ship_a_catches": _single_stream_ship_a(record)}
    missing = [name for name, row in rows.items() if row is None]
    if missing:
        raise ValueError(f"match rule: {missing} has no paired difference in the record")
    matches = all(row["interval"]["lower"] <= 0 <= row["interval"]["upper"]
                  for row in rows.values())
    return {"rule": "single-stream checks matches checks-w1 when the paired 95% interval of the "
                    "difference contains 0 on both metrics; then one stream suffices in the app",
            "rows": rows, "matches": matches}


def _single_stream_ship_a(record: Mapping) -> dict | None:
    """The paired single-stream minus checks-w1 difference on Ship A's mistake sites."""
    return paired(record["ship_a"]["mistakes_flagged"], f"zipformer-{ZIP_STREAM}",
                  f"zipformer-{ZIP_CHECKS}")


def ratchet_free(record: Mapping) -> dict:
    """The report-only ratchet-free lines: qdat's per-class error from the whole-item decodes,
    and Ship A's whole-item outcomes, beside (never inside) the verdict."""
    return {"rule": "whole-item decodes only, no app check sequence and no ratchet: the reading "
                    "of each model's ability without Muraja's engine in the loop, each on its "
                    "metric arm (stream_b0 for ours, spans for zipformer)",
            "qdat_per_class": _qdat_per_class(),
            "ship_a_whole_item": record["ship_a"]["whole_item"]}


def _qdat_per_class() -> dict:
    """qdat's whole-item per-class error against the mushaf, reciter-clustered, each model on
    its own metric arm (``stream_b0`` for ours, ``spans`` for zipformer, which has no
    ``stream_b0``): the report-only line reads the arm map, never a mixed comparison."""
    from training.nonpro_gap import rates, tally

    items, decodes = qdat_set()
    arms = {"zipformer": zipformer_lines("qdat"),
            **{model: decodes[model] for model in STUDENTS}}
    out = {}
    for model, source in arms.items():
        rows = [(item.cluster,
                 tally(source[item.key][WHOLE_ITEM[model]],
                       item.reference.reference))
                for item in items]
        out[model] = _class_errors(rates(rows))
    return out


# --- the report and the doc ----------------------------------------------------------------


def report() -> dict:
    """Every section of the committed record."""
    harness = _harness()
    plan = json.loads((ZIPFORMER_DIR / "disjointness.json").read_text(encoding="utf-8"))["plan"]
    probe = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    if probe.get("decoded_audio_unit") != "seconds":
        raise ValueError(f"{PROBE_PATH}: no decoded_audio_unit: seconds")
    record = {
        "issue": 245,
        "schema": 1,
        "gates": gates(),
        "disjointness": disjointness(),
        "arm_map": arm_map(),
        "exposure": exposure(),
        "probe": {"measured_rate": round(probe["decoded_audio_per_second"], 2),
                  "measured_unit": probe["decoded_audio_unit"],
                  "measured_wall_seconds": probe["probe"]["wall_seconds"],
                  "projected_hours": round(probe["projected_seconds"] / 3600.0, 2),
                  "horizon_hours": HORIZON_HOURS, "stop": bool(probe["stop"]),
                  "rule": "the checks-w1 store's audio-seconds per wall-second on 20 qdat "
                          "items, projected over every job; the batch stops past 12 h "
                          "(architect r2, AC 6)"},
        "compute": compute_table(),
        "benchmark": benchmark_rates(set(plan["verdict"]), set(plan["same_group"])),
        "qdat": qdat_false_flags(harness),
        "edits": edits_rates(),
        "commit": commit_lines(),
        "ship_a": ship_a_outcomes(harness),
        "look_diversity": look_diversity(),
    }
    record["match_rule"] = match_rule(record)
    record["report_only"] = ratchet_free(record)
    record["verdict"] = verdict(record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("report", "doc"))
    args = parser.parse_args(argv)
    if args.command == "report":
        record = report()
        write_json(record, REPORT_PATH)
        print(json.dumps(record["verdict"], ensure_ascii=False, indent=1))
        return 0
    from training.zipformer_comparison_doc import render

    DOC_PATH.write_text(render(json.loads(REPORT_PATH.read_text(encoding="utf-8"))),
                        encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

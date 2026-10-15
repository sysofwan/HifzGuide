"""The QUD pilot, measured without listening (#144): carriers, disagreements, the app outcome.

Two professional Hafs mushafs no model has heard (:mod:`tadabur.qud_overlap`) are decoded by
the base teacher and ``h448`` under the truth baseline's three protocols
(:mod:`training.qud_decode`: ``spans``, ``stream_b0`` and the recorded ``checks``). This module
reads those decodes against the mushaf and writes what #144 asks of a pilot:

**Items.** Each of QUD's waqf-aware segments of an ayah is an item: its recited words mapped
onto quran-transcript's Uthmani words (:mod:`tadabur.qud_text`), its realized reference the
phonetizer's output over them, last word in waqf. An ayah whose words cannot be mapped, and a
segment the phonetizer cannot handle, are listed and left out.

**Carriers and rates.** Every target-pair carrier (D9, eighteen directions) and in-word geminate
of every item, read with :mod:`training.carrier_windowing`'s rules and rates
(:mod:`training.direction_table`: ``mushaf``, ``swap``, ``other``) per model and protocol,
against the **mushaf**. Each mushaf is one reciter, so its intervals are clustered by surah.

**Sites (#144's rule, ADR-0011 §3).** A carrier is **agreed** when the base teacher decoding the
segment whole and ``h448`` streamed at b = 0 (the two decodes Ship A's frame reads) both commit
the mushaf's letter (both hold the geminate): a truth site with ``heard`` = the mushaf's letter
and ``assumes_competent_reciter``. It is a **disagreement** when either commits something else:
a ``pending`` site, and one row of the worklist, which is **not queued** for listening. Where
neither disagrees but one commits nothing at the carrier it is **abstained**: counted, and kept
out of the site files. A carrier of a letter in two pairs is a site per direction.

**App outcome.** Each item's recorded check sequence, per model, is replayed through Muraja's
engine (:mod:`training.muraja_policy`) under the shipped default and with ADR-0012's allowances
off, and every word of the item is graded; a flagged word's causes are
:mod:`training.word_flags`'. The recitation is presumed correct, so every flag is presumed
false; set beside the Tadabur word sample's false flags (``truth_baseline/report.json``).

Files (``tadabur/qud_eval/pilot/``): ``decodes/<slug>.<model>.jsonl.gz`` (every item's three
decodes, merged from the GPU box), ``truth_sites/<slug>.jsonl.gz``, ``worklist.jsonl``,
``report.json``; the doc ``docs/qud-pilot.md``; the exposure use ``qud.correct_recitation``.

Usage (from ``tools/``; torch-free; ``app`` needs the Muraja harness)::

  python -m training.qud_pilot merge --decodes-dir <the GPU box's decodes, copied>
  python -m training.qud_pilot build
  python -m training.qud_pilot app
  python -m training.qud_pilot doc
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.listening_session import ANSWER_SECONDS, EXCERPT_PAD_S, PLAYS_PER_SITE
from tadabur.qud_overlap import PILOT
from tadabur.qud_selection import QUD_DIR, SELECTION_PATH
from tadabur.qud_source import QUD_REVISION, QudRow, mushaf_metadata
from tadabur.qud_text import ItemReference, Unmapped, item_reference, map_words
from tadabur.truth_sites import PENDING, QUD, SHADDAH, TruthSite, write_truth_sites
from training.carrier_windowing import (
    APP_PROTOCOLS,
    MODELS,
    SPANS,
    Clip,
    arm,
    gap_report,
    geminate_positions,
    pair_positions,
    site_rows,
)
from training.decoding import committed_text
from training.direction_table import direction
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import CHECKS
from training.truth_scorer import SiteRow

PILOT_DIR = QUD_DIR / "pilot"
DECODES_DIR = PILOT_DIR / "decodes"
SITES_DIR = PILOT_DIR / "truth_sites"
WORKLIST_PATH = PILOT_DIR / "worklist.jsonl"
REPORT_PATH = PILOT_DIR / "report.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "qud-pilot.md"

#: The two decodes #144's rule reads (Ship A's frame): the base teacher on the whole segment,
#: and ``h448`` as shipped, streamed at b = 0.
RULE_ARMS = (arm("base", SPANS), arm("h448", "stream_b0"))
AGREED, DISAGREED, ABSTAINED = "agreed", "disagreed", "abstained"
CLASSES = (AGREED, DISAGREED, ABSTAINED)
#: The listening budgets #144 asks the estimate at: disagreements drawn per stratum.
BUDGETS = (20, 50)


# --- items ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class PilotItem:
    """One QUD segment: where it is, its audio's provenance and its realized reference."""

    key: str
    slug: str
    row: QudRow
    segment_index: int
    start_sample: int
    end_sample: int
    mp3_sha256: str
    reference: ItemReference

    @property
    def audio_filename(self) -> str:
        return self.row.audio_filename


def decodes_file(slug: str, model: str) -> Path:
    return DECODES_DIR / f"{slug}.{model}.jsonl.gz"


def read_decodes(slug: str, model: str) -> dict[str, dict]:
    with gzip.open(decodes_file(slug, model), "rt", encoding="utf-8") as f:
        return {line["key"]: line for line in map(json.loads, f)}


def build_items(slug: str, rows: Sequence[QudRow], decodes: Mapping[str, dict],
                canonical_of, reference_of) -> tuple[list[PilotItem], dict]:
    """Every segment of a mushaf with a reference, and what was left out and why."""
    items, left_out = [], {"unmapped_ayat": [], "phonetizer_unsupported": [],
                           "repeated_words_ayat": 0}
    for row in rows:
        canonical = canonical_of(row.surah_ayah)
        try:
            mapping = map_words(row.text_uthmani.split(), canonical)
        except Unmapped as error:
            left_out["unmapped_ayat"].append(f"{row.surah_ayah}: {error}")
            continue
        left_out["repeated_words_ayat"] += bool(mapping.repeats)
        for k, segment in enumerate(row.segments):
            key = f"{row.audio_filename}#{k}"
            try:
                reference = item_reference(segment, canonical, reference_of)
            except (KeyError, IndexError, Unmapped):
                left_out["phonetizer_unsupported"].append(key)
                continue
            line = decodes[key]
            items.append(PilotItem(key, slug, row, k, line["start_sample"], line["end_sample"],
                                   line["mp3_sha256"], reference))
    return items, left_out


def cluster_of(item: PilotItem) -> int:
    """The bootstrap cluster: one per mushaf and surah (each mushaf is one reciter)."""
    return PILOT.index(item.slug) * 1000 + item.row.surah


def clip_of(item: PilotItem, decodes: Mapping[str, Mapping[str, dict]]) -> Clip:
    """The item as :mod:`training.carrier_windowing` reads it, with all six arms."""
    reference, offsets = item.reference.reference, item.reference.offsets
    return Clip(
        audio_filename=item.audio_filename, surah_ayah=item.row.surah_ayah,
        reciter_id=cluster_of(item), weight=1.0, references=(reference,),
        starts=(item.start_sample,),
        positions=tuple(sorted(pair_positions(0, reference)
                               + geminate_positions(0, reference, offsets))),
        by_segment={},
        by_span={
            **{arm(m, p): decodes[m][item.key][p] for m in MODELS for p in ("spans", "stream_b0")},
            **{arm(m, CHECKS): committed_text(decodes[m][item.key][CHECKS]) for m in MODELS},
        },
    )


# --- the rule ------------------------------------------------------------------------------


def classify(row: SiteRow) -> str:
    """#144's class of one carrier from the two rule arms (module docstring)."""
    committed = [row.outcomes[a].committed for a in RULE_ARMS]
    if all(c == row.site.prescribed for c in committed):
        return AGREED
    if any(c is not None and c != row.site.prescribed for c in committed):
        return DISAGREED
    return ABSTAINED


def carrier_key(row: SiteRow) -> tuple:
    return (row.site.audio_filename, row.site.start_sample, row.site.reference_index,
            row.site.mark == SHADDAH)


def stratum(row: SiteRow) -> str:
    """A carrier's listening stratum: its letter, or the in-word geminates."""
    return "geminate" if row.site.mark == SHADDAH else f"letter:{row.site.prescribed}"


def direction_counts(rows: Sequence[SiteRow]) -> dict[str, dict]:
    """Per direction: carriers and each class, and what each rule arm committed at the
    disagreements (the partner, another letter or a single consonant)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        d = direction(row.site)
        cls = classify(row)
        out[d]["carriers"] += 1
        out[d][cls] += 1
        if cls == DISAGREED:
            for a in RULE_ARMS:
                c = row.outcomes[a].committed
                if c is not None and c != row.site.prescribed:
                    out[d][f"{a} committed another"] += 1
    return {d: dict(c) for d, c in sorted(out.items())}


# --- truth sites and the worklist --------------------------------------------------------------


def truth_sites(rows: Sequence[SiteRow], items: Mapping[tuple[str, int], PilotItem]
                ) -> list[TruthSite]:
    """The agreed and disagreed sites (abstained are left out), each stratum a census of its
    mushaf, direction and class."""
    kept = [(row, classify(row)) for row in rows]
    kept = [(row, cls) for row, cls in kept if cls != ABSTAINED]

    def stratum_name(row: SiteRow, cls: str) -> str:
        item = items[(row.site.audio_filename, row.site.start_sample)]
        return f"qud:{item.slug}:{direction(row.site)}:{cls}"

    population = Counter(stratum_name(row, cls) for row, cls in kept)
    sites = []
    for row, cls in kept:
        item = items[(row.site.audio_filename, row.site.start_sample)]
        name = stratum_name(row, cls)
        sites.append(TruthSite(
            site_id=f"qud:{item.key}:{row.site.reference_index}:{row.site.mark}",
            source=QUD, assumes_competent_reciter=cls == AGREED,
            audio_filename=item.audio_filename, shard=None, start_sample=item.start_sample,
            end_sample=item.end_sample, audio_sha256=item.mp3_sha256,
            surah_ayah=item.row.surah_ayah, reference=row.site.reference,
            reference_index=row.site.reference_index, mark=row.site.mark,
            prescribed=row.site.prescribed,
            heard=row.site.prescribed if cls == AGREED else PENDING,
            stratum=name, stratum_population=population[name]))
    return sorted(sites, key=lambda s: s.site_id)


def excerpt(item: PilotItem, reference_index: int) -> tuple[int, int]:
    """The carrier's word and one either side, from QUD's word timestamps (canonical word
    indices; a re-read word has one entry per recital, so the one inside the item's own time
    span is taken), padded by #87's :data:`~tadabur.listening_session.EXCERPT_PAD_S` and
    clamped to the item. The whole item where the timestamps cannot place the words."""
    position = bisect.bisect_right(item.reference.offsets, reference_index) - 1
    words = item.reference.word_indices
    first, last = words[max(0, position - 1)] + 1, words[min(len(words) - 1, position + 1)] + 1
    lo, hi = item.row.segments[item.segment_index][2:4]
    times = {w[0]: (w[1], w[2]) for w in item.row.word_timestamps if lo <= w[1] and w[2] <= hi}
    if first not in times or last not in times:
        return item.start_sample, item.end_sample
    pad = EXCERPT_PAD_S * TARGET_SAMPLE_RATE
    start = round(times[first][0] * TARGET_SAMPLE_RATE / 1000 - pad)
    end = round(times[last][1] * TARGET_SAMPLE_RATE / 1000 + pad)
    return max(item.start_sample, start), min(item.end_sample, max(end, start + 1))


def worklist(rows: Sequence[SiteRow], items: Mapping[tuple[str, int], PilotItem]
             ) -> list[dict]:
    """One row per disagreed carrier (its sites, every arm's commit, its excerpt): what a
    listening sample would be drawn from. Nothing here is queued."""
    by_carrier: dict[tuple, list[SiteRow]] = defaultdict(list)
    for row in rows:
        if classify(row) == DISAGREED:
            by_carrier[carrier_key(row)].append(row)
    out = []
    for key, group in sorted(by_carrier.items()):
        row = group[0]
        item = items[(row.site.audio_filename, row.site.start_sample)]
        start, end = excerpt(item, row.site.reference_index)
        out.append({
            "site_ids": sorted(f"qud:{item.key}:{r.site.reference_index}:{r.site.mark}"
                               for r in group),
            "stratum": stratum(row), "slug": item.slug, "item": item.key,
            "audio_filename": item.audio_filename, "mp3_sha256": item.mp3_sha256,
            "surah_ayah": item.row.surah_ayah, "reference": row.site.reference,
            "reference_index": row.site.reference_index, "prescribed": row.site.prescribed,
            "committed": {a: o.committed for a, o in sorted(row.outcomes.items())},
            "excerpt_start_sample": start, "excerpt_end_sample": end,
            "listen_seconds": round(PLAYS_PER_SITE * (end - start) / TARGET_SAMPLE_RATE
                                    + ANSWER_SECONDS, 2),
        })
    return out


def listening_estimate(rows: Sequence[dict]) -> dict:
    """Minutes to listen to every disagreement, and to :data:`BUDGETS` per stratum (at the
    stratum's mean excerpt time), per mushaf and pooled."""
    def minutes(group: Sequence[dict]) -> dict:
        strata: dict[str, list[float]] = defaultdict(list)
        for r in group:
            strata[r["stratum"]].append(r["listen_seconds"])
        return {
            "disagreed_carriers": len(group),
            "per_stratum": {s: len(v) for s, v in sorted(strata.items())},
            "census_minutes": round(sum(r["listen_seconds"] for r in group) / 60, 1),
            **{f"k{k}_minutes": round(sum(min(k, len(v)) * sum(v) / len(v)
                                          for v in strata.values()) / 60, 1) for k in BUDGETS},
        }

    return {"pooled": minutes(rows),
            **{slug: minutes([r for r in rows if r["slug"] == slug]) for slug in PILOT}}


# --- merging the GPU box's decodes -----------------------------------------------------------


def merge_decodes(source: Path, slug: str, model: str, expected: int) -> dict:
    """One mushaf's per-shard decode files (``training.qud_decode``) as one gzipped file,
    sorted by item, each item once (a resumed shard holds no torn record: the decoder cut it
    off before resuming); its identity record must be the one every shard ran with."""
    lines: dict[str, str] = {}
    for path in sorted(source.glob(f"{slug}.*.{model}.jsonl")):
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                lines.setdefault(json.loads(raw)["key"], raw)
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"{path}:{number} is not a decode record") from error
    if len(lines) != expected:
        raise ValueError(f"{slug} {model}: {len(lines)} items decoded, {expected} expected")
    identity = json.loads((source / f"{slug}.{model}.identity.json").read_text(encoding="utf-8"))
    DECODES_DIR.mkdir(parents=True, exist_ok=True)
    body = "".join(lines[k] + "\n" for k in sorted(lines)).encode("utf-8")
    decodes_file(slug, model).write_bytes(gzip.compress(body, mtime=0))
    return identity


def compute_record(decodes_dir: Path, logs_dir: Path, gpu: str) -> dict:
    """What the decode cost: the GPU lock's wall time (the first run's start, written by the
    launcher, to the last decode written), the segment audio decoded (every merged item), how
    many shard runs completed (a run the OOM killer stopped resumed in a later one), and the
    peak disk the pilot's scratch directory held (sampled each minute, decoding and the
    overlap check together)."""
    from datetime import datetime

    timings = [json.loads(line) for path in sorted(decodes_dir.glob("*.timing.jsonl"))
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = datetime.fromisoformat(
        (logs_dir / "decode_start").read_text(encoding="utf-8").strip().replace("Z", "+00:00"))
    end = max(path.stat().st_mtime for path in decodes_dir.glob("*.jsonl"))
    samples = [line.split() for line in (logs_dir / "df.log").read_text(encoding="utf-8")
               .splitlines() if len(line.split()) == 3]
    def stamps(name: str) -> list[datetime]:
        return [datetime.fromisoformat(line.replace("Z", "+00:00"))
                for line in (logs_dir / name).read_text(encoding="utf-8").split()]

    # One start and one end per overlap-check run (a stopped run's end is its successor's
    # start); the check's wall time is their sum.
    overlap_runs = list(zip(stamps("overlap_start"), stamps("overlap_end"), strict=True))
    return {
        "gpu": gpu,
        "gpu_hours": round((end - start.timestamp()) / 3600, 2),
        "audio_hours": round(sum(line["end_sample"] - line["start_sample"]
                                 for slug in PILOT for line in read_decodes(slug, "base").values())
                             / TARGET_SAMPLE_RATE / 3600, 2),
        "shard_runs_completed": len(timings),
        "parallel_runs": "up to 6",
        "peak_disk_gb": round(max(int(s[2]) for s in samples) / 1e9, 2),
        "disk_sampled_from": samples[0][0],
        "overlap_check_wall_hours": round(sum((end - start).total_seconds()
                                              for start, end in overlap_runs) / 3600, 2),
        "overlap_check_runs": len(overlap_runs),
        "committed_decodes_mb": round(sum(p.stat().st_size for p in DECODES_DIR.glob("*.gz"))
                                      / 1e6, 1),
    }


# --- the app outcome ---------------------------------------------------------------------------


def placed_words(item: PilotItem, phoneme_word) -> list[tuple]:
    """Every Muraja phoneme word the item recites, once, as ``(sampled word, site)``: the
    canonical run's words carried to Muraja's numbering (``phoneme_word(surah, ayah, printed
    word)``, ``word_map``), never re-inferred from the reference, so an item that starts
    mid-ayah is graded on its own words. A sampled word's carrier is the first letter of the
    first canonical word in that phoneme word."""
    from tadabur.word_sample import SampledWord
    from training.muraja_policy import MurajaSite

    surah, ayah = item.row.surah, item.row.ayah
    out, seen = [], set()
    for position, word_index in enumerate(item.reference.word_indices):
        word = phoneme_word(surah, ayah, word_index + 1)
        if word in seen:
            continue
        seen.add(word)
        out.append((SampledWord(
            site_id=f"{item.key}:w{word}", audio_filename=item.audio_filename, shard=-1,
            audio_sha256=item.mp3_sha256, reciter_id=cluster_of(item),
            surah_ayah=item.row.surah_ayah, segment_index=item.segment_index,
            start_sample=item.start_sample, end_sample=item.end_sample,
            reference=item.reference.reference, word_index=word_index,
            reference_index=item.reference.offsets[position], stratum=f"qud:{item.slug}",
            stratum_population=1), MurajaSite(surah, ayah, word, None)))
    return out


def placement(item: PilotItem, placed: Sequence[tuple]):
    """The item's session on Muraja's reference: its ayah, starting on its first word."""
    from training.muraja_policy import Placement

    return Placement(item.row.surah, item.row.ayah, placed[0][1].word,
                     {word.site_id: site for word, site in placed})


def app_outcome(items: Sequence[PilotItem], decodes: Mapping[str, Mapping[str, dict]],
                harness) -> dict:
    """Per model (its recorded checks) and scoring, every word's grade: words graded, flagged,
    the flags' causes, and the consonant flags' substitutions, expected → heard and by expected
    letter (:func:`training.word_flags.substitution_tally`: a word counts once per key)."""
    from training.device_logs import per_1000
    from training.muraja_policy import (
        NOT_GRADED,
        SCORINGS,
        SHIPPED_DEFAULT,
        UNPLACED,
        WORD_FLAGGED,
        Item,
        MurajaText,
        grade_placed,
    )
    from training.word_flags import CAUSES, SKIPPED, flag, substitution_tally

    scorings = [SHIPPED_DEFAULT, next(s for s in SCORINGS if s.name == "allowances_off")]
    text = MurajaText(harness.quran_db)
    placed = {item.key: placed_words(item, text.phoneme_word) for item in items}
    requests = [Item(f"{model}#{item.key}", tuple(w for w, _ in placed[item.key]),
                     tuple(decodes[model][item.key][CHECKS]))
                for model in MODELS for item in items]
    placements = {f"{model}#{item.key}": placement(item, placed[item.key])
                  for model in MODELS for item in items}
    graded = grade_placed(requests, placements, scorings, harness)
    out: dict = {"muraja_build": graded.build, "scorings": [s.as_dict() for s in scorings]}
    for model in MODELS:
        out[model] = {}
        for scoring in scorings:
            tally: Counter = Counter()
            causes: Counter = Counter()
            marks: Counter = Counter()
            by_mushaf: dict[str, Counter] = defaultdict(Counter)
            skipped_at: Counter = Counter()
            consonant_flags: list = []
            for item in items:
                last = len(placed[item.key]) - 1
                for position, (word, site) in enumerate(placed[item.key]):
                    grade = graded.grades[scoring.name][(f"{model}#{item.key}", word.site_id)]
                    tally["words"] += 1
                    if grade.outcome in (NOT_GRADED, UNPLACED):
                        tally[grade.outcome] += 1
                        continue
                    by_mushaf[item.slug]["graded"] += 1
                    if grade.outcome not in WORD_FLAGGED:
                        continue
                    tally["flagged"] += 1
                    by_mushaf[item.slug]["flagged"] += 1
                    reason = flag(grade, text.word_phonemes(site))
                    causes.update(reason.causes)
                    marks.update(set(reason.marks))
                    consonant_flags.append(reason.consonants)
                    if SKIPPED in reason.causes:
                        skipped_at["first word" if position == 0 else
                                   "last word" if position == last else "inside"] += 1
            graded_words = tally["words"] - tally[NOT_GRADED] - tally[UNPLACED]
            out[model][scoring.name] = {
                **dict(tally), "graded": graded_words,
                "per_1000": round(per_1000(tally["flagged"] / graded_words), 2)
                if graded_words else None,
                "causes": {c: causes[c] for c in CAUSES},
                "tashkeel_marks": dict(sorted(marks.items())),
                "skipped_where": dict(sorted(skipped_at.items())),
                "substitutions": substitution_tally(consonant_flags),
                "substitutions_by_expected": substitution_tally(consonant_flags,
                                                                by_expected=True),
                "per_mushaf": {s: {**dict(c), "per_1000": round(per_1000(c["flagged"] / c["graded"]), 2)}
                               for s, c in sorted(by_mushaf.items())},
            }
    return out


def device_log_rates() -> dict:
    """Muraja's own device sessions' false flags per 1,000 words (``training.device_logs``, #151),
    the teacher's sessions over all protocol eras and its 1 s-hop sessions alone, per scoring:
    with the explained words (skipped, re-read, never recited) left out of flags and words, the
    comparison #151 makes, and every graded word as a separate diagnostic."""
    from training.device_logs import (
        ALL_WORDS,
        REPORT_PATH as DEVICE_REPORT,
        TEACHER,
        TEACHER_ONE_SECOND,
        UNEXPLAINED,
    )

    groups = json.loads(DEVICE_REPORT.read_text(encoding="utf-8"))["groups"]
    return {group: {scoring: {view: groups[group][scoring][view]["false_flags"]
                              for view in (UNEXPLAINED, ALL_WORDS)}
                    for scoring in ("shipped_default", "allowances_off")}
            for group in (TEACHER, TEACHER_ONE_SECOND)}


def tadabur_word_sample() -> dict:
    """The Tadabur word sample's false flags on its recorded checks, per model and scoring,
    from the committed truth baseline (``truth_baseline/report.json``)."""
    from training.truth_baseline import REPORT_PATH as BASELINE_REPORT

    report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))["word_sample"]
    out = {}
    for model in MODELS:
        out[model] = {}
        for scoring in ("shipped_default", "allowances_off"):
            view = report["arms"][arm(model, CHECKS)][scoring]
            out[model][scoring] = {
                "correct_words": view["correct_words"],
                "false_flags": view["false_flags"]["words"],
                "rate": view["false_flags"]["point"],
                "interval": [view["false_flags"]["lower"], view["false_flags"]["upper"]],
                "causes": {c: v["words"] for c, v in view["causes"].items()},
            }
    return out


# --- running it ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pilot:
    """Both mushafs' items, every item's decodes per model, what was left out, and the
    models' identity records."""

    items: list[PilotItem]
    decodes: dict[str, dict[str, dict]]
    left_out: dict[str, dict]


def load_pilot() -> Pilot:
    from tadabur.waqf_segments import _uthmani_words, hafs_segment_reference

    reference_of = hafs_segment_reference()
    items: list[PilotItem] = []
    decodes: dict[str, dict[str, dict]] = {m: {} for m in MODELS}
    left_out = {}
    for slug in PILOT:
        rows = mushaf_metadata(slug, _shards(slug), QUD_REVISION)
        mushaf_decodes = {m: read_decodes(slug, m) for m in MODELS}
        built, left_out[slug] = build_items(slug, rows, mushaf_decodes["base"], _uthmani_words,
                                            reference_of)
        items += built
        for m in MODELS:
            decodes[m].update(mushaf_decodes[m])
    return Pilot(items, decodes, left_out)


def qud_exposures(decodes: Mapping[str, Mapping[str, dict]]) -> list:
    """Every decoded QUD recording (one ayah clip, whether or not it yielded an item) as the
    exposure use :data:`tadabur.exposure.QUD_CORRECT_RECITATION`, with the revision it was read
    at and its MP3's checksum."""
    from tadabur.exposure import QudExposure

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    reciter = {m["slug"]: m["reciter_id"] for m in selection["mushafs"]}
    return sorted({QudExposure(QUD_REVISION, slug, line["surah_ayah"], line["mp3_sha256"],
                               reciter[slug])
                   for slug in PILOT for line in decodes[slug].values()})


def build(workers: int) -> dict:
    """Carriers, rates, classes, truth sites, the worklist and the exposure use."""
    from tadabur.exposure import QUD_CORRECT_RECITATION, write_qud_use

    pilot = load_pilot()
    by_place = {(i.audio_filename, i.start_sample): i for i in pilot.items}
    rows = site_rows([clip_of(i, pilot.decodes) for i in pilot.items], workers)
    slug_of = {key: item.slug for key, item in by_place.items()}

    def of(slug: str) -> list[SiteRow]:
        return [r for r in rows if slug_of[(r.site.audio_filename, r.site.start_sample)] == slug]

    for slug in PILOT:
        write_truth_sites(truth_sites(of(slug), by_place), SITES_DIR / f"{slug}.jsonl.gz")
    work = worklist(rows, by_place)
    write_text_atomically(WORKLIST_PATH, "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in work))
    write_qud_use(QUD_CORRECT_RECITATION, qud_exposures(
        {slug: read_decodes(slug, "base") for slug in PILOT}))
    carriers = {carrier_key(r): r for r in rows}
    return {
        "issue": 144,
        "qud_revision": QUD_REVISION,
        "rule_arms": list(RULE_ARMS),
        "items": {slug: {
            "ayat": len({i.row.surah_ayah for i in pilot.items if i.slug == slug}),
            "items": sum(i.slug == slug for i in pilot.items),
            "hours": round(sum(i.end_sample - i.start_sample for i in pilot.items
                               if i.slug == slug) / TARGET_SAMPLE_RATE / 3600, 2),
            **{k: (v if isinstance(v, int) else len(v)) for k, v in pilot.left_out[slug].items()},
            "unmapped_ayat_list": pilot.left_out[slug]["unmapped_ayat"],
            "phonetizer_unsupported_list": pilot.left_out[slug]["phonetizer_unsupported"],
        } for slug in PILOT},
        "classes": {slug: dict(Counter(classify(r) for k, r in carriers.items()
                                       if slug_of[(k[0], k[1])] == slug)) for slug in PILOT},
        "directions": {"pooled": direction_counts(rows),
                       **{slug: direction_counts(of(slug)) for slug in PILOT}},
        "rates": {slug: gap_report(of(slug), APP_PROTOCOLS, SPANS) for slug in PILOT},
        "listening": listening_estimate(work),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge", help="gather the GPU box's per-shard decodes")
    merge.add_argument("--decodes-dir", type=Path, required=True)
    merge.add_argument("--logs-dir", type=Path, required=True)
    merge.add_argument("--gpu", default="NVIDIA GeForce RTX 5060 Ti")
    built = commands.add_parser("build", help="carriers, rates, sites, worklist, exposure")
    built.add_argument("--workers", type=int, default=8)
    commands.add_parser("app", help="the app outcome through the Muraja harness")
    commands.add_parser("doc", help="render docs/qud-pilot.md from the report")
    args = parser.parse_args()

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    if args.command == "merge":
        expected = {slug: sum(len(r.segments) for r in mushaf_metadata(
            slug, _shards(slug), QUD_REVISION)) for slug in PILOT}
        report["models"] = {f"{slug}/{m}": merge_decodes(args.decodes_dir, slug, m,
                                                         expected[slug])
                            for slug in PILOT for m in MODELS}
        report["compute"] = compute_record(args.decodes_dir, args.logs_dir, args.gpu)
    elif args.command == "build":
        report.update(build(args.workers))
    elif args.command == "app":
        from training.muraja_policy import require_harness

        pilot = load_pilot()
        report["app"] = app_outcome(pilot.items, pilot.decodes, require_harness())
        report["tadabur_word_sample"] = tadabur_word_sample()
        report["device_logs"] = device_log_rates()
    else:
        from training.qud_pilot_doc import render

        write_text_atomically(DOC_PATH, render(report))
        print(f"wrote {DOC_PATH}")
        return
    write_text_atomically(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=1,
                                                  sort_keys=True) + "\n")
    print(f"wrote {REPORT_PATH}")


def _shards(slug: str) -> list[str]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    (mushaf,) = [m for m in selection["mushafs"] if m["slug"] == slug]
    return mushaf["shards"]


if __name__ == "__main__":
    main()

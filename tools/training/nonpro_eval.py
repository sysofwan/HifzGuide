"""The non-professional eval pilot, measured without listening (#156).

Recitation by **non-professionals**, which every earlier unselected evaluation lacked (QUD,
#144, is professional; Tadabur is mostly professional and is ``h448``'s training audio):
Tarteel v1 app users, RetaSy's volunteers and qdat's amateur reciters
(:mod:`tadabur.nonpro_sources`). Each selected clip is decoded by the base teacher and ``h448``
under the truth baseline's three protocols (:mod:`training.nonpro_decode`) and read against the
**mushaf** by :mod:`training.external_corpus`, the path QUD's pilot takes too:

* **Items**: one per clip, the whole clip, its realized reference the phonetizer's output over
  the canonical words it recites (Tarteel v1: its ``sura``/``ayah``, the whole ayah; RetaSy: the
  run its text names; qdat: :data:`~tadabur.nonpro_sources.QDAT_PASSAGE`, checked against the
  card's other wording on the audio, :func:`passage_check`), last word in waqf. A clip whose
  words the phonetizer cannot handle is listed and left out.
* **Sites, worklists, rates**: #144's rule per target-pair direction and in-word geminate (truth
  sites, the worklist, the clustered rates), and per every consonant (counted, with its own
  worklist). Only RetaSy's ``correct`` clips and every Tarteel v1 and qdat clip are presumed
  correct; RetaSy's ``in_correct`` clips are read only beside its label (:func:`label_contrast`).
* **App outcome**: every word through Muraja's engine (protocol 3) per model and scoring, with
  causes and substitutions, set beside QUD (#144), Muraja's device sessions (#151), the owner's
  ``h448`` sessions (#153) and the Tadabur word sample (#90) (:func:`comparison`). For qdat also
  per word of the passage, per reciter and split by its tajweed labels (which mark tajweed
  lapses, not consonant errors); for RetaSy, against its human label.
* **qdat_bench** (:func:`bench_report`): each model's decode, both protocols, against the
  bench's expert ``phonetic_transcript`` letter by letter (consonant, haraka, shaddah); the
  transcript against the mushaf; and the app outcome on its 159 clips, each flagged word placed
  against where the experts heard the reciter leave the mushaf.

The recitation is unverified: every flag is "a flag on presumed-correct non-professional
recitation", never a measured false-flag rate. Nothing here trains, sets a threshold or touches
the sealed panel.

**What is committed.** Tarteel v1 (CC BY 4.0) and qdat (MIT): the merged decodes, truth sites
and both worklists, under ``tadabur/nonpro_eval/<dataset>/``. RetaSy (licence unknown) and
qdat_bench (licence unstated): aggregates and clip ids only; their per-clip outputs are written
under ``tadabur/nonpro_eval/restricted/`` (ignored by git). ``report.json`` holds aggregates;
the exposure use ``nonpro.evaluation_only`` every decoded clip and every bench clip.

Usage (from ``tools/``; torch-free; ``app`` and ``bench`` need the Muraja harness)::

  python -m training.nonpro_eval merge --decodes-dir <the GPU box's decodes> --logs-dir <its logs>
  python -m training.nonpro_eval build
  python -m training.nonpro_eval app
  python -m training.nonpro_eval doc
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.contrast_attribution import align_on_carriers
from tadabur.listening_session import held_geminates, run_start
from tadabur.nonpro_sources import (
    CORRECT,
    DATASETS,
    DECODED,
    QDAT,
    QDAT_ALTERNATIVE,
    QDAT_BENCH,
    QDAT_PASSAGE,
    QDAT_TAJWEED,
    RETASY,
    Dataset,
    LeftOut,
    metadata,
    select,
)
from tadabur.qud_text import ItemReference, Unmapped, item_reference
from tadabur.truth_sites import CONSONANTS, write_truth_sites
from training.carrier_windowing import APP_PROTOCOLS, MODELS, SPANS, arm, gap_report, site_rows
from training.external_corpus import (
    RULE_ARMS,
    WordGrade,
    app_scorings,
    carrier_key,
    classify,
    clip_of,
    consonant_carriers,
    consonant_counts,
    consonant_worklist,
    device_log_rates,
    direction_counts,
    flag_view,
    listening_estimate,
    merge_decode_files,
    read_decode_file,
    tadabur_word_sample,
    truth_sites,
    word_grades,
    worklist,
    write_jsonl,
)
from training.site_outcomes import pair_outcome, shaddah_outcome
from training.tashkeel_eval import SHORT_VOWELS, carrier_readings, write_text_atomically

NONPRO_DIR = Path(__file__).parent.parent / "tadabur" / "nonpro_eval"
#: Where a dataset whose licence allows only aggregates keeps its per-clip outputs (ignored
#: by git).
RESTRICTED_DIR = NONPRO_DIR / "restricted"
REPORT_PATH = NONPRO_DIR / "report.json"
OVERLAP_PATH = NONPRO_DIR / "overlap.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "nonpro-pilot.md"
QUD_REPORT = Path(__file__).parent.parent / "tadabur" / "qud_eval" / "pilot" / "report.json"
OWNER_REPORT = Path(__file__).parent.parent / "device_logs" / "owner_report.json"
#: The arms qdat_bench's transcript is compared with: both models, decoded whole and streamed.
BENCH_ARMS = tuple(arm(m, p) for m in MODELS for p in (SPANS, "stream_b0"))
AGREE, OTHER, NONE = "agree", "other", "none"
LETTER_KINDS = ("consonant", "haraka", "shaddah")


def part_dir(dataset: Dataset) -> Path:
    """Where a dataset's per-clip outputs go: committed when its licence allows."""
    return (NONPRO_DIR if dataset.open else RESTRICTED_DIR) / dataset.name


def decodes_file(dataset: Dataset, model: str) -> Path:
    return part_dir(dataset) / f"decodes.{model}.jsonl.gz"


# --- items ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipItem:
    """One clip as an external-corpus item (:class:`training.external_corpus.CorpusItem`)."""

    key: str
    slug: str
    audio_filename: str
    surah: int
    ayah: int
    start_sample: int
    end_sample: int
    audio_sha256: str
    reference: ItemReference
    reciter: str
    cluster: int
    labels: dict = field(default_factory=dict, compare=False)

    @property
    def source(self) -> str:
        """The dataset's name is its truth-site source (``tadabur.truth_sites.NONPRO_SOURCES``)."""
        return self.slug

    @property
    def surah_ayah(self) -> str:
        return f"{self.surah}:{self.ayah}"

    @property
    def provenance(self) -> dict:
        return {"audio_filename": self.audio_filename, "audio_sha256": self.audio_sha256}

    def excerpt(self, reference_index: int) -> tuple[int, int]:
        """The whole clip: no corpus here times its words, and a clip is one ayah or less."""
        return self.start_sample, self.end_sample


def clusters(dataset: Dataset, lines: Sequence[dict]) -> dict[str, int]:
    """Each clip's bootstrap cluster: its reciter where the dataset names one, else (Tarteel v1)
    its surah, as one app user's clips most likely share a surah."""
    if dataset.name == "tarteel_v1":
        return {line["key"]: int(line["surah_ayah"].split(":")[0]) for line in lines}
    reciters = sorted({line["reciter"] for line in lines})
    return {line["key"]: reciters.index(line["reciter"]) for line in lines}


def build_items(dataset: Dataset, lines: Sequence[dict], reference_of
                ) -> tuple[list[ClipItem], list[str]]:
    """Every decoded clip with a reference, and the clips the phonetizer cannot handle."""
    from tadabur.waqf_segments import _uthmani_words

    cluster = clusters(dataset, lines)
    items, unsupported = [], []
    for line in sorted(lines, key=lambda line: line["key"]):
        surah, ayah = map(int, line["surah_ayah"].split(":"))
        try:
            reference = item_reference((line["word_from"], line["word_to"]),
                                       _uthmani_words(line["surah_ayah"]), reference_of)
        except (KeyError, IndexError, Unmapped):
            unsupported.append(line["key"])
            continue
        items.append(ClipItem(line["key"], dataset.name, line["audio_filename"], surah, ayah,
                              line["start_sample"], line["end_sample"], line["audio_sha256"],
                              reference, line["reciter"], cluster[line["key"]], line["labels"]))
    return items, unsupported


@dataclass(frozen=True)
class Corpus:
    """Every decoded dataset's clips (``lines``, the base model's decode records by key), its
    items, the clips the phonetizer cannot handle, and every decode per model."""

    lines: dict[str, dict[str, dict]]
    items: dict[str, list[ClipItem]]
    unsupported: dict[str, list[str]]
    decodes: dict[str, dict[str, dict]]

    def presumed_correct(self, name: str) -> list[ClipItem]:
        """The items the sites and the app rates are read on: RetaSy's ``correct`` clips only."""
        return [i for i in self.items[name]
                if name != RETASY.name or i.labels["final_label"] == CORRECT]


def load_corpus() -> Corpus:
    from tadabur.waqf_segments import hafs_segment_reference

    reference_of = hafs_segment_reference()
    lines, items, unsupported, decodes = {}, {}, {}, {m: {} for m in MODELS}
    for dataset in DECODED:
        by_model = {m: read_decode_file(decodes_file(dataset, m)) for m in MODELS}
        lines[dataset.name] = by_model["base"]
        items[dataset.name], unsupported[dataset.name] = build_items(
            dataset, list(by_model["base"].values()), reference_of)
        for m in MODELS:
            if clash := set(decodes[m]) & set(by_model[m]):
                raise ValueError(f"{dataset.name} reuses another dataset's keys: {sorted(clash)[:3]}")
            decodes[m].update(by_model[m])
    return Corpus(lines, items, unsupported, decodes)


# --- which passage qdat recites -------------------------------------------------------------


def passage_check(items: Sequence[ClipItem], decodes: Mapping[str, Mapping[str, dict]]) -> dict:
    """qdat's audio against both wordings its cards give (:data:`QDAT_PASSAGE`, 5:109's words
    8-15; :data:`QDAT_ALTERNATIVE`, 2:32): per clip, the edit distance of the base teacher's
    whole-clip decode to each realized reference, normalized; how many clips are nearer each,
    and the median distances."""
    import statistics

    from tadabur.normalization import normalize_phonemes
    from tadabur.waqf_segments import _uthmani_words, hafs_segment_reference
    from training.decode_evalset import levenshtein

    reference_of = hafs_segment_reference()

    def reference(passage) -> str:
        surah, ayah, word_from, word_to = passage
        words = _uthmani_words(f"{surah}:{ayah}")[word_from - 1:word_to]
        return normalize_phonemes(reference_of(words)[0]).normalized

    candidates = {"5:109 words 8-15": reference(QDAT_PASSAGE),
                  "2:32": reference(QDAT_ALTERNATIVE)}
    distances: dict[str, list[float]] = {name: [] for name in candidates}
    nearer: Counter = Counter()
    for item in items:
        decode = normalize_phonemes(decodes["base"][item.key][SPANS]).normalized
        per = {name: levenshtein(decode, ref) / len(ref) for name, ref in candidates.items()}
        for name, d in per.items():
            distances[name].append(d)
        nearer[min(per, key=per.get)] += 1
    return {"clips": len(items), "nearer": dict(nearer),
            "median_distance": {n: round(statistics.median(d), 3) for n, d in distances.items()}}


# --- build -------------------------------------------------------------------------------------


def exposures(corpus: Corpus, bench_ids: Sequence[str]) -> list:
    """Every decoded clip (whether or not it yielded an item), and every qdat_bench clip (the
    qdat clip of its id), as the exposure use :data:`tadabur.exposure.NONPRO_EVALUATION`."""
    from tadabur.exposure import ClipExposure

    def exposure(dataset: Dataset, line: dict) -> ClipExposure:
        return ClipExposure(dataset.repo, dataset.revision, line["audio_filename"],
                            line["surah_ayah"], line["audio_sha256"], line["reciter"] or "unknown")

    qdat = {line["audio_filename"]: line for line in corpus.lines[QDAT.name].values()}
    return ([exposure(d, line) for d in DECODED for line in corpus.lines[d.name].values()]
            + [exposure(QDAT_BENCH, qdat[clip]) for clip in bench_ids])


def build_dataset(dataset: Dataset, corpus: Corpus, workers: int) -> dict:
    """One dataset's carriers, rates, classes, truth sites and both worklists, over its
    presumed-correct items."""
    items = corpus.presumed_correct(dataset.name)
    by_place = {(i.audio_filename, i.start_sample): i for i in items}
    rows = site_rows([clip_of(i, corpus.decodes) for i in items], workers)
    out_dir = part_dir(dataset)
    write_truth_sites(truth_sites(rows, by_place), out_dir / "truth_sites.jsonl.gz")
    work = worklist(rows, by_place)
    write_jsonl(out_dir / "worklist.jsonl", work)
    carriers = consonant_carriers(items, corpus.decodes, workers)
    consonant_work = consonant_worklist(carriers, {i.key: i for i in items})
    write_jsonl(out_dir / "consonant_worklist.jsonl", consonant_work)
    unique = {carrier_key(r): r for r in rows}
    labels = Counter(i.labels.get("final_label") for i in corpus.items[dataset.name])
    return {
        "items": {
            "clips_decoded": len(corpus.lines[dataset.name]),
            "phonetizer_unsupported": len(corpus.unsupported[dataset.name]),
            "items": len(corpus.items[dataset.name]),
            **({"labels": dict(labels)} if dataset is RETASY else {}),
            "presumed_correct": len(items),
            "hours": round(sum(i.end_sample - i.start_sample for i in items)
                           / TARGET_SAMPLE_RATE / 3600, 2),
            "ayat": len({i.surah_ayah for i in items}),
            "surahs": len({i.surah for i in items}),
            "reciters": len({i.reciter for i in items} - {""}),
            "words": sum(len(i.reference.word_indices) for i in items),
        },
        "classes": dict(Counter(classify(r) for r in unique.values())),
        "directions": direction_counts(rows),
        "consonants": consonant_counts(carriers),
        "rates": gap_report(rows, APP_PROTOCOLS, SPANS),
        "listening": {"pairs_and_geminates": listening_estimate(work),
                      "consonants": listening_estimate(consonant_work)},
    }


def build(workers: int) -> dict:
    """Every dataset's sites, worklists and rates, qdat's passage check, the bench's letters,
    and the exposure use."""
    from tadabur.exposure import NONPRO_EVALUATION, write_clip_use

    corpus = load_corpus()
    bench = bench_transcripts()
    write_clip_use(NONPRO_EVALUATION, exposures(corpus, sorted(bench)))
    return {
        "issue": 156,
        "rule_arms": list(RULE_ARMS),
        "datasets": {d.name: {"repo": d.repo, "revision": d.revision, "licence": d.licence,
                              "per_clip_outputs_committed": d.open}
                     for d in DATASETS.values()},
        "passage": passage_check(corpus.items[QDAT.name], corpus.decodes),
        "sources": {d.name: build_dataset(d, corpus, workers) for d in DECODED},
        "bench": {"letters": bench_letters(corpus, bench)},
    }


# --- qdat_bench: letters --------------------------------------------------------------------


def bench_transcripts() -> dict[str, str]:
    """qdat_bench's expert ``phonetic_transcript`` per clip id (read, never committed)."""
    return {row["id"]: row["phonetic_transcript"] for row in metadata(QDAT_BENCH)}


def _judge(expected: str, committed: str | None) -> str:
    return NONE if committed is None else AGREE if committed == expected else OTHER


def letter_outcomes(decode: str, reference: str) -> dict[str, list[tuple[int, str]]]:
    """``decode`` read against ``reference`` letter by letter, each outcome :data:`AGREE`,
    :data:`OTHER` or :data:`NONE` (committed nothing) at its reference index:

    * **consonant**: the first letter of every run of a consonant, by the pair-site rule
      (:func:`training.site_outcomes.pair_outcome`);
    * **haraka**: every consonant a fatha, damma or kasra follows, by the tashkeel-site rule
      (the carrier reproduced, one distinct haraka on it; :func:`training.tashkeel_eval.carrier_readings`);
    * **shaddah**: every geminate, held or not (:func:`training.site_outcomes.shaddah_outcome`).
    """
    from tadabur.truth_sites import HELD

    alignment = align_on_carriers(decode, reference)
    readings = carrier_readings(decode, reference)

    def haraka(i: int) -> str | None:
        reading = readings.get(i)
        if reading is None or reading.decoded != reference[i]:
            return None
        marks = list(dict.fromkeys(reading.marks))
        return marks[0] if len(marks) == 1 else None if not marks else "multiple"

    return {
        "consonant": [(i, _judge(c, pair_outcome(alignment, i).committed))
                      for i, c in enumerate(reference)
                      if c in CONSONANTS and run_start(reference, i) == i],
        "haraka": [(i, _judge(reference[i + 1], haraka(i))) for i, c in enumerate(reference[:-1])
                   if c in CONSONANTS and reference[i + 1] in SHORT_VOWELS],
        "shaddah": [(i, _judge(HELD, shaddah_outcome(alignment, i).committed))
                    for i in held_geminates(reference)],
    }


def _tally(outcomes: Sequence[dict[str, list[tuple[int, str]]]]) -> dict:
    out = {}
    for kind in LETTER_KINDS:
        counts = Counter(o for clip in outcomes for _, o in clip[kind])
        committed = counts[AGREE] + counts[OTHER]
        total = committed + counts[NONE]
        out[kind] = {**{k: counts[k] for k in (AGREE, OTHER, NONE)},
                     "agreement": round(counts[AGREE] / total, 4) if total else None,
                     "agreement_committed": round(counts[AGREE] / committed, 4)
                     if committed else None}
    return out


def bench_items(corpus: Corpus, bench: Mapping[str, str]) -> list[ClipItem]:
    """The qdat items of the bench's clips (the same audio, so the same decodes)."""
    by_clip = {i.audio_filename: i for i in corpus.items[QDAT.name]}
    return [by_clip[clip] for clip in sorted(bench)]


def bench_letters(corpus: Corpus, bench: Mapping[str, str]) -> dict:
    """Each arm of :data:`BENCH_ARMS` against the experts' transcript, letter by letter; and the
    transcript itself against the mushaf (where the experts heard the reciter leave it)."""
    items = bench_items(corpus, bench)
    out = {"clips": len(items), "arms": {}}
    for a in BENCH_ARMS:
        model, protocol = a.split("/")
        out["arms"][a] = _tally([letter_outcomes(corpus.decodes[model][i.key][protocol],
                                                 bench[i.audio_filename]) for i in items])
    against_mushaf = [letter_outcomes(bench[i.audio_filename], i.reference.reference)
                      for i in items]
    out["transcript_vs_mushaf"] = {
        **_tally(against_mushaf),
        "clips_leaving_the_mushaf": {kind: sum(any(o != AGREE for _, o in clip[kind])
                                               for clip in against_mushaf)
                                     for kind in LETTER_KINDS},
    }
    return out


def bench_departures(item: ClipItem, transcript: str) -> set[int]:
    """The canonical words (0-based) of a bench clip where the experts' transcript leaves the
    mushaf at a consonant, a haraka or a geminate."""
    import bisect

    outcomes = letter_outcomes(transcript, item.reference.reference)
    offsets = item.reference.offsets
    return {item.reference.word_indices[bisect.bisect_right(offsets, i) - 1]
            for kind in LETTER_KINDS for i, o in outcomes[kind] if o != AGREE}


# --- the app outcome ---------------------------------------------------------------------------


def _short_view(grades: Sequence[WordGrade]) -> dict:
    """A subgroup's words, flags, rate, causes and substitutions (:func:`flag_view`)."""
    view = flag_view(grades)
    return {k: view[k] for k in ("graded", "per_1000", "causes", "substitutions")} | {
        "flagged": view.get("flagged", 0)}


def by_model_scoring(grades: Sequence[WordGrade], view) -> dict:
    groups: dict[tuple[str, str], list[WordGrade]] = defaultdict(list)
    for g in grades:
        groups[(g.model, g.scoring)].append(g)
    return {m: {s.name: view(groups[(m, s.name)]) for s in app_scorings()} for m in MODELS}


def qdat_views(grades: Sequence[WordGrade]) -> dict:
    """qdat per word of the passage, per reciter, and split by its tajweed labels."""
    def per_word(group: Sequence[WordGrade]) -> list[dict]:
        words: dict[int, list[WordGrade]] = defaultdict(list)
        for g in group:
            words[g.word].append(g)
        return [{"word": w, "canonical_word": g[0].word_index + 1,
                 "text": g[0].item.reference.words[g[0].word_index - g[0].item.reference.word_indices[0]],
                 **_short_view(g)} for w, g in sorted(words.items())]

    def per_reciter(group: Sequence[WordGrade]) -> dict:
        flagged = [g for g in group if g.flagged]

        def spread(keys_of, names: tuple[str, ...]) -> list[dict]:
            reciters: dict[tuple, set[str]] = defaultdict(set)
            words: Counter = Counter()
            for g in flagged:
                for key in keys_of(g):
                    reciters[key].add(g.item.reciter)
                    words[key] += 1
            return [{**dict(zip(names, key)), "reciters": len(r), "words": words[key]}
                    for key, r in sorted(reciters.items(), key=lambda kv: (-len(kv[1]), kv[0]))]

        return {
            "reciters": len({g.item.reciter for g in group}),
            "reciters_flagged": len({g.item.reciter for g in flagged}),
            "flags_per_reciter": dict(sorted(Counter(Counter(g.item.reciter for g in flagged)
                                                     .values()).items())),
            "by_cause": spread(lambda g: {(c,) for c in g.reason.causes}, ("cause",)),
            "by_substitution": spread(lambda g: set(g.reason.consonants),
                                      ("cause", "substitution")),
        }

    def per_label(group: Sequence[WordGrade]) -> dict:
        return {label: {str(value): _short_view([g for g in group if g.item.labels[label] == value])
                        for value in sorted({g.item.labels[label] for g in group})}
                for label in QDAT_TAJWEED}

    return {"per_word": by_model_scoring(grades, per_word),
            "per_reciter": by_model_scoring(grades, per_reciter),
            "per_tajweed_label": by_model_scoring(grades, per_label)}


def label_contrast(grades: Sequence[WordGrade]) -> dict:
    """RetaSy's app outcome beside its human clip label: per label, clips, clips with a flagged
    word, words graded and flagged."""
    def view(group: Sequence[WordGrade]) -> dict:
        out = {}
        for label in sorted({g.item.labels["final_label"] for g in group}):
            mine = [g for g in group if g.item.labels["final_label"] == label]
            clips = {g.item.key for g in mine}
            flagged = {g.item.key for g in mine if g.flagged}
            out[label] = {"clips": len(clips), "clips_flagged": len(flagged),
                          **_short_view(mine)}
        return out

    return by_model_scoring(grades, view)


def bench_app(grades: Sequence[WordGrade], corpus: Corpus, bench: Mapping[str, str]) -> dict:
    """The app outcome on the bench's clips, each word placed against the experts: a flag on a
    word where the transcript leaves the mushaf, or on one where it does not (a false flag by
    the experts' ear)."""
    items = {i.key: i for i in bench_items(corpus, bench)}
    departures = {k: bench_departures(i, bench[i.audio_filename]) for k, i in items.items()}
    mine = [g for g in grades if g.item.key in items]

    def view(group: Sequence[WordGrade]) -> dict:
        def departs(g: WordGrade) -> bool:
            return g.word_index in departures[g.item.key]

        return {**_short_view(group),
                "words_where_experts_heard_a_departure": sum(g.graded and departs(g) for g in group),
                "flagged_where_experts_heard_a_departure": sum(g.flagged and departs(g)
                                                               for g in group),
                "flagged_where_experts_heard_the_mushaf": sum(g.flagged and not departs(g)
                                                              for g in group),
                "graded_where_experts_heard_the_mushaf": sum(g.graded and not departs(g)
                                                             for g in group)}

    return by_model_scoring(mine, view)


def comparison(app: Mapping[str, dict]) -> dict:
    """Per model and scoring, flags per 1,000 graded words here (per dataset) beside QUD (#144),
    Muraja's device sessions (#151: the teacher, explained words left out), the owner's ``h448``
    sessions (#153: the owner's mistakes left out) and the Tadabur word sample (#90)."""
    qud = json.loads(QUD_REPORT.read_text(encoding="utf-8"))["app"]
    owner = json.loads(OWNER_REPORT.read_text(encoding="utf-8"))["rates"]
    device = device_log_rates()
    sample = tadabur_word_sample()

    def rate(flags: int, words: int) -> dict:
        return {"flags": flags, "words": words,
                "per_1000": round(1000 * flags / words, 2) if words else None}

    out = {}
    for model in MODELS:
        out[model] = {}
        for scoring in (s.name for s in app_scorings()):
            out[model][scoring] = {
                **{name: rate(v[model][scoring].get("flagged", 0), v[model][scoring]["graded"])
                   for name, v in app.items()},
                "qud_144": rate(qud[model][scoring]["flagged"], qud[model][scoring]["graded"]),
                "device_151_teacher": rate(**{"flags": device["teacher"][scoring]["unexplained"]
                                              ["flags"], "words": device["teacher"][scoring]
                                              ["unexplained"]["words"]}) if model == "base" else None,
                "owner_153_h448": rate(owner[scoring]["unexplained"]["false_flags"]["flags"],
                                       owner[scoring]["unexplained"]["false_flags"]["words"])
                if model == "h448" else None,
                "word_sample_90": {**rate(sample[model][scoring]["false_flags"],
                                          sample[model][scoring]["correct_words"]),
                                   "weighted_per_1000": round(1000 * sample[model][scoring]["rate"], 1)},
            }
    return out


def app() -> dict:
    """The app outcome per dataset, qdat's and RetaSy's views, the bench's and the comparison."""
    from training.muraja_policy import require_harness

    corpus = load_corpus()
    bench = bench_transcripts()
    items = [i for d in DECODED for i in corpus.items[d.name]]
    build, grades = word_grades(items, corpus.decodes, require_harness())
    by_dataset: dict[str, list[WordGrade]] = defaultdict(list)
    for g in grades:
        by_dataset[g.item.slug].append(g)
    presumed = {d.name: {i.key for i in corpus.presumed_correct(d.name)} for d in DECODED}
    outcome = {name: by_model_scoring([g for g in group if g.item.key in presumed[name]],
                                      flag_view)
               for name, group in sorted(by_dataset.items())}
    return {
        "muraja_build": build,
        "scorings": [s.as_dict() for s in app_scorings()],
        "outcome": outcome,
        "qdat": qdat_views(by_dataset[QDAT.name]),
        "retasy_labels": label_contrast(by_dataset[RETASY.name]),
        "bench": bench_app(by_dataset[QDAT.name], corpus, bench),
        "comparison": comparison(outcome),
    }


# --- merging the GPU box's decodes, and what it cost -------------------------------------------


def selection(dataset: Dataset) -> dict:
    """What the dataset holds and what was selected, counted from its metadata (no audio)."""
    rows = metadata(dataset)
    chosen = [select(dataset, row) for row in rows]
    left_out = Counter(c.reason for c in chosen if isinstance(c, LeftOut))
    return {"rows": len(rows), "selected": len(rows) - sum(left_out.values()),
            "left_out": dict(sorted(left_out.items()))}


def merge(decodes_dir: Path) -> dict:
    """Every decoded dataset's per-shard decodes as one file per model, each checked against
    the count its selection gives; the models' identity records."""
    out = {"selection": {}, "models": {}}
    for dataset in DECODED:
        out["selection"][dataset.name] = selection(dataset)
        for model in MODELS:
            out["models"][f"{dataset.name}/{model}"] = merge_decode_files(
                sorted(decodes_dir.glob(f"{dataset.name}.*.{model}.jsonl")),
                decodes_dir / f"{dataset.name}.{model}.identity.json",
                decodes_file(dataset, model), out["selection"][dataset.name]["selected"])
    return out


def compute_record(decodes_dir: Path, logs_dir: Path, gpu: str) -> dict:
    """What the decode cost: the GPU lock's wall time (the launcher's start to the last decode
    written), the audio decoded, the shard runs, and the peak disk of this issue's scratch
    directory (sampled each minute)."""
    from datetime import datetime

    timings = [json.loads(line) for path in sorted(decodes_dir.glob("*.timing.jsonl"))
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = datetime.fromisoformat((logs_dir / "decode_start").read_text(encoding="utf-8")
                                   .strip().replace("Z", "+00:00"))
    end = max(path.stat().st_mtime for path in decodes_dir.glob("*.jsonl"))
    samples = [line.split() for line in (logs_dir / "df.log").read_text(encoding="utf-8")
               .splitlines() if len(line.split()) == 3]
    return {
        "gpu": gpu,
        "gpu_hours": round((end - start.timestamp()) / 3600, 2),
        "audio_hours": round(sum(t["audio_seconds"] for t in timings) / 3600, 2),
        "shard_runs_completed": len(timings),
        "parallel_runs": 5,
        "peak_scratch_gb": round(max(int(s[2]) for s in samples) / 1e9, 2),
        "disk_sampled_from": samples[0][0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    merged = commands.add_parser("merge", help="gather the GPU box's per-shard decodes")
    merged.add_argument("--decodes-dir", type=Path, required=True)
    merged.add_argument("--logs-dir", type=Path, required=True)
    merged.add_argument("--gpu", default="NVIDIA GeForce RTX 5060 Ti")
    built = commands.add_parser("build", help="sites, worklists, rates, bench letters, exposure")
    built.add_argument("--workers", type=int, default=8)
    commands.add_parser("app", help="the app outcome through the Muraja harness")
    commands.add_parser("doc", help="render docs/nonpro-pilot.md from the report")
    args = parser.parse_args()

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    if args.command == "merge":
        report.update(merge(args.decodes_dir))
        report["compute"] = compute_record(args.decodes_dir, args.logs_dir, args.gpu)
    elif args.command == "build":
        report.update(build(args.workers))
    elif args.command == "app":
        report["app"] = app()
    else:
        from training.nonpro_eval_doc import render

        overlap = json.loads(OVERLAP_PATH.read_text(encoding="utf-8"))
        write_text_atomically(DOC_PATH, render(report, overlap))
        print(f"wrote {DOC_PATH}")
        return
    write_text_atomically(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=1,
                                                  sort_keys=True) + "\n")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

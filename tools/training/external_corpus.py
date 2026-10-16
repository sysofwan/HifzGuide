"""An external corpus measured without listening: carriers, disagreements, the app outcome.

An **external corpus** is recitation from outside Tadabur that neither the base teacher nor
``h448`` has heard: QUD's professional mushafs (#144, :mod:`training.qud_pilot`) and the
non-professional app-user corpora (#156, :mod:`training.nonpro_eval`). Each is decoded by
:mod:`training.external_decode` under the truth baseline's three protocols (``spans``,
``stream_b0`` and the recorded ``checks``) and read here against the **mushaf**, the same way
for every corpus. A corpus hands this module its items (:class:`CorpusItem`); everything a
corpus differs in (where its audio comes from, how its items cluster, how an excerpt is cut)
is the item's.

**Carriers and rates.** Every target-pair carrier (D9, eighteen directions) and in-word geminate
of every item, read with :mod:`training.carrier_windowing`'s rules and rates
(:mod:`training.direction_table`: ``mushaf``, ``swap``, ``other``) per model and protocol
(:func:`clip_of`).

**Sites (#144's rule, ADR-0011 §3).** A carrier is **agreed** when the base teacher decoding the
item whole and ``h448`` streamed at b = 0 (the two decodes Ship A's frame reads, :data:`RULE_ARMS`)
both commit the mushaf's letter (both hold the geminate): a truth site with ``heard`` = the
mushaf's letter and ``assumes_competent_reciter``. It is a **disagreement** when either commits
something else: a ``pending`` site, and one row of the worklist, which is **not queued** for
listening. Where neither disagrees but one commits nothing at the carrier it is **abstained**:
counted, and kept out of the site files. A carrier of a letter in two pairs is a site per
direction (:func:`truth_sites`, :func:`worklist`).

**Every consonant (option B, #155).** The same rule at the first letter of every run of a
consonant (:func:`consonant_carriers`), whatever its pair: counted per letter, with what each rule
arm committed instead at a disagreement (:func:`consonant_counts`), and each disagreement a row of
its own worklist (:func:`consonant_worklist`). A truth site names a mark under test, and "this
consonant" is none, so these carriers are counted and listed, never written as sites.

**App outcome.** Each item's recorded check sequence, per model, is replayed through Muraja's
engine (:mod:`training.muraja_policy`) under the shipped default and with ADR-0012's allowances
off, and every word of the item is graded (:func:`word_grades`); a flagged word's causes are
:mod:`training.word_flags`'. The recitation is presumed correct, so every flag is presumed false
(:func:`app_outcome`).
"""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.contrast_attribution import align_on_carriers
from tadabur.listening_session import ANSWER_SECONDS, PLAYS_PER_SITE, run_start
from tadabur.qud_text import ItemReference
from tadabur.truth_sites import CONSONANTS, PENDING, SHADDAH, TruthSite
from training.carrier_windowing import (
    MODELS,
    SPANS,
    Clip,
    arm,
    geminate_positions,
    pair_positions,
)
from training.decoding import committed_text
from training.direction_table import direction
from training.site_outcomes import pair_outcome
from training.truth_baseline import CHECKS
from training.truth_scorer import SiteRow

if TYPE_CHECKING:
    from training.muraja_policy import SiteGrade

#: The two decodes #144's rule reads (Ship A's frame): the base teacher on the whole item,
#: and ``h448`` as shipped, streamed at b = 0.
RULE_ARMS = (arm("base", SPANS), arm("h448", "stream_b0"))
AGREED, DISAGREED, ABSTAINED = "agreed", "disagreed", "abstained"
CLASSES = (AGREED, DISAGREED, ABSTAINED)
#: The listening budgets #144 asks the estimate at: disagreements drawn per stratum.
BUDGETS = (20, 50)


class CorpusItem(Protocol):
    """One item of an external corpus: a span of one recording reciting a run of canonical
    words, with its realized reference."""

    key: str
    slug: str
    start_sample: int
    end_sample: int
    reference: ItemReference

    @property
    def source(self) -> str:
        """The corpus's truth-site source (``tadabur.truth_sites``)."""

    @property
    def audio_filename(self) -> str: ...

    @property
    def surah(self) -> int: ...

    @property
    def ayah(self) -> int: ...

    @property
    def surah_ayah(self) -> str: ...

    @property
    def audio_sha256(self) -> str:
        """The SHA-256 of the recording's audio as its corpus publishes it."""

    @property
    def cluster(self) -> int:
        """The bootstrap cluster the item's rates resample by."""

    @property
    def provenance(self) -> dict:
        """The recording's name and checksum as a worklist row carries them."""

    def excerpt(self, reference_index: int) -> tuple[int, int]:
        """The sample span a listener would hear for the carrier at ``reference_index``."""


def clip_of(item: CorpusItem, decodes: Mapping[str, Mapping[str, dict]]) -> Clip:
    """The item as :mod:`training.carrier_windowing` reads it, with all six arms."""
    reference, offsets = item.reference.reference, item.reference.offsets
    return Clip(
        audio_filename=item.audio_filename, surah_ayah=item.surah_ayah,
        reciter_id=item.cluster, weight=1.0, references=(reference,),
        starts=(item.start_sample,),
        positions=tuple(sorted(pair_positions(0, reference)
                               + geminate_positions(0, reference, offsets))),
        by_segment={},
        by_span={
            **{arm(m, p): decodes[m][item.key][p] for m in MODELS for p in ("spans", "stream_b0")},
            **{arm(m, CHECKS): committed_text(decodes[m][item.key][CHECKS]) for m in MODELS},
        },
    )


def listen_seconds(start: int, end: int) -> float:
    """An excerpt played :data:`~tadabur.listening_session.PLAYS_PER_SITE` times, plus the time
    to answer (#87's model)."""
    return round(PLAYS_PER_SITE * (end - start) / TARGET_SAMPLE_RATE + ANSWER_SECONDS, 2)


# --- the rule ------------------------------------------------------------------------------


def rule_class(prescribed: str, committed: Sequence[str | None]) -> str:
    """#144's class of one carrier from what the rule arms committed (module docstring)."""
    if all(c == prescribed for c in committed):
        return AGREED
    if any(c is not None and c != prescribed for c in committed):
        return DISAGREED
    return ABSTAINED


def classify(row: SiteRow) -> str:
    """#144's class of one target-pair or geminate carrier."""
    return rule_class(row.site.prescribed, [row.outcomes[a].committed for a in RULE_ARMS])


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


def _item_of(row: SiteRow, items: Mapping[tuple[str, int], CorpusItem]) -> CorpusItem:
    return items[(row.site.audio_filename, row.site.start_sample)]


def site_id(item: CorpusItem, reference_index: int, mark: str) -> str:
    return f"{item.source}:{item.key}:{reference_index}:{mark}"


def truth_sites(rows: Sequence[SiteRow], items: Mapping[tuple[str, int], CorpusItem]
                ) -> list[TruthSite]:
    """The agreed and disagreed sites (abstained are left out), each stratum a census of its
    corpus part (``slug``), direction and class."""
    kept = [(row, classify(row)) for row in rows]
    kept = [(row, cls) for row, cls in kept if cls != ABSTAINED]

    def stratum_name(row: SiteRow, cls: str) -> str:
        item = _item_of(row, items)
        return f"{item.source}:{item.slug}:{direction(row.site)}:{cls}"

    population = Counter(stratum_name(row, cls) for row, cls in kept)
    sites = []
    for row, cls in kept:
        item = _item_of(row, items)
        name = stratum_name(row, cls)
        sites.append(TruthSite(
            site_id=site_id(item, row.site.reference_index, row.site.mark),
            source=item.source, assumes_competent_reciter=cls == AGREED,
            audio_filename=item.audio_filename, shard=None, start_sample=item.start_sample,
            end_sample=item.end_sample, audio_sha256=item.audio_sha256,
            surah_ayah=item.surah_ayah, reference=row.site.reference,
            reference_index=row.site.reference_index, mark=row.site.mark,
            prescribed=row.site.prescribed,
            heard=row.site.prescribed if cls == AGREED else PENDING,
            stratum=name, stratum_population=population[name]))
    return sorted(sites, key=lambda s: s.site_id)


def _work_row(item: CorpusItem, reference_index: int, prescribed: str, committed: dict,
              stratum_name: str) -> dict:
    start, end = item.excerpt(reference_index)
    return {
        "stratum": stratum_name, "slug": item.slug, "item": item.key, **item.provenance,
        "surah_ayah": item.surah_ayah, "reference": item.reference.reference,
        "reference_index": reference_index, "prescribed": prescribed,
        "committed": dict(sorted(committed.items())),
        "excerpt_start_sample": start, "excerpt_end_sample": end,
        "listen_seconds": listen_seconds(start, end),
    }


def worklist(rows: Sequence[SiteRow], items: Mapping[tuple[str, int], CorpusItem]
             ) -> list[dict]:
    """One row per disagreed carrier (its sites, every arm's commit, its excerpt): what a
    listening sample would be drawn from. Nothing here is queued."""
    by_carrier: dict[tuple, list[SiteRow]] = defaultdict(list)
    for row in rows:
        if classify(row) == DISAGREED:
            by_carrier[carrier_key(row)].append(row)
    out = []
    for _, group in sorted(by_carrier.items()):
        row = group[0]
        item = _item_of(row, items)
        out.append({
            "site_ids": sorted(site_id(item, r.site.reference_index, r.site.mark)
                               for r in group),
            **_work_row(item, row.site.reference_index, row.site.prescribed,
                        {a: o.committed for a, o in row.outcomes.items()}, stratum(row)),
        })
    return out


def listening_estimate(rows: Sequence[dict]) -> dict:
    """Minutes to listen to every disagreement, and to :data:`BUDGETS` per stratum (at the
    stratum's mean excerpt time), per corpus part and pooled."""
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
            **{slug: minutes([r for r in rows if r["slug"] == slug])
               for slug in sorted({r["slug"] for r in rows})}}


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Rows as sorted-key JSON lines, atomically (a worklist)."""
    from training.tashkeel_eval import write_text_atomically

    write_text_atomically(path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                        for r in rows))


# --- every consonant (option B) -------------------------------------------------------------


@dataclass(frozen=True)
class ConsonantCarrier:
    """The first letter of a run of a consonant in an item, and what each rule arm committed
    there (:func:`training.site_outcomes.pair_outcome`, by position)."""

    item: str
    reference_index: int
    prescribed: str
    committed: tuple[tuple[str, str | None], ...]

    @property
    def cls(self) -> str:
        return rule_class(self.prescribed, [c for _, c in self.committed])


def _item_consonants(job: tuple[str, str, dict[str, str]]) -> list[ConsonantCarrier]:
    key, reference, decodes = job
    alignments = {a: align_on_carriers(decodes[a], reference) for a in RULE_ARMS}
    return [ConsonantCarrier(key, i, char, tuple((a, pair_outcome(alignments[a], i).committed)
                                                 for a in RULE_ARMS))
            for i, char in enumerate(reference)
            if char in CONSONANTS and run_start(reference, i) == i]


def consonant_carriers(items: Sequence[CorpusItem], decodes: Mapping[str, Mapping[str, dict]],
                       workers: int) -> list[ConsonantCarrier]:
    """Every consonant carrier of every item, with the rule arms' commits, in ``workers``
    processes (alignment is the cost)."""
    jobs = [(item.key, item.reference.reference,
             {a: decodes[a.split("/")[0]][item.key][a.split("/")[1]] for a in RULE_ARMS})
            for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [c for carriers in pool.map(_item_consonants, jobs, chunksize=16)
                for c in carriers]


def consonant_counts(carriers: Sequence[ConsonantCarrier]) -> dict:
    """Per consonant: carriers and each class; and, per rule arm, what it committed instead at
    the disagreements (``expected→committed``), most first."""
    letters: dict[str, Counter] = defaultdict(Counter)
    instead: dict[str, Counter] = {a: Counter() for a in RULE_ARMS}
    for c in carriers:
        letters[c.prescribed]["carriers"] += 1
        letters[c.prescribed][c.cls] += 1
        if c.cls == DISAGREED:
            for a, committed in c.committed:
                if committed is not None and committed != c.prescribed:
                    instead[a][f"{c.prescribed}→{committed}"] += 1
    totals = Counter(cls for c in carriers for cls in ("carriers", c.cls))
    return {
        "letters": {letter: dict(counts) for letter, counts in sorted(letters.items())},
        "total": dict(totals),
        "committed_instead": {a: [{"substitution": s, "carriers": n}
                                  for s, n in sorted(t.items(), key=lambda kv: (-kv[1], kv[0]))]
                              for a, t in instead.items()},
    }


def consonant_worklist(carriers: Sequence[ConsonantCarrier], items: Mapping[str, CorpusItem]
                       ) -> list[dict]:
    """One row per disagreed consonant carrier, as :func:`worklist` (``items`` by key)."""
    return [{"carrier": f"{items[c.item].source}:{c.item}:{c.reference_index}",
             **_work_row(items[c.item], c.reference_index, c.prescribed, dict(c.committed),
                         f"consonant:{c.prescribed}")}
            for c in sorted(carriers, key=lambda c: (c.item, c.reference_index))
            if c.cls == DISAGREED]


# --- merging the GPU box's decodes -----------------------------------------------------------


def merge_decode_files(paths: Sequence[Path], identity: Path, destination: Path,
                       expected: int) -> dict:
    """Per-shard decode files (``training.external_decode``) as one gzipped file, sorted by
    item, each item once (a resumed shard holds no torn record: the decoder cut it off before
    resuming); returns the identity record every shard ran with."""
    lines: dict[str, str] = {}
    for path in paths:
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                lines.setdefault(json.loads(raw)["key"], raw)
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"{path}:{number} is not a decode record") from error
    if len(lines) != expected:
        raise ValueError(f"{destination.name}: {len(lines)} items decoded, {expected} expected")
    record = json.loads(identity.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(lines[k] + "\n" for k in sorted(lines)).encode("utf-8")
    destination.write_bytes(gzip.compress(body, mtime=0))
    return record


def read_decode_file(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {line["key"]: line for line in map(json.loads, f)}


# --- the app outcome ---------------------------------------------------------------------------


def placed_words(item: CorpusItem, phoneme_word) -> list[tuple]:
    """Every Muraja phoneme word the item recites, once, as ``(sampled word, site)``: the
    canonical run's words carried to Muraja's numbering (``phoneme_word(surah, ayah, printed
    word)``, ``word_map``), never re-inferred from the reference, so an item that starts
    mid-ayah is graded on its own words. A sampled word's carrier is the first letter of the
    first canonical word in that phoneme word."""
    from tadabur.word_sample import SampledWord
    from training.muraja_policy import MurajaSite

    surah, ayah = item.surah, item.ayah
    out, seen = [], set()
    for position, word_index in enumerate(item.reference.word_indices):
        word = phoneme_word(surah, ayah, word_index + 1)
        if word in seen:
            continue
        seen.add(word)
        out.append((SampledWord(
            site_id=f"{item.key}:w{word}", audio_filename=item.audio_filename, shard=-1,
            audio_sha256=item.audio_sha256, reciter_id=item.cluster,
            surah_ayah=item.surah_ayah, segment_index=0,
            start_sample=item.start_sample, end_sample=item.end_sample,
            reference=item.reference.reference, word_index=word_index,
            reference_index=item.reference.offsets[position],
            stratum=f"{item.source}:{item.slug}", stratum_population=1),
            MurajaSite(surah, ayah, word, None)))
    return out


def placement(item: CorpusItem, placed: Sequence[tuple]):
    """The item's session on Muraja's reference: its ayah, starting on its first word."""
    from training.muraja_policy import Placement

    return Placement(item.surah, item.ayah, placed[0][1].word,
                     {word.site_id: site for word, site in placed})


@dataclass(frozen=True)
class WordGrade:
    """One word of one item under one model's checks and one scoring: where it sits in the
    item (``position``, 0-based, of ``words``), its Muraja phoneme word and the canonical word
    it starts on (0-based ``word_index``), the grade the engine kept (its outcome, errors and the
    checks that set it) and, for a flagged word, why (:func:`training.word_flags.flag`)."""

    model: str
    scoring: str
    item: CorpusItem
    position: int
    words: int
    word: int
    word_index: int
    kept: SiteGrade
    reason: object | None

    @property
    def outcome(self) -> str:
        return self.kept.outcome

    @property
    def graded(self) -> bool:
        from training.muraja_policy import NOT_GRADED, UNPLACED

        return self.outcome not in (NOT_GRADED, UNPLACED)

    @property
    def flagged(self) -> bool:
        return self.reason is not None


def app_scorings() -> list:
    """The two scorings every external corpus is graded under: the shipped default, and
    ADR-0012's allowances off."""
    from training.muraja_policy import SCORINGS, SHIPPED_DEFAULT

    return [SHIPPED_DEFAULT, next(s for s in SCORINGS if s.name == "allowances_off")]


def word_grades(items: Sequence[CorpusItem], decodes: Mapping[str, Mapping[str, dict]],
                harness) -> tuple[dict, list[WordGrade]]:
    """Every word of every item graded per model (its recorded checks) and scoring through the
    harness, and the harness build that graded them."""
    from training.muraja_policy import WORD_FLAGGED, Item, MurajaText, grade_placed
    from training.word_flags import flag

    scorings = app_scorings()
    text = MurajaText(harness.quran_db)
    placed = {item.key: placed_words(item, text.phoneme_word) for item in items}
    requests = [Item(f"{model}#{item.key}", tuple(w for w, _ in placed[item.key]),
                     tuple(decodes[model][item.key][CHECKS]))
                for model in MODELS for item in items]
    placements = {f"{model}#{item.key}": placement(item, placed[item.key])
                  for model in MODELS for item in items}
    graded = grade_placed(requests, placements, scorings, harness)
    out = []
    for model in MODELS:
        for scoring in scorings:
            for item in items:
                words = placed[item.key]
                for position, (word, site) in enumerate(words):
                    grade = graded.grades[scoring.name][(f"{model}#{item.key}", word.site_id)]
                    reason = (flag(grade, text.word_phonemes(site))
                              if grade.outcome in WORD_FLAGGED else None)
                    out.append(WordGrade(model, scoring.name, item, position, len(words),
                                         site.word, word.word_index, grade, reason))
    return graded.build, out


def flag_view(grades: Sequence[WordGrade]) -> dict:
    """Words, graded and flagged words, the flags' causes, and the consonant flags'
    substitutions, expected → heard and by expected letter
    (:func:`training.word_flags.substitution_tally`: a word counts once per key), with where a
    skipped word sits in its item."""
    from training.device_logs import per_1000
    from training.word_flags import CAUSES, SKIPPED, substitution_tally

    tally: Counter = Counter()
    causes: Counter = Counter()
    marks: Counter = Counter()
    by_slug: dict[str, Counter] = defaultdict(Counter)
    skipped_at: Counter = Counter()
    consonant_flags: list = []
    for g in grades:
        tally["words"] += 1
        if not g.graded:
            tally[g.outcome] += 1
            continue
        by_slug[g.item.slug]["graded"] += 1
        if not g.flagged:
            continue
        tally["flagged"] += 1
        by_slug[g.item.slug]["flagged"] += 1
        causes.update(g.reason.causes)
        marks.update(set(g.reason.marks))
        consonant_flags.append(g.reason.consonants)
        if SKIPPED in g.reason.causes:
            skipped_at["first word" if g.position == 0 else
                       "last word" if g.position == g.words - 1 else "inside"] += 1
    graded_words = sum(c["graded"] for c in by_slug.values())
    return {
        **dict(tally), "graded": graded_words,
        "per_1000": round(per_1000(tally["flagged"] / graded_words), 2)
        if graded_words else None,
        "causes": {c: causes[c] for c in CAUSES},
        "tashkeel_marks": dict(sorted(marks.items())),
        "skipped_where": dict(sorted(skipped_at.items())),
        "substitutions": substitution_tally(consonant_flags),
        "substitutions_by_expected": substitution_tally(consonant_flags, by_expected=True),
        "per_mushaf": {s: {**dict(c), "per_1000": round(per_1000(c["flagged"] / c["graded"]), 2)}
                       for s, c in sorted(by_slug.items())},
    }


def app_outcome(items: Sequence[CorpusItem], decodes: Mapping[str, Mapping[str, dict]],
                harness) -> dict:
    """Per model (its recorded checks) and scoring, every word's grade (:func:`flag_view`;
    ``per_mushaf`` is per corpus part)."""
    build, grades = word_grades(items, decodes, harness)
    out: dict = {"muraja_build": build, "scorings": [s.as_dict() for s in app_scorings()]}
    for model in MODELS:
        out[model] = {s.name: flag_view([g for g in grades
                                          if g.model == model and g.scoring == s.name])
                      for s in app_scorings()}
    return out


# --- what the outcome is set beside ------------------------------------------------------------


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

"""The populations #155 classifies (:mod:`training.consonant_flags`), read from what the repo
commits and replayed through the Muraja harness at the pin.

* **QUD pilot** (#144): both models, every word of every item regraded from the committed check
  sequences; each item's realized reference is known; clustered by mushaf and surah.
* **#151 device logs**: the included teacher sessions replayed at the pin, the words #151's docs
  explain left out. The log's checks are read (never written) to place the letter in the
  session's queries; the session's pauses are not known.
* **#153 owner sessions**: the committed per-word table only (word references, substitutions,
  the owner's verdicts); the owner's mistakes left out.
* **#90 word sample** and **Ship A's held-out verified sites** (correct side): ``h448``'s
  recorded checks, regraded torch-free from the committed check sequences.

A graded word is counted once per recital (an item's, a log's, a session's word), in the
denominators and the numerators alike.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from functools import cache

from training.consonant_flags import (
    DEVICE_LOGS,
    GRADING,
    HELD_OUT_SITES,
    IBTIDA_HERE,
    INTO_NEXT,
    FROM_PREVIOUS,
    MODEL,
    OWNER,
    QUD,
    SCORING_NAMES,
    WAQF_HERE,
    WORD_SAMPLE,
    Ayah,
    FlaggedWord,
    FlagRecord,
    Form,
    Population,
    Recital,
    covering_checks,
    error_without_group,
    owner_checks,
    phonetizer_slips,
    population_view,
    records,
    spellings,
    word_start,
)
from training.muraja_policy import NOT_FLAGGED, TASHKEEL_ERROR, WORD_FLAGGED, MurajaText, SiteGrade
from training.word_flags import SKIPPED_QUALITY

class Ayat:
    """Each ayah's :class:`~training.consonant_flags.Ayah`: Muraja's reference, the phonetizer's
    slips against the mushaf, how the mushaf spells each phoneme's letter, and the forms around a
    word, from quran-transcript (computed once per ayah)."""

    def __init__(self, text: MurajaText):
        self.text = text

    @cache
    def uthmani_words(self, surah: int, ayah: int) -> tuple[str, ...]:
        from tadabur.waqf_segments import _uthmani_words

        return tuple(_uthmani_words(f"{surah}:{ayah}"))

    def _phonetized(self, words: Sequence[str]) -> str | None:
        from hafs_phonetizer import phonetize

        try:
            return phonetize(" ".join(words)).phonemes
        except (KeyError, IndexError):
            return None  # quran-transcript 0.5.2's known gap

    @cache
    def get(self, surah: int, ayah: int) -> Ayah:
        from hafs_phonetizer import phonetize

        phonemes = self.text.phonemes(surah, ayah)
        uthmani = " ".join(self.uthmani_words(surah, ayah))
        try:
            out = phonetize(uthmani)
        except (KeyError, IndexError):
            out = None
        if out is None or out.phonemes.replace(" ", "") != phonemes.replace(" ", ""):
            return Ayah(phonemes, checked=False, forms=lambda word: self.forms(surah, ayah, word))
        spans = [None if m.deleted else m.pos for m in out.mappings]
        return Ayah(phonemes, phonetizer_slips(uthmani, out.phonemes, spans), True,
                    spellings(uthmani, out.phonemes, spans), lambda word: self.forms(surah, ayah, word))

    def every_ayah(self) -> Iterable[tuple[int, int]]:
        from quran_transcript import Aya

        for surah in range(1, 115):
            yield from ((surah, ayah) for ayah in range(1, Aya(surah, 1).get().num_ayat_in_sura + 1))

    @cache
    def forms(self, surah: int, ayah: int, word: int) -> dict[str, Form]:
        """The :data:`~training.consonant_flags.FORMS` a phoneme word's context can take (those
        the phonetizer handles)."""
        words = self.uthmani_words(surah, ayah)
        reference = self.text.phonemes(surah, ayah)
        start = word_start(reference, word)
        end = start + len(reference.split(" ")[word - 1])
        printed = self.text.text_words(surah, ayah, word)
        first, last = printed[0], printed[-1]
        spans: dict[str, tuple[list[str], int, int]] = {}
        if last < len(words):
            spans[WAQF_HERE] = (list(words[:last]), 0, end)
        if first > 1:
            spans[IBTIDA_HERE] = (list(words[first - 1:]), start, len(reference))
        if first == 1 and ayah > 1:
            spans[FROM_PREVIOUS] = ([self.uthmani_words(surah, ayah - 1)[-1], *words], 0, len(reference))
        if last == len(words) and self._has(surah, ayah + 1):
            spans[INTO_NEXT] = ([*words, self.uthmani_words(surah, ayah + 1)[0]], 0, len(reference))
        return {name: Form(phonemes, lo, hi) for name, (w, lo, hi) in spans.items()
                if (phonemes := self._phonetized(w)) is not None}

    @cache
    def _has(self, surah: int, ayah: int) -> bool:
        from quran_transcript import Aya

        return ayah <= Aya(surah, 1).get().num_ayat_in_sura


def phonetizer_check(ayat: Ayat) -> dict:
    """The orthography guard (:func:`~training.consonant_flags.mushaf_reason`) over every ayah,
    not only the flagged ones: the ayat checked, those it cannot check (the phonetizer does not
    reproduce Muraja's reference there), and every slip found."""
    readings = {(s, a): ayat.get(s, a) for s, a in ayat.every_ayah()}
    return {"ayat_checked": sum(r.checked for r in readings.values()),
            "not_checked": [f"{s}:{a}" for (s, a), r in readings.items() if not r.checked],
            "slips": [{"ayah": f"{s}:{a}", "ordinal": q, "reason": why}
                      for (s, a), r in readings.items() for q, why in sorted(r.slips.items())]}


def build_population(name: str, model: str, scoring: str, graded: Iterable[str],
                     flagged: Iterable[FlaggedWord], ayat: Ayat, clustered: bool) -> Population:
    """A population from its graded word occurrences' clusters and its flagged words, classified."""
    out: list[FlagRecord] = []
    for f in flagged:
        out += records(f, ayat.get(f.surah, f.ayah), ayat.text.printed(f.surah, f.ayah, f.word))
    return Population(name, model, scoring, dict(sorted(Counter(graded).items())), clustered, tuple(out))


def _consonant_flagged(grade: SiteGrade) -> bool:
    """Flagged for consonant errors: shown as not correct, neither a tashkeel error nor skipped."""
    return grade.outcome in WORD_FLAGGED and grade.quality not in (TASHKEEL_ERROR, SKIPPED_QUALITY)


def qud_populations(harness, ayat: Ayat) -> list[Population]:
    """The QUD pilot, both models and both scorings; each flag read against the queries of the
    checks that set its kept grade (:attr:`~training.muraja_policy.SiteGrade.setting_checks`)."""
    from training.qud_pilot import MODELS, grade_words, load_pilot
    from training.truth_baseline import CHECKS

    pilot = load_pilot()
    words = grade_words(pilot.items, pilot.decodes, harness)
    out = []
    for model in MODELS:
        for scoring in SCORING_NAMES:
            graded, flagged = [], []
            for item in pilot.items:
                checks = tuple(pilot.decodes[model][item.key][CHECKS])
                offsets = item.reference.offsets
                last = words.placed[item.key][-1][0].site_id
                for word, site in words.placed[item.key]:
                    grade = words.grade(scoring, model, item, word)
                    if grade.outcome not in (*WORD_FLAGGED, NOT_FLAGGED):
                        continue
                    cluster = f"{item.slug}:{item.row.surah}"
                    graded.append(cluster)
                    if not _consonant_flagged(grade):
                        continue
                    k = item.reference.word_indices.index(word.word_index)
                    flagged.append(FlaggedWord(
                        QUD, model, scoring, site.surah, site.ayah, site.word, grade.errors,
                        Recital(item.reference.reference, word.site_id == last, checks,
                                lambda position, grade=grade: grade.setting_checks),
                        cluster, word.site_id, (offsets[k], offsets[k + 1])))
            out.append(build_population(QUD, model, scoring, graded, flagged, ayat, clustered=True))
    return out


def device_log_populations(ayat: Ayat) -> list[Population]:
    """#151's included teacher sessions replayed at the pin, the explained words left out. The
    replay keeps no per-check grade, so a flag is read against every check whose query can hold
    its letter (:func:`~training.consonant_flags.covering_checks`); a log with no check parsed is
    no evidence."""
    from training import device_logs as dl
    from training.muraja_policy import MURAJA_COMMIT, SCORINGS

    pin = dl.require(MURAJA_COMMIT)
    entries = [e for e in dl.load_inventory() if e.included and e.model == dl.TEACHER]
    paths = {e.log: dl.log_dir(MURAJA_COMMIT) / f"{e.log}.log" for e in entries}
    scorings = [s for s in SCORINGS if s.name in SCORING_NAMES]
    _, replays = dl.replay_logs(pin, paths, scorings)
    checks = {log: tuple(dl.session_checks(path.read_text(encoding="utf-8"))) for log, path in paths.items()}
    out = []
    for scoring in scorings:
        graded, flagged = [], []
        for e in entries:
            for g in dl.word_grades(e, replays[e.log], [scoring], ayat.text):
                if g.explained:
                    continue
                graded.append(e.log)
                if g.flagged and g.consonants:
                    surah, ayah, word = dl.position(g.word)
                    reference = ayat.get(surah, ayah).phonemes
                    session = checks[e.log] or None
                    flagged.append(FlaggedWord(
                        DEVICE_LOGS, dl.TEACHER, scoring.name, surah, ayah, word, g.errors,
                        Recital(checks=session, setting=lambda position, session=session, reference=reference:
                                covering_checks(session, reference, position)),
                        e.log, f"{e.log}:{g.word}"))
        out.append(build_population(DEVICE_LOGS, dl.TEACHER, scoring.name, graded, flagged, ayat, clustered=True))
    return out


#: The owner's verdict on a word the phone showed, as a class (``owner_sessions.json``).
OWNER_VERDICTS = {"decode": MODEL, "boundary": GRADING}


def owner_populations(ayat: Ayat) -> list[Population]:
    """#153's committed per-word table (no group index: every unit holding the letter is a
    candidate); the owner's mistakes left out; no interval (three sessions)."""
    from training.device_exports import MODEL_ERROR, REPORT_PATH as OWNER_REPORT
    from training.device_logs import UNEXPLAINED, position

    report = json.loads(OWNER_REPORT.read_text(encoding="utf-8"))
    out = []
    for scoring in SCORING_NAMES:
        words = report["rates"][scoring][UNEXPLAINED]["false_flags"]["words"]
        flagged = []
        for row in report["words"]:
            if row["scoring"] != scoring or row["explained"] is not None or not row["substitutions"]:
                continue
            surah, ayah, word = position(row["word"])
            verdict = row["verdict"]
            evidence = OWNER_VERDICTS.get(verdict.get("explained_by")) if verdict["status"] == MODEL_ERROR else None
            errors = tuple(error_without_group(label) for label in row["substitutions"])
            flagged.append(FlaggedWord(OWNER, "h448", scoring, surah, ayah, word, errors,
                                       Recital(verdict=evidence), row["log"], f"{row['log']}:{row['word']}"))
        # three sessions carry no interval: the words are counted, not clustered
        out.append(build_population(OWNER, "h448", scoring, ["sessions"] * words, flagged, ayat, clustered=False))
    return out


def tadabur_populations(harness, ayat: Ayat) -> list[Population]:
    """#90's word sample and Ship A's held-out verified sites (correct side) on ``h448``'s
    recorded checks; a word occurrence is an item's word, however many sites sit on it."""
    from tadabur.exclusions import current
    from tadabur.listening_session import adjudicated, read_verdicts
    from tadabur.ship_a import HELD_OUT
    from tadabur.ship_a_verified import SHIP_A, load_designs
    from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips
    from tadabur.word_sample import load_words, verified_words
    from training import truth_baseline as tb
    from training.decoding import committed_text
    from training.site_outcomes import CORRECT_SIDE
    from training.truth_scorer import item_key

    arm = tb.arm("h448", tb.CHECKS)
    verdicts = read_verdicts()
    designs = load_designs()
    excluded = current()
    sites = [s for s in adjudicated(tb.load_sites(tb.truth_site_files()), verdicts)
             if s.site_id in designs and (designs[s.site_id].frame, designs[s.site_id].half) == (SHIP_A, HELD_OUT)]
    words = verified_words(load_words(), verdicts)
    registry = load_staged_clips(REGISTRY_PATH)
    items = tb.items_of([*sites, *(w.site for w in words)], registry)
    _, recorded = tb.model_caches("h448", tb.DEFAULT_MODELS["h448"], items, registry, None)
    sequences = {arm: {k: v[tb.CHECKS] for k, v in recorded["items"].items() if k in items}}
    decodes = {arm: {k: committed_text(c) for k, c in sequences[arm].items()}}
    _, kept = tb.app_outcomes(sites, words, decodes, sequences, harness)
    correct_sites = [s for s in sites if s.heard == s.prescribed and s.site_id not in excluded]
    correct_words = [w.site for w in words if w.side == CORRECT_SIDE and w.site.site_id not in excluded]
    out = []
    for name, located in ((WORD_SAMPLE, correct_words), (HELD_OUT_SITES, correct_sites)):
        for scoring in SCORING_NAMES:
            grades: Mapping[str, SiteGrade] = kept.grades[scoring][arm]
            graded, flagged, seen = [], [], set()
            for site in located:
                grade = grades[site.site_id]
                where = ayat.text.locate(site)
                occurrence = (item_key(site), where.word) if where else None
                if where is None or grade.outcome not in (*WORD_FLAGGED, NOT_FLAGGED) or occurrence in seen:
                    continue
                seen.add(occurrence)
                graded.append(item_key(site))
                if not _consonant_flagged(grade):
                    continue
                span = _word_span(site.reference, site.reference_index)
                flagged.append(FlaggedWord(
                    name, "h448", scoring, where.surah, where.ayah, where.word, grade.errors,
                    Recital(site.reference, span[1] >= len(site.reference.rstrip()),
                            tuple(sequences[arm][item_key(site)]), lambda position, grade=grade: grade.setting_checks),
                    item_key(site), f"{item_key(site)}:w{where.word}", span))
            out.append(build_population(name, "h448", scoring, graded, flagged, ayat, clustered=False))
    return out


def _word_span(reference: str, index: int) -> tuple[int, int]:
    """The space-delimited word of ``reference`` around ``index``."""
    start = reference.rfind(" ", 0, index) + 1
    end = reference.find(" ", index)
    return start, len(reference) if end < 0 else end


def build() -> dict:
    from training.muraja_policy import MURAJA_COMMIT, require_harness

    harness = require_harness()
    ayat = Ayat(MurajaText(harness.quran_db))
    populations = [*qud_populations(harness, ayat), *device_log_populations(ayat), *owner_populations(ayat),
                   *tadabur_populations(harness, ayat)]
    return {
        "issue": 155,
        "muraja_commit": MURAJA_COMMIT,
        "phonetizer_check": phonetizer_check(ayat),
        "populations": [population_view(p) for p in populations],
        # each population and model, its scorings' flags together (a word once)
        "owner_checks": [{"population": name, "model": model, "checks": owner_checks(
            [r for p in populations if (p.name, p.model) == (name, model) for r in p.records])}
            for name, model in dict.fromkeys((p.name, p.model) for p in populations)],
        "records": [{k: v for k, v in asdict(r).items() if v is not None or k not in ("printed", "context", "decode")}
                    for p in populations for r in p.records],
    }

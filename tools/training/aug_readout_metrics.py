"""The augmentation readout's metrics (#95): observations per model, compared in pairs.

Every number the readout reports (:mod:`training.aug_readout`) is a **ratio of sums over
observations**, read per model and compared between models on the same observations, the way
``docs/acceptance-rules.md`` §1 reads a rate (:mod:`training.acceptance_stats`): each
observation carries a numerator and a denominator per model and the cluster it belongs to, and
an interval is the clustered bootstrap (10,000 draws, seed 20261008, one draw shared by every
statistic on a set, so a difference is **paired**). The clusters are #128's: Tarteel v1 by
surah, qdat (and qdat_bench, its clips) by reciter, QUD by mushaf and surah, Tadabur by reciter.

The observations (:class:`Terms`), one builder each:

* **false flags** (:func:`false_flag_terms`): one per word Muraja's engine places, flagged over
  graded, from :class:`training.external_corpus.WordGrade` s; per 1,000 graded words;
* **word agreement with the teacher** (:func:`word_agreement_terms`): one per item, the reference
  words whose decoded letters equal the teacher's (:func:`word_segments`) over its words;
* **teacher agreement on Tadabur** (:func:`teacher_agreement_terms`): §2's agreement, one per
  clip (:func:`training.acceptance_stats.agreement_terms`);
* **agreement with qdat_bench's experts** (:func:`bench_terms`): one per clip and letter kind,
  :func:`training.nonpro_eval.letter_outcomes` agreeing over all;
* **leniency** (descriptive): the blank share of the argmax frames (:func:`blank_terms`), the
  consonant carriers a decode commits nothing at (:func:`abstention_terms`), and the words the
  engine leaves ungraded (:func:`ungraded_terms`).

Torch-free and harness-free: decodes, references and word grades in, numbers out.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from training.acceptance_stats import (
    LOWER_QUANTILE,
    UPPER_QUANTILE,
    VETO_QUANTILES,
    AgreementRecord,
    Interval,
    Resample,
    agreement_terms,
    difference_interval,
    project_to_teacher,
    ratio,
    ratio_interval,
)
from training.word_flags import (OTHER_CONSONANT, SHADDAH_CAUSE, SOFT_PAIR, TARGET_PAIR,
                                 TASHKEEL)

#: Scales a ratio is reported on.
PER_1000 = 1000.0
POINTS = 100.0
#: The models every table reads, by role (:mod:`training.aug_readout` binds the arms).
CANDIDATE, CONTROL, H448, BASE = "candidate", "control", "h448", "base"
#: The consonant round's paired baseline (#174).
C95 = "c95"
#: The paired differences every readout table shows: against ``h448`` and the control (#95,
#: "every table is shown twice"), and against C95, the consonant round's baseline (#174). A
#: comparison whose model a set does not carry is left out, so #95's tables are unchanged.
COMPARISONS = ((CANDIDATE, H448), (CANDIDATE, CONTROL), (CANDIDATE, C95))
#: The consonant round's cause cells (#174, and round 2's reported shaddah line): a consonant
#: substitution other than the madd alif, a tashkeel error as :mod:`training.word_flags` reads
#: it, or a collapsed geminate.
CONSONANT, TASHKEEL_CAUSE = "consonant", "tashkeel"
CAUSES = (CONSONANT, TASHKEEL_CAUSE, SHADDAH_CAUSE)
#: The flag causes a consonant error carries; ``ا→∅`` is the madd alif (Muraja#263).
CONSONANT_FLAG_CAUSES = frozenset({OTHER_CONSONANT, SOFT_PAIR, TARGET_PAIR})
MADD_ALIF = "ا→∅"
#: The teacher-departure subsets of agreement (#174, bar 4b), and round 2's reported shaddah
#: line: the words whose teacher decode holds a geminate (a doubled consonant run).
DEPARTS, EQUALS, GEMINATE = "departs", "equals", "geminate"
SUBSETS = (DEPARTS, EQUALS, GEMINATE)


def carries_cause(flag, cause: str) -> bool:
    """Whether one flagged word's :class:`training.word_flags.Flag` carries ``cause`` in the
    round's sense: ``tashkeel`` and ``shaddah`` exactly as ``word_flags`` records them;
    ``consonant`` through a consonant substitution other than the madd alif (``ا→∅``,
    Muraja#263)."""
    if cause == TASHKEEL_CAUSE:
        return TASHKEEL in flag.causes
    if cause == SHADDAH_CAUSE:
        return SHADDAH_CAUSE in flag.causes
    if cause == CONSONANT:
        return any(c in CONSONANT_FLAG_CAUSES and substitution != MADD_ALIF
                   for c, substitution in flag.consonants)
    raise ValueError(f"{cause!r} is no cause; the round reads {CAUSES}")


def comparison_name(a: str, b: str) -> str:
    return f"{a}−{b}"


@dataclass(frozen=True)
class Terms:
    """One set's observations: each one's cluster, and per model its numerators and
    denominators (aligned with ``clusters``)."""

    clusters: tuple[int, ...]
    by_model: Mapping[str, tuple[np.ndarray, np.ndarray]]

    def __post_init__(self) -> None:
        for model, (num, den) in self.by_model.items():
            if len(num) != len(self.clusters) or len(den) != len(self.clusters):
                raise ValueError(f"{model}: {len(num)}/{len(den)} terms for "
                                 f"{len(self.clusters)} observations")


def _terms(clusters: Sequence[int], rows: Mapping[str, Sequence[tuple[float, float]]]) -> Terms:
    return Terms(tuple(int(c) for c in clusters),
                 {m: (np.array([n for n, _ in r], dtype=float),
                      np.array([d for _, d in r], dtype=float)) for m, r in rows.items()})


def _scaled(interval: Interval, scale: float) -> dict:
    out = interval.as_dict()
    for end in ("lower", "upper"):
        value = getattr(interval, end)
        if value is not None and math.isfinite(value):
            out[end] = value * scale
    return out


def _point(value: float, scale: float) -> float | None:
    return None if math.isnan(value) else value * scale


def compare(terms: Terms, scale: float,
            comparisons: Sequence[tuple[str, str]] = COMPARISONS,
            quantiles: tuple[float, float] = (LOWER_QUANTILE, UPPER_QUANTILE),
            resample: Resample | None = None) -> dict:
    """Each model's ratio with its interval, and each comparison's paired difference with its
    interval, every number on ``scale``; one bootstrap draw serves every statistic (``resample``
    lets a caller share it), ``quantiles`` sets the two-sided width, and a comparison whose
    models are not both present is left out."""
    resample = resample or Resample.by_cluster(terms.clusters)
    clusters = list(terms.clusters)
    models = {}
    for model, (num, den) in terms.by_model.items():
        models[model] = {"num": float(num.sum()), "den": float(den.sum()),
                         "point": _point(ratio(num, den), scale),
                         "interval": _scaled(ratio_interval(clusters, num, den,
                                                            resample=resample,
                                                            quantiles=quantiles), scale)}
    differences = {}
    for a, b in comparisons:
        if a not in terms.by_model or b not in terms.by_model:
            continue
        pa, pb = ratio(*terms.by_model[a]), ratio(*terms.by_model[b])
        differences[comparison_name(a, b)] = {
            "point": _point(pa - pb, scale),
            "interval": _scaled(difference_interval(clusters, terms.by_model[a],
                                                    terms.by_model[b], resample=resample,
                                                    quantiles=quantiles), scale)}
    return {"observations": len(clusters), "clusters": len(set(clusters)), "scale": scale,
            "models": models, "comparisons": differences}


def paired_se(terms: Terms, a: str, b: str, scale: float,
              resample: Resample | None = None) -> float:
    """The clustered paired bootstrap's standard error of ``a`` − ``b``, on ``scale``: the
    standard deviation of the replicate differences (the same draw and clusters ``compare``
    reports an interval from). A line's width, not a bound: it decides nothing."""
    resample = resample or Resample.by_cluster(terms.clusters)
    gap = resample.ratios(*terms.by_model[a]) - resample.ratios(*terms.by_model[b])
    return float(np.std(gap)) * scale


# --- false flags --------------------------------------------------------------------------------


def _word_key(grade) -> tuple[str, int]:
    return (grade.item.key, grade.position)


def _by_model(grades: Sequence, models: Sequence[str], scoring: str) -> dict[str, dict]:
    """Each model's grades under ``scoring``, by word; every model must grade the same words."""
    out: dict[str, dict] = {m: {} for m in models}
    for g in grades:
        if g.model in out and g.scoring == scoring:
            out[g.model][_word_key(g)] = g
    keys = {m: set(v) for m, v in out.items()}
    first = next(iter(keys.values()))
    if any(k != first for k in keys.values()) or not first:
        raise ValueError(f"the models place different words: {[len(k) for k in keys.values()]}")
    return out


def false_flag_terms(grades: Sequence, models: Sequence[str], scoring: str,
                     keep: Callable | None = None, cause: str | None = None) -> Terms:
    """One observation per word every model places: flagged over graded, clustered by the
    word's item; ``keep(grade)`` restricts the words (a word kept for one model is kept for
    all, so it must not read the model), and ``cause`` keeps only the flagged words whose
    :class:`training.word_flags.Flag` carries it (:func:`carries_cause`)."""
    if cause is not None and cause not in CAUSES:
        raise ValueError(f"{cause!r} is no cause; the round reads {CAUSES}")
    by_model = _by_model(grades, models, scoring)
    first = by_model[models[0]]
    words = sorted(k for k, g in first.items() if keep is None or keep(g))
    clusters = [first[k].item.cluster for k in words]

    def counted(grade) -> bool:
        """A flagged word counts; with a cause, only when its own flag carries it. The
        denominator is every graded word either way."""
        return grade.flagged and (cause is None or carries_cause(grade.reason, cause))

    return _terms(clusters, {m: [(float(counted(by_model[m][k])),
                                  float(by_model[m][k].graded)) for k in words] for m in models})


def ungraded_terms(grades: Sequence, models: Sequence[str], scoring: str) -> Terms:
    """Leniency: one observation per placed word, left ungraded (pending, or off Muraja's
    reference) over one."""
    by_model = _by_model(grades, models, scoring)
    words = sorted(by_model[models[0]])
    clusters = [by_model[models[0]][k].item.cluster for k in words]
    return _terms(clusters, {m: [(float(not by_model[m][k].graded), 1.0) for k in words]
                             for m in models})


def clips_flagged_terms(grades: Sequence, models: Sequence[str], scoring: str) -> Terms:
    """One observation per item: whether any of its words is flagged, over one."""
    by_model = _by_model(grades, models, scoring)
    items: dict[str, object] = {}
    for g in by_model[models[0]].values():
        items[g.item.key] = g.item
    keys = sorted(items)
    flagged = {m: {g.item.key for g in by_model[m].values() if g.flagged} for m in models}
    return _terms([items[k].cluster for k in keys],
                  {m: [(float(k in flagged[m]), 1.0) for k in keys] for m in models})


# --- word agreement with the teacher -------------------------------------------------------------


def word_segments(decode: str, reference: str) -> tuple[str, ...]:
    """``decode`` cut at the word boundaries of ``reference`` (words separated by spaces).

    The decode is aligned to the reference's letters by the Levenshtein backtrace
    (:func:`training.decode_evalset.align`, diagonal first). Each decoded character belongs to
    the word of the reference character it is matched or substituted against; an inserted
    character to the word of the last reference character consumed before it (the first word,
    before any). A word the decode leaves out is an empty segment."""
    from training.decode_evalset import align

    words = reference.split(" ")
    letters = "".join(words)
    word_of = [k for k, word in enumerate(words) for _ in word]
    out: list[list[str]] = [[] for _ in words]
    current = 0
    for _, i, j in align(letters, decode):
        if i is not None:
            current = word_of[i]
        if j is not None:
            out[current].append(decode[j])
    return tuple("".join(chars) for chars in out)


def _geminate_words(teacher: str, reference: str) -> list[int]:
    """The reference words whose teacher decode holds a geminate: a doubled consonant run the
    decode commits at (``shaddah``, :func:`training.site_outcomes.shaddah_outcome`)."""
    from bisect import bisect_right

    from tadabur.contrast_attribution import align_on_carriers
    from tadabur.listening_session import run_start
    from tadabur.truth_sites import CONSONANTS
    from training.site_outcomes import HELD, consonant_run, shaddah_outcome

    starts, offset = [], 0
    for word in reference.split(" "):
        starts.append(offset)
        offset += len(word) + 1
    alignment = align_on_carriers(teacher, reference)
    raw = alignment.raw_reference
    held = set()
    for index, char in enumerate(raw):
        if (char in CONSONANTS and run_start(raw, index) == index
                and len(consonant_run(raw, index)) > 1
                and shaddah_outcome(alignment, index).committed == HELD):
            held.add(bisect_right(starts, index) - 1)
    return sorted(held)


def _agreeing_words(job: tuple[str, str, str, str | None]) -> tuple[int, int]:
    """The reference words the decoded ones equal, over the words ``subset`` keeps: the words
    whose teacher word differs from the reference word (:data:`DEPARTS`), those where it is
    equal (:data:`EQUALS`), those the teacher holds a geminate in (:data:`GEMINATE`), or every
    word (``None``)."""
    decode, teacher, reference, subset = job
    mine, theirs = word_segments(decode, reference), word_segments(teacher, reference)
    words = reference.split(" ")
    if subset == GEMINATE:
        kept = _geminate_words(teacher, reference)
    else:
        kept = (range(len(theirs)) if subset is None else
                [k for k, word in enumerate(theirs)
                 if (word == words[k]) == (subset == EQUALS)])
    return sum(mine[k] == theirs[k] for k in kept), len(kept)


def _map(function, jobs: Sequence, workers: int) -> list:
    if workers <= 1 or len(jobs) < 64:
        return [function(j) for j in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(function, jobs, chunksize=32))


def word_agreement_terms(items: Sequence, decodes: Mapping[str, Mapping[str, str]],
                         models: Sequence[str], teacher: str = BASE,
                         workers: int = 1, subset: str | None = None) -> Terms:
    """One observation per item: its reference words whose decoded letters are the teacher's
    (both cut by :func:`word_segments`; every decode first projected onto the teacher's
    vocabulary, :func:`training.acceptance_stats.project_to_teacher`) over its words.
    ``decodes[model][item key]`` is the decode compared (the committed text of the checks).
    ``subset`` restricts the words to those the teacher's word departs from the reference at
    (:data:`DEPARTS`), those where it is equal (:data:`EQUALS`), or those it holds a geminate
    in (:data:`GEMINATE`, round 2's reported shaddah line); ``None`` pools them."""
    if subset is not None and subset not in SUBSETS:
        raise ValueError(f"{subset!r} is no subset; the round reads {SUBSETS}")
    rows: dict[str, list[tuple[float, float]]] = {}
    for model in models:
        jobs = [(project_to_teacher(decodes[model][i.key]),
                 project_to_teacher(decodes[teacher][i.key]), i.reference.reference, subset)
                for i in items]
        rows[model] = [(float(a), float(n)) for a, n in _map(_agreeing_words, jobs, workers)]
    return _terms([i.cluster for i in items], rows)


def observed_substitutions(grades: Sequence, models: Sequence[str],
                           scoring: str) -> tuple[str, ...]:
    """Every substitution a model's flagged word carries under ``scoring``, sorted: what the
    readout reports descriptively. The frozen veto list is a separate selection (#174)."""
    found: set[str] = set()
    for grade in grades:
        if grade.model in models and grade.scoring == scoring and grade.flagged \
                and grade.reason is not None:
            found.update(substitution for _, substitution in grade.reason.consonants)
    return tuple(sorted(found))


def substitution_terms(grades: Sequence, models: Sequence[str], scoring: str,
                       substitutions: Sequence[str]) -> dict[str, Terms]:
    """Each substitution's observations, one per word every model places: flagged **and**
    carrying that substitution over graded (:func:`training.word_flags.Flag.consonants`), so
    every substitution keeps the same denominator and a model's own flag decides its
    numerator."""
    by_model = _by_model(grades, models, scoring)
    first = by_model[models[0]]
    words = sorted(first)
    clusters = [first[k].item.cluster for k in words]

    def carries(grade, substitution: str) -> bool:
        flag = grade.reason
        return bool(grade.flagged and flag and any(s == substitution
                                                   for _, s in flag.consonants))

    return {substitution: _terms(
        clusters, {m: [(float(carries(by_model[m][k], substitution)),
                        float(by_model[m][k].graded)) for k in words] for m in models})
        for substitution in substitutions}


def substitution_views(terms: Mapping[str, Terms], scale: float,
                       comparisons: Sequence[tuple[str, str]] | None = None) -> dict[str, dict]:
    """Each substitution's :func:`compare` view with the round's 99.5 % veto interval on every
    comparison (``veto_interval``), from the same bootstrap draw as its 95 % interval.
    ``comparisons`` defaults to the candidate against every other model present."""
    out = {}
    for substitution, one in terms.items():
        pairs = comparisons or tuple((CANDIDATE, m) for m in sorted(one.by_model)
                                     if m != CANDIDATE)
        resample = Resample.by_cluster(one.clusters)
        view = compare(one, scale, pairs, resample=resample)
        for a, b in pairs:
            name = comparison_name(a, b)
            if name in view["comparisons"]:
                view["comparisons"][name]["veto_interval"] = _scaled(
                    difference_interval(one.clusters, one.by_model[a], one.by_model[b],
                                        resample=resample, quantiles=VETO_QUANTILES), scale)
        out[substitution] = view
    return out


# --- teacher agreement on Tadabur (§2) ----------------------------------------------------------


def teacher_agreement_terms(clips: Sequence, decodes: Mapping[str, Mapping[str, str]],
                            models: Sequence[str]) -> Terms:
    """§2's teacher agreement per clip (``clips``: :class:`training.nonpro_gap.DevClip`, the
    teacher text the dev manifest caches), ``decodes[model][filename]`` the clip's decode."""
    rows = {}
    clusters: list[int] = []
    for model in models:
        records = [AgreementRecord(c.reciter_id, c.cached_teacher, decodes[model][c.filename])
                   for c in clips]
        clusters, num, den = agreement_terms(records)
        rows[model] = list(zip(num, den))
    return _terms(clusters, rows)


# --- qdat_bench: the experts' transcript -------------------------------------------------------


def bench_terms(items: Sequence, decodes: Mapping[str, Mapping[str, str]],
                transcripts: Mapping[str, str], models: Sequence[str]) -> dict[str, Terms]:
    """Per letter kind (consonant, haraka, shaddah), one observation per bench clip: the slots
    of the experts' transcript the decode agrees at over all of them
    (:func:`training.nonpro_eval.letter_outcomes`; an empty slot disagrees)."""
    from training.nonpro_eval import AGREE, LETTER_KINDS, letter_outcomes

    outcomes = {m: [letter_outcomes(decodes[m][i.key], transcripts[i.audio_filename])
                    for i in items] for m in models}
    clusters = [i.cluster for i in items]
    return {kind: _terms(clusters, {m: [(float(sum(o == AGREE for _, o in clip[kind])),
                                         float(len(clip[kind]))) for clip in outcomes[m]]
                                    for m in models})
            for kind in LETTER_KINDS}


# --- leniency ---------------------------------------------------------------------------------


def blank_terms(clips: Sequence, blanks: Mapping[str, Mapping[str, Sequence[int]]],
                models: Sequence[str]) -> Terms:
    """One observation per clip: its blank argmax frames over all its frames
    (``blanks[model][filename]`` = ``[blank, frames]``)."""
    return _terms([c.reciter_id for c in clips],
                  {m: [tuple(map(float, blanks[m][c.filename])) for c in clips] for m in models})


def _abstained(job: tuple[str, str]) -> tuple[int, int]:
    from tadabur.contrast_attribution import align_on_carriers
    from tadabur.listening_session import run_start
    from tadabur.truth_sites import CONSONANTS
    from training.site_outcomes import pair_outcome

    decode, reference = job
    alignment = align_on_carriers(decode, reference)
    carriers = [i for i, c in enumerate(reference)
                if c in CONSONANTS and run_start(reference, i) == i]
    return sum(pair_outcome(alignment, i).committed is None for i in carriers), len(carriers)


def abstention_terms(items: Sequence, decodes: Mapping[str, Mapping[str, str]],
                     models: Sequence[str], workers: int = 1) -> Terms:
    """One observation per item: the consonant carriers (the first letter of every run of a
    consonant) at which the decode commits nothing, by the pair-site rule
    (:func:`training.site_outcomes.pair_outcome`), over all of them."""
    return _terms([i.cluster for i in items],
                  {m: [(float(a), float(n)) for a, n in _map(
                      _abstained, [(decodes[m][i.key], i.reference.reference) for i in items],
                      workers)] for m in models})

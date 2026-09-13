"""Neighbour-ayah bleed detection — the recitation a staged clip carries but does not own.

A Tadabur clip is cut from a continuous recitation, so it "often carries a neighbour-ayah
lead-in or trailing bleed — audio of the previous/next ayah the reciter ran into"
(ADR-0002). For a *passing* clip :func:`tadabur.waqf_detect.segment_clip` re-cuts that
away and records the span on :class:`~tadabur.clip_status.ClipStatus`. For a **rejected**
clip it does not: ``repeated_recitation`` is a skip reason, and a skipped clip's
"recitation span defaults to the whole clip". The reject pile is exactly the population
Muraja ADR-0016 mines, so bleed reaches that corpus unfiltered. This module is the
detector that closes the gap (HifzGuide #67).

Three signals were tried first and none worked; the report
``docs/tadabur-bleed-recut.md`` scores them. They fail for one shared reason, and naming
it is what motivates the method here:

* the edge trims (:func:`~tadabur.smith_waterman.edge_insertion_trims`) and
  ``waqf_detect``'s edge re-cut both read the **whole-clip alignment span**, and
* a per-ayah reference comparison asks which *single* ayah the clip best matches, which a
  re-read's repeated tail answers correctly and uselessly — "this one".

The whole-clip alignment is the problem. Smith-Waterman is *local* with cheap affine gap
extension, so when the query runs off the end of its reference it does not stop; it
**drags** the alignment through the unmatched material rather than paying to restart
(Muraja ADR-0009 documents the same drag consuming unrelated reference text). A dragged
bleed leaves ``leading_trim == trailing_trim == 0``, so every detector reading the trims
is blind to it by construction.

The fix is to stop asking the aligner to explain bleed with reference text that cannot
explain it. **Give it the neighbours.** The query is aligned once against
``prev | this | next`` concatenated, and bleed is read off as the reference positions the
alignment consumed *outside* this ayah's region. Drag does not survive the change: where
bleed is real, the neighbour's text scores ``MATCH_SCORE`` per phoneme and the alignment
takes it; where there is none, the alignment can still wander into a neighbour but only
over mismatches and gaps. So the two are told apart by **purity** — the share of consumed
neighbour reference phonemes that matched exactly — not by span, and not by score.

The detector is deliberately **torch-free and audio-free**: it runs off
``RejectRecord.predicted_phonemes``, the decode the gate already stored, so the whole
reject pile can be re-scored offline with no GPU. Converting a detected bleed to a *time*
does need the timed decode, and that is :mod:`tadabur.bleed_recut`'s job, not this one's.

Usage:
  python -m tadabur.bleed_detect --rejects reject_run/rejects.jsonl
    [--labels eval_fixtures/reject_bleed_labels.jsonl] [--json bleed.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .normalization import PhonemeNormalization, normalize_phonemes
from .reference_phonemes import load_reference_phonemes
from .smith_waterman import AlignmentResult, smith_waterman

# Reference regions are joined by a single space, the same word separator the normalized
# references already carry. The query is spaceless after normalization, so a space is
# always a gap — it marks the boundary without offering the aligner anything to match.
REFERENCE_SEPARATOR = " "

# A bleed must be at least this many exactly-matched neighbour phonemes. Two is low on
# purpose: the shortest real bleeds in the labelled set are two phonemes (a reciter
# starting "fa-in" of the next ayah), and the cost of acting on a two-phoneme false
# positive is bounded — the re-cut removes material the alignment attributed to the
# *neighbour*, never to this ayah, and :mod:`tadabur.bleed_recut` re-gates afterwards.
MIN_BLEED_PHONEMES = 2

# ...and at least this share of the neighbour reference phonemes it consumed must be
# exact matches. This is the term that separates bleed from drag. Measured on the
# labelled set, real bleed runs 0.80–1.00 and drag runs 0.27–0.40; 0.6 sits in the empty
# middle. Reference spaces are excluded from the denominator (they are word separators a
# spaceless query can never match, so counting them would penalise long bleeds only).
MIN_BLEED_PURITY = 0.6

# Surah-initial clips take the basmallah as an additional lead-in candidate: a reciter
# beginning a surah says it before the first ayah, so it is the likeliest thing to bleed
# into ayah 1 — likelier than the previous surah's last ayah, which is also offered.
# Al-Fatiha holds the basmallah as its own first ayah and At-Tawbah has none.
BASMALLAH_KEY = "1:1"
SURAHS_WITHOUT_BASMALLAH = frozenset({1, 9})


@dataclass(frozen=True)
class EdgeBleed:
    """What one edge of a clip took from its neighbouring ayah (or ayat).

    ``matched`` is the count of neighbour reference phonemes the alignment matched
    **exactly** and ``span`` the count it consumed at all (matches, substitutions and
    skipped reference phonemes alike, excluding word-separator spaces); ``purity`` is
    their ratio and is the signal that separates real bleed from a dragged alignment.
    ``query_start``/``query_end`` are the half-open range of *normalized decode* indices
    the neighbour claimed — the phoneme-space boundary a re-cut converts to time.
    ``neighbours`` names the reference ayat offered at this edge, in concatenation order.
    ``detected`` is the verdict at the thresholds the detector was run with, carried as a
    field rather than recomputed from a property so a threshold sweep stays honest about
    which setting produced a given verdict.
    """

    neighbours: tuple[str, ...] = ()
    matched: int = 0
    span: int = 0
    query_start: int = 0
    query_end: int = 0
    detected: bool = False

    @property
    def purity(self) -> float:
        """Share of consumed neighbour reference phonemes that matched exactly."""
        return self.matched / self.span if self.span else 0.0

    @property
    def query_phonemes(self) -> int:
        """Decoded phonemes this edge attributes to the neighbour."""
        return max(0, self.query_end - self.query_start)


@dataclass(frozen=True)
class BleedVerdict:
    """Both edges of one clip, plus how much of its own ayah the decode covered.

    ``query_covered_start``/``query_covered_end`` are the half-open *normalized decode*
    range the alignment attributed to this ayah — the recitation a re-cut must keep whole,
    and the inner bound every cut is clamped against.
    ``ref_covered_start``/``ref_covered_end`` are the half-open reference span of *this*
    ayah the alignment reached, and ``ref_length`` the ayah's full reference length, so
    ``uncovered_tail`` measures the mirror case of bleed: a clip that stops before the
    ayah ends (40:16 in the labelled set — "audio does not finish the ayah"). Truncation
    is reported, never re-cut: there is nothing to remove, only a covered range to shorten.
    """

    surah_ayah: str
    leading: EdgeBleed = EdgeBleed()
    trailing: EdgeBleed = EdgeBleed()
    query_length: int = 0
    query_covered_start: int = 0
    query_covered_end: int = 0
    ref_covered_start: int = 0
    ref_covered_end: int = 0
    ref_length: int = 0

    @property
    def detected(self) -> bool:
        """Whether either edge carries bleed."""
        return self.leading.detected or self.trailing.detected

    @property
    def uncovered_head(self) -> int:
        """Reference phonemes before the covered span — an ayah begun late."""
        return self.ref_covered_start

    @property
    def uncovered_tail(self) -> int:
        """Reference phonemes after the covered span — an ayah left unfinished."""
        return max(0, self.ref_length - self.ref_covered_end)


@dataclass(frozen=True)
class BleedDetector:
    """The reference set plus the mushaf ordering needed to name a clip's neighbours.

    Build with :meth:`load` (which reads the cached reference phonemes) or pass an
    explicit ``references`` mapping — the same ``surah:ayah`` → normalized-phoneme dict
    :mod:`tadabur.reference_phonemes` produces. The mushaf order is computed once here
    rather than per clip, because the detector is run over whole reject piles.
    """

    references: dict[str, str]
    order: tuple[str, ...]
    position: dict[str, int]

    @classmethod
    def of(cls, references: dict[str, str]) -> "BleedDetector":
        """Index ``references`` in mushaf order (surah, then ayah, both numeric)."""
        order = tuple(
            sorted(references, key=lambda key: tuple(int(p) for p in key.split(":")))
        )
        return cls(references, order, {key: i for i, key in enumerate(order)})

    @classmethod
    def load(cls) -> "BleedDetector":
        """Build from the cached reference phonemes (see :mod:`tadabur.reference_phonemes`)."""
        return cls.of(load_reference_phonemes())

    def leading_keys(self, surah_ayah: str) -> tuple[str, ...]:
        """Ayat that could have bled into the *start* of a clip of ``surah_ayah``.

        The previous ayah in mushaf order, plus the basmallah when the clip is the first
        ayah of a surah that opens with one. Both are offered to the aligner at once (see
        :meth:`detect`); it takes whichever the audio actually contains, or neither.
        """
        keys: list[str] = []
        index = self.position.get(surah_ayah)
        if index is None:
            return ()
        if index > 0:
            keys.append(self.order[index - 1])
        surah, ayah = (int(p) for p in surah_ayah.split(":"))
        if ayah == 1 and surah not in SURAHS_WITHOUT_BASMALLAH:
            keys.append(BASMALLAH_KEY)
        return tuple(keys)

    def trailing_keys(self, surah_ayah: str) -> tuple[str, ...]:
        """The ayah that could have bled into the *end* of a clip of ``surah_ayah``."""
        index = self.position.get(surah_ayah)
        if index is None or index + 1 >= len(self.order):
            return ()
        return (self.order[index + 1],)

    def detect(
        self,
        predicted: str,
        surah_ayah: str,
        *,
        min_phonemes: int = MIN_BLEED_PHONEMES,
        min_purity: float = MIN_BLEED_PURITY,
        normalization: PhonemeNormalization | None = None,
    ) -> BleedVerdict:
        """Locate neighbour-ayah bleed at either edge of ``predicted``.

        ``predicted`` is the model's raw decode (it is normalized here, exactly as
        :meth:`tadabur.scorer.Scorer.gate` normalizes it, so the query indices this
        returns are directly comparable with the gate's trims). Pass ``normalization``
        to reuse one already computed. The clip's own ayah must be in ``references``;
        an unknown key yields an empty verdict rather than an exception, matching the
        filter's habit of skipping a clip it cannot reference.
        """
        this = self.references.get(surah_ayah)
        if not this:
            return BleedVerdict(surah_ayah)
        if normalization is None:
            normalization = normalize_phonemes(predicted)
        query = normalization.normalized
        if not query:
            return BleedVerdict(surah_ayah, ref_length=len(this))

        leading_keys = self.leading_keys(surah_ayah)
        trailing_keys = self.trailing_keys(surah_ayah)
        lead_text = REFERENCE_SEPARATOR.join(
            self.references.get(key, "") for key in leading_keys
        )
        trail_text = REFERENCE_SEPARATOR.join(
            self.references.get(key, "") for key in trailing_keys
        )
        this_start = len(lead_text) + len(REFERENCE_SEPARATOR)
        trail_start = this_start + len(this) + len(REFERENCE_SEPARATOR)
        concatenated = (
            lead_text + REFERENCE_SEPARATOR + this + REFERENCE_SEPARATOR + trail_text
        )

        alignment = smith_waterman(query=query, reference=concatenated)
        leading = _edge_bleed(
            alignment, concatenated, 0, this_start, leading_keys, min_phonemes, min_purity
        )
        trailing = _edge_bleed(
            alignment,
            concatenated,
            trail_start,
            len(concatenated),
            trailing_keys,
            min_phonemes,
            min_purity,
        )
        covered_start, covered_end = _covered_reference_span(
            alignment, this_start, this_start + len(this)
        )
        query_start, query_end = _covered_query_span(
            alignment, this_start, this_start + len(this)
        )
        return BleedVerdict(
            surah_ayah=surah_ayah,
            leading=leading,
            trailing=trailing,
            query_length=sum(1 for ch in query if ch != " "),
            query_covered_start=query_start,
            query_covered_end=query_end,
            ref_covered_start=covered_start,
            ref_covered_end=covered_end,
            ref_length=len(this),
        )


def _edge_bleed(
    alignment: AlignmentResult,
    reference: str,
    lo: int,
    hi: int,
    neighbours: tuple[str, ...],
    min_phonemes: int,
    min_purity: float,
) -> EdgeBleed:
    """Attribute the alignment's content inside concatenated-reference ``[lo, hi)``.

    Word-separator spaces are skipped: the normalized query has none, so every reference
    space is a gap whatever the audio contained, and counting them would make purity a
    function of how many words the neighbour region happens to hold.
    """
    matched = span = 0
    query_start = query_end = -1
    for local, info in enumerate(alignment.ref_matches):
        position = alignment.ref_start + local
        if not (lo <= position < hi) or reference[position] == REFERENCE_SEPARATOR:
            continue
        span += 1
        if info.kind == "match":
            matched += 1
        query_index = alignment.ref_to_query[local]
        if query_index >= 0:
            query_start = (
                query_index if query_start < 0 else min(query_start, query_index)
            )
            query_end = max(query_end, query_index + 1)
    purity = matched / span if span else 0.0
    return EdgeBleed(
        neighbours=neighbours,
        matched=matched,
        span=span,
        query_start=max(query_start, 0),
        query_end=max(query_end, 0),
        detected=matched >= min_phonemes and purity >= min_purity,
    )


def _covered_query_span(
    alignment: AlignmentResult, lo: int, hi: int
) -> tuple[int, int]:
    """The half-open normalized-query range aligned to concatenated-reference ``[lo, hi)``.

    This is the span a re-cut must not eat into: whatever the edges are called, the
    recitation of *this* ayah lies between these two decoded phonemes.
    """
    first = last = -1
    for local, info in enumerate(alignment.ref_matches):
        position = alignment.ref_start + local
        if not (lo <= position < hi):
            continue
        query_index = alignment.ref_to_query[local]
        if query_index < 0:
            continue
        if first < 0:
            first = query_index
        last = query_index
    if first < 0:
        return 0, 0
    return first, last + 1


def _covered_reference_span(
    alignment: AlignmentResult, lo: int, hi: int
) -> tuple[int, int]:
    """This ayah's own covered reference span, as offsets into the ayah (not the concat).

    Trailing reference gaps are excluded from the end: a dragged alignment can run its
    ``ref_end`` past the last phoneme it actually matched, and reading truncation off
    that span would under-report exactly the clips that stop early.
    """
    first = last = -1
    for local, info in enumerate(alignment.ref_matches):
        position = alignment.ref_start + local
        if not (lo <= position < hi) or info.kind == "gap":
            continue
        if first < 0:
            first = position
        last = position
    if first < 0:
        return 0, 0
    return first - lo, last - lo + 1


# --- The labelled regression set -------------------------------------------------------
#
# HifzGuide #67 fixes the nine human-adjudicated low-band rejects as the detector's
# regression set. The reviewer's verdicts are free text, so the per-edge labels in
# ``eval_fixtures/reject_bleed_labels.jsonl`` restate them as booleans, each with the
# ``basis`` it was read from. One of the nine departs from the reviewer's note and says
# so in its basis; see ``docs/tadabur-bleed-recut.md``.

DEFAULT_LABELS_PATH = Path(__file__).parent / "eval_fixtures" / "reject_bleed_labels.jsonl"


@dataclass(frozen=True)
class BleedLabel:
    """Ground truth for one adjudicated clip: which edges carry bleed, and why we say so.

    ``truncated`` is the mirror case — an ayah the clip never finishes — and is labelled
    alongside because the same alignment answers both questions and a detector that
    confused them would be worth catching.

    The clip's own ``predicted_phonemes`` and the gate signals the baselines read
    (``match_ratio``, the two trims) are carried on the label rather than joined from a
    reject sink, so re-scoring the detector needs only this file and the reference
    phonemes — not the filtering run that produced it, whose output is not tracked.
    """

    audio_ref: str
    clip_id: str
    surah_ayah: str
    reciter_id: int
    leading_bleed: bool
    trailing_bleed: bool
    truncated: bool
    predicted_phonemes: str = ""
    match_ratio: float = 0.0
    leading_trim: int = 0
    trailing_trim: int = 0
    basis: str = ""


def read_bleed_labels(path: Path = DEFAULT_LABELS_PATH) -> list[BleedLabel]:
    """Load every :class:`BleedLabel` from ``path`` in file order."""
    labels: list[BleedLabel] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            labels.append(
                BleedLabel(
                    audio_ref=data["audio_ref"],
                    clip_id=data["clip_id"],
                    surah_ayah=data["surah_ayah"],
                    reciter_id=data["reciter_id"],
                    leading_bleed=data["leading_bleed"],
                    trailing_bleed=data["trailing_bleed"],
                    truncated=data["truncated"],
                    predicted_phonemes=data.get("predicted_phonemes", ""),
                    match_ratio=data.get("match_ratio", 0.0),
                    leading_trim=data.get("leading_trim", 0),
                    trailing_trim=data.get("trailing_trim", 0),
                    basis=data.get("basis", ""),
                )
            )
    return labels


@dataclass(frozen=True)
class Score:
    """Precision/recall of one predicate against a labelled set.

    ``true_negatives`` is carried because the labelled set is small enough that "fired on
    nothing the reviewer heard as clean" is itself a result worth reading — the issue's
    bar is that a detector firing on a clean clip is *worse* than none, since it would
    clip real recitation.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @classmethod
    def of(cls, outcomes: list[tuple[bool, bool]]) -> "Score":
        """Tally ``(predicted, actual)`` pairs."""
        return cls(
            true_positives=sum(1 for p, a in outcomes if p and a),
            false_positives=sum(1 for p, a in outcomes if p and not a),
            false_negatives=sum(1 for p, a in outcomes if not p and a),
            true_negatives=sum(1 for p, a in outcomes if not p and not a),
        )

    @property
    def precision(self) -> float:
        fired = self.true_positives + self.false_positives
        return self.true_positives / fired if fired else 1.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 1.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
        }


# The edge-trim baseline the issue measured first: fire when the local aligner trimmed
# this many or more non-space query phonemes off an edge. Reproduced here so the report's
# comparison is recomputed rather than quoted.
TRIM_BASELINE_PHONEMES = 5

# The per-ayah reference-comparison baseline: cut the query into this many equal slices
# and ask which of prev / this / next each end slice aligns to best.
REFERENCE_BASELINE_SLICES = 4


def trim_baseline(record) -> tuple[bool, bool]:  # RejectRecord, typed loosely to stay import-light
    """``(leading, trailing)`` verdicts of the edge-trim baseline for one reject."""
    return (
        record.leading_trim >= TRIM_BASELINE_PHONEMES,
        record.trailing_trim >= TRIM_BASELINE_PHONEMES,
    )


def reference_baseline(
    detector: BleedDetector, predicted: str, surah_ayah: str
) -> tuple[bool, bool]:
    """``(leading, trailing)`` verdicts of the per-ayah reference-comparison baseline.

    The first and last query slice are each scored against the previous, this and next
    ayah separately; an end owned by a neighbour is called bleed. It finds nothing on a
    re-read, because a re-read's repeated tail is *this* ayah's words and so legitimately
    scores best against this ayah — which is the point of reproducing it here.
    """
    from .smith_waterman import local_alignment_score

    this = detector.references.get(surah_ayah, "")
    if not this:
        return False, False
    query = normalize_phonemes(predicted).normalized
    if len(query) < REFERENCE_BASELINE_SLICES:
        return False, False
    width = len(query) // REFERENCE_BASELINE_SLICES
    leading_keys = detector.leading_keys(surah_ayah)
    trailing_keys = detector.trailing_keys(surah_ayah)

    def owner(slice_text: str, neighbour_keys: tuple[str, ...]) -> bool:
        own = local_alignment_score(slice_text, this)
        return any(
            local_alignment_score(slice_text, detector.references.get(key, "")) > own
            for key in neighbour_keys
        )

    return owner(query[:width], leading_keys), owner(query[-width:], trailing_keys)


# The listening-set bands the shard-20 adjudication was organised into: the repeat-carrying
# rejects (``max_insertion_run >= MAX_INSERTION_RUN``) split by where ``match_ratio`` falls
# relative to the dropped clean-re-read floor, with added-shadda clips held aside. Restated
# here so the prevalence table is keyed the same way the human verdicts are.
BAND_SHADDA = "shadda"
BAND_CLEAN = "clean"
BAND_MARGINAL = "band"
BAND_LOW = "low"
BAND_NOT_REPEAT = "no_repeat"
REJECT_BANDS = (BAND_LOW, BAND_MARGINAL, BAND_CLEAN, BAND_SHADDA, BAND_NOT_REPEAT)


def reject_band(record) -> str:
    """Which adjudication band a reject falls in (see :data:`REJECT_BANDS`)."""
    from .rejects import CLEAN_RE_READ_MIN_RATIO
    from .scorer import MAX_INSERTION_RUN

    if record.max_insertion_run < MAX_INSERTION_RUN:
        return BAND_NOT_REPEAT
    if record.added_shadda:
        return BAND_SHADDA
    if record.match_ratio >= CLEAN_RE_READ_MIN_RATIO:
        return BAND_CLEAN
    if record.match_ratio >= 0.70:
        return BAND_MARGINAL
    return BAND_LOW


def _score_against_labels(detector: BleedDetector, labels: list[BleedLabel]) -> dict:
    """Score the detector and both reproduced baselines against the labelled clips.

    Reads everything from the labels themselves — each row carries its own decode and
    gate signals — so a re-score is one command against a tracked fixture.
    """
    predicates: dict[str, dict[str, list[tuple[bool, bool]]]] = {
        name: {"leading": [], "trailing": [], "clip": []}
        for name in ("detector", "trim_baseline", "reference_baseline")
    }
    rows: list[dict] = []
    for label in labels:
        verdict = detector.detect(label.predicted_phonemes, label.surah_ayah)
        calls = {
            "detector": (verdict.leading.detected, verdict.trailing.detected),
            "trim_baseline": trim_baseline(label),
            "reference_baseline": reference_baseline(
                detector, label.predicted_phonemes, label.surah_ayah
            ),
        }
        for name, (leading, trailing) in calls.items():
            predicates[name]["leading"].append((leading, label.leading_bleed))
            predicates[name]["trailing"].append((trailing, label.trailing_bleed))
            predicates[name]["clip"].append(
                (leading or trailing, label.leading_bleed or label.trailing_bleed)
            )
        rows.append(
            {
                "surah_ayah": label.surah_ayah,
                "audio_ref": label.audio_ref,
                "match_ratio": label.match_ratio,
                "label": {
                    "leading": label.leading_bleed,
                    "trailing": label.trailing_bleed,
                    "truncated": label.truncated,
                },
                "detector": {
                    "leading": _edge_dict(verdict.leading),
                    "trailing": _edge_dict(verdict.trailing),
                    "uncovered_head": verdict.uncovered_head,
                    "uncovered_tail": verdict.uncovered_tail,
                },
                "trim_baseline": list(calls["trim_baseline"]),
                "reference_baseline": list(calls["reference_baseline"]),
            }
        )
    truncation = Score.of(
        [
            (
                detector.detect(label.predicted_phonemes, label.surah_ayah).uncovered_tail
                >= TRUNCATION_PHONEMES,
                label.truncated,
            )
            for label in labels
        ]
    )
    return {
        "labelled_clips": len(rows),
        "scores": {
            name: {edge: Score.of(pairs).as_dict() for edge, pairs in edges.items()}
            for name, edges in predicates.items()
        },
        "truncation": truncation.as_dict(),
        "clips": rows,
    }


def _edge_dict(edge: EdgeBleed) -> dict:
    return {
        "detected": edge.detected,
        "matched": edge.matched,
        "span": edge.span,
        "purity": round(edge.purity, 3),
        "query_start": edge.query_start,
        "query_end": edge.query_end,
        "neighbours": list(edge.neighbours),
    }


def _prevalence(detector: BleedDetector, records: list) -> dict:
    """Bleed prevalence over a reject pile, by adjudication band."""
    bands: dict[str, Counter] = {band: Counter() for band in REJECT_BANDS}
    for record in records:
        band = bands[reject_band(record)]
        verdict = detector.detect(record.predicted_phonemes, record.surah_ayah)
        band["clips"] += 1
        band["leading"] += int(verdict.leading.detected)
        band["trailing"] += int(verdict.trailing.detected)
        band["either"] += int(verdict.detected)
        band["truncated"] += int(verdict.uncovered_tail >= TRUNCATION_PHONEMES)
        if verdict.detected:
            band["bleed_phonemes"] += (
                verdict.leading.query_phonemes * int(verdict.leading.detected)
                + verdict.trailing.query_phonemes * int(verdict.trailing.detected)
            )
    return {
        band: {
            "clips": counts["clips"],
            "leading": counts["leading"],
            "trailing": counts["trailing"],
            "either": counts["either"],
            "truncated": counts["truncated"],
            "bleed_phonemes": counts["bleed_phonemes"],
            "share_with_bleed": round(counts["either"] / counts["clips"], 3)
            if counts["clips"]
            else 0.0,
        }
        for band, counts in bands.items()
        if counts["clips"]
    }


# An ayah is called *truncated* when this many of its reference phonemes fall after the
# decode's last matched position — roughly a short word. Below it, an unmatched tail is
# more likely the model dropping a final phoneme than the reciter stopping early.
TRUNCATION_PHONEMES = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rejects",
        type=Path,
        help="Reject sink JSONL to measure bleed prevalence over (optional).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=f"Labelled regression set (default: {DEFAULT_LABELS_PATH}).",
    )
    parser.add_argument("--json", type=Path, help="Write the full report here.")
    args = parser.parse_args()

    detector = BleedDetector.load()
    report: dict = {
        "labelled": _score_against_labels(detector, read_bleed_labels(args.labels)),
        "thresholds": {
            "min_bleed_phonemes": MIN_BLEED_PHONEMES,
            "min_bleed_purity": MIN_BLEED_PURITY,
            "trim_baseline_phonemes": TRIM_BASELINE_PHONEMES,
            "truncation_phonemes": TRUNCATION_PHONEMES,
        },
    }

    if args.rejects:
        from .rejects import read_reject_records

        report["prevalence"] = _prevalence(detector, read_reject_records(args.rejects))

    for name, scores in report["labelled"]["scores"].items():
        clip = scores["clip"]
        print(
            f"{name:20s} clip P={clip['precision']:.2f} R={clip['recall']:.2f} "
            f"(tp={clip['true_positives']} fp={clip['false_positives']} "
            f"fn={clip['false_negatives']} tn={clip['true_negatives']})"
        )
    print()
    for band, counts in report.get("prevalence", {}).items():
        print(
            f"{band:10s} {counts['clips']:3d} clips  bleed {counts['either']:3d} "
            f"({counts['share_with_bleed']:.0%})  lead {counts['leading']:3d} "
            f"trail {counts['trailing']:3d}  truncated {counts['truncated']:3d}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

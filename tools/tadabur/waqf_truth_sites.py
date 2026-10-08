"""Convert the adjudicated waqf boundaries into tashkeel truth sites (#79).

The 2,050 boundaries in ``waqf_event_fixtures/waqf_events.{calibration,test}.jsonl`` are
human verdicts on whole Tadabur clips: after word ``word_index`` the reciter made
``waqf`` (stopped), ``wasl`` (continued), or the silence was a ``mid_word_closure``.
ADR-0011 retired the waqf head they were collected for, but each verdict still fixes the
tashkeel a correct recitation carries on the boundary word's last letter:

* ``waqf`` -> the word is recited in waqf form. Where that ends in a consonant, the site is
  **sukun at a pause**, and the pause the human heard is the evidence for it.
* ``wasl`` -> the word keeps its wasl ending: a **word-final haraka** (or the sukun the
  mushaf writes there). The human heard continuation, not the mark, so these sites
  ``assumes_competent_reciter`` — the weaker label.
* ``mid_word_closure`` -> not a word boundary; excluded and counted.

A site's position is a ``reference_index`` into the **whole clip's realized reference**:
the clip's words split at every human ``waqf``, each recited run phonetized on its own
(terminal word in waqf form, the rest in wasl), the runs joined by a space. A reciter who
re-reads appears as a run that restarts at an earlier word. The clip is the item, so its
provenance is ``start_sample = 0``; ``shard``, ``end_sample`` (the staged clip's length)
and ``audio_sha256`` come from the staged-clip registry (:mod:`tadabur.staged_audio`, #83),
and a clip the registry lacks keeps them ``null`` and is listed in the summary.

Every fixture row ends in exactly one place — a site, or one exclusion
bucket — and the summary written beside the sites accounts for all of them. The fixture
JSONL is read directly; this module depends on nothing of the retired waqf eval.

Usage (from ``tools/``, needs ``quran-transcript``):
  python -m tadabur.waqf_truth_sites
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hafs_phonetizer import phonetize
# Re-exported while #83 still imports it from here; new code imports hafs_phonetizer.
from hafs_phonetizer import pausal_taa_marbuta  # noqa: F401
from training.tashkeel_worklist import VOWEL_NAMES

from .staged_audio import REGISTRY_PATH, fill_staging, load_staged_clips
from .truth_sites import (
    CONSONANTS,
    SUKUN,
    TRUTH_SITES_DIR,
    WAQF_BOUNDARY,
    TruthSite,
    write_truth_sites,
)

_FIXTURE_DIR = Path(__file__).parent / "waqf_event_fixtures"
FIXTURE_PATHS = (
    _FIXTURE_DIR / "waqf_events.calibration.jsonl",
    _FIXTURE_DIR / "waqf_events.test.jsonl",
)
SITES_PATH = TRUTH_SITES_DIR / "waqf_boundaries.jsonl"
SUMMARY_PATH = TRUTH_SITES_DIR / "waqf_boundaries.summary.json"

# The fixture's three verdict classes.
WAQF = "waqf"
WASL = "wasl"
MID_WORD_CLOSURE = "mid_word_closure"
_VERDICTS = frozenset({WAQF, WASL, MID_WORD_CLOSURE})

# Why a fixture row yields no site. Each row lands in exactly one bucket or one site.
#: The human said the silence was a closure inside a word, not a word boundary.
EXCLUDED_MID_WORD_CLOSURE = "mid_word_closure"
#: A closure-candidate row folded into the regular edge after the same word; its ``waqf``
#: verdict, if any, was applied to that edge.
EXCLUDED_CLOSURE_MERGED = "closure_merged"
#: A clip with a closure judged ``waqf`` on a word it recites twice (a re-read): the pause
#: cannot be placed on either pass, so the clip has no trustworthy reference.
EXCLUDED_CLOSURE_AMBIGUOUS = "closure_ambiguous"
#: The clip's edges are not one recitation (a step back to an earlier word with no waqf
#: before it, or a word past the ayah's end), so no realized reference can be built.
EXCLUDED_INCONSISTENT_CLIP = "inconsistent_clip"
#: quran-transcript cannot phonetize one of the clip's recited runs (it ends in waqf on a
#: leen letter before the final consonant, e.g. ``شَىْءٍ``), so the clip has no reference.
EXCLUDED_PHONETIZER_UNSUPPORTED = "phonetizer_unsupported"
#: The word's last letter is not realized with a mark of its own: absorbed into the next
#: word (idgham) or nasalized into it (ikhfa / iqlab).
EXCLUDED_FINAL_LETTER_ASSIMILATED = "final_letter_assimilated"


class ClipExcluded(Exception):
    """A clip yields no realized reference; ``reason`` is the exclusion bucket its rows take."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Boundary:
    """The fields of one adjudicated fixture row this conversion reads."""

    clip_id: str
    surah_ayah: str
    boundary_index: int
    word_index: int
    predicted: str
    verdict: str


def read_boundaries(paths: Sequence[Path] = FIXTURE_PATHS) -> list[Boundary]:
    """Every fixture row, in file order, failing loudly on a row that is not a boundary.

    ``audio_ref`` must equal ``clip_id``: the boundary times are whole-clip times, which is
    what lets the whole clip be the truth-site item without re-segmentation.
    """
    boundaries: list[Boundary] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                where = f"{path}:{lineno}"
                if row["audio_ref"] != row["clip_id"]:
                    raise ValueError(f"{where}: audio_ref is not the whole clip")
                if row["predicted"] not in _VERDICTS or row["verdict"] not in _VERDICTS:
                    raise ValueError(f"{where}: unknown boundary class")
                boundaries.append(
                    Boundary(
                        clip_id=row["clip_id"],
                        surah_ayah=row["surah_ayah"],
                        boundary_index=row["boundary_index"],
                        word_index=row["word_index"],
                        predicted=row["predicted"],
                        verdict=row["verdict"],
                    )
                )
    return boundaries


@dataclass(frozen=True)
class Edge:
    """One human-adjudicated word edge: a pause (``waqf``) or not, after ``word_index``.

    ``boundary_index`` is the fixture row the edge's site is keyed by.
    """

    boundary_index: int
    word_index: int
    waqf: bool


def clip_edges(rows: list[Boundary]) -> tuple[list[Edge], dict[int, str]]:
    """One clip's word edges in time order, and the exclusion reason of every other row.

    Regular rows (a segment boundary or an interior word edge the detector proposed) are
    edges. A closure-candidate row is a VAD silence the detector placed inside a word; its
    word comes from the phoneme alignment, while the regular edges' times are interpolated,
    so where it sits in time says nothing about which edge it is. A closure the human
    called a stop or a continuation is therefore reconciled to the regular edge **after the
    same word**:

    * one such edge -> the row is merged into it, and a ``waqf`` verdict makes it a pause
      (the human's explicit stop outranks the detector's unflipped default there);
    * none -> the row is an edge of its own, at its place in time;
    * several (a re-read passes the word twice) -> a ``waqf`` cannot be placed on either
      pass, and a reference that continues through a known pause would assert false
      tashkeel, so the whole clip is excluded (:class:`ClipExcluded`). A ``wasl`` there
      changes nothing and is merged.
    """
    rows = sorted(rows, key=lambda r: r.boundary_index)
    excluded = {
        r.boundary_index: EXCLUDED_MID_WORD_CLOSURE for r in rows if r.verdict == MID_WORD_CLOSURE
    }
    judged = [r for r in rows if r.boundary_index not in excluded]
    regular_at: dict[int, list[Boundary]] = {}
    for row in judged:
        if row.predicted != MID_WORD_CLOSURE:
            regular_at.setdefault(row.word_index, []).append(row)
    pauses = {r.boundary_index for r in judged if r.verdict == WAQF}

    edge_rows: list[Boundary] = []
    for row in judged:
        matches = regular_at.get(row.word_index, [])
        if row.predicted != MID_WORD_CLOSURE or not matches:
            edge_rows.append(row)
            continue
        if row.verdict == WAQF and len(matches) > 1:
            raise ClipExcluded(EXCLUDED_CLOSURE_AMBIGUOUS)
        excluded[row.boundary_index] = EXCLUDED_CLOSURE_MERGED
        if row.verdict == WAQF:
            pauses.add(matches[0].boundary_index)

    edges = [Edge(r.boundary_index, r.word_index, r.boundary_index in pauses) for r in edge_rows]
    return edges, excluded


@dataclass(frozen=True)
class RecitedRun:
    """Words ``[word_start, word_end)`` recited without a pause, and the edges inside it.

    The run's last edge is its terminal pause when it ends in one; every other edge is a
    continuation between two of its words.
    """

    word_start: int
    word_end: int
    edges: tuple[Edge, ...]


def recited_runs(edges: list[Edge], n_words: int) -> list[RecitedRun]:
    """Split a clip's edges into the runs recited between pauses.

    A run ends at each ``waqf`` edge. After a pause the reciter resumes at the next edge's
    word when that is not past the stop (a re-read), else at the word after the stop. The
    clip's last run ends one word after its last edge (a run of two or more words would
    have had an edge inside it). Raises :class:`ClipExcluded` when the edges cannot be one
    recitation: a step back to an earlier word with no pause before it, or a word past the
    ayah's end.
    """
    if not edges:
        return []
    runs: list[RecitedRun] = []
    start = edges[0].word_index
    current: list[Edge] = []
    previous: Edge | None = None
    for edge in edges:
        if edge.word_index >= n_words:
            raise ClipExcluded(EXCLUDED_INCONSISTENT_CLIP)
        if previous is not None and previous.waqf:
            start = min(edge.word_index, previous.word_index + 1)
        elif previous is not None and edge.word_index <= previous.word_index:
            raise ClipExcluded(EXCLUDED_INCONSISTENT_CLIP)
        current.append(edge)
        if edge.waqf:
            runs.append(RecitedRun(start, edge.word_index + 1, tuple(current)))
            current = []
        previous = edge

    last = edges[-1]
    if not last.waqf:
        if last.word_index + 2 > n_words:
            raise ClipExcluded(EXCLUDED_INCONSISTENT_CLIP)
        runs.append(RecitedRun(start, last.word_index + 2, tuple(current)))
    elif last.word_index + 1 < n_words:
        runs.append(RecitedRun(last.word_index + 1, last.word_index + 2, ()))
    return runs


@dataclass(frozen=True)
class RealizedRun:
    """A recited run's realized reference and, per word, the carrier of its final mark.

    ``word_ends[i]`` indexes ``phonemes`` at the consonant carrying word ``i``'s final mark,
    or is ``None`` when that word's last letter has no mark of its own (assimilated).
    """

    phonemes: str
    word_ends: tuple[int | None, ...]


#: Turns a run's Uthmani words into its realized reference (terminal word in waqf form),
#: raising :class:`ClipExcluded` when it cannot.
Realizer = Callable[[list[str]], RealizedRun]
#: The Uthmani words of ``"surah:ayah"``.
UthmaniWords = Callable[[str], list[str]]


def final_mark(phonemes: str, carrier: int) -> str:
    """The tashkeel on ``phonemes[carrier]``: the haraka after it, else sukun."""
    return VOWEL_NAMES.get(phonemes[carrier + 1 : carrier + 2], SUKUN)


def convert(
    boundaries: list[Boundary], uthmani_words: UthmaniWords, realize: Realizer
) -> tuple[list[TruthSite], dict]:
    """Truth sites for every convertible boundary, plus the summary accounting for all rows.

    Clips are processed in ``clip_id`` order and sites emitted in time order within a clip,
    so the output is a pure function of the fixture rows.
    """
    by_clip: dict[str, list[Boundary]] = {}
    for boundary in boundaries:
        by_clip.setdefault(boundary.clip_id, []).append(boundary)

    # Where each fixture row went: a site's stratum, or an exclusion reason.
    outcomes: dict[tuple[str, int], str] = {}
    drafts: list[tuple[Boundary, Edge, str, int]] = []  # row, edge, reference, carrier index
    for clip_id in sorted(by_clip):
        rows = {r.boundary_index: r for r in by_clip[clip_id]}
        if len(rows) != len(by_clip[clip_id]):
            raise ValueError(f"{clip_id}: duplicate boundary_index")
        words = uthmani_words(next(iter(rows.values())).surah_ayah)
        try:
            edges, dropped = clip_edges(list(rows.values()))
            reference, carriers = _locate_edges(edges, words, realize)
        except ClipExcluded as excluded:
            outcomes.update(
                ((clip_id, index), _clip_exclusion(row, excluded.reason))
                for index, row in rows.items()
            )
            continue
        outcomes.update(((clip_id, index), reason) for index, reason in dropped.items())
        for edge, carrier in zip(edges, carriers):
            if carrier is None:
                outcomes[(clip_id, edge.boundary_index)] = EXCLUDED_FINAL_LETTER_ASSIMILATED
                continue
            outcomes[(clip_id, edge.boundary_index)] = _stratum(edge)
            drafts.append((rows[edge.boundary_index], edge, reference, carrier))

    populations = Counter(_stratum(edge) for _, edge, _, _ in drafts)
    sites = []
    for row, edge, reference, index in drafts:
        mark = final_mark(reference, index)
        sites.append(
            TruthSite(
                site_id=f"{WAQF_BOUNDARY}:{row.clip_id}#{row.boundary_index}",
                source=WAQF_BOUNDARY,
                assumes_competent_reciter=not (edge.waqf and mark == SUKUN),
                audio_filename=row.clip_id,
                shard=None,
                start_sample=0,
                end_sample=None,
                audio_sha256=None,
                surah_ayah=row.surah_ayah,
                reference=reference,
                reference_index=index,
                mark=mark,
                prescribed=mark,
                heard=mark,
                stratum=_stratum(edge),
                stratum_population=populations[_stratum(edge)],
            )
        )
    return sites, _summary(boundaries, sites, outcomes)


def _locate_edges(
    edges: list[Edge], words: list[str], realize: Realizer
) -> tuple[str, list[int | None]]:
    """The clip's realized reference and each edge's carrier index in it, in edge order
    (``None`` when the boundary word's last letter is assimilated)."""
    runs = recited_runs(edges, len(words))
    realized = [realize(words[run.word_start : run.word_end]) for run in runs]

    carriers: list[int | None] = []
    offset = 0
    for run, real in zip(runs, realized):
        for edge in run.edges:
            carrier = real.word_ends[edge.word_index - run.word_start]
            carriers.append(None if carrier is None else offset + carrier)
        offset += len(real.phonemes) + 1  # + the space joining the runs
    return " ".join(real.phonemes for real in realized), carriers


def _clip_exclusion(row: Boundary, reason: str) -> str:
    """A row's outcome in an excluded clip: a mid-word closure stays one."""
    return EXCLUDED_MID_WORD_CLOSURE if row.verdict == MID_WORD_CLOSURE else reason


def _stratum(edge: Edge) -> str:
    return f"{WAQF_BOUNDARY}:{WAQF if edge.waqf else WASL}"


def _summary(
    boundaries: list[Boundary], sites: list[TruthSite], outcomes: dict[tuple[str, int], str]
) -> dict:
    """Where every fixture row went, by its verdict. Raises if any row is unaccounted for."""
    keys = {(b.clip_id, b.boundary_index) for b in boundaries}
    if outcomes.keys() != keys:
        raise AssertionError(f"{len(keys - outcomes.keys())} fixture rows have no outcome")
    rows_by_verdict: dict[str, Counter[str]] = {}
    for b in boundaries:
        outcome = outcomes[(b.clip_id, b.boundary_index)]
        rows_by_verdict.setdefault(b.verdict, Counter())[outcome] += 1
    marks_by_stratum: dict[str, Counter[str]] = {}
    for site in sites:
        marks_by_stratum.setdefault(site.stratum, Counter())[site.mark] += 1
    return {
        "fixture_rows": len(boundaries),
        "clips": len({b.clip_id for b in boundaries}),
        "clips_with_sites": len({s.audio_filename for s in sites}),
        "sites": len(sites),
        "sites_assuming_competent_reciter": sum(s.assumes_competent_reciter for s in sites),
        "sites_by_stratum_and_mark": _sorted(marks_by_stratum),
        "rows_by_verdict_and_outcome": _sorted(rows_by_verdict),
    }


def _sorted(table: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(sorted(counts.items())) for key, counts in sorted(table.items())}


# --- the Hafs phonetizer realizer ------------------------------------------------------

#: A long vowel's extension in the realized reference: the mark sits on the letter before.
_MADD = frozenset("\u0627\u06e5\u06e6")  # ا ۥ ۦ
#: Uthmani letters the phonetizer drops without moving them into the next word: a silent
#: alif, or a madd letter shortened before hamzat wasl.
_DROPPABLE_LETTERS = frozenset("\u0627\u0649\u0648\u064a")  # ا ى و ي
#: Ghunna noon / meem: a noon or meem sakin nasalized into the next letter.
_GHUNNA = frozenset("\u06ba\u06fe")  # ں ۾
#: The qalqala marker the phonetizer writes after a bouncing sakin letter.
_QALQALA = "\u0687"  # ڇ
_SHADDA = "\u0651"
#: Hamza written as a mark (on a tatweel or a seat, e.g. ``شَىْـًٔا``): a letter of its own.
_HAMZA_MARKS = frozenset("\u0654\u0655")
#: The consonants an Uthmani letter is realized as when it keeps its own identity; any
#: other letter is realized as itself. A final letter realized as a *different* consonant
#: was assimilated into the next word (noon sakin into ي / و: ``أَن يَ`` -> ``ءَييَ``).
_REALIZED_AS = {
    "\u0629": "\u062a\u0647",  # ة -> ت (wasl) / ه (waqf)
    "\u0649": "\u064a",  # ى -> ي
    # every hamza form (ء آ أ ؤ إ ئ and the hamza marks) -> ء
    **dict.fromkeys("\u0621\u0622\u0623\u0624\u0625\u0626\u0654\u0655", "\u0621"),
}


def final_carrier(
    text: str, phonemes: str, spans: list[tuple[int, int]], start: int, end: int
) -> int | None:
    """Index in ``phonemes`` of the consonant carrying the final mark of ``text[start:end]``.

    ``spans[i]`` is the ``[from, to)`` slice of ``phonemes`` that input character ``i``
    became (the phonetizer's char mappings). Walks the word's letters from the end. A
    letter's realization is its own span joined with its shaddah's (the doubled consonant;
    the mark sits on the second). A letter that became nothing is skipped when it is a
    droppable madd/silent letter, and otherwise was absorbed into the next word (``None``);
    a letter realized as a madd extension passes the mark to the letter before it; a
    letter realized as ghunna, or as a consonant other than itself, was assimilated into
    the next word (``None``).
    """
    letters = [
        i for i in range(start, end)
        if unicodedata.category(text[i]) == "Lo" or text[i] in _HAMZA_MARKS
    ]
    for position in reversed(range(len(letters))):
        letter = letters[position]
        span_start, span_end = spans[letter]
        if span_start == span_end:
            if text[letter] in _DROPPABLE_LETTERS:
                continue
            return None
        marks_end = letters[position + 1] if position + 1 < len(letters) else end
        for mark in range(letter + 1, marks_end):
            if text[mark] == _SHADDA:
                span_end = max(span_end, spans[mark][1])
        realized = phonemes[span_start:span_end].rstrip(_QALQALA)
        if realized[-1] in _MADD:
            continue
        if realized[-1] in _GHUNNA:
            return None
        if realized[-1] not in CONSONANTS:
            raise ValueError(
                f"{text[start:end]!r}: {text[letter]!r} realized as {realized!r}"
            )
        if realized[-1] not in _REALIZED_AS.get(text[letter], text[letter]):
            return None
        return span_start + len(realized) - 1
    raise ValueError(f"{text[start:end]!r} has no realized letter")


def hafs_realizer() -> Realizer:
    """A :data:`Realizer` over :func:`hafs_phonetizer.phonetize`.

    The repo's one phonetizer entry point (Hafs moshaf, pausal ``ةً`` fixed), so a run's
    phonemes are exactly what :func:`tadabur.waqf_segments.hafs_phonetizer` would give for
    the same words.
    """

    def realize(words: list[str]) -> RealizedRun:
        text = " ".join(words)
        try:
            out = phonetize(text)
        except (KeyError, IndexError) as exc:  # a waqf on a leen ending, e.g. شَىْءٍ
            raise ClipExcluded(EXCLUDED_PHONETIZER_UNSUPPORTED) from exc
        spans = [m.pos for m in out.mappings]
        ends: list[int | None] = []
        cursor = 0
        for word in words:
            ends.append(final_carrier(text, out.phonemes, spans, cursor, cursor + len(word)))
            cursor += len(word) + 1
        return RealizedRun(out.phonemes, tuple(ends))

    return realize


def summary_table(summary: dict) -> str:
    """The summary's two count tables in Markdown, as ``truth_sites/README.md`` carries them."""
    return "\n\n".join(
        _markdown_table(corner, summary[key])
        for corner, key in (
            ("stratum \\ mark", "sites_by_stratum_and_mark"),
            ("fixture verdict \\ outcome", "rows_by_verdict_and_outcome"),
        )
    )


def _markdown_table(corner: str, table: dict[str, dict[str, int]]) -> str:
    columns = sorted({column for counts in table.values() for column in counts})
    lines = [
        f"| {corner} | " + " | ".join(f"`{c}`" for c in columns) + " | total |",
        "|---" * (len(columns) + 2) + "|",
    ]
    for row, counts in table.items():
        cells = " | ".join(str(counts.get(c, 0)) for c in columns)
        lines.append(f"| `{row}` | {cells} | {sum(counts.values())} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=SITES_PATH,
                        help="truth-site JSONL to (over)write.")
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH,
                        help="summary JSON to (over)write.")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH,
                        help="staged-clip registry the staging fields are filled from (#83).")
    args = parser.parse_args()

    from .waqf_segments import _uthmani_words

    sites, summary = convert(read_boundaries(), _uthmani_words, hafs_realizer())
    sites, not_restaged = fill_staging(sites, load_staged_clips(args.registry))
    summary["clips_not_restaged"] = not_restaged
    write_truth_sites(sites, args.out)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary_table(summary))
    print(f"\nWrote {len(sites)} sites to {args.out} and the summary to {args.summary}")


if __name__ == "__main__":
    main()

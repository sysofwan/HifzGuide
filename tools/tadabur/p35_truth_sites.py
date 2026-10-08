"""Re-locate the P3.5 fixtures on today's segments and convert them to truth sites (#83).

The P3.5 poison audit labelled **segments**: each fixture id ``<clip>__seg<n>.wav`` names
segment ``n`` of a whole clip, and the human judged one contrast bucket on it (a soft
pair, ``shadda``, or the ``marginal`` band), with ``accept`` or ``reject``. The segment
boundaries were lost with ``audit_run/``. :mod:`tadabur.resegment` re-derives every
clip's segments with today's VAD and segmentation and decodes each with the base
teacher; this module decides, per fixture, whether its label still has a site.

**The re-location rule** (:func:`relocate`). A fixture keeps its label only if

1. its bucket names a contrast (``marginal`` has no site: excluded);
2. its clip was re-staged, segmented, and segment ``n`` exists and survived
   :mod:`tadabur.segment_score`'s drop rules; and
3. the base decode of that segment still shows the labelled contrast against its
   realized reference (:func:`tadabur.contrast_attribution.contrast_sites`): the soft-pair
   substitution for a pair bucket, a gemination mismatch for ``shadda``. Each occurrence
   is one site, on its carrier in the realized reference, provided the truth-site schema
   accepts that carrier (``contrast_not_expressible`` otherwise).

Anything else is dropped and counted by reason, so no label is attached to audio it was
not given on.

**What a verdict says at a site.** An ``accept`` judged the clip acceptable for that
contrast: the mushaf's letter (or gemination state) was said at every occurrence, so
``heard = prescribed``. A ``reject`` is a **clip-level** verdict and does not establish
that the other letter was said at a given site: several carry notes such as "Not hafs",
"Segmentation issue" or "Unclear reading". Re-located reject sites therefore get
``heard = pending`` (:data:`tadabur.truth_sites.PENDING`): defined, staged, and waiting
for the site-level re-adjudication the listening session gives all nominal rejects (#87).

For ``shadda`` the decode either has one consonant where the reference geminates
(prescribed ``held``) or doubles a single one (prescribed ``not_held``).

Every site is ``assumes_competent_reciter: false`` (the human judged the contrast
itself), and its stratum ``p35_fixture:<bucket>`` is a census of the re-located
fixtures: population = site count, weight 1. These are targeted safeguards, not a sample
that weights back to the corpus.

Usage (from ``tools/``, on the output of ``tadabur.resegment --use p35_fixture``)::

  python -m tadabur.p35_truth_sites --seg-dir stage/seg_p35
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .contrast_attribution import (
    DROPPED,
    MARGINAL_CONTRAST,
    SHADDA_CONTRAST,
    contrast_sites,
)
from .eval_fixtures import ACCEPT, EvalFixtureEntry, load_should_accept, load_should_reject
from .segment_score import parse_segment_id, segment_sample_bounds
from .staged_audio import StagedClip, load_staged_clips
from .truth_sites import (
    HELD,
    NOT_HELD,
    P35_FIXTURE,
    PENDING,
    SHADDAH,
    TRUTH_SITES_DIR,
    TruthSite,
    parse_site,
    write_truth_sites,
)

SITES_PATH = TRUTH_SITES_DIR / "p35_fixtures.jsonl"
RELOCATION_PATH = TRUTH_SITES_DIR / "p35_fixtures.relocation.jsonl"
SUMMARY_PATH = TRUTH_SITES_DIR / "p35_fixtures.summary.json"

KEPT = "kept"
# Why a fixture yields no site. Each fixture lands in exactly one outcome.
EXCLUDED_MARGINAL = "marginal_no_site"
DROPPED_NOT_STAGED = "clip_not_staged"
DROPPED_UNSEGMENTED = "clip_not_segmented"
DROPPED_SEGMENT_MISSING = "segment_missing"
DROPPED_SEGMENT_DROPPED = "segment_dropped"
DROPPED_CONTRAST_ABSENT = "contrast_absent"
DROPPED_NOT_EXPRESSIBLE = "contrast_not_expressible"


@dataclass(frozen=True)
class Segment:
    """One re-derived segment of a whole clip, as :mod:`tadabur.resegment` left it.

    ``reference`` (the realized reference) and ``decode`` (the base teacher's) are
    ``None`` for a segment the drop rules removed; ``drops`` is then its clip's tally.
    """

    audio_filename: str
    segment_index: int
    start_s: float
    end_s: float
    kept: bool
    reference: str | None
    decode: str | None
    drops: tuple[str, ...]


@dataclass(frozen=True)
class Relocation:
    """Where one fixture went: its outcome, the segment it was looked for in, its sites."""

    fixture: EvalFixtureEntry
    outcome: str
    segment: Segment | None
    sites: tuple[TruthSite, ...]


def read_segments(seg_dir: Path) -> dict[tuple[str, int], Segment]:
    """Every re-derived segment in a ``tadabur.resegment`` output, keyed by
    ``(audio_filename, segment_index)``."""
    kept_rows = {
        (row["clip_audio_filename"], row["segment_index"]): row
        for row in _jsonl(seg_dir / "segment_manifest.jsonl")
    }
    segments: dict[tuple[str, int], Segment] = {}
    for clip in _jsonl(seg_dir / "segmentation.jsonl"):
        drops = tuple(sorted(clip["drops"]))
        for seg in clip["segments"]:
            key = (clip["audio_filename"], seg["segment_index"])
            row = kept_rows.get(key)
            if seg["kept"] != (row is not None):
                raise ValueError(f"{seg_dir}: segmentation and manifest disagree on {key}")
            segments[key] = Segment(
                audio_filename=clip["audio_filename"],
                segment_index=seg["segment_index"],
                start_s=seg["start_s"],
                end_s=seg["end_s"],
                kept=seg["kept"],
                reference=None if row is None else row["raw_reference_phonemes"],
                decode=None if row is None else row["predicted_phonemes"],
                drops=() if seg["kept"] else drops,
            )
    return segments


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(raw) for raw in f if raw.strip()]


def relocate(
    fixture: EvalFixtureEntry,
    segments: dict[tuple[str, int], Segment],
    staged: dict[str, StagedClip],
    segmented_clips: set[str],
) -> Relocation:
    """Apply the re-location rule (module docstring) to one fixture.

    Sites are returned without a stratum population; :func:`convert` sets it once every
    fixture has been re-located.
    """
    clip_name, index = parse_segment_id(fixture.clip_id)
    if fixture.contrast == MARGINAL_CONTRAST:
        return Relocation(fixture, EXCLUDED_MARGINAL, None, ())
    clip = staged.get(clip_name)
    if clip is None:
        return Relocation(fixture, DROPPED_NOT_STAGED, None, ())
    if clip.surah_ayah != fixture.surah_ayah:
        raise ValueError(f"{fixture.clip_id}: fixture and staged clip name different ayat")
    if clip_name not in segmented_clips:
        return Relocation(fixture, DROPPED_UNSEGMENTED, None, ())
    segment = segments.get((clip_name, index))
    if segment is None:
        return Relocation(fixture, DROPPED_SEGMENT_MISSING, None, ())
    if not segment.kept:
        return Relocation(fixture, DROPPED_SEGMENT_DROPPED, segment, ())

    located = contrast_sites(segment.decode, segment.reference, fixture.contrast)
    if not located:
        return Relocation(fixture, DROPPED_CONTRAST_ABSENT, segment, ())
    start, end = segment_sample_bounds(clip.num_samples, segment.start_s, segment.end_s)
    sites = []
    for site in located:
        if fixture.contrast == SHADDA_CONTRAST:
            mark, prescribed = SHADDAH, HELD if site.change == DROPPED else NOT_HELD
        else:
            mark, prescribed = fixture.contrast, segment.reference[site.reference_index]
        candidate = TruthSite(
            site_id=f"{P35_FIXTURE}:{fixture.clip_id}:{fixture.contrast}#{site.reference_index}",
            source=P35_FIXTURE,
            assumes_competent_reciter=False,
            audio_filename=clip_name,
            shard=clip.shard,
            start_sample=start,
            end_sample=end,
            audio_sha256=clip.audio_sha256,
            surah_ayah=fixture.surah_ayah,
            reference=segment.reference,
            reference_index=site.reference_index,
            mark=mark,
            prescribed=prescribed,
            heard=prescribed if fixture.verdict == ACCEPT else PENDING,
            stratum=f"{P35_FIXTURE}:{fixture.contrast}",
            stratum_population=1,  # set by convert() once the stratum is complete
        )
        try:
            parse_site(asdict(candidate), candidate.site_id)
        except ValueError:
            continue  # a carrier the schema cannot hold, e.g. a folded ghunna noon
        sites.append(candidate)
    if not sites:
        return Relocation(fixture, DROPPED_NOT_EXPRESSIBLE, segment, ())
    return Relocation(fixture, KEPT, segment, tuple(sites))


def convert(
    fixtures: list[EvalFixtureEntry],
    segments: dict[tuple[str, int], Segment],
    staged: dict[str, StagedClip],
    segmented_clips: set[str],
) -> tuple[list[TruthSite], list[Relocation]]:
    """Every fixture's :class:`Relocation`, and the sites with stratum populations set.

    Fixtures are processed in ``(clip_id, contrast)`` order and sites emitted in carrier
    order within one, so the output is a pure function of the inputs.
    """
    ordered = sorted(fixtures, key=lambda f: (f.clip_id, f.contrast))
    relocations = [relocate(f, segments, staged, segmented_clips) for f in ordered]
    drafts = [site for r in relocations for site in r.sites]
    population = Counter(site.stratum for site in drafts)
    sites = [
        TruthSite(**{**asdict(site), "stratum_population": population[site.stratum]})
        for site in drafts
    ]
    return sites, relocations


def relocation_rows(relocations: list[Relocation]) -> list[dict]:
    """The committed record of every fixture's re-location: the fixture, the segment it
    was looked for in (span, realized reference, base decode) and the sites it gave."""
    rows = []
    for r in relocations:
        seg = r.segment
        rows.append(
            {
                "fixture_clip_id": r.fixture.clip_id,
                "contrast": r.fixture.contrast,
                "verdict": r.fixture.verdict,
                "note": r.fixture.note,
                "outcome": r.outcome,
                "segment_start_s": None if seg is None else seg.start_s,
                "segment_end_s": None if seg is None else seg.end_s,
                "segment_drops": None if seg is None or seg.kept else list(seg.drops),
                "reference": None if seg is None else seg.reference,
                "decode": None if seg is None else seg.decode,
                "site_ids": [s.site_id for s in r.sites],
            }
        )
    return rows


def summary(sites: list[TruthSite], relocations: list[Relocation], run: dict) -> dict:
    """Kept vs dropped per bucket and verdict, sites per stratum, and the decode used."""
    outcomes: dict[str, Counter] = {}
    for r in relocations:
        outcomes.setdefault(f"{r.fixture.contrast} / {r.fixture.verdict}", Counter())[
            r.outcome
        ] += 1
    heard: dict[str, Counter] = {}
    for site in sites:
        heard.setdefault(site.stratum, Counter())[
            "pending" if site.heard == PENDING else "heard_as_prescribed"
        ] += 1
    return {
        "fixtures": len(relocations),
        "fixtures_kept": sum(r.outcome == KEPT for r in relocations),
        "sites": len(sites),
        "outcomes_by_bucket": {k: dict(sorted(v.items())) for k, v in sorted(outcomes.items())},
        "sites_by_stratum": {k: dict(sorted(v.items())) for k, v in sorted(heard.items())},
        "decode_fingerprint": run["decode_fingerprint"],
        "pausal_taa_marbuta": run["pausal_taa_marbuta"],
    }


def outcome_table(relocations: list[Relocation]) -> str:
    """Kept / dropped per bucket in Markdown, as ``truth_sites/README.md`` carries it."""
    outcomes = sorted({r.outcome for r in relocations})
    lines = [
        "| bucket | verdict | " + " | ".join(f"`{o}`" for o in outcomes) + " | total | sites |",
        "|---" * (len(outcomes) + 4) + "|",
    ]
    keys = sorted({(r.fixture.contrast, r.fixture.verdict) for r in relocations})
    for contrast, verdict in keys:
        group = [r for r in relocations if (r.fixture.contrast, r.fixture.verdict) == (contrast, verdict)]
        counts = Counter(r.outcome for r in group)
        cells = " | ".join(str(counts.get(o, 0)) for o in outcomes)
        sites = sum(len(r.sites) for r in group)
        lines.append(f"| `{contrast}` | {verdict} | {cells} | {len(group)} | {sites} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seg-dir", type=Path, required=True,
                        help="output of `tadabur.resegment --use p35_fixture`")
    parser.add_argument("--registry", type=Path, default=None,
                        help="staged-clip registry (default: the committed one)")
    parser.add_argument("--out", type=Path, default=SITES_PATH)
    parser.add_argument("--relocation", type=Path, default=RELOCATION_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    staged = load_staged_clips() if args.registry is None else load_staged_clips(args.registry)
    segments = read_segments(args.seg_dir)
    segmented = {name for name, _ in segments}
    run = json.loads((args.seg_dir / "run.json").read_text(encoding="utf-8"))
    sites, relocations = convert(
        [*load_should_accept(), *load_should_reject()], segments, staged, segmented
    )
    write_truth_sites(sites, args.out)
    with open(args.relocation, "w", encoding="utf-8") as f:
        for row in relocation_rows(relocations):
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.summary.write_text(
        json.dumps(summary(sites, relocations, run), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(outcome_table(relocations))
    print(f"\nWrote {len(sites)} sites to {args.out}")


if __name__ == "__main__":
    main()

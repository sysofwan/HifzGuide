"""Word snips (#264): which samples of each worklist clip the listening UI plays.

The owner asked not to hear the whole ayah for every item (2026-11-02). Pair, shaddah and
tashkeel sites already play an excerpt (:func:`tadabur.listening_session.excerpt_span`), but a
catch-loss item plays its whole tlog clip, because tlog has no word timings until phase A
aligns it (:mod:`tadabur.tlog_mining.word_times`). This module turns those committed word
times into a **sidecar** beside the worklist: per worklist id, the ``[start_sample,
end_sample)`` of the decoded 16 kHz clip the UI plays, or the whole clip.

The rule is exactly :func:`excerpt_span`'s — the word and one either side, padded by
:data:`~tadabur.listening_session.EXCERPT_PAD_S`, at least
:data:`~tadabur.listening_session.MIN_EXCERPT_S` — so the two can never drift: the builder
calls it rather than reimplementing it. A clip plays whole when the record is missing, is
``aligned: false``, sits below the likelihood gate, names another ayah than the row, has no
such word, or the rule itself falls back.

A word-time record is joined to a worklist row by the clip's ``audio_sha256`` **and** the
row's ``surah_ayah``: phase A records the **filename** ayah while phase D records a mined
row's **assigned** one, so the same clip can appear under either, and only the record for the
ayah the worklist names may place its words.

Usage (from ``tools/``)::

  python -m tadabur.word_snips build --worklist tadabur/<run>/catch_loss/worklist.jsonl \
      --word-times tadabur/tlog_mining/word_times/look.jsonl.gz \
      --out tadabur/<run>/catch_loss/snips.jsonl \
      --report tadabur/<run>/catch_loss/snips.report.json

It prints the fallback rate and its causes; the file is committed beside the worklist before
any listening, so a re-answer never rewrites what the owner heard.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .listening_session import excerpt_span
from .mining_pool import PoolSegment

#: An aligned word's per-frame log-likelihood below this is not trusted to place the word:
#: the clip plays whole, exactly as ``aligned: false`` does. Provisional (-3.5, phase A's
#: aligned p5) until the owner confirms it for the S5 draw (#311); it depends only on the
#: reference and the teacher, so it leaks nothing to the page.
MIN_LL_PER_FRAME = -3.5

#: An excerpt cut from the clip's word alignment.
ALIGNMENT = "alignment"
#: The clip itself: no alignment, or one not trusted to place this word.
WHOLE_CLIP = "whole_clip"
_SOURCES = frozenset({ALIGNMENT, WHOLE_CLIP})

_SNIP_FIELDS = frozenset({"id", "audio_sha256", "start_sample", "end_sample", "snip_source"})
_ROW_FIELDS = ("id", "audio_sha256", "surah_ayah", "word_index", "start_sample", "end_sample")


@dataclass(frozen=True)
class Snip:
    """One worklist item's audio: ``[start_sample, end_sample)`` of its decoded 16 kHz clip,
    or the whole clip (``snip_source == WHOLE_CLIP``, both bounds ``None``)."""

    site_id: str
    audio_sha256: str
    start_sample: int | None
    end_sample: int | None
    snip_source: str

    def as_json(self) -> dict:
        return {"id": self.site_id, "audio_sha256": self.audio_sha256,
                "start_sample": self.start_sample, "end_sample": self.end_sample,
                "snip_source": self.snip_source}


def _index(value: object) -> bool:
    """Whether ``value`` is a sample index: a non-negative ``int``, not a bool."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _snip_from_row(data: object, where: str) -> Snip:
    if not isinstance(data, dict) or set(data) != _SNIP_FIELDS:
        raise ValueError(f"{where}: a snip row is {sorted(_SNIP_FIELDS)}")
    site_id, digest, source = data["id"], data["audio_sha256"], data["snip_source"]
    if not isinstance(site_id, str) or not site_id:
        raise ValueError(f"{where}: id must be a non-empty string")
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"{where}: audio_sha256 must be a non-empty string")
    start, end = data["start_sample"], data["end_sample"]
    if source == ALIGNMENT:
        if not (_index(start) and _index(end) and start < end):
            raise ValueError(f"{where}: an alignment snip is 0 <= start < end")
    elif source == WHOLE_CLIP:
        if start is not None or end is not None:
            raise ValueError(f"{where}: a whole-clip snip has no bounds")
    else:
        raise ValueError(f"{where}: snip_source is {ALIGNMENT!r} or {WHOLE_CLIP!r}, "
                         f"not {source!r}")
    return Snip(site_id, digest, start, end, source)


def read_snips(path: Path) -> dict[str, Snip]:
    """Every snip keyed by worklist id; a malformed row or a second snip for one id is
    refused."""
    snips: dict[str, Snip] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        snip = _snip_from_row(json.loads(raw), f"{path}:{lineno}")
        if snip.site_id in snips:
            raise ValueError(f"{path}:{lineno}: a second snip for {snip.site_id}")
        snips[snip.site_id] = snip
    return snips


def write_snips(snips: Iterable[Snip], path: Path) -> None:
    """Rewrite the whole file atomically, one row per id sorted by id, so an interrupted
    build leaves the previous sidecar intact and a re-build never reorders the diff."""
    by_id: dict[str, Snip] = {}
    for snip in snips:
        if snip.site_id in by_id:
            raise ValueError(f"two snips for {snip.site_id}")
        by_id[snip.site_id] = snip
    write_text_atomically(path, "".join(
        json.dumps(by_id[key].as_json(), ensure_ascii=False, sort_keys=True) + "\n"
        for key in sorted(by_id)
    ))


@dataclass(frozen=True)
class _Placement:
    """A row's snip and, when it plays whole, why: one cause per whole-clip row."""

    snip: Snip
    reason: str | None


def build_snips(
    rows: Sequence[Mapping[str, object]], records: Iterable[Mapping[str, object]], *,
    min_ll_per_frame: float = MIN_LL_PER_FRAME,
) -> list[Snip]:
    """One snip per worklist row, sorted by id, from the rows and their word-time records.

    ``rows`` are worklist rows as raw dicts (``id, audio_sha256, surah_ayah, word_index,
    start_sample, end_sample``, where ``end_sample`` is the decoded 16 kHz clip's length);
    ``records`` are :mod:`tadabur.tlog_mining.word_times` records, whose ``word_times`` are
    each word's ``(start, end)`` pair in seconds.
    """
    return sorted((placement.snip for placement in _placements(rows, records, min_ll_per_frame)),
                  key=lambda snip: snip.site_id)


def _placements(
    rows: Sequence[Mapping[str, object]], records: Iterable[Mapping[str, object]],
    min_ll_per_frame: float,
) -> list[_Placement]:
    by_clip = _records_by_clip(records)
    placements = [_placement(row, by_clip, min_ll_per_frame) for row in rows]
    ids = [placement.snip.site_id for placement in placements]
    duplicate = next((site_id for site_id, count in Counter(ids).items() if count > 1), None)
    if duplicate is not None:
        raise ValueError(f"two worklist rows share the id {duplicate}")
    return placements


def _records_by_clip(
    records: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
    """The records keyed by ``(audio_sha256, surah_ayah)``: only the record for the ayah a
    row names may place its words. Two records for one key must agree on the alignment they
    would place; otherwise which one wins would depend on the files' order."""
    by_clip: dict[tuple[str, str], Mapping[str, object]] = {}
    for record in records:
        key = (record["audio_sha256"], record["surah_ayah"])
        previous = by_clip.get(key)
        if previous is not None and _timing(previous) != _timing(record):
            raise ValueError(f"two word-time records for {key[0]} under {key[1]} disagree")
        by_clip[key] = record
    return by_clip


def _timing(record: Mapping[str, object]) -> tuple:
    return (record["aligned"], record["ll_per_frame"],
            tuple(tuple(pair) for pair in record["word_times"] or ()))


def _row_fields(row: Mapping[str, object]) -> tuple[str, str, str, int, int]:
    """The row's ``(id, audio_sha256, surah_ayah, word_index, clip_samples)``, refusing a row
    that cannot be joined or that has no such word or clip."""
    missing = [name for name in _ROW_FIELDS if name not in row]
    if missing:
        raise ValueError(f"a worklist row has no {missing[0]}")
    site_id, digest, ayah = row["id"], row["audio_sha256"], row["surah_ayah"]
    word_index, clip_samples = row["word_index"], row["end_sample"]
    if not isinstance(site_id, str) or not site_id:
        raise ValueError("a worklist row's id must be a non-empty string")
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"{site_id}: audio_sha256 must be a non-empty string")
    if not isinstance(ayah, str) or not ayah:
        raise ValueError(f"{site_id}: surah_ayah must be a non-empty string")
    if not _index(word_index):
        raise ValueError(f"{site_id}: word_index must be a non-negative integer")
    if not _index(clip_samples) or clip_samples == 0:
        raise ValueError(f"{site_id}: end_sample must be the clip's positive length")
    return site_id, digest, ayah, word_index, clip_samples


def _placement(
    row: Mapping[str, object], records: Mapping[tuple[str, str], Mapping[str, object]],
    min_ll_per_frame: float,
) -> _Placement:
    site_id, digest, ayah, word_index, clip_samples = _row_fields(row)
    record = records.get((digest, ayah))
    reason = _whole_reason(record, word_index, min_ll_per_frame)
    if reason is None:
        start, end = _excerpt(record, word_index, clip_samples)
        if (start, end) != (0, clip_samples):
            return _Placement(Snip(site_id, digest, start, end, ALIGNMENT), None)
        reason = "the excerpt covers the whole clip"
    return _Placement(Snip(site_id, digest, None, None, WHOLE_CLIP), reason)


def _whole_reason(
    record: Mapping[str, object] | None, word_index: int, min_ll_per_frame: float
) -> str | None:
    """Why ``record`` cannot place ``word_index``, or ``None`` when it can."""
    if record is None:
        return "no word-time record for the clip's ayah"
    if not record["aligned"]:
        return "the clip is not aligned"
    likelihood = record["ll_per_frame"]
    if likelihood is None or likelihood < min_ll_per_frame:
        return "ll_per_frame is below the gate"
    if word_index >= len(record["word_times"] or ()):
        return "the word is past the alignment's words"
    return None


def _excerpt(
    record: Mapping[str, object], word_index: int, clip_samples: int
) -> tuple[int, int]:
    """The rule's span, over a segment that is the whole clip and the record's words: each
    word's onset plus the last word's end are the boundaries :func:`excerpt_span` reads."""
    times = [tuple(pair) for pair in record["word_times"]]
    boundaries = (*[start for start, _ in times], times[-1][1])
    segment = PoolSegment(0, 0, len(times), 0, clip_samples, "", tuple(range(len(times) + 1)),
                          True)
    return excerpt_span(segment, boundaries, 0, word_index, clip_samples)


def _report(placements: Sequence[_Placement]) -> dict:
    reasons = Counter(p.reason for p in placements if p.reason is not None)
    rows = len(placements)
    whole = sum(reasons.values())
    return {"rows": rows, "alignment": rows - whole, "whole_clip": whole,
            "fallback_rate": whole / rows if rows else 0.0, "reasons": dict(reasons)}


def _read_jsonl(path: Path) -> list[dict]:
    """A JSONL file, gzipped when its name says so."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.worklist)
    records = [record for path in args.word_times for record in _read_jsonl(path)]
    placements = _placements(rows, records, args.min_ll_per_frame)
    write_snips([placement.snip for placement in placements], args.out)
    report = _report(placements)
    print(f"{report['rows']} rows: {report['alignment']} play an alignment excerpt, "
          f"{report['whole_clip']} play whole "
          f"({report['fallback_rate']:.1%} fallback)", flush=True)
    for reason, count in sorted(report["reasons"].items()):
        print(f"  {count:5d}  {reason}", flush=True)
    if args.report is not None:
        write_text_atomically(args.report,
                              json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True)
                              + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tadabur.word_snips", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="write the snip sidecar beside a worklist")
    build.add_argument("--worklist", type=Path, required=True,
                       help="the worklist whose ids the sidecar is keyed by")
    build.add_argument("--word-times", type=Path, nargs="+", required=True,
                       help="committed word-time records (phase A's look/confirm.1/2, and "
                            "phase D's mined.confirm.1/2): plain or .gz JSONL")
    build.add_argument("--out", type=Path, required=True, help="the sidecar to write")
    build.add_argument("--report", type=Path, default=None,
                       help="also write the fallback counts as JSON here")
    build.add_argument("--min-ll-per-frame", type=float, default=MIN_LL_PER_FRAME,
                       help="the reliability gate: an aligned record below it plays whole "
                            f"(default: {MIN_LL_PER_FRAME})")
    build.set_defaults(handler=_build)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

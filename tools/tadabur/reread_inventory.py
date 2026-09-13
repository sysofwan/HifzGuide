"""Inventory the re-read material already on disk, before paying to stream more.

Muraja ADR-0016 needs natural re-reads. Its decision 2 mines them from the
**reject** pile, which does not exist yet — but a prior filtering run's *passing*
artifacts survive, and before a re-stream is commissioned the honest question is
whether they already carry enough. This module answers it, torch-free, from three
files a finished run leaves behind:

* the passing manifest (:mod:`tadabur.manifest`) — whose ``predicted_phonemes``
  let the ``.balanced`` gate be **re-run offline**, recovering the
  ``max_insertion_run`` the original run computed and discarded;
* the per-clip segmentation status (:mod:`tadabur.clip_status`) — whose ``re_reads``
  counts the seams ``tadabur.waqf_detect`` cut a clip at;
* the staged clip directory — because a manifest row whose WAV is gone is not a
  usable clip.

The two re-read signals are deliberately *both* reported, because they disagree and
the disagreement is the finding. ``re_reads`` is a word-space count from the
segmenter and will flag a clip whose word-edge snapping wobbled; ``max_insertion_run``
is the phoneme-space length of the longest repeated span and is what the mining
predicate (:func:`~tadabur.rejects.is_clean_re_read`) actually keys on. A passing clip
cannot reach the predicate's floor — clearing it is what "passing" means — so the
counts here bound what the existing artifacts can yield, and
:attr:`Inventory.clean_re_reads` is the number that decides whether a stream is needed.

Usage:
  python -m tadabur.reread_inventory --passing passing_subset_full.jsonl \
      --clip-status seg/manifest.jsonl.clip_status.jsonl --clips-dir clips_v2/ \
      [--json inventory.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .clip_status import ClipStatus, read_clip_status
from .manifest import ManifestRecord, read_records
from .reference_phonemes import load_reference_phonemes
from .rejects import is_clean_re_read
from .scorer import BALANCED_SCORER, Scorer


@dataclass(frozen=True)
class Inventory:
    """What a finished filtering run's surviving artifacts hold, for re-read mining.

    ``clips`` / ``clips_with_audio`` size the whole passing subset. The two histograms
    are keyed by the signal in their name and count clips, not seams. ``re_read_clips``
    is how many the segmenter flagged; ``clean_re_reads`` how many meet the mining
    predicate — the deciding number, and structurally 0 over a passing subset (see the
    module docstring). ``skip_reasons`` carries the segmenter's own verdicts, of which
    ``repeated_recitation`` is the one that means "gave up because of a repeat" and is
    therefore re-read-adjacent evidence in its own right.
    """

    clips: int
    clips_with_audio: int
    re_read_clips: int
    re_read_clips_with_audio: int
    clean_re_reads: int
    reciters: int
    re_read_reciters: int
    re_read_ayat: int
    re_read_audio_seconds: float
    re_reads_histogram: dict[str, int] = field(default_factory=dict)
    skip_reasons: dict[str, int] = field(default_factory=dict)
    insertion_run_histogram: dict[str, int] = field(default_factory=dict)
    re_read_insertion_run_histogram: dict[str, int] = field(default_factory=dict)
    repeated_recitation_clips: int = 0
    repeated_recitation_with_audio: int = 0


def take_inventory(
    records: list[ManifestRecord],
    statuses: list[ClipStatus],
    clips_dir: Path | None,
    references: dict[str, str],
    scorer: Scorer = BALANCED_SCORER,
) -> Inventory:
    """Cross ``records`` with ``statuses`` and the staged audio into an :class:`Inventory`.

    Each record is re-gated against its cached reference — the same ``.balanced`` gate
    the original run used, on the decode it stored — which is what recovers
    ``max_insertion_run``. A record with no matching status row is still counted in the
    totals and the insertion-run histogram: it was filtered, it just was never
    segmented, and silently dropping it would understate the subset.
    """
    by_clip = {status.audio_filename: status for status in statuses}
    re_reads = Counter()
    skip_reasons = Counter()
    insertion_runs = Counter()
    re_read_runs = Counter()
    clips_with_audio = 0
    re_read_clips = re_read_with_audio = clean_re_reads = 0
    repeated_recitation = repeated_recitation_with_audio = 0
    re_read_reciters: set[int] = set()
    re_read_ayat: set[str] = set()
    re_read_seconds = 0.0
    reciters: set[int] = set()

    for record in records:
        reciters.add(record.reciter_id)
        has_audio = clips_dir is None or (clips_dir / record.audio_filename).exists()
        clips_with_audio += int(has_audio)

        result = scorer.gate(record.predicted_phonemes, references[record.surah_ayah])
        insertion_runs[result.max_insertion_run] += 1
        if is_clean_re_read(
            result.match_ratio, result.max_insertion_run, result.added_shadda
        ):
            clean_re_reads += 1

        status = by_clip.get(record.audio_filename)
        if status is None:
            continue
        re_reads[status.re_reads] += 1
        skip_reasons[status.skip_reason or "none"] += 1
        if status.skip_reason == "repeated_recitation":
            repeated_recitation += 1
            repeated_recitation_with_audio += int(has_audio)
        if status.re_reads >= 1:
            re_read_clips += 1
            re_read_with_audio += int(has_audio)
            re_read_runs[result.max_insertion_run] += 1
            re_read_reciters.add(record.reciter_id)
            re_read_ayat.add(record.surah_ayah)
            re_read_seconds += status.duration_s

    return Inventory(
        clips=len(records),
        clips_with_audio=clips_with_audio,
        re_read_clips=re_read_clips,
        re_read_clips_with_audio=re_read_with_audio,
        clean_re_reads=clean_re_reads,
        reciters=len(reciters),
        re_read_reciters=len(re_read_reciters),
        re_read_ayat=len(re_read_ayat),
        re_read_audio_seconds=round(re_read_seconds, 1),
        re_reads_histogram=_histogram(re_reads),
        skip_reasons=_histogram(skip_reasons),
        insertion_run_histogram=_histogram(insertion_runs),
        re_read_insertion_run_histogram=_histogram(re_read_runs),
        repeated_recitation_clips=repeated_recitation,
        repeated_recitation_with_audio=repeated_recitation_with_audio,
    )


def _histogram(counter: Counter) -> dict[str, int]:
    """Render a counter as a key-ordered ``{str: count}`` dict, for stable JSON."""
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passing", type=Path, required=True, help="Passing-subset JSONL manifest."
    )
    parser.add_argument(
        "--clip-status",
        type=Path,
        required=True,
        help="Per-clip segmentation status JSONL (segment_score's .clip_status.jsonl).",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="Directory of staged whole-clip WAVs; omitted, audio presence is assumed.",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write the inventory as JSON here."
    )
    args = parser.parse_args()

    inventory = take_inventory(
        read_records(args.passing),
        read_clip_status(args.clip_status),
        args.clips_dir,
        load_reference_phonemes(),
    )
    payload = json.dumps(asdict(inventory), ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

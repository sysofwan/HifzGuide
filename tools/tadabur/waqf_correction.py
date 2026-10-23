"""The frozen pause corrections (#130): where the guard heard a ta marbuta at a pause.

A ta marbuta is read ``ت`` in wasl and ``ه`` at a waqf, but the segment reference assumes
wasl throughout except at its last word. When the reciter pauses mid-segment on a word that
ends in a ta marbuta, the reference keeps the wasl ``ت`` and a correct ``ه`` is scored as a
pair mistake (all 70 #126 pair sites of the pool were ``ت``→``ه``). The guard's own pause
decision (:mod:`tadabur.waqf_guard`: the owner's note first, then the local pause detector at
its calibration operating point) says where the reciter stopped; this module freezes those
boundaries as a committed manifest so the derived references are reproducible, not a
per-build guess.

A correction names the clip, the segment it applies to (its ``start_sample``), the ayah word
the pause precedes (``cut_word``) and its provenance. :func:`derive_corrections` reads only
frozen inputs — the committed boundary scores, the guard's calibration operating point and
the clips' ayahs — never the rebuilt guard rows, whose dependent boundaries disappear once
the references carry the pause. :func:`paused_segment` applies a clip's cuts to a segment;
:func:`load_cuts` is the map the site derivation and the guard read by default.

Paths repeat :mod:`tadabur.waqf_guard`'s directory because :mod:`tadabur.listening_session`
loads this module; importing the guard here would close that cycle. Only paused ta-marbuta
boundaries are corrected: every other pause stays #134's exclusion.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .local_pause import fires
from .mining_pool import CLIPS_PATH as POOL_CLIPS_PATH
from .mining_pool import PoolSegment, load_manifest
from .waqf_segments import _uthmani_words, hafs_paused_reference

SHIP_A_DIR = Path(__file__).resolve().parent / "ship_a"
CORRECTION_PATH = SHIP_A_DIR / "waqf_correction.jsonl"
EXTENSION_CLIPS_PATH = SHIP_A_DIR / "mining" / "clips.jsonl"
BOUNDARIES_PATH = SHIP_A_DIR / "waqf_guard.boundaries.jsonl"
GUARD_SUMMARY_PATH = SHIP_A_DIR / "waqf_guard.summary.json"
#: The owner's answers, repeated from :mod:`tadabur.listening_session` (the import cycle the
#: module docstring names).
VERDICTS_PATH = Path(__file__).resolve().parent / "listening_session" / "verdicts.jsonl"

#: A segment's cuts: absolute ayah word indices the reference carries a waqf at.
Cuts = tuple[int, ...]
#: A clip's segment start sample to that segment's cuts.
CutMap = Mapping[tuple[str, int], Cuts]


@dataclass(frozen=True)
class Correction:
    """One paused ta-marbuta boundary: the clip, the segment it is inside, the ayah word
    the pause precedes, and who decided it."""

    audio_filename: str
    surah_ayah: str
    start_sample: int
    cut_word: int
    provenance: str


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(raw) for raw in f if raw.strip()]


def load_corrections(path: Path = CORRECTION_PATH) -> list[Correction]:
    """The committed corrections, in file order."""
    return [Correction(**row) for row in _read_jsonl(path)]


def cut_map(rows: Iterable[Correction]) -> dict[tuple[str, int], Cuts]:
    """Corrections keyed by clip and segment start sample, each segment's cuts sorted."""
    grouped: dict[tuple[str, int], set[int]] = {}
    for row in rows:
        grouped.setdefault((row.audio_filename, row.start_sample), set()).add(row.cut_word)
    return {key: tuple(sorted(cuts)) for key, cuts in grouped.items()}


def load_cuts(path: Path = CORRECTION_PATH) -> dict[tuple[str, int], Cuts]:
    """The committed corrections as :data:`CutMap`."""
    return cut_map(load_corrections(path))


@cache
def _words(surah_ayah: str) -> tuple[str, ...]:
    return tuple(_uthmani_words(surah_ayah))


def _clips() -> dict[str, tuple[str, tuple[PoolSegment, ...]]]:
    """Every pool and mining-extension clip's ayah and segments, by clip, from the clips'
    committed manifests (the same two the guard's windows are read on)."""
    from .ship_a_mining import FILES as EXTENSION_FILES

    clips = [*load_manifest(POOL_CLIPS_PATH),
             *load_manifest(EXTENSION_CLIPS_PATH, None, EXTENSION_FILES.use,
                            EXTENSION_FILES.strata)]
    return {clip.audio_filename: (clip.surah_ayah, clip.segments) for clip in clips}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependent(by_word: Mapping[tuple[str, int], list], name: str, word: int) -> list[str]:
    """The items whose prescribed mark a waqf at the boundary changes
    (:func:`tadabur.waqf_guard.classify`): those whose own word is the one after the
    boundary or the one before it, read on the realization without cuts so a boundary a
    correction removed still has its site."""
    from .waqf_guard import classify, wasl_item

    return [item.item_id
            for start in (word - 1, word)
            for item in by_word.get((name, start), ())
            if any(d.word == word for d in classify(wasl_item(item)).dependences)]


def _owner_decision(item_ids: Iterable[str], verdicts: Mapping[str, Verdict]
                    ) -> tuple[bool, str | None]:
    """The owner's reading of a boundary's pause, when any of its dependent items' notes
    states one (:func:`tadabur.waqf_guard.owner_says`), and the item that said it. Notes
    that disagree on one boundary are an error, not a tie-break."""
    from .waqf_guard import owner_says

    said = [(item_id, owner_says(verdicts[item_id].note))
            for item_id in item_ids if item_id in verdicts]
    pauses = [item_id for item_id, says in said if says is True]
    wasls = [item_id for item_id, says in said if says is False]
    if pauses and wasls:
        raise ValueError(f"{pauses[0]} and {wasls[0]} disagree on their shared boundary")
    if pauses:
        return True, pauses[0]
    return False, (wasls[0] if wasls else None)


def derive_corrections(verdicts_path: Path = VERDICTS_PATH,
                       boundaries_path: Path = BOUNDARIES_PATH) -> list[Correction]:
    """The guard's paused ta-marbuta boundaries: the owner's note first, then the local
    pause check, at the current verdicts (:func:`tadabur.waqf_guard.resolve`).

    A boundary is corrected when the word before it carries a ta marbuta, it is interior to
    a kept segment (a pause at a segment end is already the reference's waqf), and the
    guard hears a pause there: a dependent item's owner note overrides the detector both
    ways, else the local pause check fires at the calibration operating point (an unplaced
    boundary is no pause without an owner note). Dependences are read on the frozen items'
    own references, so a boundary stays visible after a correction removed its site. One row
    is written per containing segment, with its start sample; the provenance names the
    deciding note or the detector and pins both inputs' checksums.
    """
    from .waqf_guard import frozen_items, read_verdicts

    operating_point = json.loads(GUARD_SUMMARY_PATH.read_text(encoding="utf-8"))[
        "calibration"]["operating_point"]
    if operating_point is None:
        raise ValueError("the guard has no calibration operating point")
    verdicts = read_verdicts(verdicts_path)
    by_word: dict[tuple[str, int], list] = defaultdict(list)
    for item in frozen_items():
        by_word[(item.audio_filename, item.word_start)].append(item)
    inputs = (f"inputs: waqf_guard.boundaries.jsonl={_sha256(boundaries_path)}, "
              f"listening_session/verdicts.jsonl={_sha256(verdicts_path)}")
    detector = (f"local_pause fires at the guard's operating point "
                f"(threshold={operating_point['threshold']}, "
                f"duration={operating_point['duration']}); {inputs}")
    clips = _clips()
    found: set[Correction] = set()
    for boundary in _read_jsonl(boundaries_path):
        if boundary["kind"] != "site":
            continue
        name, word = boundary["audio_filename"], boundary["word"]
        if name not in clips:
            continue  # a frozen worklist or P3.5 clip: its reference is not rebuilt here
        ayah, segments = clips[name]
        if "ة" not in _words(ayah)[word - 1]:
            continue
        paused, decided_by = _owner_decision(_dependent(by_word, name, word), verdicts)
        if decided_by is None:
            if boundary["window"] is None:
                continue  # unplaced: the guard reads no pause without an owner note
            if not fires(boundary["scores"], operating_point["threshold"],
                         operating_point["duration"]):
                continue
            provenance = detector
        elif not paused:
            continue  # the owner says wasl: no correction
        else:
            provenance = f"the owner's note says a pause: {decided_by}; {inputs}"
        for segment in segments:
            if segment.kept and segment.word_start < word < segment.word_end:
                found.add(Correction(name, ayah, segment.start_sample, word, provenance))
    return sorted(found, key=lambda row: (row.audio_filename, row.start_sample, row.cut_word))


def paused_segment(segment: PoolSegment, words: Sequence[str], cuts: Iterable[int]) -> PoolSegment:
    """The segment realized with a waqf at each of ``cuts`` (absolute ayah word indices).

    A cut outside the segment leaves it unchanged, so a correction list may be applied to
    every segment of a clip."""
    local = tuple(sorted({cut - segment.word_start for cut in cuts
                          if segment.word_start < cut < segment.word_end}))
    if not local:
        return segment
    reference, offsets = hafs_paused_reference(
        list(words[segment.word_start:segment.word_end]), local)
    return replace(segment, reference=reference, raw_word_offsets=tuple(offsets))


def write_corrections(rows: list[Correction], path: Path = CORRECTION_PATH) -> None:
    """Write the manifest, one row per line, sorted."""
    write_text_atomically(path, "".join(
        json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in sorted(rows, key=lambda r: (r.audio_filename, r.start_sample, r.cut_word))))


def main() -> None:
    write_corrections(derive_corrections())


if __name__ == "__main__":
    main()

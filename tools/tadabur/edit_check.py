"""The synthetic-edit check in the owner's listening session (#107, for #88 and #61).

ADR-0011 §3 admits a synthetic edit type as training signal only after a blind listen
confirms the edits sound real. #88 drew that listen: ``synthetic_edits/blind_check.jsonl``,
truth-site skeletons of edits and their unedited decoys, mixed, under opaque ids and file
names. This module puts those items in the listening session, in the same shuffled queue
as the session's sites and behind the same blinding (:mod:`tadabur.tashkeel_audit_ui`).
Each item asks two things: does it sound natural
(:data:`tadabur.listening_session.NATURALNESS`), and what was said at the carrier (its
mark's question: held / not held, or which letter of the pair). Both go into one verdict.

**What the page can learn.** The UI reads only the truth-site file, never the edit manifest
(``synthetic_edits/edits.jsonl``), so no operation, label, role, donor or render path is
even loaded. A row carries the *source's* reference, carrier, mark and prescription, which
an edit and its decoy share; its id and file name are opaque hashes. An item **plays whole**
(Play and Whole alike): an excerpt around the carrier would have to be placed from the
manifest, and its length would differ between an edit and its decoy by the edit's length
change, while an edit and its decoy are the same length whole.

**Words.** The page hides every site's answer wherever its word appears (the same ayah,
word index and realized word), so an item needs its Uthmani word offsets in its whole-ayah
reference. They come from the phonetizer (``quran-transcript``), which the UI does not
need, so they are computed once per draw and committed:
``listening_session/edit_check_words.json``.

**Audio.** The items were rendered on the GPU box (:data:`EDIT_AUDIO_REMOTE`).
``python -m tadabur.listening_session fetch`` copies them next to the session's clips, and
the UI verifies every item's checksum before it serves anything (:func:`verify_audio`).

The per-operation summary #90 needs is :mod:`tadabur.edit_check_summary`; the UI never
imports it.

Usage (from ``tools/``; needs ``quran-transcript``, run once per blind-check draw)::

  python -m tadabur.edit_check words
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .listening_session import SESSION_DIR, mode_of
from .truth_sites import PENDING, SYNTHETIC_EDIT, TruthSite, load_truth_sites
from .truth_sites import verify_audio as verify_checksums

#: #88's blind-check worklist (``tadabur.synthetic_edits.WORKLIST_PATH``; that module is
#: not imported here, because it reads the edit manifest).
BLIND_CHECK_PATH = Path(__file__).parent / "synthetic_edits" / "blind_check.jsonl"
WORDS_PATH = SESSION_DIR / "edit_check_words.json"
#: Where the items were rendered on the GPU box (``generate --out-dir stage/edits``), last
#: by #117 after it deduplicated the recordings.
EDIT_AUDIO_REMOTE = "/root/scratch/issue-117/stage/edits/audio"


@dataclass(frozen=True)
class EditCheckSite:
    """One blind-check item as the session asks it: its truth-site skeleton and the word
    offsets of its whole-ayah reference. The excerpt is the whole item."""

    site: TruthSite
    word_offsets: tuple[int, ...]

    @property
    def mode(self) -> str:
        return mode_of(self.site.mark)

    @property
    def word_start(self) -> int:
        return 0  # the reference is the whole ayah

    @property
    def excerpt_start_sample(self) -> int:
        return self.site.start_sample

    @property
    def excerpt_end_sample(self) -> int:
        return self.site.end_sample


def read_words(path: Path = WORDS_PATH) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Each ayah's whole-ayah reference and its word offsets, checked to partition it."""
    record = json.loads(path.read_text(encoding="utf-8"))
    words = {}
    for surah_ayah, entry in record["ayahs"].items():
        reference, offsets = entry["reference"], entry["word_offsets"]
        if not (offsets and all(type(o) is int for o in offsets) and offsets[0] == 0
                and offsets[-1] == len(reference) and offsets == sorted(offsets)):
            raise ValueError(f"{path}: {surah_ayah}'s word offsets do not partition it")
        words[surah_ayah] = (reference, tuple(offsets))
    return words


def load_edit_check(
    path: Path = BLIND_CHECK_PATH, words_path: Path = WORDS_PATH
) -> list[EditCheckSite]:
    """The blind check in file order, each item with its words. Every row must be a
    staged, unanswered synthetic-edit item that is a whole file, and its reference must be
    the one its words were computed on."""
    words = read_words(words_path)
    rows = []
    for site in load_truth_sites(path):
        where = f"{path}: {site.site_id}"
        if site.source != SYNTHETIC_EDIT or site.heard != PENDING:
            raise ValueError(f"{where} is not a pending synthetic-edit item")
        if site.start_sample != 0 or site.audio_sha256 is None:
            raise ValueError(f"{where} is not a whole, staged item")
        reference, offsets = words.get(site.surah_ayah, (None, ()))
        if reference != site.reference:
            raise ValueError(f"{where}: {words_path} has no words for its reference; "
                             f"run python -m tadabur.edit_check words")
        rows.append(EditCheckSite(site, offsets))
    return rows


def verify_audio(rows: list[EditCheckSite], audio_dir: Path) -> None:
    """Fail unless ``audio_dir`` holds every item with its recorded checksum (which fixes
    its length too: the row's ``end_sample`` is the rendered file's)."""
    missing = sorted(r.site.audio_filename for r in rows
                     if not (audio_dir / r.site.audio_filename).is_file())
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} edit-check item(s) are not in {audio_dir}, e.g. {missing[0]}; "
            f"copy them with python -m tadabur.listening_session fetch")
    verify_checksums([row.site for row in rows], audio_dir)


def ayah_words(surah_ayahs) -> dict[str, dict]:
    """Each ayah's whole-ayah realized reference (wasl inside, waqf at its end, as
    ``mining_pool.whole_ayah_references`` makes it) with its Uthmani word offsets."""
    from .waqf_segments import _uthmani_words, hafs_segment_reference

    segment_reference = hafs_segment_reference()
    words = {}
    for surah_ayah in sorted(set(surah_ayahs)):
        reference, offsets = segment_reference(_uthmani_words(surah_ayah))
        words[surah_ayah] = {"reference": reference, "word_offsets": list(offsets)}
    return words


def _words(args) -> None:
    import hafs_phonetizer

    sites = load_truth_sites(args.blind_check)
    record = {"phonetizer_revision": hafs_phonetizer.REVISION,
              "ayahs": ayah_words(site.surah_ayah for site in sites)}
    write_text_atomically(args.out, json.dumps(record, ensure_ascii=False, indent=1,
                                               sort_keys=True) + "\n")
    load_edit_check(args.blind_check, args.out)
    print(f"Wrote the words of {len(record['ayahs'])} ayahs to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    words = commands.add_parser("words", help="write the blind check's word offsets")
    words.add_argument("--blind-check", type=Path, default=BLIND_CHECK_PATH)
    words.add_argument("--out", type=Path, default=WORDS_PATH)
    args = parser.parse_args()
    _words(args)


if __name__ == "__main__":
    main()

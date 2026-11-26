"""The S5 listening UI (#311 PR3): the draw's pilot sitting, blind, then the main one.

The page asks one drawn word at a time — was it recited correctly? — in the order the draw
committed, and writes the answer where the scorer reads it. It is the catch-loss word question
(``tadabur.tashkeel_audit_ui``) with three things changed for S5:

* **Order.** ``s5_draw/pilot.jsonl`` then, only behind ``--with-main``, ``s5_draw/main.jsonl``,
  each in its committed order: the repeats are separate ids already in those files, so the page
  asks them again with the same behaviour and never fills one from the other. The main sitting
  waits behind the flag because the pre-registration may resize it after the pilot is read.
* **Vocabulary.** :data:`tadabur.followalong_audit.s5_score.VERDICTS` and ``MISTAKE_KINDS``,
  with ``SUB_REASONS`` on ``unusable audio`` — the schema :func:`s5_score.read_verdicts` holds
  the file to. :func:`s5_score.check_verdict` is the one rule both readers apply.
* **Audio.** The Mac's WAVs (``<audio_sha256>.wav``), verified against the committed
  ``app_arm/audio_manifest.jsonl.gz`` before the server starts, sliced at the item's snip
  (:mod:`tadabur.word_snips`). A whole-clip snip always plays whole; the page's Whole control
  plays the clip for any item, and the verdict records that it did.

The page is told the opaque key, the ayah with the word highlighted, and the answer already
given. It is never told the stratum, cell, inclusion probability, an arm's grade, or which
sitting an item came from — :func:`tadabur.catch_loss_worklist.read_catch_loss_worklist` refuses a
worklist whose word is not the ayah's, and the snips and WAVs are checked against it.

Usage (from ``tools/``; ``scripts/issue-311-audit.sh`` stages it on cuda-dev)::

  python -m tadabur.followalong_audit.s5_ui --wav-dir ~/wav --verdicts ~/verdicts.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import wave
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from tadabur.audit_http import AuditHandler, serve
from tadabur.catch_loss_worklist import CatchLossRow, read_catch_loss_worklist
from tadabur.followalong_audit.app_arm import read_jsonl, read_jsonl_gz
from tadabur.followalong_audit.clips import SAMPLE_RATE, verify_wav
from tadabur.followalong_audit.s5_score import (
    MISTAKE_KINDS,
    SUB_REASONS,
    VERDICTS,
    check_verdict,
    read_verdicts,
)
from tadabur.tashkeel_audit_ui import blind_reference, page_key, verify_snips
from tadabur.word_snips import ALIGNMENT, WHOLE_CLIP, Snip, read_snips
from tadabur.waqf_segments import uthmani_words
from training.tashkeel_eval import write_text_atomically

AUDIT = Path(__file__).resolve().parent
DRAW_DIR = AUDIT / "s5_draw"
APP_ARM = AUDIT / "app_arm"
PAGE = AUDIT / "s5_ui_page.html"
#: The repository root, for the Quran text database the wasl-merge check reads.
REPO = Path(__file__).resolve().parents[3]
QURAN_DB = REPO / "data" / "quran.db"

#: The sittings, in the order the round reads them.
SITTINGS = ("pilot", "main")
#: What the page's answer carries back, and what a verdict row's answer half is.
ANSWER_FIELDS = ("verdict", "kinds", "sub_reason", "note")
#: What ``record`` and the POST route take.
ANSWER_REQUEST = ("key", *ANSWER_FIELDS, "previous")


class StaleAnswer(Exception):
    """The page answered a different stored answer than the one it was shown."""


# --- the files ------------------------------------------------------------------------------------

def read_audio_manifest(path: Path) -> dict[str, dict]:
    """The committed WAV manifest by ``audio_sha256``: each row's ``wav_sha256`` and ``samples``."""
    return {row["audio_sha256"]: row for row in read_jsonl_gz(Path(path))}


def load_rows(worklist: Path, snips: Path, orders: Mapping[str, Path], *,
              uthmani_words: Callable[[str], list[str]],
              references: Callable[[str], object] | None = None,
              sittings: Sequence[str] = ("pilot",)) -> list[CatchLossRow]:
    """The worklist's rows in served order: each served sitting's file, in file order.

    Refused: a row whose word is not its ayah's (:func:`read_catch_loss_worklist`), an order id
    the worklist lacks, an id served twice, a served row with no snip or a snip built for another
    clip or running past it (:func:`verify_snips`), and — with ``references`` — a row whose
    engine word is recited together with another canonical word, which would give the page two
    printed words for one phoneme word.
    """
    by_id = _worklist_rows(worklist, uthmani_words)
    return _served(by_id, snips, orders, sittings, references)


def _worklist_rows(worklist: Path,
                   uthmani_words: Callable[[str], list[str]]) -> dict[str, CatchLossRow]:
    return {row.site.site_id: row
            for row in read_catch_loss_worklist(Path(worklist), uthmani_words)}


def _served(by_id: Mapping[str, CatchLossRow], snips: Path, orders: Mapping[str, Path],
            sittings: Sequence[str],
            references: Callable[[str], object] | None) -> list[CatchLossRow]:
    served: list[CatchLossRow] = []
    seen: set[str] = set()
    for sitting in sittings:
        for row in read_jsonl(Path(orders[sitting])):
            ident = row["id"]
            if ident not in by_id:
                raise ValueError(f"{orders[sitting]} serves {ident}, which the worklist lacks")
            if ident in seen:
                raise ValueError(f"{orders[sitting]} serves {ident} twice")
            seen.add(ident)
            served.append(by_id[ident])
    verify_snips(served, read_snips(Path(snips)))
    if references is not None:
        _refuse_wasl_merged(served, references)
    return served


def _refuse_wasl_merged(rows: Sequence[CatchLossRow],
                        references: Callable[[str], object]) -> None:
    """Fail on a drawn word its ayah's reference reads with a neighbour (a wasl merge).

    The page highlights one printed word; an engine word that spans two canonical words would be
    asking about both, so the draw must never contain one. The check reads the oracle's own
    reference (:func:`tadabur.followalong_audit.sweep.ayah_references`).
    """
    for row in rows:
        word_of = tuple(references(row.site.surah_ayah).word_of)
        if not 0 <= row.word_index < len(word_of):
            raise ValueError(f"{row.site.site_id}: word {row.word_index} is outside "
                             f"{row.site.surah_ayah}'s {len(word_of)} reference words")
        engine = word_of[row.word_index]
        if word_of.count(engine) > 1:
            raise ValueError(f"{row.site.site_id}: word {row.word_index} of "
                             f"{row.site.surah_ayah} is recited with another canonical word")


# --- the session ----------------------------------------------------------------------------------

class SessionState:
    """The served rows, their audio and the verdicts file the page writes as it goes.

    Every answer is validated against :func:`s5_score.check_verdict` and written whole, sorted by
    id, so a restart resumes from the file and an answer the scorer could not read never lands.
    Rows the file holds for ids this sitting does not serve are kept: the pilot sitting's file
    already answers for main, and a relaunch must not drop them.
    """

    def __init__(self, rows: Sequence[CatchLossRow], snips: Mapping[str, Snip], *, wav_dir: Path,
                 manifest: Mapping[str, Mapping], verdicts: Path,
                 stored: Mapping[str, Mapping]) -> None:
        self.rows = list(rows)
        self._by_id = {row.site.site_id: row for row in self.rows}
        self._ids_by_key = {page_key(row.site.site_id): row.site.site_id for row in self.rows}
        self.snips = snips
        self.wav_dir = Path(wav_dir)
        self.manifest = manifest
        self.verdicts_path = Path(verdicts)
        self._verdicts = dict(stored)
        self._served_whole: set[str] = set()
        self._lock = threading.Lock()

    @classmethod
    def load(cls, *, worklist: Path, snips: Path, orders: Mapping[str, Path], wav_dir: Path,
             wav_manifest: Path, verdicts: Path,
             uthmani_words: Callable[[str], list[str]],
             references: Callable[[str], object] | None = None,
             sittings: Sequence[str] = ("pilot",)) -> SessionState:
        """The served rows, their snips, their verified WAVs and the verdicts already given."""
        by_id = _worklist_rows(worklist, uthmani_words)
        rows = _served(by_id, snips, orders, sittings, references)
        manifest = read_audio_manifest(wav_manifest)
        for digest in sorted({row.site.audio_sha256 for row in rows}):
            _verify_clip(Path(wav_dir) / f"{digest}.wav", manifest.get(digest))
        stored = read_verdicts(Path(verdicts), [{"id": ident} for ident in by_id])
        return cls(rows, read_snips(Path(snips)), wav_dir=wav_dir, manifest=manifest,
                   verdicts=verdicts, stored=stored)

    def payload(self) -> dict:
        """What the page is told: its sites in served order, progress, and the vocabulary.

        A site is the opaque key, the ayah split around the highlighted word, and the answer
        already given — nothing of the draw's strata, weights or arms.
        """
        return {"sites": [self._site(row) for row in self.rows],
                "progress": {"answered": sum(1 for row in self.rows
                                             if row.site.site_id in self._verdicts),
                             "total": len(self.rows)},
                "vocabulary": {"verdicts": list(VERDICTS), "kinds": list(MISTAKE_KINDS),
                               "sub_reasons": list(SUB_REASONS)}}

    def _site(self, row: CatchLossRow) -> dict:
        before, carrier, after = blind_reference(row, {})
        return {"key": page_key(row.site.site_id), "surah_ayah": row.site.surah_ayah,
                "before": before, "carrier": carrier, "after": after,
                "answer": _answer(self._verdicts.get(row.site.site_id))}

    def record(self, key: str, verdict: str, kinds: Sequence[str], sub_reason: str | None,
               note: str, previous: Mapping | None) -> dict:
        """Save one answer over the one the page was shown; returns the progress and the answer.

        ``KeyError`` for a key this sitting does not serve, :class:`StaleAnswer` when ``previous``
        is not the stored answer (another tab or a re-answer moved it), ``ValueError`` for an
        answer the scorer would refuse — and then nothing is written.
        """
        ident = self._ident(key)
        with self._lock:
            stored = self._verdicts.get(ident)
            if _answer(stored) != previous:
                raise StaleAnswer(f"{key}: the answer shown was {previous!r}, the file holds "
                                  f"{_answer(stored)!r}")
            row = {"id": ident, "verdict": verdict, "kinds": list(kinds),
                   "sub_reason": sub_reason, "whole_played": self._whole_played(ident, stored),
                   "note": note}
            check_verdict(row)
            proposed = {**self._verdicts, ident: row}
            self._write(proposed)
            self._verdicts = proposed
        return {"progress": self.payload()["progress"], "answer": _answer(row)}

    def audio(self, key: str, whole: bool) -> bytes:
        """The item's snip as a 16 kHz mono 16-bit WAV, or the whole clip.

        The whole clip plays for ``whole`` and always for a whole-clip snip; playing it is
        remembered for the item's verdict, rewriting one already given.
        """
        ident = self._ident(key)
        snip = self.snips[ident]
        row = self._by_id[ident]
        source = self.wav_dir / f"{row.site.audio_sha256}.wav"
        if snip.snip_source == WHOLE_CLIP:
            return _wav_slice(source)
        if whole:
            with self._lock:
                self._served_whole.add(ident)
                self._note_whole_played(ident)
            return _wav_slice(source)
        return _wav_slice(source, snip.start_sample, snip.end_sample)

    def _ident(self, key: str) -> str:
        try:
            return self._ids_by_key[key]
        except KeyError:
            raise KeyError(f"{key} is not a served key of this sitting") from None

    def _whole_played(self, ident: str, stored: Mapping | None) -> bool:
        return (self.snips[ident].snip_source == WHOLE_CLIP or ident in self._served_whole
                or bool(stored and stored["whole_played"]))

    def _note_whole_played(self, ident: str) -> None:
        """Rewrite an answer already given so it records the whole clip just played.

        Called under the lock; an unwritable file leaves the answer as it was, so the next save
        records it again.
        """
        stored = self._verdicts.get(ident)
        if stored is None or stored["whole_played"]:
            return
        proposed = {**self._verdicts, ident: {**stored, "whole_played": True}}
        try:
            self._write(proposed)
        except OSError as error:  # the clip still plays; a re-answer records it too
            print(f"verdict not updated with the whole clip: {error}", flush=True)
            return
        self._verdicts = proposed

    def _write(self, verdicts: Mapping[str, Mapping]) -> None:
        write_text_atomically(self.verdicts_path, "".join(
            json.dumps(verdicts[ident], ensure_ascii=False, sort_keys=True) + "\n"
            for ident in sorted(verdicts)))


def _answer(row: Mapping | None) -> dict | None:
    """A stored verdict as the page names the answer it saw, or ``None``."""
    return None if row is None else {name: row[name] for name in ANSWER_FIELDS}


def _verify_clip(path: Path, expected: Mapping | None) -> None:
    """Refuse a clip the manifest does not vouch for: missing, other bytes, other length."""
    if not path.is_file():
        raise ValueError(f"{path} is not there: the manifest vouches for this clip")
    verify_wav(path, expected)


def _wav_slice(path: Path, start: int | None = None, stop: int | None = None) -> bytes:
    """The clip's samples ``[start, stop)`` (the whole clip without bounds), as a WAV.

    The clips are already mono 16 kHz 16-bit (``clips.verify_wav``), so the slice is taken in
    their own samples: re-encoding through floats would move values at the scale's edges.
    """
    with wave.open(str(path), "rb") as source:
        frames = source.readframes(source.getnframes())
    first = 0 if start is None else start
    last = len(frames) // 2 if stop is None else stop
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(SAMPLE_RATE)
        sink.writeframes(frames[2 * first:2 * last])
    return buffer.getvalue()


# --- the routes -----------------------------------------------------------------------------------

class Handler(AuditHandler):
    """Routes the page, the blind site list, an item's audio, and an answer."""

    state: SessionState

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self.send_bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/sites":
            self.send_json(self.state.payload())
        elif path == "/api/audio":
            query = parse_qs(parsed.query)
            whole = query.get("whole", ["0"])[0] == "1"
            try:
                self.send_bytes(self.state.audio(query.get("key", [""])[0], whole), "audio/wav")
            except KeyError as error:
                self.send_json({"error": error.args[0]}, status=404)
        else:
            self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if unquote(urlparse(self.path).path) != "/api/verdict":
            self.send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) if length else b"")
            answer = {name: body[name] for name in ANSWER_REQUEST}
        except (ValueError, KeyError, TypeError):
            self.send_json({"error": f"an answer is a JSON object of {list(ANSWER_REQUEST)}"},
                           status=400)
            return
        try:
            self.send_json(self.state.record(**answer))
        except StaleAnswer as error:
            self.send_json({"error": str(error)}, status=409)
        except KeyError as error:
            self.send_json({"error": error.args[0]}, status=404)
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)
        except OSError as error:
            self.send_json({"error": f"not saved: {error}"}, status=500)


# --- the command line -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worklist", type=Path, default=DRAW_DIR / "worklist.jsonl",
                        help="The draw's blind worklist (default: the committed one).")
    parser.add_argument("--snips", type=Path, default=DRAW_DIR / "snips.jsonl",
                        help="The draw's snip sidecar (default: the committed one).")
    parser.add_argument("--pilot", type=Path, default=DRAW_DIR / "pilot.jsonl",
                        help="The pilot sitting's served order (default: the committed one).")
    parser.add_argument("--main", type=Path, default=DRAW_DIR / "main.jsonl",
                        help="The main sitting's served order (default: the committed one).")
    parser.add_argument("--wav-dir", type=Path, required=True,
                        help="The clips' WAVs, named <audio_sha256>.wav.")
    parser.add_argument("--wav-manifest", type=Path,
                        default=APP_ARM / "audio_manifest.jsonl.gz",
                        help="The committed digest manifest every WAV is verified against "
                             "(default: the committed one).")
    parser.add_argument("--verdicts", type=Path, required=True,
                        help="Where the answers are written, one row per id (s5_score reads it).")
    parser.add_argument("--with-main", action="store_true",
                        help="Serve the main sitting after the pilot (default: the pilot alone).")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: loopback; 0.0.0.0 for the LAN).")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from tadabur.followalong_audit.sweep import ayah_references
    from training.muraja_policy import MurajaText

    sittings = SITTINGS if args.with_main else ("pilot",)
    state = SessionState.load(
        worklist=args.worklist, snips=args.snips,
        orders={"pilot": args.pilot, "main": args.main}, wav_dir=args.wav_dir,
        wav_manifest=args.wav_manifest, verdicts=args.verdicts, uthmani_words=uthmani_words,
        references=ayah_references(MurajaText(QURAN_DB)), sittings=sittings)
    progress = state.payload()["progress"]
    httpd = serve(Handler, state, args.port, args.host)
    print(f"S5 listening UI: {progress['answered']} of {progress['total']} answered, "
          f"sittings {', '.join(sittings)}; http://{args.host}:{args.port}/  (Ctrl-C to stop)",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

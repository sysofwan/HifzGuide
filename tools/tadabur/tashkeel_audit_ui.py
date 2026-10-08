"""Blind web UI for the owner's listening session (#87 for #61; first built for #60).

Serves the session worklist (:mod:`tadabur.listening_session`) so the owner can listen to
each site and record **what the reciter said**, one shuffled queue across three questions:

* **tashkeel** — which mark is on the highlighted letter: fatha, damma, kasra, sukun or
  unclear. Prescribed-haraka and prescribed-sukun sites ask the same question and look the
  same on the page;
* **shaddah** — was the highlighted consonant held (doubled), not held, or unclear;
* **consonant** — which of the pair's two letters was said, or unclear. The P3.5 nominal
  rejects come up here and in shaddah, indistinguishable from the mined sites.

**The page is blind** (ADR-0007). It never receives a model's output, the mark or letter
the mushaf prescribes, the site's stratum, source, id or sampling weight, or any running
tally: :meth:`SessionState.payload` builds the only thing it is sent. The displayed text
hides the answer of **every** session site wherever its word appears, not only the site on
screen, because the page receives the whole queue at once: two sites in one segment (or in
the same word of the same ayah recited by another reciter) would otherwise show each
other's answer. Per question, the answer and the phonetic cues that would give it away are
hidden: for tashkeel the haraka with any madd or qalqala after the carrier; for shaddah the
doubled consonant (shown once) and the haraka after it; for consonant the letter (shown as
``◌``, a geminate once) and a qalqala mark after it, which only some letters take. Sites
are addressed by an opaque key, so not even the id says where a site came from. Progress
is only "n of N answered".

Each answer is written straight into the tracked verdicts file
(``listening_session/verdicts.jsonl``) keyed by site id, so the UI resumes from it and
committing it is the only step after the session. Audio is the site's **excerpt** (the
carrier's word and one either side) sliced from the staged clip, or the whole segment on
request; every clip is checked against the staged-clip registry's checksum and length
before the server starts.

Usage (from ``tools/``)::

  python -m tadabur.tashkeel_audit_ui --audio-dir <staged clips> --host 0.0.0.0 [--port 8000]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .audio import TARGET_SAMPLE_RATE
from .audit_http import AuditHandler, serve
from .listening_session import (
    MADD,
    QALQALA,
    SALT,
    SHADDAH_MODE,
    TASHKEEL_MODE,
    VERDICTS_PATH,
    WORKLIST_PATH,
    SessionSite,
    Verdict,
    hearable,
    load_worklist,
    read_verdicts,
    run_start,
    write_verdicts,
)
from .staged_audio import StagedClip, load_staged_clips, verify_staged
from .truth_sites import HARAKA_CHARS, HELD, NOT_HELD, SUKUN, UNCLEAR

_PAGE_PATH = Path(__file__).parent / "tashkeel_audit_page.html"
#: The page's logic, a module with no DOM in it, so it is tested on its own.
_SCRIPT_PATH = Path(__file__).parent / "tashkeel_audit_session.mjs"

#: What the page shows in place of the letter under test in consonant mode.
HIDDEN_LETTER = "◌"
_HARAKAT = frozenset(HARAKA_CHARS.values())
#: Characters after a tashkeel carrier that would give its mark away.
_MARK_TELLS = _HARAKAT | MADD | {QALQALA}

#: The answers offered per mode, in a fixed order that never depends on the site.
_TASHKEEL_CHOICES = (*HARAKA_CHARS, SUKUN, UNCLEAR)  # fatha, damma, kasra
_SHADDAH_CHOICES = (HELD, NOT_HELD, UNCLEAR)


def choices(row: SessionSite) -> tuple[str, ...]:
    """The answers the page offers for ``row``: the same for every site of its mode, so
    the order cannot hint at the prescribed one (a pair's letters in label order)."""
    if row.mode == TASHKEEL_MODE:
        return _TASHKEEL_CHOICES
    if row.mode == SHADDAH_MODE:
        return _SHADDAH_CHOICES
    return (*row.site.mark.split("↔"), UNCLEAR)


@dataclass(frozen=True)
class Target:
    """Where one session site's answer sits, in terms every row reciting the same word
    shares: the ayah, the Uthmani word index and the word's realized text, the carrier's
    offset in that word, and the question asked there."""

    word: tuple[str, int, str]
    offset: int
    mode: str


def _words(row: SessionSite) -> list[tuple[tuple[str, int, str], int]]:
    """Each word of the row's segment as ``(Target.word, its start in the reference)``."""
    reference, offsets = row.site.reference, row.word_offsets
    return [
        ((row.site.surah_ayah, row.word_start + k, reference[start:end]), start)
        for k, (start, end) in enumerate(zip(offsets, offsets[1:]))
    ]


def target_of(row: SessionSite) -> Target:
    index = row.site.reference_index
    word, start = [w for w in _words(row) if w[1] <= index][-1]
    return Target(word, index - start, row.mode)


def _hide(reference: str, shown: list[str], index: int, mode: str) -> None:
    """Blank (or replace) in ``shown`` what would give away ``mode``'s answer at ``index``."""
    def blank_while(start: int, chars) -> int:
        while start < len(reference) and reference[start] in chars:
            shown[start] = ""
            start += 1
        return start

    if mode == TASHKEEL_MODE:
        blank_while(index + 1, _MARK_TELLS)
        return
    after_run = blank_while(index + 1, {reference[index]})
    if mode == SHADDAH_MODE:
        blank_while(after_run, _HARAKAT)
    else:
        shown[index] = HIDDEN_LETTER
        blank_while(after_run, {QALQALA})


def masked(row: SessionSite, targets: Mapping[tuple[str, int, str], list[Target]]) -> list[str]:
    """The row's reference as the page shows it, character by character, with the answer of
    every target in any of its words hidden."""
    reference = row.site.reference
    shown = list(reference)
    for word, start in _words(row):
        for target in targets.get(word, ()):
            _hide(reference, shown, start + target.offset, target.mode)
    return shown


def blind_reference(
    row: SessionSite, targets: Mapping[tuple[str, int, str], list[Target]]
) -> tuple[str, str, str]:
    """:func:`masked` split around the row's carrier. The highlight runs from the first
    letter of the carrier's geminate (the carrier itself if it is single) to the carrier,
    then over what is read with it: the blanked answer and any harakat left after it. It
    stops at the next letter, even one equal to the carrier."""
    reference, shown = row.site.reference, masked(row, targets)
    anchor = run_start(reference, row.site.reference_index)
    end = row.site.reference_index + 1
    while end < len(reference) and (not shown[end] or reference[end] in _HARAKAT):
        end += 1
    return "".join(shown[:anchor]), "".join(shown[anchor:end]), "".join(shown[end:])


def page_key(site_id: str) -> str:
    """The opaque key the page addresses a site by: stable, and says nothing about it."""
    return hashlib.sha256(f"{SALT}:page:{site_id}".encode("utf-8")).hexdigest()[:16]


def encode_wav(samples: np.ndarray) -> bytes:
    """A mono 16 kHz 16-bit PCM RIFF file for ``samples``, built in memory. Values are
    clipped before scaling so a peak above unity rails instead of wrapping in sign."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def verify_audio(
    rows: list[SessionSite], registry: dict[str, StagedClip], audio_dir: Path
) -> None:
    """Fail unless every clip the worklist plays is in the registry under the checksum its
    sites record, every excerpt lies inside its clip, and ``audio_dir`` holds exactly that
    file (checksum and length)."""
    by_clip: dict[str, list[SessionSite]] = {}
    for row in rows:
        by_clip.setdefault(row.site.audio_filename, []).append(row)
    for name, clip_rows in sorted(by_clip.items()):
        clip = registry.get(name)
        if clip is None:
            raise ValueError(f"{name} is not in the staged-clip registry")
        if {row.site.audio_sha256 for row in clip_rows} != {clip.audio_sha256}:
            raise ValueError(f"{name}: the worklist and the registry disagree on its checksum")
        if any(row.excerpt_end_sample > clip.num_samples for row in clip_rows):
            raise ValueError(f"{name}: an excerpt runs past the end of the clip")
        verify_staged(clip, audio_dir)


class StaleAnswer(Exception):
    """The answer replaces one the client did not see: the site was answered since."""


@dataclass
class SessionState:
    """The worklist, the verdicts and the audio. ``verdicts`` is replaced only after the
    file holding it has been rewritten, so the page is never told an answer is saved that
    is not on disk. A verdict for a site outside the worklist (a re-mine dropped it) is kept
    on save and never shown."""

    rows: list[SessionSite]
    verdicts_path: Path
    audio_dir: Path
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    _by_key: dict[str, SessionSite] = field(default_factory=dict)
    _targets: dict[tuple[str, int, str], list[Target]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._by_key = {page_key(row.site.site_id): row for row in self.rows}
        if len(self._by_key) != len(self.rows):
            raise ValueError("two sites share a page key")
        for row in self.rows:
            target = target_of(row)
            self._targets.setdefault(target.word, []).append(target)

    @classmethod
    def load(
        cls, worklist: Path, verdicts: Path, audio_dir: Path, registry: dict[str, StagedClip]
    ) -> "SessionState":
        rows = load_worklist(worklist)
        verify_audio(rows, registry, audio_dir)
        return cls(rows=rows, verdicts_path=verdicts, audio_dir=audio_dir,
                   verdicts=read_verdicts(verdicts))

    def row(self, key: str) -> SessionSite:
        return self._by_key[key]

    def view(self, row: SessionSite) -> dict:
        """Everything the page may know about one site, and nothing else."""
        before, carrier, after = blind_reference(row, self._targets)
        verdict = self.verdicts.get(row.site.site_id)
        return {
            "key": page_key(row.site.site_id),
            "mode": row.mode,
            "surah_ayah": row.site.surah_ayah,
            "before": before,
            "carrier": carrier,
            "after": after,
            "choices": list(choices(row)),
            "heard": verdict.heard if verdict else None,
            "note": verdict.note if verdict else "",
        }

    def payload(self) -> dict:
        """The whole response ``/api/sites`` sends."""
        return {"sites": [self.view(row) for row in self.rows], "progress": self.progress()}

    def record(self, key: str, heard: object, note: str, previous: object) -> Verdict:
        """Persist one answer, replacing ``previous`` (the answer the client last saw for
        the site, ``None`` for none). Raises ``KeyError`` for an unknown key, ``ValueError``
        for an answer the question does not offer, :class:`StaleAnswer` when the stored
        answer is no longer ``previous``, and ``OSError`` when the file cannot be written;
        nothing changes in any of those cases."""
        row = self.row(key)
        if heard not in hearable(row.site.mark):
            raise ValueError(f"{heard!r} is not an answer to this question")
        verdict = Verdict(site_id=row.site.site_id, heard=heard, note=note)
        with self._lock:
            stored = self.verdicts.get(verdict.site_id)
            if (stored.heard if stored else None) != previous:
                raise StaleAnswer("this site was answered elsewhere since; reload the page")
            proposed = {**self.verdicts, verdict.site_id: verdict}
            write_verdicts(proposed, self.verdicts_path)
            self.verdicts = proposed
        return verdict

    def progress(self) -> dict:
        answered = sum(row.site.site_id in self.verdicts for row in self.rows)
        return {"answered": answered, "total": len(self.rows)}

    def audio(self, key: str, whole: bool) -> bytes:
        """The site's excerpt, or its whole segment, as a WAV."""
        import soundfile as sf

        row = self.row(key)
        start, end = (
            (row.site.start_sample, row.site.end_sample)
            if whole else (row.excerpt_start_sample, row.excerpt_end_sample)
        )
        samples, _ = sf.read(self.audio_dir / row.site.audio_filename, start=start,
                             stop=end, dtype="float32")
        return encode_wav(samples)


class SessionHandler(AuditHandler):
    """Routes: the page, the blind site list, site audio, and answer submission."""

    state: SessionState

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            self.send_bytes(_PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif parsed.path == "/api/sites":
            self.send_json(self.state.payload())
        elif parsed.path == "/session.mjs":
            self.send_bytes(_SCRIPT_PATH.read_bytes(), "text/javascript; charset=utf-8")
        elif parsed.path == "/api/audio":
            key = (query.get("key") or [""])[0]
            whole = (query.get("whole") or ["0"])[0] == "1"
            try:
                self.send_bytes(self.state.audio(key, whole), "audio/wav")
            except KeyError:
                self.send_json({"error": "unknown site"}, status=404)
        else:
            self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        if urlparse(self.path).path != "/api/verdict":
            self.send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (TypeError, ValueError):
            self.send_json({"error": "body must be JSON"}, status=400)
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("note", ""), str):
            self.send_json({"error": "body must be {key, heard, note, previous}"}, status=400)
            return
        try:
            self.state.record(str(payload.get("key", "")), payload.get("heard"),
                              payload.get("note", ""), payload.get("previous"))
        except StaleAnswer as error:
            self.send_json({"error": str(error)}, status=409)
            return
        except KeyError:
            self.send_json({"error": "unknown site"}, status=404)
            return
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)
            return
        except OSError as error:
            self.send_json({"error": f"not saved: {error}"}, status=500)
            return
        self.send_json({"progress": self.state.progress()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="directory holding the staged 16 kHz clips the worklist plays")
    parser.add_argument("--worklist", type=Path, default=WORKLIST_PATH)
    parser.add_argument("--verdicts", type=Path, default=VERDICTS_PATH,
                        help="the tracked verdicts JSONL; resumed if present")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; 0.0.0.0 to answer from a phone on the LAN")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = SessionState.load(args.worklist, args.verdicts, args.audio_dir,
                              load_staged_clips())
    progress = state.progress()
    print(f"{progress['answered']} of {progress['total']} answered. "
          f"Serving on http://{args.host}:{args.port}/", flush=True)
    serve(SessionHandler, state, args.port, args.host).serve_forever()


if __name__ == "__main__":
    main()

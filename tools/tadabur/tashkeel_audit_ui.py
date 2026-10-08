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
tally: :meth:`SessionState.view` builds the only thing it is sent. The displayed reference
hides the answer at the carrier: the haraka (and the madd or qalqala that would give it
away) is deleted for tashkeel, the doubled consonant is shown once for shaddah, and the
letter itself is replaced by ``◌`` for consonant. Sites are addressed by an opaque key, so
not even the id says where a site came from. Progress is only "n of N answered".

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
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .audio import TARGET_SAMPLE_RATE
from .audit_http import AuditHandler, serve
from .listening_session import (
    CONSONANT_MODE,
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
    write_verdicts,
)
from .staged_audio import StagedClip, load_staged_clips, verify_staged
from .truth_sites import HARAKA_CHARS, HELD, NOT_HELD, SUKUN, UNCLEAR

_PAGE_PATH = Path(__file__).parent / "tashkeel_audit_page.html"

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


def blind_reference(row: SessionSite) -> tuple[str, str, str]:
    """The segment's reference as the page shows it, split around the carrier, with the
    answer hidden (module docstring). The carrier part keeps the harakat after it, so the
    highlight covers the letter as it is read."""
    reference, index = row.site.reference, row.site.reference_index
    carrier = reference[index]
    rest = index + 1
    if row.mode == TASHKEEL_MODE:
        while rest < len(reference) and reference[rest] in _MARK_TELLS:
            rest += 1
    else:
        while rest < len(reference) and reference[rest] == carrier:
            rest += 1  # a geminate shows once: its doubling is the shaddah answer
        if row.mode == CONSONANT_MODE:
            carrier = HIDDEN_LETTER
    marks = rest
    while marks < len(reference) and reference[marks] in _HARAKAT:
        marks += 1
    return reference[:index], carrier + reference[rest:marks], reference[marks:]


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
    sites record, and ``audio_dir`` holds exactly that file (checksum and length)."""
    recorded: dict[str, set[str | None]] = {}
    for row in rows:
        recorded.setdefault(row.site.audio_filename, set()).add(row.site.audio_sha256)
    for name, checksums in sorted(recorded.items()):
        clip = registry.get(name)
        if clip is None:
            raise ValueError(f"{name} is not in the staged-clip registry")
        if checksums != {clip.audio_sha256}:
            raise ValueError(f"{name}: the worklist and the registry disagree on its checksum")
        verify_staged(clip, audio_dir)


@dataclass
class SessionState:
    """The worklist, the verdicts and the audio; ``verdicts`` is rewritten on every save,
    so the file and the in-memory view never disagree. A verdict for a site outside the
    worklist (a re-mine dropped it) is kept on save and never shown."""

    rows: list[SessionSite]
    verdicts_path: Path
    audio_dir: Path
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    _by_key: dict[str, SessionSite] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._by_key = {page_key(row.site.site_id): row for row in self.rows}
        if len(self._by_key) != len(self.rows):
            raise ValueError("two sites share a page key")

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
        before, carrier, after = blind_reference(row)
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

    def record(self, key: str, heard: object, note: str) -> Verdict:
        """Persist one answer, replacing any earlier one for the same site."""
        row = self.row(key)
        if heard not in hearable(row.site.mark):
            raise ValueError(f"{heard!r} is not an answer to this question")
        verdict = Verdict(site_id=row.site.site_id, heard=heard, note=note)
        with self._lock:
            self.verdicts[verdict.site_id] = verdict
            write_verdicts(self.verdicts, self.verdicts_path)
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
            self.send_json({
                "sites": [self.state.view(row) for row in self.state.rows],
                "progress": self.state.progress(),
            })
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
            self.send_json({"error": "body must be {key, heard, note}"}, status=400)
            return
        try:
            self.state.record(str(payload.get("key", "")), payload.get("heard"),
                              payload.get("note", ""))
        except KeyError:
            self.send_json({"error": "unknown site"}, status=404)
            return
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)
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

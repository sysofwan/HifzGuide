"""Local web UI for adjudicating mined tashkeel sites (#60).

Serves the worklist from :mod:`training.tashkeel_worklist` so a human can listen to each
position where the base and fine-tuned checkpoints disagreed about a short vowel, and record
what the **reciter actually said** (:mod:`tadabur.tashkeel_fixtures`). That verdict is the
ground truth neither the corpus nor the mushaf reference can supply: the reference states
the vowel the text prescribes, not the one the reciter produced, so a model declining to
mark it is ambiguous between the model being over-strict and the reciter being wrong.

**The audit is blind.** The API never sends ``direction`` or either checkpoint's outcome,
and the worklist is shuffled across buckets before it is written, so a listener cannot tell
whether saying "I heard the reference vowel" credits the fine-tune or convicts it. The
question on screen is only ever about the audio.

Audio is served as the **exact window span** both checkpoints were decoded on, sliced from
the staged clip and re-encoded as 16-bit PCM WAV in memory — not the whole clip. A listener
grading a vowel the model never heard would be adjudicating a different question. A padded
"with context" take is available separately, because a bare 5 s window can cut mid-word and
the surrounding syllable is often what makes a case ending audible.

There is no database and no framework: verdicts are persisted straight into the
adjudications JSONL, so the UI resumes from — and is interchangeable with — whatever that
file already holds.

Usage:
  python -m tadabur.tashkeel_audit_ui \\
    --worklist audit_run/seg_v21/tashkeel_worklist.jsonl \\
    --adjudications audit_run/seg_v21/tashkeel_adjudications.jsonl \\
    --audio-dir audit_run/clips_v2 [--port 8000]
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from training.tashkeel_eval import SHORT_VOWELS
from training.tashkeel_worklist import TashkeelSite, read_worklist

from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from .audit_http import AuditHandler, serve
from .audit_sampler import local_audio_path
from .tashkeel_acceptance import compare
from .tashkeel_fixtures import (
    VERDICTS,
    Adjudication,
    read_adjudications,
    write_adjudications,
)

_PAGE_PATH = Path(__file__).parent / "tashkeel_audit_page.html"

#: Seconds of surrounding clip audio the "with context" take adds on each side. A window
#: boundary can fall mid-word, and a case ending is far easier to hear with the syllable
#: that follows it.
DEFAULT_CONTEXT_PAD_S = 1.0

#: The largest padding a request may ask for, so a crafted query cannot make the server
#: encode an entire clip per request.
MAX_CONTEXT_PAD_S = 5.0


def _carrier_index(display_reference: str, vowel_index: int) -> int | None:
    """Index of the letter the removed harakah sat on, in the *display* reference.

    Deleting the target vowel shifts nothing before it, so the carrier keeps its index —
    it is the nearest preceding non-vowel character. Returns ``None`` when the vowel opened
    the window and there is no preceding letter to point at.
    """
    for index in range(min(vowel_index, len(display_reference)) - 1, -1, -1):
        if display_reference[index] not in SHORT_VOWELS:
            return index
    return None


class ClipCache:
    """Decode each staged clip at most once, shared across handler threads.

    The threading server handles requests concurrently and a listener replays the same
    window repeatedly, so decoding per request would re-read the same file on every click.
    Only the most recent few clips are retained: the worklist is shuffled, so there is no
    clip locality to exploit and an unbounded cache would grow to the whole corpus.
    """

    def __init__(self, audio_dir: Path, capacity: int = 8) -> None:
        self._audio_dir = audio_dir
        self._capacity = capacity
        self._lock = threading.Lock()
        self._cache: dict[str, np.ndarray] = {}

    def _path(self, clip_audio_filename: str) -> Path:
        """Resolve either staged layout: hash-prefixed sampler name, or the plain name."""
        hashed = self._audio_dir / local_audio_path(clip_audio_filename)
        if hashed.is_file():
            return hashed
        plain = self._audio_dir / clip_audio_filename
        if plain.is_file():
            return plain
        raise FileNotFoundError(
            f"clip audio for {clip_audio_filename!r} not found under {self._audio_dir} "
            "under either the hash-prefixed (tadabur.audit_sampler) or plain name."
        )

    def waveform(self, clip_audio_filename: str) -> np.ndarray:
        with self._lock:
            cached = self._cache.get(clip_audio_filename)
        if cached is None:
            cached = decode_to_mono_16k(self._path(clip_audio_filename).read_bytes())
            with self._lock:
                if len(self._cache) >= self._capacity:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[clip_audio_filename] = cached
        return cached


def encode_wav(samples: np.ndarray) -> bytes:
    """A mono 16 kHz 16-bit PCM RIFF file for ``samples``, built in memory.

    Written by hand rather than through a codec library because the browser needs a
    container it can play from a plain ``<audio src>``, and the staged clips are float32
    arrays after decoding — there is no original byte range to hand back once a slice is
    taken. Values are clipped before scaling so a clip that peaks above unity wraps to the
    rails instead of overflowing into the opposite sign.
    """
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


@dataclass
class AuditState:
    """Everything a request handler needs: the worklist, the verdicts, and the audio.

    ``adjudications`` is mutated in place and rewritten on every save, so the JSONL on disk
    and the in-memory view can never disagree — a crash mid-session loses nothing but the
    verdict being submitted.
    """

    sites: list[TashkeelSite]
    adjudications_path: Path
    clips: ClipCache
    population: dict
    adjudications: dict[str, Adjudication] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def load(
        cls, worklist: Path, adjudications: Path, audio_dir: Path, population: dict
    ) -> "AuditState":
        state = cls(
            sites=read_worklist(worklist),
            adjudications_path=adjudications,
            clips=ClipCache(audio_dir),
            population=population,
        )
        state.adjudications = read_adjudications(adjudications)
        return state

    def site(self, site_id: str) -> TashkeelSite | None:
        return next((s for s in self.sites if s.site_id == site_id), None)

    def record(self, site_id: str, verdict: str, note: str) -> Adjudication:
        """Persist one verdict, replacing any earlier verdict for the same site."""
        site = self.site(site_id)
        if site is None:
            raise KeyError(site_id)
        entry = Adjudication(
            site_id=site_id,
            verdict=verdict,
            clip_audio_filename=site.clip_audio_filename,
            reference_index=site.reference_index,
            note=note,
        )
        with self._lock:
            self.adjudications[site_id] = entry
            write_adjudications(self.adjudications_path, dict(self.adjudications))
        return entry

    def view(self, site: TashkeelSite) -> dict:
        """The blind view of one site — everything the listener may see, and nothing else.

        ``direction``, ``base_outcome`` and ``candidate_outcome`` are deliberately absent. A
        listener who knows the fine-tune recovered this position is no longer answering "what
        did the reciter say"; the whole value of the audit is that they cannot tell.

        **The reference vowel is withheld too**, and the reference is rendered with that one
        harakah deleted. Showing it would prime the listener toward the very answer that
        counts as "confirmed over-strictness" — the reading that inflates the fine-tune's
        gain. Surrounding harakat stay, because they are context rather than the answer, and
        ``carrier_index`` points at the letter to judge.
        """
        verdict = self.adjudications.get(site.site_id)
        index = site.reference_index
        display = site.reference[:index] + site.reference[index + 1:]
        return {
            "site_id": site.site_id,
            "surah_ayah": site.surah_ayah,
            "reciter_id": site.reciter_id,
            "clip_audio_filename": site.clip_audio_filename,
            "reference": display,
            "carrier": site.carrier,
            "carrier_index": _carrier_index(display, index),
            "verdict": verdict.verdict if verdict else None,
            "note": verdict.note if verdict else "",
        }

    def progress(self) -> dict:
        judged = sum(1 for s in self.sites if s.site_id in self.adjudications)
        return {"total": len(self.sites), "judged": judged}

    def results(self) -> dict:
        """The live over-strictness comparison, so progress is visible mid-audit.

        Shown behind a click rather than on the grading screen: a listener who can watch the
        gain move as they grade is being handed exactly the bias the blind view removes.
        """
        return compare(self.sites, self.adjudications, self.population)


class TashkeelAuditHandler(AuditHandler):
    """Routes: the page, the worklist API, window audio, and verdict submission."""

    state: AuditState

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            self.send_bytes(_PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif parsed.path == "/api/sites":
            self.send_json(
                {
                    "sites": [self.state.view(site) for site in self.state.sites],
                    "progress": self.state.progress(),
                }
            )
        elif parsed.path == "/api/results":
            self.send_json(self.state.results())
        elif parsed.path == "/api/audio":
            self._send_window_audio(query)
        else:
            self.send_json({"error": "not found"}, status=404)

    def _send_window_audio(self, query: dict[str, list[str]]) -> None:
        """Stream the site's window span, optionally padded, as a WAV."""
        site = self.state.site((query.get("site") or [""])[0])
        if site is None:
            self.send_json({"error": "unknown site"}, status=404)
            return
        try:
            pad_s = float((query.get("pad") or ["0"])[0])
        except ValueError:
            self.send_json({"error": "pad must be a number of seconds"}, status=400)
            return
        pad = int(min(max(pad_s, 0.0), MAX_CONTEXT_PAD_S) * TARGET_SAMPLE_RATE)
        try:
            waveform = self.state.clips.waveform(site.clip_audio_filename)
        except FileNotFoundError as error:
            self.send_json({"error": str(error)}, status=404)
            return
        start = max(0, site.start_sample - pad)
        end = min(len(waveform), site.start_sample + site.num_samples + pad)
        self.send_bytes(encode_wav(waveform[start:end]), "audio/wav")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        if urlparse(self.path).path != "/api/verdict":
            self.send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        verdict = payload.get("verdict")
        if verdict not in VERDICTS:
            self.send_json(
                {"error": f"verdict must be one of {sorted(VERDICTS)}"}, status=400
            )
            return
        try:
            self.state.record(payload.get("site_id", ""), verdict, payload.get("note", ""))
        except KeyError:
            self.send_json({"error": "unknown site"}, status=404)
            return
        self.send_json({"progress": self.state.progress()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True,
                        help="mined worklist JSONL (training.tashkeel_worklist).")
    parser.add_argument("--adjudications", type=Path, required=True,
                        help="verdict JSONL; created on the first save, resumed if present.")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="staged 16 kHz clip directory the windows are sliced from.")
    parser.add_argument("--summary", type=Path, default=None,
                        help="the worklist's '.summary.json' sidecar (default: beside it).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; pass 0.0.0.0 to grade from another device on "
                             "the LAN.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_path = args.summary or args.worklist.with_suffix(
        args.worklist.suffix + ".summary.json"
    )
    if not summary_path.is_file():
        raise SystemExit(
            f"{summary_path} not found — it holds the population counts the live results "
            "view needs. Pass --summary."
        )
    population = json.loads(summary_path.read_text(encoding="utf-8"))["population"]
    state = AuditState.load(args.worklist, args.adjudications, args.audio_dir, population)
    progress = state.progress()
    print(
        f"{progress['judged']}/{progress['total']} sites already judged. "
        f"Serving on http://{args.host}:{args.port}/"
    )
    serve(TashkeelAuditHandler, state, args.port, args.host).serve_forever()


if __name__ == "__main__":
    main()

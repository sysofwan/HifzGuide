"""Local web UI for recording the tashkeel counterfactual takes (#10).

`training.minimal_pairs` showed that the fine-tune's ~0.98 short-vowel recall is fully
explainable without any hearing: a text-only baseline that never touches audio scores 0.9734
on the ambiguous slice. Settling whether the phoneme head **hears** a harakah or reconstructs
it from the Quran's fixed text needs audio where the spoken vowel and the canonical vowel
disagree — audio that does not exist in any corpus, because every corpus clip is correct
recitation. `tools/tadabur/tashkeel_counterfactual_fixtures/` freezes the 47 words to record;
this module is the thing that captures them.

Each item is recited **twice by the same reciter**: a ``control`` take (the phrase as written)
and a ``counterfactual`` take (the same phrase with one vowel swapped). The control take is
what makes a negative result interpretable, so the UI treats an item as done only when both
takes exist.

Two properties matter more than the interface:

* **Every accepted take is readable by the pipeline.** Uploads are decoded through
  :func:`tadabur.audio.decode_to_mono_16k` — the exact loader the scorer will use — *before*
  they are written to disk. A take the pipeline cannot read is rejected with a 400 the reciter
  sees immediately, rather than discovered after all 94 clips are recorded. (The browser's
  ``MediaRecorder`` emits WebM/Opus, which ``soundfile`` cannot read and there is no ffmpeg
  here to transcode; the page therefore encodes 16 kHz mono RIFF itself, and this check is
  what keeps that contract honest.)
* **Progress is resumable.** The output directory *is* the state: a take is recorded iff its
  file is there, so the UI can be stopped and restarted mid-sheet and lands on the first item
  still missing a take. Nobody does 94 clips in one sitting.

Recorded audio is written to ``--out-dir`` (gitignored ``audit_run/`` by default), never into
the tracked fixtures directory.

Usage::

    python -m tadabur.counterfactual_record_ui [--items <jsonl>] [--out-dir <dir>] [--port 8000]

**Serve it on loopback and open the browser on the same machine.** Browsers expose
``navigator.mediaDevices`` only in a secure context — HTTPS, ``localhost`` or ``127.0.0.1`` —
so the LAN-exposed ``--host 0.0.0.0`` pattern the other audit UIs use leaves the microphone
dead. To record from another device, SSH-forward the port (``ssh -L 8000:127.0.0.1:8000 host``)
rather than binding a routable address; the page says so too when it detects the problem.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import soundfile as sf

from training.counterfactual_script import VOWEL_NAMES, CounterfactualItem

from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from .audit_http import AuditHandler, serve

_PAGE_PATH = Path(__file__).parent / "counterfactual_record_ui_page.html"

FIXTURES_DIR = Path(__file__).parent / "tashkeel_counterfactual_fixtures"
DEFAULT_ITEMS = FIXTURES_DIR / "counterfactual_items.jsonl"
DEFAULT_OUT_DIR = Path(__file__).parent / "audit_run" / "counterfactual_audio"

CONTROL = "control"
COUNTERFACTUAL = "counterfactual"
#: An item is done only when both takes exist — see the module docstring on the control take.
TAKES = (CONTROL, COUNTERFACTUAL)

# A take shorter than this is a mis-click, not a recitation: every phrase is a multi-word
# segment. Rejecting it costs one re-record; accepting it costs a silently unscorable item.
MIN_TAKE_SECONDS = 0.5
# Generous ceiling: a 5 MB-ish 16 kHz mono take is ~2.5 minutes, far past any single phrase,
# so anything longer is a recorder left running rather than a take.
MAX_TAKE_SECONDS = 150.0


def take_filename(item_id: str, take: str) -> str:
    """The filename the recording sheet already promises for ``take`` of ``item_id``.

    The sheet's ``take_1_file`` / ``take_2_file`` columns are literal
    ``<item_id>_control.wav`` / ``<item_id>_counterfactual.wav``, so the scorer can find the
    audio by construction and nothing has to record a path.
    """
    if take not in TAKES:
        raise ValueError(f"unknown take {take!r}; expected one of {TAKES}")
    return f"{item_id}_{take}.wav"


def load_items(path: Path) -> list[CounterfactualItem]:
    """Read the frozen item manifest (JSONL) into :class:`CounterfactualItem` rows, in order."""
    items: list[CounterfactualItem] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(CounterfactualItem(**json.loads(line)))
    return items


def validate_take(data: bytes) -> float:
    """Return the take's duration in seconds, or raise if the pipeline could not read it.

    This is the gate that keeps the UI honest: the bytes are decoded with the *same*
    :func:`~tadabur.audio.decode_to_mono_16k` the scorer will use, so a format ``soundfile``
    cannot open (notably ``MediaRecorder``'s WebM/Opus default) fails here and now instead of
    after 94 clips are in the can. Raises :class:`ValueError` with a reciter-readable reason.
    """
    if not data:
        raise ValueError("empty upload")
    try:
        waveform = decode_to_mono_16k(data)
    except Exception as exc:  # soundfile raises its own error types for every bad container
        raise ValueError(
            f"the pipeline cannot decode this audio ({exc}); it must be a WAV file"
        ) from exc
    seconds = len(waveform) / TARGET_SAMPLE_RATE
    if seconds < MIN_TAKE_SECONDS:
        raise ValueError(f"take is only {seconds:.2f}s long; recite the whole phrase")
    if seconds > MAX_TAKE_SECONDS:
        raise ValueError(f"take is {seconds:.0f}s long; the recorder was probably left running")
    return seconds


@dataclass
class TakeStore:
    """The recorded takes on disk. The directory is the state, so resume is free.

    There is no index file and no database: a take is recorded iff
    ``out_dir/<item_id>_<take>.wav`` exists. That makes the UI's progress identical to what
    the scorer will actually find, and survives the server being restarted mid-sheet.
    """

    out_dir: Path

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def path_of(self, item_id: str, take: str) -> Path:
        return self.out_dir / take_filename(item_id, take)

    def seconds_of(self, item_id: str, take: str) -> float | None:
        """Duration of a recorded take, or ``None`` if it is not there.

        Read from the WAV header rather than by decoding, so listing every item stays cheap.
        """
        path = self.path_of(item_id, take)
        if not path.is_file() or path.stat().st_size == 0:
            return None
        try:
            info = sf.info(str(path))
        except Exception:
            return None
        return info.frames / info.samplerate if info.samplerate else None

    def save(self, item_id: str, take: str, data: bytes) -> float:
        """Validate ``data`` and write it as ``item_id``'s ``take``, returning its duration.

        Writing is atomic-by-rename so an interrupted upload cannot leave a truncated WAV that
        the next run would count as a finished take. Re-recording simply overwrites.
        """
        seconds = validate_take(data)
        target = self.path_of(item_id, take)
        staging = target.with_suffix(".wav.part")
        staging.write_bytes(data)
        staging.replace(target)
        return seconds


def item_view(item: CounterfactualItem, store: TakeStore) -> dict[str, object]:
    """One item as the page needs it: what to say, what to say differently, what exists.

    ``words`` is the phrase pre-split on whitespace so the page can highlight ``word_index``
    without re-deriving the tokenization the item's index is defined against.
    """
    takes = {}
    for take in TAKES:
        seconds = store.seconds_of(item.item_id, take)
        takes[take] = {
            "recorded": seconds is not None,
            "seconds": seconds,
            "filename": take_filename(item.item_id, take),
        }
    return {
        "item_id": item.item_id,
        "surah_ayah": item.surah_ayah,
        "words": item.segment_text.split(),
        "word_index": item.word_index,
        "target_word": item.target_word,
        "spoken_word": item.spoken_word,
        "canonical_vowel": VOWEL_NAMES[item.canonical_vowel],
        "spoken_vowel": VOWEL_NAMES[item.spoken_vowel],
        "reference_phonemes": item.reference_phonemes,
        "takes": takes,
        "done": all(takes[take]["recorded"] for take in TAKES),
    }


@dataclass
class RecordingSession:
    """Server state: the frozen items plus the takes recorded for them so far."""

    items: list[CounterfactualItem]
    store: TakeStore

    def __post_init__(self) -> None:
        self._by_id = {item.item_id: item for item in self.items}

    def state(self) -> dict[str, object]:
        """The full UI payload: every item's view plus recording progress."""
        views = [item_view(item, self.store) for item in self.items]
        return {"items": views, "progress": self._progress(views)}

    def _progress(self, views: list[dict[str, object]]) -> dict[str, int]:
        return {
            "items": len(views),
            "items_done": sum(1 for view in views if view["done"]),
            "takes": len(views) * len(TAKES),
            "takes_recorded": sum(
                1 for view in views for take in TAKES if view["takes"][take]["recorded"]  # type: ignore[index]
            ),
        }

    def save_take(self, item_id: str, take: str, data: bytes) -> dict[str, object]:
        """Persist one take, returning the item's refreshed view and overall progress.

        An ``item_id`` outside the frozen manifest is rejected rather than written: the 47
        items are the experiment, and a stray request must not be able to add a 48th (nor,
        since the id becomes a filename, to name a path outside the output directory).
        """
        item = self._by_id.get(item_id)
        if item is None:
            raise KeyError(f"no counterfactual item {item_id!r}")
        self.store.save(item_id, take, data)
        views = [item_view(row, self.store) for row in self.items]
        return {
            "item": next(view for view in views if view["item_id"] == item_id),
            "progress": self._progress(views),
        }


class _Handler(AuditHandler):
    """Routes ``/`` (page), ``/api/state``, ``/api/take/<item>/<take>`` and ``/audio/<file>``."""

    state: RecordingSession  # bound onto the subclass by serve()

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self.send_bytes(_PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send_json(self.state.state())
        elif path.startswith("/audio/"):
            self.serve_audio(self.state.store.out_dir, path[len("/audio/"):])
        else:
            self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        parts = path.strip("/").split("/")
        if len(parts) != 4 or parts[0] != "api" or parts[1] != "take":
            self.send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length) if length else b""
        try:
            self.send_json(self.state.save_take(parts[2], parts[3], data))
        except KeyError as exc:
            # KeyError's str() is a repr; the page shows this text to the reciter verbatim.
            self.send_json({"error": exc.args[0]}, status=400)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=400)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS,
                        help="Frozen counterfactual item manifest (JSONL).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Where to write the recorded takes (must not be the tracked "
                             "fixtures directory; default: audit_run/counterfactual_audio).")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind. Keep the default: browsers disable the "
                             "microphone outside a secure context, so a LAN address cannot "
                             "record. SSH-forward the port instead.")
    args = parser.parse_args()

    if args.out_dir.resolve() == FIXTURES_DIR.resolve():
        parser.error("--out-dir must not be the tracked fixtures directory; audio does not belong in git")

    session = RecordingSession(load_items(args.items), TakeStore(args.out_dir))
    progress = session.state()["progress"]
    httpd = serve(_Handler, session, args.port, args.host)
    print(f"{progress['items']} counterfactual items, {progress['takes']} takes to record; "
          f"{progress['takes_recorded']} already in {args.out_dir}.")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: the microphone only works in a secure context (https, localhost or "
              "127.0.0.1). On a non-loopback address the browser will report no microphone. "
              "Bind loopback and SSH-forward the port instead.")
    print(f"Recording UI on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

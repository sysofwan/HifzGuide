"""Local web UI for the P7.F0 waqf event-adjudication gate (#27).

Serves the candidate-boundary worklist (:mod:`tadabur.waqf_event_sampler`) in the
browser so a human can play each **whole** clip, seek to a candidate boundary, and
mark it — the analogue of the poison-audit UI (:mod:`tadabur.audit_ui`) for the
event-level signal ADR-0004 needs. A silence VAD cannot tell a real **waqf** from a
mid-word stop-consonant/hamza **closure**, nor confirm that a continued boundary was
a genuine **wasl**; the reviewer calls each, and the verdict is persisted straight
into the canonical waqf event-fixture JSONL (:mod:`tadabur.waqf_event_fixtures`), so
the UI resumes from — and is interchangeable with — whatever that file already holds.

Each worklist row ``(clip_id, boundary_index)`` is one adjudication unit and one
fixture line. The clip's Uthmani ayah text (which the worklist omits) is recovered
from ``quran.db`` for context, and the clip audio is served from ``--audio-dir`` —
the whole-clip staging directory :mod:`tadabur.waqf_segments` writes, where each clip
already lives under its raw ``audio_filename`` (the row's ``local_audio_path``).

Usage:
  python -m tadabur.waqf_audit_ui --worklist waqf_worklist.jsonl \\
    --audio-dir clips/ [--port 8000] [--host 0.0.0.0]

  ``--audio-dir`` is the same directory ``tadabur.waqf_segments`` staged the whole
  passing clips into (the audio the VAD/segmentation pass analysed to propose these
  candidate boundaries); no separate audio-export step is needed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .audit_http import AuditHandler, serve
from .audit_ui import DEFAULT_QURAN_DB, uthmani_index
from .waqf_event_fixtures import (
    WAQF_EVENT_CLASSES,
    WAQF_EVENTS_PATH,
    WaqfEventEntry,
    load_waqf_events,
    write_waqf_events,
)
from .waqf_event_sampler import WaqfCandidateItem

_PAGE_PATH = Path(__file__).parent / "waqf_audit_ui_page.html"

# The adjudication unit's key: one candidate boundary within a clip.
BoundaryKey = tuple[str, int]


def load_worklist(path: Path) -> list[WaqfCandidateItem]:
    """Read the sampler worklist (JSONL) into :class:`WaqfCandidateItem` rows, in order."""
    items: list[WaqfCandidateItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(WaqfCandidateItem(**json.loads(line)))
    return items


# One clip's full set of candidate boundaries, shown as context on the card so the reviewer
# sees every waqf/wasl/closure point in the clip (dimmed) while judging the active one.
BOUNDARY_CONTEXT_FIELDS = ("boundary_index", "word_index", "start_s", "end_s", "predicted")


def _boundary_context(row: dict) -> dict[str, object]:
    return {field: row[field] for field in BOUNDARY_CONTEXT_FIELDS}


def clip_boundaries_from_candidates(path: Path) -> dict[str, list[dict[str, object]]]:
    """Group *all* candidate boundaries (JSONL of ``WaqfCandidate``) by clip id.

    Read from the full candidate manifest so a clip's boundaries that the sampler did not
    draw into the worklist still show as context. Each clip's boundaries are ordered by
    ``boundary_index``.
    """
    by_clip: dict[str, list[dict[str, object]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_clip.setdefault(row["clip_id"], []).append(_boundary_context(row))
    for boundaries in by_clip.values():
        boundaries.sort(key=lambda b: b["boundary_index"])
    return by_clip


def clip_boundaries_from_items(items: list[WaqfCandidateItem]) -> dict[str, list[dict[str, object]]]:
    """Fallback per-clip boundary context built from the worklist itself.

    Used when no full candidate manifest is supplied: shows the clip's boundaries that
    *are* in the worklist (the sampled subset), still ordered by ``boundary_index``.
    """
    by_clip: dict[str, list[dict[str, object]]] = {}
    for item in items:
        by_clip.setdefault(item.clip_id, []).append(_boundary_context(asdict(item)))
    for boundaries in by_clip.values():
        boundaries.sort(key=lambda b: b["boundary_index"])
    return by_clip


@dataclass
class WaqfEventStore:
    """The current per-boundary verdicts, persisted as the waqf event-fixture set.

    Verdicts are keyed by ``(clip_id, boundary_index)`` — one worklist row, one
    fixture line. :meth:`set` and :meth:`clear` rewrite the file atomically through
    the schema module, so the on-disk fixtures always equal the UI state and a
    restart resumes exactly where the human left off.
    """

    path: Path
    entries: dict[BoundaryKey, WaqfEventEntry]

    @classmethod
    def load(cls, path: Path) -> "WaqfEventStore":
        """Build a store from any already-adjudicated entries in the fixture file."""
        entries = {(e.clip_id, e.boundary_index): e for e in load_waqf_events(path)}
        return cls(path, entries)

    def verdict_of(self, key: BoundaryKey) -> str | None:
        entry = self.entries.get(key)
        return entry.verdict if entry else None

    def note_of(self, key: BoundaryKey) -> str:
        entry = self.entries.get(key)
        return entry.note if entry else ""

    def set(self, entry: WaqfEventEntry) -> None:
        """Record (or overwrite) a boundary's verdict and persist the fixture set.

        The change is staged in a copy and the on-disk file rewritten *before*
        ``self.entries`` is swapped in, so a rejected entry (e.g. an invalid verdict
        class the schema refuses to write) leaves both the store and the fixture file
        exactly as they were — the adjudication session never holds a line that could
        not be persisted.
        """
        staged = dict(self.entries)
        staged[(entry.clip_id, entry.boundary_index)] = entry
        self._persist(staged)
        self.entries = staged

    def clear(self, key: BoundaryKey) -> None:
        """Un-adjudicate a boundary (moves it back to pending) and persist."""
        if key not in self.entries:
            return
        staged = dict(self.entries)
        del staged[key]
        self._persist(staged)
        self.entries = staged

    def _persist(self, entries: dict[BoundaryKey, WaqfEventEntry]) -> None:
        ordered = sorted(entries.values(), key=lambda e: (e.clip_id, e.boundary_index))
        write_waqf_events(ordered, self.path)


def class_stats(items: list[WaqfCandidateItem], store: WaqfEventStore) -> list[dict[str, object]]:
    """Per predicted-class progress and the verdict confusion over the worklist.

    For each predicted class, ``verdicts`` counts how the adjudicated boundaries in
    that stratum actually resolved — the confusion ADR-0004's eval reads (e.g. a
    ``predicted=wasl`` boundary the human calls ``waqf`` is a **false-wasl**; a
    ``predicted=waqf`` boundary called ``mid_word_closure`` is a bad snap). Classes
    are returned in :data:`WAQF_EVENT_CLASSES` order.
    """
    counts = {
        c: {"total": 0, "labelled": 0, "verdicts": {v: 0 for v in WAQF_EVENT_CLASSES}}
        for c in WAQF_EVENT_CLASSES
    }
    for item in items:
        bucket = counts[item.predicted]
        bucket["total"] += 1
        verdict = store.verdict_of((item.clip_id, item.boundary_index))
        if verdict is not None:
            bucket["labelled"] += 1
            bucket["verdicts"][verdict] += 1
    return [
        {"predicted": c, "total": counts[c]["total"], "labelled": counts[c]["labelled"],
         "verdicts": counts[c]["verdicts"]}
        for c in WAQF_EVENT_CLASSES
    ]


def item_view(server: "WaqfAuditServer", item: WaqfCandidateItem) -> dict[str, object]:
    """One worklist boundary as an adjudication row: its span, class, and current verdict.

    A row inside a clip page (see :func:`clip_view`); carries the boundary's identity and
    time span so the reviewer can seek to it and call waqf / wasl / mid-word-closure, plus
    the verdict/note already on file so the page resumes where they left off.
    """
    key = (item.clip_id, item.boundary_index)
    return {
        "clip_id": item.clip_id,
        "surah_ayah": item.surah_ayah,
        "boundary_index": item.boundary_index,
        "word_index": item.word_index,
        "start_s": item.start_s,
        "end_s": item.end_s,
        "predicted": item.predicted,
        "verdict": server.store.verdict_of(key),
        "note": server.store.note_of(key),
    }


def clip_view(server: "WaqfAuditServer", clip_id: str, items: list[WaqfCandidateItem]) -> dict[str, object]:
    """One clip page: its audio + ayah, the boundaries to adjudicate, and stop context.

    ``judge`` is every worklist boundary in this clip — the rows the reviewer marks, ordered
    by clip time. ``clip_boundaries`` is the clip's full candidate set (or the sampled subset
    when no candidate manifest was given), from which the page draws the *stop* markers
    (waqf / mid-word-closure) so the reviewer sees all pause points while playing the clip
    once. Audio is served under the clip's staged filename.
    """
    first = items[0]
    return {
        "clip_id": clip_id,
        "surah_ayah": first.surah_ayah,
        "uthmani": server.uthmani.get(first.surah_ayah, ""),
        "audio_url": f"/audio/{first.local_audio_path}",
        "audio_available": (server.audio_dir / first.local_audio_path).is_file(),
        "judge": [item_view(server, i) for i in sorted(items, key=lambda i: (i.start_s, i.boundary_index))],
        "clip_boundaries": server.clip_boundaries.get(
            clip_id, [_boundary_context(asdict(i)) for i in items]),
    }


class WaqfAuditServer:
    """Holds the loaded worklist, Uthmani index, event store and audio dir.

    A thin state object the request handler reads; keeps the handler free of globals
    and makes the request logic unit-testable in isolation.
    """

    def __init__(
        self,
        items: list[WaqfCandidateItem],
        uthmani: dict[str, str],
        store: WaqfEventStore,
        audio_dir: Path,
        clip_boundaries: dict[str, list[dict[str, object]]] | None = None,
    ) -> None:
        self.items = items
        self.uthmani = uthmani
        self.store = store
        self.audio_dir = audio_dir
        self.clip_boundaries = (
            clip_boundaries if clip_boundaries is not None
            else clip_boundaries_from_items(items)
        )
        self._by_key: dict[BoundaryKey, WaqfCandidateItem] = {
            (i.clip_id, i.boundary_index): i for i in items
        }
        # Clips in first-appearance order, each with its worklist boundaries.
        self._clips: dict[str, list[WaqfCandidateItem]] = {}
        for item in items:
            self._clips.setdefault(item.clip_id, []).append(item)

    def state(self) -> dict[str, object]:
        """The full UI payload: one page per clip, plus per-class stats."""
        return {
            "clips": [clip_view(self, clip_id, clip_items)
                      for clip_id, clip_items in self._clips.items()],
            "stats": class_stats(self.items, self.store),
            "classes": list(WAQF_EVENT_CLASSES),
        }

    def apply_label(self, payload: dict) -> dict[str, object]:
        """Handle a verdict POST: set or clear one boundary's verdict, return stats.

        ``verdict`` of ``None`` (or missing) clears the boundary. An unknown
        ``(clip_id, boundary_index)`` — not in the worklist — is rejected so a stray
        request cannot write a fixture line with no adjudication unit behind it. An
        invalid ``verdict`` class is rejected by the schema on write.
        """
        key = (payload["clip_id"], payload["boundary_index"])
        item = self._by_key.get(key)
        if item is None:
            raise KeyError(f"no worklist boundary for {key!r}")

        verdict = payload.get("verdict")
        if verdict is None:
            self.store.clear(key)
        else:
            self.store.set(WaqfEventEntry(
                clip_id=item.clip_id,
                audio_ref=item.audio_ref,
                surah_ayah=item.surah_ayah,
                boundary_index=item.boundary_index,
                word_index=item.word_index,
                start_s=item.start_s,
                end_s=item.end_s,
                predicted=item.predicted,
                verdict=verdict,
                note=payload.get("note", ""),
            ))
        return {"stats": class_stats(self.items, self.store)}


class _Handler(AuditHandler):
    """Routes ``/`` (page), ``/api/state``, ``/api/label`` and ``/audio/<file>``."""

    state: WaqfAuditServer  # bound onto the subclass by serve()

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self.send_bytes(_PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send_json(self.state.state())
        elif path.startswith("/audio/"):
            self.serve_audio(self.state.audio_dir, path[len("/audio/"):])
        else:
            self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path != "/api/label":
            self.send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            self.send_json(self.state.apply_label(payload))
        except (KeyError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=400)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worklist", type=Path, required=True, help="Sampler worklist (JSONL).")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Whole-clip staging dir from tadabur.waqf_segments (clips served by audio_filename).")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="Full candidate manifest (tadabur.waqf_candidates) for per-clip boundary "
                             "context; defaults to the worklist's own (sampled) boundaries.")
    parser.add_argument("--fixtures", type=Path, default=WAQF_EVENTS_PATH,
                        help="Waqf event-fixture file to write (default: canonical path).")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1; use 0.0.0.0 to expose on the LAN).")
    parser.add_argument("--quran-db", type=Path, default=DEFAULT_QURAN_DB,
                        help="quran.db for Uthmani ayah text (default: repo data/quran.db).")
    args = parser.parse_args()

    items = load_worklist(args.worklist)
    store = WaqfEventStore.load(args.fixtures)
    uthmani = uthmani_index(args.quran_db, {i.surah_ayah for i in items})
    clip_boundaries = (
        clip_boundaries_from_candidates(args.candidates) if args.candidates is not None else None
    )
    server_state = WaqfAuditServer(items, uthmani, store, args.audio_dir, clip_boundaries)

    httpd = serve(_Handler, server_state, args.port, args.host)
    labelled = sum(1 for i in items if store.verdict_of((i.clip_id, i.boundary_index)))
    print(f"Loaded {len(items)} candidate boundaries ({labelled} already adjudicated); "
          f"{len(uthmani)} ayat with Uthmani text.")
    print(f"Waqf audit UI on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

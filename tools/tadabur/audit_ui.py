"""Local web UI for the P3.5 poison audit + eval-set curation (#6).

Serves the per-contrast audit worklist (``tadabur.audit_sampler``) in the browser
so a human can listen to each admitted clip and mark it **B (accept)** — an
acceptable-imperfect recitation the model should admit — or **C (reject)** — a
genuinely-wrong substitution it must still reject (ADR-0001). One click does both
jobs the audit needs at once: it records the B-vs-C verdict that measures the
per-contrast *poison rate*, and it curates the ``should_accept`` / ``should_reject``
eval fixtures (``tadabur.eval_fixtures``) that #7's harness later reads back.

There is no database and no framework: labels are persisted straight into the two
canonical fixture JSONL files through :func:`eval_fixtures.write_eval_fixtures`, so
the UI resumes from — and is interchangeable with — whatever those files already
hold. Each worklist row ``(clip_id, contrast)`` is one audit unit and one fixture
entry; a clip sampled into several contrast buckets is judged once per bucket.

The audit unit's ``surah_ayah`` (which the worklist omits but the fixture schema
requires) is recovered from the filter manifest, and each clip's exported audio is
served from ``--audio-dir`` by the ``local_audio_path`` the sampler assigned it.

Usage:
  python -m tadabur.audit_ui --worklist audit_worklist.jsonl \\
    --manifest passing_subset.jsonl --audio-dir audit_audio/ [--port 8000]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import eval_fixtures
from .audit_sampler import WorklistItem
from .contrast_attribution import MARGINAL_CONTRAST, all_contrasts
from .eval_fixtures import ACCEPT, REJECT, EvalFixtureEntry
from .manifest import read_records

_PAGE_PATH = Path(__file__).parent / "audit_ui_page.html"

# Bucket order shown in the UI: the seven attribution buckets, then marginal.
CONTRAST_ORDER: tuple[str, ...] = all_contrasts() + (MARGINAL_CONTRAST,)


def load_worklist(path: Path) -> list[WorklistItem]:
    """Read the sampler worklist (JSONL) into :class:`WorklistItem` rows, in order."""
    items: list[WorklistItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(WorklistItem(**json.loads(line)))
    return items


def surah_ayah_index(manifest_path: Path) -> dict[str, str]:
    """Map ``clip_id`` -> ``"surah:ayah"`` from the filter manifest.

    The worklist carries no ``surah_ayah`` but the fixture schema requires it, so
    the UI looks it up here by the shared ``audio_filename`` key.
    """
    return {r.audio_filename: r.surah_ayah for r in read_records(manifest_path)}


def sniff_audio_content_type(data: bytes) -> str:
    """Best-effort ``Content-Type`` for raw audio bytes, by magic number.

    Tadabur clips are exported as their original bytes with no reliable extension,
    so the browser ``<audio>`` element needs the type sniffed from the header
    (RIFF/WAVE, ID3 or MPEG-frame MP3, Ogg, fLaC). Falls back to
    ``application/octet-stream`` when unrecognised.
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    return "application/octet-stream"


@dataclass
class LabelStore:
    """The current B/C verdicts, persisted as the two eval-fixture sets.

    Verdicts are keyed by ``(clip_id, contrast)`` — one worklist row, one fixture
    entry. :meth:`set` and :meth:`clear` rewrite the affected file(s) atomically
    through the schema module, so the on-disk fixtures always equal the UI state
    and a restart resumes exactly where the human left off.
    """

    accept_path: Path
    reject_path: Path
    entries: dict[tuple[str, str], EvalFixtureEntry]

    @classmethod
    def load(cls, accept_path: Path, reject_path: Path) -> "LabelStore":
        """Build a store from any already-labelled entries in the two fixture files."""
        entries: dict[tuple[str, str], EvalFixtureEntry] = {}
        for entry in eval_fixtures.load_eval_fixtures(accept_path, ACCEPT):
            entries[(entry.clip_id, entry.contrast)] = entry
        for entry in eval_fixtures.load_eval_fixtures(reject_path, REJECT):
            entries[(entry.clip_id, entry.contrast)] = entry
        return cls(accept_path, reject_path, entries)

    def verdict_of(self, clip_id: str, contrast: str) -> str | None:
        entry = self.entries.get((clip_id, contrast))
        return entry.verdict if entry else None

    def note_of(self, clip_id: str, contrast: str) -> str:
        entry = self.entries.get((clip_id, contrast))
        return entry.note if entry else ""

    def set(self, entry: EvalFixtureEntry) -> None:
        """Record (or overwrite) a verdict and persist both fixture sets."""
        if entry.verdict not in (ACCEPT, REJECT):
            raise ValueError(f"verdict must be {ACCEPT!r} or {REJECT!r}, got {entry.verdict!r}")
        self.entries[(entry.clip_id, entry.contrast)] = entry
        self._persist()

    def clear(self, clip_id: str, contrast: str) -> None:
        """Un-label a row (moves it back to pending) and persist."""
        if self.entries.pop((clip_id, contrast), None) is not None:
            self._persist()

    def _persist(self) -> None:
        accept = [e for e in self.entries.values() if e.verdict == ACCEPT]
        reject = [e for e in self.entries.values() if e.verdict == REJECT]
        eval_fixtures.write_eval_fixtures(accept, self.accept_path, ACCEPT)
        eval_fixtures.write_eval_fixtures(reject, self.reject_path, REJECT)


def contrast_stats(
    items: list[WorklistItem], store: LabelStore
) -> list[dict[str, object]]:
    """Per-contrast progress and poison rate over the worklist.

    ``poison_rate`` is ``reject / (accept + reject)`` among *labelled* rows in the
    bucket — the direct measurement the P3.5 go/no-go gate reads (ADR-0001) — and is
    ``None`` until the bucket has a verdict. Buckets are returned in
    :data:`CONTRAST_ORDER`.
    """
    order = {c: i for i, c in enumerate(CONTRAST_ORDER)}
    counts: dict[str, dict[str, int]] = {}
    for item in items:
        bucket = counts.setdefault(item.contrast, {"total": 0, "accept": 0, "reject": 0})
        bucket["total"] += 1
        verdict = store.verdict_of(item.clip_id, item.contrast)
        if verdict == ACCEPT:
            bucket["accept"] += 1
        elif verdict == REJECT:
            bucket["reject"] += 1

    stats: list[dict[str, object]] = []
    for contrast, c in sorted(counts.items(), key=lambda kv: order.get(kv[0], len(order))):
        labelled = c["accept"] + c["reject"]
        stats.append({
            "contrast": contrast,
            "total": c["total"],
            "labelled": labelled,
            "accept": c["accept"],
            "reject": c["reject"],
            "poison_rate": (c["reject"] / labelled) if labelled else None,
        })
    return stats


def item_view(item: WorklistItem, store: LabelStore, sa: dict[str, str], audio_dir: Path) -> dict[str, object]:
    """The JSON shape one worklist row is sent to the browser as."""
    return {
        "clip_id": item.clip_id,
        "contrast": item.contrast,
        "match_ratio": item.match_ratio,
        "surah_ayah": sa.get(item.clip_id, ""),
        "audio_url": f"/audio/{item.local_audio_path}",
        "audio_available": (audio_dir / item.local_audio_path).is_file(),
        "verdict": store.verdict_of(item.clip_id, item.contrast),
        "note": store.note_of(item.clip_id, item.contrast),
    }


class AuditServer:
    """Holds the loaded worklist, manifest index, label store and audio dir.

    A thin state object the request handler reads; keeps the handler free of
    globals and makes the request logic unit-testable in isolation.
    """

    def __init__(
        self,
        items: list[WorklistItem],
        surah_ayah: dict[str, str],
        store: LabelStore,
        audio_dir: Path,
    ) -> None:
        self.items = items
        self.surah_ayah = surah_ayah
        self.store = store
        self.audio_dir = audio_dir
        self._by_key = {(i.clip_id, i.contrast): i for i in items}

    def state(self) -> dict[str, object]:
        """The full UI payload: every row's view plus per-contrast stats."""
        return {
            "items": [item_view(i, self.store, self.surah_ayah, self.audio_dir) for i in self.items],
            "stats": contrast_stats(self.items, self.store),
            "contrast_order": list(CONTRAST_ORDER),
        }

    def apply_label(self, payload: dict) -> dict[str, object]:
        """Handle a label POST: set or clear one row's verdict, return fresh stats.

        ``verdict`` of ``None`` (or missing) clears the row. An unknown
        ``(clip_id, contrast)`` — not in the worklist — is rejected so a stray
        request cannot write a fixture entry with no audit unit behind it.
        """
        clip_id = payload["clip_id"]
        contrast = payload["contrast"]
        item = self._by_key.get((clip_id, contrast))
        if item is None:
            raise KeyError(f"no worklist row for {(clip_id, contrast)!r}")

        verdict = payload.get("verdict")
        if verdict is None:
            self.store.clear(clip_id, contrast)
        else:
            self.store.set(EvalFixtureEntry(
                clip_id=clip_id,
                audio_ref=item.audio_ref,
                surah_ayah=self.surah_ayah.get(clip_id, ""),
                contrast=contrast,
                verdict=verdict,
                note=payload.get("note", ""),
            ))
        return {"stats": contrast_stats(self.items, self.store)}


class _Handler(BaseHTTPRequestHandler):
    """Routes ``/`` (page), ``/api/state``, ``/api/label`` and ``/audio/<file>``."""

    server_state: AuditServer  # injected via functools.partial

    def log_message(self, *args: object) -> None:  # noqa: D401 - quiet the default stderr spam
        return

    def _send_json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send_bytes(_PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send_json(self.server_state.state())
        elif path.startswith("/audio/"):
            self._serve_audio(path[len("/audio/"):])
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path != "/api/label":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            self._send_json(self.server_state.apply_label(payload))
        except (KeyError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _serve_audio(self, name: str) -> None:
        # ``name`` is a sampler ``local_audio_path`` (a flat, separator-free name);
        # resolve under audio_dir and refuse anything that escapes it.
        audio_dir = self.server_state.audio_dir.resolve()
        target = (audio_dir / name).resolve()
        if audio_dir not in target.parents or not target.is_file():
            self._send_json({"error": "audio not found"}, status=404)
            return
        data = target.read_bytes()
        self._send_bytes(data, sniff_audio_content_type(data))


def serve(server_state: AuditServer, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Build (but do not block on) the threading HTTP server for ``server_state``.

    The state is bound onto a per-server handler subclass so each request handler
    instance can reach the loaded worklist and label store without globals.
    ``host`` defaults to loopback; pass ``0.0.0.0`` to expose the UI on the LAN so
    a human on another device can grade the clips.
    """

    class _BoundHandler(_Handler):
        pass

    _BoundHandler.server_state = server_state
    return ThreadingHTTPServer((host, port), _BoundHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worklist", type=Path, required=True, help="Sampler worklist (JSONL).")
    parser.add_argument("--manifest", type=Path, required=True, help="Filter manifest (for surah:ayah).")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory of exported clip audio.")
    parser.add_argument("--accept", type=Path, default=eval_fixtures.SHOULD_ACCEPT_PATH,
                        help="should-accept fixture file to write (default: canonical path).")
    parser.add_argument("--reject", type=Path, default=eval_fixtures.SHOULD_REJECT_PATH,
                        help="should-reject fixture file to write (default: canonical path).")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1; use 0.0.0.0 to expose on the LAN).")
    args = parser.parse_args()

    items = load_worklist(args.worklist)
    surah_ayah = surah_ayah_index(args.manifest)
    store = LabelStore.load(args.accept, args.reject)
    server_state = AuditServer(items, surah_ayah, store, args.audio_dir)

    httpd = serve(server_state, args.port, args.host)
    labelled = sum(1 for i in items if store.verdict_of(i.clip_id, i.contrast))
    print(f"Loaded {len(items)} worklist rows ({labelled} already labelled).")
    print(f"Audit UI on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
